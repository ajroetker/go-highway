// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gguf

//go:generate go run ../../../cmd/hwygen -input gguf_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch gguf

import (
	"math"

	"github.com/ajroetker/go-highway/hwy"
)

// Block sizes in bytes for each GGUF quantization format.
const (
	BlockSizeQ4_0  = 18 // fp16 scale (2) + 16 nibble bytes
	BlockSizeQ8_0  = 34 // fp16 scale (2) + 32 int8 quants
	BlockSizeIQ4NL = 18 // fp16 scale (2) + 16 nibble bytes
	QK             = 32 // Number of values per block
)

// kvaluesIQ4NL is the non-linear lookup table for IQ4_NL dequantization.
// From llama.cpp ggml-common.h kvalues_iq4nl.
var kvaluesIQ4NL = [16]int8{
	-127, -104, -83, -65, -49, -35, -22, -10,
	1, 13, 25, 38, 53, 69, 89, 113,
}

// BaseDequantizeQ8_0 converts Q8_0 quantized blocks to float32.
// Each block is 34 bytes: 2-byte fp16 scale + 32 int8 quants.
//
//	output[i] = d * int8(qs[i])
func BaseDequantizeQ8_0(data []uint8, output []float32) {
	if len(data) == 0 {
		return
	}
	nblocks := len(data) / BlockSizeQ8_0

	lanes := hwy.NumLanes[float32]()
	buf := make([]float32, lanes)

	for b := range nblocks {
		blockData := data[b*BlockSizeQ8_0 : (b+1)*BlockSizeQ8_0]

		// Read fp16 scale (little-endian) and convert to float32.
		// Inline bit manipulation instead of hwy.Float16ToFloat32 for C/GOAT compatibility.
		raw := uint32(blockData[0]) | uint32(blockData[1])<<8
		sign := raw >> 15
		exp := (raw >> 10) & 0x1F
		mant := raw & 0x3FF
		var d float32
		if exp == 0 {
			d = math.Float32frombits(sign << 31)
		} else {
			d = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}
		scaleVec := hwy.Set(d)

		qs := blockData[2:] // 32 int8 quants
		outOff := b * QK

		i := 0
		for ; i+lanes <= QK; i += lanes {
			for j := range lanes {
				buf[j] = float32(int8(qs[i+j]))
			}
			v := hwy.Load(buf)
			result := hwy.Mul(v, scaleVec)
			hwy.Store(result, output[outOff+i:])
		}

		// Scalar tail
		for ; i < QK; i++ {
			output[outOff+i] = d * float32(int8(qs[i]))
		}
	}
}

// BaseDequantizeQ4_0 converts Q4_0 quantized blocks to float32.
// Each block is 18 bytes: 2-byte fp16 scale + 16 nibble bytes.
// GGUF uses split nibble layout: low nibbles produce the first 16 values,
// high nibbles produce the last 16 values.
//
//	output[j]    = d * (lo_nibble - 8)
//	output[j+16] = d * (hi_nibble - 8)
func BaseDequantizeQ4_0(data []uint8, output []float32) {
	if len(data) == 0 {
		return
	}
	nblocks := len(data) / BlockSizeQ4_0

	lanes := hwy.NumLanes[float32]()
	buf := make([]float32, lanes)

	for b := range nblocks {
		blockData := data[b*BlockSizeQ4_0 : (b+1)*BlockSizeQ4_0]

		// Read fp16 scale (little-endian) and convert to float32.
		raw := uint32(blockData[0]) | uint32(blockData[1])<<8
		sign := raw >> 15
		exp := (raw >> 10) & 0x1F
		mant := raw & 0x3FF
		var d float32
		if exp == 0 {
			d = math.Float32frombits(sign << 31)
		} else {
			d = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}
		scaleVec := hwy.Set(d)

		qs := blockData[2:] // 16 nibble bytes
		outOff := b * QK

		// Low nibbles -> first 16 values
		i := 0
		for ; i+lanes <= 16; i += lanes {
			for j := range lanes {
				lo := int(qs[i+j] & 0x0F)
				buf[j] = float32(lo - 8)
			}
			v := hwy.Load(buf)
			result := hwy.Mul(v, scaleVec)
			hwy.Store(result, output[outOff+i:])
		}
		for ; i < 16; i++ {
			lo := int(qs[i] & 0x0F)
			output[outOff+i] = d * float32(lo-8)
		}

		// High nibbles -> last 16 values
		i = 0
		for ; i+lanes <= 16; i += lanes {
			for j := range lanes {
				hi := int((qs[i+j] >> 4) & 0x0F)
				buf[j] = float32(hi - 8)
			}
			v := hwy.Load(buf)
			result := hwy.Mul(v, scaleVec)
			hwy.Store(result, output[outOff+16+i:])
		}
		for ; i < 16; i++ {
			hi := int((qs[i] >> 4) & 0x0F)
			output[outOff+16+i] = d * float32(hi-8)
		}
	}
}

// BaseDequantizeIQ4NL converts IQ4_NL quantized blocks to float32.
// Each block is 18 bytes: 2-byte fp16 scale + 16 nibble bytes.
// Same split nibble layout as Q4_0, but uses a non-linear lookup table
// instead of linear (nibble - 8) mapping.
//
//	output[j]    = d * kvaluesIQ4NL[lo_nibble]
//	output[j+16] = d * kvaluesIQ4NL[hi_nibble]
func BaseDequantizeIQ4NL(data []uint8, output []float32) {
	if len(data) == 0 {
		return
	}
	nblocks := len(data) / BlockSizeIQ4NL

	// Lookup table as float32 to avoid int8→float32 conversion in the hot loop
	// and to keep data on the stack (no static data section for GoAT compatibility).
	var lut [16]float32
	lut[0] = -127
	lut[1] = -104
	lut[2] = -83
	lut[3] = -65
	lut[4] = -49
	lut[5] = -35
	lut[6] = -22
	lut[7] = -10
	lut[8] = 1
	lut[9] = 13
	lut[10] = 25
	lut[11] = 38
	lut[12] = 53
	lut[13] = 69
	lut[14] = 89
	lut[15] = 113

	lanes := hwy.NumLanes[float32]()
	buf := make([]float32, lanes)

	for b := range nblocks {
		blockData := data[b*BlockSizeIQ4NL : (b+1)*BlockSizeIQ4NL]

		// Read fp16 scale (little-endian) and convert to float32.
		raw := uint32(blockData[0]) | uint32(blockData[1])<<8
		sign := raw >> 15
		exp := (raw >> 10) & 0x1F
		mant := raw & 0x3FF
		var d float32
		if exp == 0 {
			d = math.Float32frombits(sign << 31)
		} else {
			d = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}
		scaleVec := hwy.Set(d)

		qs := blockData[2:] // 16 nibble bytes
		outOff := b * QK

		// Low nibbles -> first 16 values
		i := 0
		for ; i+lanes <= 16; i += lanes {
			for j := range lanes {
				lo := qs[i+j] & 0x0F
				buf[j] = lut[lo]
			}
			v := hwy.Load(buf)
			result := hwy.Mul(v, scaleVec)
			hwy.Store(result, output[outOff+i:])
		}
		for ; i < 16; i++ {
			lo := qs[i] & 0x0F
			output[outOff+i] = d * lut[lo]
		}

		// High nibbles -> last 16 values
		i = 0
		for ; i+lanes <= 16; i += lanes {
			for j := range lanes {
				hi := (qs[i+j] >> 4) & 0x0F
				buf[j] = lut[hi]
			}
			v := hwy.Load(buf)
			result := hwy.Mul(v, scaleVec)
			hwy.Store(result, output[outOff+16+i:])
		}
		for ; i < 16; i++ {
			hi := (qs[i] >> 4) & 0x0F
			output[outOff+16+i] = d * lut[hi]
		}
	}
}
