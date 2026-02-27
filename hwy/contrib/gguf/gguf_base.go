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

	// K-quant formats: 256 values per super-block.
	QK_K         = 256
	BlockSizeQ2K = 84  // scales(16) + d(2) + dmin(2) + qs(64)
	BlockSizeQ3K = 110 // hmask(32) + qs(64) + scales(12) + d(2)
	BlockSizeQ4K = 144 // d(2) + dmin(2) + scales(12) + qs(128)
	BlockSizeQ5K = 176 // d(2) + dmin(2) + scales(12) + qs(128) + qh(32)
	BlockSizeQ6K = 210 // ql(128) + qh(64) + scales(16) + d(2)
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

// BaseDequantizeQ6K converts Q6_K quantized super-blocks to float32.
// Each super-block is 210 bytes: ql(128) + qh(64) + scales(16) + d(2).
// 16 sub-blocks of 16 values with int8 scales.
//
//	output[i] = d * scale * (q6 - 32)
func BaseDequantizeQ6K(data []uint8, output []float32) {
	if len(data) == 0 {
		return
	}
	nblocks := len(data) / BlockSizeQ6K

	lanes := hwy.NumLanes[float32]()
	buf := make([]float32, lanes)

	for b := range nblocks {
		blockData := data[b*BlockSizeQ6K : (b+1)*BlockSizeQ6K]

		ql := blockData[0:128]
		qh := blockData[128:192]
		sc := blockData[192:208]

		// Read fp16 scale (little-endian) at offset 208.
		raw := uint32(blockData[208]) | uint32(blockData[209])<<8
		sign := raw >> 15
		exp := (raw >> 10) & 0x1F
		mant := raw & 0x3FF
		var d float32
		if exp == 0 {
			d = math.Float32frombits(sign << 31)
		} else {
			d = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}

		outOff := b * QK_K

		// Process 16 sub-blocks of 16 values each.
		// Quant extraction is inlined to avoid a large stack array
		// (which generates a bzero call that GOAT cannot resolve).
		//
		// Sub-block j maps to ql/qh as follows:
		//   half       = j / 8         (two 128-value halves)
		//   group      = (j % 8) / 2   (which of q0..q3)
		//   lBase      = (j % 2) * 16  (first or second 16 of 32)
		//   qlOff      = half*64 + (group&1)*32
		//   qhOff      = half * 32
		//   nibbleShift = (group/2) * 4  (0 for low nibble, 4 for high)
		//   qhShift    = group * 2
		for j := 0; j < 16; j++ {
			scaleVal := d * float32(int8(sc[j]))
			scaleVec := hwy.Set(scaleVal)
			baseOut := outOff + j*16

			half := j / 8
			group := (j % 8) / 2
			lBase := (j % 2) * 16
			qlOff := half*64 + (group&1)*32
			qhOff := half * 32
			qhShift := uint(group * 2)
			nibbleShift := uint((group / 2) * 4)

			i := 0
			for ; i+lanes <= 16; i += lanes {
				for k := range lanes {
					l := lBase + i + k
					low4 := int((ql[qlOff+l] >> nibbleShift) & 0xF)
					high2 := int((qh[qhOff+l] >> qhShift) & 3)
					buf[k] = float32((low4 | (high2 << 4)) - 32)
				}
				v := hwy.Load(buf)
				result := hwy.Mul(v, scaleVec)
				hwy.Store(result, output[baseOut+i:])
			}
			for ; i < 16; i++ {
				l := lBase + i
				low4 := int((ql[qlOff+l] >> nibbleShift) & 0xF)
				high2 := int((qh[qhOff+l] >> qhShift) & 3)
				output[baseOut+i] = scaleVal * float32((low4|(high2<<4))-32)
			}
		}
	}
}

// BaseDequantizeQ4K converts Q4_K quantized super-blocks to float32.
// Each super-block is 144 bytes: d(2) + dmin(2) + scales(12) + qs(128).
// 8 sub-blocks of 32 values with 6-bit packed scales and mins.
//
//	output[i] = d * sc * q4 - dmin * m
func BaseDequantizeQ4K(data []uint8, output []float32) {
	if len(data) == 0 {
		return
	}
	nblocks := len(data) / BlockSizeQ4K

	lanes := hwy.NumLanes[float32]()
	buf := make([]float32, lanes)

	for b := range nblocks {
		blockData := data[b*BlockSizeQ4K : (b+1)*BlockSizeQ4K]

		// Read fp16 d at offset 0.
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

		// Read fp16 dmin at offset 2.
		raw = uint32(blockData[2]) | uint32(blockData[3])<<8
		sign = raw >> 15
		exp = (raw >> 10) & 0x1F
		mant = raw & 0x3FF
		var dmin float32
		if exp == 0 {
			dmin = math.Float32frombits(sign << 31)
		} else {
			dmin = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}

		scales := blockData[4:16]
		qs := blockData[16:144]

		// Unpack 8 (scale, min) pairs from 12-byte packed format.
		// get_scale_min_k4 from llama.cpp.
		var scs [8]float32
		var mns [8]float32
		for j := 0; j < 4; j++ {
			scs[j] = float32(scales[j] & 63)
			mns[j] = float32(scales[j+4] & 63)
		}
		for j := 4; j < 8; j++ {
			scs[j] = float32((scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4))
			mns[j] = float32((scales[j+4] >> 4) | ((scales[j] >> 6) << 4))
		}

		outOff := b * QK_K

		// Process 4 chunks of 64 values (2 sub-blocks of 32 each).
		qOff := 0
		outIdx := outOff
		for chunk := 0; chunk < 4; chunk++ {
			is := chunk * 2
			dsc0 := d * scs[is]
			dmm0 := dmin * mns[is]
			dsc1 := d * scs[is+1]
			dmm1 := dmin * mns[is+1]

			dscVec0 := hwy.Set(dsc0)
			dmmVec0 := hwy.Set(dmm0)
			dscVec1 := hwy.Set(dsc1)
			dmmVec1 := hwy.Set(dmm1)

			// Sub-block 0: low nibbles → 32 values.
			i := 0
			for ; i+lanes <= 32; i += lanes {
				for k := range lanes {
					buf[k] = float32(qs[qOff+i+k] & 0xF)
				}
				v := hwy.Load(buf)
				scaled := hwy.Mul(v, dscVec0)
				result := hwy.Sub(scaled, dmmVec0)
				hwy.Store(result, output[outIdx+i:])
			}
			for ; i < 32; i++ {
				output[outIdx+i] = dsc0*float32(qs[qOff+i]&0xF) - dmm0
			}

			// Sub-block 1: high nibbles → 32 values.
			i = 0
			for ; i+lanes <= 32; i += lanes {
				for k := range lanes {
					buf[k] = float32(qs[qOff+i+k] >> 4)
				}
				v := hwy.Load(buf)
				scaled := hwy.Mul(v, dscVec1)
				result := hwy.Sub(scaled, dmmVec1)
				hwy.Store(result, output[outIdx+32+i:])
			}
			for ; i < 32; i++ {
				output[outIdx+32+i] = dsc1*float32(qs[qOff+i]>>4) - dmm1
			}

			qOff += 32
			outIdx += 64
		}
	}
}

// BaseDequantizeQ5K converts Q5_K quantized super-blocks to float32.
// Each super-block is 176 bytes: d(2) + dmin(2) + scales(12) + qs(128) + qh(32).
// 8 sub-blocks of 32 values. Same scale packing as Q4_K, plus high bits in qh.
//
//	output[i] = d * sc * q5 - dmin * m
func BaseDequantizeQ5K(data []uint8, output []float32) {
	if len(data) == 0 {
		return
	}
	nblocks := len(data) / BlockSizeQ5K

	lanes := hwy.NumLanes[float32]()
	buf := make([]float32, lanes)

	for b := range nblocks {
		blockData := data[b*BlockSizeQ5K : (b+1)*BlockSizeQ5K]

		// Read fp16 d at offset 0.
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

		// Read fp16 dmin at offset 2.
		raw = uint32(blockData[2]) | uint32(blockData[3])<<8
		sign = raw >> 15
		exp = (raw >> 10) & 0x1F
		mant = raw & 0x3FF
		var dmin float32
		if exp == 0 {
			dmin = math.Float32frombits(sign << 31)
		} else {
			dmin = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}

		scales := blockData[4:16]
		ql := blockData[16:144]
		qh := blockData[144:176]

		// Unpack 8 (scale, min) pairs — same as Q4_K.
		var scs [8]float32
		var mns [8]float32
		for j := 0; j < 4; j++ {
			scs[j] = float32(scales[j] & 63)
			mns[j] = float32(scales[j+4] & 63)
		}
		for j := 4; j < 8; j++ {
			scs[j] = float32((scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4))
			mns[j] = float32((scales[j+4] >> 4) | ((scales[j] >> 6) << 4))
		}

		outOff := b * QK_K

		// Process 4 chunks of 64 values (2 sub-blocks of 32 each).
		// Each chunk's two sub-blocks use consecutive high-bit positions
		// in qh: chunk c uses bits c*2 and c*2+1.
		// We use shift-based extraction instead of mask variables to avoid
		// a C operator precedence issue (& vs != in generated C code).
		qlOff := 0
		outIdx := outOff
		for chunk := 0; chunk < 4; chunk++ {
			is := chunk * 2
			dsc0 := d * scs[is]
			dmm0 := dmin * mns[is]
			dsc1 := d * scs[is+1]
			dmm1 := dmin * mns[is+1]

			dscVec0 := hwy.Set(dsc0)
			dmmVec0 := hwy.Set(dmm0)
			dscVec1 := hwy.Set(dsc1)
			dmmVec1 := hwy.Set(dmm1)

			hbShift0 := uint(chunk * 2)
			hbShift1 := uint(chunk*2 + 1)

			// Sub-block 0: low nibbles + high bit → 32 values.
			i := 0
			for ; i+lanes <= 32; i += lanes {
				for k := range lanes {
					l := i + k
					q := int(ql[qlOff+l]&0xF) + int((qh[l]>>hbShift0)&1)*16
					buf[k] = float32(q)
				}
				v := hwy.Load(buf)
				scaled := hwy.Mul(v, dscVec0)
				result := hwy.Sub(scaled, dmmVec0)
				hwy.Store(result, output[outIdx+i:])
			}
			for ; i < 32; i++ {
				q := int(ql[qlOff+i]&0xF) + int((qh[i]>>hbShift0)&1)*16
				output[outIdx+i] = dsc0*float32(q) - dmm0
			}

			// Sub-block 1: high nibbles + high bit → 32 values.
			i = 0
			for ; i+lanes <= 32; i += lanes {
				for k := range lanes {
					l := i + k
					q := int(ql[qlOff+l]>>4) + int((qh[l]>>hbShift1)&1)*16
					buf[k] = float32(q)
				}
				v := hwy.Load(buf)
				scaled := hwy.Mul(v, dscVec1)
				result := hwy.Sub(scaled, dmmVec1)
				hwy.Store(result, output[outIdx+32+i:])
			}
			for ; i < 32; i++ {
				q := int(ql[qlOff+i]>>4) + int((qh[i]>>hbShift1)&1)*16
				output[outIdx+32+i] = dsc1*float32(q) - dmm1
			}

			qlOff += 32
			outIdx += 64
		}
	}
}

// BaseDequantizeQ2K converts Q2_K quantized super-blocks to float32.
// Each super-block is 84 bytes: scales(16) + d(2) + dmin(2) + qs(64).
// 16 sub-blocks of 16 values with 4-bit packed scales and mins.
//
//	output[i] = d * sc * q2 - dmin * m
func BaseDequantizeQ2K(data []uint8, output []float32) {
	if len(data) == 0 {
		return
	}
	nblocks := len(data) / BlockSizeQ2K

	lanes := hwy.NumLanes[float32]()
	buf := make([]float32, lanes)

	for b := range nblocks {
		blockData := data[b*BlockSizeQ2K : (b+1)*BlockSizeQ2K]

		scalesRaw := blockData[0:16]

		// Read fp16 d at offset 16.
		raw := uint32(blockData[16]) | uint32(blockData[17])<<8
		sign := raw >> 15
		exp := (raw >> 10) & 0x1F
		mant := raw & 0x3FF
		var d float32
		if exp == 0 {
			d = math.Float32frombits(sign << 31)
		} else {
			d = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}

		// Read fp16 dmin at offset 18.
		raw = uint32(blockData[18]) | uint32(blockData[19])<<8
		sign = raw >> 15
		exp = (raw >> 10) & 0x1F
		mant = raw & 0x3FF
		var dmin float32
		if exp == 0 {
			dmin = math.Float32frombits(sign << 31)
		} else {
			dmin = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}

		qs := blockData[20:84]
		outOff := b * QK_K

		// Process 16 sub-blocks of 16 values each.
		// Quant extraction inlined to avoid large stack array (bzero issue).
		//
		// Sub-block is maps to qs as:
		//   chunk = is / 8
		//   group = (is % 8) / 2
		//   lBase = (is % 2) * 16
		//   q = (qs[chunk*32 + lBase + i] >> (group*2)) & 3
		for is := 0; is < 16; is++ {
			sc := float32(scalesRaw[is] & 0x0F)
			m := float32(scalesRaw[is] >> 4)
			dsc := d * sc
			dmm := dmin * m
			dscVec := hwy.Set(dsc)
			dmmVec := hwy.Set(dmm)

			baseOut := outOff + is*16
			chunk := is / 8
			group := (is % 8) / 2
			lBase := (is % 2) * 16
			qBase := chunk * 32
			shift := uint(group * 2)

			i := 0
			for ; i+lanes <= 16; i += lanes {
				for k := range lanes {
					buf[k] = float32((qs[qBase+lBase+i+k] >> shift) & 3)
				}
				v := hwy.Load(buf)
				scaled := hwy.Mul(v, dscVec)
				result := hwy.Sub(scaled, dmmVec)
				hwy.Store(result, output[baseOut+i:])
			}
			for ; i < 16; i++ {
				output[baseOut+i] = dsc*float32((qs[qBase+lBase+i]>>shift)&3) - dmm
			}
		}
	}
}

// BaseDequantizeQ3K converts Q3_K quantized super-blocks to float32.
// Each super-block is 110 bytes: hmask(32) + qs(64) + scales(12) + d(2).
// 16 sub-blocks of 16 values with 6-bit packed scales and hmask high bits.
//
//	output[i] = d * (scale - 32) * (q3 - 4)
func BaseDequantizeQ3K(data []uint8, output []float32) {
	if len(data) == 0 {
		return
	}
	nblocks := len(data) / BlockSizeQ3K

	lanes := hwy.NumLanes[float32]()
	buf := make([]float32, lanes)

	for b := range nblocks {
		blockData := data[b*BlockSizeQ3K : (b+1)*BlockSizeQ3K]

		hmask := blockData[0:32]
		qs := blockData[32:96]
		scaleData := blockData[96:108]

		// Read fp16 d at offset 108.
		raw := uint32(blockData[108]) | uint32(blockData[109])<<8
		sign := raw >> 15
		exp := (raw >> 10) & 0x1F
		mant := raw & 0x3FF
		var d float32
		if exp == 0 {
			d = math.Float32frombits(sign << 31)
		} else {
			d = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}

		outOff := b * QK_K

		// Unpack 16 six-bit scale values from 12 bytes.
		// Each scale is 0..63, with -32 applied during dequantization.
		var rawScales [16]int
		for i := 0; i < 4; i++ {
			rawScales[i] = int(scaleData[i]&0x0F) | (int(scaleData[8+i]&0x03) << 4)
			rawScales[i+4] = int(scaleData[4+i]&0x0F) | (int((scaleData[8+i]>>2)&0x03) << 4)
			rawScales[i+8] = int((scaleData[i]>>4)&0x0F) | (int((scaleData[8+i]>>4)&0x03) << 4)
			rawScales[i+12] = int((scaleData[4+i]>>4)&0x0F) | (int((scaleData[8+i]>>6)&0x03) << 4)
		}

		// Process 16 sub-blocks of 16 values each.
		// Quant extraction inlined to avoid large stack array (bzero issue).
		//
		// Sub-block j maps to qs/hmask as:
		//   chunk = j / 8
		//   group = (j % 8) / 2
		//   lBase = (j % 2) * 16
		//   shift = group * 2
		//   hmBit = chunk * 4 + group
		//   low2 = (qs[chunk*32 + l] >> shift) & 3
		//   high1 = (hmask[l] >> hmBit) & 1
		//   q = low2 + high1*4 - 4
		for j := 0; j < 16; j++ {
			scaleVal := d * float32(rawScales[j]-32)
			scaleVec := hwy.Set(scaleVal)
			baseOut := outOff + j*16

			chunk := j / 8
			group := (j % 8) / 2
			lBase := (j % 2) * 16
			qBase := chunk * 32
			shift := uint(group * 2)
			hmBit := uint(chunk*4 + group)

			i := 0
			for ; i+lanes <= 16; i += lanes {
				for k := range lanes {
					l := lBase + i + k
					low2 := int((qs[qBase+l] >> shift) & 3)
					high1 := int((hmask[l] >> hmBit) & 1)
					buf[k] = float32(low2 + high1*4 - 4)
				}
				v := hwy.Load(buf)
				result := hwy.Mul(v, scaleVec)
				hwy.Store(result, output[baseOut+i:])
			}
			for ; i < 16; i++ {
				l := lBase + i
				low2 := int((qs[qBase+l] >> shift) & 3)
				high1 := int((hmask[l] >> hmBit) & 1)
				output[baseOut+i] = scaleVal * float32(low2+high1*4-4)
			}
		}
	}
}
