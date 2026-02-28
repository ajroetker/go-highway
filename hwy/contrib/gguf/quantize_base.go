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

//go:generate go run ../../../cmd/hwygen -input quantize_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch ggufquant

import (
	"math"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseQuantizeQ8_0 quantizes float32 values to Q8_0 blocks.
// Input length must be a multiple of 32 (QK). Output must be pre-allocated
// to (len(input)/QK) * BlockSizeQ8_0 bytes.
//
// For each block of 32 values:
//
//	d = max(|input[i]|) / 127.0
//	qs[i] = round(input[i] / d) clamped to [-128, 127]
//
// The scale d is stored as fp16 little-endian in the first 2 bytes.
func BaseQuantizeQ8_0(input []float32, output []uint8) {
	if len(input) == 0 {
		return
	}
	nblocks := len(input) / QK

	lanes := hwy.NumLanes[float32]()
	buf := make([]float32, lanes)
	minVec := hwy.Set[float32](-128.0)
	maxVec := hwy.Set[float32](127.0)

	for b := range nblocks {
		inOff := b * QK
		outOff := b * BlockSizeQ8_0
		block := output[outOff : outOff+BlockSizeQ8_0]

		// Find amax = max(|input[i]|) over the block.
		amax := float32(0)
		i := 0
		for ; i+lanes <= QK; i += lanes {
			v := hwy.Load(input[inOff+i:])
			absV := hwy.Abs(v)
			hwy.Store(absV, buf)
			for j := range lanes {
				if buf[j] > amax {
					amax = buf[j]
				}
			}
		}
		for ; i < QK; i++ {
			av := input[inOff+i]
			if av < 0 {
				av = -av
			}
			if av > amax {
				amax = av
			}
		}

		// Compute scale d and inverse id.
		d := amax / 127.0
		var id float32
		if d > 0 {
			id = 127.0 / amax
		}

		// Encode d as fp16 little-endian.
		// Inline conversion for GoAT compatibility (no hwy.Float32ToFloat16 call).
		f32bits := math.Float32bits(d)
		f32sign := (f32bits >> 31) & 1
		f32exp := int((f32bits>>23)&0xFF) - 127
		f32mant := f32bits & 0x7FFFFF

		var fp16bits uint16
		if d == 0 {
			fp16bits = uint16(f32sign << 15)
		} else if f32exp > 15 {
			fp16bits = uint16((f32sign << 15) | (0x1F << 10))
		} else if f32exp < -14 {
			fp16bits = uint16(f32sign << 15)
		} else {
			fp16bits = uint16((f32sign << 15) | uint32(f32exp+15)<<10 | (f32mant >> 13))
		}
		block[0] = uint8(fp16bits)
		block[1] = uint8(fp16bits >> 8)

		// Quantize values: round(input * id), clamped to [-128, 127].
		qs := block[2:]
		i = 0
		idVec := hwy.Set(id)
		for ; i+lanes <= QK; i += lanes {
			v := hwy.Load(input[inOff+i:])
			scaled := hwy.Mul(v, idVec)
			clamped := hwy.Clamp(hwy.Round(scaled), minVec, maxVec)
			hwy.Store(clamped, buf)
			for j := range lanes {
				qs[i+j] = uint8(int8(buf[j]))
			}
		}
		for ; i < QK; i++ {
			val := input[inOff+i] * id
			var q int32
			if val >= 0 {
				q = int32(val + 0.5)
			} else {
				q = int32(val - 0.5)
			}
			if q > 127 {
				q = 127
			} else if q < -128 {
				q = -128
			}
			qs[i] = uint8(int8(q))
		}
	}
}
