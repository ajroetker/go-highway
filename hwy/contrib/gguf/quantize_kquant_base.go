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

//go:generate go run ../../../cmd/hwygen -input quantize_kquant_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch ggufkqquant

import (
	"math"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseQuantizeQ8_K quantizes float32 values to Q8_K blocks.
// Input length must be a multiple of 256 (QK_K). Output must be pre-allocated
// to (len(input)/QK_K) * BlockSizeQ8K bytes.
//
// Q8_K format (292 bytes per block of 256 values):
//
//	d      (4 bytes): float32 scale, little-endian
//	qs   (256 bytes): int8 quantized values
//	bsums (32 bytes): 16 x int16 sub-block sums, little-endian
//
// For each block of 256 values:
//
//	d = max(|input[i]|) / 127.0
//	qs[i] = round(input[i] / d) clamped to [-128, 127]
//	bsums[j] = sum(qs[j*16 .. j*16+15])
func BaseQuantizeQ8_K(input []float32, output []uint8) {
	if len(input) == 0 {
		return
	}
	nblocks := len(input) / QK_K

	lanes := hwy.NumLanes[float32]()
	buf := make([]float32, lanes)
	minVec := hwy.Set[float32](-128.0)
	maxVec := hwy.Set[float32](127.0)

	for b := range nblocks {
		inOff := b * QK_K
		outOff := b * BlockSizeQ8K
		block := output[outOff : outOff+BlockSizeQ8K]

		// Find amax = max(|input[i]|) over the block.
		amax := float32(0)
		i := 0
		for ; i+lanes <= QK_K; i += lanes {
			v := hwy.Load(input[inOff+i:])
			absV := hwy.Abs(v)
			hwy.Store(absV, buf)
			for j := range lanes {
				if buf[j] > amax {
					amax = buf[j]
				}
			}
		}
		for ; i < QK_K; i++ {
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

		// Encode d as float32 little-endian (4 bytes).
		f32bits := math.Float32bits(d)
		block[0] = uint8(f32bits)
		block[1] = uint8(f32bits >> 8)
		block[2] = uint8(f32bits >> 16)
		block[3] = uint8(f32bits >> 24)

		// Quantize values: round(input * id), clamped to [-128, 127].
		qs := block[4 : 4+QK_K]
		i = 0
		idVec := hwy.Set(id)
		for ; i+lanes <= QK_K; i += lanes {
			v := hwy.Load(input[inOff+i:])
			scaled := hwy.Mul(v, idVec)
			clamped := hwy.Clamp(hwy.Round(scaled), minVec, maxVec)
			hwy.Store(clamped, buf)
			for j := range lanes {
				qs[i+j] = uint8(int8(buf[j]))
			}
		}
		for ; i < QK_K; i++ {
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

		// Compute 16 bsums: sum of 16 consecutive qs values, stored as int16 LE.
		bsums := block[4+QK_K:]
		for j := range 16 {
			var sum int16
			for k := range 16 {
				sum += int16(int8(qs[j*16+k]))
			}
			bsums[j*2] = uint8(uint16(sum))
			bsums[j*2+1] = uint8(uint16(sum) >> 8)
		}
	}
}
