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

//go:generate go run ../../../cmd/hwygen -input vecdot_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch ggufvecdot

import (
	"math"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseVecDotQ4_0Q8_0 computes the dot product between Q4_0 weight blocks and
// Q8_0 activation blocks. Returns the scalar dot product.
//
// For each block pair:
//
//	result += scale_w * scale_a * sum_i(dequant_w[i] * quant_a[i])
//
// The inner sum uses float32 SIMD accumulation. All integer values fit exactly
// in float32 (weight quants [-8,7], activation quants [-128,127], products max
// 889, block sum max 28448), so this is numerically equivalent to integer
// accumulation.
func BaseVecDotQ4_0Q8_0(wdata []uint8, adata []uint8, nblocks int) float32 {
	var sumf float32

	lanes := hwy.NumLanes[float32]()
	wbuf := make([]float32, lanes)
	abuf := make([]float32, lanes)

	for b := range nblocks {
		wb := wdata[b*BlockSizeQ4_0 : (b+1)*BlockSizeQ4_0]
		ab := adata[b*BlockSizeQ8_0 : (b+1)*BlockSizeQ8_0]

		// Extract fp16 weight scale.
		raw := uint32(wb[0]) | uint32(wb[1])<<8
		sign := raw >> 15
		exp := (raw >> 10) & 0x1F
		mant := raw & 0x3FF
		var dw float32
		if exp == 0 {
			dw = math.Float32frombits(sign << 31)
		} else {
			dw = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}

		// Extract fp16 activation scale.
		raw = uint32(ab[0]) | uint32(ab[1])<<8
		sign = raw >> 15
		exp = (raw >> 10) & 0x1F
		mant = raw & 0x3FF
		var da float32
		if exp == 0 {
			da = math.Float32frombits(sign << 31)
		} else {
			da = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}

		wqs := wb[2:]  // 16 nibble bytes
		aqs := ab[2:]  // 32 int8 quants

		// Vectorized dot product over 32 values.
		accVec := hwy.Zero[float32]()

		// Low nibbles: weight values 0..15 against activation values 0..15.
		i := 0
		for ; i+lanes <= 16; i += lanes {
			for j := range lanes {
				wbuf[j] = float32(int(wqs[i+j]&0x0F) - 8)
				abuf[j] = float32(int8(aqs[i+j]))
			}
			wVec := hwy.Load(wbuf)
			aVec := hwy.Load(abuf)
			accVec = hwy.MulAdd(wVec, aVec, accVec)
		}
		// Scalar tail for low nibbles.
		var tailSum float32
		for ; i < 16; i++ {
			tailSum += float32(int(wqs[i]&0x0F)-8) * float32(int8(aqs[i]))
		}

		// High nibbles: weight values 16..31 against activation values 16..31.
		i = 0
		for ; i+lanes <= 16; i += lanes {
			for j := range lanes {
				wbuf[j] = float32(int((wqs[i+j]>>4)&0x0F) - 8)
				abuf[j] = float32(int8(aqs[16+i+j]))
			}
			wVec := hwy.Load(wbuf)
			aVec := hwy.Load(abuf)
			accVec = hwy.MulAdd(wVec, aVec, accVec)
		}
		for ; i < 16; i++ {
			tailSum += float32(int((wqs[i]>>4)&0x0F)-8) * float32(int8(aqs[16+i]))
		}

		// Horizontal sum of accumulator vector.
		hwy.Store(accVec, wbuf)
		blockSum := tailSum
		for j := range lanes {
			blockSum += wbuf[j]
		}

		sumf += dw * da * blockSum
	}
	return sumf
}

// BaseVecDotQ8_0Q8_0 computes the dot product between Q8_0 weight blocks and
// Q8_0 activation blocks. Returns the scalar dot product.
//
// For each block pair:
//
//	result += scale_w * scale_a * sum_i(int8(wqs[i]) * int8(aqs[i]))
func BaseVecDotQ8_0Q8_0(wdata []uint8, adata []uint8, nblocks int) float32 {
	var sumf float32

	lanes := hwy.NumLanes[float32]()
	wbuf := make([]float32, lanes)
	abuf := make([]float32, lanes)

	for b := range nblocks {
		wb := wdata[b*BlockSizeQ8_0 : (b+1)*BlockSizeQ8_0]
		ab := adata[b*BlockSizeQ8_0 : (b+1)*BlockSizeQ8_0]

		// Extract fp16 weight scale.
		raw := uint32(wb[0]) | uint32(wb[1])<<8
		sign := raw >> 15
		exp := (raw >> 10) & 0x1F
		mant := raw & 0x3FF
		var dw float32
		if exp == 0 {
			dw = math.Float32frombits(sign << 31)
		} else {
			dw = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}

		// Extract fp16 activation scale.
		raw = uint32(ab[0]) | uint32(ab[1])<<8
		sign = raw >> 15
		exp = (raw >> 10) & 0x1F
		mant = raw & 0x3FF
		var da float32
		if exp == 0 {
			da = math.Float32frombits(sign << 31)
		} else {
			da = math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
		}

		wqs := wb[2:] // 32 int8 quants
		aqs := ab[2:] // 32 int8 quants

		// Vectorized dot product over 32 values.
		accVec := hwy.Zero[float32]()

		i := 0
		for ; i+lanes <= QK; i += lanes {
			for j := range lanes {
				wbuf[j] = float32(int8(wqs[i+j]))
				abuf[j] = float32(int8(aqs[i+j]))
			}
			wVec := hwy.Load(wbuf)
			aVec := hwy.Load(abuf)
			accVec = hwy.MulAdd(wVec, aVec, accVec)
		}

		// Horizontal sum of accumulator vector + scalar tail.
		hwy.Store(accVec, wbuf)
		var blockSum float32
		for j := range lanes {
			blockSum += wbuf[j]
		}
		for ; i < QK; i++ {
			blockSum += float32(int8(wqs[i])) * float32(int8(aqs[i]))
		}

		sumf += dw * da * blockSum
	}
	return sumf
}
