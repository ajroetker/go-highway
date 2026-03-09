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

//go:generate go run ../../../cmd/hwygen -input vecdot_kquant_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch ggufkqvecdot

import (
	"github.com/ajroetker/go-highway/hwy"
)

// BaseVecDotQ4_KQ8_K computes the dot product between Q4_K weight blocks and
// Q8_K activation blocks. Returns the scalar dot product.
//
// Q4_K block (144 bytes): d(fp16,2) + dmin(fp16,2) + scales(12) + qs(128 nibbles)
// Q8_K block (292 bytes): d(f32,4) + qs(256 int8) + bsums(16×int16)
//
// For each super-block pair of 256 values:
//
//	result += dw * da * sum_j(sc_j * dot(nibble_j, aqs_j))
//	        - dmin_w * da * sum_j(m_j * (bsums[2j] + bsums[2j+1]))
//
// The inner dot product uses float32 SIMD accumulation per sub-block.
// The min correction uses pre-computed bsums for O(8) instead of O(256).
func BaseVecDotQ4_KQ8_K(wdata []uint8, adata []uint8, nblocks int) float32 {
	var sumf float32

	lanes := hwy.NumLanes[float32]()
	wbuf := make([]float32, lanes)
	abuf := make([]float32, lanes)

	for b := range nblocks {
		wb := wdata[b*BlockSizeQ4K : (b+1)*BlockSizeQ4K]
		ab := adata[b*BlockSizeQ8K : (b+1)*BlockSizeQ8K]

		dw := fp16LE(wb[0], wb[1])
		dminw := fp16LE(wb[2], wb[3])
		scales := wb[4:16]
		wqs := wb[16:144]

		da := f32LE(ab[0], ab[1], ab[2], ab[3])
		aqs := ab[4 : 4+QK_K]
		bsumsData := ab[4+QK_K:]

		// Unpack 8 (scale, min) pairs from 12-byte packed format.
		// Same as BaseDequantizeQ4K.
		var scs [8]int
		var mns [8]int
		for j := range 4 {
			scs[j] = int(scales[j] & 63)
			mns[j] = int(scales[j+4] & 63)
		}
		for j := 4; j < 8; j++ {
			scs[j] = int((scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4))
			mns[j] = int((scales[j+4] >> 4) | ((scales[j] >> 6) << 4))
		}

		// Min correction via bsums: sum_j(mns[j] * (bsums[2j] + bsums[2j+1])).
		var sumMins int32
		for j := range 8 {
			bs0 := i16LE(bsumsData[j*4], bsumsData[j*4+1])
			bs1 := i16LE(bsumsData[j*4+2], bsumsData[j*4+3])
			sumMins += int32(mns[j]) * int32(bs0+bs1)
		}

		// Float32 SIMD dot product per sub-block.
		// 4 chunks of 64 values (2 sub-blocks of 32 each).
		var blockDot float32
		qOff := 0
		aOff := 0
		for chunk := range 4 {
			is := chunk * 2

			// Sub-block is: low nibbles × activation.
			accVec := hwy.Zero[float32]()
			i := 0
			for ; i+lanes <= 32; i += lanes {
				for j := range lanes {
					wbuf[j] = float32(wqs[qOff+i+j] & 0xF)
					abuf[j] = float32(int8(aqs[aOff+i+j]))
				}
				wVec := hwy.Load(wbuf)
				aVec := hwy.Load(abuf)
				accVec = hwy.MulAdd(wVec, aVec, accVec)
			}
			hwy.Store(accVec, wbuf)
			var subSum float32
			for j := range lanes {
				subSum += wbuf[j]
			}
			for ; i < 32; i++ {
				subSum += float32(wqs[qOff+i]&0xF) * float32(int8(aqs[aOff+i]))
			}
			blockDot += float32(scs[is]) * subSum

			// Sub-block is+1: high nibbles × activation.
			accVec = hwy.Zero[float32]()
			i = 0
			for ; i+lanes <= 32; i += lanes {
				for j := range lanes {
					wbuf[j] = float32(wqs[qOff+i+j] >> 4)
					abuf[j] = float32(int8(aqs[aOff+32+i+j]))
				}
				wVec := hwy.Load(wbuf)
				aVec := hwy.Load(abuf)
				accVec = hwy.MulAdd(wVec, aVec, accVec)
			}
			hwy.Store(accVec, wbuf)
			subSum = 0
			for j := range lanes {
				subSum += wbuf[j]
			}
			for ; i < 32; i++ {
				subSum += float32(wqs[qOff+i]>>4) * float32(int8(aqs[aOff+32+i]))
			}
			blockDot += float32(scs[is+1]) * subSum

			qOff += 32
			aOff += 64
		}

		sumf += dw*da*blockDot - dminw*da*float32(sumMins)
	}
	return sumf
}

// BaseVecDotQ6_KQ8_K computes the dot product between Q6_K weight blocks and
// Q8_K activation blocks. Returns the scalar dot product.
//
// Q6_K block (210 bytes): ql(128) + qh(64) + scales(16×int8) + d(fp16,2)
// Q8_K block (292 bytes): d(f32,4) + qs(256 int8) + bsums(16×int16)
//
// For each super-block pair of 256 values (16 sub-blocks of 16):
//
//	result += dw * da * sum_j(sc_j * dot((q6_j - 32), aqs_j))
//
// Q6_K has no min correction (no dmin).
func BaseVecDotQ6_KQ8_K(wdata []uint8, adata []uint8, nblocks int) float32 {
	var sumf float32

	lanes := hwy.NumLanes[float32]()
	wbuf := make([]float32, lanes)
	abuf := make([]float32, lanes)

	for b := range nblocks {
		wb := wdata[b*BlockSizeQ6K : (b+1)*BlockSizeQ6K]
		ab := adata[b*BlockSizeQ8K : (b+1)*BlockSizeQ8K]

		ql := wb[0:128]
		qh := wb[128:192]
		sc := wb[192:208]
		dw := fp16LE(wb[208], wb[209])

		da := f32LE(ab[0], ab[1], ab[2], ab[3])
		aqs := ab[4 : 4+QK_K]

		// Process 16 sub-blocks of 16 values each.
		// Quant extraction mirrors BaseDequantizeQ6K.
		var blockDot float32
		for j := range 16 {
			scaleVal := float32(int8(sc[j]))

			half := j / 8
			group := (j % 8) / 2
			lBase := (j % 2) * 16
			qlOff := half*64 + (group&1)*32
			qhOff := half * 32
			qhShift := uint(group * 2)
			nibbleShift := uint((group / 2) * 4)

			aOff := j * 16

			accVec := hwy.Zero[float32]()
			i := 0
			for ; i+lanes <= 16; i += lanes {
				for k := range lanes {
					l := lBase + i + k
					low4 := int((ql[qlOff+l] >> nibbleShift) & 0xF)
					high2 := int((qh[qhOff+l] >> qhShift) & 3)
					wbuf[k] = float32((low4 | (high2 << 4)) - 32)
					abuf[k] = float32(int8(aqs[aOff+i+k]))
				}
				wVec := hwy.Load(wbuf)
				aVec := hwy.Load(abuf)
				accVec = hwy.MulAdd(wVec, aVec, accVec)
			}
			hwy.Store(accVec, wbuf)
			var subSum float32
			for k := range lanes {
				subSum += wbuf[k]
			}
			for ; i < 16; i++ {
				l := lBase + i
				low4 := int((ql[qlOff+l] >> nibbleShift) & 0xF)
				high2 := int((qh[qhOff+l] >> qhShift) & 3)
				subSum += float32((low4|(high2<<4))-32) * float32(int8(aqs[aOff+i]))
			}
			blockDot += scaleVal * subSum
		}

		sumf += dw * da * blockDot
	}
	return sumf
}

// BaseVecDotQ2_KQ8_K computes the dot product between Q2_K weight blocks and
// Q8_K activation blocks. Returns the scalar dot product.
//
// Q2_K block (84 bytes): scales(16) + d(fp16,2) + dmin(fp16,2) + qs(64)
// Q8_K block (292 bytes): d(f32,4) + qs(256 int8) + bsums(16×int16)
//
// 16 sub-blocks of 16 values each. Quant extraction mirrors BaseDequantizeQ2K.
func BaseVecDotQ2_KQ8_K(wdata []uint8, adata []uint8, nblocks int) float32 {
	var sumf float32

	lanes := hwy.NumLanes[float32]()
	wbuf := make([]float32, lanes)
	abuf := make([]float32, lanes)

	for b := range nblocks {
		wb := wdata[b*BlockSizeQ2K : (b+1)*BlockSizeQ2K]
		ab := adata[b*BlockSizeQ8K : (b+1)*BlockSizeQ8K]

		scalesRaw := wb[0:16]
		dw := fp16LE(wb[16], wb[17])
		dminw := fp16LE(wb[18], wb[19])
		wqs := wb[20:84]

		da := f32LE(ab[0], ab[1], ab[2], ab[3])
		aqs := ab[4 : 4+QK_K]
		bsumsData := ab[4+QK_K:]

		// Min correction via bsums: sum_j(m_j * bsums[j]).
		// Q2_K has 16 sub-blocks, each with its own min (high nibble of scalesRaw).
		// Each sub-block has 16 values → one bsum entry per sub-block.
		var sumMins int32
		for j := range 16 {
			m := int32(scalesRaw[j] >> 4)
			bs := int32(i16LE(bsumsData[j*2], bsumsData[j*2+1]))
			sumMins += m * bs
		}

		// Float32 SIMD dot product per sub-block.
		var blockDot float32
		for is := range 16 {
			sc := float32(scalesRaw[is] & 0x0F)

			chunk := is / 8
			group := (is % 8) / 2
			lBase := (is % 2) * 16
			qBase := chunk * 32
			shift := uint(group * 2)

			aOff := is * 16

			accVec := hwy.Zero[float32]()
			i := 0
			for ; i+lanes <= 16; i += lanes {
				for k := range lanes {
					wbuf[k] = float32((wqs[qBase+lBase+i+k] >> shift) & 3)
					abuf[k] = float32(int8(aqs[aOff+i+k]))
				}
				wVec := hwy.Load(wbuf)
				aVec := hwy.Load(abuf)
				accVec = hwy.MulAdd(wVec, aVec, accVec)
			}
			hwy.Store(accVec, wbuf)
			var subSum float32
			for k := range lanes {
				subSum += wbuf[k]
			}
			for ; i < 16; i++ {
				subSum += float32((wqs[qBase+lBase+i]>>shift)&3) * float32(int8(aqs[aOff+i]))
			}
			blockDot += sc * subSum
		}

		sumf += dw*da*blockDot - dminw*da*float32(sumMins)
	}
	return sumf
}

// BaseVecDotQ3_KQ8_K computes the dot product between Q3_K weight blocks and
// Q8_K activation blocks. Returns the scalar dot product.
//
// Q3_K block (110 bytes): hmask(32) + qs(64) + scales(12) + d(fp16,2)
// Q8_K block (292 bytes): d(f32,4) + qs(256 int8) + bsums(16×int16)
//
// 16 sub-blocks of 16 values with 6-bit packed scales and hmask high bits.
// Q3_K has no min correction (uses scale-32 centering instead).
func BaseVecDotQ3_KQ8_K(wdata []uint8, adata []uint8, nblocks int) float32 {
	var sumf float32

	lanes := hwy.NumLanes[float32]()
	wbuf := make([]float32, lanes)
	abuf := make([]float32, lanes)

	for b := range nblocks {
		wb := wdata[b*BlockSizeQ3K : (b+1)*BlockSizeQ3K]
		ab := adata[b*BlockSizeQ8K : (b+1)*BlockSizeQ8K]

		hmask := wb[0:32]
		wqs := wb[32:96]
		scaleData := wb[96:108]
		dw := fp16LE(wb[108], wb[109])

		da := f32LE(ab[0], ab[1], ab[2], ab[3])
		aqs := ab[4 : 4+QK_K]

		// Unpack 16 six-bit scale values from 12 bytes.
		// Same as BaseDequantizeQ3K.
		var rawScales [16]int
		for i := range 4 {
			rawScales[i] = int(scaleData[i]&0x0F) | (int(scaleData[8+i]&0x03) << 4)
			rawScales[i+4] = int(scaleData[4+i]&0x0F) | (int((scaleData[8+i]>>2)&0x03) << 4)
			rawScales[i+8] = int((scaleData[i]>>4)&0x0F) | (int((scaleData[8+i]>>4)&0x03) << 4)
			rawScales[i+12] = int((scaleData[4+i]>>4)&0x0F) | (int((scaleData[8+i]>>6)&0x03) << 4)
		}

		// Float32 SIMD dot product per sub-block.
		var blockDot float32
		for j := range 16 {
			scaleVal := float32(rawScales[j] - 32)

			chunk := j / 8
			group := (j % 8) / 2
			lBase := (j % 2) * 16
			qBase := chunk * 32
			shift := uint(group * 2)
			hmBit := uint(chunk*4 + group)

			aOff := j * 16

			accVec := hwy.Zero[float32]()
			i := 0
			for ; i+lanes <= 16; i += lanes {
				for k := range lanes {
					l := lBase + i + k
					low2 := int((wqs[qBase+l] >> shift) & 3)
					high1 := int((hmask[l] >> hmBit) & 1)
					wbuf[k] = float32(low2 + high1*4 - 4)
					abuf[k] = float32(int8(aqs[aOff+i+k]))
				}
				wVec := hwy.Load(wbuf)
				aVec := hwy.Load(abuf)
				accVec = hwy.MulAdd(wVec, aVec, accVec)
			}
			hwy.Store(accVec, wbuf)
			var subSum float32
			for k := range lanes {
				subSum += wbuf[k]
			}
			for ; i < 16; i++ {
				l := lBase + i
				low2 := int((wqs[qBase+l] >> shift) & 3)
				high1 := int((hmask[l] >> hmBit) & 1)
				subSum += float32(low2+high1*4-4) * float32(int8(aqs[aOff+i]))
			}
			blockDot += scaleVal * subSum
		}

		sumf += dw * da * blockDot
	}
	return sumf
}

// BaseVecDotQ5_KQ8_K computes the dot product between Q5_K weight blocks and
// Q8_K activation blocks. Returns the scalar dot product.
//
// Q5_K block (176 bytes): d(fp16,2) + dmin(fp16,2) + scales(12) + qs(128) + qh(32)
// Q8_K block (292 bytes): d(f32,4) + qs(256 int8) + bsums(16×int16)
//
// 8 sub-blocks of 32 values. Same scale packing as Q4_K, plus high bits in qh.
func BaseVecDotQ5_KQ8_K(wdata []uint8, adata []uint8, nblocks int) float32 {
	var sumf float32

	lanes := hwy.NumLanes[float32]()
	wbuf := make([]float32, lanes)
	abuf := make([]float32, lanes)

	for b := range nblocks {
		wb := wdata[b*BlockSizeQ5K : (b+1)*BlockSizeQ5K]
		ab := adata[b*BlockSizeQ8K : (b+1)*BlockSizeQ8K]

		dw := fp16LE(wb[0], wb[1])
		dminw := fp16LE(wb[2], wb[3])
		scales := wb[4:16]
		ql := wb[16:144]
		qh := wb[144:176]

		da := f32LE(ab[0], ab[1], ab[2], ab[3])
		aqs := ab[4 : 4+QK_K]
		bsumsData := ab[4+QK_K:]

		// Unpack 8 (scale, min) pairs — same as Q4_K.
		var scs [8]int
		var mns [8]int
		for j := range 4 {
			scs[j] = int(scales[j] & 63)
			mns[j] = int(scales[j+4] & 63)
		}
		for j := 4; j < 8; j++ {
			scs[j] = int((scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4))
			mns[j] = int((scales[j+4] >> 4) | ((scales[j] >> 6) << 4))
		}

		// Min correction via bsums.
		var sumMins int32
		for j := range 8 {
			bs0 := i16LE(bsumsData[j*4], bsumsData[j*4+1])
			bs1 := i16LE(bsumsData[j*4+2], bsumsData[j*4+3])
			sumMins += int32(mns[j]) * int32(bs0+bs1)
		}

		// Float32 SIMD dot product per sub-block.
		var blockDot float32
		qlOff := 0
		aOff := 0
		for chunk := range 4 {
			is := chunk * 2
			hbShift0 := uint(chunk * 2)
			hbShift1 := uint(chunk*2 + 1)

			// Sub-block is: low nibbles + high bit × activation.
			accVec := hwy.Zero[float32]()
			i := 0
			for ; i+lanes <= 32; i += lanes {
				for k := range lanes {
					l := i + k
					q := int(ql[qlOff+l]&0xF) + int((qh[l]>>hbShift0)&1)*16
					wbuf[k] = float32(q)
					abuf[k] = float32(int8(aqs[aOff+i+k]))
				}
				wVec := hwy.Load(wbuf)
				aVec := hwy.Load(abuf)
				accVec = hwy.MulAdd(wVec, aVec, accVec)
			}
			hwy.Store(accVec, wbuf)
			var subSum float32
			for k := range lanes {
				subSum += wbuf[k]
			}
			for ; i < 32; i++ {
				q := int(ql[qlOff+i]&0xF) + int((qh[i]>>hbShift0)&1)*16
				subSum += float32(q) * float32(int8(aqs[aOff+i]))
			}
			blockDot += float32(scs[is]) * subSum

			// Sub-block is+1: high nibbles + high bit × activation.
			accVec = hwy.Zero[float32]()
			i = 0
			for ; i+lanes <= 32; i += lanes {
				for k := range lanes {
					l := i + k
					q := int(ql[qlOff+l]>>4) + int((qh[l]>>hbShift1)&1)*16
					wbuf[k] = float32(q)
					abuf[k] = float32(int8(aqs[aOff+32+i+k]))
				}
				wVec := hwy.Load(wbuf)
				aVec := hwy.Load(abuf)
				accVec = hwy.MulAdd(wVec, aVec, accVec)
			}
			hwy.Store(accVec, wbuf)
			subSum = 0
			for k := range lanes {
				subSum += wbuf[k]
			}
			for ; i < 32; i++ {
				q := int(ql[qlOff+i]>>4) + int((qh[i]>>hbShift1)&1)*16
				subSum += float32(q) * float32(int8(aqs[aOff+32+i]))
			}
			blockDot += float32(scs[is+1]) * subSum

			qlOff += 32
			aOff += 64
		}

		sumf += dw*da*blockDot - dminw*da*float32(sumMins)
	}
	return sumf
}
