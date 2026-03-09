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

// Per-sub-block quant extraction helpers for SMOPA panel packing.
//
// Each function extracts raw integer quantization values from a GGUF block
// for a single sub-block. These are used to fill SMOPA panels before
// calling the tile kernel.
//
// Naming convention:
//   - Q4_K, Q5_K, Q2_K: unsigned quants → dst is []uint8 (for SUMOPA)
//   - Q6_K, Q3_K: signed quants (after bias subtraction) → dst is []int8 (for SMOPA)

// SubBlockInfo describes the sub-block structure for a K-quant type.
type SubBlockInfo struct {
	NumSubBlocks int // number of sub-blocks per super-block
	SubBlockSize int // number of values per sub-block
	Signed       bool // true if quants are signed (Q6_K, Q3_K)
}

// GetSubBlockInfo returns sub-block structure for a K-quant type.
func GetSubBlockInfo(qt QuantType) SubBlockInfo {
	switch qt {
	case TypeQ4_K:
		return SubBlockInfo{NumSubBlocks: 8, SubBlockSize: 32, Signed: false}
	case TypeQ5_K:
		return SubBlockInfo{NumSubBlocks: 8, SubBlockSize: 32, Signed: false}
	case TypeQ6_K:
		return SubBlockInfo{NumSubBlocks: 16, SubBlockSize: 16, Signed: true}
	case TypeQ2_K:
		return SubBlockInfo{NumSubBlocks: 16, SubBlockSize: 16, Signed: false}
	case TypeQ3_K:
		return SubBlockInfo{NumSubBlocks: 16, SubBlockSize: 16, Signed: true}
	default:
		return SubBlockInfo{}
	}
}

// --- Q4_K extraction ---
// Block layout: d(2) + dmin(2) + scales(12) + qs(128)
// 8 sub-blocks of 32 values. Quants are 4-bit unsigned [0, 15].
// Sub-blocks are organized as 4 chunks of 64 values (low/high nibbles).

// extractQ4KSubBlock extracts 32 unsigned 4-bit quants from Q4_K sub-block j.
// j ∈ [0, 8). dst must have length >= 32.
func extractQ4KSubBlock(block []uint8, j int, dst []uint8) {
	qs := block[16:144]
	chunk := j / 2
	isHigh := j % 2

	qOff := chunk * 32
	if isHigh == 0 {
		// Low nibbles.
		for i := range 32 {
			dst[i] = qs[qOff+i] & 0x0F
		}
	} else {
		// High nibbles.
		for i := range 32 {
			dst[i] = qs[qOff+i] >> 4
		}
	}
}

// unpackQ4KScaleMins unpacks 8 (scale, min) pairs from Q4_K's 12-byte packed format.
func unpackQ4KScaleMins(scales []uint8, scs, mns *[8]float32) {
	for j := range 4 {
		scs[j] = float32(scales[j] & 63)
		mns[j] = float32(scales[j+4] & 63)
	}
	for j := 4; j < 8; j++ {
		scs[j] = float32((scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4))
		mns[j] = float32((scales[j+4] >> 4) | ((scales[j] >> 6) << 4))
	}
}

// --- Q5_K extraction ---
// Block layout: d(2) + dmin(2) + scales(12) + qs(128) + qh(32)
// 8 sub-blocks of 32 values. Quants are 5-bit unsigned [0, 31].
// Same scale packing as Q4_K, plus high bits in qh.

// extractQ5KSubBlock extracts 32 unsigned 5-bit quants from Q5_K sub-block j.
// j ∈ [0, 8). dst must have length >= 32.
func extractQ5KSubBlock(block []uint8, j int, dst []uint8) {
	ql := block[16:144]
	qh := block[144:176]

	chunk := j / 2
	isHigh := j % 2

	qlOff := chunk * 32
	hbShift := uint(j)

	if isHigh == 0 {
		// Low nibbles + high bit.
		for i := range 32 {
			dst[i] = (ql[qlOff+i] & 0x0F) + ((qh[i]>>hbShift)&1)*16
		}
	} else {
		// High nibbles + high bit.
		for i := range 32 {
			dst[i] = (ql[qlOff+i] >> 4) + ((qh[i]>>hbShift)&1)*16
		}
	}
}

// --- Q2_K extraction ---
// Block layout: scales(16) + d(2) + dmin(2) + qs(64)
// 16 sub-blocks of 16 values. Quants are 2-bit unsigned [0, 3].

// extractQ2KSubBlock extracts 16 unsigned 2-bit quants from Q2_K sub-block j.
// j ∈ [0, 16). dst must have length >= 16.
func extractQ2KSubBlock(block []uint8, j int, dst []uint8) {
	qs := block[20:84]

	chunk := j / 8
	group := (j % 8) / 2
	lBase := (j % 2) * 16
	qBase := chunk * 32
	shift := uint(group * 2)

	for i := range 16 {
		dst[i] = (qs[qBase+lBase+i] >> shift) & 3
	}
}

// extractQ2KScaleMin extracts the scale and min for Q2_K sub-block j.
func extractQ2KScaleMin(block []uint8, j int) (sc, mn float32) {
	scalesRaw := block[0:16]
	sc = float32(scalesRaw[j] & 0x0F)
	mn = float32(scalesRaw[j] >> 4)
	return sc, mn
}

// --- Q6_K extraction ---
// Block layout: ql(128) + qh(64) + scales(16) + d(2)
// 16 sub-blocks of 16 values. Quants are 6-bit signed [-32, 31] (after -32 bias).

// extractQ6KSubBlock extracts 16 signed 6-bit quants from Q6_K sub-block j.
// j ∈ [0, 16). dst must have length >= 16. Values are in [-32, 31].
func extractQ6KSubBlock(block []uint8, j int, dst []int8) {
	ql := block[0:128]
	qh := block[128:192]

	half := j / 8
	group := (j % 8) / 2
	lBase := (j % 2) * 16
	qlOff := half*64 + (group&1)*32
	qhOff := half * 32
	qhShift := uint(group * 2)
	nibbleShift := uint((group / 2) * 4)

	for i := range 16 {
		l := lBase + i
		low4 := int((ql[qlOff+l] >> nibbleShift) & 0xF)
		high2 := int((qh[qhOff+l] >> qhShift) & 3)
		dst[i] = int8((low4 | (high2 << 4)) - 32)
	}
}

// --- Q3_K extraction ---
// Block layout: hmask(32) + qs(64) + scales(12) + d(2)
// 16 sub-blocks of 16 values. Quants are 3-bit signed [-4, 3] (after -4 bias).

// extractQ3KSubBlock extracts 16 signed 3-bit quants from Q3_K sub-block j.
// j ∈ [0, 16). dst must have length >= 16. Values are in [-4, 3].
func extractQ3KSubBlock(block []uint8, j int, dst []int8) {
	hmask := block[0:32]
	qs := block[32:96]

	chunk := j / 8
	group := (j % 8) / 2
	lBase := (j % 2) * 16
	qBase := chunk * 32
	shift := uint(group * 2)
	hmBit := uint(chunk*4 + group)

	for i := range 16 {
		l := lBase + i
		low2 := int((qs[qBase+l] >> shift) & 3)
		high1 := int((hmask[l] >> hmBit) & 1)
		dst[i] = int8(low2 + high1*4 - 4)
	}
}

// unpackQ3KScales unpacks 16 six-bit scale values from Q3_K's 12-byte packed format.
// Each scale has -32 subtracted during use: effective_scale = d * (rawScales[j] - 32).
func unpackQ3KScales(scaleData []uint8, rawScales *[16]int) {
	for i := range 4 {
		rawScales[i] = int(scaleData[i]&0x0F) | (int(scaleData[8+i]&0x03) << 4)
		rawScales[i+4] = int(scaleData[4+i]&0x0F) | (int((scaleData[8+i]>>2)&0x03) << 4)
		rawScales[i+8] = int((scaleData[i]>>4)&0x0F) | (int((scaleData[8+i]>>4)&0x03) << 4)
		rawScales[i+12] = int((scaleData[4+i]>>4)&0x0F) | (int((scaleData[8+i]>>6)&0x03) << 4)
	}
}
