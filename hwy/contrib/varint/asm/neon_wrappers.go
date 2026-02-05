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

//go:build !noasm && arm64

// NEON-optimized varint (LEB128) operations for ARM64
// These provide SIMD acceleration for finding varint boundaries and decoding.
package asm

import "unsafe"

//go:generate go tool goat ../c/varint_neon_arm64.c -O3 --target arm64 -e=-march=armv8-a+simd+fp

// ============================================================================
// SIMD Varint Boundary Detection
// ============================================================================

// FindVarintEnds returns a bitmask where bit i is set if src[i] < 0x80.
// This identifies varint terminator bytes (bytes without the continuation bit set).
// Examines up to min(len(src), 64) bytes.
func FindVarintEnds(src []byte) uint64 {
	if len(src) == 0 {
		return 0
	}
	var result int64
	n := int64(len(src))
	find_varint_ends_u8(unsafe.Pointer(&src[0]), n, unsafe.Pointer(&result))
	return uint64(result)
}

// ============================================================================
// Batch Varint Decoding
// ============================================================================

// DecodeUvarint64Batch decodes up to n unsigned varints from src into dst.
// Returns the number of values decoded and bytes consumed.
// Decoding stops on incomplete varint, overflow, or when n values are decoded.
func DecodeUvarint64Batch(src []byte, dst []uint64, n int) (decoded, consumed int) {
	if len(src) == 0 || len(dst) == 0 || n <= 0 {
		return 0, 0
	}
	var decodedOut, consumedOut int64
	srcLen := int64(len(src))
	dstLen := int64(len(dst))
	nInt := int64(n)
	decode_uvarint64_batch(
		unsafe.Pointer(&src[0]), srcLen,
		unsafe.Pointer(&dst[0]), dstLen,
		nInt,
		unsafe.Pointer(&decodedOut), unsafe.Pointer(&consumedOut),
	)
	return int(decodedOut), int(consumedOut)
}

// ============================================================================
// Group Varint Decoding
// ============================================================================

// DecodeGroupVarint32 decodes 4 uint32 values from group varint format.
// Group varint uses a 1-byte control header followed by variable-length values.
// Returns the 4 decoded values and bytes consumed (0 if error/incomplete).
func DecodeGroupVarint32(src []byte) (values [4]uint32, consumed int) {
	if len(src) == 0 {
		return [4]uint32{}, 0
	}
	var consumedOut int64
	decode_group_varint32(
		unsafe.Pointer(&src[0]), int64(len(src)),
		unsafe.Pointer(&values[0]),
		unsafe.Pointer(&consumedOut),
	)
	return values, int(consumedOut)
}

// DecodeGroupVarint64 decodes 4 uint64 values from group varint format.
// Group varint uses a 2-byte control header followed by variable-length values.
// Returns the 4 decoded values and bytes consumed (0 if error/incomplete).
func DecodeGroupVarint64(src []byte) (values [4]uint64, consumed int) {
	if len(src) == 0 {
		return [4]uint64{}, 0
	}
	var consumedOut int64
	decode_group_varint64(
		unsafe.Pointer(&src[0]), int64(len(src)),
		unsafe.Pointer(&values[0]),
		unsafe.Pointer(&consumedOut),
	)
	return values, int(consumedOut)
}

// ============================================================================
// Single Varint Decoding
// ============================================================================

// DecodeUvarint64 decodes a single unsigned varint from src.
// Returns the decoded value and bytes consumed (0 if incomplete/error).
func DecodeUvarint64(src []byte) (value uint64, consumed int) {
	if len(src) == 0 {
		return 0, 0
	}
	var consumedOut int64
	decode_uvarint64(
		unsafe.Pointer(&src[0]), int64(len(src)),
		unsafe.Pointer(&value),
		unsafe.Pointer(&consumedOut),
	)
	return value, int(consumedOut)
}

// ============================================================================
// Multi-Varint Decoding (Fixed Count)
// ============================================================================

// Decode2Uvarint64 decodes exactly 2 unsigned varints from src.
// This is optimized for the freq/norm pair pattern in search indexes.
// Returns the 2 decoded values and bytes consumed (0 if incomplete/error).
func Decode2Uvarint64(src []byte) (v1, v2 uint64, consumed int) {
	if len(src) == 0 {
		return 0, 0, 0
	}
	var consumedOut int64
	decode_2uvarint64(
		unsafe.Pointer(&src[0]), int64(len(src)),
		unsafe.Pointer(&v1), unsafe.Pointer(&v2),
		unsafe.Pointer(&consumedOut),
	)
	return v1, v2, int(consumedOut)
}

// Decode5Uvarint64 decodes exactly 5 unsigned varints from src.
// This is optimized for the location fields pattern in search indexes.
// Returns the 5 decoded values and bytes consumed (0 if incomplete/error).
func Decode5Uvarint64(src []byte) (values [5]uint64, consumed int) {
	if len(src) == 0 {
		return [5]uint64{}, 0
	}
	var consumedOut int64
	decode_5uvarint64(
		unsafe.Pointer(&src[0]), int64(len(src)),
		unsafe.Pointer(&values[0]),
		unsafe.Pointer(&consumedOut),
	)
	return values, int(consumedOut)
}

// ============================================================================
// Stream-VByte SIMD Decoding
// ============================================================================

// DecodeStreamVByte32Into decodes uint32 values from Stream-VByte format using SIMD.
// This uses NEON TBL instruction for shuffle-based decoding of 4 values at a time.
// control: control bytes (1 per 4 values)
// data: packed value bytes
// dst: output slice for decoded values
// Returns (values decoded, data bytes consumed).
func DecodeStreamVByte32Into(control, data []byte, dst []uint32) (decoded, dataConsumed int) {
	if len(control) == 0 || len(data) == 0 || len(dst) == 0 {
		return 0, 0
	}
	var dataConsumedOut int64
	n := int64(len(dst))
	decode_streamvbyte32_batch(
		unsafe.Pointer(&control[0]), int64(len(control)),
		unsafe.Pointer(&data[0]), int64(len(data)),
		unsafe.Pointer(&dst[0]), n,
		unsafe.Pointer(&dataConsumedOut),
	)
	// Calculate how many values were decoded based on data consumed
	numGroups := int64(len(control))
	maxVals := min(numGroups*4, n)
	return int(maxVals), int(dataConsumedOut)
}

// Assembly function declarations are in varint_neon_arm64.go (generated by GoAT)
