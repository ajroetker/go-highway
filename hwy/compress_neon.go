// Copyright 2025 The Go Highway Authors
// SPDX-License-Identifier: Apache-2.0
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

//go:build arm64

package hwy

import "github.com/ajroetker/go-highway/hwy/asm"

// BitsFromMask_NEON_Uint8x16 converts a NEON byte comparison result to a bitmask.
// Each byte in the input is either 0xFF (true) or 0x00 (false).
// Returns a 16-bit mask where bit i is set if byte i has the high bit set.
func BitsFromMask_NEON_Uint8x16(mask asm.Uint8x16) uint64 {
	return asm.BitsFromMask(mask)
}

// BitsFromMask_NEON_Float32x4 converts a NEON float32 comparison result to a bitmask.
// Each lane in the input is either all-ones (true) or all-zeros (false).
// Returns a 4-bit mask where bit i is set if lane i has the high bit set.
func BitsFromMask_NEON_Float32x4(mask asm.Int32x4) uint64 {
	var buf [4]int32
	mask.Store(&buf)
	var result uint64
	for i := range 4 {
		if buf[i] < 0 { // high bit set means negative
			result |= 1 << i
		}
	}
	return result
}

// BitsFromMask_NEON_Float64x2 converts a NEON float64 comparison result to a bitmask.
// Each lane in the input is either all-ones (true) or all-zeros (false).
// Returns a 2-bit mask where bit i is set if lane i has the high bit set.
func BitsFromMask_NEON_Float64x2(mask asm.Int64x2) uint64 {
	var buf [2]int64
	mask.Store(&buf)
	var result uint64
	for i := range 2 {
		if buf[i] < 0 { // high bit set means negative
			result |= 1 << i
		}
	}
	return result
}

// BitsFromMask_NEON_Int32x4 converts a NEON int32 comparison result to a bitmask.
func BitsFromMask_NEON_Int32x4(mask asm.Int32x4) uint64 {
	return BitsFromMask_NEON_Float32x4(mask)
}

// BitsFromMask_NEON_Int64x2 converts a NEON int64 comparison result to a bitmask.
func BitsFromMask_NEON_Int64x2(mask asm.Int64x2) uint64 {
	return BitsFromMask_NEON_Float64x2(mask)
}

// BitsFromMask_NEON_Uint32x4 converts a NEON uint32 comparison result to a bitmask.
func BitsFromMask_NEON_Uint32x4(mask asm.Uint32x4) uint64 {
	var buf [4]uint32
	mask.Store(&buf)
	var result uint64
	for i := range 4 {
		if buf[i]&0x80000000 != 0 {
			result |= 1 << i
		}
	}
	return result
}

// BitsFromMask_NEON_Uint64x2 converts a NEON uint64 comparison result to a bitmask.
func BitsFromMask_NEON_Uint64x2(mask asm.Uint64x2) uint64 {
	var buf [2]uint64
	mask.Store(&buf)
	var result uint64
	for i := range 2 {
		if buf[i]&0x8000000000000000 != 0 {
			result |= 1 << i
		}
	}
	return result
}
