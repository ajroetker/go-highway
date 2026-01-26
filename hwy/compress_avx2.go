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

//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides AVX2 SIMD implementations of compress and expand operations.
// These work directly with archsimd vector and mask types.
// Since archsimd doesn't have vcompressps, we use store/scalar/load pattern.
//
// Note: archsimd Mask32x8.ToBits() requires AVX-512 (KMOV instructions).
// For AVX2-only machines, we use helper functions that extract mask bits
// by converting to Int32x8 and checking sign bits.
// mask32x8ToBits converts a Mask32x8 to a bitmask without using AVX-512.
// This works on AVX2-only machines by converting to Int32x8 (all-ones for true,
// all-zeros for false) and extracting the sign bits.
func mask32x8ToBits(mask archsimd.Mask32x8) uint8 {
	// Convert mask to int32x8 (true lanes = -1 = 0xFFFFFFFF, false = 0)
	v := mask.ToInt32x8()
	var data [8]int32
	v.StoreSlice(data[:])

	var bits uint8
	for i := 0; i < 8; i++ {
		if data[i] < 0 { // Sign bit set means all-ones (true)
			bits |= 1 << i
		}
	}
	return bits
}

// mask64x4ToBits converts a Mask64x4 to a bitmask without using AVX-512.
func mask64x4ToBits(mask archsimd.Mask64x4) uint8 {
	v := mask.ToInt64x4()
	var data [4]int64
	v.StoreSlice(data[:])

	var bits uint8
	for i := 0; i < 4; i++ {
		if data[i] < 0 {
			bits |= 1 << i
		}
	}
	return bits
}


// Compress_AVX2_F32x8 compresses elements where mask is true to the front.
// The mask should be the result of a comparison operation (e.g., Less, Equal).
// Returns compressed vector and count of valid elements.
func Compress_AVX2_F32x8(v archsimd.Float32x8, mask archsimd.Mask32x8) (archsimd.Float32x8, int) {
	var data [8]float32
	v.StoreSlice(data[:])

	bits := mask32x8ToBits(mask)
	var result [8]float32
	count := 0

	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat32x8Slice(result[:]), count
}

// Compress_AVX2_F64x4 compresses elements where mask is true to the front.
// Returns compressed vector and count of valid elements.
func Compress_AVX2_F64x4(v archsimd.Float64x4, mask archsimd.Mask64x4) (archsimd.Float64x4, int) {
	var data [4]float64
	v.StoreSlice(data[:])

	bits := mask64x4ToBits(mask)
	var result [4]float64
	count := 0

	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat64x4Slice(result[:]), count
}

// Expand_AVX2_F32x8 expands elements into positions where mask is true.
func Expand_AVX2_F32x8(v archsimd.Float32x8, mask archsimd.Mask32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])

	bits := mask32x8ToBits(mask)
	var result [8]float32
	srcIdx := 0

	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadFloat32x8Slice(result[:])
}

// Expand_AVX2_F64x4 expands elements into positions where mask is true.
func Expand_AVX2_F64x4(v archsimd.Float64x4, mask archsimd.Mask64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])

	bits := mask64x4ToBits(mask)
	var result [4]float64
	srcIdx := 0

	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadFloat64x4Slice(result[:])
}

// CompressStore_AVX2_F32x8 compresses and stores directly to slice.
// Returns number of elements stored.
func CompressStore_AVX2_F32x8(v archsimd.Float32x8, mask archsimd.Mask32x8, dst []float32) int {
	var data [8]float32
	v.StoreSlice(data[:])

	bits := mask32x8ToBits(mask)
	count := 0
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CompressStore_AVX2_F64x4 compresses and stores directly to slice.
// Returns number of elements stored.
func CompressStore_AVX2_F64x4(v archsimd.Float64x4, mask archsimd.Mask64x4, dst []float64) int {
	var data [4]float64
	v.StoreSlice(data[:])

	bits := mask64x4ToBits(mask)
	count := 0
	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CountTrue_AVX2_F32x8 counts true lanes in mask.
func CountTrue_AVX2_F32x8(mask archsimd.Mask32x8) int {
	bits := mask32x8ToBits(mask)
	count := 0
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			count++
		}
	}
	return count
}

// CountTrue_AVX2_F64x4 counts true lanes in mask.
func CountTrue_AVX2_F64x4(mask archsimd.Mask64x4) int {
	bits := mask64x4ToBits(mask)
	count := 0
	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			count++
		}
	}
	return count
}

// AllTrue_AVX2_F32x8 returns true if all lanes are true.
func AllTrue_AVX2_F32x8(mask archsimd.Mask32x8) bool {
	return mask32x8ToBits(mask) == 0xFF
}

// AllTrue_AVX2_F64x4 returns true if all lanes are true.
func AllTrue_AVX2_F64x4(mask archsimd.Mask64x4) bool {
	return mask64x4ToBits(mask) == 0xF
}

// AllFalse_AVX2_F32x8 returns true if all lanes are false.
func AllFalse_AVX2_F32x8(mask archsimd.Mask32x8) bool {
	return mask32x8ToBits(mask) == 0
}

// AllFalse_AVX2_F64x4 returns true if all lanes are false.
func AllFalse_AVX2_F64x4(mask archsimd.Mask64x4) bool {
	return mask64x4ToBits(mask) == 0
}

// FindFirstTrue_AVX2_F32x8 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX2_F32x8(mask archsimd.Mask32x8) int {
	bits := mask32x8ToBits(mask)
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			return i
		}
	}
	return -1
}

// FindFirstTrue_AVX2_F64x4 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX2_F64x4(mask archsimd.Mask64x4) int {
	bits := mask64x4ToBits(mask)
	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX2_F32x8 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX2_F32x8(mask archsimd.Mask32x8) int {
	bits := mask32x8ToBits(mask)
	for i := 7; i >= 0; i-- {
		if (bits & (1 << i)) != 0 {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX2_F64x4 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX2_F64x4(mask archsimd.Mask64x4) int {
	bits := mask64x4ToBits(mask)
	for i := 3; i >= 0; i-- {
		if (bits & (1 << i)) != 0 {
			return i
		}
	}
	return -1
}

// FirstN_AVX2_F32x8 creates a mask with the first n lanes set to true.
func FirstN_AVX2_F32x8(n int) archsimd.Mask32x8 {
	indices := archsimd.LoadInt32x8Slice([]int32{0, 1, 2, 3, 4, 5, 6, 7})
	threshold := archsimd.BroadcastInt32x8(int32(n))
	return indices.Less(threshold)
}

// FirstN_AVX2_F64x4 creates a mask with the first n lanes set to true.
func FirstN_AVX2_F64x4(n int) archsimd.Mask64x4 {
	indices := archsimd.LoadInt64x4Slice([]int64{0, 1, 2, 3})
	threshold := archsimd.BroadcastInt64x4(int64(n))
	return indices.Less(threshold)
}

// LastN_AVX2_F32x8 creates a mask with the last n lanes set to true.
// For n lanes set, we compute: bits = ((1 << n) - 1) << (8 - n)
// e.g., LastN(3) = 0b11100000 = 0xE0
func LastN_AVX2_F32x8(n int) archsimd.Mask32x8 {
	if n <= 0 {
		return archsimd.Mask32x8FromBits(0)
	}
	if n >= 8 {
		return archsimd.Mask32x8FromBits(0xFF)
	}
	bits := uint8(((1 << n) - 1) << (8 - n))
	return archsimd.Mask32x8FromBits(bits)
}

// LastN_AVX2_F64x4 creates a mask with the last n lanes set to true.
func LastN_AVX2_F64x4(n int) archsimd.Mask64x4 {
	if n <= 0 {
		return archsimd.Mask64x4FromBits(0)
	}
	if n >= 4 {
		return archsimd.Mask64x4FromBits(0xF)
	}
	bits := uint8(((1 << n) - 1) << (4 - n))
	return archsimd.Mask64x4FromBits(bits)
}

// MaskFromBits_AVX2_F32x8 creates a mask from a bitmask integer.
func MaskFromBits_AVX2_F32x8(bits uint64) archsimd.Mask32x8 {
	var vals [8]int32
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = 1
		}
	}
	vec := archsimd.LoadInt32x8Slice(vals[:])
	zero := archsimd.BroadcastInt32x8(0)
	return vec.Greater(zero)
}

// MaskFromBits_AVX2_F64x4 creates a mask from a bitmask integer.
func MaskFromBits_AVX2_F64x4(bits uint64) archsimd.Mask64x4 {
	var vals [4]int64
	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = 1
		}
	}
	vec := archsimd.LoadInt64x4Slice(vals[:])
	zero := archsimd.BroadcastInt64x4(0)
	return vec.Greater(zero)
}

// BitsFromMask_AVX2_F32x8 converts mask to bitmask integer.
func BitsFromMask_AVX2_F32x8(mask archsimd.Mask32x8) uint64 {
	return uint64(mask32x8ToBits(mask))
}

// BitsFromMask_AVX2_F64x4 converts mask to bitmask integer.
func BitsFromMask_AVX2_F64x4(mask archsimd.Mask64x4) uint64 {
	return uint64(mask64x4ToBits(mask))
}

// BitsFromMask_AVX2_Uint8x16 converts a byte mask to bitmask integer.
// Each bit in the result corresponds to one byte lane.
func BitsFromMask_AVX2_Uint8x16(mask archsimd.Mask8x16) uint64 {
	return uint64(mask.ToBits())
}

// BitsFromMask_AVX2_Uint8x32 converts a 32-byte mask to bitmask integer.
// Each bit in the result corresponds to one byte lane.
func BitsFromMask_AVX2_Uint8x32(mask archsimd.Mask8x32) uint64 {
	return uint64(mask.ToBits())
}

// ============================================================================
// Integer type variants (I32x8, I64x4)
// ============================================================================

// Compress_AVX2_I32x8 compresses elements where mask is true to the front.
func Compress_AVX2_I32x8(v archsimd.Int32x8, mask archsimd.Mask32x8) (archsimd.Int32x8, int) {
	var data [8]int32
	v.StoreSlice(data[:])

	bits := mask32x8ToBits(mask)
	var result [8]int32
	count := 0

	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadInt32x8Slice(result[:]), count
}

// Compress_AVX2_I64x4 compresses elements where mask is true to the front.
func Compress_AVX2_I64x4(v archsimd.Int64x4, mask archsimd.Mask64x4) (archsimd.Int64x4, int) {
	var data [4]int64
	v.StoreSlice(data[:])

	bits := mask64x4ToBits(mask)
	var result [4]int64
	count := 0

	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadInt64x4Slice(result[:]), count
}

// Expand_AVX2_I32x8 expands elements into positions where mask is true.
func Expand_AVX2_I32x8(v archsimd.Int32x8, mask archsimd.Mask32x8) archsimd.Int32x8 {
	var data [8]int32
	v.StoreSlice(data[:])

	bits := mask32x8ToBits(mask)
	var result [8]int32
	srcIdx := 0

	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadInt32x8Slice(result[:])
}

// Expand_AVX2_I64x4 expands elements into positions where mask is true.
func Expand_AVX2_I64x4(v archsimd.Int64x4, mask archsimd.Mask64x4) archsimd.Int64x4 {
	var data [4]int64
	v.StoreSlice(data[:])

	bits := mask64x4ToBits(mask)
	var result [4]int64
	srcIdx := 0

	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadInt64x4Slice(result[:])
}

// CompressStore_AVX2_I32x8 compresses and stores directly to slice.
func CompressStore_AVX2_I32x8(v archsimd.Int32x8, mask archsimd.Mask32x8, dst []int32) int {
	var data [8]int32
	v.StoreSlice(data[:])

	bits := mask32x8ToBits(mask)
	count := 0
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CompressStore_AVX2_I64x4 compresses and stores directly to slice.
func CompressStore_AVX2_I64x4(v archsimd.Int64x4, mask archsimd.Mask64x4, dst []int64) int {
	var data [4]int64
	v.StoreSlice(data[:])

	bits := mask64x4ToBits(mask)
	count := 0
	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CountTrue_AVX2_I32x8 counts true lanes in mask.
func CountTrue_AVX2_I32x8(mask archsimd.Mask32x8) int {
	return CountTrue_AVX2_F32x8(mask)
}

// CountTrue_AVX2_I64x4 counts true lanes in mask.
func CountTrue_AVX2_I64x4(mask archsimd.Mask64x4) int {
	return CountTrue_AVX2_F64x4(mask)
}

// FirstN_AVX2_I32x8 creates a mask with the first n lanes set to true.
func FirstN_AVX2_I32x8(n int) archsimd.Mask32x8 {
	return FirstN_AVX2_F32x8(n)
}

// FirstN_AVX2_I64x4 creates a mask with the first n lanes set to true.
func FirstN_AVX2_I64x4(n int) archsimd.Mask64x4 {
	return FirstN_AVX2_F64x4(n)
}

// LastN_AVX2_I32x8 creates a mask with the last n lanes set to true.
func LastN_AVX2_I32x8(n int) archsimd.Mask32x8 {
	return LastN_AVX2_F32x8(n)
}

// LastN_AVX2_I64x4 creates a mask with the last n lanes set to true.
func LastN_AVX2_I64x4(n int) archsimd.Mask64x4 {
	return LastN_AVX2_F64x4(n)
}

// AllTrue_AVX2_I32x8 returns true if all lanes are true.
func AllTrue_AVX2_I32x8(mask archsimd.Mask32x8) bool {
	return AllTrue_AVX2_F32x8(mask)
}

// AllTrue_AVX2_I64x4 returns true if all lanes are true.
func AllTrue_AVX2_I64x4(mask archsimd.Mask64x4) bool {
	return AllTrue_AVX2_F64x4(mask)
}

// AllFalse_AVX2_I32x8 returns true if all lanes are false.
func AllFalse_AVX2_I32x8(mask archsimd.Mask32x8) bool {
	return AllFalse_AVX2_F32x8(mask)
}

// AllFalse_AVX2_I64x4 returns true if all lanes are false.
func AllFalse_AVX2_I64x4(mask archsimd.Mask64x4) bool {
	return AllFalse_AVX2_F64x4(mask)
}

// FindFirstTrue_AVX2_I32x8 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX2_I32x8(mask archsimd.Mask32x8) int {
	return FindFirstTrue_AVX2_F32x8(mask)
}

// FindFirstTrue_AVX2_I64x4 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX2_I64x4(mask archsimd.Mask64x4) int {
	return FindFirstTrue_AVX2_F64x4(mask)
}

// FindLastTrue_AVX2_I32x8 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX2_I32x8(mask archsimd.Mask32x8) int {
	return FindLastTrue_AVX2_F32x8(mask)
}

// FindLastTrue_AVX2_I64x4 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX2_I64x4(mask archsimd.Mask64x4) int {
	return FindLastTrue_AVX2_F64x4(mask)
}

// ============================================================================
// Unsigned integer wrappers (use same masks as signed)
// ============================================================================

// Compress_AVX2_Uint32x8 compresses elements where mask is true to the front.
func Compress_AVX2_Uint32x8(v archsimd.Uint32x8, mask archsimd.Mask32x8) (archsimd.Uint32x8, int) {
	var data [8]uint32
	v.StoreSlice(data[:])

	bits := mask32x8ToBits(mask)
	var result [8]uint32
	count := 0

	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadUint32x8Slice(result[:]), count
}

// Compress_AVX2_Uint64x4 compresses elements where mask is true to the front.
func Compress_AVX2_Uint64x4(v archsimd.Uint64x4, mask archsimd.Mask64x4) (archsimd.Uint64x4, int) {
	var data [4]uint64
	v.StoreSlice(data[:])

	bits := mask64x4ToBits(mask)
	var result [4]uint64
	count := 0

	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadUint64x4Slice(result[:]), count
}

// CompressStore_AVX2_Uint32x8 compresses and stores directly to slice.
func CompressStore_AVX2_Uint32x8(v archsimd.Uint32x8, mask archsimd.Mask32x8, dst []uint32) int {
	var data [8]uint32
	v.StoreSlice(data[:])

	bits := mask32x8ToBits(mask)
	count := 0
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CompressStore_AVX2_Uint64x4 compresses and stores directly to slice.
func CompressStore_AVX2_Uint64x4(v archsimd.Uint64x4, mask archsimd.Mask64x4, dst []uint64) int {
	var data [4]uint64
	v.StoreSlice(data[:])

	bits := mask64x4ToBits(mask)
	count := 0
	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CountTrue_AVX2_Uint32x8 counts true lanes in mask.
func CountTrue_AVX2_Uint32x8(mask archsimd.Mask32x8) int {
	return CountTrue_AVX2_F32x8(mask)
}

// CountTrue_AVX2_Uint64x4 counts true lanes in mask.
func CountTrue_AVX2_Uint64x4(mask archsimd.Mask64x4) int {
	return CountTrue_AVX2_F64x4(mask)
}

// FirstN_AVX2_Uint32x8 creates a mask with the first n lanes set to true.
func FirstN_AVX2_Uint32x8(n int) archsimd.Mask32x8 {
	return FirstN_AVX2_F32x8(n)
}

// FirstN_AVX2_Uint64x4 creates a mask with the first n lanes set to true.
func FirstN_AVX2_Uint64x4(n int) archsimd.Mask64x4 {
	return FirstN_AVX2_F64x4(n)
}

// LastN_AVX2_Uint32x8 creates a mask with the last n lanes set to true.
func LastN_AVX2_Uint32x8(n int) archsimd.Mask32x8 {
	return LastN_AVX2_F32x8(n)
}

// LastN_AVX2_Uint64x4 creates a mask with the last n lanes set to true.
func LastN_AVX2_Uint64x4(n int) archsimd.Mask64x4 {
	return LastN_AVX2_F64x4(n)
}

// AllTrue_AVX2_Uint32x8 returns true if all lanes are true.
func AllTrue_AVX2_Uint32x8(mask archsimd.Mask32x8) bool {
	return AllTrue_AVX2_F32x8(mask)
}

// AllTrue_AVX2_Uint64x4 returns true if all lanes are true.
func AllTrue_AVX2_Uint64x4(mask archsimd.Mask64x4) bool {
	return AllTrue_AVX2_F64x4(mask)
}

// AllFalse_AVX2_Uint32x8 returns true if all lanes are false.
func AllFalse_AVX2_Uint32x8(mask archsimd.Mask32x8) bool {
	return AllFalse_AVX2_F32x8(mask)
}

// AllFalse_AVX2_Uint64x4 returns true if all lanes are false.
func AllFalse_AVX2_Uint64x4(mask archsimd.Mask64x4) bool {
	return AllFalse_AVX2_F64x4(mask)
}

// FindFirstTrue_AVX2_Uint32x8 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX2_Uint32x8(mask archsimd.Mask32x8) int {
	return FindFirstTrue_AVX2_F32x8(mask)
}

// FindFirstTrue_AVX2_Uint64x4 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX2_Uint64x4(mask archsimd.Mask64x4) int {
	return FindFirstTrue_AVX2_F64x4(mask)
}

// FindLastTrue_AVX2_Uint32x8 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX2_Uint32x8(mask archsimd.Mask32x8) int {
	return FindLastTrue_AVX2_F32x8(mask)
}

// FindLastTrue_AVX2_Uint64x4 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX2_Uint64x4(mask archsimd.Mask64x4) int {
	return FindLastTrue_AVX2_F64x4(mask)
}

// ============================================================================
// IfThenElse wrappers
// ============================================================================

// IfThenElse_AVX2_F32x8 selects elements from a where mask is true, else from b.
func IfThenElse_AVX2_F32x8(mask archsimd.Mask32x8, a, b archsimd.Float32x8) archsimd.Float32x8 {
	var aBuf, bBuf [8]float32
	a.StoreSlice(aBuf[:])
	b.StoreSlice(bBuf[:])

	bits := mask32x8ToBits(mask)
	var result [8]float32
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = aBuf[i]
		} else {
			result[i] = bBuf[i]
		}
	}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// IfThenElse_AVX2_F64x4 selects elements from a where mask is true, else from b.
func IfThenElse_AVX2_F64x4(mask archsimd.Mask64x4, a, b archsimd.Float64x4) archsimd.Float64x4 {
	var aBuf, bBuf [4]float64
	a.StoreSlice(aBuf[:])
	b.StoreSlice(bBuf[:])

	bits := mask64x4ToBits(mask)
	var result [4]float64
	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = aBuf[i]
		} else {
			result[i] = bBuf[i]
		}
	}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// IfThenElse_AVX2_I32x8 selects elements from a where mask is true, else from b.
func IfThenElse_AVX2_I32x8(mask archsimd.Mask32x8, a, b archsimd.Int32x8) archsimd.Int32x8 {
	var aBuf, bBuf [8]int32
	a.StoreSlice(aBuf[:])
	b.StoreSlice(bBuf[:])

	bits := mask32x8ToBits(mask)
	var result [8]int32
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = aBuf[i]
		} else {
			result[i] = bBuf[i]
		}
	}
	return archsimd.LoadInt32x8Slice(result[:])
}

// IfThenElse_AVX2_I64x4 selects elements from a where mask is true, else from b.
func IfThenElse_AVX2_I64x4(mask archsimd.Mask64x4, a, b archsimd.Int64x4) archsimd.Int64x4 {
	var aBuf, bBuf [4]int64
	a.StoreSlice(aBuf[:])
	b.StoreSlice(bBuf[:])

	bits := mask64x4ToBits(mask)
	var result [4]int64
	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = aBuf[i]
		} else {
			result[i] = bBuf[i]
		}
	}
	return archsimd.LoadInt64x4Slice(result[:])
}

// MaskAnd_AVX2_F32x8 combines two masks with AND operation.
func MaskAnd_AVX2_F32x8(a, b archsimd.Mask32x8) archsimd.Mask32x8 {
	return a.And(b)
}

// MaskAnd_AVX2_F64x4 combines two masks with AND operation.
func MaskAnd_AVX2_F64x4(a, b archsimd.Mask64x4) archsimd.Mask64x4 {
	return a.And(b)
}

// MaskOr_AVX2_F32x8 combines two masks with OR operation.
func MaskOr_AVX2_F32x8(a, b archsimd.Mask32x8) archsimd.Mask32x8 {
	return a.Or(b)
}

// MaskOr_AVX2_F64x4 combines two masks with OR operation.
func MaskOr_AVX2_F64x4(a, b archsimd.Mask64x4) archsimd.Mask64x4 {
	return a.Or(b)
}

// ============================================================================
// Bitwise operations for float types
// archsimd doesn't have Xor/Not on float types, so we cast to int and back
// ============================================================================

// SignBit_AVX2_F32x8 returns a vector with the sign bit set in all lanes.
func SignBit_AVX2_F32x8() archsimd.Float32x8 {
	// 0x80000000 is the sign bit for float32
	signBit := archsimd.BroadcastInt32x8(int32(-0x80000000))
	return signBit.AsFloat32x8()
}

// SignBit_AVX2_F64x4 returns a vector with the sign bit set in all lanes.
func SignBit_AVX2_F64x4() archsimd.Float64x4 {
	// 0x8000000000000000 is the sign bit for float64
	signBit := archsimd.BroadcastInt64x4(int64(-0x8000000000000000))
	return signBit.AsFloat64x4()
}

// Not_AVX2_F32x8 returns bitwise NOT of the vector.
func Not_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	// XOR with all 1s
	allOnes := archsimd.BroadcastInt32x8(-1)
	return v.AsInt32x8().Xor(allOnes).AsFloat32x8()
}

// Not_AVX2_F64x4 returns bitwise NOT of the vector.
func Not_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	// XOR with all 1s
	allOnes := archsimd.BroadcastInt64x4(-1)
	return v.AsInt64x4().Xor(allOnes).AsFloat64x4()
}

// Xor_AVX2_F32x8 returns bitwise XOR of two vectors.
func Xor_AVX2_F32x8(a, b archsimd.Float32x8) archsimd.Float32x8 {
	return a.AsInt32x8().Xor(b.AsInt32x8()).AsFloat32x8()
}

// Xor_AVX2_F64x4 returns bitwise XOR of two vectors.
func Xor_AVX2_F64x4(a, b archsimd.Float64x4) archsimd.Float64x4 {
	return a.AsInt64x4().Xor(b.AsInt64x4()).AsFloat64x4()
}

// And_AVX2_F32x8 returns bitwise AND of two vectors.
func And_AVX2_F32x8(a, b archsimd.Float32x8) archsimd.Float32x8 {
	return a.AsInt32x8().And(b.AsInt32x8()).AsFloat32x8()
}

// And_AVX2_F64x4 returns bitwise AND of two vectors.
func And_AVX2_F64x4(a, b archsimd.Float64x4) archsimd.Float64x4 {
	return a.AsInt64x4().And(b.AsInt64x4()).AsFloat64x4()
}

// ============================================================================
// Reduction operations
// ============================================================================

// ReduceMin_AVX2_F32x8 returns the minimum element of the vector.
func ReduceMin_AVX2_F32x8(v archsimd.Float32x8) float32 {
	var data [8]float32
	v.StoreSlice(data[:])
	min := data[0]
	for i := 1; i < 8; i++ {
		if data[i] < min {
			min = data[i]
		}
	}
	return min
}

// ReduceMin_AVX2_F64x4 returns the minimum element of the vector.
func ReduceMin_AVX2_F64x4(v archsimd.Float64x4) float64 {
	var data [4]float64
	v.StoreSlice(data[:])
	min := data[0]
	for i := 1; i < 4; i++ {
		if data[i] < min {
			min = data[i]
		}
	}
	return min
}

// ReduceMax_AVX2_F32x8 returns the maximum element of the vector.
func ReduceMax_AVX2_F32x8(v archsimd.Float32x8) float32 {
	var data [8]float32
	v.StoreSlice(data[:])
	max := data[0]
	for i := 1; i < 8; i++ {
		if data[i] > max {
			max = data[i]
		}
	}
	return max
}

// ReduceMax_AVX2_F64x4 returns the maximum element of the vector.
func ReduceMax_AVX2_F64x4(v archsimd.Float64x4) float64 {
	var data [4]float64
	v.StoreSlice(data[:])
	max := data[0]
	for i := 1; i < 4; i++ {
		if data[i] > max {
			max = data[i]
		}
	}
	return max
}

// ReduceMax_AVX2_I32x8 returns the maximum element of the vector.
func ReduceMax_AVX2_I32x8(v archsimd.Int32x8) int32 {
	var data [8]int32
	v.StoreSlice(data[:])
	max := data[0]
	for i := 1; i < 8; i++ {
		if data[i] > max {
			max = data[i]
		}
	}
	return max
}

// ReduceMax_AVX2_I64x4 returns the maximum element of the vector.
func ReduceMax_AVX2_I64x4(v archsimd.Int64x4) int64 {
	var data [4]int64
	v.StoreSlice(data[:])
	max := data[0]
	for i := 1; i < 4; i++ {
		if data[i] > max {
			max = data[i]
		}
	}
	return max
}
