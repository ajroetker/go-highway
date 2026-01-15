//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides AVX-512 SIMD implementations of compress and expand operations.
// These work directly with archsimd vector and mask types.
// Note: archsimd uses proper Mask types (Mask32x16, Mask64x8) for comparison results.
// These Mask types have ToBits() method to convert to a bitmask integer.

// Compress_AVX512_F32x16 compresses elements where mask is true to the front.
// The mask should be the result of a comparison operation (e.g., Less, Equal).
// Returns compressed vector and count of valid elements.
func Compress_AVX512_F32x16(v archsimd.Float32x16, mask archsimd.Mask32x16) (archsimd.Float32x16, int) {
	var data [16]float32
	v.StoreSlice(data[:])

	bits := mask.ToBits()
	var result [16]float32
	count := 0

	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat32x16Slice(result[:]), count
}

// Compress_AVX512_F64x8 compresses elements where mask is true to the front.
// Returns compressed vector and count of valid elements.
func Compress_AVX512_F64x8(v archsimd.Float64x8, mask archsimd.Mask64x8) (archsimd.Float64x8, int) {
	var data [8]float64
	v.StoreSlice(data[:])

	bits := mask.ToBits()
	var result [8]float64
	count := 0

	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat64x8Slice(result[:]), count
}

// Expand_AVX512_F32x16 expands elements into positions where mask is true.
func Expand_AVX512_F32x16(v archsimd.Float32x16, mask archsimd.Mask32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])

	bits := mask.ToBits()
	var result [16]float32
	srcIdx := 0

	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadFloat32x16Slice(result[:])
}

// Expand_AVX512_F64x8 expands elements into positions where mask is true.
func Expand_AVX512_F64x8(v archsimd.Float64x8, mask archsimd.Mask64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])

	bits := mask.ToBits()
	var result [8]float64
	srcIdx := 0

	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadFloat64x8Slice(result[:])
}

// CompressStore_AVX512_F32x16 compresses and stores directly to slice.
// Returns number of elements stored.
func CompressStore_AVX512_F32x16(v archsimd.Float32x16, mask archsimd.Mask32x16, dst []float32) int {
	var data [16]float32
	v.StoreSlice(data[:])

	bits := mask.ToBits()
	count := 0
	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CompressStore_AVX512_F64x8 compresses and stores directly to slice.
// Returns number of elements stored.
func CompressStore_AVX512_F64x8(v archsimd.Float64x8, mask archsimd.Mask64x8, dst []float64) int {
	var data [8]float64
	v.StoreSlice(data[:])

	bits := mask.ToBits()
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

// CountTrue_AVX512_F32x16 counts true lanes in mask.
func CountTrue_AVX512_F32x16(mask archsimd.Mask32x16) int {
	bits := mask.ToBits()
	count := 0
	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			count++
		}
	}
	return count
}

// CountTrue_AVX512_F64x8 counts true lanes in mask.
func CountTrue_AVX512_F64x8(mask archsimd.Mask64x8) int {
	bits := mask.ToBits()
	count := 0
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			count++
		}
	}
	return count
}

// AllTrue_AVX512_F32x16 returns true if all lanes are true.
func AllTrue_AVX512_F32x16(mask archsimd.Mask32x16) bool {
	return mask.ToBits() == 0xFFFF
}

// AllTrue_AVX512_F64x8 returns true if all lanes are true.
func AllTrue_AVX512_F64x8(mask archsimd.Mask64x8) bool {
	return mask.ToBits() == 0xFF
}

// AllFalse_AVX512_F32x16 returns true if all lanes are false.
func AllFalse_AVX512_F32x16(mask archsimd.Mask32x16) bool {
	return mask.ToBits() == 0
}

// AllFalse_AVX512_F64x8 returns true if all lanes are false.
func AllFalse_AVX512_F64x8(mask archsimd.Mask64x8) bool {
	return mask.ToBits() == 0
}

// FindFirstTrue_AVX512_F32x16 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX512_F32x16(mask archsimd.Mask32x16) int {
	bits := mask.ToBits()
	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			return i
		}
	}
	return -1
}

// FindFirstTrue_AVX512_F64x8 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX512_F64x8(mask archsimd.Mask64x8) int {
	bits := mask.ToBits()
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX512_F32x16 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX512_F32x16(mask archsimd.Mask32x16) int {
	bits := mask.ToBits()
	for i := 15; i >= 0; i-- {
		if (bits & (1 << i)) != 0 {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX512_F64x8 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX512_F64x8(mask archsimd.Mask64x8) int {
	bits := mask.ToBits()
	for i := 7; i >= 0; i-- {
		if (bits & (1 << i)) != 0 {
			return i
		}
	}
	return -1
}

// FirstN_AVX512_F32x16 creates a mask with the first n lanes set to true.
func FirstN_AVX512_F32x16(n int) archsimd.Mask32x16 {
	indices := archsimd.LoadInt32x16Slice([]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
	threshold := archsimd.BroadcastInt32x16(int32(n))
	return indices.Less(threshold)
}

// FirstN_AVX512_F64x8 creates a mask with the first n lanes set to true.
func FirstN_AVX512_F64x8(n int) archsimd.Mask64x8 {
	indices := archsimd.LoadInt64x8Slice([]int64{0, 1, 2, 3, 4, 5, 6, 7})
	threshold := archsimd.BroadcastInt64x8(int64(n))
	return indices.Less(threshold)
}

// LastN_AVX512_F32x16 creates a mask with the last n lanes set to true.
// For n lanes set, we compute: bits = ((1 << n) - 1) << (16 - n)
func LastN_AVX512_F32x16(n int) archsimd.Mask32x16 {
	if n <= 0 {
		return archsimd.Mask32x16FromBits(0)
	}
	if n >= 16 {
		return archsimd.Mask32x16FromBits(0xFFFF)
	}
	bits := uint16(((1 << n) - 1) << (16 - n))
	return archsimd.Mask32x16FromBits(bits)
}

// LastN_AVX512_F64x8 creates a mask with the last n lanes set to true.
func LastN_AVX512_F64x8(n int) archsimd.Mask64x8 {
	if n <= 0 {
		return archsimd.Mask64x8FromBits(0)
	}
	if n >= 8 {
		return archsimd.Mask64x8FromBits(0xFF)
	}
	bits := uint8(((1 << n) - 1) << (8 - n))
	return archsimd.Mask64x8FromBits(bits)
}

// MaskFromBits_AVX512_F32x16 creates a mask from a bitmask integer.
func MaskFromBits_AVX512_F32x16(bits uint64) archsimd.Mask32x16 {
	var vals [16]int32
	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = 1
		}
	}
	vec := archsimd.LoadInt32x16Slice(vals[:])
	zero := archsimd.BroadcastInt32x16(0)
	return vec.Greater(zero)
}

// MaskFromBits_AVX512_F64x8 creates a mask from a bitmask integer.
func MaskFromBits_AVX512_F64x8(bits uint64) archsimd.Mask64x8 {
	var vals [8]int64
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = 1
		}
	}
	vec := archsimd.LoadInt64x8Slice(vals[:])
	zero := archsimd.BroadcastInt64x8(0)
	return vec.Greater(zero)
}

// BitsFromMask_AVX512_F32x16 converts mask to bitmask integer.
func BitsFromMask_AVX512_F32x16(mask archsimd.Mask32x16) uint64 {
	return uint64(mask.ToBits())
}

// BitsFromMask_AVX512_F64x8 converts mask to bitmask integer.
func BitsFromMask_AVX512_F64x8(mask archsimd.Mask64x8) uint64 {
	return uint64(mask.ToBits())
}

// ============================================================================
// Integer type variants (I32x16, I64x8)
// ============================================================================

// Compress_AVX512_I32x16 compresses elements where mask is true to the front.
func Compress_AVX512_I32x16(v archsimd.Int32x16, mask archsimd.Mask32x16) (archsimd.Int32x16, int) {
	var data [16]int32
	v.StoreSlice(data[:])

	bits := mask.ToBits()
	var result [16]int32
	count := 0

	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadInt32x16Slice(result[:]), count
}

// Compress_AVX512_I64x8 compresses elements where mask is true to the front.
func Compress_AVX512_I64x8(v archsimd.Int64x8, mask archsimd.Mask64x8) (archsimd.Int64x8, int) {
	var data [8]int64
	v.StoreSlice(data[:])

	bits := mask.ToBits()
	var result [8]int64
	count := 0

	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadInt64x8Slice(result[:]), count
}

// Expand_AVX512_I32x16 expands elements into positions where mask is true.
func Expand_AVX512_I32x16(v archsimd.Int32x16, mask archsimd.Mask32x16) archsimd.Int32x16 {
	var data [16]int32
	v.StoreSlice(data[:])

	bits := mask.ToBits()
	var result [16]int32
	srcIdx := 0

	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadInt32x16Slice(result[:])
}

// Expand_AVX512_I64x8 expands elements into positions where mask is true.
func Expand_AVX512_I64x8(v archsimd.Int64x8, mask archsimd.Mask64x8) archsimd.Int64x8 {
	var data [8]int64
	v.StoreSlice(data[:])

	bits := mask.ToBits()
	var result [8]int64
	srcIdx := 0

	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadInt64x8Slice(result[:])
}

// CompressStore_AVX512_I32x16 compresses and stores directly to slice.
func CompressStore_AVX512_I32x16(v archsimd.Int32x16, mask archsimd.Mask32x16, dst []int32) int {
	var data [16]int32
	v.StoreSlice(data[:])

	bits := mask.ToBits()
	count := 0
	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CompressStore_AVX512_I64x8 compresses and stores directly to slice.
func CompressStore_AVX512_I64x8(v archsimd.Int64x8, mask archsimd.Mask64x8, dst []int64) int {
	var data [8]int64
	v.StoreSlice(data[:])

	bits := mask.ToBits()
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

// CountTrue_AVX512_I32x16 counts true lanes in mask.
func CountTrue_AVX512_I32x16(mask archsimd.Mask32x16) int {
	return CountTrue_AVX512_F32x16(mask)
}

// CountTrue_AVX512_I64x8 counts true lanes in mask.
func CountTrue_AVX512_I64x8(mask archsimd.Mask64x8) int {
	return CountTrue_AVX512_F64x8(mask)
}

// FirstN_AVX512_I32x16 creates a mask with the first n lanes set to true.
func FirstN_AVX512_I32x16(n int) archsimd.Mask32x16 {
	return FirstN_AVX512_F32x16(n)
}

// FirstN_AVX512_I64x8 creates a mask with the first n lanes set to true.
func FirstN_AVX512_I64x8(n int) archsimd.Mask64x8 {
	return FirstN_AVX512_F64x8(n)
}

// LastN_AVX512_I32x16 creates a mask with the last n lanes set to true.
func LastN_AVX512_I32x16(n int) archsimd.Mask32x16 {
	return LastN_AVX512_F32x16(n)
}

// LastN_AVX512_I64x8 creates a mask with the last n lanes set to true.
func LastN_AVX512_I64x8(n int) archsimd.Mask64x8 {
	return LastN_AVX512_F64x8(n)
}

// AllTrue_AVX512_I32x16 returns true if all lanes are true.
func AllTrue_AVX512_I32x16(mask archsimd.Mask32x16) bool {
	return AllTrue_AVX512_F32x16(mask)
}

// AllTrue_AVX512_I64x8 returns true if all lanes are true.
func AllTrue_AVX512_I64x8(mask archsimd.Mask64x8) bool {
	return AllTrue_AVX512_F64x8(mask)
}

// AllFalse_AVX512_I32x16 returns true if all lanes are false.
func AllFalse_AVX512_I32x16(mask archsimd.Mask32x16) bool {
	return AllFalse_AVX512_F32x16(mask)
}

// AllFalse_AVX512_I64x8 returns true if all lanes are false.
func AllFalse_AVX512_I64x8(mask archsimd.Mask64x8) bool {
	return AllFalse_AVX512_F64x8(mask)
}

// FindFirstTrue_AVX512_I32x16 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX512_I32x16(mask archsimd.Mask32x16) int {
	return FindFirstTrue_AVX512_F32x16(mask)
}

// FindFirstTrue_AVX512_I64x8 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX512_I64x8(mask archsimd.Mask64x8) int {
	return FindFirstTrue_AVX512_F64x8(mask)
}

// FindLastTrue_AVX512_I32x16 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX512_I32x16(mask archsimd.Mask32x16) int {
	return FindLastTrue_AVX512_F32x16(mask)
}

// FindLastTrue_AVX512_I64x8 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX512_I64x8(mask archsimd.Mask64x8) int {
	return FindLastTrue_AVX512_F64x8(mask)
}

// ============================================================================
// IfThenElse wrappers
// ============================================================================

// IfThenElse_AVX512_F32x16 selects elements from a where mask is true, else from b.
func IfThenElse_AVX512_F32x16(mask archsimd.Mask32x16, a, b archsimd.Float32x16) archsimd.Float32x16 {
	var aBuf, bBuf [16]float32
	a.StoreSlice(aBuf[:])
	b.StoreSlice(bBuf[:])

	bits := mask.ToBits()
	var result [16]float32
	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = aBuf[i]
		} else {
			result[i] = bBuf[i]
		}
	}
	return archsimd.LoadFloat32x16Slice(result[:])
}

// IfThenElse_AVX512_F64x8 selects elements from a where mask is true, else from b.
func IfThenElse_AVX512_F64x8(mask archsimd.Mask64x8, a, b archsimd.Float64x8) archsimd.Float64x8 {
	var aBuf, bBuf [8]float64
	a.StoreSlice(aBuf[:])
	b.StoreSlice(bBuf[:])

	bits := mask.ToBits()
	var result [8]float64
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = aBuf[i]
		} else {
			result[i] = bBuf[i]
		}
	}
	return archsimd.LoadFloat64x8Slice(result[:])
}

// IfThenElse_AVX512_I32x16 selects elements from a where mask is true, else from b.
func IfThenElse_AVX512_I32x16(mask archsimd.Mask32x16, a, b archsimd.Int32x16) archsimd.Int32x16 {
	var aBuf, bBuf [16]int32
	a.StoreSlice(aBuf[:])
	b.StoreSlice(bBuf[:])

	bits := mask.ToBits()
	var result [16]int32
	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = aBuf[i]
		} else {
			result[i] = bBuf[i]
		}
	}
	return archsimd.LoadInt32x16Slice(result[:])
}

// IfThenElse_AVX512_I64x8 selects elements from a where mask is true, else from b.
func IfThenElse_AVX512_I64x8(mask archsimd.Mask64x8, a, b archsimd.Int64x8) archsimd.Int64x8 {
	var aBuf, bBuf [8]int64
	a.StoreSlice(aBuf[:])
	b.StoreSlice(bBuf[:])

	bits := mask.ToBits()
	var result [8]int64
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			result[i] = aBuf[i]
		} else {
			result[i] = bBuf[i]
		}
	}
	return archsimd.LoadInt64x8Slice(result[:])
}

// MaskAnd_AVX512_F32x16 combines two masks with AND operation.
func MaskAnd_AVX512_F32x16(a, b archsimd.Mask32x16) archsimd.Mask32x16 {
	return a.And(b)
}

// MaskAnd_AVX512_F64x8 combines two masks with AND operation.
func MaskAnd_AVX512_F64x8(a, b archsimd.Mask64x8) archsimd.Mask64x8 {
	return a.And(b)
}

// MaskOr_AVX512_F32x16 combines two masks with OR operation.
func MaskOr_AVX512_F32x16(a, b archsimd.Mask32x16) archsimd.Mask32x16 {
	return a.Or(b)
}

// MaskOr_AVX512_F64x8 combines two masks with OR operation.
func MaskOr_AVX512_F64x8(a, b archsimd.Mask64x8) archsimd.Mask64x8 {
	return a.Or(b)
}

// ============================================================================
// Bitwise operations for float types
// archsimd doesn't have Xor/Not on float types, so we cast to int and back
// ============================================================================

// SignBit_AVX512_F32x16 returns a vector with the sign bit set in all lanes.
func SignBit_AVX512_F32x16() archsimd.Float32x16 {
	// 0x80000000 is the sign bit for float32
	signBit := archsimd.BroadcastInt32x16(int32(-0x80000000))
	return signBit.AsFloat32x16()
}

// SignBit_AVX512_F64x8 returns a vector with the sign bit set in all lanes.
func SignBit_AVX512_F64x8() archsimd.Float64x8 {
	// 0x8000000000000000 is the sign bit for float64
	signBit := archsimd.BroadcastInt64x8(int64(-0x8000000000000000))
	return signBit.AsFloat64x8()
}

// Not_AVX512_F32x16 returns bitwise NOT of the vector.
func Not_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	// XOR with all 1s
	allOnes := archsimd.BroadcastInt32x16(-1)
	return v.AsInt32x16().Xor(allOnes).AsFloat32x16()
}

// Not_AVX512_F64x8 returns bitwise NOT of the vector.
func Not_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	// XOR with all 1s
	allOnes := archsimd.BroadcastInt64x8(-1)
	return v.AsInt64x8().Xor(allOnes).AsFloat64x8()
}

// Xor_AVX512_F32x16 returns bitwise XOR of two vectors.
func Xor_AVX512_F32x16(a, b archsimd.Float32x16) archsimd.Float32x16 {
	return a.AsInt32x16().Xor(b.AsInt32x16()).AsFloat32x16()
}

// Xor_AVX512_F64x8 returns bitwise XOR of two vectors.
func Xor_AVX512_F64x8(a, b archsimd.Float64x8) archsimd.Float64x8 {
	return a.AsInt64x8().Xor(b.AsInt64x8()).AsFloat64x8()
}

// And_AVX512_F32x16 returns bitwise AND of two vectors.
func And_AVX512_F32x16(a, b archsimd.Float32x16) archsimd.Float32x16 {
	return a.AsInt32x16().And(b.AsInt32x16()).AsFloat32x16()
}

// And_AVX512_F64x8 returns bitwise AND of two vectors.
func And_AVX512_F64x8(a, b archsimd.Float64x8) archsimd.Float64x8 {
	return a.AsInt64x8().And(b.AsInt64x8()).AsFloat64x8()
}
