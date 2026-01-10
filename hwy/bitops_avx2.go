//go:build amd64 && goexperiment.simd

package hwy

import (
	"math/bits"
	"simd/archsimd"
)

// This file provides AVX2 SIMD implementations of bit manipulation operations.
// AVX2 doesn't have native SIMD popcount instructions, so we use store/scalar/load pattern.

// PopCount_AVX2_I32x8 counts set bits in each lane.
func PopCount_AVX2_I32x8(v archsimd.Int32x8) archsimd.Int32x8 {
	var data [8]int32
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = int32(bits.OnesCount32(uint32(data[i])))
	}
	return archsimd.LoadInt32x8Slice(data[:])
}

// PopCount_AVX2_I64x4 counts set bits in each lane.
func PopCount_AVX2_I64x4(v archsimd.Int64x4) archsimd.Int64x4 {
	var data [4]int64
	v.StoreSlice(data[:])
	for i := 0; i < 4; i++ {
		data[i] = int64(bits.OnesCount64(uint64(data[i])))
	}
	return archsimd.LoadInt64x4Slice(data[:])
}

// LeadingZeroCount_AVX2_I32x8 counts leading zeros in each lane.
func LeadingZeroCount_AVX2_I32x8(v archsimd.Int32x8) archsimd.Int32x8 {
	var data [8]int32
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = int32(bits.LeadingZeros32(uint32(data[i])))
	}
	return archsimd.LoadInt32x8Slice(data[:])
}

// LeadingZeroCount_AVX2_I64x4 counts leading zeros in each lane.
func LeadingZeroCount_AVX2_I64x4(v archsimd.Int64x4) archsimd.Int64x4 {
	var data [4]int64
	v.StoreSlice(data[:])
	for i := 0; i < 4; i++ {
		data[i] = int64(bits.LeadingZeros64(uint64(data[i])))
	}
	return archsimd.LoadInt64x4Slice(data[:])
}

// TrailingZeroCount_AVX2_I32x8 counts trailing zeros in each lane.
func TrailingZeroCount_AVX2_I32x8(v archsimd.Int32x8) archsimd.Int32x8 {
	var data [8]int32
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		if data[i] == 0 {
			data[i] = 32
		} else {
			data[i] = int32(bits.TrailingZeros32(uint32(data[i])))
		}
	}
	return archsimd.LoadInt32x8Slice(data[:])
}

// TrailingZeroCount_AVX2_I64x4 counts trailing zeros in each lane.
func TrailingZeroCount_AVX2_I64x4(v archsimd.Int64x4) archsimd.Int64x4 {
	var data [4]int64
	v.StoreSlice(data[:])
	for i := 0; i < 4; i++ {
		if data[i] == 0 {
			data[i] = 64
		} else {
			data[i] = int64(bits.TrailingZeros64(uint64(data[i])))
		}
	}
	return archsimd.LoadInt64x4Slice(data[:])
}

// RotateRight_AVX2_I32x8 rotates bits right in each lane.
func RotateRight_AVX2_I32x8(v archsimd.Int32x8, count int) archsimd.Int32x8 {
	var data [8]int32
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = int32(bits.RotateLeft32(uint32(data[i]), -count))
	}
	return archsimd.LoadInt32x8Slice(data[:])
}

// RotateRight_AVX2_I64x4 rotates bits right in each lane.
func RotateRight_AVX2_I64x4(v archsimd.Int64x4, count int) archsimd.Int64x4 {
	var data [4]int64
	v.StoreSlice(data[:])
	for i := 0; i < 4; i++ {
		data[i] = int64(bits.RotateLeft64(uint64(data[i]), -count))
	}
	return archsimd.LoadInt64x4Slice(data[:])
}

// ReverseBits_AVX2_I32x8 reverses bit order in each lane.
func ReverseBits_AVX2_I32x8(v archsimd.Int32x8) archsimd.Int32x8 {
	var data [8]int32
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = int32(bits.Reverse32(uint32(data[i])))
	}
	return archsimd.LoadInt32x8Slice(data[:])
}

// ReverseBits_AVX2_I64x4 reverses bit order in each lane.
func ReverseBits_AVX2_I64x4(v archsimd.Int64x4) archsimd.Int64x4 {
	var data [4]int64
	v.StoreSlice(data[:])
	for i := 0; i < 4; i++ {
		data[i] = int64(bits.Reverse64(uint64(data[i])))
	}
	return archsimd.LoadInt64x4Slice(data[:])
}

// HighestSetBitIndex_AVX2_I32x8 returns index of highest set bit (-1 if zero).
func HighestSetBitIndex_AVX2_I32x8(v archsimd.Int32x8) archsimd.Int32x8 {
	var data [8]int32
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		if data[i] == 0 {
			data[i] = -1
		} else {
			data[i] = int32(bits.Len32(uint32(data[i])) - 1)
		}
	}
	return archsimd.LoadInt32x8Slice(data[:])
}

// HighestSetBitIndex_AVX2_I64x4 returns index of highest set bit (-1 if zero).
func HighestSetBitIndex_AVX2_I64x4(v archsimd.Int64x4) archsimd.Int64x4 {
	var data [4]int64
	v.StoreSlice(data[:])
	for i := 0; i < 4; i++ {
		if data[i] == 0 {
			data[i] = -1
		} else {
			data[i] = int64(bits.Len64(uint64(data[i])) - 1)
		}
	}
	return archsimd.LoadInt64x4Slice(data[:])
}
