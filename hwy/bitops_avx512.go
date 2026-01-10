//go:build amd64 && goexperiment.simd

package hwy

import (
	"math/bits"
	"simd/archsimd"
)

// This file provides AVX-512 SIMD implementations of bit manipulation operations.
// AVX-512 VPOPCNTDQ provides native popcount for 32/64-bit elements, but we use
// store/scalar/load pattern for consistency and portability across all AVX-512 variants.

// PopCount_AVX512_I32x16 counts set bits in each lane.
func PopCount_AVX512_I32x16(v archsimd.Int32x16) archsimd.Int32x16 {
	var data [16]int32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i++ {
		data[i] = int32(bits.OnesCount32(uint32(data[i])))
	}
	return archsimd.LoadInt32x16Slice(data[:])
}

// PopCount_AVX512_I64x8 counts set bits in each lane.
func PopCount_AVX512_I64x8(v archsimd.Int64x8) archsimd.Int64x8 {
	var data [8]int64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = int64(bits.OnesCount64(uint64(data[i])))
	}
	return archsimd.LoadInt64x8Slice(data[:])
}

// LeadingZeroCount_AVX512_I32x16 counts leading zeros in each lane.
func LeadingZeroCount_AVX512_I32x16(v archsimd.Int32x16) archsimd.Int32x16 {
	var data [16]int32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i++ {
		data[i] = int32(bits.LeadingZeros32(uint32(data[i])))
	}
	return archsimd.LoadInt32x16Slice(data[:])
}

// LeadingZeroCount_AVX512_I64x8 counts leading zeros in each lane.
func LeadingZeroCount_AVX512_I64x8(v archsimd.Int64x8) archsimd.Int64x8 {
	var data [8]int64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = int64(bits.LeadingZeros64(uint64(data[i])))
	}
	return archsimd.LoadInt64x8Slice(data[:])
}

// TrailingZeroCount_AVX512_I32x16 counts trailing zeros in each lane.
func TrailingZeroCount_AVX512_I32x16(v archsimd.Int32x16) archsimd.Int32x16 {
	var data [16]int32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i++ {
		if data[i] == 0 {
			data[i] = 32
		} else {
			data[i] = int32(bits.TrailingZeros32(uint32(data[i])))
		}
	}
	return archsimd.LoadInt32x16Slice(data[:])
}

// TrailingZeroCount_AVX512_I64x8 counts trailing zeros in each lane.
func TrailingZeroCount_AVX512_I64x8(v archsimd.Int64x8) archsimd.Int64x8 {
	var data [8]int64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		if data[i] == 0 {
			data[i] = 64
		} else {
			data[i] = int64(bits.TrailingZeros64(uint64(data[i])))
		}
	}
	return archsimd.LoadInt64x8Slice(data[:])
}

// RotateRight_AVX512_I32x16 rotates bits right in each lane.
func RotateRight_AVX512_I32x16(v archsimd.Int32x16, count int) archsimd.Int32x16 {
	var data [16]int32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i++ {
		data[i] = int32(bits.RotateLeft32(uint32(data[i]), -count))
	}
	return archsimd.LoadInt32x16Slice(data[:])
}

// RotateRight_AVX512_I64x8 rotates bits right in each lane.
func RotateRight_AVX512_I64x8(v archsimd.Int64x8, count int) archsimd.Int64x8 {
	var data [8]int64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = int64(bits.RotateLeft64(uint64(data[i]), -count))
	}
	return archsimd.LoadInt64x8Slice(data[:])
}

// ReverseBits_AVX512_I32x16 reverses bit order in each lane.
func ReverseBits_AVX512_I32x16(v archsimd.Int32x16) archsimd.Int32x16 {
	var data [16]int32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i++ {
		data[i] = int32(bits.Reverse32(uint32(data[i])))
	}
	return archsimd.LoadInt32x16Slice(data[:])
}

// ReverseBits_AVX512_I64x8 reverses bit order in each lane.
func ReverseBits_AVX512_I64x8(v archsimd.Int64x8) archsimd.Int64x8 {
	var data [8]int64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = int64(bits.Reverse64(uint64(data[i])))
	}
	return archsimd.LoadInt64x8Slice(data[:])
}

// HighestSetBitIndex_AVX512_I32x16 returns index of highest set bit (-1 if zero).
func HighestSetBitIndex_AVX512_I32x16(v archsimd.Int32x16) archsimd.Int32x16 {
	var data [16]int32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i++ {
		if data[i] == 0 {
			data[i] = -1
		} else {
			data[i] = int32(bits.Len32(uint32(data[i])) - 1)
		}
	}
	return archsimd.LoadInt32x16Slice(data[:])
}

// HighestSetBitIndex_AVX512_I64x8 returns index of highest set bit (-1 if zero).
func HighestSetBitIndex_AVX512_I64x8(v archsimd.Int64x8) archsimd.Int64x8 {
	var data [8]int64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		if data[i] == 0 {
			data[i] = -1
		} else {
			data[i] = int64(bits.Len64(uint64(data[i])) - 1)
		}
	}
	return archsimd.LoadInt64x8Slice(data[:])
}
