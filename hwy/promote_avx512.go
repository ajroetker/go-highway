//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides AVX-512 SIMD implementations of type promotion and demotion operations.
// These work directly with archsimd vector types.
//
// AVX-512 has hardware support for various conversions:
// - VCVTPS2PD: float32 -> float64 (8 lanes)
// - VCVTPD2PS: float64 -> float32 (8 lanes)
// - VPMOVSXDQ/VPMOVZXDQ: int32 -> int64
// - etc.
// Since archsimd may not expose all of these, we use store/scalar/load pattern.

// PromoteF32ToF64_AVX512_Lower promotes lower 8 float32 lanes to 8 float64 lanes.
func PromoteF32ToF64_AVX512_Lower(v archsimd.Float32x16) archsimd.Float64x8 {
	var data [16]float32
	v.StoreSlice(data[:])

	var result [8]float64
	for i := 0; i < 8; i++ {
		result[i] = float64(data[i])
	}
	return archsimd.LoadFloat64x8Slice(result[:])
}

// PromoteF32ToF64_AVX512_Upper promotes upper 8 float32 lanes to 8 float64 lanes.
func PromoteF32ToF64_AVX512_Upper(v archsimd.Float32x16) archsimd.Float64x8 {
	var data [16]float32
	v.StoreSlice(data[:])

	var result [8]float64
	for i := 0; i < 8; i++ {
		result[i] = float64(data[8+i])
	}
	return archsimd.LoadFloat64x8Slice(result[:])
}

// DemoteF64ToF32_AVX512 demotes 8 float64 lanes to 8 float32 lanes.
// The result occupies the lower 8 lanes of a Float32x16.
func DemoteF64ToF32_AVX512(v archsimd.Float64x8) archsimd.Float32x16 {
	var data [8]float64
	v.StoreSlice(data[:])

	var result [16]float32
	for i := 0; i < 8; i++ {
		result[i] = float32(data[i])
	}
	// Upper 8 lanes are zero
	return archsimd.LoadFloat32x16Slice(result[:])
}

// DemoteTwoF64ToF32_AVX512 demotes two Float64x8 vectors to one Float32x16.
func DemoteTwoF64ToF32_AVX512(lo, hi archsimd.Float64x8) archsimd.Float32x16 {
	var loData [8]float64
	lo.StoreSlice(loData[:])

	var hiData [8]float64
	hi.StoreSlice(hiData[:])

	var result [16]float32
	for i := 0; i < 8; i++ {
		result[i] = float32(loData[i])
	}
	for i := 0; i < 8; i++ {
		result[8+i] = float32(hiData[i])
	}
	return archsimd.LoadFloat32x16Slice(result[:])
}

// PromoteI32ToI64_AVX512_Lower promotes lower 8 int32 lanes to 8 int64 lanes.
func PromoteI32ToI64_AVX512_Lower(v archsimd.Int32x16) archsimd.Int64x8 {
	var data [16]int32
	v.StoreSlice(data[:])

	var result [8]int64
	for i := 0; i < 8; i++ {
		result[i] = int64(data[i])
	}
	return archsimd.LoadInt64x8Slice(result[:])
}

// PromoteI32ToI64_AVX512_Upper promotes upper 8 int32 lanes to 8 int64 lanes.
func PromoteI32ToI64_AVX512_Upper(v archsimd.Int32x16) archsimd.Int64x8 {
	var data [16]int32
	v.StoreSlice(data[:])

	var result [8]int64
	for i := 0; i < 8; i++ {
		result[i] = int64(data[8+i])
	}
	return archsimd.LoadInt64x8Slice(result[:])
}

// DemoteI64ToI32_AVX512 demotes 8 int64 lanes to 8 int32 lanes (saturating).
func DemoteI64ToI32_AVX512(v archsimd.Int64x8) archsimd.Int32x16 {
	var data [8]int64
	v.StoreSlice(data[:])

	var result [16]int32
	for i := 0; i < 8; i++ {
		val := data[i]
		if val > 2147483647 {
			result[i] = 2147483647
		} else if val < -2147483648 {
			result[i] = -2147483648
		} else {
			result[i] = int32(val)
		}
	}
	// Upper 8 lanes are zero
	return archsimd.LoadInt32x16Slice(result[:])
}

// DemoteTwoI64ToI32_AVX512 demotes two Int64x8 vectors to one Int32x16 (saturating).
func DemoteTwoI64ToI32_AVX512(lo, hi archsimd.Int64x8) archsimd.Int32x16 {
	var loData [8]int64
	lo.StoreSlice(loData[:])

	var hiData [8]int64
	hi.StoreSlice(hiData[:])

	var result [16]int32
	for i := 0; i < 8; i++ {
		val := loData[i]
		if val > 2147483647 {
			result[i] = 2147483647
		} else if val < -2147483648 {
			result[i] = -2147483648
		} else {
			result[i] = int32(val)
		}
	}
	for i := 0; i < 8; i++ {
		val := hiData[i]
		if val > 2147483647 {
			result[8+i] = 2147483647
		} else if val < -2147483648 {
			result[8+i] = -2147483648
		} else {
			result[8+i] = int32(val)
		}
	}
	return archsimd.LoadInt32x16Slice(result[:])
}

// TruncateI64ToI32_AVX512 demotes 8 int64 lanes to 8 int32 lanes (truncating).
func TruncateI64ToI32_AVX512(v archsimd.Int64x8) archsimd.Int32x16 {
	var data [8]int64
	v.StoreSlice(data[:])

	var result [16]int32
	for i := 0; i < 8; i++ {
		result[i] = int32(data[i])
	}
	return archsimd.LoadInt32x16Slice(result[:])
}

// PromoteU32ToU64_AVX512_Lower promotes lower 8 uint32 lanes to 8 uint64 lanes.
func PromoteU32ToU64_AVX512_Lower(v archsimd.Int32x16) archsimd.Int64x8 {
	var data [16]int32
	v.StoreSlice(data[:])

	var result [8]int64
	for i := 0; i < 8; i++ {
		// Zero-extend by treating int32 bits as uint32
		result[i] = int64(uint32(data[i]))
	}
	return archsimd.LoadInt64x8Slice(result[:])
}

// PromoteU32ToU64_AVX512_Upper promotes upper 8 uint32 lanes to 8 uint64 lanes.
func PromoteU32ToU64_AVX512_Upper(v archsimd.Int32x16) archsimd.Int64x8 {
	var data [16]int32
	v.StoreSlice(data[:])

	var result [8]int64
	for i := 0; i < 8; i++ {
		result[i] = int64(uint32(data[8+i]))
	}
	return archsimd.LoadInt64x8Slice(result[:])
}

// DemoteU64ToU32_AVX512 demotes 8 uint64 lanes to 8 uint32 lanes (saturating).
func DemoteU64ToU32_AVX512(v archsimd.Int64x8) archsimd.Int32x16 {
	var data [8]int64
	v.StoreSlice(data[:])

	var result [16]int32
	for i := 0; i < 8; i++ {
		val := uint64(data[i])
		if val > 0xFFFFFFFF {
			result[i] = -1 // 0xFFFFFFFF as int32
		} else {
			result[i] = int32(uint32(val))
		}
	}
	return archsimd.LoadInt32x16Slice(result[:])
}

// DemoteTwoU64ToU32_AVX512 demotes two vectors to one (saturating).
func DemoteTwoU64ToU32_AVX512(lo, hi archsimd.Int64x8) archsimd.Int32x16 {
	var loData [8]int64
	lo.StoreSlice(loData[:])

	var hiData [8]int64
	hi.StoreSlice(hiData[:])

	var result [16]int32
	for i := 0; i < 8; i++ {
		val := uint64(loData[i])
		if val > 0xFFFFFFFF {
			result[i] = -1
		} else {
			result[i] = int32(uint32(val))
		}
	}
	for i := 0; i < 8; i++ {
		val := uint64(hiData[i])
		if val > 0xFFFFFFFF {
			result[8+i] = -1
		} else {
			result[8+i] = int32(uint32(val))
		}
	}
	return archsimd.LoadInt32x16Slice(result[:])
}

// TruncateU64ToU32_AVX512 demotes 8 uint64 lanes to 8 uint32 lanes (truncating).
func TruncateU64ToU32_AVX512(v archsimd.Int64x8) archsimd.Int32x16 {
	var data [8]int64
	v.StoreSlice(data[:])

	var result [16]int32
	for i := 0; i < 8; i++ {
		result[i] = int32(uint32(data[i]))
	}
	return archsimd.LoadInt32x16Slice(result[:])
}
