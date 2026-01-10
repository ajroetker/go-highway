//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
	"unsafe"
)

// This file provides AVX2 SIMD implementations of type promotion and demotion operations.
// These work directly with archsimd vector types.
//
// AVX2 has hardware support for some conversions:
// - VCVTPS2PD: float32 -> float64 (4 lanes)
// - VCVTPD2PS: float64 -> float32 (4 lanes)
// - VPMOVSXWD/VPMOVZXWD: int16 -> int32
// - etc.
// Since archsimd may not expose all of these, we use store/scalar/load pattern.

// PromoteF32ToF64_AVX2 promotes 4 float32 lanes to 4 float64 lanes.
// Note: AVX2 Float32x8 has 8 lanes, but promotion doubles the width,
// so we take lower 4 float32 and produce 4 float64.
func PromoteF32ToF64_AVX2_Lower(v archsimd.Float32x8) archsimd.Float64x4 {
	var data [8]float32
	v.StoreSlice(data[:])

	var result [4]float64
	for i := 0; i < 4; i++ {
		result[i] = float64(data[i])
	}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// PromoteF32ToF64_AVX2_Upper promotes upper 4 float32 lanes to 4 float64 lanes.
func PromoteF32ToF64_AVX2_Upper(v archsimd.Float32x8) archsimd.Float64x4 {
	var data [8]float32
	v.StoreSlice(data[:])

	var result [4]float64
	for i := 0; i < 4; i++ {
		result[i] = float64(data[4+i])
	}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// DemoteF64ToF32_AVX2 demotes 4 float64 lanes to 4 float32 lanes.
// The result occupies the lower 4 lanes of a Float32x8.
func DemoteF64ToF32_AVX2(v archsimd.Float64x4) archsimd.Float32x8 {
	var data [4]float64
	v.StoreSlice(data[:])

	var result [8]float32
	for i := 0; i < 4; i++ {
		result[i] = float32(data[i])
	}
	// Upper 4 lanes are zero
	return archsimd.LoadFloat32x8Slice(result[:])
}

// DemoteTwoF64ToF32_AVX2 demotes two Float64x4 vectors to one Float32x8.
func DemoteTwoF64ToF32_AVX2(lo, hi archsimd.Float64x4) archsimd.Float32x8 {
	var loData [4]float64
	lo.StoreSlice(loData[:])

	var hiData [4]float64
	hi.StoreSlice(hiData[:])

	var result [8]float32
	for i := 0; i < 4; i++ {
		result[i] = float32(loData[i])
	}
	for i := 0; i < 4; i++ {
		result[4+i] = float32(hiData[i])
	}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// PromoteI16ToI32_AVX2_Lower promotes lower 4 int16 lanes to 4 int32 lanes.
// Note: Input would be Int16x16, but we work with Int32x8 sized operations.
func PromoteI16ToI32_AVX2_Lower(v archsimd.Int32x8) archsimd.Int32x8 {
	// Interpret as 16 int16 values, promote lower 8 to int32
	var data [8]int32
	v.StoreSlice(data[:])

	// Reinterpret the int32 array as int16 values
	int16Data := (*[16]int16)(unsafe.Pointer(&data))

	var result [8]int32
	for i := 0; i < 8; i++ {
		result[i] = int32(int16Data[i])
	}
	return archsimd.LoadInt32x8Slice(result[:])
}

// PromoteI32ToI64_AVX2_Lower promotes lower 4 int32 lanes to 4 int64 lanes.
func PromoteI32ToI64_AVX2_Lower(v archsimd.Int32x8) archsimd.Int64x4 {
	var data [8]int32
	v.StoreSlice(data[:])

	var result [4]int64
	for i := 0; i < 4; i++ {
		result[i] = int64(data[i])
	}
	return archsimd.LoadInt64x4Slice(result[:])
}

// PromoteI32ToI64_AVX2_Upper promotes upper 4 int32 lanes to 4 int64 lanes.
func PromoteI32ToI64_AVX2_Upper(v archsimd.Int32x8) archsimd.Int64x4 {
	var data [8]int32
	v.StoreSlice(data[:])

	var result [4]int64
	for i := 0; i < 4; i++ {
		result[i] = int64(data[4+i])
	}
	return archsimd.LoadInt64x4Slice(result[:])
}

// DemoteI64ToI32_AVX2 demotes 4 int64 lanes to 4 int32 lanes (saturating).
func DemoteI64ToI32_AVX2(v archsimd.Int64x4) archsimd.Int32x8 {
	var data [4]int64
	v.StoreSlice(data[:])

	var result [8]int32
	for i := 0; i < 4; i++ {
		val := data[i]
		if val > 2147483647 {
			result[i] = 2147483647
		} else if val < -2147483648 {
			result[i] = -2147483648
		} else {
			result[i] = int32(val)
		}
	}
	// Upper 4 lanes are zero
	return archsimd.LoadInt32x8Slice(result[:])
}

// DemoteTwoI64ToI32_AVX2 demotes two Int64x4 vectors to one Int32x8 (saturating).
func DemoteTwoI64ToI32_AVX2(lo, hi archsimd.Int64x4) archsimd.Int32x8 {
	var loData [4]int64
	lo.StoreSlice(loData[:])

	var hiData [4]int64
	hi.StoreSlice(hiData[:])

	var result [8]int32
	for i := 0; i < 4; i++ {
		val := loData[i]
		if val > 2147483647 {
			result[i] = 2147483647
		} else if val < -2147483648 {
			result[i] = -2147483648
		} else {
			result[i] = int32(val)
		}
	}
	for i := 0; i < 4; i++ {
		val := hiData[i]
		if val > 2147483647 {
			result[4+i] = 2147483647
		} else if val < -2147483648 {
			result[4+i] = -2147483648
		} else {
			result[4+i] = int32(val)
		}
	}
	return archsimd.LoadInt32x8Slice(result[:])
}

// TruncateI64ToI32_AVX2 demotes 4 int64 lanes to 4 int32 lanes (truncating).
func TruncateI64ToI32_AVX2(v archsimd.Int64x4) archsimd.Int32x8 {
	var data [4]int64
	v.StoreSlice(data[:])

	var result [8]int32
	for i := 0; i < 4; i++ {
		result[i] = int32(data[i])
	}
	return archsimd.LoadInt32x8Slice(result[:])
}

// PromoteU32ToU64_AVX2_Lower promotes lower 4 uint32 lanes to 4 uint64 lanes.
func PromoteU32ToU64_AVX2_Lower(v archsimd.Int32x8) archsimd.Int64x4 {
	var data [8]int32
	v.StoreSlice(data[:])

	var result [4]int64
	for i := 0; i < 4; i++ {
		// Zero-extend by treating int32 bits as uint32
		result[i] = int64(uint32(data[i]))
	}
	return archsimd.LoadInt64x4Slice(result[:])
}

// PromoteU32ToU64_AVX2_Upper promotes upper 4 uint32 lanes to 4 uint64 lanes.
func PromoteU32ToU64_AVX2_Upper(v archsimd.Int32x8) archsimd.Int64x4 {
	var data [8]int32
	v.StoreSlice(data[:])

	var result [4]int64
	for i := 0; i < 4; i++ {
		result[i] = int64(uint32(data[4+i]))
	}
	return archsimd.LoadInt64x4Slice(result[:])
}

// DemoteU64ToU32_AVX2 demotes 4 uint64 lanes to 4 uint32 lanes (saturating).
func DemoteU64ToU32_AVX2(v archsimd.Int64x4) archsimd.Int32x8 {
	var data [4]int64
	v.StoreSlice(data[:])

	var result [8]int32
	for i := 0; i < 4; i++ {
		val := uint64(data[i])
		if val > 0xFFFFFFFF {
			result[i] = -1 // 0xFFFFFFFF as int32
		} else {
			result[i] = int32(uint32(val))
		}
	}
	return archsimd.LoadInt32x8Slice(result[:])
}

// DemoteTwoU64ToU32_AVX2 demotes two vectors to one (saturating).
func DemoteTwoU64ToU32_AVX2(lo, hi archsimd.Int64x4) archsimd.Int32x8 {
	var loData [4]int64
	lo.StoreSlice(loData[:])

	var hiData [4]int64
	hi.StoreSlice(hiData[:])

	var result [8]int32
	for i := 0; i < 4; i++ {
		val := uint64(loData[i])
		if val > 0xFFFFFFFF {
			result[i] = -1
		} else {
			result[i] = int32(uint32(val))
		}
	}
	for i := 0; i < 4; i++ {
		val := uint64(hiData[i])
		if val > 0xFFFFFFFF {
			result[4+i] = -1
		} else {
			result[4+i] = int32(uint32(val))
		}
	}
	return archsimd.LoadInt32x8Slice(result[:])
}

// TruncateU64ToU32_AVX2 demotes 4 uint64 lanes to 4 uint32 lanes (truncating).
func TruncateU64ToU32_AVX2(v archsimd.Int64x4) archsimd.Int32x8 {
	var data [4]int64
	v.StoreSlice(data[:])

	var result [8]int32
	for i := 0; i < 4; i++ {
		result[i] = int32(uint32(data[i]))
	}
	return archsimd.LoadInt32x8Slice(result[:])
}
