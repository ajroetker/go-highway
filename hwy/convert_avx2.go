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
	"math"
	"simd/archsimd"
)

// This file provides AVX2 SIMD implementations of type conversion operations.
// These work directly with archsimd vector types for maximum performance.

// ConvertToInt32_AVX2_F32x8 converts float32 to int32 using AVX2 VCVTTPS2DQ.
// Uses truncation toward zero (same as C-style float to int cast).
func ConvertToInt32_AVX2_F32x8(v archsimd.Float32x8) archsimd.Int32x8 {
	return v.ConvertToInt32()
}

// ConvertToFloat32_AVX2_I32x8 converts int32 to float32 using AVX2 VCVTDQ2PS.
func ConvertToFloat32_AVX2_I32x8(v archsimd.Int32x8) archsimd.Float32x8 {
	return v.ConvertToFloat32()
}

// BitCastF32ToI32_AVX2 reinterprets Float32x8 bits as Int32x8.
func BitCastF32ToI32_AVX2(v archsimd.Float32x8) archsimd.Int32x8 {
	return v.AsInt32x8()
}

// BitCastI32ToF32_AVX2 reinterprets Int32x8 bits as Float32x8.
func BitCastI32ToF32_AVX2(v archsimd.Int32x8) archsimd.Float32x8 {
	return v.AsFloat32x8()
}

// BitCastF64ToI64_AVX2 reinterprets Float64x4 bits as Int64x4.
func BitCastF64ToI64_AVX2(v archsimd.Float64x4) archsimd.Int64x4 {
	return v.AsInt64x4()
}

// BitCastI64ToF64_AVX2 reinterprets Int64x4 bits as Float64x4.
func BitCastI64ToF64_AVX2(v archsimd.Int64x4) archsimd.Float64x4 {
	return v.AsFloat64x4()
}

// Round_AVX2_F32x8 rounds to nearest integer using store/scalar/load pattern.
// AVX2 has VROUNDPS but archsimd may not expose it directly.
func Round_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreArray(&data)
	for i := 0; i < 8; i++ {
		data[i] = float32(math.Round(float64(data[i])))
	}
	return archsimd.LoadFloat32x8(data[:])
}

// Round_AVX2_F64x4 rounds to nearest integer using store/scalar/load pattern.
func Round_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreArray(&data)
	for i := 0; i < 4; i++ {
		data[i] = math.Round(data[i])
	}
	return archsimd.LoadFloat64x4(data[:])
}

// Trunc_AVX2_F32x8 truncates toward zero using store/scalar/load pattern.
func Trunc_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreArray(&data)
	for i := 0; i < 8; i++ {
		data[i] = float32(math.Trunc(float64(data[i])))
	}
	return archsimd.LoadFloat32x8(data[:])
}

// Trunc_AVX2_F64x4 truncates toward zero using store/scalar/load pattern.
func Trunc_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreArray(&data)
	for i := 0; i < 4; i++ {
		data[i] = math.Trunc(data[i])
	}
	return archsimd.LoadFloat64x4(data[:])
}

// Ceil_AVX2_F32x8 rounds up toward positive infinity.
func Ceil_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreArray(&data)
	for i := 0; i < 8; i++ {
		data[i] = float32(math.Ceil(float64(data[i])))
	}
	return archsimd.LoadFloat32x8(data[:])
}

// Ceil_AVX2_F64x4 rounds up toward positive infinity.
func Ceil_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreArray(&data)
	for i := 0; i < 4; i++ {
		data[i] = math.Ceil(data[i])
	}
	return archsimd.LoadFloat64x4(data[:])
}

// Floor_AVX2_F32x8 rounds down toward negative infinity.
func Floor_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreArray(&data)
	for i := 0; i < 8; i++ {
		data[i] = float32(math.Floor(float64(data[i])))
	}
	return archsimd.LoadFloat32x8(data[:])
}

// Floor_AVX2_F64x4 rounds down toward negative infinity.
func Floor_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreArray(&data)
	for i := 0; i < 4; i++ {
		data[i] = math.Floor(data[i])
	}
	return archsimd.LoadFloat64x4(data[:])
}

// NearestInt_AVX2_F32x8 rounds to nearest even integer.
func NearestInt_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreArray(&data)
	for i := 0; i < 8; i++ {
		data[i] = float32(math.RoundToEven(float64(data[i])))
	}
	return archsimd.LoadFloat32x8(data[:])
}

// NearestInt_AVX2_F64x4 rounds to nearest even integer.
func NearestInt_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreArray(&data)
	for i := 0; i < 4; i++ {
		data[i] = math.RoundToEven(data[i])
	}
	return archsimd.LoadFloat64x4(data[:])
}

// Pow2_AVX2_F32x8 computes 2^k for each lane using IEEE 754 bit manipulation.
// Takes Int32x8 exponents and returns Float32x8 results.
func Pow2_AVX2_F32x8(k archsimd.Int32x8) archsimd.Float32x8 {
	// For float32: 2^k = ((k + 127) << 23) as float32 bits
	bias := archsimd.BroadcastInt32x8(127)
	biased := k.Add(bias)
	expBits := biased.ShiftAllLeft(23)
	return expBits.AsFloat32x8()
}

// Pow2_AVX2_F64x4 computes 2^k for each lane using IEEE 754 bit manipulation.
// Takes Int32x4 exponents (widened from int32) and returns Float64x4 results.
// Note: For float64, we need 64-bit operations which AVX2 doesn't handle well,
// so we use scalar fallback.
func Pow2_AVX2_F64x4(k archsimd.Int32x4) archsimd.Float64x4 {
	// For float64: 2^k = ((k + 1023) << 52) as float64 bits
	// AVX2 lacks proper 64-bit integer shift, so use scalar
	var kData [4]int32
	var result [4]float64
	k.StoreArray(&kData)
	for i := 0; i < 4; i++ {
		ki := kData[i]
		if ki < -1022 {
			result[i] = 0
		} else if ki > 1023 {
			result[i] = math.Inf(1)
		} else {
			bits := uint64(int64(ki)+1023) << 52
			result[i] = math.Float64frombits(bits)
		}
	}
	return archsimd.LoadFloat64x4(result[:])
}
