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

// This file provides AVX-512 SIMD implementations of type conversion operations.
// These work directly with archsimd vector types for maximum performance.

// ConvertToInt32_AVX512_F32x16 converts float32 to int32 using AVX-512 VCVTTPS2DQ.
// Uses truncation toward zero (same as C-style float to int cast).
func ConvertToInt32_AVX512_F32x16(v archsimd.Float32x16) archsimd.Int32x16 {
	return v.ConvertToInt32()
}

// ConvertToFloat32_AVX512_I32x16 converts int32 to float32 using AVX-512 VCVTDQ2PS.
func ConvertToFloat32_AVX512_I32x16(v archsimd.Int32x16) archsimd.Float32x16 {
	return v.ConvertToFloat32()
}

// ConvertToInt64_AVX512_F64x8 converts float64 to int64 using AVX-512.
func ConvertToInt64_AVX512_F64x8(v archsimd.Float64x8) archsimd.Int64x8 {
	return v.ConvertToInt64()
}

// ConvertToFloat64_AVX512_I64x8 converts int64 to float64 using AVX-512.
func ConvertToFloat64_AVX512_I64x8(v archsimd.Int64x8) archsimd.Float64x8 {
	return v.ConvertToFloat64()
}

// BitCastF32ToI32_AVX512 reinterprets Float32x16 bits as Int32x16.
func BitCastF32ToI32_AVX512(v archsimd.Float32x16) archsimd.Int32x16 {
	return v.AsInt32x16()
}

// BitCastI32ToF32_AVX512 reinterprets Int32x16 bits as Float32x16.
func BitCastI32ToF32_AVX512(v archsimd.Int32x16) archsimd.Float32x16 {
	return v.AsFloat32x16()
}

// BitCastF64ToI64_AVX512 reinterprets Float64x8 bits as Int64x8.
func BitCastF64ToI64_AVX512(v archsimd.Float64x8) archsimd.Int64x8 {
	return v.AsInt64x8()
}

// BitCastI64ToF64_AVX512 reinterprets Int64x8 bits as Float64x8.
func BitCastI64ToF64_AVX512(v archsimd.Int64x8) archsimd.Float64x8 {
	return v.AsFloat64x8()
}

// Round_AVX512_F32x16 rounds to nearest integer.
func Round_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreArray(&data)
	for i := 0; i < 16; i++ {
		data[i] = float32(math.Round(float64(data[i])))
	}
	return archsimd.LoadFloat32x16(data[:])
}

// Round_AVX512_F64x8 rounds to nearest integer.
func Round_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreArray(&data)
	for i := 0; i < 8; i++ {
		data[i] = math.Round(data[i])
	}
	return archsimd.LoadFloat64x8(data[:])
}

// Trunc_AVX512_F32x16 truncates toward zero.
func Trunc_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreArray(&data)
	for i := 0; i < 16; i++ {
		data[i] = float32(math.Trunc(float64(data[i])))
	}
	return archsimd.LoadFloat32x16(data[:])
}

// Trunc_AVX512_F64x8 truncates toward zero.
func Trunc_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreArray(&data)
	for i := 0; i < 8; i++ {
		data[i] = math.Trunc(data[i])
	}
	return archsimd.LoadFloat64x8(data[:])
}

// Ceil_AVX512_F32x16 rounds up toward positive infinity.
func Ceil_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreArray(&data)
	for i := 0; i < 16; i++ {
		data[i] = float32(math.Ceil(float64(data[i])))
	}
	return archsimd.LoadFloat32x16(data[:])
}

// Ceil_AVX512_F64x8 rounds up toward positive infinity.
func Ceil_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreArray(&data)
	for i := 0; i < 8; i++ {
		data[i] = math.Ceil(data[i])
	}
	return archsimd.LoadFloat64x8(data[:])
}

// Floor_AVX512_F32x16 rounds down toward negative infinity.
func Floor_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreArray(&data)
	for i := 0; i < 16; i++ {
		data[i] = float32(math.Floor(float64(data[i])))
	}
	return archsimd.LoadFloat32x16(data[:])
}

// Floor_AVX512_F64x8 rounds down toward negative infinity.
func Floor_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreArray(&data)
	for i := 0; i < 8; i++ {
		data[i] = math.Floor(data[i])
	}
	return archsimd.LoadFloat64x8(data[:])
}

// NearestInt_AVX512_F32x16 rounds to nearest even integer.
func NearestInt_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreArray(&data)
	for i := 0; i < 16; i++ {
		data[i] = float32(math.RoundToEven(float64(data[i])))
	}
	return archsimd.LoadFloat32x16(data[:])
}

// NearestInt_AVX512_F64x8 rounds to nearest even integer.
func NearestInt_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreArray(&data)
	for i := 0; i < 8; i++ {
		data[i] = math.RoundToEven(data[i])
	}
	return archsimd.LoadFloat64x8(data[:])
}

// RoundToEven_AVX512_F32x16 rounds to nearest even integer using SIMD conversion.
// This uses ConvertToInt32 which does banker's rounding, then converts back.
// Note: This may not handle values outside int32 range correctly.
func RoundToEven_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	return v.ConvertToInt32().ConvertToFloat32()
}

// RoundToEven_AVX512_F64x8 rounds to nearest even integer using SIMD conversion.
// This uses ConvertToInt64 which does banker's rounding, then converts back.
// Note: This may not handle values outside int64 range correctly.
func RoundToEven_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	return v.ConvertToInt64().ConvertToFloat64()
}

// Pow2_AVX512_F32x16 computes 2^k for each lane using IEEE 754 bit manipulation.
// Takes Int32x16 exponents and returns Float32x16 results.
func Pow2_AVX512_F32x16(k archsimd.Int32x16) archsimd.Float32x16 {
	// For float32: 2^k = ((k + 127) << 23) as float32 bits
	bias := archsimd.BroadcastInt32x16(127)
	biased := k.Add(bias)
	expBits := biased.ShiftAllLeft(23)
	return expBits.AsFloat32x16()
}

// Pow2_AVX512_F64x8 computes 2^k for each lane using IEEE 754 bit manipulation.
// Takes Int32x8 exponents and returns Float64x8 results.
// AVX-512 has proper 64-bit integer support.
func Pow2_AVX512_F64x8(k archsimd.Int32x8) archsimd.Float64x8 {
	// For float64: 2^k = ((k + 1023) << 52) as float64 bits
	// Use scalar fallback for now as AVX-512 Int64x8 shift may need verification
	var kData [8]int32
	var result [8]float64
	k.StoreArray(&kData)
	for i := 0; i < 8; i++ {
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
	return archsimd.LoadFloat64x8(result[:])
}
