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

// This file provides low-level AVX-512 SIMD operations that work directly with
// archsimd vector types. These are core ops (direct hardware instructions),
// not transcendental functions which belong in contrib/math.

// ReduceSum_AVX512_F32x16 returns the sum of all 16 float32 elements.
func ReduceSum_AVX512_F32x16(v archsimd.Float32x16) float32 {
	// Reduce 16 -> 8 using AVX2 function
	lo := v.GetLo() // Float32x8
	hi := v.GetHi() // Float32x8
	sum8 := lo.Add(hi)
	return ReduceSum_AVX2_F32x8(sum8)
}

// ReduceSum_AVX512_F64x8 returns the sum of all 8 float64 elements.
func ReduceSum_AVX512_F64x8(v archsimd.Float64x8) float64 {
	lo := v.GetLo() // Float64x4
	hi := v.GetHi() // Float64x4
	sum4 := lo.Add(hi)
	return ReduceSum_AVX2_F64x4(sum4)
}

// Sqrt_AVX512_F32x16 computes sqrt(x) for a single Float32x16 vector.
// Uses the hardware VSQRTPS instruction which provides correctly rounded results.
func Sqrt_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	return x.Sqrt()
}

// Sqrt_AVX512_F64x8 computes sqrt(x) for a single Float64x8 vector.
// Uses the hardware VSQRTPD instruction which provides correctly rounded results.
func Sqrt_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	return x.Sqrt()
}

// RSqrt_AVX512_F32x16 computes approximate 1/sqrt(x) for 16 float32 values.
// Uses the hardware VRSQRT14PS instruction which provides ~14-bit precision.
// For values where x <= 0, result is undefined.
func RSqrt_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	return x.ReciprocalSqrt()
}

// RSqrt_AVX512_F64x8 computes approximate 1/sqrt(x) for 8 float64 values.
// Uses the hardware VRSQRT14PD instruction which provides ~14-bit precision.
// For values where x <= 0, result is undefined.
func RSqrt_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	return x.ReciprocalSqrt()
}

// RSqrtNewtonRaphson_AVX512_F32x16 computes 1/sqrt(x) with one Newton-Raphson refinement.
// Provides ~23-bit precision (sufficient for float32).
// Formula: y = y * (1.5 - 0.5 * x * y * y)
func RSqrtNewtonRaphson_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	half := archsimd.BroadcastFloat32x16(0.5)
	threeHalf := archsimd.BroadcastFloat32x16(1.5)

	// Initial approximation
	y := x.ReciprocalSqrt()

	// One Newton-Raphson iteration: y = y * (1.5 - 0.5 * x * y * y)
	xHalf := x.Mul(half)
	yy := y.Mul(y)
	xyy := xHalf.Mul(yy)
	correction := threeHalf.Sub(xyy)
	return y.Mul(correction)
}

// RSqrtNewtonRaphson_AVX512_F64x8 computes 1/sqrt(x) with one Newton-Raphson refinement.
// Provides improved precision over the approximate version.
// Formula: y = y * (1.5 - 0.5 * x * y * y)
func RSqrtNewtonRaphson_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	half := archsimd.BroadcastFloat64x8(0.5)
	threeHalf := archsimd.BroadcastFloat64x8(1.5)

	// Initial approximation
	y := x.ReciprocalSqrt()

	// One Newton-Raphson iteration: y = y * (1.5 - 0.5 * x * y * y)
	xHalf := x.Mul(half)
	yy := y.Mul(y)
	xyy := xHalf.Mul(yy)
	correction := threeHalf.Sub(xyy)
	return y.Mul(correction)
}

// RSqrtPrecise_AVX512_F32x16 computes precise 1/sqrt(x) via sqrt + reciprocal.
// Uses VSQRTPS + VRCP14PS for high precision at higher latency.
func RSqrtPrecise_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	one := archsimd.BroadcastFloat32x16(1.0)
	sqrtX := x.Sqrt()
	return one.Div(sqrtX)
}

// RSqrtPrecise_AVX512_F64x8 computes precise 1/sqrt(x) via sqrt + division.
// Uses VSQRTPD + VDIVPD for full precision.
func RSqrtPrecise_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	one := archsimd.BroadcastFloat64x8(1.0)
	sqrtX := x.Sqrt()
	return one.Div(sqrtX)
}

// Iota_AVX512_F32x16 returns a vector with lane indices [0, 1, ..., 15].
func Iota_AVX512_F32x16() archsimd.Float32x16 {
	return archsimd.LoadFloat32x16Slice([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
}

// Iota_AVX512_F64x8 returns a vector with lane indices [0, 1, ..., 7].
func Iota_AVX512_F64x8() archsimd.Float64x8 {
	return archsimd.LoadFloat64x8Slice([]float64{0, 1, 2, 3, 4, 5, 6, 7})
}

// ReduceMax_AVX512_Uint32x16 returns the maximum element in the vector.
func ReduceMax_AVX512_Uint32x16(v archsimd.Uint32x16) uint32 {
	// Reduce 16 -> 8 -> 4 -> scalar
	lo := v.GetLo()
	hi := v.GetHi()
	max8 := lo.Max(hi)
	return ReduceMax_AVX2_Uint32x8(max8)
}

// ReduceMax_AVX512_Uint64x8 returns the maximum element in the vector.
func ReduceMax_AVX512_Uint64x8(v archsimd.Uint64x8) uint64 {
	// Reduce 8 -> 4 -> 2 -> scalar
	lo := v.GetLo()
	hi := v.GetHi()
	max4 := lo.Max(hi)
	return ReduceMax_AVX2_Uint64x4(max4)
}

// GetLane_AVX512_Uint32x16 extracts the element at the given lane index.
func GetLane_AVX512_Uint32x16(v archsimd.Uint32x16, lane int) uint32 {
	// Uint32x16 -> GetLo/GetHi -> Uint32x8 -> GetLo/GetHi -> Uint32x4 -> GetElem
	if lane < 8 {
		return GetLane_AVX2_Uint32x8(v.GetLo(), lane)
	}
	return GetLane_AVX2_Uint32x8(v.GetHi(), lane-8)
}

// GetLane_AVX512_Uint64x8 extracts the element at the given lane index.
func GetLane_AVX512_Uint64x8(v archsimd.Uint64x8, lane int) uint64 {
	// Uint64x8 -> GetLo/GetHi -> Uint64x4 -> GetLo/GetHi -> Uint64x2 -> GetElem
	if lane < 4 {
		return GetLane_AVX2_Uint64x4(v.GetLo(), lane)
	}
	return GetLane_AVX2_Uint64x4(v.GetHi(), lane-4)
}

// ===== Load4 wrappers for 4x loop unrolling =====
// Unlike NEON which has a single ld1 {v0,v1,v2,v3} instruction,
// AVX-512 loads are already 512-bit, so we simply perform 4 separate loads.

// Load4_AVX512_F32x16 loads 4 consecutive Float32x16 vectors (64 floats = 256 bytes).
func Load4_AVX512_F32x16(s []float32) (archsimd.Float32x16, archsimd.Float32x16, archsimd.Float32x16, archsimd.Float32x16) {
	v0 := archsimd.LoadFloat32x16Slice(s)
	v1 := archsimd.LoadFloat32x16Slice(s[16:])
	v2 := archsimd.LoadFloat32x16Slice(s[32:])
	v3 := archsimd.LoadFloat32x16Slice(s[48:])
	return v0, v1, v2, v3
}

// Load4_AVX512_F64x8 loads 4 consecutive Float64x8 vectors (32 doubles = 256 bytes).
func Load4_AVX512_F64x8(s []float64) (archsimd.Float64x8, archsimd.Float64x8, archsimd.Float64x8, archsimd.Float64x8) {
	v0 := archsimd.LoadFloat64x8Slice(s)
	v1 := archsimd.LoadFloat64x8Slice(s[8:])
	v2 := archsimd.LoadFloat64x8Slice(s[16:])
	v3 := archsimd.LoadFloat64x8Slice(s[24:])
	return v0, v1, v2, v3
}

// Load4_AVX512_I32x16 loads 4 consecutive Int32x16 vectors (64 ints = 256 bytes).
func Load4_AVX512_I32x16(s []int32) (archsimd.Int32x16, archsimd.Int32x16, archsimd.Int32x16, archsimd.Int32x16) {
	v0 := archsimd.LoadInt32x16Slice(s)
	v1 := archsimd.LoadInt32x16Slice(s[16:])
	v2 := archsimd.LoadInt32x16Slice(s[32:])
	v3 := archsimd.LoadInt32x16Slice(s[48:])
	return v0, v1, v2, v3
}

// Load4_AVX512_I64x8 loads 4 consecutive Int64x8 vectors (32 longs = 256 bytes).
func Load4_AVX512_I64x8(s []int64) (archsimd.Int64x8, archsimd.Int64x8, archsimd.Int64x8, archsimd.Int64x8) {
	v0 := archsimd.LoadInt64x8Slice(s)
	v1 := archsimd.LoadInt64x8Slice(s[8:])
	v2 := archsimd.LoadInt64x8Slice(s[16:])
	v3 := archsimd.LoadInt64x8Slice(s[24:])
	return v0, v1, v2, v3
}

// Load4_AVX512_Uint32x16 loads 4 consecutive Uint32x16 vectors (64 uints = 256 bytes).
func Load4_AVX512_Uint32x16(s []uint32) (archsimd.Uint32x16, archsimd.Uint32x16, archsimd.Uint32x16, archsimd.Uint32x16) {
	v0 := archsimd.LoadUint32x16Slice(s)
	v1 := archsimd.LoadUint32x16Slice(s[16:])
	v2 := archsimd.LoadUint32x16Slice(s[32:])
	v3 := archsimd.LoadUint32x16Slice(s[48:])
	return v0, v1, v2, v3
}

// Load4_AVX512_Uint64x8 loads 4 consecutive Uint64x8 vectors (32 ulongs = 256 bytes).
func Load4_AVX512_Uint64x8(s []uint64) (archsimd.Uint64x8, archsimd.Uint64x8, archsimd.Uint64x8, archsimd.Uint64x8) {
	v0 := archsimd.LoadUint64x8Slice(s)
	v1 := archsimd.LoadUint64x8Slice(s[8:])
	v2 := archsimd.LoadUint64x8Slice(s[16:])
	v3 := archsimd.LoadUint64x8Slice(s[24:])
	return v0, v1, v2, v3
}

// Load4_AVX512_Vec loads 4 consecutive Vec vectors (for Float16/BFloat16).
// Falls back to the generic hwy.Load4 implementation.
func Load4_AVX512_Vec[T Lanes](s []T) (Vec[T], Vec[T], Vec[T], Vec[T]) {
	return Load4(s)
}
