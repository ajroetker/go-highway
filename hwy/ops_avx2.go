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

// This file provides low-level AVX2 SIMD operations that work directly with
// archsimd vector types. These are core ops (direct hardware instructions),
// not transcendental functions which belong in contrib/math.
//
// These functions are used by contrib/algo transforms and can be used directly
// by users who want to work with raw SIMD types instead of the Vec abstraction.

// ReduceSum_AVX2_F32x8 returns the sum of all 8 float32 elements.
func ReduceSum_AVX2_F32x8(v archsimd.Float32x8) float32 {
	// Reduce 8 -> 4 -> 2 -> 1 using horizontal adds
	lo := v.GetLo() // Float32x4
	hi := v.GetHi() // Float32x4
	sum4 := lo.Add(hi)
	// sum4 is Float32x4, extract and sum
	e0 := sum4.GetElem(0)
	e1 := sum4.GetElem(1)
	e2 := sum4.GetElem(2)
	e3 := sum4.GetElem(3)
	return e0 + e1 + e2 + e3
}

// ReduceSum_AVX2_F64x4 returns the sum of all 4 float64 elements.
func ReduceSum_AVX2_F64x4(v archsimd.Float64x4) float64 {
	lo := v.GetLo() // Float64x2
	hi := v.GetHi() // Float64x2
	sum2 := lo.Add(hi)
	e0 := sum2.GetElem(0)
	e1 := sum2.GetElem(1)
	return e0 + e1
}

// Sqrt_AVX2_F32x8 computes sqrt(x) for a single Float32x8 vector.
// Uses the hardware VSQRTPS instruction which provides correctly rounded results.
func Sqrt_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	return x.Sqrt()
}

// Sqrt_AVX2_F64x4 computes sqrt(x) for a single Float64x4 vector.
// Uses the hardware VSQRTPD instruction which provides correctly rounded results.
func Sqrt_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	return x.Sqrt()
}

// RSqrt_AVX2_F32x8 computes approximate 1/sqrt(x) for 8 float32 values.
// Uses the hardware VRSQRTPS instruction which provides ~12-bit precision.
// For values where x <= 0, result is undefined.
func RSqrt_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	return x.ReciprocalSqrt()
}

// RSqrt_AVX2_F64x4 computes approximate 1/sqrt(x) for 4 float64 values.
// Uses the hardware VRSQRTPD instruction which provides ~12-bit precision.
// For values where x <= 0, result is undefined.
func RSqrt_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	return x.ReciprocalSqrt()
}

// RSqrtNewtonRaphson_AVX2_F32x8 computes 1/sqrt(x) with one Newton-Raphson refinement.
// Provides ~23-bit precision (sufficient for float32).
// Formula: y = y * (1.5 - 0.5 * x * y * y)
func RSqrtNewtonRaphson_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	half := archsimd.BroadcastFloat32x8(0.5)
	threeHalf := archsimd.BroadcastFloat32x8(1.5)

	// Initial approximation
	y := x.ReciprocalSqrt()

	// One Newton-Raphson iteration: y = y * (1.5 - 0.5 * x * y * y)
	xHalf := x.Mul(half)
	yy := y.Mul(y)
	xyy := xHalf.Mul(yy)
	correction := threeHalf.Sub(xyy)
	return y.Mul(correction)
}

// RSqrtNewtonRaphson_AVX2_F64x4 computes 1/sqrt(x) with one Newton-Raphson refinement.
// Provides improved precision over the approximate version.
// Formula: y = y * (1.5 - 0.5 * x * y * y)
func RSqrtNewtonRaphson_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	half := archsimd.BroadcastFloat64x4(0.5)
	threeHalf := archsimd.BroadcastFloat64x4(1.5)

	// Initial approximation
	y := x.ReciprocalSqrt()

	// One Newton-Raphson iteration: y = y * (1.5 - 0.5 * x * y * y)
	xHalf := x.Mul(half)
	yy := y.Mul(y)
	xyy := xHalf.Mul(yy)
	correction := threeHalf.Sub(xyy)
	return y.Mul(correction)
}

// RSqrtPrecise_AVX2_F32x8 computes precise 1/sqrt(x) via sqrt + reciprocal.
// Uses VSQRTPS + VRCPPS for high precision at higher latency.
func RSqrtPrecise_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	one := archsimd.BroadcastFloat32x8(1.0)
	sqrtX := x.Sqrt()
	return one.Div(sqrtX)
}

// RSqrtPrecise_AVX2_F64x4 computes precise 1/sqrt(x) via sqrt + division.
// Uses VSQRTPD + VDIVPD for full precision.
func RSqrtPrecise_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	one := archsimd.BroadcastFloat64x4(1.0)
	sqrtX := x.Sqrt()
	return one.Div(sqrtX)
}

// ReduceMax_AVX2_Uint32x8 returns the maximum element in the vector.
func ReduceMax_AVX2_Uint32x8(v archsimd.Uint32x8) uint32 {
	// Reduce 8 -> 4 -> 2 -> 1
	lo := v.GetLo()
	hi := v.GetHi()
	max4 := lo.Max(hi)
	// Now max4 is Uint32x4, reduce further
	e0 := max4.GetElem(0)
	e1 := max4.GetElem(1)
	e2 := max4.GetElem(2)
	e3 := max4.GetElem(3)
	m := e0
	if e1 > m {
		m = e1
	}
	if e2 > m {
		m = e2
	}
	if e3 > m {
		m = e3
	}
	return m
}

// ReduceMax_AVX2_Uint64x4 returns the maximum element in the vector.
func ReduceMax_AVX2_Uint64x4(v archsimd.Uint64x4) uint64 {
	// AVX2 doesn't have VPMAXUQ (unsigned 64-bit max), so extract all elements
	// and compare scalarly
	lo := v.GetLo()
	hi := v.GetHi()
	e0 := lo.GetElem(0)
	e1 := lo.GetElem(1)
	e2 := hi.GetElem(0)
	e3 := hi.GetElem(1)
	m := e0
	if e1 > m {
		m = e1
	}
	if e2 > m {
		m = e2
	}
	if e3 > m {
		m = e3
	}
	return m
}

// Max_AVX2_Uint64x4 returns the element-wise maximum of two Uint64x4 vectors.
// AVX2 doesn't have VPMAXUQ (unsigned 64-bit max), so we use scalar comparison.
func Max_AVX2_Uint64x4(a, b archsimd.Uint64x4) archsimd.Uint64x4 {
	var result [4]uint64
	aLo, aHi := a.GetLo(), a.GetHi()
	bLo, bHi := b.GetLo(), b.GetHi()
	a0, a1 := aLo.GetElem(0), aLo.GetElem(1)
	a2, a3 := aHi.GetElem(0), aHi.GetElem(1)
	b0, b1 := bLo.GetElem(0), bLo.GetElem(1)
	b2, b3 := bHi.GetElem(0), bHi.GetElem(1)
	if a0 > b0 {
		result[0] = a0
	} else {
		result[0] = b0
	}
	if a1 > b1 {
		result[1] = a1
	} else {
		result[1] = b1
	}
	if a2 > b2 {
		result[2] = a2
	} else {
		result[2] = b2
	}
	if a3 > b3 {
		result[3] = a3
	} else {
		result[3] = b3
	}
	return archsimd.LoadUint64x4Slice(result[:])
}

// Max_AVX2_Int64x4 returns the element-wise maximum of two Int64x4 vectors.
// AVX2 doesn't have VPMAXSQ (signed 64-bit max), so we use scalar comparison.
func Max_AVX2_Int64x4(a, b archsimd.Int64x4) archsimd.Int64x4 {
	var result [4]int64
	aLo, aHi := a.GetLo(), a.GetHi()
	bLo, bHi := b.GetLo(), b.GetHi()
	a0, a1 := aLo.GetElem(0), aLo.GetElem(1)
	a2, a3 := aHi.GetElem(0), aHi.GetElem(1)
	b0, b1 := bLo.GetElem(0), bLo.GetElem(1)
	b2, b3 := bHi.GetElem(0), bHi.GetElem(1)
	if a0 > b0 {
		result[0] = a0
	} else {
		result[0] = b0
	}
	if a1 > b1 {
		result[1] = a1
	} else {
		result[1] = b1
	}
	if a2 > b2 {
		result[2] = a2
	} else {
		result[2] = b2
	}
	if a3 > b3 {
		result[3] = a3
	} else {
		result[3] = b3
	}
	return archsimd.LoadInt64x4Slice(result[:])
}

// Min_AVX2_Uint64x4 returns the element-wise minimum of two Uint64x4 vectors.
// AVX2 doesn't have VPMINUQ (unsigned 64-bit min), so we use scalar comparison.
func Min_AVX2_Uint64x4(a, b archsimd.Uint64x4) archsimd.Uint64x4 {
	var result [4]uint64
	aLo, aHi := a.GetLo(), a.GetHi()
	bLo, bHi := b.GetLo(), b.GetHi()
	a0, a1 := aLo.GetElem(0), aLo.GetElem(1)
	a2, a3 := aHi.GetElem(0), aHi.GetElem(1)
	b0, b1 := bLo.GetElem(0), bLo.GetElem(1)
	b2, b3 := bHi.GetElem(0), bHi.GetElem(1)
	if a0 < b0 {
		result[0] = a0
	} else {
		result[0] = b0
	}
	if a1 < b1 {
		result[1] = a1
	} else {
		result[1] = b1
	}
	if a2 < b2 {
		result[2] = a2
	} else {
		result[2] = b2
	}
	if a3 < b3 {
		result[3] = a3
	} else {
		result[3] = b3
	}
	return archsimd.LoadUint64x4Slice(result[:])
}

// Min_AVX2_Int64x4 returns the element-wise minimum of two Int64x4 vectors.
// AVX2 doesn't have VPMINSQ (signed 64-bit min), so we use scalar comparison.
func Min_AVX2_Int64x4(a, b archsimd.Int64x4) archsimd.Int64x4 {
	var result [4]int64
	aLo, aHi := a.GetLo(), a.GetHi()
	bLo, bHi := b.GetLo(), b.GetHi()
	a0, a1 := aLo.GetElem(0), aLo.GetElem(1)
	a2, a3 := aHi.GetElem(0), aHi.GetElem(1)
	b0, b1 := bLo.GetElem(0), bLo.GetElem(1)
	b2, b3 := bHi.GetElem(0), bHi.GetElem(1)
	if a0 < b0 {
		result[0] = a0
	} else {
		result[0] = b0
	}
	if a1 < b1 {
		result[1] = a1
	} else {
		result[1] = b1
	}
	if a2 < b2 {
		result[2] = a2
	} else {
		result[2] = b2
	}
	if a3 < b3 {
		result[3] = a3
	} else {
		result[3] = b3
	}
	return archsimd.LoadInt64x4Slice(result[:])
}

// GetLane_AVX2_Uint32x8 extracts the element at the given lane index.
func GetLane_AVX2_Uint32x8(v archsimd.Uint32x8, lane int) uint32 {
	if lane < 4 {
		return v.GetLo().GetElem(uint8(lane))
	}
	return v.GetHi().GetElem(uint8(lane - 4))
}

// GetLane_AVX2_Uint64x4 extracts the element at the given lane index.
func GetLane_AVX2_Uint64x4(v archsimd.Uint64x4, lane int) uint64 {
	if lane < 2 {
		return v.GetLo().GetElem(uint8(lane))
	}
	return v.GetHi().GetElem(uint8(lane - 2))
}

// GetLane_AVX2_I32x8 extracts the element at the given lane index.
func GetLane_AVX2_I32x8(v archsimd.Int32x8, lane int) int32 {
	if lane < 4 {
		return v.GetLo().GetElem(uint8(lane))
	}
	return v.GetHi().GetElem(uint8(lane - 4))
}

// GetLane_AVX2_I64x4 extracts the element at the given lane index.
func GetLane_AVX2_I64x4(v archsimd.Int64x4, lane int) int64 {
	if lane < 2 {
		return v.GetLo().GetElem(uint8(lane))
	}
	return v.GetHi().GetElem(uint8(lane - 2))
}

// GetLane_AVX2_F32x8 extracts the element at the given lane index.
func GetLane_AVX2_F32x8(v archsimd.Float32x8, lane int) float32 {
	if lane < 4 {
		return v.GetLo().GetElem(uint8(lane))
	}
	return v.GetHi().GetElem(uint8(lane - 4))
}

// GetLane_AVX2_F64x4 extracts the element at the given lane index.
func GetLane_AVX2_F64x4(v archsimd.Float64x4, lane int) float64 {
	if lane < 2 {
		return v.GetLo().GetElem(uint8(lane))
	}
	return v.GetHi().GetElem(uint8(lane - 2))
}

// ===== Load4 wrappers for 4x loop unrolling =====
// Unlike NEON which has a single ld1 {v0,v1,v2,v3} instruction,
// AVX2 loads are already 256-bit, so we simply perform 4 separate loads.

// Load4_AVX2_F32x8 loads 4 consecutive Float32x8 vectors (32 floats = 128 bytes).
func Load4_AVX2_F32x8(s []float32) (archsimd.Float32x8, archsimd.Float32x8, archsimd.Float32x8, archsimd.Float32x8) {
	v0 := archsimd.LoadFloat32x8Slice(s)
	v1 := archsimd.LoadFloat32x8Slice(s[8:])
	v2 := archsimd.LoadFloat32x8Slice(s[16:])
	v3 := archsimd.LoadFloat32x8Slice(s[24:])
	return v0, v1, v2, v3
}

// Load4_AVX2_F64x4 loads 4 consecutive Float64x4 vectors (16 doubles = 128 bytes).
func Load4_AVX2_F64x4(s []float64) (archsimd.Float64x4, archsimd.Float64x4, archsimd.Float64x4, archsimd.Float64x4) {
	v0 := archsimd.LoadFloat64x4Slice(s)
	v1 := archsimd.LoadFloat64x4Slice(s[4:])
	v2 := archsimd.LoadFloat64x4Slice(s[8:])
	v3 := archsimd.LoadFloat64x4Slice(s[12:])
	return v0, v1, v2, v3
}

// Load4_AVX2_I32x8 loads 4 consecutive Int32x8 vectors (32 ints = 128 bytes).
func Load4_AVX2_I32x8(s []int32) (archsimd.Int32x8, archsimd.Int32x8, archsimd.Int32x8, archsimd.Int32x8) {
	v0 := archsimd.LoadInt32x8Slice(s)
	v1 := archsimd.LoadInt32x8Slice(s[8:])
	v2 := archsimd.LoadInt32x8Slice(s[16:])
	v3 := archsimd.LoadInt32x8Slice(s[24:])
	return v0, v1, v2, v3
}

// Load4_AVX2_I64x4 loads 4 consecutive Int64x4 vectors (16 longs = 128 bytes).
func Load4_AVX2_I64x4(s []int64) (archsimd.Int64x4, archsimd.Int64x4, archsimd.Int64x4, archsimd.Int64x4) {
	v0 := archsimd.LoadInt64x4Slice(s)
	v1 := archsimd.LoadInt64x4Slice(s[4:])
	v2 := archsimd.LoadInt64x4Slice(s[8:])
	v3 := archsimd.LoadInt64x4Slice(s[12:])
	return v0, v1, v2, v3
}

// Load4_AVX2_Uint32x8 loads 4 consecutive Uint32x8 vectors (32 uints = 128 bytes).
func Load4_AVX2_Uint32x8(s []uint32) (archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8) {
	v0 := archsimd.LoadUint32x8Slice(s)
	v1 := archsimd.LoadUint32x8Slice(s[8:])
	v2 := archsimd.LoadUint32x8Slice(s[16:])
	v3 := archsimd.LoadUint32x8Slice(s[24:])
	return v0, v1, v2, v3
}

// Load4_AVX2_Uint64x4 loads 4 consecutive Uint64x4 vectors (16 ulongs = 128 bytes).
func Load4_AVX2_Uint64x4(s []uint64) (archsimd.Uint64x4, archsimd.Uint64x4, archsimd.Uint64x4, archsimd.Uint64x4) {
	v0 := archsimd.LoadUint64x4Slice(s)
	v1 := archsimd.LoadUint64x4Slice(s[4:])
	v2 := archsimd.LoadUint64x4Slice(s[8:])
	v3 := archsimd.LoadUint64x4Slice(s[12:])
	return v0, v1, v2, v3
}

// Load4_AVX2_Vec loads 4 consecutive Vec vectors (for Float16/BFloat16).
// Falls back to the generic hwy.Load4 implementation.
func Load4_AVX2_Vec[T Lanes](s []T) (Vec[T], Vec[T], Vec[T], Vec[T]) {
	return Load4(s)
}
