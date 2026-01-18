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
	// Reduce 4 -> 2 -> 1
	lo := v.GetLo()
	hi := v.GetHi()
	max2 := lo.Max(hi)
	// Now max2 is Uint64x2, reduce further
	e0 := max2.GetElem(0)
	e1 := max2.GetElem(1)
	if e1 > e0 {
		return e1
	}
	return e0
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
