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
