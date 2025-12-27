//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides low-level AVX-512 SIMD operations that work directly with
// archsimd vector types. These are core ops (direct hardware instructions),
// not transcendental functions which belong in contrib/math.

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
