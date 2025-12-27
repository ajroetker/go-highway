// Package math provides high-performance SIMD transcendental math functions.
// This package corresponds to Google Highway's hwy/contrib/math directory.
//
// # Low-Level SIMD Functions
//
// Direct SIMD vector functions for library authors building custom operations.
// These work with archsimd vector types and require GOEXPERIMENT=simd.
//
// Float32x8 functions (AVX2):
//
// Exponential and logarithmic:
//   - Exp_AVX2_F32x8(x Float32x8) Float32x8 - e^x
//   - Exp2_AVX2_F32x8(x Float32x8) Float32x8 - 2^x
//   - Exp10_AVX2_F32x8(x Float32x8) Float32x8 - 10^x
//   - Log_AVX2_F32x8(x Float32x8) Float32x8 - ln(x)
//   - Log2_AVX2_F32x8(x Float32x8) Float32x8 - log₂(x)
//   - Log10_AVX2_F32x8(x Float32x8) Float32x8 - log₁₀(x)
//
// Trigonometric:
//   - Sin_AVX2_F32x8(x Float32x8) Float32x8
//   - Cos_AVX2_F32x8(x Float32x8) Float32x8
//   - SinCos_AVX2_F32x8(x Float32x8) (sin, cos Float32x8)
//
// Hyperbolic:
//   - Sinh_AVX2_F32x8(x Float32x8) Float32x8
//   - Cosh_AVX2_F32x8(x Float32x8) Float32x8
//   - Tanh_AVX2_F32x8(x Float32x8) Float32x8
//   - Asinh_AVX2_F32x8(x Float32x8) Float32x8
//   - Acosh_AVX2_F32x8(x Float32x8) Float32x8
//   - Atanh_AVX2_F32x8(x Float32x8) Float32x8
//
// Activation functions:
//   - Sigmoid_AVX2_F32x8(x Float32x8) Float32x8
//   - Erf_AVX2_F32x8(x Float32x8) Float32x8
//
// Float64x4 functions (AVX2):
//
// Exponential and logarithmic:
//   - Exp_AVX2_F64x4(x Float64x4) Float64x4 - e^x
//   - Exp2_AVX2_F64x4(x Float64x4) Float64x4 - 2^x
//   - Exp10_AVX2_F64x4(x Float64x4) Float64x4 - 10^x
//   - Log_AVX2_F64x4(x Float64x4) Float64x4 - ln(x)
//   - Log2_AVX2_F64x4(x Float64x4) Float64x4 - log₂(x)
//   - Log10_AVX2_F64x4(x Float64x4) Float64x4 - log₁₀(x)
//
// Trigonometric:
//   - Sin_AVX2_F64x4(x Float64x4) Float64x4
//   - Cos_AVX2_F64x4(x Float64x4) Float64x4
//   - SinCos_AVX2_F64x4(x Float64x4) (sin, cos Float64x4)
//
// Hyperbolic:
//   - Sinh_AVX2_F64x4(x Float64x4) Float64x4
//   - Cosh_AVX2_F64x4(x Float64x4) Float64x4
//   - Tanh_AVX2_F64x4(x Float64x4) Float64x4
//   - Asinh_AVX2_F64x4(x Float64x4) Float64x4
//   - Acosh_AVX2_F64x4(x Float64x4) Float64x4
//   - Atanh_AVX2_F64x4(x Float64x4) Float64x4
//
// Activation functions:
//   - Sigmoid_AVX2_F64x4(x Float64x4) Float64x4
//   - Erf_AVX2_F64x4(x Float64x4) Float64x4
//
// Note: Sqrt is a core op in the hwy package (uses hardware VSQRT* instructions).
// Use hwy.Sqrt_AVX2_F32x8 and hwy.Sqrt_AVX2_F64x4 for SIMD sqrt.
//
// AVX-512 variants are also available (e.g., Exp_AVX512_F32x16).
//
// # Polynomial Utilities
//
// Horner's method for efficient polynomial evaluation:
//   - Horner[T](x Vec[T], coeffs []T) Vec[T]
//   - Horner5[T](x Vec[T], c0, c1, c2, c3, c4, c5 T) Vec[T]
//   - Horner7[T](x Vec[T], c0, ..., c7 T) Vec[T]
//   - Horner13[T](x Vec[T], coeffs [14]T) Vec[T]
//
// # Accuracy
//
// All functions are designed to provide reasonable accuracy for typical
// machine learning and signal processing applications:
//   - Maximum error: ~4 ULP for most functions
//   - Special value handling: ±Inf, NaN, denormals
//
// # Example Usage
//
//	import (
//	    "simd/archsimd"
//	    "github.com/ajroetker/go-highway/hwy/contrib/math"
//	)
//
//	// Direct SIMD function usage
//	func ProcessVectors(x archsimd.Float32x8) archsimd.Float32x8 {
//	    expX := math.Exp_AVX2_F32x8(x)
//	    return expX.Mul(x)  // x * exp(x)
//	}
//
//	// Using polynomial utilities
//	func EvaluatePolynomial(x hwy.Vec[float32]) hwy.Vec[float32] {
//	    coeffs := []float32{1.0, 2.0, 3.0}  // 1 + 2x + 3x²
//	    return math.Horner(x, coeffs)
//	}
//
// # Build Requirements
//
// The SIMD implementations require:
//   - GOEXPERIMENT=simd build flag
//   - AMD64 architecture with AVX2 or AVX-512 support
package math
