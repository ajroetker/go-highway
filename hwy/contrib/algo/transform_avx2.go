//go:build amd64 && goexperiment.simd

package algo

import (
	stdmath "math"
	"simd/archsimd"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

// This file provides zero-allocation Transform functions using AVX2 SIMD.
// These functions process slices directly without the hwy.Vec abstraction overhead.

// Function types for generic Transform operations.
// These allow users to pass custom SIMD operations to Transform32/Transform64.
type (
	// VecFunc32 is a SIMD operation on 8 float32 values.
	VecFunc32 func(archsimd.Float32x8) archsimd.Float32x8

	// VecFunc64 is a SIMD operation on 4 float64 values.
	VecFunc64 func(archsimd.Float64x4) archsimd.Float64x4

	// ScalarFunc32 is a scalar operation on a single float32.
	ScalarFunc32 func(float32) float32

	// ScalarFunc64 is a scalar operation on a single float64.
	ScalarFunc64 func(float64) float64
)

// Transform32 applies a SIMD operation to each element of input, storing results in output.
// This is the generic primitive that all named transforms (ExpTransform, etc.) build upon.
//
// The simd function processes 8 elements at a time using AVX2.
// The scalar function handles remaining elements (0-7) that don't fill a full vector.
//
// Example usage:
//
//	// Apply x² + x to all elements
//	Transform32(input, output,
//	    func(x archsimd.Float32x8) archsimd.Float32x8 {
//	        return x.Mul(x).Add(x)
//	    },
//	    func(x float32) float32 { return x*x + x },
//	)
func Transform32(input, output []float32, simd VecFunc32, scalar ScalarFunc32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		transform32AVX2(input, output, simd, scalar)
	} else {
		transform32Scalar(input, output, scalar)
	}
}

// Transform64 applies a SIMD operation to each element of input, storing results in output.
// This is the generic primitive that all named transforms (ExpTransform64, etc.) build upon.
//
// The simd function processes 4 elements at a time using AVX2.
// The scalar function handles remaining elements (0-3) that don't fill a full vector.
func Transform64(input, output []float64, simd VecFunc64, scalar ScalarFunc64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		transform64AVX2(input, output, simd, scalar)
	} else {
		transform64Scalar(input, output, scalar)
	}
}

func transform32AVX2(input, output []float32, simd VecFunc32, scalar ScalarFunc32) {
	n := min(len(input), len(output))

	// Process 8 float32s at a time
	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(input[i:])
		simd(x).StoreSlice(output[i:])
	}

	// Scalar tail
	for i := (n / 8) * 8; i < n; i++ {
		output[i] = scalar(input[i])
	}
}

func transform64AVX2(input, output []float64, simd VecFunc64, scalar ScalarFunc64) {
	n := min(len(input), len(output))

	// Process 4 float64s at a time
	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(input[i:])
		simd(x).StoreSlice(output[i:])
	}

	// Scalar tail
	for i := (n / 4) * 4; i < n; i++ {
		output[i] = scalar(input[i])
	}
}

func transform32Scalar(input, output []float32, scalar ScalarFunc32) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = scalar(input[i])
	}
}

func transform64Scalar(input, output []float64, scalar ScalarFunc64) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = scalar(input[i])
	}
}

// Scalar helper functions for tail elements
func exp32Scalar(x float32) float32       { return float32(stdmath.Exp(float64(x))) }
func exp64Scalar(x float64) float64       { return stdmath.Exp(x) }
func log32Scalar(x float32) float32       { return float32(stdmath.Log(float64(x))) }
func log64Scalar(x float64) float64       { return stdmath.Log(x) }
func log2_32Scalar(x float32) float32     { return float32(stdmath.Log2(float64(x))) }
func log2_64Scalar(x float64) float64     { return stdmath.Log2(x) }
func log10_32Scalar(x float32) float32    { return float32(stdmath.Log10(float64(x))) }
func log10_64Scalar(x float64) float64    { return stdmath.Log10(x) }
func exp2_32Scalar(x float32) float32     { return float32(stdmath.Exp2(float64(x))) }
func exp2_64Scalar(x float64) float64     { return stdmath.Exp2(x) }
func sin32Scalar(x float32) float32       { return float32(stdmath.Sin(float64(x))) }
func sin64Scalar(x float64) float64       { return stdmath.Sin(x) }
func cos32Scalar(x float32) float32       { return float32(stdmath.Cos(float64(x))) }
func cos64Scalar(x float64) float64       { return stdmath.Cos(x) }
func tanh32Scalar(x float32) float32      { return float32(stdmath.Tanh(float64(x))) }
func tanh64Scalar(x float64) float64      { return stdmath.Tanh(x) }
func sinh32Scalar(x float32) float32      { return float32(stdmath.Sinh(float64(x))) }
func sinh64Scalar(x float64) float64      { return stdmath.Sinh(x) }
func cosh32Scalar(x float32) float32      { return float32(stdmath.Cosh(float64(x))) }
func cosh64Scalar(x float64) float64      { return stdmath.Cosh(x) }
func sqrt32Scalar(x float32) float32      { return float32(stdmath.Sqrt(float64(x))) }
func sqrt64Scalar(x float64) float64      { return stdmath.Sqrt(x) }
func sigmoid32Scalar(x float32) float32   { return float32(1.0 / (1.0 + stdmath.Exp(-float64(x)))) }
func sigmoid64Scalar(x float64) float64   { return 1.0 / (1.0 + stdmath.Exp(-x)) }
func erf32Scalar(x float32) float32       { return float32(stdmath.Erf(float64(x))) }
func erf64Scalar(x float64) float64       { return stdmath.Erf(x) }

// ExpTransform applies exp(x) to each element with zero allocations.
// Caller must ensure len(output) >= len(input).
func ExpTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Exp_AVX512_F32x16, exp32Scalar)
	} else {
		Transform32(input, output, math.Exp_AVX2_F32x8, exp32Scalar)
	}
}

// ExpTransform64 applies exp(x) to each float64 element with zero allocations.
func ExpTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Exp_AVX512_F64x8, exp64Scalar)
	} else {
		Transform64(input, output, math.Exp_AVX2_F64x4, exp64Scalar)
	}
}

// LogTransform applies ln(x) to each element with zero allocations.
func LogTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Log_AVX512_F32x16, log32Scalar)
	} else {
		Transform32(input, output, math.Log_AVX2_F32x8, log32Scalar)
	}
}

// LogTransform64 applies ln(x) to each float64 element with zero allocations.
func LogTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Log_AVX512_F64x8, log64Scalar)
	} else {
		Transform64(input, output, math.Log_AVX2_F64x4, log64Scalar)
	}
}

// SinTransform applies sin(x) to each element with zero allocations.
func SinTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Sin_AVX512_F32x16, sin32Scalar)
	} else {
		Transform32(input, output, math.Sin_AVX2_F32x8, sin32Scalar)
	}
}

// SinTransform64 applies sin(x) to each float64 element with zero allocations.
func SinTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Sin_AVX512_F64x8, sin64Scalar)
	} else {
		Transform64(input, output, math.Sin_AVX2_F64x4, sin64Scalar)
	}
}

// CosTransform applies cos(x) to each element with zero allocations.
func CosTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Cos_AVX512_F32x16, cos32Scalar)
	} else {
		Transform32(input, output, math.Cos_AVX2_F32x8, cos32Scalar)
	}
}

// CosTransform64 applies cos(x) to each float64 element with zero allocations.
func CosTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Cos_AVX512_F64x8, cos64Scalar)
	} else {
		Transform64(input, output, math.Cos_AVX2_F64x4, cos64Scalar)
	}
}

// TanhTransform applies tanh(x) to each element with zero allocations.
func TanhTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Tanh_AVX512_F32x16, tanh32Scalar)
	} else {
		Transform32(input, output, math.Tanh_AVX2_F32x8, tanh32Scalar)
	}
}

// TanhTransform64 applies tanh(x) to each float64 element with zero allocations.
func TanhTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Tanh_AVX512_F64x8, tanh64Scalar)
	} else {
		Transform64(input, output, math.Tanh_AVX2_F64x4, tanh64Scalar)
	}
}

// SigmoidTransform applies sigmoid(x) = 1/(1+exp(-x)) to each element with zero allocations.
func SigmoidTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Sigmoid_AVX512_F32x16, sigmoid32Scalar)
	} else {
		Transform32(input, output, math.Sigmoid_AVX2_F32x8, sigmoid32Scalar)
	}
}

// SigmoidTransform64 applies sigmoid(x) to each float64 element with zero allocations.
func SigmoidTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Sigmoid_AVX512_F64x8, sigmoid64Scalar)
	} else {
		Transform64(input, output, math.Sigmoid_AVX2_F64x4, sigmoid64Scalar)
	}
}

// ErfTransform applies erf(x) to each element with zero allocations.
func ErfTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Erf_AVX512_F32x16, erf32Scalar)
	} else {
		Transform32(input, output, math.Erf_AVX2_F32x8, erf32Scalar)
	}
}

// ErfTransform64 applies erf(x) to each float64 element with zero allocations.
func ErfTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Erf_AVX512_F64x8, erf64Scalar)
	} else {
		Transform64(input, output, math.Erf_AVX2_F64x4, erf64Scalar)
	}
}

// Log2Transform applies log₂(x) to each element with zero allocations.
func Log2Transform(input, output []float32) {
	Transform32(input, output, math.Log2_AVX2_F32x8, log2_32Scalar)
}

// Log2Transform64 applies log₂(x) to each float64 element with zero allocations.
func Log2Transform64(input, output []float64) {
	Transform64(input, output, math.Log2_AVX2_F64x4, log2_64Scalar)
}

// Log10Transform applies log₁₀(x) to each element with zero allocations.
func Log10Transform(input, output []float32) {
	Transform32(input, output, math.Log10_AVX2_F32x8, log10_32Scalar)
}

// Log10Transform64 applies log₁₀(x) to each float64 element with zero allocations.
func Log10Transform64(input, output []float64) {
	Transform64(input, output, math.Log10_AVX2_F64x4, log10_64Scalar)
}

// Exp2Transform applies 2^x to each element with zero allocations.
func Exp2Transform(input, output []float32) {
	Transform32(input, output, math.Exp2_AVX2_F32x8, exp2_32Scalar)
}

// Exp2Transform64 applies 2^x to each float64 element with zero allocations.
func Exp2Transform64(input, output []float64) {
	Transform64(input, output, math.Exp2_AVX2_F64x4, exp2_64Scalar)
}

// SinhTransform applies sinh(x) to each element with zero allocations.
func SinhTransform(input, output []float32) {
	Transform32(input, output, math.Sinh_AVX2_F32x8, sinh32Scalar)
}

// SinhTransform64 applies sinh(x) to each float64 element with zero allocations.
func SinhTransform64(input, output []float64) {
	Transform64(input, output, math.Sinh_AVX2_F64x4, sinh64Scalar)
}

// CoshTransform applies cosh(x) to each element with zero allocations.
func CoshTransform(input, output []float32) {
	Transform32(input, output, math.Cosh_AVX2_F32x8, cosh32Scalar)
}

// CoshTransform64 applies cosh(x) to each float64 element with zero allocations.
func CoshTransform64(input, output []float64) {
	Transform64(input, output, math.Cosh_AVX2_F64x4, cosh64Scalar)
}

// SqrtTransform applies sqrt(x) to each element with zero allocations.
// Note: Sqrt is a core op (hardware instruction), not a transcendental.
func SqrtTransform(input, output []float32) {
	Transform32(input, output, hwy.Sqrt_AVX2_F32x8, sqrt32Scalar)
}

// SqrtTransform64 applies sqrt(x) to each float64 element with zero allocations.
// Note: Sqrt is a core op (hardware instruction), not a transcendental.
func SqrtTransform64(input, output []float64) {
	Transform64(input, output, hwy.Sqrt_AVX2_F64x4, sqrt64Scalar)
}
