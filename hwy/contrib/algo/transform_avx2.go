//go:build amd64 && goexperiment.simd

package algo

import (
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

// ExpTransform applies exp(x) to each element with zero allocations.
// Caller must ensure len(output) >= len(input).
func ExpTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Exp_AVX512_F32x16, math.Exp32Scalar)
	} else {
		Transform32(input, output, math.Exp_AVX2_F32x8, math.Exp32Scalar)
	}
}

// ExpTransform64 applies exp(x) to each float64 element with zero allocations.
func ExpTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Exp_AVX512_F64x8, math.Exp64Scalar)
	} else {
		Transform64(input, output, math.Exp_AVX2_F64x4, math.Exp64Scalar)
	}
}

// LogTransform applies ln(x) to each element with zero allocations.
func LogTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Log_AVX512_F32x16, math.Log32Scalar)
	} else {
		Transform32(input, output, math.Log_AVX2_F32x8, math.Log32Scalar)
	}
}

// LogTransform64 applies ln(x) to each float64 element with zero allocations.
func LogTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Log_AVX512_F64x8, math.Log64Scalar)
	} else {
		Transform64(input, output, math.Log_AVX2_F64x4, math.Log64Scalar)
	}
}

// SinTransform applies sin(x) to each element with zero allocations.
func SinTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Sin_AVX512_F32x16, math.Sin32Scalar)
	} else {
		Transform32(input, output, math.Sin_AVX2_F32x8, math.Sin32Scalar)
	}
}

// SinTransform64 applies sin(x) to each float64 element with zero allocations.
func SinTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Sin_AVX512_F64x8, math.Sin64Scalar)
	} else {
		Transform64(input, output, math.Sin_AVX2_F64x4, math.Sin64Scalar)
	}
}

// CosTransform applies cos(x) to each element with zero allocations.
func CosTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Cos_AVX512_F32x16, math.Cos32Scalar)
	} else {
		Transform32(input, output, math.Cos_AVX2_F32x8, math.Cos32Scalar)
	}
}

// CosTransform64 applies cos(x) to each float64 element with zero allocations.
func CosTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Cos_AVX512_F64x8, math.Cos64Scalar)
	} else {
		Transform64(input, output, math.Cos_AVX2_F64x4, math.Cos64Scalar)
	}
}

// TanhTransform applies tanh(x) to each element with zero allocations.
func TanhTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Tanh_AVX512_F32x16, math.Tanh32Scalar)
	} else {
		Transform32(input, output, math.Tanh_AVX2_F32x8, math.Tanh32Scalar)
	}
}

// TanhTransform64 applies tanh(x) to each float64 element with zero allocations.
func TanhTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Tanh_AVX512_F64x8, math.Tanh64Scalar)
	} else {
		Transform64(input, output, math.Tanh_AVX2_F64x4, math.Tanh64Scalar)
	}
}

// SigmoidTransform applies sigmoid(x) = 1/(1+exp(-x)) to each element with zero allocations.
func SigmoidTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Sigmoid_AVX512_F32x16, math.Sigmoid32Scalar)
	} else {
		Transform32(input, output, math.Sigmoid_AVX2_F32x8, math.Sigmoid32Scalar)
	}
}

// SigmoidTransform64 applies sigmoid(x) to each float64 element with zero allocations.
func SigmoidTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Sigmoid_AVX512_F64x8, math.Sigmoid64Scalar)
	} else {
		Transform64(input, output, math.Sigmoid_AVX2_F64x4, math.Sigmoid64Scalar)
	}
}

// ErfTransform applies erf(x) to each element with zero allocations.
func ErfTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform32x16(input, output, math.Erf_AVX512_F32x16, math.Erf32Scalar)
	} else {
		Transform32(input, output, math.Erf_AVX2_F32x8, math.Erf32Scalar)
	}
}

// ErfTransform64 applies erf(x) to each float64 element with zero allocations.
func ErfTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		Transform64x8(input, output, math.Erf_AVX512_F64x8, math.Erf64Scalar)
	} else {
		Transform64(input, output, math.Erf_AVX2_F64x4, math.Erf64Scalar)
	}
}

// Log2Transform applies log₂(x) to each element with zero allocations.
func Log2Transform(input, output []float32) {
	Transform32(input, output, math.Log2_AVX2_F32x8, math.Log2_32Scalar)
}

// Log2Transform64 applies log₂(x) to each float64 element with zero allocations.
func Log2Transform64(input, output []float64) {
	Transform64(input, output, math.Log2_AVX2_F64x4, math.Log2_64Scalar)
}

// Log10Transform applies log₁₀(x) to each element with zero allocations.
func Log10Transform(input, output []float32) {
	Transform32(input, output, math.Log10_AVX2_F32x8, math.Log10_32Scalar)
}

// Log10Transform64 applies log₁₀(x) to each float64 element with zero allocations.
func Log10Transform64(input, output []float64) {
	Transform64(input, output, math.Log10_AVX2_F64x4, math.Log10_64Scalar)
}

// Exp2Transform applies 2^x to each element with zero allocations.
func Exp2Transform(input, output []float32) {
	Transform32(input, output, math.Exp2_AVX2_F32x8, math.Exp2_32Scalar)
}

// Exp2Transform64 applies 2^x to each float64 element with zero allocations.
func Exp2Transform64(input, output []float64) {
	Transform64(input, output, math.Exp2_AVX2_F64x4, math.Exp2_64Scalar)
}

// SinhTransform applies sinh(x) to each element with zero allocations.
func SinhTransform(input, output []float32) {
	Transform32(input, output, math.Sinh_AVX2_F32x8, math.Sinh32Scalar)
}

// SinhTransform64 applies sinh(x) to each float64 element with zero allocations.
func SinhTransform64(input, output []float64) {
	Transform64(input, output, math.Sinh_AVX2_F64x4, math.Sinh64Scalar)
}

// CoshTransform applies cosh(x) to each element with zero allocations.
func CoshTransform(input, output []float32) {
	Transform32(input, output, math.Cosh_AVX2_F32x8, math.Cosh32Scalar)
}

// CoshTransform64 applies cosh(x) to each float64 element with zero allocations.
func CoshTransform64(input, output []float64) {
	Transform64(input, output, math.Cosh_AVX2_F64x4, math.Cosh64Scalar)
}

// SqrtTransform applies sqrt(x) to each element with zero allocations.
// Note: Sqrt is a core op (hardware instruction), not a transcendental.
func SqrtTransform(input, output []float32) {
	Transform32(input, output, hwy.Sqrt_AVX2_F32x8, math.Sqrt32Scalar)
}

// SqrtTransform64 applies sqrt(x) to each float64 element with zero allocations.
// Note: Sqrt is a core op (hardware instruction), not a transcendental.
func SqrtTransform64(input, output []float64) {
	Transform64(input, output, hwy.Sqrt_AVX2_F64x4, math.Sqrt64Scalar)
}
