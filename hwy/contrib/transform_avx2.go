//go:build amd64 && goexperiment.simd

package contrib

import (
	"math"
	"simd/archsimd"

	"github.com/ajroetker/go-highway/hwy"
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
func exp32Scalar(x float32) float32   { return float32(math.Exp(float64(x))) }
func exp64Scalar(x float64) float64   { return math.Exp(x) }
func log32Scalar(x float32) float32   { return float32(math.Log(float64(x))) }
func log64Scalar(x float64) float64   { return math.Log(x) }
func sin32Scalar(x float32) float32   { return float32(math.Sin(float64(x))) }
func sin64Scalar(x float64) float64   { return math.Sin(x) }
func cos32Scalar(x float32) float32   { return float32(math.Cos(float64(x))) }
func cos64Scalar(x float64) float64   { return math.Cos(x) }
func tanh32Scalar(x float32) float32  { return float32(math.Tanh(float64(x))) }
func tanh64Scalar(x float64) float64  { return math.Tanh(x) }
func sigmoid32Scalar(x float32) float32 { return float32(1.0 / (1.0 + math.Exp(-float64(x)))) }
func sigmoid64Scalar(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
func erf32Scalar(x float32) float32   { return float32(math.Erf(float64(x))) }
func erf64Scalar(x float64) float64   { return math.Erf(x) }

// ExpTransform applies exp(x) to each element with zero allocations.
// Caller must ensure len(output) >= len(input).
func ExpTransform(input, output []float32) {
	Transform32(input, output, Exp_AVX2_F32x8, exp32Scalar)
}

// ExpTransform64 applies exp(x) to each float64 element with zero allocations.
func ExpTransform64(input, output []float64) {
	Transform64(input, output, Exp_AVX2_F64x4, exp64Scalar)
}

// LogTransform applies ln(x) to each element with zero allocations.
func LogTransform(input, output []float32) {
	Transform32(input, output, Log_AVX2_F32x8, log32Scalar)
}

// LogTransform64 applies ln(x) to each float64 element with zero allocations.
func LogTransform64(input, output []float64) {
	Transform64(input, output, Log_AVX2_F64x4, log64Scalar)
}

// SinTransform applies sin(x) to each element with zero allocations.
func SinTransform(input, output []float32) {
	Transform32(input, output, Sin_AVX2_F32x8, sin32Scalar)
}

// SinTransform64 applies sin(x) to each float64 element with zero allocations.
func SinTransform64(input, output []float64) {
	Transform64(input, output, Sin_AVX2_F64x4, sin64Scalar)
}

// CosTransform applies cos(x) to each element with zero allocations.
func CosTransform(input, output []float32) {
	Transform32(input, output, Cos_AVX2_F32x8, cos32Scalar)
}

// CosTransform64 applies cos(x) to each float64 element with zero allocations.
func CosTransform64(input, output []float64) {
	Transform64(input, output, Cos_AVX2_F64x4, cos64Scalar)
}

// TanhTransform applies tanh(x) to each element with zero allocations.
func TanhTransform(input, output []float32) {
	Transform32(input, output, Tanh_AVX2_F32x8, tanh32Scalar)
}

// TanhTransform64 applies tanh(x) to each float64 element with zero allocations.
func TanhTransform64(input, output []float64) {
	Transform64(input, output, Tanh_AVX2_F64x4, tanh64Scalar)
}

// SigmoidTransform applies sigmoid(x) = 1/(1+exp(-x)) to each element with zero allocations.
func SigmoidTransform(input, output []float32) {
	Transform32(input, output, Sigmoid_AVX2_F32x8, sigmoid32Scalar)
}

// SigmoidTransform64 applies sigmoid(x) to each float64 element with zero allocations.
func SigmoidTransform64(input, output []float64) {
	Transform64(input, output, Sigmoid_AVX2_F64x4, sigmoid64Scalar)
}

// ErfTransform applies erf(x) to each element with zero allocations.
func ErfTransform(input, output []float32) {
	Transform32(input, output, Erf_AVX2_F32x8, erf32Scalar)
}

// ErfTransform64 applies erf(x) to each float64 element with zero allocations.
func ErfTransform64(input, output []float64) {
	Transform64(input, output, Erf_AVX2_F64x4, erf64Scalar)
}
