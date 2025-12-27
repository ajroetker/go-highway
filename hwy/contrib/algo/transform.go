//go:build !amd64 || !goexperiment.simd

package algo

import "math"

// This file provides scalar fallback implementations for non-SIMD builds.
// The AVX2 implementations in transform_avx2.go override these on amd64.

// Function types for generic Transform operations.
// These allow users to pass custom operations to Transform32/Transform64.
// On non-SIMD builds, the SIMD function is ignored and only scalar is used.
type (
	// VecFunc32 is a placeholder for SIMD operations (ignored on non-SIMD builds).
	VecFunc32 func(interface{}) interface{}

	// VecFunc64 is a placeholder for SIMD operations (ignored on non-SIMD builds).
	VecFunc64 func(interface{}) interface{}

	// ScalarFunc32 is a scalar operation on a single float32.
	ScalarFunc32 func(float32) float32

	// ScalarFunc64 is a scalar operation on a single float64.
	ScalarFunc64 func(float64) float64
)

// Transform32 applies a scalar operation to each element of input, storing results in output.
// On non-SIMD builds, the simd function is ignored.
//
// Example usage:
//
//	Transform32(input, output, nil, func(x float32) float32 { return x*x + x })
func Transform32(input, output []float32, _ VecFunc32, scalar ScalarFunc32) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = scalar(input[i])
	}
}

// Transform64 applies a scalar operation to each element of input, storing results in output.
// On non-SIMD builds, the simd function is ignored.
func Transform64(input, output []float64, _ VecFunc64, scalar ScalarFunc64) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = scalar(input[i])
	}
}

// Scalar helper functions
func exp32Scalar(x float32) float32       { return float32(math.Exp(float64(x))) }
func exp64Scalar(x float64) float64       { return math.Exp(x) }
func log32Scalar(x float32) float32       { return float32(math.Log(float64(x))) }
func log64Scalar(x float64) float64       { return math.Log(x) }
func log2_32Scalar(x float32) float32     { return float32(math.Log2(float64(x))) }
func log2_64Scalar(x float64) float64     { return math.Log2(x) }
func log10_32Scalar(x float32) float32    { return float32(math.Log10(float64(x))) }
func log10_64Scalar(x float64) float64    { return math.Log10(x) }
func exp2_32Scalar(x float32) float32     { return float32(math.Exp2(float64(x))) }
func exp2_64Scalar(x float64) float64     { return math.Exp2(x) }
func sin32Scalar(x float32) float32       { return float32(math.Sin(float64(x))) }
func sin64Scalar(x float64) float64       { return math.Sin(x) }
func cos32Scalar(x float32) float32       { return float32(math.Cos(float64(x))) }
func cos64Scalar(x float64) float64       { return math.Cos(x) }
func tanh32Scalar(x float32) float32      { return float32(math.Tanh(float64(x))) }
func tanh64Scalar(x float64) float64      { return math.Tanh(x) }
func sinh32Scalar(x float32) float32      { return float32(math.Sinh(float64(x))) }
func sinh64Scalar(x float64) float64      { return math.Sinh(x) }
func cosh32Scalar(x float32) float32      { return float32(math.Cosh(float64(x))) }
func cosh64Scalar(x float64) float64      { return math.Cosh(x) }
func sqrt32Scalar(x float32) float32      { return float32(math.Sqrt(float64(x))) }
func sqrt64Scalar(x float64) float64      { return math.Sqrt(x) }
func sigmoid32Scalar(x float32) float32   { return float32(1.0 / (1.0 + math.Exp(-float64(x)))) }
func sigmoid64Scalar(x float64) float64   { return 1.0 / (1.0 + math.Exp(-x)) }
func erf32Scalar(x float32) float32       { return float32(math.Erf(float64(x))) }
func erf64Scalar(x float64) float64       { return math.Erf(x) }

// ExpTransform applies exp(x) to each element.
// Caller must ensure len(output) >= len(input).
func ExpTransform(input, output []float32) {
	Transform32(input, output, nil, exp32Scalar)
}

// ExpTransform64 applies exp(x) to each float64 element.
func ExpTransform64(input, output []float64) {
	Transform64(input, output, nil, exp64Scalar)
}

// LogTransform applies ln(x) to each element.
func LogTransform(input, output []float32) {
	Transform32(input, output, nil, log32Scalar)
}

// LogTransform64 applies ln(x) to each float64 element.
func LogTransform64(input, output []float64) {
	Transform64(input, output, nil, log64Scalar)
}

// SinTransform applies sin(x) to each element.
func SinTransform(input, output []float32) {
	Transform32(input, output, nil, sin32Scalar)
}

// SinTransform64 applies sin(x) to each float64 element.
func SinTransform64(input, output []float64) {
	Transform64(input, output, nil, sin64Scalar)
}

// CosTransform applies cos(x) to each element.
func CosTransform(input, output []float32) {
	Transform32(input, output, nil, cos32Scalar)
}

// CosTransform64 applies cos(x) to each float64 element.
func CosTransform64(input, output []float64) {
	Transform64(input, output, nil, cos64Scalar)
}

// TanhTransform applies tanh(x) to each element.
func TanhTransform(input, output []float32) {
	Transform32(input, output, nil, tanh32Scalar)
}

// TanhTransform64 applies tanh(x) to each float64 element.
func TanhTransform64(input, output []float64) {
	Transform64(input, output, nil, tanh64Scalar)
}

// SigmoidTransform applies sigmoid(x) = 1/(1+exp(-x)) to each element.
func SigmoidTransform(input, output []float32) {
	Transform32(input, output, nil, sigmoid32Scalar)
}

// SigmoidTransform64 applies sigmoid(x) to each float64 element.
func SigmoidTransform64(input, output []float64) {
	Transform64(input, output, nil, sigmoid64Scalar)
}

// ErfTransform applies erf(x) to each element.
func ErfTransform(input, output []float32) {
	Transform32(input, output, nil, erf32Scalar)
}

// ErfTransform64 applies erf(x) to each float64 element.
func ErfTransform64(input, output []float64) {
	Transform64(input, output, nil, erf64Scalar)
}

// Log2Transform applies log₂(x) to each element.
func Log2Transform(input, output []float32) {
	Transform32(input, output, nil, log2_32Scalar)
}

// Log2Transform64 applies log₂(x) to each float64 element.
func Log2Transform64(input, output []float64) {
	Transform64(input, output, nil, log2_64Scalar)
}

// Log10Transform applies log₁₀(x) to each element.
func Log10Transform(input, output []float32) {
	Transform32(input, output, nil, log10_32Scalar)
}

// Log10Transform64 applies log₁₀(x) to each float64 element.
func Log10Transform64(input, output []float64) {
	Transform64(input, output, nil, log10_64Scalar)
}

// Exp2Transform applies 2^x to each element.
func Exp2Transform(input, output []float32) {
	Transform32(input, output, nil, exp2_32Scalar)
}

// Exp2Transform64 applies 2^x to each float64 element.
func Exp2Transform64(input, output []float64) {
	Transform64(input, output, nil, exp2_64Scalar)
}

// SinhTransform applies sinh(x) to each element.
func SinhTransform(input, output []float32) {
	Transform32(input, output, nil, sinh32Scalar)
}

// SinhTransform64 applies sinh(x) to each float64 element.
func SinhTransform64(input, output []float64) {
	Transform64(input, output, nil, sinh64Scalar)
}

// CoshTransform applies cosh(x) to each element.
func CoshTransform(input, output []float32) {
	Transform32(input, output, nil, cosh32Scalar)
}

// CoshTransform64 applies cosh(x) to each float64 element.
func CoshTransform64(input, output []float64) {
	Transform64(input, output, nil, cosh64Scalar)
}

// SqrtTransform applies sqrt(x) to each element.
func SqrtTransform(input, output []float32) {
	Transform32(input, output, nil, sqrt32Scalar)
}

// SqrtTransform64 applies sqrt(x) to each float64 element.
func SqrtTransform64(input, output []float64) {
	Transform64(input, output, nil, sqrt64Scalar)
}
