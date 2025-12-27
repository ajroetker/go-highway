package math

import stdmath "math"

// Scalar helper functions for single-element operations.
// These are useful for processing tail elements when SIMD vector lengths
// don't evenly divide the input size.

// Exp32Scalar computes e^x for a single float32.
func Exp32Scalar(x float32) float32 { return float32(stdmath.Exp(float64(x))) }

// Exp64Scalar computes e^x for a single float64.
func Exp64Scalar(x float64) float64 { return stdmath.Exp(x) }

// Log32Scalar computes ln(x) for a single float32.
func Log32Scalar(x float32) float32 { return float32(stdmath.Log(float64(x))) }

// Log64Scalar computes ln(x) for a single float64.
func Log64Scalar(x float64) float64 { return stdmath.Log(x) }

// Log2_32Scalar computes log₂(x) for a single float32.
func Log2_32Scalar(x float32) float32 { return float32(stdmath.Log2(float64(x))) }

// Log2_64Scalar computes log₂(x) for a single float64.
func Log2_64Scalar(x float64) float64 { return stdmath.Log2(x) }

// Log10_32Scalar computes log₁₀(x) for a single float32.
func Log10_32Scalar(x float32) float32 { return float32(stdmath.Log10(float64(x))) }

// Log10_64Scalar computes log₁₀(x) for a single float64.
func Log10_64Scalar(x float64) float64 { return stdmath.Log10(x) }

// Exp2_32Scalar computes 2^x for a single float32.
func Exp2_32Scalar(x float32) float32 { return float32(stdmath.Exp2(float64(x))) }

// Exp2_64Scalar computes 2^x for a single float64.
func Exp2_64Scalar(x float64) float64 { return stdmath.Exp2(x) }

// Sin32Scalar computes sin(x) for a single float32.
func Sin32Scalar(x float32) float32 { return float32(stdmath.Sin(float64(x))) }

// Sin64Scalar computes sin(x) for a single float64.
func Sin64Scalar(x float64) float64 { return stdmath.Sin(x) }

// Cos32Scalar computes cos(x) for a single float32.
func Cos32Scalar(x float32) float32 { return float32(stdmath.Cos(float64(x))) }

// Cos64Scalar computes cos(x) for a single float64.
func Cos64Scalar(x float64) float64 { return stdmath.Cos(x) }

// Tanh32Scalar computes tanh(x) for a single float32.
func Tanh32Scalar(x float32) float32 { return float32(stdmath.Tanh(float64(x))) }

// Tanh64Scalar computes tanh(x) for a single float64.
func Tanh64Scalar(x float64) float64 { return stdmath.Tanh(x) }

// Sinh32Scalar computes sinh(x) for a single float32.
func Sinh32Scalar(x float32) float32 { return float32(stdmath.Sinh(float64(x))) }

// Sinh64Scalar computes sinh(x) for a single float64.
func Sinh64Scalar(x float64) float64 { return stdmath.Sinh(x) }

// Cosh32Scalar computes cosh(x) for a single float32.
func Cosh32Scalar(x float32) float32 { return float32(stdmath.Cosh(float64(x))) }

// Cosh64Scalar computes cosh(x) for a single float64.
func Cosh64Scalar(x float64) float64 { return stdmath.Cosh(x) }

// Sqrt32Scalar computes sqrt(x) for a single float32.
func Sqrt32Scalar(x float32) float32 { return float32(stdmath.Sqrt(float64(x))) }

// Sqrt64Scalar computes sqrt(x) for a single float64.
func Sqrt64Scalar(x float64) float64 { return stdmath.Sqrt(x) }

// Sigmoid32Scalar computes sigmoid(x) = 1/(1+exp(-x)) for a single float32.
func Sigmoid32Scalar(x float32) float32 { return float32(1.0 / (1.0 + stdmath.Exp(-float64(x)))) }

// Sigmoid64Scalar computes sigmoid(x) = 1/(1+exp(-x)) for a single float64.
func Sigmoid64Scalar(x float64) float64 { return 1.0 / (1.0 + stdmath.Exp(-x)) }

// Erf32Scalar computes erf(x) for a single float32.
func Erf32Scalar(x float32) float32 { return float32(stdmath.Erf(float64(x))) }

// Erf64Scalar computes erf(x) for a single float64.
func Erf64Scalar(x float64) float64 { return stdmath.Erf(x) }
