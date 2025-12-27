//go:build amd64 && goexperiment.simd

package math

import (
	"simd/archsimd"

	"github.com/ajroetker/go-highway/hwy"
)

// Constants for hyperbolic functions
var (
	hyp32_half = archsimd.BroadcastFloat32x8(0.5)
	hyp64_half = archsimd.BroadcastFloat64x4(0.5)
)

// Sinh_AVX2_F32x8 computes sinh(x) for a single Float32x8 vector.
//
// Uses the formula: sinh(x) = (e^x - e^(-x)) / 2
func Sinh_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Compute e^x
	expX := Exp_AVX2_F32x8(x)

	// Compute e^(-x)
	negX := archsimd.BroadcastFloat32x8(0.0).Sub(x)
	expNegX := Exp_AVX2_F32x8(negX)

	// sinh(x) = (e^x - e^(-x)) / 2
	diff := expX.Sub(expNegX)
	return diff.Mul(hyp32_half)
}

// Sinh_AVX2_F64x4 computes sinh(x) for a single Float64x4 vector.
func Sinh_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	expX := Exp_AVX2_F64x4(x)
	negX := archsimd.BroadcastFloat64x4(0.0).Sub(x)
	expNegX := Exp_AVX2_F64x4(negX)
	diff := expX.Sub(expNegX)
	return diff.Mul(hyp64_half)
}

// Cosh_AVX2_F32x8 computes cosh(x) for a single Float32x8 vector.
//
// Uses the formula: cosh(x) = (e^x + e^(-x)) / 2
func Cosh_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Compute e^x
	expX := Exp_AVX2_F32x8(x)

	// Compute e^(-x)
	negX := archsimd.BroadcastFloat32x8(0.0).Sub(x)
	expNegX := Exp_AVX2_F32x8(negX)

	// cosh(x) = (e^x + e^(-x)) / 2
	sum := expX.Add(expNegX)
	return sum.Mul(hyp32_half)
}

// Cosh_AVX2_F64x4 computes cosh(x) for a single Float64x4 vector.
func Cosh_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	expX := Exp_AVX2_F64x4(x)
	negX := archsimd.BroadcastFloat64x4(0.0).Sub(x)
	expNegX := Exp_AVX2_F64x4(negX)
	sum := expX.Add(expNegX)
	return sum.Mul(hyp64_half)
}

// Asinh_AVX2_F32x8 computes asinh(x) for a single Float32x8 vector.
//
// Uses the formula: asinh(x) = ln(x + sqrt(x² + 1))
func Asinh_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	one := archsimd.BroadcastFloat32x8(1.0)
	xSquared := x.Mul(x)
	sqrtArg := xSquared.Add(one)
	sqrtVal := hwy.Sqrt_AVX2_F32x8(sqrtArg)
	arg := x.Add(sqrtVal)
	return Log_AVX2_F32x8(arg)
}

// Asinh_AVX2_F64x4 computes asinh(x) for a single Float64x4 vector.
func Asinh_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	one := archsimd.BroadcastFloat64x4(1.0)
	xSquared := x.Mul(x)
	sqrtArg := xSquared.Add(one)
	sqrtVal := hwy.Sqrt_AVX2_F64x4(sqrtArg)
	arg := x.Add(sqrtVal)
	return Log_AVX2_F64x4(arg)
}

// Acosh_AVX2_F32x8 computes acosh(x) for a single Float32x8 vector.
//
// Uses the formula: acosh(x) = ln(x + sqrt(x² - 1))
// Note: Result is NaN for x < 1
func Acosh_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	one := archsimd.BroadcastFloat32x8(1.0)
	xSquared := x.Mul(x)
	sqrtArg := xSquared.Sub(one)
	sqrtVal := hwy.Sqrt_AVX2_F32x8(sqrtArg)
	arg := x.Add(sqrtVal)
	return Log_AVX2_F32x8(arg)
}

// Acosh_AVX2_F64x4 computes acosh(x) for a single Float64x4 vector.
func Acosh_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	one := archsimd.BroadcastFloat64x4(1.0)
	xSquared := x.Mul(x)
	sqrtArg := xSquared.Sub(one)
	sqrtVal := hwy.Sqrt_AVX2_F64x4(sqrtArg)
	arg := x.Add(sqrtVal)
	return Log_AVX2_F64x4(arg)
}

// Atanh_AVX2_F32x8 computes atanh(x) for a single Float32x8 vector.
//
// Uses the formula: atanh(x) = 0.5 * ln((1 + x) / (1 - x))
// Note: Result is NaN for |x| >= 1
func Atanh_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	one := archsimd.BroadcastFloat32x8(1.0)
	numerator := one.Add(x)
	denominator := one.Sub(x)
	ratio := numerator.Div(denominator)
	logRatio := Log_AVX2_F32x8(ratio)
	return logRatio.Mul(hyp32_half)
}

// Atanh_AVX2_F64x4 computes atanh(x) for a single Float64x4 vector.
func Atanh_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	one := archsimd.BroadcastFloat64x4(1.0)
	numerator := one.Add(x)
	denominator := one.Sub(x)
	ratio := numerator.Div(denominator)
	logRatio := Log_AVX2_F64x4(ratio)
	return logRatio.Mul(hyp64_half)
}
