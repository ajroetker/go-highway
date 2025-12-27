//go:build amd64 && goexperiment.simd

package math

import (
	"simd/archsimd"

	"github.com/ajroetker/go-highway/hwy"
)

// Constants for hyperbolic functions
var (
	hyp512_32_half archsimd.Float32x16
	hyp512_64_half archsimd.Float64x8
	hyp512_32_zero archsimd.Float32x16
	hyp512_64_zero archsimd.Float64x8
	hyp512_32_one  archsimd.Float32x16
	hyp512_64_one  archsimd.Float64x8
)

// Sinh_AVX512_F32x16 computes sinh(x) for a single Float32x16 vector.
//
// Uses the formula: sinh(x) = (e^x - e^(-x)) / 2
func Sinh_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	// Lazy initialization
	hyp512_32_half = archsimd.BroadcastFloat32x16(0.5)
	hyp512_32_zero = archsimd.BroadcastFloat32x16(0.0)

	// Compute e^x
	expX := Exp_AVX512_F32x16(x)

	// Compute e^(-x)
	negX := hyp512_32_zero.Sub(x)
	expNegX := Exp_AVX512_F32x16(negX)

	// sinh(x) = (e^x - e^(-x)) / 2
	diff := expX.Sub(expNegX)
	return diff.Mul(hyp512_32_half)
}

// Sinh_AVX512_F64x8 computes sinh(x) for a single Float64x8 vector.
func Sinh_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	// Lazy initialization
	hyp512_64_half = archsimd.BroadcastFloat64x8(0.5)
	hyp512_64_zero = archsimd.BroadcastFloat64x8(0.0)

	expX := Exp_AVX512_F64x8(x)
	negX := hyp512_64_zero.Sub(x)
	expNegX := Exp_AVX512_F64x8(negX)
	diff := expX.Sub(expNegX)
	return diff.Mul(hyp512_64_half)
}

// Cosh_AVX512_F32x16 computes cosh(x) for a single Float32x16 vector.
//
// Uses the formula: cosh(x) = (e^x + e^(-x)) / 2
func Cosh_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	// Lazy initialization
	hyp512_32_half = archsimd.BroadcastFloat32x16(0.5)
	hyp512_32_zero = archsimd.BroadcastFloat32x16(0.0)

	// Compute e^x
	expX := Exp_AVX512_F32x16(x)

	// Compute e^(-x)
	negX := hyp512_32_zero.Sub(x)
	expNegX := Exp_AVX512_F32x16(negX)

	// cosh(x) = (e^x + e^(-x)) / 2
	sum := expX.Add(expNegX)
	return sum.Mul(hyp512_32_half)
}

// Cosh_AVX512_F64x8 computes cosh(x) for a single Float64x8 vector.
func Cosh_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	// Lazy initialization
	hyp512_64_half = archsimd.BroadcastFloat64x8(0.5)
	hyp512_64_zero = archsimd.BroadcastFloat64x8(0.0)

	expX := Exp_AVX512_F64x8(x)
	negX := hyp512_64_zero.Sub(x)
	expNegX := Exp_AVX512_F64x8(negX)
	sum := expX.Add(expNegX)
	return sum.Mul(hyp512_64_half)
}

// Asinh_AVX512_F32x16 computes asinh(x) for a single Float32x16 vector.
//
// Uses the formula: asinh(x) = ln(x + sqrt(x² + 1))
func Asinh_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	// Lazy initialization
	hyp512_32_one = archsimd.BroadcastFloat32x16(1.0)

	xSquared := x.Mul(x)
	sqrtArg := xSquared.Add(hyp512_32_one)
	sqrtVal := hwy.Sqrt_AVX512_F32x16(sqrtArg)
	arg := x.Add(sqrtVal)
	return Log_AVX512_F32x16(arg)
}

// Asinh_AVX512_F64x8 computes asinh(x) for a single Float64x8 vector.
func Asinh_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	// Lazy initialization
	hyp512_64_one = archsimd.BroadcastFloat64x8(1.0)

	xSquared := x.Mul(x)
	sqrtArg := xSquared.Add(hyp512_64_one)
	sqrtVal := hwy.Sqrt_AVX512_F64x8(sqrtArg)
	arg := x.Add(sqrtVal)
	return Log_AVX512_F64x8(arg)
}

// Acosh_AVX512_F32x16 computes acosh(x) for a single Float32x16 vector.
//
// Uses the formula: acosh(x) = ln(x + sqrt(x² - 1))
// Note: Result is NaN for x < 1
func Acosh_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	// Lazy initialization
	hyp512_32_one = archsimd.BroadcastFloat32x16(1.0)

	xSquared := x.Mul(x)
	sqrtArg := xSquared.Sub(hyp512_32_one)
	sqrtVal := hwy.Sqrt_AVX512_F32x16(sqrtArg)
	arg := x.Add(sqrtVal)
	return Log_AVX512_F32x16(arg)
}

// Acosh_AVX512_F64x8 computes acosh(x) for a single Float64x8 vector.
func Acosh_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	// Lazy initialization
	hyp512_64_one = archsimd.BroadcastFloat64x8(1.0)

	xSquared := x.Mul(x)
	sqrtArg := xSquared.Sub(hyp512_64_one)
	sqrtVal := hwy.Sqrt_AVX512_F64x8(sqrtArg)
	arg := x.Add(sqrtVal)
	return Log_AVX512_F64x8(arg)
}

// Atanh_AVX512_F32x16 computes atanh(x) for a single Float32x16 vector.
//
// Uses the formula: atanh(x) = 0.5 * ln((1 + x) / (1 - x))
// Note: Result is NaN for |x| >= 1
func Atanh_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	// Lazy initialization
	hyp512_32_one = archsimd.BroadcastFloat32x16(1.0)
	hyp512_32_half = archsimd.BroadcastFloat32x16(0.5)

	numerator := hyp512_32_one.Add(x)
	denominator := hyp512_32_one.Sub(x)
	ratio := numerator.Div(denominator)
	logRatio := Log_AVX512_F32x16(ratio)
	return logRatio.Mul(hyp512_32_half)
}

// Atanh_AVX512_F64x8 computes atanh(x) for a single Float64x8 vector.
func Atanh_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	// Lazy initialization
	hyp512_64_one = archsimd.BroadcastFloat64x8(1.0)
	hyp512_64_half = archsimd.BroadcastFloat64x8(0.5)

	numerator := hyp512_64_one.Add(x)
	denominator := hyp512_64_one.Sub(x)
	ratio := numerator.Div(denominator)
	logRatio := Log_AVX512_F64x8(ratio)
	return logRatio.Mul(hyp512_64_half)
}
