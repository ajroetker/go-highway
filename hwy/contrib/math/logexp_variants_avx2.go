//go:build amd64 && goexperiment.simd

package math

import (
	"simd/archsimd"
)

// Constants for log/exp base conversions
var (
	// log2(e) = 1.4426950408889634
	log2e32 = archsimd.BroadcastFloat32x8(1.4426950408889634)
	log2e64 = archsimd.BroadcastFloat64x4(1.4426950408889634)

	// log10(e) = 0.4342944819032518
	log10e32 = archsimd.BroadcastFloat32x8(0.4342944819032518)
	log10e64 = archsimd.BroadcastFloat64x4(0.4342944819032518)

	// ln(2) = 0.6931471805599453
	ln2_32 = archsimd.BroadcastFloat32x8(0.6931471805599453)
	ln2_64 = archsimd.BroadcastFloat64x4(0.6931471805599453)

	// ln(10) = 2.302585092994046
	ln10_32 = archsimd.BroadcastFloat32x8(2.302585092994046)
	ln10_64 = archsimd.BroadcastFloat64x4(2.302585092994046)
)

// Log2_AVX2_F32x8 computes log₂(x) for a single Float32x8 vector.
//
// Uses the identity: log₂(x) = ln(x) / ln(2) = ln(x) * log₂(e)
func Log2_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	lnX := Log_AVX2_F32x8(x)
	return lnX.Mul(log2e32)
}

// Log2_AVX2_F64x4 computes log₂(x) for a single Float64x4 vector.
func Log2_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	lnX := Log_AVX2_F64x4(x)
	return lnX.Mul(log2e64)
}

// Log10_AVX2_F32x8 computes log₁₀(x) for a single Float32x8 vector.
//
// Uses the identity: log₁₀(x) = ln(x) / ln(10) = ln(x) * log₁₀(e)
func Log10_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	lnX := Log_AVX2_F32x8(x)
	return lnX.Mul(log10e32)
}

// Log10_AVX2_F64x4 computes log₁₀(x) for a single Float64x4 vector.
func Log10_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	lnX := Log_AVX2_F64x4(x)
	return lnX.Mul(log10e64)
}

// Exp2_AVX2_F32x8 computes 2^x for a single Float32x8 vector.
//
// Uses the identity: 2^x = e^(x * ln(2))
func Exp2_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	xLn2 := x.Mul(ln2_32)
	return Exp_AVX2_F32x8(xLn2)
}

// Exp2_AVX2_F64x4 computes 2^x for a single Float64x4 vector.
func Exp2_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	xLn2 := x.Mul(ln2_64)
	return Exp_AVX2_F64x4(xLn2)
}

// Exp10_AVX2_F32x8 computes 10^x for a single Float32x8 vector.
//
// Uses the identity: 10^x = e^(x * ln(10))
func Exp10_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	xLn10 := x.Mul(ln10_32)
	return Exp_AVX2_F32x8(xLn10)
}

// Exp10_AVX2_F64x4 computes 10^x for a single Float64x4 vector.
func Exp10_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	xLn10 := x.Mul(ln10_64)
	return Exp_AVX2_F64x4(xLn10)
}
