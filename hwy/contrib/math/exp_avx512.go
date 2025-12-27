//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
	"sync"
)

// Lazy initialization for AVX-512 constants to avoid executing AVX-512
// instructions at package load time on machines without AVX-512 support.

var exp512Init sync.Once

// AVX-512 vectorized constants for exp32
var (
	exp512_32_ln2Hi     archsimd.Float32x16
	exp512_32_ln2Lo     archsimd.Float32x16
	exp512_32_invLn2    archsimd.Float32x16
	exp512_32_one       archsimd.Float32x16
	exp512_32_zero      archsimd.Float32x16
	exp512_32_inf       archsimd.Float32x16
	exp512_32_overflow  archsimd.Float32x16
	exp512_32_underflow archsimd.Float32x16
	exp512_32_c1        archsimd.Float32x16
	exp512_32_c2        archsimd.Float32x16
	exp512_32_c3        archsimd.Float32x16
	exp512_32_c4        archsimd.Float32x16
	exp512_32_c5        archsimd.Float32x16
	exp512_32_c6        archsimd.Float32x16
	exp512_32_bias      archsimd.Int32x16
)

// AVX-512 vectorized constants for exp64
var (
	exp512_64_ln2Hi     archsimd.Float64x8
	exp512_64_ln2Lo     archsimd.Float64x8
	exp512_64_invLn2    archsimd.Float64x8
	exp512_64_one       archsimd.Float64x8
	exp512_64_zero      archsimd.Float64x8
	exp512_64_inf       archsimd.Float64x8
	exp512_64_overflow  archsimd.Float64x8
	exp512_64_underflow archsimd.Float64x8
	exp512_64_c1        archsimd.Float64x8
	exp512_64_c2        archsimd.Float64x8
	exp512_64_c3        archsimd.Float64x8
	exp512_64_c4        archsimd.Float64x8
	exp512_64_c5        archsimd.Float64x8
	exp512_64_c6        archsimd.Float64x8
	exp512_64_c7        archsimd.Float64x8
	exp512_64_c8        archsimd.Float64x8
	exp512_64_c9        archsimd.Float64x8
	exp512_64_c10       archsimd.Float64x8
	exp512_64_bias      archsimd.Int64x8
)

func initExp512Constants() {
	// Float32 constants
	exp512_32_ln2Hi = archsimd.BroadcastFloat32x16(0.693359375)
	exp512_32_ln2Lo = archsimd.BroadcastFloat32x16(-2.12194440e-4)
	exp512_32_invLn2 = archsimd.BroadcastFloat32x16(1.44269504088896341)
	exp512_32_one = archsimd.BroadcastFloat32x16(1.0)
	exp512_32_zero = archsimd.BroadcastFloat32x16(0.0)
	exp512_32_inf = archsimd.BroadcastFloat32x16(float32PositiveInf())
	exp512_32_overflow = archsimd.BroadcastFloat32x16(88.72283905206835)
	exp512_32_underflow = archsimd.BroadcastFloat32x16(-87.33654475055310)
	exp512_32_c1 = archsimd.BroadcastFloat32x16(1.0)
	exp512_32_c2 = archsimd.BroadcastFloat32x16(0.5)
	exp512_32_c3 = archsimd.BroadcastFloat32x16(0.16666666666666666)
	exp512_32_c4 = archsimd.BroadcastFloat32x16(0.041666666666666664)
	exp512_32_c5 = archsimd.BroadcastFloat32x16(0.008333333333333333)
	exp512_32_c6 = archsimd.BroadcastFloat32x16(0.001388888888888889)
	exp512_32_bias = archsimd.BroadcastInt32x16(127)

	// Float64 constants
	exp512_64_ln2Hi = archsimd.BroadcastFloat64x8(0.6931471803691238)
	exp512_64_ln2Lo = archsimd.BroadcastFloat64x8(1.9082149292705877e-10)
	exp512_64_invLn2 = archsimd.BroadcastFloat64x8(1.4426950408889634)
	exp512_64_one = archsimd.BroadcastFloat64x8(1.0)
	exp512_64_zero = archsimd.BroadcastFloat64x8(0.0)
	exp512_64_inf = archsimd.BroadcastFloat64x8(math.Inf(1))
	exp512_64_overflow = archsimd.BroadcastFloat64x8(709.782712893384)
	exp512_64_underflow = archsimd.BroadcastFloat64x8(-708.3964185322641)
	exp512_64_c1 = archsimd.BroadcastFloat64x8(1.0)
	exp512_64_c2 = archsimd.BroadcastFloat64x8(0.5)
	exp512_64_c3 = archsimd.BroadcastFloat64x8(0.16666666666666666)
	exp512_64_c4 = archsimd.BroadcastFloat64x8(0.041666666666666664)
	exp512_64_c5 = archsimd.BroadcastFloat64x8(0.008333333333333333)
	exp512_64_c6 = archsimd.BroadcastFloat64x8(0.001388888888888889)
	exp512_64_c7 = archsimd.BroadcastFloat64x8(0.0001984126984126984)
	exp512_64_c8 = archsimd.BroadcastFloat64x8(2.48015873015873e-05)
	exp512_64_c9 = archsimd.BroadcastFloat64x8(2.7557319223985893e-06)
	exp512_64_c10 = archsimd.BroadcastFloat64x8(2.755731922398589e-07)
	exp512_64_bias = archsimd.BroadcastInt64x8(1023)
}

// Exp_AVX512_F32x16 computes e^x for a single Float32x16 vector.
//
// Algorithm:
// 1. Range reduction: x = k*ln(2) + r, where |r| <= ln(2)/2
// 2. Polynomial approximation: e^r ≈ 1 + r + r²/2! + r³/3! + ...
// 3. Reconstruction: e^x = 2^k * e^r
func Exp_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	exp512Init.Do(initExp512Constants)

	// Create masks for special cases
	overflowMask := x.Greater(exp512_32_overflow)
	underflowMask := x.Less(exp512_32_underflow)

	// Range reduction: k = round(x / ln(2))
	// k = round(x * (1/ln(2)))
	kFloat := x.Mul(exp512_32_invLn2).RoundToEvenScaled(0)

	// r = x - k * ln(2) using high/low split for precision
	// r = x - k*ln2Hi - k*ln2Lo
	r := x.Sub(kFloat.Mul(exp512_32_ln2Hi))
	r = r.Sub(kFloat.Mul(exp512_32_ln2Lo))

	// Polynomial approximation using Horner's method
	// p = 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
	// Horner's method from inside out
	p := exp512_32_c6.MulAdd(r, exp512_32_c5) // c6*r + c5
	p = p.MulAdd(r, exp512_32_c4)             // p*r + c4
	p = p.MulAdd(r, exp512_32_c3)             // p*r + c3
	p = p.MulAdd(r, exp512_32_c2)             // p*r + c2
	p = p.MulAdd(r, exp512_32_c1)             // p*r + c1
	p = p.MulAdd(r, exp512_32_one)            // p*r + 1

	// Scale by 2^k using IEEE 754 bit manipulation
	// float32 = sign(1) | exponent(8) | mantissa(23)
	// 2^k is represented as: exponent = k + 127, mantissa = 0
	// So we create (k + 127) << 23 and reinterpret as float
	kInt := kFloat.ConvertToInt32()
	expBits := kInt.Add(exp512_32_bias).ShiftAllLeft(23)
	scale := expBits.AsFloat32x16()

	result := p.Mul(scale)

	// Handle overflow: return +Inf where x > threshold
	// Note: Merge semantics are inverted - a.Merge(b, mask) returns a when TRUE, b when FALSE
	result = exp512_32_inf.Merge(result, overflowMask)

	// Handle underflow: return 0 where x < threshold
	result = exp512_32_zero.Merge(result, underflowMask)

	return result
}

// Exp_AVX512_F64x8 computes e^x for a single Float64x8 vector.
func Exp_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	exp512Init.Do(initExp512Constants)

	overflowMask := x.Greater(exp512_64_overflow)
	underflowMask := x.Less(exp512_64_underflow)

	// Range reduction
	kFloat := x.Mul(exp512_64_invLn2).RoundToEvenScaled(0)
	r := x.Sub(kFloat.Mul(exp512_64_ln2Hi))
	r = r.Sub(kFloat.Mul(exp512_64_ln2Lo))

	// Polynomial approximation (degree 10 for float64)
	p := exp512_64_c10.MulAdd(r, exp512_64_c9)
	p = p.MulAdd(r, exp512_64_c8)
	p = p.MulAdd(r, exp512_64_c7)
	p = p.MulAdd(r, exp512_64_c6)
	p = p.MulAdd(r, exp512_64_c5)
	p = p.MulAdd(r, exp512_64_c4)
	p = p.MulAdd(r, exp512_64_c3)
	p = p.MulAdd(r, exp512_64_c2)
	p = p.MulAdd(r, exp512_64_c1)
	p = p.MulAdd(r, exp512_64_one)

	// Scale by 2^k
	kInt := kFloat.ConvertToInt64()
	expBits := kInt.Add(exp512_64_bias).ShiftAllLeft(52)
	scale := expBits.AsFloat64x8()

	result := p.Mul(scale)

	// Handle special cases (Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	result = exp512_64_inf.Merge(result, overflowMask)
	result = exp512_64_zero.Merge(result, underflowMask)

	return result
}

// Constants for log/exp base conversions (AVX-512)
var (
	// log2(e) = 1.4426950408889634
	log2e512_32 archsimd.Float32x16
	log2e512_64 archsimd.Float64x8

	// log10(e) = 0.4342944819032518
	log10e512_32 archsimd.Float32x16
	log10e512_64 archsimd.Float64x8

	// ln(2) = 0.6931471805599453
	ln2_512_32 archsimd.Float32x16
	ln2_512_64 archsimd.Float64x8

	// ln(10) = 2.302585092994046
	ln10_512_32 archsimd.Float32x16
	ln10_512_64 archsimd.Float64x8
)

var expVariants512Init sync.Once

func initExpVariants512Constants() {
	log2e512_32 = archsimd.BroadcastFloat32x16(1.4426950408889634)
	log2e512_64 = archsimd.BroadcastFloat64x8(1.4426950408889634)

	log10e512_32 = archsimd.BroadcastFloat32x16(0.4342944819032518)
	log10e512_64 = archsimd.BroadcastFloat64x8(0.4342944819032518)

	ln2_512_32 = archsimd.BroadcastFloat32x16(0.6931471805599453)
	ln2_512_64 = archsimd.BroadcastFloat64x8(0.6931471805599453)

	ln10_512_32 = archsimd.BroadcastFloat32x16(2.302585092994046)
	ln10_512_64 = archsimd.BroadcastFloat64x8(2.302585092994046)
}

// Exp2_AVX512_F32x16 computes 2^x for a single Float32x16 vector.
//
// Uses the identity: 2^x = e^(x * ln(2))
func Exp2_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	expVariants512Init.Do(initExpVariants512Constants)
	xLn2 := x.Mul(ln2_512_32)
	return Exp_AVX512_F32x16(xLn2)
}

// Exp2_AVX512_F64x8 computes 2^x for a single Float64x8 vector.
func Exp2_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	expVariants512Init.Do(initExpVariants512Constants)
	xLn2 := x.Mul(ln2_512_64)
	return Exp_AVX512_F64x8(xLn2)
}

// Exp10_AVX512_F32x16 computes 10^x for a single Float32x16 vector.
//
// Uses the identity: 10^x = e^(x * ln(10))
func Exp10_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	expVariants512Init.Do(initExpVariants512Constants)
	xLn10 := x.Mul(ln10_512_32)
	return Exp_AVX512_F32x16(xLn10)
}

// Exp10_AVX512_F64x8 computes 10^x for a single Float64x8 vector.
func Exp10_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	expVariants512Init.Do(initExpVariants512Constants)
	xLn10 := x.Mul(ln10_512_64)
	return Exp_AVX512_F64x8(xLn10)
}
