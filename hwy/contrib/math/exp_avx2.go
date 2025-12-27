//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
)

// AVX2 vectorized constants for exp32
var (
	// ln(2) split for precision
	exp32_ln2Hi  = archsimd.BroadcastFloat32x8(0.693359375)
	exp32_ln2Lo  = archsimd.BroadcastFloat32x8(-2.12194440e-4)
	exp32_invLn2 = archsimd.BroadcastFloat32x8(1.44269504088896341)
	exp32_one    = archsimd.BroadcastFloat32x8(1.0)
	exp32_zero   = archsimd.BroadcastFloat32x8(0.0)
	exp32_inf    = archsimd.BroadcastFloat32x8(float32PositiveInf())

	// Overflow/underflow thresholds
	exp32_overflow  = archsimd.BroadcastFloat32x8(88.72283905206835)
	exp32_underflow = archsimd.BroadcastFloat32x8(-87.33654475055310)

	// Polynomial coefficients for exp(r) on [-ln(2)/2, ln(2)/2]
	// Using Taylor series: 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5! + r⁶/6!
	exp32_c1 = archsimd.BroadcastFloat32x8(1.0)
	exp32_c2 = archsimd.BroadcastFloat32x8(0.5)
	exp32_c3 = archsimd.BroadcastFloat32x8(0.16666666666666666)
	exp32_c4 = archsimd.BroadcastFloat32x8(0.041666666666666664)
	exp32_c5 = archsimd.BroadcastFloat32x8(0.008333333333333333)
	exp32_c6 = archsimd.BroadcastFloat32x8(0.001388888888888889)

	// Constants for 2^k computation via IEEE 754 bit manipulation
	exp32_bias = archsimd.BroadcastInt32x8(127)
)

func float32PositiveInf() float32 {
	bits := uint32(0x7F800000)
	return math.Float32frombits(bits)
}

// Exp_AVX2_F32x8 computes e^x for a single Float32x8 vector.
//
// Algorithm:
// 1. Range reduction: x = k*ln(2) + r, where |r| <= ln(2)/2
// 2. Polynomial approximation: e^r ≈ 1 + r + r²/2! + r³/3! + ...
// 3. Reconstruction: e^x = 2^k * e^r
func Exp_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Create masks for special cases
	overflowMask := x.Greater(exp32_overflow)
	underflowMask := x.Less(exp32_underflow)

	// Range reduction: k = round(x / ln(2))
	// k = round(x * (1/ln(2)))
	kFloat := x.Mul(exp32_invLn2).RoundToEven()

	// r = x - k * ln(2) using high/low split for precision
	// r = x - k*ln2Hi - k*ln2Lo
	r := x.Sub(kFloat.Mul(exp32_ln2Hi))
	r = r.Sub(kFloat.Mul(exp32_ln2Lo))

	// Polynomial approximation using Horner's method
	// p = 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
	// Horner's method from inside out
	p := exp32_c6.MulAdd(r, exp32_c5) // c6*r + c5
	p = p.MulAdd(r, exp32_c4)          // p*r + c4
	p = p.MulAdd(r, exp32_c3)          // p*r + c3
	p = p.MulAdd(r, exp32_c2)          // p*r + c2
	p = p.MulAdd(r, exp32_c1)          // p*r + c1
	p = p.MulAdd(r, exp32_one)         // p*r + 1

	// Scale by 2^k using IEEE 754 bit manipulation
	// float32 = sign(1) | exponent(8) | mantissa(23)
	// 2^k is represented as: exponent = k + 127, mantissa = 0
	// So we create (k + 127) << 23 and reinterpret as float
	kInt := kFloat.ConvertToInt32()
	expBits := kInt.Add(exp32_bias).ShiftAllLeft(23)
	scale := expBits.AsFloat32x8()

	result := p.Mul(scale)

	// Handle overflow: return +Inf where x > threshold
	// Note: Merge semantics are inverted - a.Merge(b, mask) returns a when TRUE, b when FALSE
	result = exp32_inf.Merge(result, overflowMask)

	// Handle underflow: return 0 where x < threshold
	result = exp32_zero.Merge(result, underflowMask)

	return result
}

// AVX2 vectorized constants for exp64
var (
	exp64_ln2Hi  = archsimd.BroadcastFloat64x4(0.6931471803691238)
	exp64_ln2Lo  = archsimd.BroadcastFloat64x4(1.9082149292705877e-10)
	exp64_invLn2 = archsimd.BroadcastFloat64x4(1.4426950408889634)
	exp64_one    = archsimd.BroadcastFloat64x4(1.0)
	exp64_zero   = archsimd.BroadcastFloat64x4(0.0)
	exp64_inf    = archsimd.BroadcastFloat64x4(stdmath.Inf(1))

	exp64_overflow  = archsimd.BroadcastFloat64x4(709.782712893384)
	exp64_underflow = archsimd.BroadcastFloat64x4(-708.3964185322641)

	// Higher-degree polynomial for float64 accuracy
	exp64_c1  = archsimd.BroadcastFloat64x4(1.0)
	exp64_c2  = archsimd.BroadcastFloat64x4(0.5)
	exp64_c3  = archsimd.BroadcastFloat64x4(0.16666666666666666)
	exp64_c4  = archsimd.BroadcastFloat64x4(0.041666666666666664)
	exp64_c5  = archsimd.BroadcastFloat64x4(0.008333333333333333)
	exp64_c6  = archsimd.BroadcastFloat64x4(0.001388888888888889)
	exp64_c7  = archsimd.BroadcastFloat64x4(0.0001984126984126984)
	exp64_c8  = archsimd.BroadcastFloat64x4(2.48015873015873e-05)
	exp64_c9  = archsimd.BroadcastFloat64x4(2.7557319223985893e-06)
	exp64_c10 = archsimd.BroadcastFloat64x4(2.755731922398589e-07)

	// Magic number constants for 2^k computation (avoids scalar int64 conversion)
	exp64_magic       = archsimd.BroadcastFloat64x4(0x1.0p52)          // 2^52
	exp64_magicOffset = archsimd.BroadcastFloat64x4(1024.0)            // Offset to make k positive
	exp64_magicAdjust = archsimd.BroadcastInt64x4(0x4330000000000000)  // Bits of 2^52
	exp64_magicOne    = archsimd.BroadcastInt64x4(1)                   // For adjustment
)

// Exp_AVX2_F64x4 computes e^x for a single Float64x4 vector.
func Exp_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	overflowMask := x.Greater(exp64_overflow)
	underflowMask := x.Less(exp64_underflow)

	// Range reduction
	kFloat := x.Mul(exp64_invLn2).RoundToEven()
	r := x.Sub(kFloat.Mul(exp64_ln2Hi))
	r = r.Sub(kFloat.Mul(exp64_ln2Lo))

	// Polynomial approximation (degree 10 for float64)
	p := exp64_c10.MulAdd(r, exp64_c9)
	p = p.MulAdd(r, exp64_c8)
	p = p.MulAdd(r, exp64_c7)
	p = p.MulAdd(r, exp64_c6)
	p = p.MulAdd(r, exp64_c5)
	p = p.MulAdd(r, exp64_c4)
	p = p.MulAdd(r, exp64_c3)
	p = p.MulAdd(r, exp64_c2)
	p = p.MulAdd(r, exp64_c1)
	p = p.MulAdd(r, exp64_one)

	// Scale by 2^k using magic number trick (pure SIMD, no scalar conversion).
	// Add offset to ensure k is positive, embed in mantissa via 2^52 addition,
	// then extract and construct the float64 bit pattern.
	kPositive := kFloat.Add(exp64_magicOffset)           // k + 1024, always positive
	kPlusMagic := kPositive.Add(exp64_magic)             // Embed in mantissa
	kInt := kPlusMagic.AsInt64x4().Sub(exp64_magicAdjust) // Extract k + 1024
	expBits := kInt.Sub(exp64_magicOne).ShiftAllLeft(52) // (k + 1023) << 52
	scale := expBits.AsFloat64x4()

	result := p.Mul(scale)

	// Handle special cases (Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	result = exp64_inf.Merge(result, overflowMask)
	result = exp64_zero.Merge(result, underflowMask)

	return result
}
