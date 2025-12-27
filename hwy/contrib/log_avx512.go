//go:build amd64 && goexperiment.simd

package contrib

import (
	"math"
	"simd/archsimd"
)

// AVX-512 vectorized constants for log32
var (
	// Polynomial coefficients for atanh-based log approximation
	// ln((1+y)/(1-y)) = 2y(1 + y²/3 + y⁴/5 + y⁶/7 + ...)
	log512_32_c1 = archsimd.BroadcastFloat32x16(0.3333333333333367565) // 1/3
	log512_32_c2 = archsimd.BroadcastFloat32x16(0.1999999999970470954) // 1/5
	log512_32_c3 = archsimd.BroadcastFloat32x16(0.1428571437183119574) // 1/7
	log512_32_c4 = archsimd.BroadcastFloat32x16(0.1111109921607489198) // 1/9
	log512_32_c5 = archsimd.BroadcastFloat32x16(0.0909178608080902506) // 1/11

	// ln(2) split for precision
	log512_32_ln2Hi = archsimd.BroadcastFloat32x16(0.693359375)
	log512_32_ln2Lo = archsimd.BroadcastFloat32x16(-2.12194440e-4)

	// Constants
	log512_32_one      = archsimd.BroadcastFloat32x16(1.0)
	log512_32_two      = archsimd.BroadcastFloat32x16(2.0)
	log512_32_zero     = archsimd.BroadcastFloat32x16(0.0)
	log512_32_sqrtHalf = archsimd.BroadcastFloat32x16(0.7071067811865476) // sqrt(2)/2

	// Special values
	log512_32_negInf = archsimd.BroadcastFloat32x16(float32(math.Inf(-1)))
	log512_32_posInf = archsimd.BroadcastFloat32x16(float32(math.Inf(1)))
	log512_32_nan    = archsimd.BroadcastFloat32x16(float32(math.NaN()))

	// Bit manipulation constants for IEEE 754
	log512_32_mantMask = archsimd.BroadcastInt32x16(0x007FFFFF) // mantissa mask
	log512_32_expBias  = archsimd.BroadcastInt32x16(127)        // exponent bias
	log512_32_normBits = archsimd.BroadcastInt32x16(0x3F800000) // 1.0 in IEEE 754
	log512_32_intOne   = archsimd.BroadcastInt32x16(1)
)

// AVX-512 vectorized constants for log64
var (
	// Higher-degree polynomial for float64
	log512_64_c1 = archsimd.BroadcastFloat64x8(0.3333333333333367565) // 1/3
	log512_64_c2 = archsimd.BroadcastFloat64x8(0.1999999999970470954) // 1/5
	log512_64_c3 = archsimd.BroadcastFloat64x8(0.1428571437183119574) // 1/7
	log512_64_c4 = archsimd.BroadcastFloat64x8(0.1111109921607489198) // 1/9
	log512_64_c5 = archsimd.BroadcastFloat64x8(0.0909178608080902506) // 1/11
	log512_64_c6 = archsimd.BroadcastFloat64x8(0.0765691884960468666) // 1/13
	log512_64_c7 = archsimd.BroadcastFloat64x8(0.0739909930255829295) // 1/15 (approx)

	log512_64_ln2Hi = archsimd.BroadcastFloat64x8(0.6931471803691238)
	log512_64_ln2Lo = archsimd.BroadcastFloat64x8(1.9082149292705877e-10)

	log512_64_one      = archsimd.BroadcastFloat64x8(1.0)
	log512_64_two      = archsimd.BroadcastFloat64x8(2.0)
	log512_64_zero     = archsimd.BroadcastFloat64x8(0.0)
	log512_64_sqrtHalf = archsimd.BroadcastFloat64x8(0.7071067811865476)

	log512_64_negInf = archsimd.BroadcastFloat64x8(math.Inf(-1))
	log512_64_posInf = archsimd.BroadcastFloat64x8(math.Inf(1))
	log512_64_nan    = archsimd.BroadcastFloat64x8(math.NaN())

	log512_64_mantMask = archsimd.BroadcastInt64x8(0x000FFFFFFFFFFFFF)
	log512_64_expBias  = archsimd.BroadcastInt64x8(1023)
	log512_64_normBits = archsimd.BroadcastInt64x8(0x3FF0000000000000)
	log512_64_intOne   = archsimd.BroadcastInt64x8(1)
)

// Log_AVX512_F32x16 computes ln(x) for a single Float32x16 vector.
//
// Algorithm:
// 1. Range reduction: x = 2^e * m where 1 <= m < 2
// 2. If m < sqrt(2)/2, adjust: m = 2*m, e = e-1 (so sqrt(2)/2 <= m < sqrt(2))
// 3. Transform: y = (m-1)/(m+1)
// 4. Polynomial: ln(m) = 2*y*(1 + c1*y^2 + c2*y^4 + ...)
// 5. Reconstruct: ln(x) = e*ln(2) + ln(m)
func Log_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	// Save input for special case handling
	origX := x

	// Extract exponent and mantissa using IEEE 754 bit manipulation
	// float32: sign(1) | exponent(8) | mantissa(23)
	xBits := x.AsInt32x16()

	// Extract exponent: shift right by 23, subtract bias
	// exp = ((xBits >> 23) & 0xFF) - 127
	exp := xBits.ShiftAllRight(23).And(archsimd.BroadcastInt32x16(0xFF)).Sub(log512_32_expBias)

	// Extract mantissa and normalize to [1, 2)
	// mantissa bits OR'd with exponent=127 (normalized form)
	// Use addition instead of OR since the exponent field is zero in mantissa
	mantOnly := xBits.And(log512_32_mantMask)
	mantBits := mantOnly.Add(log512_32_normBits)
	m := mantBits.AsFloat32x16()

	// If m < sqrt(2)/2 (~0.707), use m*2 and e-1 for better accuracy
	// This centers the range around 1.0
	adjustMask := m.Less(log512_32_sqrtHalf)
	// For lanes where m < sqrt(2)/2: m = m*2, exp = exp-1
	mAdjusted := m.Mul(log512_32_two)
	expAdjusted := exp.Sub(log512_32_intOne)
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	m = mAdjusted.Merge(m, adjustMask)
	// Use float conversion to apply mask (exp values are small integers, no precision loss)
	expFloat := exp.ConvertToFloat32()
	expAdjustedFloat := expAdjusted.ConvertToFloat32()
	exp = expAdjustedFloat.Merge(expFloat, adjustMask).ConvertToInt32()

	// Transform: y = (m-1)/(m+1)
	// This maps m in [sqrt(2)/2, sqrt(2)] to y in [-0.17, 0.17]
	mMinus1 := m.Sub(log512_32_one)
	mPlus1 := m.Add(log512_32_one)
	y := mMinus1.Div(mPlus1)
	y2 := y.Mul(y)

	// Polynomial approximation for 2*atanh(y) = ln((1+y)/(1-y))
	// ln(m) = 2*y*(1 + c1*y^2 + c2*y^4 + c3*y^6 + c4*y^8 + c5*y^10)
	// Using Horner's method
	p := log512_32_c5.MulAdd(y2, log512_32_c4)
	p = p.MulAdd(y2, log512_32_c3)
	p = p.MulAdd(y2, log512_32_c2)
	p = p.MulAdd(y2, log512_32_c1)
	p = p.MulAdd(y2, log512_32_one) // p = 1 + c1*y^2 + ...

	// ln(m) = 2*y*p
	lnM := log512_32_two.Mul(y).Mul(p)

	// Reconstruct: ln(x) = e*ln(2) + ln(m)
	// Use high/low split for ln(2) to maintain precision
	expFloat = exp.ConvertToFloat32()
	result := expFloat.Mul(log512_32_ln2Hi)
	result = result.Add(expFloat.Mul(log512_32_ln2Lo))
	result = result.Add(lnM)

	// Handle special cases (Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	// x <= 0: return NaN (log of negative or zero)
	// x == 0: return -Inf
	// x == +Inf: return +Inf
	// x is NaN: return NaN
	zeroMask := origX.Equal(log512_32_zero)
	negMask := origX.Less(log512_32_zero)
	infMask := origX.Equal(log512_32_posInf)

	result = log512_32_negInf.Merge(result, zeroMask)
	result = log512_32_nan.Merge(result, negMask)
	result = log512_32_posInf.Merge(result, infMask)

	return result
}

// Log_AVX512_F64x8 computes ln(x) for a single Float64x8 vector.
//
// Algorithm: Same as F32x16 but with higher-degree polynomial for float64 precision.
func Log_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	// Save input for special case handling
	origX := x

	// Extract exponent and mantissa using IEEE 754 bit manipulation
	// float64: sign(1) | exponent(11) | mantissa(52)
	xBits := x.AsInt64x8()

	// Extract exponent: shift right by 52, subtract bias (1023)
	exp := xBits.ShiftAllRight(52).And(archsimd.BroadcastInt64x8(0x7FF)).Sub(log512_64_expBias)

	// Extract mantissa and normalize to [1, 2)
	// Use addition instead of OR since the exponent field is zero in mantissa
	mantOnly := xBits.And(log512_64_mantMask)
	mantBits := mantOnly.Add(log512_64_normBits)
	m := mantBits.AsFloat64x8()

	// If m < sqrt(2)/2 (~0.707), use m*2 and e-1 for better accuracy
	adjustMask := m.Less(log512_64_sqrtHalf)
	mAdjusted := m.Mul(log512_64_two)
	expAdjusted := exp.Sub(log512_64_intOne)
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	m = mAdjusted.Merge(m, adjustMask)
	// Use float conversion to apply mask (exp values are small integers, no precision loss)
	expFloat := exp.ConvertToFloat64()
	expAdjustedFloat := expAdjusted.ConvertToFloat64()
	exp = expAdjustedFloat.Merge(expFloat, adjustMask).ConvertToInt64()

	// Transform: y = (m-1)/(m+1)
	mMinus1 := m.Sub(log512_64_one)
	mPlus1 := m.Add(log512_64_one)
	y := mMinus1.Div(mPlus1)
	y2 := y.Mul(y)

	// Polynomial approximation (higher degree for float64)
	// ln(m) = 2*y*(1 + c1*y^2 + c2*y^4 + c3*y^6 + c4*y^8 + c5*y^10 + c6*y^12 + c7*y^14)
	p := log512_64_c7.MulAdd(y2, log512_64_c6)
	p = p.MulAdd(y2, log512_64_c5)
	p = p.MulAdd(y2, log512_64_c4)
	p = p.MulAdd(y2, log512_64_c3)
	p = p.MulAdd(y2, log512_64_c2)
	p = p.MulAdd(y2, log512_64_c1)
	p = p.MulAdd(y2, log512_64_one)

	// ln(m) = 2*y*p
	lnM := log512_64_two.Mul(y).Mul(p)

	// Reconstruct: ln(x) = e*ln(2) + ln(m)
	expFloat = exp.ConvertToFloat64()
	result := expFloat.Mul(log512_64_ln2Hi)
	result = result.Add(expFloat.Mul(log512_64_ln2Lo))
	result = result.Add(lnM)

	// Handle special cases (Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	zeroMask := origX.Equal(log512_64_zero)
	negMask := origX.Less(log512_64_zero)
	infMask := origX.Equal(log512_64_posInf)

	result = log512_64_negInf.Merge(result, zeroMask)
	result = log512_64_nan.Merge(result, negMask)
	result = log512_64_posInf.Merge(result, infMask)

	return result
}
