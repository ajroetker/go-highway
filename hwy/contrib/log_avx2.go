//go:build amd64 && goexperiment.simd

package contrib

import (
	"math"
	"simd/archsimd"
)

// AVX2 vectorized constants for log32
var (
	// Polynomial coefficients for atanh-based log approximation
	// ln((1+y)/(1-y)) = 2y(1 + y²/3 + y⁴/5 + y⁶/7 + ...)
	log32_c1 = archsimd.BroadcastFloat32x8(0.3333333333333367565) // 1/3
	log32_c2 = archsimd.BroadcastFloat32x8(0.1999999999970470954) // 1/5
	log32_c3 = archsimd.BroadcastFloat32x8(0.1428571437183119574) // 1/7
	log32_c4 = archsimd.BroadcastFloat32x8(0.1111109921607489198) // 1/9
	log32_c5 = archsimd.BroadcastFloat32x8(0.0909178608080902506) // 1/11

	// ln(2) split for precision
	log32_ln2Hi = archsimd.BroadcastFloat32x8(0.693359375)
	log32_ln2Lo = archsimd.BroadcastFloat32x8(-2.12194440e-4)

	// Constants
	log32_one      = archsimd.BroadcastFloat32x8(1.0)
	log32_two      = archsimd.BroadcastFloat32x8(2.0)
	log32_zero     = archsimd.BroadcastFloat32x8(0.0)
	log32_sqrtHalf = archsimd.BroadcastFloat32x8(0.7071067811865476) // sqrt(2)/2

	// Special values
	log32_negInf = archsimd.BroadcastFloat32x8(float32(math.Inf(-1)))
	log32_posInf = archsimd.BroadcastFloat32x8(float32(math.Inf(1)))
	log32_nan    = archsimd.BroadcastFloat32x8(float32(math.NaN()))

	// Bit manipulation constants for IEEE 754
	log32_mantMask = archsimd.BroadcastInt32x8(0x007FFFFF)  // mantissa mask
	log32_expBias  = archsimd.BroadcastInt32x8(127)         // exponent bias
	log32_normBits = archsimd.BroadcastInt32x8(0x3F800000)  // 1.0 in IEEE 754
	log32_intOne   = archsimd.BroadcastInt32x8(1)
)

// AVX2 vectorized constants for log64
var (
	// Higher-degree polynomial for float64
	log64_c1 = archsimd.BroadcastFloat64x4(0.3333333333333367565) // 1/3
	log64_c2 = archsimd.BroadcastFloat64x4(0.1999999999970470954) // 1/5
	log64_c3 = archsimd.BroadcastFloat64x4(0.1428571437183119574) // 1/7
	log64_c4 = archsimd.BroadcastFloat64x4(0.1111109921607489198) // 1/9
	log64_c5 = archsimd.BroadcastFloat64x4(0.0909178608080902506) // 1/11
	log64_c6 = archsimd.BroadcastFloat64x4(0.0765691884960468666) // 1/13
	log64_c7 = archsimd.BroadcastFloat64x4(0.0739909930255829295) // 1/15 (approx)

	log64_ln2Hi = archsimd.BroadcastFloat64x4(0.6931471803691238)
	log64_ln2Lo = archsimd.BroadcastFloat64x4(1.9082149292705877e-10)

	log64_one      = archsimd.BroadcastFloat64x4(1.0)
	log64_two      = archsimd.BroadcastFloat64x4(2.0)
	log64_zero     = archsimd.BroadcastFloat64x4(0.0)
	log64_sqrtHalf = archsimd.BroadcastFloat64x4(0.7071067811865476)

	log64_negInf = archsimd.BroadcastFloat64x4(math.Inf(-1))
	log64_posInf = archsimd.BroadcastFloat64x4(math.Inf(1))
	log64_nan    = archsimd.BroadcastFloat64x4(math.NaN())

	log64_mantMask = archsimd.BroadcastInt64x4(0x000FFFFFFFFFFFFF)
	log64_expBias  = archsimd.BroadcastInt64x4(1023)
	log64_normBits = archsimd.BroadcastInt64x4(0x3FF0000000000000)
	log64_intOne   = archsimd.BroadcastInt64x4(1)
)

// Log_AVX2_F32x8 computes ln(x) for a single Float32x8 vector.
//
// Algorithm:
// 1. Range reduction: x = 2^e * m where 1 <= m < 2
// 2. If m < sqrt(2)/2, adjust: m = 2*m, e = e-1 (so sqrt(2)/2 <= m < sqrt(2))
// 3. Transform: y = (m-1)/(m+1)
// 4. Polynomial: ln(m) = 2*y*(1 + c1*y^2 + c2*y^4 + ...)
// 5. Reconstruct: ln(x) = e*ln(2) + ln(m)
func Log_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Save input for special case handling
	origX := x

	// Extract exponent and mantissa using IEEE 754 bit manipulation
	// float32: sign(1) | exponent(8) | mantissa(23)
	xBits := x.AsInt32x8()

	// Extract exponent: shift right by 23, subtract bias
	// exp = ((xBits >> 23) & 0xFF) - 127
	exp := xBits.ShiftAllRight(23).And(archsimd.BroadcastInt32x8(0xFF)).Sub(log32_expBias)

	// Extract mantissa and normalize to [1, 2)
	// mantissa bits OR'd with exponent=127 (normalized form)
	mantBits := xBits.And(log32_mantMask).Or(log32_normBits)
	m := mantBits.AsFloat32x8()

	// If m < sqrt(2)/2 (~0.707), use m*2 and e-1 for better accuracy
	// This centers the range around 1.0
	adjustMask := m.Less(log32_sqrtHalf)
	// For lanes where m < sqrt(2)/2: m = m*2, exp = exp-1
	mAdjusted := m.Mul(log32_two)
	expAdjusted := exp.Sub(log32_intOne)
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	m = mAdjusted.Merge(m, adjustMask)
	// Use float conversion to apply mask (exp values are small integers, no precision loss)
	expFloat := exp.ConvertToFloat32()
	expAdjustedFloat := expAdjusted.ConvertToFloat32()
	exp = expAdjustedFloat.Merge(expFloat, adjustMask).ConvertToInt32()

	// Transform: y = (m-1)/(m+1)
	// This maps m in [sqrt(2)/2, sqrt(2)] to y in [-0.17, 0.17]
	mMinus1 := m.Sub(log32_one)
	mPlus1 := m.Add(log32_one)
	y := mMinus1.Div(mPlus1)
	y2 := y.Mul(y)

	// Polynomial approximation for 2*atanh(y) = ln((1+y)/(1-y))
	// ln(m) = 2*y*(1 + c1*y^2 + c2*y^4 + c3*y^6 + c4*y^8 + c5*y^10)
	// Using Horner's method
	p := log32_c5.MulAdd(y2, log32_c4)
	p = p.MulAdd(y2, log32_c3)
	p = p.MulAdd(y2, log32_c2)
	p = p.MulAdd(y2, log32_c1)
	p = p.MulAdd(y2, log32_one) // p = 1 + c1*y^2 + ...

	// ln(m) = 2*y*p
	lnM := log32_two.Mul(y).Mul(p)

	// Reconstruct: ln(x) = e*ln(2) + ln(m)
	// Use high/low split for ln(2) to maintain precision
	expFloat = exp.ConvertToFloat32()
	result := expFloat.Mul(log32_ln2Hi)
	result = result.Add(expFloat.Mul(log32_ln2Lo))
	result = result.Add(lnM)

	// Handle special cases (Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	// x <= 0: return NaN (log of negative or zero)
	// x == 0: return -Inf
	// x == +Inf: return +Inf
	// x is NaN: return NaN
	zeroMask := origX.Equal(log32_zero)
	negMask := origX.Less(log32_zero)
	infMask := origX.Equal(log32_posInf)

	result = log32_negInf.Merge(result, zeroMask)
	result = log32_nan.Merge(result, negMask)
	result = log32_posInf.Merge(result, infMask)

	return result
}

// Log_AVX2_F64x4 computes ln(x) for a single Float64x4 vector.
//
// Algorithm: Same as F32x8 but with higher-degree polynomial for float64 precision.
func Log_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	// Save input for special case handling
	origX := x

	// Extract exponent and mantissa using IEEE 754 bit manipulation
	// float64: sign(1) | exponent(11) | mantissa(52)
	xBits := x.AsInt64x4()

	// Extract exponent: shift right by 52, subtract bias (1023)
	exp := xBits.ShiftAllRight(52).And(archsimd.BroadcastInt64x4(0x7FF)).Sub(log64_expBias)

	// Extract mantissa and normalize to [1, 2)
	mantBits := xBits.And(log64_mantMask).Or(log64_normBits)
	m := mantBits.AsFloat64x4()

	// If m < sqrt(2)/2 (~0.707), use m*2 and e-1 for better accuracy
	adjustMask := m.Less(log64_sqrtHalf)
	mAdjusted := m.Mul(log64_two)
	expAdjusted := exp.Sub(log64_intOne)
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	m = mAdjusted.Merge(m, adjustMask)
	// Use float conversion to apply mask (exp values are small integers, no precision loss)
	expFloat := exp.ConvertToFloat64()
	expAdjustedFloat := expAdjusted.ConvertToFloat64()
	exp = expAdjustedFloat.Merge(expFloat, adjustMask).ConvertToInt64()

	// Transform: y = (m-1)/(m+1)
	mMinus1 := m.Sub(log64_one)
	mPlus1 := m.Add(log64_one)
	y := mMinus1.Div(mPlus1)
	y2 := y.Mul(y)

	// Polynomial approximation (higher degree for float64)
	// ln(m) = 2*y*(1 + c1*y^2 + c2*y^4 + c3*y^6 + c4*y^8 + c5*y^10 + c6*y^12 + c7*y^14)
	p := log64_c7.MulAdd(y2, log64_c6)
	p = p.MulAdd(y2, log64_c5)
	p = p.MulAdd(y2, log64_c4)
	p = p.MulAdd(y2, log64_c3)
	p = p.MulAdd(y2, log64_c2)
	p = p.MulAdd(y2, log64_c1)
	p = p.MulAdd(y2, log64_one)

	// ln(m) = 2*y*p
	lnM := log64_two.Mul(y).Mul(p)

	// Reconstruct: ln(x) = e*ln(2) + ln(m)
	expFloat = exp.ConvertToFloat64()
	result := expFloat.Mul(log64_ln2Hi)
	result = result.Add(expFloat.Mul(log64_ln2Lo))
	result = result.Add(lnM)

	// Handle special cases (Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	zeroMask := origX.Equal(log64_zero)
	negMask := origX.Less(log64_zero)
	infMask := origX.Equal(log64_posInf)

	result = log64_negInf.Merge(result, zeroMask)
	result = log64_nan.Merge(result, negMask)
	result = log64_posInf.Merge(result, infMask)

	return result
}
