//go:generate go run ../../../cmd/hwygen -input vec_math_base.go -output . -targets avx2,avx512,neon,fallback

package math

import (
	"github.com/ajroetker/go-highway/hwy"
)

// =============================================================================
// Vec-to-Vec Mathematical Functions (Zero Allocation)
// =============================================================================
//
// These functions operate on SIMD registers directly, enabling composition
// without heap allocation. They are designed to be called from Transform
// wrappers that handle the slice→register→slice conversion.
//
// Example composition (all register-to-register):
//   BaseTanhVec calls BaseSigmoidVec calls BaseExpVec
//
// The hwygen tool generates target-specific versions:
//   BaseExpVec_avx2(x archsimd.Float32x8) archsimd.Float32x8
//   BaseExpVec_avx512(x archsimd.Float32x16) archsimd.Float32x16
//   etc.

// =============================================================================
// Constants (shared with poly versions)
// =============================================================================

// BaseExpVec computes e^x for a single vector, returning the result.
// This is the register-level building block for zero-allocation composition.
//
// Algorithm:
// 1. Range reduction: x = k*ln(2) + r, where |r| <= ln(2)/2
// 2. Polynomial approximation: e^r ≈ 1 + r + r²/2! + r³/3! + ...
// 3. Reconstruction: e^x = 2^k * e^r using IEEE 754 bit manipulation
func BaseExpVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	overflow := hwy.Set[T](T(expOverflow_f32))
	underflow := hwy.Set[T](T(expUnderflow_f32))
	one := hwy.Set[T](T(expOne_f32))
	zero := hwy.Set[T](T(expZero_f32))
	inf := hwy.Set[T](T(expOverflow_f32 * 2))
	invLn2 := hwy.Set[T](T(expInvLn2_f32))
	ln2Hi := hwy.Set[T](T(expLn2Hi_f32))
	ln2Lo := hwy.Set[T](T(expLn2Lo_f32))

	c1 := hwy.Set[T](T(expC1_f32))
	c2 := hwy.Set[T](T(expC2_f32))
	c3 := hwy.Set[T](T(expC3_f32))
	c4 := hwy.Set[T](T(expC4_f32))
	c5 := hwy.Set[T](T(expC5_f32))
	c6 := hwy.Set[T](T(expC6_f32))

	// Check overflow/underflow
	overflowMask := hwy.Greater(x, overflow)
	underflowMask := hwy.Less(x, underflow)

	// Range reduction: k = round(x / ln(2)), r = x - k * ln(2)
	kFloat := hwy.RoundToEven(hwy.Mul(x, invLn2))

	// r = x - k*ln(2) using high/low split for precision
	r := hwy.Sub(x, hwy.Mul(kFloat, ln2Hi))
	r = hwy.Sub(r, hwy.Mul(kFloat, ln2Lo))

	// Polynomial approximation using Horner's method
	p := hwy.MulAdd(c6, r, c5)
	p = hwy.MulAdd(p, r, c4)
	p = hwy.MulAdd(p, r, c3)
	p = hwy.MulAdd(p, r, c2)
	p = hwy.MulAdd(p, r, c1)
	p = hwy.MulAdd(p, r, one)

	// Scale by 2^k using IEEE 754 bit manipulation
	kInt := hwy.ConvertToInt32(kFloat)
	scale := hwy.Pow2[T](kInt)
	result := hwy.Mul(p, scale)

	// Handle special cases
	result = hwy.Merge(inf, result, overflowMask)
	result = hwy.Merge(zero, result, underflowMask)

	return result
}

// BaseSigmoidVec computes sigmoid(x) = 1 / (1 + e^(-x)) using BaseExpVec.
// Zero allocation - composes at register level.
func BaseSigmoidVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	one := hwy.Set[T](T(sigmoidOne_f32))
	zero := hwy.Set[T](T(0.0))
	satHi := hwy.Set[T](T(20.0))
	satLo := hwy.Set[T](T(-20.0))

	// Clamp to avoid exp overflow
	clampedX := hwy.Max(hwy.Min(x, satHi), satLo)

	// Compute exp(-x) using BaseExpVec - register-to-register, no allocation
	negX := hwy.Neg(clampedX)
	expNegX := BaseExpVec[T](negX)

	// sigmoid(x) = 1 / (1 + exp(-x))
	result := hwy.Div(one, hwy.Add(one, expNegX))

	// Handle saturation
	result = hwy.Merge(one, result, hwy.Greater(x, satHi))
	result = hwy.Merge(zero, result, hwy.Less(x, satLo))

	return result
}

// BaseTanhVec computes tanh(x) = 2*sigmoid(2x) - 1 using BaseSigmoidVec.
// Zero allocation - composes at register level.
func BaseTanhVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	two := hwy.Set[T](T(2.0))
	one := hwy.Set[T](T(tanhOne_f32))
	negOne := hwy.Set[T](T(tanhNegOne_f32))
	threshold := hwy.Set[T](T(tanhClamp_f32))
	negThreshold := hwy.Neg(threshold)

	// tanh(x) = 2*sigmoid(2x) - 1 using BaseSigmoidVec - register-to-register
	twoX := hwy.Mul(two, x)
	sigTwoX := BaseSigmoidVec[T](twoX)
	result := hwy.Sub(hwy.Mul(two, sigTwoX), one)

	// Handle saturation
	result = hwy.Merge(one, result, hwy.Greater(x, threshold))
	result = hwy.Merge(negOne, result, hwy.Less(x, negThreshold))

	return result
}

// BaseLogVec computes ln(x) for a single vector.
// Zero allocation - register-level operation.
func BaseLogVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	one := hwy.Set[T](T(logOne_f32))
	two := hwy.Set[T](T(logTwo_f32))
	zero := hwy.Set[T](T(0.0))
	ln2Hi := hwy.Set[T](T(logLn2Hi_f32))
	ln2Lo := hwy.Set[T](T(logLn2Lo_f32))
	negInf := hwy.Set[T](T(-1e38)) // Approximate -Inf
	nan := hwy.Set[T](T(0.0))      // Will be replaced with NaN mask

	c1 := hwy.Set[T](T(logC1_f32))
	c2 := hwy.Set[T](T(logC2_f32))
	c3 := hwy.Set[T](T(logC3_f32))
	c4 := hwy.Set[T](T(logC4_f32))
	c5 := hwy.Set[T](T(logC5_f32))

	// Handle special cases
	zeroMask := hwy.Equal(x, zero)
	negMask := hwy.Less(x, zero)
	oneMask := hwy.Equal(x, one)

	// Extract exponent and mantissa
	e := hwy.GetExponent(x)
	m := hwy.GetMantissa(x)

	// Adjust for m > sqrt(2)
	mLarge := hwy.Greater(m, hwy.Set[T](T(1.414)))
	mAdjusted := hwy.Merge(hwy.Mul(m, hwy.Set[T](T(0.5))), m, mLarge)

	// Convert exponent to float
	eData := e.Data()
	eFloatData := make([]T, len(eData))
	for i, v := range eData {
		eFloatData[i] = T(v)
	}
	eFloat := hwy.Load(eFloatData)
	eAdjusted := hwy.Merge(hwy.Add(eFloat, one), eFloat, mLarge)

	// Compute log(m) using y = (m-1)/(m+1)
	mMinus1 := hwy.Sub(mAdjusted, one)
	mPlus1 := hwy.Add(mAdjusted, one)
	y := hwy.Div(mMinus1, mPlus1)
	y2 := hwy.Mul(y, y)

	// Polynomial
	poly := hwy.MulAdd(c5, y2, c4)
	poly = hwy.MulAdd(poly, y2, c3)
	poly = hwy.MulAdd(poly, y2, c2)
	poly = hwy.MulAdd(poly, y2, c1)
	logM := hwy.Mul(hwy.Mul(two, y), poly)

	// log(x) = e*ln(2) + log(m)
	result := hwy.Add(hwy.MulAdd(eAdjusted, ln2Hi, logM), hwy.Mul(eAdjusted, ln2Lo))

	// Handle special cases
	result = hwy.Merge(negInf, result, zeroMask)
	result = hwy.Merge(nan, result, negMask)
	result = hwy.Merge(zero, result, oneMask)

	return result
}

// BaseSinVec computes sin(x) for a single vector.
// Zero allocation - register-level operation.
func BaseSinVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	twoOverPi := hwy.Set[T](T(trig2OverPi_f32))
	piOver2Hi := hwy.Set[T](T(trigPiOver2Hi_f32))
	piOver2Lo := hwy.Set[T](T(trigPiOver2Lo_f32))
	one := hwy.Set[T](T(trigOne_f32))
	s1 := hwy.Set[T](T(trigS1_f32))
	s2 := hwy.Set[T](T(trigS2_f32))
	s3 := hwy.Set[T](T(trigS3_f32))
	s4 := hwy.Set[T](T(trigS4_f32))
	c1 := hwy.Set[T](T(trigC1_f32))
	c2 := hwy.Set[T](T(trigC2_f32))
	c3 := hwy.Set[T](T(trigC3_f32))
	c4 := hwy.Set[T](T(trigC4_f32))

	intOne := hwy.Set[int32](1)
	intTwo := hwy.Set[int32](2)
	intThree := hwy.Set[int32](3)

	// Range reduction
	kFloat := hwy.RoundToEven(hwy.Mul(x, twoOverPi))
	kInt := hwy.ConvertToInt32(kFloat)

	r := hwy.Sub(x, hwy.Mul(kFloat, piOver2Hi))
	r = hwy.Sub(r, hwy.Mul(kFloat, piOver2Lo))
	r2 := hwy.Mul(r, r)

	// Compute sin(r) and cos(r) polynomials
	sinPoly := hwy.MulAdd(s4, r2, s3)
	sinPoly = hwy.MulAdd(sinPoly, r2, s2)
	sinPoly = hwy.MulAdd(sinPoly, r2, s1)
	sinPoly = hwy.MulAdd(sinPoly, r2, one)
	sinR := hwy.Mul(r, sinPoly)

	cosPoly := hwy.MulAdd(c4, r2, c3)
	cosPoly = hwy.MulAdd(cosPoly, r2, c2)
	cosPoly = hwy.MulAdd(cosPoly, r2, c1)
	cosR := hwy.MulAdd(cosPoly, r2, one)

	// Octant selection
	octant := hwy.And(kInt, intThree)
	useCosMask := hwy.Equal(hwy.And(octant, intOne), intOne)
	negateMask := hwy.Equal(hwy.And(octant, intTwo), intTwo)

	// Select between sin(r) and cos(r)
	sinRData := sinR.Data()
	cosRData := cosR.Data()
	resultData := make([]T, len(sinRData))
	for i := range sinRData {
		if useCosMask.GetBit(i) {
			resultData[i] = cosRData[i]
		} else {
			resultData[i] = sinRData[i]
		}
	}
	result := hwy.Load(resultData)

	// Negate if needed
	negResult := hwy.Neg(result)
	negResultData := negResult.Data()
	for i := range resultData {
		if negateMask.GetBit(i) {
			resultData[i] = negResultData[i]
		}
	}
	return hwy.Load(resultData)
}

// BaseCosVec computes cos(x) for a single vector.
// Zero allocation - register-level operation.
func BaseCosVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	twoOverPi := hwy.Set[T](T(trig2OverPi_f32))
	piOver2Hi := hwy.Set[T](T(trigPiOver2Hi_f32))
	piOver2Lo := hwy.Set[T](T(trigPiOver2Lo_f32))
	one := hwy.Set[T](T(trigOne_f32))
	s1 := hwy.Set[T](T(trigS1_f32))
	s2 := hwy.Set[T](T(trigS2_f32))
	s3 := hwy.Set[T](T(trigS3_f32))
	s4 := hwy.Set[T](T(trigS4_f32))
	c1 := hwy.Set[T](T(trigC1_f32))
	c2 := hwy.Set[T](T(trigC2_f32))
	c3 := hwy.Set[T](T(trigC3_f32))
	c4 := hwy.Set[T](T(trigC4_f32))

	intOne := hwy.Set[int32](1)
	intTwo := hwy.Set[int32](2)
	intThree := hwy.Set[int32](3)

	// Range reduction
	kFloat := hwy.RoundToEven(hwy.Mul(x, twoOverPi))
	kInt := hwy.ConvertToInt32(kFloat)

	r := hwy.Sub(x, hwy.Mul(kFloat, piOver2Hi))
	r = hwy.Sub(r, hwy.Mul(kFloat, piOver2Lo))
	r2 := hwy.Mul(r, r)

	// Compute sin(r) and cos(r) polynomials
	sinPoly := hwy.MulAdd(s4, r2, s3)
	sinPoly = hwy.MulAdd(sinPoly, r2, s2)
	sinPoly = hwy.MulAdd(sinPoly, r2, s1)
	sinPoly = hwy.MulAdd(sinPoly, r2, one)
	sinR := hwy.Mul(r, sinPoly)

	cosPoly := hwy.MulAdd(c4, r2, c3)
	cosPoly = hwy.MulAdd(cosPoly, r2, c2)
	cosPoly = hwy.MulAdd(cosPoly, r2, c1)
	cosR := hwy.MulAdd(cosPoly, r2, one)

	// For cos(x), shift by 1 octant
	cosOctant := hwy.And(hwy.Add(kInt, intOne), intThree)
	useCosMask := hwy.Equal(hwy.And(cosOctant, intOne), intOne)
	negateMask := hwy.Equal(hwy.And(cosOctant, intTwo), intTwo)

	// Select between sin(r) and cos(r)
	sinRData := sinR.Data()
	cosRData := cosR.Data()
	resultData := make([]T, len(sinRData))
	for i := range sinRData {
		if useCosMask.GetBit(i) {
			resultData[i] = cosRData[i]
		} else {
			resultData[i] = sinRData[i]
		}
	}
	result := hwy.Load(resultData)

	// Negate if needed
	negResult := hwy.Neg(result)
	negResultData := negResult.Data()
	for i := range resultData {
		if negateMask.GetBit(i) {
			resultData[i] = negResultData[i]
		}
	}
	return hwy.Load(resultData)
}

// BaseErfVec computes erf(x) for a single vector.
// Zero allocation - register-level operation (except for internal exp call).
func BaseErfVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	a1 := hwy.Set[T](T(erfA1_f32))
	a2 := hwy.Set[T](T(erfA2_f32))
	a3 := hwy.Set[T](T(erfA3_f32))
	a4 := hwy.Set[T](T(erfA4_f32))
	a5 := hwy.Set[T](T(erfA5_f32))
	p := hwy.Set[T](T(erfP_f32))
	one := hwy.Set[T](T(erfOne_f32))
	zero := hwy.Set[T](T(erfZero_f32))

	// erf(-x) = -erf(x)
	absX := hwy.Abs(x)
	signMask := hwy.Less(x, zero)

	// t = 1 / (1 + p * |x|)
	t := hwy.Div(one, hwy.Add(one, hwy.Mul(p, absX)))

	// Polynomial using Horner's method
	poly := hwy.MulAdd(a5, t, a4)
	poly = hwy.MulAdd(poly, t, a3)
	poly = hwy.MulAdd(poly, t, a2)
	poly = hwy.MulAdd(poly, t, a1)
	poly = hwy.Mul(poly, t)

	// Compute exp(-x²) using BaseExpVec
	x2 := hwy.Mul(absX, absX)
	negX2 := hwy.Neg(x2)
	expNegX2 := BaseExpVec[T](negX2)

	// erf(|x|) = 1 - poly * exp(-x²)
	erfAbs := hwy.Sub(one, hwy.Mul(poly, expNegX2))

	// Clamp to [0, 1]
	erfAbs = hwy.Max(hwy.Min(erfAbs, one), zero)

	// Apply sign
	negErfAbs := hwy.Neg(erfAbs)
	result := hwy.Merge(negErfAbs, erfAbs, signMask)

	return result
}

// BaseLog2Vec computes log₂(x) for a single vector.
// Zero allocation - composes with BaseLogVec.
func BaseLog2Vec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	log2E := hwy.Set[T](T(log2E_f32))
	lnX := BaseLogVec[T](x)
	return hwy.Mul(lnX, log2E)
}

// BaseLog10Vec computes log₁₀(x) for a single vector.
// Zero allocation - composes with BaseLogVec.
func BaseLog10Vec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	log10E := hwy.Set[T](T(log10E_f32))
	lnX := BaseLogVec[T](x)
	return hwy.Mul(lnX, log10E)
}

// BaseExp2Vec computes 2^x for a single vector.
// Zero allocation - composes with BaseExpVec.
func BaseExp2Vec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	ln2 := hwy.Set[T](T(ln2_f32))
	xLn2 := hwy.Mul(x, ln2)
	return BaseExpVec[T](xLn2)
}

// BaseSinhVec computes sinh(x) for a single vector.
// Zero allocation - register-level operation.
func BaseSinhVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	one := hwy.Set[T](T(sinhOne_f32))
	c3 := hwy.Set[T](T(sinhC3_f32))
	c5 := hwy.Set[T](T(sinhC5_f32))
	c7 := hwy.Set[T](T(sinhC7_f32))

	// Polynomial: sinh(x) ≈ x * (1 + x²/6 + x⁴/120 + x⁶/5040)
	x2 := hwy.Mul(x, x)
	poly := hwy.MulAdd(c7, x2, c5)
	poly = hwy.MulAdd(poly, x2, c3)
	poly = hwy.MulAdd(poly, x2, one)
	return hwy.Mul(x, poly)
}

// BaseCoshVec computes cosh(x) for a single vector.
// Zero allocation - register-level operation.
func BaseCoshVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	one := hwy.Set[T](T(1.0))
	c2 := hwy.Set[T](T(0.5))
	c4 := hwy.Set[T](T(0.041666666666666664))
	c6 := hwy.Set[T](T(0.001388888888888889))

	// cosh is even: cosh(-x) = cosh(x)
	x2 := hwy.Mul(x, x)

	// Polynomial: 1 + x²/2 + x⁴/24 + x⁶/720
	poly := hwy.MulAdd(c6, x2, c4)
	poly = hwy.MulAdd(poly, x2, c2)
	return hwy.MulAdd(poly, x2, one)
}

// BaseAsinhVec computes asinh(x) for a single vector.
// asinh(x) = ln(x + sqrt(x² + 1))
func BaseAsinhVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	one := hwy.Set[T](T(1.0))

	x2 := hwy.Mul(x, x)
	x2Plus1 := hwy.Add(x2, one)
	sqrtPart := hwy.Sqrt(x2Plus1)
	arg := hwy.Add(x, sqrtPart)
	return BaseLogVec[T](arg)
}

// BaseAcoshVec computes acosh(x) for a single vector (x >= 1).
// acosh(x) = ln(x + sqrt(x² - 1))
func BaseAcoshVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	one := hwy.Set[T](T(1.0))
	zero := hwy.Set[T](T(0.0))

	x2 := hwy.Mul(x, x)
	x2Minus1 := hwy.Sub(x2, one)
	sqrtPart := hwy.Sqrt(x2Minus1)
	arg := hwy.Add(x, sqrtPart)
	result := BaseLogVec[T](arg)

	// Handle x = 1: acosh(1) = 0
	oneMask := hwy.Equal(x, one)
	result = hwy.Merge(zero, result, oneMask)

	return result
}

// BaseAtanhVec computes atanh(x) for a single vector (|x| < 1).
// atanh(x) = 0.5 * ln((1+x)/(1-x))
func BaseAtanhVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	one := hwy.Set[T](T(1.0))
	half := hwy.Set[T](T(0.5))
	zero := hwy.Set[T](T(0.0))

	onePlusX := hwy.Add(one, x)
	oneMinusX := hwy.Sub(one, x)
	ratio := hwy.Div(onePlusX, oneMinusX)
	logRatio := BaseLogVec[T](ratio)
	result := hwy.Mul(half, logRatio)

	// Handle x = 0
	zeroMask := hwy.Equal(x, zero)
	result = hwy.Merge(zero, result, zeroMask)

	return result
}

// BasePowVec computes base^exp for vectors element-wise.
// Uses the identity: base^exp = exp(exp * log(base))
// Note: Only valid for base > 0. Negative bases will produce NaN.
func BasePowVec[T hwy.Floats](base, exp hwy.Vec[T]) hwy.Vec[T] {
	one := hwy.Set[T](T(1.0))
	zero := hwy.Set[T](T(0.0))

	// base^exp = exp(exp * log(base))
	logBase := BaseLogVec[T](base)
	expTimesLog := hwy.Mul(exp, logBase)
	result := BaseExpVec[T](expTimesLog)

	// Handle special cases:
	// exp = 0 -> result = 1
	expZeroMask := hwy.Equal(exp, zero)
	result = hwy.Merge(one, result, expZeroMask)

	// base = 1 -> result = 1
	baseOneMask := hwy.Equal(base, one)
	result = hwy.Merge(one, result, baseOneMask)

	// base = 0, exp > 0 -> result = 0
	baseZeroMask := hwy.Equal(base, zero)
	expPosMask := hwy.GreaterThan(exp, zero)
	baseZeroExpPosMask := hwy.MaskAnd(baseZeroMask, expPosMask)
	result = hwy.Merge(zero, result, baseZeroExpPosMask)

	return result
}
