//go:build amd64 && goexperiment.simd

package contrib

import (
	"math"
	"simd/archsimd"
)

// AVX-512 vectorized constants for trig32
var (
	// Range reduction constants (Cody-Waite)
	// Using 2/π for reduction to [-π/4, π/4] with π/2 intervals
	trig512_32_2overPi   = archsimd.BroadcastFloat32x16(0.6366197723675814)     // 2/π
	trig512_32_piOver2Hi = archsimd.BroadcastFloat32x16(1.5707963267948966)     // π/2 high
	trig512_32_piOver2Lo = archsimd.BroadcastFloat32x16(6.123233995736766e-17)  // π/2 low

	// sin(x) polynomial coefficients for |x| <= π/4
	// sin(x) ≈ x * (1 + s1*x² + s2*x⁴ + s3*x⁶ + s4*x⁸)
	trig512_32_s1 = archsimd.BroadcastFloat32x16(-0.16666666641626524)    // -1/3!
	trig512_32_s2 = archsimd.BroadcastFloat32x16(0.008333329385889463)    // 1/5!
	trig512_32_s3 = archsimd.BroadcastFloat32x16(-0.00019839334836096632) // -1/7!
	trig512_32_s4 = archsimd.BroadcastFloat32x16(2.718311493989822e-6)    // 1/9!

	// cos(x) polynomial coefficients for |x| <= π/4
	// cos(x) ≈ 1 + c1*x² + c2*x⁴ + c3*x⁶ + c4*x⁸
	trig512_32_c1 = archsimd.BroadcastFloat32x16(-0.4999999963229337)   // -1/2!
	trig512_32_c2 = archsimd.BroadcastFloat32x16(0.04166662453689337)   // 1/4!
	trig512_32_c3 = archsimd.BroadcastFloat32x16(-0.001388731625493765) // -1/6!
	trig512_32_c4 = archsimd.BroadcastFloat32x16(2.443315711809948e-5)  // 1/8!

	// Constants
	trig512_32_zero   = archsimd.BroadcastFloat32x16(0.0)
	trig512_32_one    = archsimd.BroadcastFloat32x16(1.0)
	trig512_32_negOne = archsimd.BroadcastFloat32x16(-1.0)
	trig512_32_nan    = archsimd.BroadcastFloat32x16(float32(math.NaN()))
	trig512_32_inf    = archsimd.BroadcastFloat32x16(float32(math.Inf(1)))
	trig512_32_negInf = archsimd.BroadcastFloat32x16(float32(math.Inf(-1)))

	// Integer constants for octant selection
	trig512_32_intOne   = archsimd.BroadcastInt32x16(1)
	trig512_32_intTwo   = archsimd.BroadcastInt32x16(2)
	trig512_32_intThree = archsimd.BroadcastInt32x16(3)
	trig512_32_intZero  = archsimd.BroadcastInt32x16(0)
)

// AVX-512 vectorized constants for trig64
var (
	trig512_64_2overPi   = archsimd.BroadcastFloat64x8(0.6366197723675814)
	trig512_64_piOver2Hi = archsimd.BroadcastFloat64x8(1.5707963267948966192313216916398)
	trig512_64_piOver2Lo = archsimd.BroadcastFloat64x8(6.123233995736766035868820147292e-17)

	// Higher-degree polynomials for float64
	trig512_64_s1 = archsimd.BroadcastFloat64x8(-0.16666666666666632)
	trig512_64_s2 = archsimd.BroadcastFloat64x8(0.008333333333332249)
	trig512_64_s3 = archsimd.BroadcastFloat64x8(-0.00019841269840885721)
	trig512_64_s4 = archsimd.BroadcastFloat64x8(2.7557316103728803e-6)
	trig512_64_s5 = archsimd.BroadcastFloat64x8(-2.5051132068021698e-8)
	trig512_64_s6 = archsimd.BroadcastFloat64x8(1.5896230157221844e-10)

	trig512_64_c1 = archsimd.BroadcastFloat64x8(-0.5)
	trig512_64_c2 = archsimd.BroadcastFloat64x8(0.04166666666666621)
	trig512_64_c3 = archsimd.BroadcastFloat64x8(-0.001388888888887411)
	trig512_64_c4 = archsimd.BroadcastFloat64x8(2.4801587288851704e-5)
	trig512_64_c5 = archsimd.BroadcastFloat64x8(-2.7557314351390663e-7)
	trig512_64_c6 = archsimd.BroadcastFloat64x8(2.0875723212981748e-9)

	trig512_64_zero   = archsimd.BroadcastFloat64x8(0.0)
	trig512_64_one    = archsimd.BroadcastFloat64x8(1.0)
	trig512_64_negOne = archsimd.BroadcastFloat64x8(-1.0)
	trig512_64_nan    = archsimd.BroadcastFloat64x8(math.NaN())
	trig512_64_inf    = archsimd.BroadcastFloat64x8(math.Inf(1))

	trig512_64_intOne   = archsimd.BroadcastInt64x8(1)
	trig512_64_intTwo   = archsimd.BroadcastInt64x8(2)
	trig512_64_intThree = archsimd.BroadcastInt64x8(3)
)

// Sin_AVX512_F32x16 computes sin(x) for a single Float32x16 vector.
//
// Algorithm:
// 1. Range reduction: k = round(x * 2/π), r = x - k*(π/2)
// 2. Compute sin(r) and cos(r) polynomials
// 3. Select based on quadrant: k mod 4
//   - 0: sin(r)
//   - 1: cos(r)
//   - 2: -sin(r)
//   - 3: -cos(r)
func Sin_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	sin, _ := sinCos512_32Core(x)
	return sin
}

// sinCos512_32Core computes both sin and cos for Float32x16.
// This is the shared implementation used by Sin, Cos, and SinCos.
func sinCos512_32Core(x archsimd.Float32x16) (sin, cos archsimd.Float32x16) {
	// Save input for special case handling
	origX := x

	// Range reduction: k = round(x * 2/π)
	// This gives us the quadrant (0-3)
	k := x.Mul(trig512_32_2overPi).RoundToEvenScaled(0)
	kInt := k.ConvertToInt32()

	// r = x - k * (π/2) using Cody-Waite high/low for precision
	r := x.Sub(k.Mul(trig512_32_piOver2Hi))
	r = r.Sub(k.Mul(trig512_32_piOver2Lo))
	r2 := r.Mul(r)

	// Compute sin(r) polynomial: sin(r) ≈ r * (1 + s1*r² + s2*r⁴ + s3*r⁶ + s4*r⁸)
	sinPoly := trig512_32_s4.MulAdd(r2, trig512_32_s3)
	sinPoly = sinPoly.MulAdd(r2, trig512_32_s2)
	sinPoly = sinPoly.MulAdd(r2, trig512_32_s1)
	sinPoly = sinPoly.MulAdd(r2, trig512_32_one)
	sinR := r.Mul(sinPoly)

	// Compute cos(r) polynomial: cos(r) ≈ 1 + c1*r² + c2*r⁴ + c3*r⁶ + c4*r⁸
	cosPoly := trig512_32_c4.MulAdd(r2, trig512_32_c3)
	cosPoly = cosPoly.MulAdd(r2, trig512_32_c2)
	cosPoly = cosPoly.MulAdd(r2, trig512_32_c1)
	cosR := cosPoly.MulAdd(r2, trig512_32_one)

	// Octant selection based on k mod 4
	// For sin(x): quadrant determines which polynomial and sign
	//   k%4 == 0: sin(r)
	//   k%4 == 1: cos(r)
	//   k%4 == 2: -sin(r)
	//   k%4 == 3: -cos(r)
	octant := kInt.And(trig512_32_intThree)

	// Determine if we should use cos instead of sin
	useCosMask := octant.And(trig512_32_intOne).Equal(trig512_32_intOne)
	// Determine if we should negate the result
	negateMask := octant.And(trig512_32_intTwo).Equal(trig512_32_intTwo)

	// For sin: select sin(r) or cos(r), then negate if needed
	// Use bit-level reinterpretation to work with int masks
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	sinRBits := sinR.AsInt32x16()
	cosRBits := cosR.AsInt32x16()
	sinResultBits := cosRBits.Merge(sinRBits, useCosMask)
	sinResult := sinResultBits.AsFloat32x16()
	negSinResult := trig512_32_zero.Sub(sinResult)
	negSinResultBits := negSinResult.AsInt32x16()
	sinBits := negSinResultBits.Merge(sinResultBits, negateMask)
	sin = sinBits.AsFloat32x16()

	// For cos: it's shifted by 1 quadrant from sin
	// cos(x) = sin(x + π/2), so we use k+1 for cos
	cosOctant := octant.Add(trig512_32_intOne).And(trig512_32_intThree)
	useCosForCosMask := cosOctant.And(trig512_32_intOne).Equal(trig512_32_intOne)
	negateCosMask := cosOctant.And(trig512_32_intTwo).Equal(trig512_32_intTwo)

	cosResultBits := cosRBits.Merge(sinRBits, useCosForCosMask)
	cosResult := cosResultBits.AsFloat32x16()
	negCosResult := trig512_32_zero.Sub(cosResult)
	negCosResultBits := negCosResult.AsInt32x16()
	cosBits := negCosResultBits.Merge(cosResultBits, negateCosMask)
	cos = cosBits.AsFloat32x16()

	// Handle special cases: ±Inf and NaN -> NaN
	infMask := origX.Equal(trig512_32_inf).Or(origX.Equal(trig512_32_negInf))
	sin = trig512_32_nan.Merge(sin, infMask)
	cos = trig512_32_nan.Merge(cos, infMask)

	return sin, cos
}

// Sin_AVX512_F64x8 computes sin(x) for a single Float64x8 vector.
func Sin_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	sin, _ := sinCos512_64Core(x)
	return sin
}

// sinCos512_64Core computes both sin and cos for Float64x8.
// This is the shared implementation used by Sin, Cos, and SinCos.
func sinCos512_64Core(x archsimd.Float64x8) (sin, cos archsimd.Float64x8) {
	// Save input for special case handling
	origX := x

	// Range reduction: k = round(x * 2/π)
	k := x.Mul(trig512_64_2overPi).RoundToEvenScaled(0)
	kInt := k.ConvertToInt64()

	// r = x - k * (π/2) using Cody-Waite high/low for precision
	r := x.Sub(k.Mul(trig512_64_piOver2Hi))
	r = r.Sub(k.Mul(trig512_64_piOver2Lo))
	r2 := r.Mul(r)

	// Compute sin(r) polynomial (higher degree for float64)
	sinPoly := trig512_64_s6.MulAdd(r2, trig512_64_s5)
	sinPoly = sinPoly.MulAdd(r2, trig512_64_s4)
	sinPoly = sinPoly.MulAdd(r2, trig512_64_s3)
	sinPoly = sinPoly.MulAdd(r2, trig512_64_s2)
	sinPoly = sinPoly.MulAdd(r2, trig512_64_s1)
	sinPoly = sinPoly.MulAdd(r2, trig512_64_one)
	sinR := r.Mul(sinPoly)

	// Compute cos(r) polynomial
	cosPoly := trig512_64_c6.MulAdd(r2, trig512_64_c5)
	cosPoly = cosPoly.MulAdd(r2, trig512_64_c4)
	cosPoly = cosPoly.MulAdd(r2, trig512_64_c3)
	cosPoly = cosPoly.MulAdd(r2, trig512_64_c2)
	cosPoly = cosPoly.MulAdd(r2, trig512_64_c1)
	cosR := cosPoly.MulAdd(r2, trig512_64_one)

	// Octant selection
	octant := kInt.And(trig512_64_intThree)
	useCosMask := octant.And(trig512_64_intOne).Equal(trig512_64_intOne)
	negateMask := octant.And(trig512_64_intTwo).Equal(trig512_64_intTwo)

	// For sin: use bit-level reinterpretation to work with int masks
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	sinRBits := sinR.AsInt64x8()
	cosRBits := cosR.AsInt64x8()
	sinResultBits := cosRBits.Merge(sinRBits, useCosMask)
	sinResult := sinResultBits.AsFloat64x8()
	negSinResult := trig512_64_zero.Sub(sinResult)
	negSinResultBits := negSinResult.AsInt64x8()
	sinBits := negSinResultBits.Merge(sinResultBits, negateMask)
	sin = sinBits.AsFloat64x8()

	// For cos (shifted by 1 quadrant)
	cosOctant := octant.Add(trig512_64_intOne).And(trig512_64_intThree)
	useCosForCosMask := cosOctant.And(trig512_64_intOne).Equal(trig512_64_intOne)
	negateCosMask := cosOctant.And(trig512_64_intTwo).Equal(trig512_64_intTwo)

	cosResultBits := cosRBits.Merge(sinRBits, useCosForCosMask)
	cosResult := cosResultBits.AsFloat64x8()
	negCosResult := trig512_64_zero.Sub(cosResult)
	negCosResultBits := negCosResult.AsInt64x8()
	cosBits := negCosResultBits.Merge(cosResultBits, negateCosMask)
	cos = cosBits.AsFloat64x8()

	// Handle special cases: ±Inf -> NaN
	infMask := origX.Equal(trig512_64_inf).Or(origX.Equal(archsimd.BroadcastFloat64x8(math.Inf(-1))))
	sin = trig512_64_nan.Merge(sin, infMask)
	cos = trig512_64_nan.Merge(cos, infMask)

	return sin, cos
}

// Cos_AVX512_F32x16 computes cos(x) for a single Float32x16 vector.
func Cos_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	_, cos := sinCos512_32Core(x)
	return cos
}

// Cos_AVX512_F64x8 computes cos(x) for a single Float64x8 vector.
func Cos_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	_, cos := sinCos512_64Core(x)
	return cos
}

// SinCos_AVX512_F32x16 computes both sin and cos for a single Float32x16 vector.
// This is more efficient than calling Sin and Cos separately as it shares
// the range reduction computation.
func SinCos_AVX512_F32x16(x archsimd.Float32x16) (sin, cos archsimd.Float32x16) {
	return sinCos512_32Core(x)
}

// SinCos_AVX512_F64x8 computes both sin and cos for a single Float64x8 vector.
// This is more efficient than calling Sin and Cos separately as it shares
// the range reduction computation.
func SinCos_AVX512_F64x8(x archsimd.Float64x8) (sin, cos archsimd.Float64x8) {
	return sinCos512_64Core(x)
}
