//go:build amd64 && goexperiment.simd

package contrib

import (
	"math"
	"simd/archsimd"
)

// AVX2 vectorized constants for trig32
var (
	// Range reduction constants (Cody-Waite)
	// Using 2/π for reduction to [-π/4, π/4] with π/2 intervals
	trig32_2overPi   = archsimd.BroadcastFloat32x8(0.6366197723675814)     // 2/π
	trig32_piOver2Hi = archsimd.BroadcastFloat32x8(1.5707963267948966)     // π/2 high
	trig32_piOver2Lo = archsimd.BroadcastFloat32x8(6.123233995736766e-17)  // π/2 low

	// sin(x) polynomial coefficients for |x| <= π/4
	// sin(x) ≈ x * (1 + s1*x² + s2*x⁴ + s3*x⁶ + s4*x⁸)
	trig32_s1 = archsimd.BroadcastFloat32x8(-0.16666666641626524)     // -1/3!
	trig32_s2 = archsimd.BroadcastFloat32x8(0.008333329385889463)     // 1/5!
	trig32_s3 = archsimd.BroadcastFloat32x8(-0.00019839334836096632)  // -1/7!
	trig32_s4 = archsimd.BroadcastFloat32x8(2.718311493989822e-6)     // 1/9!

	// cos(x) polynomial coefficients for |x| <= π/4
	// cos(x) ≈ 1 + c1*x² + c2*x⁴ + c3*x⁶ + c4*x⁸
	trig32_c1 = archsimd.BroadcastFloat32x8(-0.4999999963229337)      // -1/2!
	trig32_c2 = archsimd.BroadcastFloat32x8(0.04166662453689337)      // 1/4!
	trig32_c3 = archsimd.BroadcastFloat32x8(-0.001388731625493765)    // -1/6!
	trig32_c4 = archsimd.BroadcastFloat32x8(2.443315711809948e-5)     // 1/8!

	// Constants
	trig32_zero   = archsimd.BroadcastFloat32x8(0.0)
	trig32_one    = archsimd.BroadcastFloat32x8(1.0)
	trig32_negOne = archsimd.BroadcastFloat32x8(-1.0)
	trig32_nan    = archsimd.BroadcastFloat32x8(float32(math.NaN()))
	trig32_inf    = archsimd.BroadcastFloat32x8(float32(math.Inf(1)))
	trig32_negInf = archsimd.BroadcastFloat32x8(float32(math.Inf(-1)))

	// Integer constants for octant selection
	trig32_intOne   = archsimd.BroadcastInt32x8(1)
	trig32_intTwo   = archsimd.BroadcastInt32x8(2)
	trig32_intThree = archsimd.BroadcastInt32x8(3)
	trig32_intZero  = archsimd.BroadcastInt32x8(0)
)

// AVX2 vectorized constants for trig64
var (
	trig64_2overPi   = archsimd.BroadcastFloat64x4(0.6366197723675814)
	trig64_piOver2Hi = archsimd.BroadcastFloat64x4(1.5707963267948966192313216916398)
	trig64_piOver2Lo = archsimd.BroadcastFloat64x4(6.123233995736766035868820147292e-17)

	// Higher-degree polynomials for float64
	trig64_s1 = archsimd.BroadcastFloat64x4(-0.16666666666666632)
	trig64_s2 = archsimd.BroadcastFloat64x4(0.008333333333332249)
	trig64_s3 = archsimd.BroadcastFloat64x4(-0.00019841269840885721)
	trig64_s4 = archsimd.BroadcastFloat64x4(2.7557316103728803e-6)
	trig64_s5 = archsimd.BroadcastFloat64x4(-2.5051132068021698e-8)
	trig64_s6 = archsimd.BroadcastFloat64x4(1.5896230157221844e-10)

	trig64_c1 = archsimd.BroadcastFloat64x4(-0.5)
	trig64_c2 = archsimd.BroadcastFloat64x4(0.04166666666666621)
	trig64_c3 = archsimd.BroadcastFloat64x4(-0.001388888888887411)
	trig64_c4 = archsimd.BroadcastFloat64x4(2.4801587288851704e-5)
	trig64_c5 = archsimd.BroadcastFloat64x4(-2.7557314351390663e-7)
	trig64_c6 = archsimd.BroadcastFloat64x4(2.0875723212981748e-9)

	trig64_zero   = archsimd.BroadcastFloat64x4(0.0)
	trig64_one    = archsimd.BroadcastFloat64x4(1.0)
	trig64_negOne = archsimd.BroadcastFloat64x4(-1.0)
	trig64_nan    = archsimd.BroadcastFloat64x4(math.NaN())
	trig64_inf    = archsimd.BroadcastFloat64x4(math.Inf(1))

	trig64_intOne   = archsimd.BroadcastInt64x4(1)
	trig64_intTwo   = archsimd.BroadcastInt64x4(2)
	trig64_intThree = archsimd.BroadcastInt64x4(3)
)

// Sin_AVX2_F32x8 computes sin(x) for a single Float32x8 vector.
//
// Algorithm:
// 1. Range reduction: k = round(x * 2/π), r = x - k*(π/2)
// 2. Compute sin(r) and cos(r) polynomials
// 3. Select based on quadrant: k mod 4
//   - 0: sin(r)
//   - 1: cos(r)
//   - 2: -sin(r)
//   - 3: -cos(r)
func Sin_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	sin, _ := sinCos32Core(x)
	return sin
}

// sinCos32Core computes both sin and cos for Float32x8.
// This is the shared implementation used by Sin, Cos, and SinCos.
func sinCos32Core(x archsimd.Float32x8) (sin, cos archsimd.Float32x8) {
	// Save input for special case handling
	origX := x

	// Range reduction: k = round(x * 2/π)
	// This gives us the quadrant (0-3)
	k := x.Mul(trig32_2overPi).RoundToEven()
	kInt := k.ConvertToInt32()

	// r = x - k * (π/2) using Cody-Waite high/low for precision
	r := x.Sub(k.Mul(trig32_piOver2Hi))
	r = r.Sub(k.Mul(trig32_piOver2Lo))
	r2 := r.Mul(r)

	// Compute sin(r) polynomial: sin(r) ≈ r * (1 + s1*r² + s2*r⁴ + s3*r⁶ + s4*r⁸)
	sinPoly := trig32_s4.MulAdd(r2, trig32_s3)
	sinPoly = sinPoly.MulAdd(r2, trig32_s2)
	sinPoly = sinPoly.MulAdd(r2, trig32_s1)
	sinPoly = sinPoly.MulAdd(r2, trig32_one)
	sinR := r.Mul(sinPoly)

	// Compute cos(r) polynomial: cos(r) ≈ 1 + c1*r² + c2*r⁴ + c3*r⁶ + c4*r⁸
	cosPoly := trig32_c4.MulAdd(r2, trig32_c3)
	cosPoly = cosPoly.MulAdd(r2, trig32_c2)
	cosPoly = cosPoly.MulAdd(r2, trig32_c1)
	cosR := cosPoly.MulAdd(r2, trig32_one)

	// Octant selection based on k mod 4
	// For sin(x): quadrant determines which polynomial and sign
	//   k%4 == 0: sin(r)
	//   k%4 == 1: cos(r)
	//   k%4 == 2: -sin(r)
	//   k%4 == 3: -cos(r)
	octant := kInt.And(trig32_intThree)

	// Determine if we should use cos instead of sin
	useCosMask := octant.And(trig32_intOne).Equal(trig32_intOne)
	// Determine if we should negate the result
	negateMask := octant.And(trig32_intTwo).Equal(trig32_intTwo)

	// For sin: select sin(r) or cos(r), then negate if needed
	// Use bit-level reinterpretation to work with int masks
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	sinRBits := sinR.AsInt32x8()
	cosRBits := cosR.AsInt32x8()
	sinResultBits := cosRBits.Merge(sinRBits, useCosMask)
	sinResult := sinResultBits.AsFloat32x8()
	negSinResult := trig32_zero.Sub(sinResult)
	negSinResultBits := negSinResult.AsInt32x8()
	sinBits := negSinResultBits.Merge(sinResultBits, negateMask)
	sin = sinBits.AsFloat32x8()

	// For cos: it's shifted by 1 quadrant from sin
	// cos(x) = sin(x + π/2), so we use k+1 for cos
	cosOctant := octant.Add(trig32_intOne).And(trig32_intThree)
	useCosForCosMask := cosOctant.And(trig32_intOne).Equal(trig32_intOne)
	negateCosMask := cosOctant.And(trig32_intTwo).Equal(trig32_intTwo)

	cosResultBits := cosRBits.Merge(sinRBits, useCosForCosMask)
	cosResult := cosResultBits.AsFloat32x8()
	negCosResult := trig32_zero.Sub(cosResult)
	negCosResultBits := negCosResult.AsInt32x8()
	cosBits := negCosResultBits.Merge(cosResultBits, negateCosMask)
	cos = cosBits.AsFloat32x8()

	// Handle special cases: ±Inf and NaN -> NaN
	infMask := origX.Equal(trig32_inf).Or(origX.Equal(trig32_negInf))
	sin = trig32_nan.Merge(sin, infMask)
	cos = trig32_nan.Merge(cos, infMask)

	return sin, cos
}

// Sin_AVX2_F64x4 computes sin(x) for a single Float64x4 vector.
func Sin_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	sin, _ := sinCos64Core(x)
	return sin
}

// sinCos64Core computes both sin and cos for Float64x4.
// This is the shared implementation used by Sin, Cos, and SinCos.
func sinCos64Core(x archsimd.Float64x4) (sin, cos archsimd.Float64x4) {
	// Save input for special case handling
	origX := x

	// Range reduction: k = round(x * 2/π)
	k := x.Mul(trig64_2overPi).RoundToEven()

	// Note: AVX2 lacks vcvttpd2qq (float64→int64), so we use scalar conversion.
	var kBuf [4]float64
	k.StoreSlice(kBuf[:])
	kIntBuf := [4]int64{int64(kBuf[0]), int64(kBuf[1]), int64(kBuf[2]), int64(kBuf[3])}
	kInt := archsimd.LoadInt64x4Slice(kIntBuf[:])

	// r = x - k * (π/2) using Cody-Waite high/low for precision
	r := x.Sub(k.Mul(trig64_piOver2Hi))
	r = r.Sub(k.Mul(trig64_piOver2Lo))
	r2 := r.Mul(r)

	// Compute sin(r) polynomial (higher degree for float64)
	sinPoly := trig64_s6.MulAdd(r2, trig64_s5)
	sinPoly = sinPoly.MulAdd(r2, trig64_s4)
	sinPoly = sinPoly.MulAdd(r2, trig64_s3)
	sinPoly = sinPoly.MulAdd(r2, trig64_s2)
	sinPoly = sinPoly.MulAdd(r2, trig64_s1)
	sinPoly = sinPoly.MulAdd(r2, trig64_one)
	sinR := r.Mul(sinPoly)

	// Compute cos(r) polynomial
	cosPoly := trig64_c6.MulAdd(r2, trig64_c5)
	cosPoly = cosPoly.MulAdd(r2, trig64_c4)
	cosPoly = cosPoly.MulAdd(r2, trig64_c3)
	cosPoly = cosPoly.MulAdd(r2, trig64_c2)
	cosPoly = cosPoly.MulAdd(r2, trig64_c1)
	cosR := cosPoly.MulAdd(r2, trig64_one)

	// Octant selection
	octant := kInt.And(trig64_intThree)
	useCosMask := octant.And(trig64_intOne).Equal(trig64_intOne)
	negateMask := octant.And(trig64_intTwo).Equal(trig64_intTwo)

	// For sin: use bit-level reinterpretation to work with int masks
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	sinRBits := sinR.AsInt64x4()
	cosRBits := cosR.AsInt64x4()
	sinResultBits := cosRBits.Merge(sinRBits, useCosMask)
	sinResult := sinResultBits.AsFloat64x4()
	negSinResult := trig64_zero.Sub(sinResult)
	negSinResultBits := negSinResult.AsInt64x4()
	sinBits := negSinResultBits.Merge(sinResultBits, negateMask)
	sin = sinBits.AsFloat64x4()

	// For cos (shifted by 1 quadrant)
	cosOctant := octant.Add(trig64_intOne).And(trig64_intThree)
	useCosForCosMask := cosOctant.And(trig64_intOne).Equal(trig64_intOne)
	negateCosMask := cosOctant.And(trig64_intTwo).Equal(trig64_intTwo)

	cosResultBits := cosRBits.Merge(sinRBits, useCosForCosMask)
	cosResult := cosResultBits.AsFloat64x4()
	negCosResult := trig64_zero.Sub(cosResult)
	negCosResultBits := negCosResult.AsInt64x4()
	cosBits := negCosResultBits.Merge(cosResultBits, negateCosMask)
	cos = cosBits.AsFloat64x4()

	// Handle special cases: ±Inf -> NaN
	infMask := origX.Equal(trig64_inf).Or(origX.Equal(archsimd.BroadcastFloat64x4(math.Inf(-1))))
	sin = trig64_nan.Merge(sin, infMask)
	cos = trig64_nan.Merge(cos, infMask)

	return sin, cos
}

// Cos_AVX2_F32x8 computes cos(x) for a single Float32x8 vector.
func Cos_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	_, cos := sinCos32Core(x)
	return cos
}

// Cos_AVX2_F64x4 computes cos(x) for a single Float64x4 vector.
func Cos_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	_, cos := sinCos64Core(x)
	return cos
}

// SinCos_AVX2_F32x8 computes both sin and cos for a single Float32x8 vector.
// This is more efficient than calling Sin and Cos separately as it shares
// the range reduction computation.
func SinCos_AVX2_F32x8(x archsimd.Float32x8) (sin, cos archsimd.Float32x8) {
	return sinCos32Core(x)
}

// SinCos_AVX2_F64x4 computes both sin and cos for a single Float64x4 vector.
// This is more efficient than calling Sin and Cos separately as it shares
// the range reduction computation.
func SinCos_AVX2_F64x4(x archsimd.Float64x4) (sin, cos archsimd.Float64x4) {
	return sinCos64Core(x)
}
