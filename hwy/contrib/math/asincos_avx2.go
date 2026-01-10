//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"

	"github.com/ajroetker/go-highway/hwy"
)

// AVX2 vectorized constants for asin/acos
var (
	// Common constants
	asin32_zero     = archsimd.BroadcastFloat32x8(0.0)
	asin32_one      = archsimd.BroadcastFloat32x8(1.0)
	asin32_negOne   = archsimd.BroadcastFloat32x8(-1.0)
	asin32_half     = archsimd.BroadcastFloat32x8(0.5)
	asin32_two      = archsimd.BroadcastFloat32x8(2.0)
	asin32_piOver2  = archsimd.BroadcastFloat32x8(1.5707963267948966)  // pi/2
	asin32_pi       = archsimd.BroadcastFloat32x8(3.141592653589793)   // pi
	asin32_nan      = archsimd.BroadcastFloat32x8(float32(stdmath.NaN()))

	// Polynomial coefficients for asin(x) approximation on |x| < 0.5
	// asin(x) ≈ x + x³*(p1 + x²*(p2 + x²*(p3 + x²*(p4 + x²*(p5 + x²*p6)))))
	// Taylor series coefficients for better accuracy at boundary
	asin32_p1 = archsimd.BroadcastFloat32x8(0.16666666666666666)   // 1/6
	asin32_p2 = archsimd.BroadcastFloat32x8(0.075)                 // 3/40
	asin32_p3 = archsimd.BroadcastFloat32x8(0.04464285714285714)   // 15/336
	asin32_p4 = archsimd.BroadcastFloat32x8(0.030381944444444444)  // 35/1152
	asin32_p5 = archsimd.BroadcastFloat32x8(0.022372159090909092)  // 63/2816
	asin32_p6 = archsimd.BroadcastFloat32x8(0.017352764423076923)  // 231/13312

	// Float64 constants
	asin64_zero     = archsimd.BroadcastFloat64x4(0.0)
	asin64_one      = archsimd.BroadcastFloat64x4(1.0)
	asin64_negOne   = archsimd.BroadcastFloat64x4(-1.0)
	asin64_half     = archsimd.BroadcastFloat64x4(0.5)
	asin64_two      = archsimd.BroadcastFloat64x4(2.0)
	asin64_piOver2  = archsimd.BroadcastFloat64x4(1.5707963267948966192313216916398)
	asin64_pi       = archsimd.BroadcastFloat64x4(3.1415926535897932384626433832795)
	asin64_nan      = archsimd.BroadcastFloat64x4(stdmath.NaN())
)

// Asin_AVX2_F32x8 computes asin(x) for a single Float32x8 vector.
//
// Algorithm:
// For |x| < 0.5: use Taylor series x + x³/6 + 3x⁵/40 + 15x⁷/336 + ...
// For |x| >= 0.5: use asin(x) = pi/2 - 2*asin(sqrt((1-x)/2)) for x > 0
//                      asin(x) = -pi/2 + 2*asin(sqrt((1+x)/2)) for x < 0
// For |x| > 1: return NaN
//
// Special cases:
//   - Asin(±0) = ±0
//   - Asin(±1) = ±pi/2
//   - Asin(x) = NaN if |x| > 1
func Asin_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Save original for sign handling
	origX := x

	// Compute |x|
	absX := x.Max(asin32_zero.Sub(x))

	// Check for out of range: |x| > 1
	outOfRangeMask := absX.Greater(asin32_one)

	// Check if we need the large argument formula: |x| >= 0.5
	// Use Greater OR Equal since GreaterOrEqual doesn't exist
	largeMask := absX.Greater(asin32_half).Or(absX.Equal(asin32_half))

	// ===== Small argument path: |x| < 0.5 =====
	// asin(x) ≈ x + x³*(p1 + x²*(p2 + x²*(p3 + x²*(p4 + x²*(p5 + x²*p6)))))
	x2Small := x.Mul(x)
	poly := asin32_p6.MulAdd(x2Small, asin32_p5)
	poly = poly.MulAdd(x2Small, asin32_p4)
	poly = poly.MulAdd(x2Small, asin32_p3)
	poly = poly.MulAdd(x2Small, asin32_p2)
	poly = poly.MulAdd(x2Small, asin32_p1)
	smallResult := x.Add(x.Mul(x2Small).Mul(poly))

	// ===== Large argument path: |x| >= 0.5 =====
	// For x >= 0.5: asin(x) = pi/2 - 2*asin(sqrt((1-x)/2))
	// For x <= -0.5: asin(x) = -pi/2 + 2*asin(sqrt((1+x)/2))

	// Compute sqrt((1 - |x|) / 2)
	oneMinusAbsX := asin32_one.Sub(absX)
	halfOneMinusAbsX := oneMinusAbsX.Mul(asin32_half)
	sqrtArg := hwy.Sqrt_AVX2_F32x8(halfOneMinusAbsX)

	// Apply polynomial to sqrtArg
	sqrtArg2 := sqrtArg.Mul(sqrtArg)
	polyLarge := asin32_p6.MulAdd(sqrtArg2, asin32_p5)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin32_p4)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin32_p3)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin32_p2)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin32_p1)
	asinSqrtArg := sqrtArg.Add(sqrtArg.Mul(sqrtArg2).Mul(polyLarge))

	// asin(|x|) = pi/2 - 2*asin(sqrt((1-|x|)/2))
	largeResultPos := asin32_piOver2.Sub(asin32_two.Mul(asinSqrtArg))

	// Select based on original sign
	// If x was negative, result = -largeResultPos
	signMask := origX.Less(asin32_zero)
	negLargeResult := asin32_zero.Sub(largeResultPos)
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	largeResultBits := negLargeResult.AsInt32x8()
	largeResultPosBits := largeResultPos.AsInt32x8()
	signMaskInt := signMask
	largeResultFinalBits := largeResultBits.Merge(largeResultPosBits, signMaskInt)
	largeResult := largeResultFinalBits.AsFloat32x8()

	// Select between small and large paths
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	smallResultBits := smallResult.AsInt32x8()
	largeResultBits2 := largeResult.AsInt32x8()
	largeMaskInt := largeMask
	resultBits := largeResultBits2.Merge(smallResultBits, largeMaskInt)
	result := resultBits.AsFloat32x8()

	// Handle out of range: return NaN for |x| > 1
	result = asin32_nan.Merge(result, outOfRangeMask)

	return result
}

// Asin_AVX2_F64x4 computes asin(x) for a single Float64x4 vector.
// Uses scalar fallback for float64 due to AVX2 limitations with 64-bit operations.
func Asin_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	var in, out [4]float64
	x.StoreSlice(in[:])
	for i := range in {
		out[i] = stdmath.Asin(in[i])
	}
	return archsimd.LoadFloat64x4Slice(out[:])
}

// Acos_AVX2_F32x8 computes acos(x) for a single Float32x8 vector.
//
// Algorithm: acos(x) = pi/2 - asin(x)
//
// Special cases:
//   - Acos(1) = 0
//   - Acos(-1) = pi
//   - Acos(0) = pi/2
//   - Acos(x) = NaN if |x| > 1
func Acos_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Compute |x|
	absX := x.Max(asin32_zero.Sub(x))

	// Check for out of range: |x| > 1
	outOfRangeMask := absX.Greater(asin32_one)

	// acos(x) = pi/2 - asin(x)
	asinX := Asin_AVX2_F32x8(x)
	result := asin32_piOver2.Sub(asinX)

	// Handle out of range: return NaN for |x| > 1
	result = asin32_nan.Merge(result, outOfRangeMask)

	return result
}

// Acos_AVX2_F64x4 computes acos(x) for a single Float64x4 vector.
// Uses scalar fallback for float64 due to AVX2 limitations with 64-bit operations.
func Acos_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	var in, out [4]float64
	x.StoreSlice(in[:])
	for i := range in {
		out[i] = stdmath.Acos(in[i])
	}
	return archsimd.LoadFloat64x4Slice(out[:])
}
