//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
	"sync"

	"github.com/ajroetker/go-highway/hwy"
)

// Lazy initialization for AVX-512 constants to avoid executing AVX-512
// instructions at package load time on machines without AVX-512 support.

var asin512Init sync.Once

// AVX-512 vectorized constants for asin/acos
var (
	// Float32x16 constants
	asin512_32_zero     archsimd.Float32x16
	asin512_32_one      archsimd.Float32x16
	asin512_32_negOne   archsimd.Float32x16
	asin512_32_half     archsimd.Float32x16
	asin512_32_two      archsimd.Float32x16
	asin512_32_piOver2  archsimd.Float32x16
	asin512_32_pi       archsimd.Float32x16
	asin512_32_nan      archsimd.Float32x16

	// Polynomial coefficients for asin(x) approximation on |x| < 0.5
	asin512_32_p1 archsimd.Float32x16
	asin512_32_p2 archsimd.Float32x16
	asin512_32_p3 archsimd.Float32x16
	asin512_32_p4 archsimd.Float32x16
	asin512_32_p5 archsimd.Float32x16
	asin512_32_p6 archsimd.Float32x16

	// Float64x8 constants
	asin512_64_zero     archsimd.Float64x8
	asin512_64_one      archsimd.Float64x8
	asin512_64_negOne   archsimd.Float64x8
	asin512_64_half     archsimd.Float64x8
	asin512_64_two      archsimd.Float64x8
	asin512_64_piOver2  archsimd.Float64x8
	asin512_64_pi       archsimd.Float64x8
	asin512_64_nan      archsimd.Float64x8

	// Polynomial coefficients for float64
	asin512_64_p1 archsimd.Float64x8
	asin512_64_p2 archsimd.Float64x8
	asin512_64_p3 archsimd.Float64x8
	asin512_64_p4 archsimd.Float64x8
	asin512_64_p5 archsimd.Float64x8
	asin512_64_p6 archsimd.Float64x8
)

func initAsin512Constants() {
	// Float32x16 constants
	asin512_32_zero = archsimd.BroadcastFloat32x16(0.0)
	asin512_32_one = archsimd.BroadcastFloat32x16(1.0)
	asin512_32_negOne = archsimd.BroadcastFloat32x16(-1.0)
	asin512_32_half = archsimd.BroadcastFloat32x16(0.5)
	asin512_32_two = archsimd.BroadcastFloat32x16(2.0)
	asin512_32_piOver2 = archsimd.BroadcastFloat32x16(1.5707963267948966)
	asin512_32_pi = archsimd.BroadcastFloat32x16(3.141592653589793)
	asin512_32_nan = archsimd.BroadcastFloat32x16(float32(stdmath.NaN()))

	// Polynomial coefficients for asin(x) approximation on |x| < 0.5
	asin512_32_p1 = archsimd.BroadcastFloat32x16(0.16666666666666666)   // 1/6
	asin512_32_p2 = archsimd.BroadcastFloat32x16(0.075)                 // 3/40
	asin512_32_p3 = archsimd.BroadcastFloat32x16(0.04464285714285714)   // 15/336
	asin512_32_p4 = archsimd.BroadcastFloat32x16(0.030381944444444444)  // 35/1152
	asin512_32_p5 = archsimd.BroadcastFloat32x16(0.022372159090909092)  // 63/2816
	asin512_32_p6 = archsimd.BroadcastFloat32x16(0.017352764423076923)  // 231/13312

	// Float64x8 constants
	asin512_64_zero = archsimd.BroadcastFloat64x8(0.0)
	asin512_64_one = archsimd.BroadcastFloat64x8(1.0)
	asin512_64_negOne = archsimd.BroadcastFloat64x8(-1.0)
	asin512_64_half = archsimd.BroadcastFloat64x8(0.5)
	asin512_64_two = archsimd.BroadcastFloat64x8(2.0)
	asin512_64_piOver2 = archsimd.BroadcastFloat64x8(1.5707963267948966192313216916398)
	asin512_64_pi = archsimd.BroadcastFloat64x8(3.1415926535897932384626433832795)
	asin512_64_nan = archsimd.BroadcastFloat64x8(stdmath.NaN())

	// Higher-degree polynomial for float64 accuracy
	asin512_64_p1 = archsimd.BroadcastFloat64x8(0.16666666666666666)     // 1/6
	asin512_64_p2 = archsimd.BroadcastFloat64x8(0.075)                   // 3/40
	asin512_64_p3 = archsimd.BroadcastFloat64x8(0.04464285714285714)     // 15/336
	asin512_64_p4 = archsimd.BroadcastFloat64x8(0.030381944444444444)    // 35/1152
	asin512_64_p5 = archsimd.BroadcastFloat64x8(0.022372159090909092)    // 63/2816
	asin512_64_p6 = archsimd.BroadcastFloat64x8(0.017352764423076923)    // 231/13312
}

// Asin_AVX512_F32x16 computes asin(x) for a single Float32x16 vector.
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
func Asin_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	asin512Init.Do(initAsin512Constants)

	// Save original for sign handling
	origX := x

	// Compute |x|
	absX := x.Max(asin512_32_zero.Sub(x))

	// Check for out of range: |x| > 1
	outOfRangeMask := absX.Greater(asin512_32_one)

	// Check if we need the large argument formula: |x| >= 0.5
	largeMask := absX.Greater(asin512_32_half).Or(absX.Equal(asin512_32_half))

	// ===== Small argument path: |x| < 0.5 =====
	x2Small := x.Mul(x)
	poly := asin512_32_p6.MulAdd(x2Small, asin512_32_p5)
	poly = poly.MulAdd(x2Small, asin512_32_p4)
	poly = poly.MulAdd(x2Small, asin512_32_p3)
	poly = poly.MulAdd(x2Small, asin512_32_p2)
	poly = poly.MulAdd(x2Small, asin512_32_p1)
	smallResult := x.Add(x.Mul(x2Small).Mul(poly))

	// ===== Large argument path: |x| >= 0.5 =====
	// Compute sqrt((1 - |x|) / 2)
	oneMinusAbsX := asin512_32_one.Sub(absX)
	halfOneMinusAbsX := oneMinusAbsX.Mul(asin512_32_half)
	sqrtArg := hwy.Sqrt_AVX512_F32x16(halfOneMinusAbsX)

	// Apply polynomial to sqrtArg
	sqrtArg2 := sqrtArg.Mul(sqrtArg)
	polyLarge := asin512_32_p6.MulAdd(sqrtArg2, asin512_32_p5)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin512_32_p4)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin512_32_p3)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin512_32_p2)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin512_32_p1)
	asinSqrtArg := sqrtArg.Add(sqrtArg.Mul(sqrtArg2).Mul(polyLarge))

	// asin(|x|) = pi/2 - 2*asin(sqrt((1-|x|)/2))
	largeResultPos := asin512_32_piOver2.Sub(asin512_32_two.Mul(asinSqrtArg))

	// Select based on original sign
	signMask := origX.Less(asin512_32_zero)
	negLargeResult := asin512_32_zero.Sub(largeResultPos)
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	largeResultBits := negLargeResult.AsInt32x16()
	largeResultPosBits := largeResultPos.AsInt32x16()
	largeResultFinalBits := largeResultBits.Merge(largeResultPosBits, signMask)
	largeResult := largeResultFinalBits.AsFloat32x16()

	// Select between small and large paths
	smallResultBits := smallResult.AsInt32x16()
	largeResultBits2 := largeResult.AsInt32x16()
	resultBits := largeResultBits2.Merge(smallResultBits, largeMask)
	result := resultBits.AsFloat32x16()

	// Handle out of range: return NaN for |x| > 1
	result = asin512_32_nan.Merge(result, outOfRangeMask)

	return result
}

// Asin_AVX512_F64x8 computes asin(x) for a single Float64x8 vector.
//
// Algorithm: Same as F32x16 but with higher-degree polynomial for better accuracy.
//
// Special cases:
//   - Asin(±0) = ±0
//   - Asin(±1) = ±pi/2
//   - Asin(x) = NaN if |x| > 1
func Asin_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	asin512Init.Do(initAsin512Constants)

	// Save original for sign handling
	origX := x

	// Compute |x|
	absX := x.Max(asin512_64_zero.Sub(x))

	// Check for out of range: |x| > 1
	outOfRangeMask := absX.Greater(asin512_64_one)

	// Check if we need the large argument formula: |x| >= 0.5
	largeMask := absX.Greater(asin512_64_half).Or(absX.Equal(asin512_64_half))

	// ===== Small argument path: |x| < 0.5 =====
	x2Small := x.Mul(x)
	poly := asin512_64_p6.MulAdd(x2Small, asin512_64_p5)
	poly = poly.MulAdd(x2Small, asin512_64_p4)
	poly = poly.MulAdd(x2Small, asin512_64_p3)
	poly = poly.MulAdd(x2Small, asin512_64_p2)
	poly = poly.MulAdd(x2Small, asin512_64_p1)
	smallResult := x.Add(x.Mul(x2Small).Mul(poly))

	// ===== Large argument path: |x| >= 0.5 =====
	// Compute sqrt((1 - |x|) / 2)
	oneMinusAbsX := asin512_64_one.Sub(absX)
	halfOneMinusAbsX := oneMinusAbsX.Mul(asin512_64_half)
	sqrtArg := hwy.Sqrt_AVX512_F64x8(halfOneMinusAbsX)

	// Apply polynomial to sqrtArg (higher degree for float64)
	sqrtArg2 := sqrtArg.Mul(sqrtArg)
	polyLarge := asin512_64_p6.MulAdd(sqrtArg2, asin512_64_p5)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin512_64_p4)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin512_64_p3)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin512_64_p2)
	polyLarge = polyLarge.MulAdd(sqrtArg2, asin512_64_p1)
	asinSqrtArg := sqrtArg.Add(sqrtArg.Mul(sqrtArg2).Mul(polyLarge))

	// asin(|x|) = pi/2 - 2*asin(sqrt((1-|x|)/2))
	largeResultPos := asin512_64_piOver2.Sub(asin512_64_two.Mul(asinSqrtArg))

	// Select based on original sign
	signMask := origX.Less(asin512_64_zero)
	negLargeResult := asin512_64_zero.Sub(largeResultPos)
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	largeResultBits := negLargeResult.AsInt64x8()
	largeResultPosBits := largeResultPos.AsInt64x8()
	largeResultFinalBits := largeResultBits.Merge(largeResultPosBits, signMask)
	largeResult := largeResultFinalBits.AsFloat64x8()

	// Select between small and large paths
	smallResultBits := smallResult.AsInt64x8()
	largeResultBits2 := largeResult.AsInt64x8()
	resultBits := largeResultBits2.Merge(smallResultBits, largeMask)
	result := resultBits.AsFloat64x8()

	// Handle out of range: return NaN for |x| > 1
	result = asin512_64_nan.Merge(result, outOfRangeMask)

	return result
}

// Acos_AVX512_F32x16 computes acos(x) for a single Float32x16 vector.
//
// Algorithm: acos(x) = pi/2 - asin(x)
//
// Special cases:
//   - Acos(1) = 0
//   - Acos(-1) = pi
//   - Acos(0) = pi/2
//   - Acos(x) = NaN if |x| > 1
func Acos_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	asin512Init.Do(initAsin512Constants)

	// Compute |x|
	absX := x.Max(asin512_32_zero.Sub(x))

	// Check for out of range: |x| > 1
	outOfRangeMask := absX.Greater(asin512_32_one)

	// acos(x) = pi/2 - asin(x)
	asinX := Asin_AVX512_F32x16(x)
	result := asin512_32_piOver2.Sub(asinX)

	// Handle out of range: return NaN for |x| > 1
	result = asin512_32_nan.Merge(result, outOfRangeMask)

	return result
}

// Acos_AVX512_F64x8 computes acos(x) for a single Float64x8 vector.
//
// Algorithm: acos(x) = pi/2 - asin(x)
//
// Special cases:
//   - Acos(1) = 0
//   - Acos(-1) = pi
//   - Acos(0) = pi/2
//   - Acos(x) = NaN if |x| > 1
func Acos_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	asin512Init.Do(initAsin512Constants)

	// Compute |x|
	absX := x.Max(asin512_64_zero.Sub(x))

	// Check for out of range: |x| > 1
	outOfRangeMask := absX.Greater(asin512_64_one)

	// acos(x) = pi/2 - asin(x)
	asinX := Asin_AVX512_F64x8(x)
	result := asin512_64_piOver2.Sub(asinX)

	// Handle out of range: return NaN for |x| > 1
	result = asin512_64_nan.Merge(result, outOfRangeMask)

	return result
}
