//go:build amd64 && goexperiment.simd

package contrib

import (
	"simd/archsimd"
)

// AVX-512 vectorized constants for special functions
var (
	// Sigmoid constants
	sig512_32_zero   = archsimd.BroadcastFloat32x16(0.0)
	sig512_32_one    = archsimd.BroadcastFloat32x16(1.0)
	sig512_32_negOne = archsimd.BroadcastFloat32x16(-1.0)
	sig512_32_satHi  = archsimd.BroadcastFloat32x16(20.0)  // sigmoid saturates to 1
	sig512_32_satLo  = archsimd.BroadcastFloat32x16(-20.0) // sigmoid saturates to 0

	sig512_64_zero  = archsimd.BroadcastFloat64x8(0.0)
	sig512_64_one   = archsimd.BroadcastFloat64x8(1.0)
	sig512_64_satHi = archsimd.BroadcastFloat64x8(36.0)
	sig512_64_satLo = archsimd.BroadcastFloat64x8(-36.0)

	// Tanh constants
	tanh512_32_two       = archsimd.BroadcastFloat32x16(2.0)
	tanh512_32_threshold = archsimd.BroadcastFloat32x16(9.0) // tanh saturates beyond this

	tanh512_64_two       = archsimd.BroadcastFloat64x8(2.0)
	tanh512_64_threshold = archsimd.BroadcastFloat64x8(19.0)

	// Erf constants (Abramowitz & Stegun approximation 7.1.26)
	erf512_32_p1 = archsimd.BroadcastFloat32x16(0.254829592)
	erf512_32_p2 = archsimd.BroadcastFloat32x16(-0.284496736)
	erf512_32_p3 = archsimd.BroadcastFloat32x16(1.421413741)
	erf512_32_p4 = archsimd.BroadcastFloat32x16(-1.453152027)
	erf512_32_p5 = archsimd.BroadcastFloat32x16(1.061405429)
	erf512_32_t  = archsimd.BroadcastFloat32x16(0.3275911)

	erf512_64_p1 = archsimd.BroadcastFloat64x8(0.254829592)
	erf512_64_p2 = archsimd.BroadcastFloat64x8(-0.284496736)
	erf512_64_p3 = archsimd.BroadcastFloat64x8(1.421413741)
	erf512_64_p4 = archsimd.BroadcastFloat64x8(-1.453152027)
	erf512_64_p5 = archsimd.BroadcastFloat64x8(1.061405429)
	erf512_64_t  = archsimd.BroadcastFloat64x8(0.3275911)
)

// Tanh_AVX512_F32x16 computes tanh(x) for a single Float32x16 vector.
//
// Algorithm: tanh(x) = 2*sigmoid(2x) - 1
// For large |x|, tanh saturates to ±1.
func Tanh_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	// tanh(x) = 2*sigmoid(2x) - 1
	twoX := tanh512_32_two.Mul(x)
	sigTwoX := Sigmoid_AVX512_F32x16(twoX)
	result := tanh512_32_two.Mul(sigTwoX).Sub(sig512_32_one)

	// Handle saturation (Merge: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	// For x > threshold, tanh ≈ 1; for x < -threshold, tanh ≈ -1
	result = sig512_32_one.Merge(result, x.Greater(tanh512_32_threshold))
	result = sig512_32_negOne.Merge(result, x.Less(sig512_32_zero.Sub(tanh512_32_threshold)))

	return result
}

// Tanh_AVX512_F64x8 computes tanh(x) for a single Float64x8 vector.
func Tanh_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	// tanh(x) = 2*sigmoid(2x) - 1
	twoX := tanh512_64_two.Mul(x)
	sigTwoX := Sigmoid_AVX512_F64x8(twoX)
	result := tanh512_64_two.Mul(sigTwoX).Sub(sig512_64_one)

	// Handle saturation (Merge: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	negThreshold := sig512_64_zero.Sub(tanh512_64_threshold)
	result = sig512_64_one.Merge(result, x.Greater(tanh512_64_threshold))
	result = archsimd.BroadcastFloat64x8(-1.0).Merge(result, x.Less(negThreshold))

	return result
}

// Sigmoid_AVX512_F32x16 computes sigmoid(x) for a single Float32x16 vector.
//
// Algorithm: sigmoid(x) = 1 / (1 + exp(-x))
// For numerical stability, we clamp x to avoid exp overflow.
func Sigmoid_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	// Clamp to avoid exp overflow
	// For x > 20, sigmoid ≈ 1; for x < -20, sigmoid ≈ 0
	clampedX := x.Max(sig512_32_satLo).Min(sig512_32_satHi)

	// Compute exp(-x)
	negX := sig512_32_zero.Sub(clampedX)
	expNegX := Exp_AVX512_F32x16(negX)

	// sigmoid(x) = 1 / (1 + exp(-x))
	result := sig512_32_one.Div(sig512_32_one.Add(expNegX))

	// Handle saturation (Merge: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	result = sig512_32_one.Merge(result, x.Greater(sig512_32_satHi))
	result = sig512_32_zero.Merge(result, x.Less(sig512_32_satLo))

	return result
}

// Sigmoid_AVX512_F64x8 computes sigmoid(x) for a single Float64x8 vector.
func Sigmoid_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	// Clamp to avoid exp overflow
	clampedX := x.Max(sig512_64_satLo).Min(sig512_64_satHi)

	// Compute exp(-x)
	negX := sig512_64_zero.Sub(clampedX)
	expNegX := Exp_AVX512_F64x8(negX)

	// sigmoid(x) = 1 / (1 + exp(-x))
	result := sig512_64_one.Div(sig512_64_one.Add(expNegX))

	// Handle saturation (Merge: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	result = sig512_64_one.Merge(result, x.Greater(sig512_64_satHi))
	result = sig512_64_zero.Merge(result, x.Less(sig512_64_satLo))

	return result
}

// Erf_AVX512_F32x16 computes erf(x) for a single Float32x16 vector.
//
// Algorithm: Abramowitz & Stegun approximation 7.1.26
// erf(x) ≈ 1 - (p1*t + p2*t² + p3*t³ + p4*t⁴ + p5*t⁵) * exp(-x²)
// where t = 1 / (1 + 0.3275911*|x|)
// This has a maximum error of 1.5×10⁻⁷
func Erf_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	// Handle sign: erf(-x) = -erf(x)
	signMask := x.Less(sig512_32_zero)
	absX := x.Max(sig512_32_zero.Sub(x)) // abs(x) = max(x, -x)

	// t = 1 / (1 + p*|x|)
	t := sig512_32_one.Div(sig512_32_one.Add(erf512_32_t.Mul(absX)))

	// Polynomial: p5*t⁵ + p4*t⁴ + p3*t³ + p2*t² + p1*t
	// Using Horner's method
	poly := erf512_32_p5.MulAdd(t, erf512_32_p4)
	poly = poly.MulAdd(t, erf512_32_p3)
	poly = poly.MulAdd(t, erf512_32_p2)
	poly = poly.MulAdd(t, erf512_32_p1)
	poly = poly.Mul(t)

	// exp(-x²)
	negX2 := sig512_32_zero.Sub(absX.Mul(absX))
	expNegX2 := Exp_AVX512_F32x16(negX2)

	// erf(|x|) = 1 - poly * exp(-x²)
	result := sig512_32_one.Sub(poly.Mul(expNegX2))

	// Apply sign: erf(-x) = -erf(x) (Merge: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	negResult := sig512_32_zero.Sub(result)
	result = negResult.Merge(result, signMask)

	return result
}

// Erf_AVX512_F64x8 computes erf(x) for a single Float64x8 vector.
func Erf_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	// Handle sign: erf(-x) = -erf(x)
	signMask := x.Less(sig512_64_zero)
	absX := x.Max(sig512_64_zero.Sub(x)) // abs(x) = max(x, -x)

	// t = 1 / (1 + p*|x|)
	t := sig512_64_one.Div(sig512_64_one.Add(erf512_64_t.Mul(absX)))

	// Polynomial using Horner's method
	poly := erf512_64_p5.MulAdd(t, erf512_64_p4)
	poly = poly.MulAdd(t, erf512_64_p3)
	poly = poly.MulAdd(t, erf512_64_p2)
	poly = poly.MulAdd(t, erf512_64_p1)
	poly = poly.Mul(t)

	// exp(-x²)
	negX2 := sig512_64_zero.Sub(absX.Mul(absX))
	expNegX2 := Exp_AVX512_F64x8(negX2)

	// erf(|x|) = 1 - poly * exp(-x²)
	result := sig512_64_one.Sub(poly.Mul(expNegX2))

	// Apply sign (Merge: a.Merge(b, mask) returns a when TRUE, b when FALSE)
	negResult := sig512_64_zero.Sub(result)
	result = negResult.Merge(result, signMask)

	return result
}
