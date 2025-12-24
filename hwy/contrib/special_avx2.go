//go:build amd64 && goexperiment.simd

package contrib

import (
	"simd/archsimd"

	"github.com/ajroetker/go-highway/hwy"
)

// AVX2 vectorized constants for special functions
var (
	// Sigmoid constants
	sig32_zero      = archsimd.BroadcastFloat32x8(0.0)
	sig32_one       = archsimd.BroadcastFloat32x8(1.0)
	sig32_negOne    = archsimd.BroadcastFloat32x8(-1.0)
	sig32_satHi     = archsimd.BroadcastFloat32x8(20.0)  // sigmoid saturates to 1
	sig32_satLo     = archsimd.BroadcastFloat32x8(-20.0) // sigmoid saturates to 0

	sig64_zero   = archsimd.BroadcastFloat64x4(0.0)
	sig64_one    = archsimd.BroadcastFloat64x4(1.0)
	sig64_satHi  = archsimd.BroadcastFloat64x4(36.0)
	sig64_satLo  = archsimd.BroadcastFloat64x4(-36.0)

	// Tanh constants
	tanh32_two       = archsimd.BroadcastFloat32x8(2.0)
	tanh32_threshold = archsimd.BroadcastFloat32x8(9.0) // tanh saturates beyond this

	tanh64_two       = archsimd.BroadcastFloat64x4(2.0)
	tanh64_threshold = archsimd.BroadcastFloat64x4(19.0)

	// Erf constants (Abramowitz & Stegun approximation 7.1.26)
	erf32_p1 = archsimd.BroadcastFloat32x8(0.254829592)
	erf32_p2 = archsimd.BroadcastFloat32x8(-0.284496736)
	erf32_p3 = archsimd.BroadcastFloat32x8(1.421413741)
	erf32_p4 = archsimd.BroadcastFloat32x8(-1.453152027)
	erf32_p5 = archsimd.BroadcastFloat32x8(1.061405429)
	erf32_t  = archsimd.BroadcastFloat32x8(0.3275911)

	erf64_p1 = archsimd.BroadcastFloat64x4(0.254829592)
	erf64_p2 = archsimd.BroadcastFloat64x4(-0.284496736)
	erf64_p3 = archsimd.BroadcastFloat64x4(1.421413741)
	erf64_p4 = archsimd.BroadcastFloat64x4(-1.453152027)
	erf64_p5 = archsimd.BroadcastFloat64x4(1.061405429)
	erf64_t  = archsimd.BroadcastFloat64x4(0.3275911)
)

func init() {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		Tanh32 = tanh32AVX2
		Tanh64 = tanh64AVX2
		Sigmoid32 = sigmoid32AVX2
		Sigmoid64 = sigmoid64AVX2
		Erf32 = erf32AVX2
		Erf64 = erf64AVX2
	}
}

// tanh32AVX2 computes tanh(x) for float32 values using AVX2 SIMD.
func tanh32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Tanh_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		result[i] = tanh32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Tanh_AVX2_F32x8 computes tanh(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
//
// Algorithm: tanh(x) = 2*sigmoid(2x) - 1
// For large |x|, tanh saturates to ±1.
func Tanh_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// tanh(x) = 2*sigmoid(2x) - 1
	twoX := tanh32_two.Mul(x)
	sigTwoX := Sigmoid_AVX2_F32x8(twoX)
	result := tanh32_two.Mul(sigTwoX).Sub(sig32_one)

	// Handle saturation for large |x|
	// For x > threshold, tanh ≈ 1; for x < -threshold, tanh ≈ -1
	result = result.Merge(sig32_one, x.Greater(tanh32_threshold))
	result = result.Merge(sig32_negOne, x.Less(sig32_zero.Sub(tanh32_threshold)))

	return result
}

func tanh32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := tanh32Base(v)
	return result.Data()[0]
}

// tanh64AVX2 computes tanh(x) for float64 values using AVX2 SIMD.
func tanh64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Tanh_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		result[i] = tanh64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Tanh_AVX2_F64x4 computes tanh(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
func Tanh_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	// tanh(x) = 2*sigmoid(2x) - 1
	twoX := tanh64_two.Mul(x)
	sigTwoX := Sigmoid_AVX2_F64x4(twoX)
	result := tanh64_two.Mul(sigTwoX).Sub(sig64_one)

	// Handle saturation
	negThreshold := sig64_zero.Sub(tanh64_threshold)
	result = result.Merge(sig64_one, x.Greater(tanh64_threshold))
	result = result.Merge(archsimd.BroadcastFloat64x4(-1.0), x.Less(negThreshold))

	return result
}

func tanh64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := tanh64Base(v)
	return result.Data()[0]
}

// sigmoid32AVX2 computes sigmoid(x) for float32 values using AVX2 SIMD.
func sigmoid32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Sigmoid_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		result[i] = sigmoid32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Sigmoid_AVX2_F32x8 computes sigmoid(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
//
// Algorithm: sigmoid(x) = 1 / (1 + exp(-x))
// For numerical stability, we clamp x to avoid exp overflow.
func Sigmoid_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Clamp to avoid exp overflow
	// For x > 20, sigmoid ≈ 1; for x < -20, sigmoid ≈ 0
	clampedX := x.Max(sig32_satLo).Min(sig32_satHi)

	// Compute exp(-x)
	negX := sig32_zero.Sub(clampedX)
	expNegX := Exp_AVX2_F32x8(negX)

	// sigmoid(x) = 1 / (1 + exp(-x))
	result := sig32_one.Div(sig32_one.Add(expNegX))

	// Handle saturation
	result = result.Merge(sig32_one, x.Greater(sig32_satHi))
	result = result.Merge(sig32_zero, x.Less(sig32_satLo))

	return result
}

func sigmoid32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := sigmoid32Base(v)
	return result.Data()[0]
}

// sigmoid64AVX2 computes sigmoid(x) for float64 values using AVX2 SIMD.
func sigmoid64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Sigmoid_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		result[i] = sigmoid64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Sigmoid_AVX2_F64x4 computes sigmoid(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
func Sigmoid_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	// Clamp to avoid exp overflow
	clampedX := x.Max(sig64_satLo).Min(sig64_satHi)

	// Compute exp(-x)
	negX := sig64_zero.Sub(clampedX)
	expNegX := Exp_AVX2_F64x4(negX)

	// sigmoid(x) = 1 / (1 + exp(-x))
	result := sig64_one.Div(sig64_one.Add(expNegX))

	// Handle saturation
	result = result.Merge(sig64_one, x.Greater(sig64_satHi))
	result = result.Merge(sig64_zero, x.Less(sig64_satLo))

	return result
}

func sigmoid64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := sigmoid64Base(v)
	return result.Data()[0]
}

// erf32AVX2 computes erf(x) for float32 values using AVX2 SIMD.
func erf32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Erf_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		result[i] = erf32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Erf_AVX2_F32x8 computes erf(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
//
// Algorithm: Abramowitz & Stegun approximation 7.1.26
// erf(x) ≈ 1 - (p1*t + p2*t² + p3*t³ + p4*t⁴ + p5*t⁵) * exp(-x²)
// where t = 1 / (1 + 0.3275911*|x|)
// This has a maximum error of 1.5×10⁻⁷
func Erf_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Handle sign: erf(-x) = -erf(x)
	signMask := x.Less(sig32_zero)
	absX := x.Abs()

	// t = 1 / (1 + p*|x|)
	t := sig32_one.Div(sig32_one.Add(erf32_t.Mul(absX)))

	// Polynomial: p5*t⁵ + p4*t⁴ + p3*t³ + p2*t² + p1*t
	// Using Horner's method
	poly := erf32_p5.MulAdd(t, erf32_p4)
	poly = poly.MulAdd(t, erf32_p3)
	poly = poly.MulAdd(t, erf32_p2)
	poly = poly.MulAdd(t, erf32_p1)
	poly = poly.Mul(t)

	// exp(-x²)
	negX2 := sig32_zero.Sub(absX.Mul(absX))
	expNegX2 := Exp_AVX2_F32x8(negX2)

	// erf(|x|) = 1 - poly * exp(-x²)
	result := sig32_one.Sub(poly.Mul(expNegX2))

	// Apply sign: erf(-x) = -erf(x)
	negResult := sig32_zero.Sub(result)
	result = result.Merge(negResult, signMask)

	return result
}

func erf32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := erf32Base(v)
	return result.Data()[0]
}

// erf64AVX2 computes erf(x) for float64 values using AVX2 SIMD.
func erf64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Erf_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		result[i] = erf64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Erf_AVX2_F64x4 computes erf(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
func Erf_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	// Handle sign: erf(-x) = -erf(x)
	signMask := x.Less(sig64_zero)
	absX := x.Abs()

	// t = 1 / (1 + p*|x|)
	t := sig64_one.Div(sig64_one.Add(erf64_t.Mul(absX)))

	// Polynomial using Horner's method
	poly := erf64_p5.MulAdd(t, erf64_p4)
	poly = poly.MulAdd(t, erf64_p3)
	poly = poly.MulAdd(t, erf64_p2)
	poly = poly.MulAdd(t, erf64_p1)
	poly = poly.Mul(t)

	// exp(-x²)
	negX2 := sig64_zero.Sub(absX.Mul(absX))
	expNegX2 := Exp_AVX2_F64x4(negX2)

	// erf(|x|) = 1 - poly * exp(-x²)
	result := sig64_one.Sub(poly.Mul(expNegX2))

	// Apply sign
	negResult := sig64_zero.Sub(result)
	result = result.Merge(negResult, signMask)

	return result
}

func erf64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := erf64Base(v)
	return result.Data()[0]
}
