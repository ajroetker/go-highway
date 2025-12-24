//go:build amd64 && goexperiment.simd

package contrib

import (
	"math"
	"simd/archsimd"

	"github.com/ajroetker/go-highway/hwy"
)

// AVX2 vectorized constants for trig32
var (
	// Range reduction constants (Cody-Waite)
	// Using 4/π for reduction to [-π/4, π/4]
	trig32_4overPi = archsimd.BroadcastFloat32x8(1.27323954473516268615) // 4/π
	trig32_piOver4Hi = archsimd.BroadcastFloat32x8(0.78539816339744830962) // π/4 high
	trig32_piOver4Lo = archsimd.BroadcastFloat32x8(3.061616997868383e-17)  // π/4 low

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
	trig64_4overPi   = archsimd.BroadcastFloat64x4(1.27323954473516268615)
	trig64_piOver4Hi = archsimd.BroadcastFloat64x4(0.7853981633974483096156608)
	trig64_piOver4Lo = archsimd.BroadcastFloat64x4(3.0616169978683830179e-17)

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

func init() {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		Sin32 = sin32AVX2
		Sin64 = sin64AVX2
		Cos32 = cos32AVX2
		Cos64 = cos64AVX2
		SinCos32 = sinCos32AVX2
		SinCos64 = sinCos64AVX2
	}
}

// sin32AVX2 computes sin(x) for float32 values using AVX2 SIMD.
func sin32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Sin_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		result[i] = sin32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Sin_AVX2_F32x8 computes sin(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
//
// Algorithm:
// 1. Range reduction: k = round(x * 4/π), r = x - k*(π/4)
// 2. Compute sin(r) and cos(r) polynomials
// 3. Select based on octant: k mod 4
//    - 0: sin(r)
//    - 1: cos(r)
//    - 2: -sin(r)
//    - 3: -cos(r)
func Sin_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	sin, _ := sinCos32Core(x)
	return sin
}

// sinCos32Core computes both sin and cos for Float32x8.
// This is the shared implementation used by Sin, Cos, and SinCos.
func sinCos32Core(x archsimd.Float32x8) (sin, cos archsimd.Float32x8) {
	// Save input for special case handling
	origX := x

	// Range reduction: k = round(x * 4/π)
	// This gives us the quadrant/octant
	k := x.Mul(trig32_4overPi).RoundToEven()
	kInt := k.ConvertToInt32()

	// r = x - k * (π/4) using Cody-Waite high/low for precision
	r := x.Sub(k.Mul(trig32_piOver4Hi))
	r = r.Sub(k.Mul(trig32_piOver4Lo))
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
	sinResult := sinR.Merge(cosR, useCosMask.AsFloat32x8Mask())
	negSinResult := trig32_zero.Sub(sinResult)
	sin = sinResult.Merge(negSinResult, negateMask.AsFloat32x8Mask())

	// For cos: it's shifted by 1 quadrant from sin
	// cos(x) = sin(x + π/2), so we use k+1 for cos
	cosOctant := octant.Add(trig32_intOne).And(trig32_intThree)
	useCosForCosMask := cosOctant.And(trig32_intOne).Equal(trig32_intOne)
	negateCosMask := cosOctant.And(trig32_intTwo).Equal(trig32_intTwo)

	cosResult := sinR.Merge(cosR, useCosForCosMask.AsFloat32x8Mask())
	negCosResult := trig32_zero.Sub(cosResult)
	cos = cosResult.Merge(negCosResult, negateCosMask.AsFloat32x8Mask())

	// Handle special cases: ±Inf and NaN -> NaN
	infMask := origX.Equal(trig32_inf).Or(origX.Equal(trig32_negInf))
	sin = sin.Merge(trig32_nan, infMask)
	cos = cos.Merge(trig32_nan, infMask)

	return sin, cos
}

func sin32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := sin32Base(v)
	return result.Data()[0]
}

// sin64AVX2 computes sin(x) for float64 values using AVX2 SIMD.
func sin64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Sin_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		result[i] = sin64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Sin_AVX2_F64x4 computes sin(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
func Sin_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	sin, _ := sinCos64Core(x)
	return sin
}

// sinCos64Core computes both sin and cos for Float64x4.
// This is the shared implementation used by Sin, Cos, and SinCos.
func sinCos64Core(x archsimd.Float64x4) (sin, cos archsimd.Float64x4) {
	// Save input for special case handling
	origX := x

	// Range reduction: k = round(x * 4/π)
	k := x.Mul(trig64_4overPi).RoundToEven()
	kInt := k.ConvertToInt64()

	// r = x - k * (π/4) using Cody-Waite high/low for precision
	r := x.Sub(k.Mul(trig64_piOver4Hi))
	r = r.Sub(k.Mul(trig64_piOver4Lo))
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

	// For sin
	sinResult := sinR.Merge(cosR, useCosMask.AsFloat64x4Mask())
	negSinResult := trig64_zero.Sub(sinResult)
	sin = sinResult.Merge(negSinResult, negateMask.AsFloat64x4Mask())

	// For cos (shifted by 1 quadrant)
	cosOctant := octant.Add(trig64_intOne).And(trig64_intThree)
	useCosForCosMask := cosOctant.And(trig64_intOne).Equal(trig64_intOne)
	negateCosMask := cosOctant.And(trig64_intTwo).Equal(trig64_intTwo)

	cosResult := sinR.Merge(cosR, useCosForCosMask.AsFloat64x4Mask())
	negCosResult := trig64_zero.Sub(cosResult)
	cos = cosResult.Merge(negCosResult, negateCosMask.AsFloat64x4Mask())

	// Handle special cases: ±Inf -> NaN
	infMask := origX.Equal(trig64_inf).Or(origX.Equal(archsimd.BroadcastFloat64x4(math.Inf(-1))))
	sin = sin.Merge(trig64_nan, infMask)
	cos = cos.Merge(trig64_nan, infMask)

	return sin, cos
}

func sin64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := sin64Base(v)
	return result.Data()[0]
}

// cos32AVX2 computes cos(x) for float32 values using AVX2 SIMD.
func cos32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Cos_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		result[i] = cos32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Cos_AVX2_F32x8 computes cos(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
func Cos_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	_, cos := sinCos32Core(x)
	return cos
}

func cos32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := cos32Base(v)
	return result.Data()[0]
}

// cos64AVX2 computes cos(x) for float64 values using AVX2 SIMD.
func cos64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Cos_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		result[i] = cos64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Cos_AVX2_F64x4 computes cos(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
func Cos_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	_, cos := sinCos64Core(x)
	return cos
}

func cos64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := cos64Base(v)
	return result.Data()[0]
}

// sinCos32AVX2 computes both sin and cos for float32 values.
func sinCos32AVX2(v hwy.Vec[float32]) (sin, cos hwy.Vec[float32]) {
	data := v.Data()
	n := v.NumLanes()

	sinResult := make([]float32, n)
	cosResult := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		s, c := SinCos_AVX2_F32x8(x)
		s.StoreSlice(sinResult[i:])
		c.StoreSlice(cosResult[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		sinResult[i] = sin32Scalar(data[i])
		cosResult[i] = cos32Scalar(data[i])
	}

	return hwy.Load(sinResult), hwy.Load(cosResult)
}

// SinCos_AVX2_F32x8 computes both sin and cos for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
// This is more efficient than calling Sin and Cos separately as it shares
// the range reduction computation.
func SinCos_AVX2_F32x8(x archsimd.Float32x8) (sin, cos archsimd.Float32x8) {
	return sinCos32Core(x)
}

// sinCos64AVX2 computes both sin and cos for float64 values.
func sinCos64AVX2(v hwy.Vec[float64]) (sin, cos hwy.Vec[float64]) {
	data := v.Data()
	n := v.NumLanes()

	sinResult := make([]float64, n)
	cosResult := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		s, c := SinCos_AVX2_F64x4(x)
		s.StoreSlice(sinResult[i:])
		c.StoreSlice(cosResult[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		sinResult[i] = sin64Scalar(data[i])
		cosResult[i] = cos64Scalar(data[i])
	}

	return hwy.Load(sinResult), hwy.Load(cosResult)
}

// SinCos_AVX2_F64x4 computes both sin and cos for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
// This is more efficient than calling Sin and Cos separately as it shares
// the range reduction computation.
func SinCos_AVX2_F64x4(x archsimd.Float64x4) (sin, cos archsimd.Float64x4) {
	return sinCos64Core(x)
}
