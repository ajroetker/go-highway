//go:build amd64 && goexperiment.simd

package contrib

import (
	"math"
	"simd/archsimd"

	"github.com/ajroetker/go-highway/hwy"
)

// AVX2 vectorized constants for log32
var (
	// Polynomial coefficients for atanh-based log approximation
	// ln((1+y)/(1-y)) = 2y(1 + y²/3 + y⁴/5 + y⁶/7 + ...)
	log32_c1 = archsimd.BroadcastFloat32x8(0.6666666666666735130)  // 2/3
	log32_c2 = archsimd.BroadcastFloat32x8(0.3999999999940941908)  // 2/5
	log32_c3 = archsimd.BroadcastFloat32x8(0.2857142874366239149)  // 2/7
	log32_c4 = archsimd.BroadcastFloat32x8(0.2222219843214978396)  // 2/9
	log32_c5 = archsimd.BroadcastFloat32x8(0.1818357216161805012)  // 2/11

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
	log32_mantMask  = archsimd.BroadcastInt32x8(0x007FFFFF)         // mantissa mask
	log32_expMask   = archsimd.BroadcastInt32x8(0x7F800000)         // exponent mask
	log32_expBias   = archsimd.BroadcastInt32x8(127)                // exponent bias
	log32_normBits  = archsimd.BroadcastInt32x8(0x3F800000)         // 1.0 in IEEE 754
	log32_intOne    = archsimd.BroadcastInt32x8(1)
	log32_intZero   = archsimd.BroadcastInt32x8(0)
)

// AVX2 vectorized constants for log64
var (
	// Higher-degree polynomial for float64
	log64_c1 = archsimd.BroadcastFloat64x4(0.6666666666666735130)
	log64_c2 = archsimd.BroadcastFloat64x4(0.3999999999940941908)
	log64_c3 = archsimd.BroadcastFloat64x4(0.2857142874366239149)
	log64_c4 = archsimd.BroadcastFloat64x4(0.2222219843214978396)
	log64_c5 = archsimd.BroadcastFloat64x4(0.1818357216161805012)
	log64_c6 = archsimd.BroadcastFloat64x4(0.1531383769920937332)
	log64_c7 = archsimd.BroadcastFloat64x4(0.1479819860511658591)

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
	log64_expMask  = archsimd.BroadcastInt64x4(0x7FF0000000000000)
	log64_expBias  = archsimd.BroadcastInt64x4(1023)
	log64_normBits = archsimd.BroadcastInt64x4(0x3FF0000000000000)
	log64_intOne   = archsimd.BroadcastInt64x4(1)
)

func init() {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		Log32 = log32AVX2
		Log64 = log64AVX2
	}
}

// log32AVX2 computes ln(x) for float32 values using AVX2 SIMD.
func log32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	// Process 8 elements at a time
	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Log_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	// Handle tail with scalar fallback
	for i := (n / 8) * 8; i < n; i++ {
		result[i] = log32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Log_AVX2_F32x8 computes ln(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
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
	m = m.Merge(mAdjusted, adjustMask)
	exp = exp.Merge(expAdjusted, adjustMask.AsInt32x8Mask())

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
	expFloat := exp.ConvertToFloat32()
	result := expFloat.Mul(log32_ln2Hi)
	result = result.Add(expFloat.Mul(log32_ln2Lo))
	result = result.Add(lnM)

	// Handle special cases
	// x <= 0: return NaN (log of negative or zero)
	// x == 0: return -Inf
	// x == +Inf: return +Inf
	// x is NaN: return NaN
	zeroMask := origX.Equal(log32_zero)
	negMask := origX.Less(log32_zero)
	infMask := origX.Equal(log32_posInf)

	result = result.Merge(log32_negInf, zeroMask)
	result = result.Merge(log32_nan, negMask)
	result = result.Merge(log32_posInf, infMask)

	return result
}

// log32Scalar is the scalar fallback for tail elements.
func log32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := log32Base(v)
	return result.Data()[0]
}

// log64AVX2 computes ln(x) for float64 values using AVX2 SIMD.
func log64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	// Process 4 elements at a time (Float64x4)
	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Log_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	// Handle tail with scalar fallback
	for i := (n / 4) * 4; i < n; i++ {
		result[i] = log64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Log_AVX2_F64x4 computes ln(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
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
	m = m.Merge(mAdjusted, adjustMask)
	exp = exp.Merge(expAdjusted, adjustMask.AsInt64x4Mask())

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
	expFloat := exp.ConvertToFloat64()
	result := expFloat.Mul(log64_ln2Hi)
	result = result.Add(expFloat.Mul(log64_ln2Lo))
	result = result.Add(lnM)

	// Handle special cases
	zeroMask := origX.Equal(log64_zero)
	negMask := origX.Less(log64_zero)
	infMask := origX.Equal(log64_posInf)

	result = result.Merge(log64_negInf, zeroMask)
	result = result.Merge(log64_nan, negMask)
	result = result.Merge(log64_posInf, infMask)

	return result
}

// log64Scalar is the scalar fallback for tail elements.
func log64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := log64Base(v)
	return result.Data()[0]
}
