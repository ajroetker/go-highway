package contrib

import (
	"math"

	"github.com/go-highway/highway/hwy"
)

// Constants for float32 exp implementation
const (
	// ln(2) split into high and low parts for better precision
	ln2Hi32 float32 = 0.693359375      // High part of ln(2)
	ln2Lo32 float32 = -2.12194440e-4   // Low part of ln(2)
	invLn2_32 float32 = 1.44269504088896341

	// Overflow and underflow thresholds for float32
	expOverflow32  float32 = 88.72283905206835
	expUnderflow32 float32 = -87.33654475055310
)

// Minimax polynomial coefficients for exp(r) on [-ln(2)/2, ln(2)/2]
// Generated for maximum accuracy with degree 6 polynomial
var expCoeffs32 = []float32{
	1.0,                       // c0
	1.0,                       // c1
	0.5,                       // c2 = 1/2!
	0.16666666666666666,       // c3 = 1/3!
	0.041666666666666664,      // c4 = 1/4!
	0.008333333333333333,      // c5 = 1/5!
	0.001388888888888889,      // c6 = 1/6!
}

// Constants for float64 exp implementation
const (
	ln2Hi64 float64 = 0.6931471803691238
	ln2Lo64 float64 = 1.9082149292705877e-10
	invLn2_64 float64 = 1.4426950408889634

	expOverflow64  float64 = 709.782712893384
	expUnderflow64 float64 = -708.3964185322641
)

// Minimax polynomial coefficients for exp(r) on [-ln(2)/2, ln(2)/2]
// Higher degree for float64 for better accuracy
var expCoeffs64 = []float64{
	1.0,                           // c0
	1.0,                           // c1
	0.5,                           // c2
	0.16666666666666666,           // c3
	0.041666666666666664,          // c4
	0.008333333333333333,          // c5
	0.001388888888888889,          // c6
	0.0001984126984126984,         // c7 = 1/7!
	2.48015873015873e-05,          // c8 = 1/8!
	2.7557319223985893e-06,        // c9 = 1/9!
	2.755731922398589e-07,         // c10 = 1/10!
	2.505210838544172e-08,         // c11 = 1/11!
	2.08767569878681e-09,          // c12 = 1/12!
	1.6059043836821613e-10,        // c13 = 1/13!
}

func init() {
	// Register base implementations as defaults
	Exp32 = exp32Base
	Exp64 = exp64Base
}

// exp32Base computes e^x for float32 using range reduction and polynomial approximation.
//
// Algorithm:
// 1. Handle special cases (NaN, ±Inf, overflow, underflow)
// 2. Range reduction: x = k*ln(2) + r, where |r| <= ln(2)/2
//    k = round(x / ln(2))
//    r = x - k*ln(2)  (using high/low parts for precision)
// 3. Polynomial approximation: e^r ≈ 1 + r + r²/2! + r³/3! + ...
// 4. Reconstruction: e^x = 2^k * e^r
func exp32Base(v hwy.Vec[float32]) hwy.Vec[float32] {
	n := v.NumLanes()
	result := make([]float32, n)
	input := v.Data()

	for i := 0; i < n; i++ {
		x := input[i]

		// Handle special cases
		if math.IsNaN(float64(x)) {
			result[i] = float32(math.NaN())
			continue
		}
		if math.IsInf(float64(x), 1) {
			result[i] = float32(math.Inf(1))
			continue
		}
		if math.IsInf(float64(x), -1) {
			result[i] = 0
			continue
		}
		if x > expOverflow32 {
			result[i] = float32(math.Inf(1))
			continue
		}
		if x < expUnderflow32 {
			result[i] = 0
			continue
		}

		// Range reduction: x = k*ln(2) + r
		k := int32(math.Round(float64(x) * float64(invLn2_32)))

		// r = x - k*ln(2) using high/low split for precision
		r := float64(x) - float64(k)*float64(ln2Hi32) - float64(k)*float64(ln2Lo32)

		// Polynomial approximation of e^r using Horner's method
		// e^r ≈ 1 + r + r²/2! + r³/3! + ...
		poly := 1.0 + r*(1.0+r*(0.5+r*(1.0/6.0+r*(1.0/24.0+r*(1.0/120.0+r/720.0)))))

		// Scale by 2^k
		result[i] = float32(math.Ldexp(poly, int(k)))
	}

	return hwy.Load(result)
}

// exp64Base computes e^x for float64 using the same algorithm as exp32Base
// but with higher-precision constants and polynomial coefficients.
func exp64Base(v hwy.Vec[float64]) hwy.Vec[float64] {
	n := v.NumLanes()
	result := make([]float64, n)
	input := v.Data()

	vInvLn2 := hwy.Set(invLn2_64)
	vLn2Hi := hwy.Set(ln2Hi64)
	vLn2Lo := hwy.Set(ln2Lo64)
	vOverflow := hwy.Set(expOverflow64)
	vUnderflow := hwy.Set(expUnderflow64)

	// Check for overflow/underflow
	overflowMask := hwy.GreaterThan(v, vOverflow)
	underflowMask := hwy.LessThan(v, vUnderflow)

	// Range reduction: k = round(x / ln(2))
	vK := hwy.Mul(v, vInvLn2)

	kData := vK.Data()
	kInt := make([]int32, n)
	for i := 0; i < n; i++ {
		kInt[i] = int32(math.Round(kData[i]))
	}

	// Compute r = x - k*ln(2)
	vKFloat := hwy.Zero[float64]()
	kFloatData := vKFloat.Data()
	for i := 0; i < n; i++ {
		kFloatData[i] = float64(kInt[i])
	}
	vKFloat = hwy.Load(kFloatData)

	r := hwy.Sub(v, hwy.Mul(vKFloat, vLn2Hi))
	r = hwy.Sub(r, hwy.Mul(vKFloat, vLn2Lo))

	// Polynomial approximation
	poly := Horner(r, expCoeffs64)

	// Scale by 2^k
	polyData := poly.Data()
	for i := 0; i < n; i++ {
		k := kInt[i]

		if k > 1023 {
			k = 1023
		}
		if k < -1022 {
			k = -1022
		}

		result[i] = math.Ldexp(polyData[i], int(k))
	}

	resultVec := hwy.Load(result)

	// Handle special cases
	inf64 := hwy.Set(math.Inf(1))
	zero64 := hwy.Set(float64(0))

	resultVec = hwy.IfThenElse(overflowMask, inf64, resultVec)
	resultVec = hwy.IfThenElse(underflowMask, zero64, resultVec)

	// Handle NaN
	for i := 0; i < n; i++ {
		if math.IsNaN(input[i]) {
			result[i] = math.NaN()
		}
	}

	return hwy.Load(result)
}
