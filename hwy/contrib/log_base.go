package contrib

import (
	"math"

	"github.com/go-highway/highway/hwy"
)

// Constants for float32 log implementation
const (
	ln2_32 float32 = 0.6931471805599453
	sqrt2Over2_32 float32 = 0.7071067811865476 // 1/sqrt(2)
)

// Polynomial coefficients for log(1+x) on [sqrt(2)/2-1, sqrt(2)-1]
// Using minimax approximation
var logCoeffs32 = []float32{
	0.0,                      // c0 (will be computed separately)
	1.0,                      // c1
	-0.5,                     // c2 = -1/2
	0.3333333333333333,       // c3 = 1/3
	-0.25,                    // c4 = -1/4
	0.2,                      // c5 = 1/5
	-0.16666666666666666,     // c6 = -1/6
	0.14285714285714285,      // c7 = 1/7
}

// Constants for float64 log implementation
const (
	ln2_64 float64 = 0.6931471805599453
	sqrt2Over2_64 float64 = 0.7071067811865476
)

var logCoeffs64 = []float64{
	0.0,
	1.0,
	-0.5,
	0.3333333333333333,
	-0.25,
	0.2,
	-0.16666666666666666,
	0.14285714285714285,
	-0.125,
	0.1111111111111111,
	-0.1,
	0.09090909090909091,
}

func init() {
	Log32 = log32Base
	Log64 = log64Base
}

// log32Base computes ln(x) for float32 using range reduction and polynomial approximation.
//
// Algorithm:
// 1. Handle special cases (NaN, ±Inf, negative, zero)
// 2. Extract exponent and mantissa: x = 2^e * m, where 1 <= m < 2
// 3. If m < sqrt(2)/2, adjust: m = 2*m, e = e-1
// 4. Compute y = (m - 1) / (m + 1) or similar transformation
// 5. Polynomial approximation: ln(m) ≈ polynomial(y)
// 6. Result: ln(x) = e*ln(2) + ln(m)
func log32Base(v hwy.Vec[float32]) hwy.Vec[float32] {
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
		if x < 0 {
			result[i] = float32(math.NaN())
			continue
		}
		if x == 0 {
			result[i] = float32(math.Inf(-1))
			continue
		}
		if math.IsInf(float64(x), 1) {
			result[i] = float32(math.Inf(1))
			continue
		}

		// Extract exponent and mantissa using math.Frexp
		// Frexp returns m in [0.5, 1), so we multiply by 2 to get [1, 2)
		m, e := math.Frexp(float64(x))
		m *= 2.0
		e -= 1

		// If mantissa < sqrt(2)/2, normalize to [sqrt(2)/2, sqrt(2)]
		if m < float64(sqrt2Over2_32) {
			m *= 2.0
			e -= 1
		}

		// Compute log using polynomial approximation
		// Use ln(m) = ln(1 + (m-1)) and approximate with polynomial
		// For better accuracy, we can use: ln(m) = 2*atanh((m-1)/(m+1))
		// where atanh(y) ≈ y + y³/3 + y⁵/5 + ...
		y := (m - 1.0) / (m + 1.0)
		y2 := y * y

		// Polynomial for atanh(y) * 2
		poly := y * (2.0 + y2*(2.0/3.0 + y2*(2.0/5.0 + y2*(2.0/7.0 + y2*2.0/9.0))))

		// Combine: ln(x) = e*ln(2) + ln(m)
		result[i] = float32(float64(e)*float64(ln2_32) + poly)
	}

	return hwy.Load(result)
}

// log64Base computes ln(x) for float64 using the same algorithm as log32Base
// but with higher-precision constants and polynomial coefficients.
func log64Base(v hwy.Vec[float64]) hwy.Vec[float64] {
	n := v.NumLanes()
	result := make([]float64, n)
	input := v.Data()

	for i := 0; i < n; i++ {
		x := input[i]

		// Handle special cases
		if math.IsNaN(x) {
			result[i] = math.NaN()
			continue
		}
		if x < 0 {
			result[i] = math.NaN()
			continue
		}
		if x == 0 {
			result[i] = math.Inf(-1)
			continue
		}
		if math.IsInf(x, 1) {
			result[i] = math.Inf(1)
			continue
		}

		// Extract exponent and mantissa
		m, e := math.Frexp(x)
		m *= 2.0
		e -= 1

		// Normalize mantissa
		if m < sqrt2Over2_64 {
			m *= 2.0
			e -= 1
		}

		// Compute log using atanh transformation
		y := (m - 1.0) / (m + 1.0)
		y2 := y * y

		// Higher-degree polynomial for float64
		poly := y * (2.0 + y2*(2.0/3.0 + y2*(2.0/5.0 + y2*(2.0/7.0 +
			y2*(2.0/9.0 + y2*(2.0/11.0 + y2*2.0/13.0))))))

		// Combine
		result[i] = float64(e)*ln2_64 + poly
	}

	return hwy.Load(result)
}
