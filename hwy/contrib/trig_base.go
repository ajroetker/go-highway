package contrib

import (
	"math"

	"github.com/go-highway/highway/hwy"
)

// Constants for trigonometric functions
const (
	pi32       float32 = 3.1415926535897932
	piOver2_32 float32 = 1.5707963267948966
	piOver4_32 float32 = 0.7853981633974483
	invPi32    float32 = 0.3183098861837907 // 1/π

	pi64       float64 = 3.1415926535897932
	piOver2_64 float64 = 1.5707963267948966
	piOver4_64 float64 = 0.7853981633974483
	invPi64    float64 = 0.3183098861837907
)

// Polynomial coefficients for sin(x) on [-π/4, π/4]
// Using minimax polynomial
var sinCoeffs32 = []float32{
	0.0,                    // c0 (sin(0) = 0)
	1.0,                    // c1
	0.0,                    // c2 (no x² term)
	-0.16666666666666666,   // c3 = -1/3!
	0.0,                    // c4
	0.008333333333333333,   // c5 = 1/5!
	0.0,                    // c6
	-0.0001984126984126984, // c7 = -1/7!
}

// Polynomial coefficients for cos(x) on [-π/4, π/4]
var cosCoeffs32 = []float32{
	1.0,                   // c0 (cos(0) = 1)
	0.0,                   // c1 (no x term)
	-0.5,                  // c2 = -1/2!
	0.0,                   // c3
	0.041666666666666664,  // c4 = 1/4!
	0.0,                   // c5
	-0.001388888888888889, // c6 = -1/6!
	0.0,                   // c7
}

var sinCoeffs64 = []float64{
	0.0,
	1.0,
	0.0,
	-0.16666666666666666,
	0.0,
	0.008333333333333333,
	0.0,
	-0.0001984126984126984,
	0.0,
	2.7557319223985893e-06,
}

var cosCoeffs64 = []float64{
	1.0,
	0.0,
	-0.5,
	0.0,
	0.041666666666666664,
	0.0,
	-0.001388888888888889,
	0.0,
	2.48015873015873e-05,
	0.0,
}

func init() {
	Sin32 = sin32Base
	Sin64 = sin64Base
	Cos32 = cos32Base
	Cos64 = cos64Base
	SinCos32 = sinCos32Base
	SinCos64 = sinCos64Base
}

// rangeReduceSin32 reduces x to the range [-π/4, π/4] and returns the reduced value
// and the quadrant (0-3).
func rangeReduceSin32(x float32) (reduced float32, quadrant int) {
	// Reduce to [-π, π]
	if x > pi32 || x < -pi32 {
		// x = x - 2π * round(x / (2π))
		k := math.Round(float64(x) / (2.0 * float64(pi32)))
		x = x - float32(k)*2.0*pi32
	}

	// Determine quadrant and reduce to [-π/4, π/4]
	if x >= 0 {
		if x <= piOver4_32 {
			return x, 0
		} else if x <= 3.0*piOver4_32 {
			return piOver2_32 - x, 1
		} else {
			return x - pi32, 2
		}
	} else {
		if x >= -piOver4_32 {
			return x, 0
		} else if x >= -3.0*piOver4_32 {
			return -piOver2_32 - x, 3
		} else {
			return x + pi32, 2
		}
	}
}

// sin32Base computes sin(x) for float32 using range reduction and polynomial approximation.
func sin32Base(v hwy.Vec[float32]) hwy.Vec[float32] {
	n := v.NumLanes()
	result := make([]float32, n)
	input := v.Data()

	for i := 0; i < n; i++ {
		x := input[i]

		// Handle special cases
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			result[i] = float32(math.NaN())
			continue
		}

		// Range reduction
		reduced, quadrant := rangeReduceSin32(x)

		// Compute sin using polynomial
		x2 := reduced * reduced
		sinVal := reduced * (1.0 + x2*(-0.16666666666666666 +
			x2*(0.008333333333333333 + x2*(-0.0001984126984126984))))

		// Adjust sign based on quadrant
		switch quadrant {
		case 0: // [0, π/4]: sin is positive
			result[i] = sinVal
		case 1: // [π/4, 3π/4]: use cos, positive
			cosVal := 1.0 + x2*(-0.5+x2*(0.041666666666666664+x2*(-0.001388888888888889)))
			result[i] = cosVal
		case 2: // [3π/4, 5π/4]: sin is negative
			result[i] = -sinVal
		case 3: // [5π/4, 7π/4]: use cos, negative
			cosVal := 1.0 + x2*(-0.5+x2*(0.041666666666666664+x2*(-0.001388888888888889)))
			result[i] = -cosVal
		}
	}

	return hwy.Load(result)
}

// sin64Base computes sin(x) for float64 using the same algorithm as sin32Base.
func sin64Base(v hwy.Vec[float64]) hwy.Vec[float64] {
	n := v.NumLanes()
	result := make([]float64, n)
	input := v.Data()

	for i := 0; i < n; i++ {
		x := input[i]

		if math.IsNaN(x) || math.IsInf(x, 0) {
			result[i] = math.NaN()
			continue
		}

		// Use math.Sin for now as a placeholder for the full implementation
		// A production version would implement the full range reduction and polynomial
		result[i] = math.Sin(x)
	}

	return hwy.Load(result)
}

// cos32Base computes cos(x) for float32 using range reduction and polynomial approximation.
func cos32Base(v hwy.Vec[float32]) hwy.Vec[float32] {
	n := v.NumLanes()
	result := make([]float32, n)
	input := v.Data()

	for i := 0; i < n; i++ {
		x := input[i]

		// Handle special cases
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			result[i] = float32(math.NaN())
			continue
		}

		// Range reduction (similar to sin but shifted by π/2)
		reduced, quadrant := rangeReduceSin32(x)

		// Compute values
		x2 := reduced * reduced
		sinVal := reduced * (1.0 + x2*(-0.16666666666666666 +
			x2*(0.008333333333333333 + x2*(-0.0001984126984126984))))
		cosVal := 1.0 + x2*(-0.5+x2*(0.041666666666666664+x2*(-0.001388888888888889)))

		// Adjust based on quadrant (cos is sin shifted by π/2)
		switch quadrant {
		case 0: // [0, π/4]
			result[i] = cosVal
		case 1: // [π/4, 3π/4]
			result[i] = sinVal
		case 2: // [3π/4, 5π/4]
			result[i] = -cosVal
		case 3: // [5π/4, 7π/4]
			result[i] = -sinVal
		}
	}

	return hwy.Load(result)
}

// cos64Base computes cos(x) for float64.
func cos64Base(v hwy.Vec[float64]) hwy.Vec[float64] {
	n := v.NumLanes()
	result := make([]float64, n)
	input := v.Data()

	for i := 0; i < n; i++ {
		x := input[i]

		if math.IsNaN(x) || math.IsInf(x, 0) {
			result[i] = math.NaN()
			continue
		}

		result[i] = math.Cos(x)
	}

	return hwy.Load(result)
}

// sinCos32Base computes both sin and cos for float32.
// This is more efficient than computing them separately since
// they share the range reduction step.
func sinCos32Base(v hwy.Vec[float32]) (sin, cos hwy.Vec[float32]) {
	n := v.NumLanes()
	sinResult := make([]float32, n)
	cosResult := make([]float32, n)
	input := v.Data()

	for i := 0; i < n; i++ {
		x := input[i]

		// Handle special cases
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			sinResult[i] = float32(math.NaN())
			cosResult[i] = float32(math.NaN())
			continue
		}

		// Range reduction
		reduced, quadrant := rangeReduceSin32(x)

		// Compute both values
		x2 := reduced * reduced
		sinVal := reduced * (1.0 + x2*(-0.16666666666666666 +
			x2*(0.008333333333333333 + x2*(-0.0001984126984126984))))
		cosVal := 1.0 + x2*(-0.5+x2*(0.041666666666666664+x2*(-0.001388888888888889)))

		// Adjust based on quadrant
		switch quadrant {
		case 0:
			sinResult[i] = sinVal
			cosResult[i] = cosVal
		case 1:
			sinResult[i] = cosVal
			cosResult[i] = sinVal
		case 2:
			sinResult[i] = -sinVal
			cosResult[i] = -cosVal
		case 3:
			sinResult[i] = -cosVal
			cosResult[i] = -sinVal
		}
	}

	return hwy.Load(sinResult), hwy.Load(cosResult)
}

// sinCos64Base computes both sin and cos for float64.
func sinCos64Base(v hwy.Vec[float64]) (sin, cos hwy.Vec[float64]) {
	n := v.NumLanes()
	sinResult := make([]float64, n)
	cosResult := make([]float64, n)
	input := v.Data()

	for i := 0; i < n; i++ {
		x := input[i]

		if math.IsNaN(x) || math.IsInf(x, 0) {
			sinResult[i] = math.NaN()
			cosResult[i] = math.NaN()
			continue
		}

		sinResult[i] = math.Sin(x)
		cosResult[i] = math.Cos(x)
	}

	return hwy.Load(sinResult), hwy.Load(cosResult)
}
