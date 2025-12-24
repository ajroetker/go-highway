package contrib

import (
	"math"

	"github.com/go-highway/highway/hwy"
)

func init() {
	Tanh32 = tanh32Base
	Tanh64 = tanh64Base
	Sigmoid32 = sigmoid32Base
	Sigmoid64 = sigmoid64Base
	Erf32 = erf32Base
	Erf64 = erf64Base
}

// tanh32Base computes tanh(x) for float32.
//
// Uses the identity: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
// For numerical stability, we use different formulas for different ranges:
//   - For |x| < 0.5: use polynomial approximation
//   - For x > 9: tanh(x) ≈ 1
//   - For x < -9: tanh(x) ≈ -1
//   - Otherwise: use the exponential formula
func tanh32Base(v hwy.Vec[float32]) hwy.Vec[float32] {
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
			result[i] = 1.0
			continue
		}
		if math.IsInf(float64(x), -1) {
			result[i] = -1.0
			continue
		}

		absX := float32(math.Abs(float64(x)))

		if absX < 0.5 {
			// For small x, use polynomial approximation: tanh(x) ≈ x - x³/3 + 2x⁵/15 - 17x⁷/315
			x2 := x * x
			result[i] = x * (1.0 + x2*(-0.3333333333333333 +
				x2*(0.13333333333333333 + x2*(-0.053968253968253970))))
		} else if absX > 9.0 {
			// For large |x|, tanh(x) ≈ sign(x)
			if x > 0 {
				result[i] = 1.0
			} else {
				result[i] = -1.0
			}
		} else {
			// General case: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
			// Rewrite as: tanh(x) = 2/(1 + e^(-2x)) - 1 for better numerical stability
			v2x := hwy.Set(2.0 * x)
			vNeg2x := hwy.Neg(v2x)
			expVal := Exp32(vNeg2x)
			expData := expVal.Data()

			result[i] = 2.0/(1.0+expData[0]) - 1.0
		}
	}

	return hwy.Load(result)
}

// tanh64Base computes tanh(x) for float64.
func tanh64Base(v hwy.Vec[float64]) hwy.Vec[float64] {
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
		if math.IsInf(x, 1) {
			result[i] = 1.0
			continue
		}
		if math.IsInf(x, -1) {
			result[i] = -1.0
			continue
		}

		absX := math.Abs(x)

		if absX < 0.5 {
			x2 := x * x
			result[i] = x * (1.0 + x2*(-0.3333333333333333 +
				x2*(0.13333333333333333 + x2*(-0.053968253968253970 +
					x2*0.021869488536155203))))
		} else if absX > 20.0 {
			if x > 0 {
				result[i] = 1.0
			} else {
				result[i] = -1.0
			}
		} else {
			v2x := hwy.Set(2.0 * x)
			vNeg2x := hwy.Neg(v2x)
			expVal := Exp64(vNeg2x)
			expData := expVal.Data()

			result[i] = 2.0/(1.0+expData[0]) - 1.0
		}
	}

	return hwy.Load(result)
}

// sigmoid32Base computes sigmoid(x) = 1/(1+exp(-x)) for float32.
//
// For numerical stability:
//   - For x > 0: sigmoid(x) = 1 / (1 + e^(-x))
//   - For x < 0: sigmoid(x) = e^x / (1 + e^x)  [equivalent, but more stable]
//   - For x > 20: sigmoid(x) ≈ 1
//   - For x < -20: sigmoid(x) ≈ 0
func sigmoid32Base(v hwy.Vec[float32]) hwy.Vec[float32] {
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
			result[i] = 1.0
			continue
		}
		if math.IsInf(float64(x), -1) {
			result[i] = 0.0
			continue
		}

		// Use numerically stable formulas for all ranges
		if x >= 0 {
			// For x >= 0: sigmoid(x) = 1 / (1 + e^(-x))
			vNegX := hwy.Set(-x)
			expVal := Exp32(vNegX)
			expData := expVal.Data()
			result[i] = 1.0 / (1.0 + expData[0])
		} else {
			// For x < 0: sigmoid(x) = e^x / (1 + e^x)
			vX := hwy.Set(x)
			expVal := Exp32(vX)
			expData := expVal.Data()
			result[i] = expData[0] / (1.0 + expData[0])
		}
	}

	return hwy.Load(result)
}

// sigmoid64Base computes sigmoid(x) for float64.
func sigmoid64Base(v hwy.Vec[float64]) hwy.Vec[float64] {
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
		if math.IsInf(x, 1) {
			result[i] = 1.0
			continue
		}
		if math.IsInf(x, -1) {
			result[i] = 0.0
			continue
		}

		if x > 40.0 {
			result[i] = 1.0
		} else if x < -40.0 {
			result[i] = 0.0
		} else if x >= 0 {
			vNegX := hwy.Set(-x)
			expVal := Exp64(vNegX)
			expData := expVal.Data()
			result[i] = 1.0 / (1.0 + expData[0])
		} else {
			vX := hwy.Set(x)
			expVal := Exp64(vX)
			expData := expVal.Data()
			result[i] = expData[0] / (1.0 + expData[0])
		}
	}

	return hwy.Load(result)
}

// erf32Base computes the error function erf(x) for float32.
//
// Uses different approximations for different ranges:
//   - For |x| < 0.5: polynomial approximation
//   - For 0.5 <= |x| < 4: rational approximation
//   - For |x| >= 4: erf(x) ≈ sign(x)
func erf32Base(v hwy.Vec[float32]) hwy.Vec[float32] {
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
			result[i] = 1.0
			continue
		}
		if math.IsInf(float64(x), -1) {
			result[i] = -1.0
			continue
		}

		absX := float32(math.Abs(float64(x)))
		sign := float32(1.0)
		if x < 0 {
			sign = -1.0
		}

		if absX < 0.5 {
			// Polynomial approximation for small x
			// erf(x) ≈ x * (a1 + x²*(a2 + x²*(a3 + x²*a4)))
			x2 := absX * absX
			result[i] = sign * absX * (1.1283791670955126 +
				x2*(-0.3761263890318375 +
					x2*(0.11283791670955126 +
						x2*(-0.026866170645131254))))
		} else if absX < 4.0 {
			// Rational approximation for medium x
			// Use Abramowitz and Stegun approximation
			t := 1.0 / (1.0 + 0.3275911*absX)
			poly := t * (0.254829592 + t*(-0.284496736 +
				t*(1.421413741 + t*(-1.453152027 + t*1.061405429))))

			// erf(x) ≈ 1 - poly * exp(-x²)
			vX2 := hwy.Set(-absX * absX)
			expVal := Exp32(vX2)
			expData := expVal.Data()

			result[i] = sign * (1.0 - poly*expData[0])
		} else {
			// For large |x|, erf(x) ≈ sign(x)
			result[i] = sign
		}
	}

	return hwy.Load(result)
}

// erf64Base computes the error function erf(x) for float64.
func erf64Base(v hwy.Vec[float64]) hwy.Vec[float64] {
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
		if math.IsInf(x, 1) {
			result[i] = 1.0
			continue
		}
		if math.IsInf(x, -1) {
			result[i] = -1.0
			continue
		}

		// Use math.Erf for float64 (placeholder for full implementation)
		result[i] = math.Erf(x)
	}

	return hwy.Load(result)
}
