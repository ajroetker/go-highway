package contrib

import "github.com/go-highway/highway/hwy"

// Horner evaluates a polynomial using Horner's method for efficient computation.
// Given coefficients [c0, c1, c2, ..., cn], computes:
//   p(x) = c0 + x*(c1 + x*(c2 + ... + x*cn))
//
// This method minimizes the number of multiplications and is well-suited
// for vectorized execution.
//
// Example:
//   coeffs := []float32{1.0, 2.0, 3.0}  // represents 1 + 2x + 3x²
//   result := Horner(x, coeffs)
func Horner[T hwy.Floats](x hwy.Vec[T], coeffs []T) hwy.Vec[T] {
	if len(coeffs) == 0 {
		return hwy.Zero[T]()
	}

	// Start with the last coefficient
	result := hwy.Set(coeffs[len(coeffs)-1])

	// Work backwards through the coefficients
	for i := len(coeffs) - 2; i >= 0; i-- {
		// result = result * x + coeffs[i]
		result = hwy.FMA(result, x, hwy.Set(coeffs[i]))
	}

	return result
}

// Horner5 evaluates a degree-5 polynomial using Horner's method.
// This is an unrolled version for better performance on common polynomial degrees.
// Coefficients are in ascending order: [c0, c1, c2, c3, c4, c5]
func Horner5[T hwy.Floats](x hwy.Vec[T], c0, c1, c2, c3, c4, c5 T) hwy.Vec[T] {
	// ((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
	v5 := hwy.Set(c5)
	v4 := hwy.Set(c4)
	v3 := hwy.Set(c3)
	v2 := hwy.Set(c2)
	v1 := hwy.Set(c1)
	v0 := hwy.Set(c0)

	result := v5
	result = hwy.FMA(result, x, v4)
	result = hwy.FMA(result, x, v3)
	result = hwy.FMA(result, x, v2)
	result = hwy.FMA(result, x, v1)
	result = hwy.FMA(result, x, v0)

	return result
}

// Horner7 evaluates a degree-7 polynomial using Horner's method.
// Coefficients are in ascending order: [c0, c1, ..., c7]
func Horner7[T hwy.Floats](x hwy.Vec[T], c0, c1, c2, c3, c4, c5, c6, c7 T) hwy.Vec[T] {
	v7 := hwy.Set(c7)
	v6 := hwy.Set(c6)
	v5 := hwy.Set(c5)
	v4 := hwy.Set(c4)
	v3 := hwy.Set(c3)
	v2 := hwy.Set(c2)
	v1 := hwy.Set(c1)
	v0 := hwy.Set(c0)

	result := v7
	result = hwy.FMA(result, x, v6)
	result = hwy.FMA(result, x, v5)
	result = hwy.FMA(result, x, v4)
	result = hwy.FMA(result, x, v3)
	result = hwy.FMA(result, x, v2)
	result = hwy.FMA(result, x, v1)
	result = hwy.FMA(result, x, v0)

	return result
}

// Horner13 evaluates a degree-13 polynomial using Horner's method.
// This is used for higher-accuracy approximations like float64 exp/log.
func Horner13[T hwy.Floats](x hwy.Vec[T], coeffs [14]T) hwy.Vec[T] {
	result := hwy.Set(coeffs[13])
	for i := 12; i >= 0; i-- {
		result = hwy.FMA(result, x, hwy.Set(coeffs[i]))
	}
	return result
}
