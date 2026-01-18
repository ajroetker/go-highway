package vec

//go:generate go run ../../../cmd/hwygen -input dot_base.go -output . -targets avx2,avx512,neon,fallback -dispatch dot

import "github.com/ajroetker/go-highway/hwy"

// BaseDot computes the dot product (inner product) of two vectors using hwy primitives.
// The result is the sum of element-wise products: Σ(a[i] * b[i]).
//
// If the slices have different lengths, the computation uses the minimum length.
// Returns 0 if either slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Example:
//
//	a := []float32{1, 2, 3}
//	b := []float32{4, 5, 6}
//	result := Dot(a, b)  // 1*4 + 2*5 + 3*6 = 32
func BaseDot[T hwy.Floats](a, b []T) T {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	n := min(len(a), len(b))
	sum := hwy.Zero[T]()
	lanes := sum.NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		prod := hwy.Mul(va, vb)
		sum = hwy.Add(sum, prod)
	}

	// Reduce vector sum to scalar
	result := hwy.ReduceSum(sum)

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		result += a[i] * b[i]
	}

	return result
}
