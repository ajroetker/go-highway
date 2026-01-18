package vec

//go:generate go run ../../../cmd/hwygen -input distance_base.go -output . -targets avx2,avx512,neon,fallback -dispatch distance

import (
	"math"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseL2SquaredDistance computes the squared Euclidean distance between two slices.
// The result is the sum of squared differences: sum((a[i] - b[i])^2).
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
//	result := L2SquaredDistance(a, b)  // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
func BaseL2SquaredDistance[T hwy.Floats](a, b []T) T {
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
		diff := hwy.Sub(va, vb)
		diffSq := hwy.Mul(diff, diff)
		sum = hwy.Add(sum, diffSq)
	}

	// Reduce vector sum to scalar
	result := hwy.ReduceSum(sum)

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		d := a[i] - b[i]
		result += d * d
	}

	return result
}

// BaseL2Distance computes the Euclidean distance (L2 norm) between two slices.
// The result is the square root of the sum of squared differences: sqrt(sum((a[i] - b[i])^2)).
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
//	result := L2Distance(a, b)  // sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt(27) ≈ 5.196
func BaseL2Distance[T hwy.Floats](a, b []T) T {
	sqDist := BaseL2SquaredDistance(a, b)
	return T(math.Sqrt(float64(sqDist)))
}
