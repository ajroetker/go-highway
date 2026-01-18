package vec

//go:generate go run ../../../cmd/hwygen -input norm_base.go -output . -targets avx2,avx512,neon,fallback -dispatch norm

import (
	"math"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseSquaredNorm computes the squared L2 norm (sum of squares) of a vector
// using hwy primitives.
// The result is equivalent to Dot(v, v): Σ(v[i] * v[i]).
//
// Returns 0 if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Example:
//
//	v := []float32{3, 4}
//	result := SquaredNorm(v)  // 3*3 + 4*4 = 25
func BaseSquaredNorm[T hwy.Floats](v []T) T {
	if len(v) == 0 {
		return 0
	}

	n := len(v)
	sum := hwy.Zero[T]()
	lanes := sum.NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		vec := hwy.Load(v[i:])
		prod := hwy.Mul(vec, vec)
		sum = hwy.Add(sum, prod)
	}

	// Reduce vector sum to scalar
	result := hwy.ReduceSum(sum)

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		result += v[i] * v[i]
	}

	return result
}

// BaseNorm computes the L2 norm (Euclidean magnitude) of a vector using hwy
// primitives.
// The result is Sqrt(Σ(v[i] * v[i])).
//
// Returns 0 if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Example:
//
//	v := []float32{3, 4}
//	result := Norm(v)  // Sqrt(3*3 + 4*4) = Sqrt(25) = 5
func BaseNorm[T hwy.Floats](v []T) T {
	squaredNorm := BaseSquaredNorm(v)
	if squaredNorm == 0 {
		return 0
	}

	// Take square root of the squared norm
	// Use standard math library for the final scalar result
	switch any(squaredNorm).(type) {
	case float32:
		return any(float32(math.Sqrt(float64(any(squaredNorm).(float32))))).(T)
	case float64:
		return any(math.Sqrt(any(squaredNorm).(float64))).(T)
	default:
		// For Float16/BFloat16, convert through float32
		return any(float32(math.Sqrt(float64(any(squaredNorm).(float32))))).(T)
	}
}
