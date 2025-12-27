//go:build !amd64 || !goexperiment.simd

package dot

// dotImpl32 is the scalar-only implementation for float32.
func dotImpl32(a, b []float32) float32 {
	return dotScalar32(a, b)
}

// dotImpl64 is the scalar-only implementation for float64.
func dotImpl64(a, b []float64) float64 {
	return dotScalar64(a, b)
}

// Scalar implementations

func dotScalar32(a, b []float32) float32 {
	n := min(len(a), len(b))
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func dotScalar64(a, b []float64) float64 {
	n := min(len(a), len(b))
	var sum float64
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}
