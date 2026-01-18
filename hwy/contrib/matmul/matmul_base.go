package matmul

// matmulScalar is the pure Go scalar implementation.
// C[i,j] = sum(A[i,p] * B[p,j]) for p in 0..K-1
// This is kept for reference and benchmarking; the generated BaseMatMul_fallback
// is used as the actual fallback implementation.
func matmulScalar(a, b, c []float32, m, n, k int) {
	// Clear output
	for i := range c[:m*n] {
		c[i] = 0
	}

	// Standard triple-loop matrix multiply
	for i := range m {
		for p := range k {
			aip := a[i*k+p]
			for j := range n {
				c[i*n+j] += aip * b[p*n+j]
			}
		}
	}
}

// matmulScalar64 is the pure Go scalar implementation for float64.
func matmulScalar64(a, b, c []float64, m, n, k int) {
	// Clear output
	for i := range c[:m*n] {
		c[i] = 0
	}

	// Standard triple-loop matrix multiply
	for i := range m {
		for p := range k {
			aip := a[i*k+p]
			for j := range n {
				c[i*n+j] += aip * b[p*n+j]
			}
		}
	}
}
