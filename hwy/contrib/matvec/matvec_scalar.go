//go:build !amd64 || !goexperiment.simd

package matvec

// matvecImpl32 is the scalar-only implementation for float32.
func matvecImpl32(m []float32, rows, cols int, v, result []float32) {
	for i := 0; i < rows; i++ {
		var sum float32
		rowStart := i * cols
		for j := 0; j < cols; j++ {
			sum += m[rowStart+j] * v[j]
		}
		result[i] = sum
	}
}

// matvecImpl64 is the scalar-only implementation for float64.
func matvecImpl64(m []float64, rows, cols int, v, result []float64) {
	for i := 0; i < rows; i++ {
		var sum float64
		rowStart := i * cols
		for j := 0; j < cols; j++ {
			sum += m[rowStart+j] * v[j]
		}
		result[i] = sum
	}
}
