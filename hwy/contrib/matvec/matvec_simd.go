//go:build amd64 && goexperiment.simd

package matvec

import (
	"github.com/ajroetker/go-highway/hwy/contrib/dot"
)

// matvecImpl32 is the SIMD-aware implementation for float32.
// Each row-vector dot product is computed using the optimized Dot function.
func matvecImpl32(m []float32, rows, cols int, v, result []float32) {
	for i := 0; i < rows; i++ {
		// Get row i as a slice
		rowStart := i * cols
		rowEnd := rowStart + cols
		row := m[rowStart:rowEnd]

		// Compute dot product of row with v
		result[i] = dot.Dot(row, v)
	}
}

// matvecImpl64 is the SIMD-aware implementation for float64.
func matvecImpl64(m []float64, rows, cols int, v, result []float64) {
	for i := 0; i < rows; i++ {
		rowStart := i * cols
		rowEnd := rowStart + cols
		row := m[rowStart:rowEnd]

		result[i] = dot.Dot64(row, v)
	}
}
