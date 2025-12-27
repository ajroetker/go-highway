// Package matvec provides high-performance SIMD matrix-vector operations.
// This package corresponds to Google Highway's hwy/contrib/matvec directory.
//
// # Matrix-Vector Product
//
// The package provides vectorized matrix-vector multiplication:
//   - MatVec(m []float32, rows, cols int, v, result []float32) - float32 M*v
//   - MatVec64(m []float64, rows, cols int, v, result []float64) - float64 M*v
//
// # Algorithm
//
// Matrix-vector multiplication computes result = M * v where:
//   - M is a matrix of shape [rows, cols] in row-major order
//   - v is a vector of length cols
//   - result is a vector of length rows
//
// Each output element result[i] is the dot product of row i with vector v.
//
// The SIMD implementation:
//   1. Iterates over each row of the matrix
//   2. Computes dot product of row with v using SIMD operations
//   3. Stores result in output vector
//
// # Example Usage
//
//	import "github.com/ajroetker/go-highway/hwy/contrib/matvec"
//
//	// Matrix-vector product: result = M * v
//	// M is 3x4 matrix in row-major order:
//	//   [1 2 3 4]
//	//   [5 6 7 8]
//	//   [9 0 1 2]
//	m := []float32{
//	    1, 2, 3, 4,
//	    5, 6, 7, 8,
//	    9, 0, 1, 2,
//	}
//	v := []float32{1, 2, 3, 4}
//	result := make([]float32, 3)
//
//	matvec.MatVec(m, 3, 4, v, result)
//	// result = [30, 70, 20]
//	// 30 = 1*1 + 2*2 + 3*3 + 4*4
//	// 70 = 5*1 + 6*2 + 7*3 + 8*4
//	// 20 = 9*1 + 0*2 + 1*3 + 2*4
//
// # Performance
//
// Matrix-vector operations use the optimized dot product internally,
// providing speedups of 4-8x over scalar code on typical matrices.
//
// # Build Requirements
//
// The SIMD implementations require:
//   - GOEXPERIMENT=simd build flag
//   - AMD64 architecture with AVX2 or AVX-512 support
package matvec
