package matvec

// MatVec computes the matrix-vector product: result = M * v
//
// Parameters:
//   - m: matrix in row-major order with shape [rows, cols]
//   - rows: number of rows in the matrix
//   - cols: number of columns in the matrix
//   - v: input vector of length cols
//   - result: output vector of length rows (must be pre-allocated)
//
// Each element result[i] is the dot product of row i with vector v.
//
// Panics if:
//   - len(m) < rows * cols
//   - len(v) < cols
//   - len(result) < rows
//
// Example:
//
//	// 2x3 matrix:
//	//   [1 2 3]
//	//   [4 5 6]
//	m := []float32{1, 2, 3, 4, 5, 6}
//	v := []float32{1, 0, 1}
//	result := make([]float32, 2)
//	MatVec(m, 2, 3, v, result)  // result = [4, 10]
func MatVec(m []float32, rows, cols int, v, result []float32) {
	if len(m) < rows*cols {
		panic("matrix slice too small")
	}
	if len(v) < cols {
		panic("vector slice too small")
	}
	if len(result) < rows {
		panic("result slice too small")
	}

	matvecImpl32(m, rows, cols, v, result)
}

// MatVec64 computes the matrix-vector product using float64: result = M * v
//
// Parameters are the same as MatVec but using float64 precision.
func MatVec64(m []float64, rows, cols int, v, result []float64) {
	if len(m) < rows*cols {
		panic("matrix slice too small")
	}
	if len(v) < cols {
		panic("vector slice too small")
	}
	if len(result) < rows {
		panic("result slice too small")
	}

	matvecImpl64(m, rows, cols, v, result)
}
