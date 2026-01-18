package matvec

//go:generate go run ../../../cmd/hwygen -input matvec_base.go -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseMatVec computes the matrix-vector product: result = M * v
//
// Parameters:
//   - m: matrix in row-major order with shape [rows, cols]
//   - rows: number of rows in the matrix
//   - cols: number of columns in the matrix
//   - v: input vector of length cols
//   - result: output vector of length rows (must be pre-allocated)
//
// Each element result[i] is the dot product of row i with vector v.
// Uses SIMD acceleration when available via the hwy package primitives.
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
func BaseMatVec[T hwy.Floats](m []T, rows, cols int, v, result []T) {
	if len(m) < rows*cols {
		panic("matrix slice too small")
	}
	if len(v) < cols {
		panic("vector slice too small")
	}
	if len(result) < rows {
		panic("result slice too small")
	}

	for i := range rows {
		row := m[i*cols : (i+1)*cols]

		// SIMD dot product for this row
		sum := hwy.Zero[T]()
		lanes := sum.NumLanes()

		var j int
		for j = 0; j+lanes <= cols; j += lanes {
			va := hwy.Load(row[j:])
			vb := hwy.Load(v[j:])
			prod := hwy.Mul(va, vb)
			sum = hwy.Add(sum, prod)
		}

		// Reduce and add scalar tail
		acc := hwy.ReduceSum(sum)
		for ; j < cols; j++ {
			acc += row[j] * v[j]
		}
		result[i] = acc
	}
}
