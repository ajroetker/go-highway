package matmul

//go:generate go run ../../../cmd/hwygen -input matmul_simd.go -dispatch matmul -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseMatMul computes C = A * B where:
//   - A is M x K (row-major)
//   - B is K x N (row-major)
//   - C is M x N (row-major)
//
// Uses the "broadcast A, stream B" algorithm which is efficient for SIMD:
// For each row i of C and each column k of A, broadcast A[i,k] and
// multiply with the corresponding row of B, accumulating into C.
//
// This function is designed for code generation by hwygen.
// It will be specialized for AVX2, AVX-512, NEON, and fallback targets.
func BaseMatMul[T hwy.Floats](a, b, c []T, m, n, k int) {
	if len(a) < m*k {
		panic("matmul: A slice too short")
	}
	if len(b) < k*n {
		panic("matmul: B slice too short")
	}
	if len(c) < m*n {
		panic("matmul: C slice too short")
	}

	// For each row i of C
	for i := range m {
		cRow := c[i*n : (i+1)*n]

		// Zero the C row using SIMD
		vZero := hwy.Zero[T]()
		lanes := vZero.NumLanes()
		var j int
		for j = 0; j+lanes <= n; j += lanes {
			hwy.Store(vZero, cRow[j:])
		}
		// Scalar tail for zeroing
		for ; j < n; j++ {
			cRow[j] = 0
		}

		// Accumulate A[i,:] * B into C[i,:]
		// For each column p of A (= row p of B)
		for p := range k {
			aip := a[i*k+p]
			vA := hwy.Set(aip) // Broadcast A[i,p]
			bRow := b[p*n : (p+1)*n]

			// Vectorized multiply-add: C[i,j:j+lanes] += A[i,p] * B[p,j:j+lanes]
			for j = 0; j+lanes <= n; j += lanes {
				vB := hwy.Load(bRow[j:])
				vC := hwy.Load(cRow[j:])
				vC = hwy.MulAdd(vA, vB, vC) // C += A * B
				hwy.Store(vC, cRow[j:])
			}
			// Scalar tail
			for ; j < n; j++ {
				cRow[j] += aip * bRow[j]
			}
		}
	}
}
