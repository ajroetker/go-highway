// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package matmul

//go:generate go run ../../../cmd/hwygen -input matmul_base.go -dispatch matmul -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

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
