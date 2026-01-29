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

//go:generate go run ../../../cmd/hwygen -input matmul_klast_base.go -dispatch matmul_klast -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseMatMulKLast computes C = A * B^T where:
//   - A is M x K (row-major, K last)
//   - B is N x K (row-major, K last - PyTorch weight format)
//   - C is M x N (row-major)
//
// This is the "K-last" layout where both input matrices have K as their
// last dimension. This is the natural format for PyTorch weights and
// enables efficient SIMD since both A rows and B rows are contiguous.
//
// Each output element: C[i,j] = dot(A[i,:], B[j,:])
//
// The algorithm vectorizes along the K dimension:
//  1. Load SIMD-width elements from A row i
//  2. Load SIMD-width elements from B row j
//  3. Multiply and accumulate into a vector accumulator
//  4. Horizontal sum at the end to produce C[i,j]
//
// Memory access pattern:
//   - A row i: A[i*K : i*K+K] - sequential (cache friendly)
//   - B row j: B[j*K : j*K+K] - sequential (cache friendly)
func BaseMatMulKLast[T hwy.Floats](a, b, c []T, m, n, k int) {
	if len(a) < m*k {
		panic("matmul: A slice too short")
	}
	if len(b) < n*k {
		panic("matmul: B slice too short")
	}
	if len(c) < m*n {
		panic("matmul: C slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()

	// Process 4 rows of A at a time for better register utilization
	var i int
	for i = 0; i+3 < m; i += 4 {
		aRow0 := i * k
		aRow1 := (i + 1) * k
		aRow2 := (i + 2) * k
		aRow3 := (i + 3) * k

		cRow0 := i * n
		cRow1 := (i + 1) * n
		cRow2 := (i + 2) * n
		cRow3 := (i + 3) * n

		// For each output column (B row)
		for j := 0; j < n; j++ {
			bRow := j * k

			// Initialize 4 accumulators
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()

			// Vectorized dot product along K
			var p int
			for p = 0; p+lanes <= k; p += lanes {
				vB := hwy.Load(b[bRow+p:])

				vA0 := hwy.Load(a[aRow0+p:])
				vA1 := hwy.Load(a[aRow1+p:])
				vA2 := hwy.Load(a[aRow2+p:])
				vA3 := hwy.Load(a[aRow3+p:])

				acc0 = hwy.MulAdd(vA0, vB, acc0)
				acc1 = hwy.MulAdd(vA1, vB, acc1)
				acc2 = hwy.MulAdd(vA2, vB, acc2)
				acc3 = hwy.MulAdd(vA3, vB, acc3)
			}

			// Horizontal sum + scalar tail
			sum0 := hwy.ReduceSum(acc0)
			sum1 := hwy.ReduceSum(acc1)
			sum2 := hwy.ReduceSum(acc2)
			sum3 := hwy.ReduceSum(acc3)

			for ; p < k; p++ {
				sum0 += a[aRow0+p] * b[bRow+p]
				sum1 += a[aRow1+p] * b[bRow+p]
				sum2 += a[aRow2+p] * b[bRow+p]
				sum3 += a[aRow3+p] * b[bRow+p]
			}

			c[cRow0+j] = sum0
			c[cRow1+j] = sum1
			c[cRow2+j] = sum2
			c[cRow3+j] = sum3
		}
	}

	// Handle remaining rows (0-3)
	for ; i < m; i++ {
		aRow := i * k
		cRow := i * n

		for j := 0; j < n; j++ {
			bRow := j * k
			acc := hwy.Zero[T]()

			var p int
			for p = 0; p+lanes <= k; p += lanes {
				vA := hwy.Load(a[aRow+p:])
				vB := hwy.Load(b[bRow+p:])
				acc = hwy.MulAdd(vA, vB, acc)
			}

			sum := hwy.ReduceSum(acc)
			for ; p < k; p++ {
				sum += a[aRow+p] * b[bRow+p]
			}

			c[cRow+j] = sum
		}
	}
}

// BaseMatMulKLastBlocked is a cache-blocked version of MatMulKLast.
// It processes the output in tiles to improve cache locality for large matrices.
//
// Block sizes are chosen to fit in L1/L2 cache:
//   - blockM, blockN: output tile dimensions
//   - blockK: reduction tile along K dimension
func BaseMatMulKLastBlocked[T hwy.Floats](a, b, c []T, m, n, k int) {
	if len(a) < m*k {
		panic("matmul: A slice too short")
	}
	if len(b) < n*k {
		panic("matmul: B slice too short")
	}
	if len(c) < m*n {
		panic("matmul: C slice too short")
	}

	// Block sizes tuned for L2 cache (~256KB)
	// A block: blockM × blockK × 4 bytes
	// B block: blockN × blockK × 4 bytes
	// C block: blockM × blockN × 4 bytes
	const blockM = 64
	const blockN = 64
	const blockK = 256

	lanes := hwy.Zero[T]().NumLanes()

	// Zero output first
	for i := range c[:m*n] {
		c[i] = 0
	}

	// Process in blocks
	for ii := 0; ii < m; ii += blockM {
		iEnd := min(ii+blockM, m)

		for jj := 0; jj < n; jj += blockN {
			jEnd := min(jj+blockN, n)

			for kk := 0; kk < k; kk += blockK {
				kEnd := min(kk+blockK, k)

				// Process block
				for i := ii; i < iEnd; i++ {
					aRow := i * k
					cRow := i * n

					for j := jj; j < jEnd; j++ {
						bRow := j * k
						acc := hwy.Zero[T]()

						var p int
						for p = kk; p+lanes <= kEnd; p += lanes {
							vA := hwy.Load(a[aRow+p:])
							vB := hwy.Load(b[bRow+p:])
							acc = hwy.MulAdd(vA, vB, acc)
						}

						sum := hwy.ReduceSum(acc)
						for ; p < kEnd; p++ {
							sum += a[aRow+p] * b[bRow+p]
						}

						c[cRow+j] += sum
					}
				}
			}
		}
	}
}
