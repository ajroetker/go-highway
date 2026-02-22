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

//go:generate go run ../../../cmd/hwygen -input matmul_klast_base.go -dispatch matmul_klast -output . -targets avx2,avx512,neon:asm,fallback

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

	// Specialized M=1 path for autoregressive decoder (vector-matrix multiply).
	// Processes 4 B rows at a time for better register utilization.
	// Memory-bandwidth bound — no packing needed.
	if m == 1 {
		aRow := 0

		var j int
		for j = 0; j+3 < n; j += 4 {
			bRow0 := j * k
			bRow1 := (j + 1) * k
			bRow2 := (j + 2) * k
			bRow3 := (j + 3) * k

			var tot0, tot1, tot2, tot3 T
			for pBlock := 0; pBlock < k; pBlock += pairwiseBlockK {
				pEnd := min(pBlock+pairwiseBlockK, k)

				acc0 := hwy.Zero[T]()
				acc1 := hwy.Zero[T]()
				acc2 := hwy.Zero[T]()
				acc3 := hwy.Zero[T]()

				var p int
				for p = pBlock; p+lanes <= pEnd; p += lanes {
					vA := hwy.Load(a[aRow+p:])
					acc0 = hwy.MulAdd(vA, hwy.Load(b[bRow0+p:]), acc0)
					acc1 = hwy.MulAdd(vA, hwy.Load(b[bRow1+p:]), acc1)
					acc2 = hwy.MulAdd(vA, hwy.Load(b[bRow2+p:]), acc2)
					acc3 = hwy.MulAdd(vA, hwy.Load(b[bRow3+p:]), acc3)
				}

				s0 := hwy.ReduceSum(acc0)
				s1 := hwy.ReduceSum(acc1)
				s2 := hwy.ReduceSum(acc2)
				s3 := hwy.ReduceSum(acc3)
				for ; p < pEnd; p++ {
					ap := a[aRow+p]
					s0 += ap * b[bRow0+p]
					s1 += ap * b[bRow1+p]
					s2 += ap * b[bRow2+p]
					s3 += ap * b[bRow3+p]
				}
				tot0 += s0
				tot1 += s1
				tot2 += s2
				tot3 += s3
			}

			c[j] = tot0
			c[j+1] = tot1
			c[j+2] = tot2
			c[j+3] = tot3
		}

		// Remainder B rows
		for ; j < n; j++ {
			bRow := j * k
			var total T
			for pBlock := 0; pBlock < k; pBlock += pairwiseBlockK {
				pEnd := min(pBlock+pairwiseBlockK, k)
				acc := hwy.Zero[T]()
				var p int
				for p = pBlock; p+lanes <= pEnd; p += lanes {
					acc = hwy.MulAdd(hwy.Load(a[aRow+p:]), hwy.Load(b[bRow+p:]), acc)
				}
				sum := hwy.ReduceSum(acc)
				for ; p < pEnd; p++ {
					sum += a[aRow+p] * b[bRow+p]
				}
				total += sum
			}
			c[j] = total
		}
		return
	}

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
		for j := range n {
			bRow := j * k

			// Pairwise summation: accumulate K in blocks
			var tot0, tot1, tot2, tot3 T
			for pBlock := 0; pBlock < k; pBlock += pairwiseBlockK {
				pEnd := min(pBlock+pairwiseBlockK, k)

				acc0 := hwy.Zero[T]()
				acc1 := hwy.Zero[T]()
				acc2 := hwy.Zero[T]()
				acc3 := hwy.Zero[T]()

				var p int
				for p = pBlock; p+lanes <= pEnd; p += lanes {
					vB := hwy.Load(b[bRow+p:])
					acc0 = hwy.MulAdd(hwy.Load(a[aRow0+p:]), vB, acc0)
					acc1 = hwy.MulAdd(hwy.Load(a[aRow1+p:]), vB, acc1)
					acc2 = hwy.MulAdd(hwy.Load(a[aRow2+p:]), vB, acc2)
					acc3 = hwy.MulAdd(hwy.Load(a[aRow3+p:]), vB, acc3)
				}

				// Horizontal sum + scalar tail for this block
				s0 := hwy.ReduceSum(acc0)
				s1 := hwy.ReduceSum(acc1)
				s2 := hwy.ReduceSum(acc2)
				s3 := hwy.ReduceSum(acc3)
				for ; p < pEnd; p++ {
					s0 += a[aRow0+p] * b[bRow+p]
					s1 += a[aRow1+p] * b[bRow+p]
					s2 += a[aRow2+p] * b[bRow+p]
					s3 += a[aRow3+p] * b[bRow+p]
				}

				tot0 += s0
				tot1 += s1
				tot2 += s2
				tot3 += s3
			}

			c[cRow0+j] = tot0
			c[cRow1+j] = tot1
			c[cRow2+j] = tot2
			c[cRow3+j] = tot3
		}
	}

	// Handle remaining rows (0-3)
	for ; i < m; i++ {
		aRow := i * k
		cRow := i * n

		for j := range n {
			bRow := j * k

			var total T
			for pBlock := 0; pBlock < k; pBlock += pairwiseBlockK {
				pEnd := min(pBlock+pairwiseBlockK, k)
				acc := hwy.Zero[T]()

				var p int
				for p = pBlock; p+lanes <= pEnd; p += lanes {
					vA := hwy.Load(a[aRow+p:])
					vB := hwy.Load(b[bRow+p:])
					acc = hwy.MulAdd(vA, vB, acc)
				}

				sum := hwy.ReduceSum(acc)
				for ; p < pEnd; p++ {
					sum += a[aRow+p] * b[bRow+p]
				}
				total += sum
			}

			c[cRow+j] = total
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

				// Process block with pairwise summation within each K block
				for i := ii; i < iEnd; i++ {
					aRow := i * k
					cRow := i * n

					for j := jj; j < jEnd; j++ {
						bRow := j * k

						var blockTotal T
						for pBlock := kk; pBlock < kEnd; pBlock += pairwiseBlockK {
							pBlockEnd := min(pBlock+pairwiseBlockK, kEnd)
							acc := hwy.Zero[T]()

							var p int
							for p = pBlock; p+lanes <= pBlockEnd; p += lanes {
								vA := hwy.Load(a[aRow+p:])
								vB := hwy.Load(b[bRow+p:])
								acc = hwy.MulAdd(vA, vB, acc)
							}

							sum := hwy.ReduceSum(acc)
							for ; p < pBlockEnd; p++ {
								sum += a[aRow+p] * b[bRow+p]
							}
							blockTotal += sum
						}

						c[cRow+j] += blockTotal
					}
				}
			}
		}
	}
}
