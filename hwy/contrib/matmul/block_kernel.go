// Copyright 2024 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

//go:generate go run ../../../cmd/hwygen -input block_kernel.go -dispatch blockkernel -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseBlockMulAdd computes C += A * B for square blocks.
//
// This is designed for cache-tiled matrix multiplication where:
//   - aT is blockDim × blockDim (PRE-TRANSPOSED A, so rows are original A columns)
//   - b is blockDim × blockDim (row-major, rows are B rows)
//   - c is blockDim × blockDim (row-major, accumulated into)
//
// The caller passes A^T (transposed A) and B (normal), and the function computes:
//
//	C += (A^T)^T * B = A * B
//
// This layout is optimal for SIMD:
//   - A^T[k, i:i+lanes] gives us A[i:i+lanes, k] (contiguous in A^T)
//   - B[k, j:j+lanes] gives us B[k, j:j+lanes] (contiguous in B)
//
// For standard matmul C = A * B where you have A and B:
//  1. Transpose A to get A^T
//  2. Call BaseBlockMulAdd(A^T, B, C, blockDim)
func BaseBlockMulAdd[T hwy.Floats](aT, b, c []T, blockDim int) {
	if len(aT) < blockDim*blockDim {
		panic("BlockMulAdd: aT slice too short")
	}
	if len(b) < blockDim*blockDim {
		panic("BlockMulAdd: B slice too short")
	}
	if len(c) < blockDim*blockDim {
		panic("BlockMulAdd: C slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()

	// Process one row of C at a time
	// C[i, j] = sum_k A[i,k] * B[k,j] = sum_k aT[k,i] * B[k,j]
	for i := range blockDim {
		cRowStart := i * blockDim

		// Accumulate contributions from all k values
		for k := range blockDim {
			// A[i,k] = aT[k,i] (transposed access)
			aik := aT[k*blockDim+i]
			vA := hwy.Set(aik) // Broadcast A[i,k]

			// B[k,:] is contiguous (row k of B)
			bRowStart := k * blockDim

			// Vectorized accumulation: C[i,j] += A[i,k] * B[k,j]
			var j int
			for j = 0; j+lanes <= blockDim; j += lanes {
				vB := hwy.Load(b[bRowStart+j:])
				vC := hwy.Load(c[cRowStart+j:])
				vC = hwy.MulAdd(vA, vB, vC)
				hwy.Store(vC, c[cRowStart+j:])
			}

			// Scalar tail
			for ; j < blockDim; j++ {
				c[cRowStart+j] += aik * b[bRowStart+j]
			}
		}
	}
}

// BaseBlockMulAdd2 computes C += A * B processing 2 rows of C at a time.
//
// Loop unrolling improves performance by reusing B loads and increasing ILP.
// Same semantics as BaseBlockMulAdd but with 2-way row unrolling.
func BaseBlockMulAdd2[T hwy.Floats](aT, b, c []T, blockDim int) {
	if len(aT) < blockDim*blockDim {
		panic("BlockMulAdd2: aT slice too short")
	}
	if len(b) < blockDim*blockDim {
		panic("BlockMulAdd2: B slice too short")
	}
	if len(c) < blockDim*blockDim {
		panic("BlockMulAdd2: C slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()

	// Process 2 rows of C at a time
	var i int
	for i = 0; i+1 < blockDim; i += 2 {
		cRow0Start := i * blockDim
		cRow1Start := (i + 1) * blockDim

		for k := range blockDim {
			// A[i,k] = aT[k,i], A[i+1,k] = aT[k,i+1]
			// These are consecutive in aT (same row k, columns i and i+1)
			a0k := aT[k*blockDim+i]
			a1k := aT[k*blockDim+i+1]
			vA0 := hwy.Set(a0k)
			vA1 := hwy.Set(a1k)

			bRowStart := k * blockDim

			var j int
			for j = 0; j+lanes <= blockDim; j += lanes {
				vB := hwy.Load(b[bRowStart+j:])

				vC0 := hwy.Load(c[cRow0Start+j:])
				vC0 = hwy.MulAdd(vA0, vB, vC0)
				hwy.Store(vC0, c[cRow0Start+j:])

				vC1 := hwy.Load(c[cRow1Start+j:])
				vC1 = hwy.MulAdd(vA1, vB, vC1)
				hwy.Store(vC1, c[cRow1Start+j:])
			}

			for ; j < blockDim; j++ {
				c[cRow0Start+j] += a0k * b[bRowStart+j]
				c[cRow1Start+j] += a1k * b[bRowStart+j]
			}
		}
	}

	// Handle odd row if blockDim is odd
	if i < blockDim {
		cRowStart := i * blockDim
		for k := range blockDim {
			aik := aT[k*blockDim+i]
			vA := hwy.Set(aik)
			bRowStart := k * blockDim

			var j int
			for j = 0; j+lanes <= blockDim; j += lanes {
				vB := hwy.Load(b[bRowStart+j:])
				vC := hwy.Load(c[cRowStart+j:])
				vC = hwy.MulAdd(vA, vB, vC)
				hwy.Store(vC, c[cRowStart+j:])
			}
			for ; j < blockDim; j++ {
				c[cRowStart+j] += aik * b[bRowStart+j]
			}
		}
	}
}

// BaseBlockMulAddRegBlocked computes C += A * B using register blocking.
//
// This is the highest-performance kernel that holds accumulators in registers
// across the entire K dimension, minimizing memory traffic.
//
// The kernel processes:
//   - 4 rows of C (Mr=4)
//   - 2 vector widths of columns (Nr=2*lanes, e.g., 32 cols for AVX-512)
//   - The full K dimension with accumulators held in registers
//
// This matches the register-blocking strategy used by high-performance BLAS
// implementations like OpenBLAS and MKL.
func BaseBlockMulAddRegBlocked[T hwy.Floats](aT, b, c []T, blockDim int) {
	if len(aT) < blockDim*blockDim {
		panic("BlockMulAddRegBlocked: aT slice too short")
	}
	if len(b) < blockDim*blockDim {
		panic("BlockMulAddRegBlocked: B slice too short")
	}
	if len(c) < blockDim*blockDim {
		panic("BlockMulAddRegBlocked: C slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()
	mr := 4       // Rows per micro-tile
	nr := lanes * 2 // Columns per micro-tile (2 vector widths)

	// Process micro-tiles of C
	var i int
	for i = 0; i+mr <= blockDim; i += mr {
		cRow0 := i * blockDim
		cRow1 := (i + 1) * blockDim
		cRow2 := (i + 2) * blockDim
		cRow3 := (i + 3) * blockDim

		// Tile the J dimension to fit Nr columns in accumulators
		var j int
		for j = 0; j+nr <= blockDim; j += nr {
			// Initialize 8 accumulators (4 rows × 2 column strips)
			// These stay in registers across the entire K loop
			acc00 := hwy.Zero[T]()
			acc01 := hwy.Zero[T]()
			acc10 := hwy.Zero[T]()
			acc11 := hwy.Zero[T]()
			acc20 := hwy.Zero[T]()
			acc21 := hwy.Zero[T]()
			acc30 := hwy.Zero[T]()
			acc31 := hwy.Zero[T]()

			// K-loop: accumulate in registers
			for k := range blockDim {
				// Load A values for 4 rows (consecutive in aT)
				aTRowK := k * blockDim
				a0k := aT[aTRowK+i]
				a1k := aT[aTRowK+i+1]
				a2k := aT[aTRowK+i+2]
				a3k := aT[aTRowK+i+3]

				vA0 := hwy.Set(a0k)
				vA1 := hwy.Set(a1k)
				vA2 := hwy.Set(a2k)
				vA3 := hwy.Set(a3k)

				// Load B values (2 vector widths)
				bRowStart := k * blockDim
				vB0 := hwy.Load(b[bRowStart+j:])
				vB1 := hwy.Load(b[bRowStart+j+lanes:])

				// Accumulate: 8 FMA operations
				acc00 = hwy.MulAdd(vA0, vB0, acc00)
				acc01 = hwy.MulAdd(vA0, vB1, acc01)
				acc10 = hwy.MulAdd(vA1, vB0, acc10)
				acc11 = hwy.MulAdd(vA1, vB1, acc11)
				acc20 = hwy.MulAdd(vA2, vB0, acc20)
				acc21 = hwy.MulAdd(vA2, vB1, acc21)
				acc30 = hwy.MulAdd(vA3, vB0, acc30)
				acc31 = hwy.MulAdd(vA3, vB1, acc31)
			}

			// Write back: Load C, add accumulator, store
			vC := hwy.Load(c[cRow0+j:])
			vC = hwy.Add(vC, acc00)
			hwy.Store(vC, c[cRow0+j:])

			vC = hwy.Load(c[cRow0+j+lanes:])
			vC = hwy.Add(vC, acc01)
			hwy.Store(vC, c[cRow0+j+lanes:])

			vC = hwy.Load(c[cRow1+j:])
			vC = hwy.Add(vC, acc10)
			hwy.Store(vC, c[cRow1+j:])

			vC = hwy.Load(c[cRow1+j+lanes:])
			vC = hwy.Add(vC, acc11)
			hwy.Store(vC, c[cRow1+j+lanes:])

			vC = hwy.Load(c[cRow2+j:])
			vC = hwy.Add(vC, acc20)
			hwy.Store(vC, c[cRow2+j:])

			vC = hwy.Load(c[cRow2+j+lanes:])
			vC = hwy.Add(vC, acc21)
			hwy.Store(vC, c[cRow2+j+lanes:])

			vC = hwy.Load(c[cRow3+j:])
			vC = hwy.Add(vC, acc30)
			hwy.Store(vC, c[cRow3+j:])

			vC = hwy.Load(c[cRow3+j+lanes:])
			vC = hwy.Add(vC, acc31)
			hwy.Store(vC, c[cRow3+j+lanes:])
		}

		// Handle remaining columns (less than Nr)
		for ; j < blockDim; j += lanes {
			// Single column strip
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()

			remaining := blockDim - j
			if remaining >= lanes {
				// Full vector
				for k := range blockDim {
					aTRowK := k * blockDim
					vA0 := hwy.Set(aT[aTRowK+i])
					vA1 := hwy.Set(aT[aTRowK+i+1])
					vA2 := hwy.Set(aT[aTRowK+i+2])
					vA3 := hwy.Set(aT[aTRowK+i+3])

					vB := hwy.Load(b[k*blockDim+j:])
					acc0 = hwy.MulAdd(vA0, vB, acc0)
					acc1 = hwy.MulAdd(vA1, vB, acc1)
					acc2 = hwy.MulAdd(vA2, vB, acc2)
					acc3 = hwy.MulAdd(vA3, vB, acc3)
				}

				vC := hwy.Load(c[cRow0+j:])
				vC = hwy.Add(vC, acc0)
				hwy.Store(vC, c[cRow0+j:])

				vC = hwy.Load(c[cRow1+j:])
				vC = hwy.Add(vC, acc1)
				hwy.Store(vC, c[cRow1+j:])

				vC = hwy.Load(c[cRow2+j:])
				vC = hwy.Add(vC, acc2)
				hwy.Store(vC, c[cRow2+j:])

				vC = hwy.Load(c[cRow3+j:])
				vC = hwy.Add(vC, acc3)
				hwy.Store(vC, c[cRow3+j:])
			} else {
				// Scalar tail
				for jj := j; jj < blockDim; jj++ {
					for k := range blockDim {
						aTRowK := k * blockDim
						bkj := b[k*blockDim+jj]
						c[cRow0+jj] += aT[aTRowK+i] * bkj
						c[cRow1+jj] += aT[aTRowK+i+1] * bkj
						c[cRow2+jj] += aT[aTRowK+i+2] * bkj
						c[cRow3+jj] += aT[aTRowK+i+3] * bkj
					}
				}
				break
			}
		}
	}

	// Handle remaining rows (less than Mr)
	for ; i < blockDim; i++ {
		cRowStart := i * blockDim
		for k := range blockDim {
			aik := aT[k*blockDim+i]
			vA := hwy.Set(aik)
			bRowStart := k * blockDim

			var j int
			for j = 0; j+lanes <= blockDim; j += lanes {
				vB := hwy.Load(b[bRowStart+j:])
				vC := hwy.Load(c[cRowStart+j:])
				vC = hwy.MulAdd(vA, vB, vC)
				hwy.Store(vC, c[cRowStart+j:])
			}
			for ; j < blockDim; j++ {
				c[cRowStart+j] += aik * b[bRowStart+j]
			}
		}
	}
}

// BaseBlockMulAdd4 computes C += A * B processing 4 rows of C at a time.
//
// 4-way loop unrolling for maximum performance on large blocks.
// Same semantics as BaseBlockMulAdd but with 4-way row unrolling.
//
// With aT layout, A[i,k], A[i+1,k], A[i+2,k], A[i+3,k] are consecutive
// in memory: aT[k*blockDim+i], aT[k*blockDim+i+1], etc.
// This provides excellent cache locality compared to the old interface.
func BaseBlockMulAdd4[T hwy.Floats](aT, b, c []T, blockDim int) {
	if len(aT) < blockDim*blockDim {
		panic("BlockMulAdd4: aT slice too short")
	}
	if len(b) < blockDim*blockDim {
		panic("BlockMulAdd4: B slice too short")
	}
	if len(c) < blockDim*blockDim {
		panic("BlockMulAdd4: C slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()

	// Process 4 rows of C at a time
	var i int
	for i = 0; i+3 < blockDim; i += 4 {
		cRow0 := i * blockDim
		cRow1 := (i + 1) * blockDim
		cRow2 := (i + 2) * blockDim
		cRow3 := (i + 3) * blockDim

		for k := range blockDim {
			// A[i+r, k] = aT[k, i+r] - consecutive in memory!
			aTRowK := k * blockDim
			a0k := aT[aTRowK+i]
			a1k := aT[aTRowK+i+1]
			a2k := aT[aTRowK+i+2]
			a3k := aT[aTRowK+i+3]

			vA0 := hwy.Set(a0k)
			vA1 := hwy.Set(a1k)
			vA2 := hwy.Set(a2k)
			vA3 := hwy.Set(a3k)

			bRowStart := k * blockDim

			var j int
			for j = 0; j+lanes <= blockDim; j += lanes {
				vB := hwy.Load(b[bRowStart+j:])

				vC0 := hwy.Load(c[cRow0+j:])
				vC0 = hwy.MulAdd(vA0, vB, vC0)
				hwy.Store(vC0, c[cRow0+j:])

				vC1 := hwy.Load(c[cRow1+j:])
				vC1 = hwy.MulAdd(vA1, vB, vC1)
				hwy.Store(vC1, c[cRow1+j:])

				vC2 := hwy.Load(c[cRow2+j:])
				vC2 = hwy.MulAdd(vA2, vB, vC2)
				hwy.Store(vC2, c[cRow2+j:])

				vC3 := hwy.Load(c[cRow3+j:])
				vC3 = hwy.MulAdd(vA3, vB, vC3)
				hwy.Store(vC3, c[cRow3+j:])
			}

			for ; j < blockDim; j++ {
				c[cRow0+j] += a0k * b[bRowStart+j]
				c[cRow1+j] += a1k * b[bRowStart+j]
				c[cRow2+j] += a2k * b[bRowStart+j]
				c[cRow3+j] += a3k * b[bRowStart+j]
			}
		}
	}

	// Handle remaining rows (0-3 rows)
	for ; i < blockDim; i++ {
		cRowStart := i * blockDim
		for k := range blockDim {
			aik := aT[k*blockDim+i]
			vA := hwy.Set(aik)
			bRowStart := k * blockDim

			var j int
			for j = 0; j+lanes <= blockDim; j += lanes {
				vB := hwy.Load(b[bRowStart+j:])
				vC := hwy.Load(c[cRowStart+j:])
				vC = hwy.MulAdd(vA, vB, vC)
				hwy.Store(vC, c[cRowStart+j:])
			}
			for ; j < blockDim; j++ {
				c[cRowStart+j] += aik * b[bRowStart+j]
			}
		}
	}
}
