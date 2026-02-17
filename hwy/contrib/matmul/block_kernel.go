// Copyright 2024 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

//go:generate go run ../../../cmd/hwygen -input block_kernel.go -dispatch blockkernel -output . -targets avx2,avx512,neon:asm,fallback

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
// Uses register-blocked accumulators: the J dimension is tiled into groups
// of 4 vector widths, with accumulators held in registers across the full
// K loop. This eliminates K-1 redundant loads and stores of C per element.
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
	tileJ := 4 * lanes

	for i := range blockDim {
		cRowStart := i * blockDim

		// Tiled J loop — 4 accumulators held in registers across full K loop
		var j int
		for j = 0; j+tileJ <= blockDim; j += tileJ {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()
			for k := range blockDim {
				aik := aT[k*blockDim+i]
				vA := hwy.Set(aik)
				bRowStart := k * blockDim
				acc0 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j:]), acc0)
				acc1 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j+lanes:]), acc1)
				acc2 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j+2*lanes:]), acc2)
				acc3 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j+3*lanes:]), acc3)
			}
			// Add accumulators to existing C (this is +=)
			vC := hwy.Load(c[cRowStart+j:])
			hwy.Store(hwy.Add(vC, acc0), c[cRowStart+j:])
			vC = hwy.Load(c[cRowStart+j+lanes:])
			hwy.Store(hwy.Add(vC, acc1), c[cRowStart+j+lanes:])
			vC = hwy.Load(c[cRowStart+j+2*lanes:])
			hwy.Store(hwy.Add(vC, acc2), c[cRowStart+j+2*lanes:])
			vC = hwy.Load(c[cRowStart+j+3*lanes:])
			hwy.Store(hwy.Add(vC, acc3), c[cRowStart+j+3*lanes:])
		}

		// Remainder: single accumulator per remaining vector strip
		for ; j+lanes <= blockDim; j += lanes {
			acc := hwy.Zero[T]()
			for k := range blockDim {
				aik := aT[k*blockDim+i]
				vA := hwy.Set(aik)
				acc = hwy.MulAdd(vA, hwy.Load(b[k*blockDim+j:]), acc)
			}
			vC := hwy.Load(c[cRowStart+j:])
			hwy.Store(hwy.Add(vC, acc), c[cRowStart+j:])
		}

		// Scalar tail
		for ; j < blockDim; j++ {
			var sum T
			for k := range blockDim {
				sum += aT[k*blockDim+i] * b[k*blockDim+j]
			}
			c[cRowStart+j] += sum
		}
	}
}

// BaseBlockMulAdd2 computes C += A * B processing 2 rows of C at a time.
//
// Uses register-blocked accumulators with 2-way row unrolling (2 rows × 4 column strips = 8 accumulators).
// Same semantics as BaseBlockMulAdd but with better ILP from processing 2 rows simultaneously.
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
	tileJ := 4 * lanes

	// Process 2 rows of C at a time
	var i int
	for i = 0; i+1 < blockDim; i += 2 {
		cRow0Start := i * blockDim
		cRow1Start := (i + 1) * blockDim

		// Tiled J loop — 8 accumulators (2 rows × 4 strips)
		var j int
		for j = 0; j+tileJ <= blockDim; j += tileJ {
			acc00 := hwy.Zero[T]()
			acc01 := hwy.Zero[T]()
			acc02 := hwy.Zero[T]()
			acc03 := hwy.Zero[T]()
			acc10 := hwy.Zero[T]()
			acc11 := hwy.Zero[T]()
			acc12 := hwy.Zero[T]()
			acc13 := hwy.Zero[T]()
			for k := range blockDim {
				aTRowK := k * blockDim
				a0k := aT[aTRowK+i]
				a1k := aT[aTRowK+i+1]
				vA0 := hwy.Set(a0k)
				vA1 := hwy.Set(a1k)
				bRowStart := k * blockDim
				vB0 := hwy.Load(b[bRowStart+j:])
				vB1 := hwy.Load(b[bRowStart+j+lanes:])
				vB2 := hwy.Load(b[bRowStart+j+2*lanes:])
				vB3 := hwy.Load(b[bRowStart+j+3*lanes:])
				acc00 = hwy.MulAdd(vA0, vB0, acc00)
				acc01 = hwy.MulAdd(vA0, vB1, acc01)
				acc02 = hwy.MulAdd(vA0, vB2, acc02)
				acc03 = hwy.MulAdd(vA0, vB3, acc03)
				acc10 = hwy.MulAdd(vA1, vB0, acc10)
				acc11 = hwy.MulAdd(vA1, vB1, acc11)
				acc12 = hwy.MulAdd(vA1, vB2, acc12)
				acc13 = hwy.MulAdd(vA1, vB3, acc13)
			}
			// Add accumulators to existing C
			vC := hwy.Load(c[cRow0Start+j:])
			hwy.Store(hwy.Add(vC, acc00), c[cRow0Start+j:])
			vC = hwy.Load(c[cRow0Start+j+lanes:])
			hwy.Store(hwy.Add(vC, acc01), c[cRow0Start+j+lanes:])
			vC = hwy.Load(c[cRow0Start+j+2*lanes:])
			hwy.Store(hwy.Add(vC, acc02), c[cRow0Start+j+2*lanes:])
			vC = hwy.Load(c[cRow0Start+j+3*lanes:])
			hwy.Store(hwy.Add(vC, acc03), c[cRow0Start+j+3*lanes:])
			vC = hwy.Load(c[cRow1Start+j:])
			hwy.Store(hwy.Add(vC, acc10), c[cRow1Start+j:])
			vC = hwy.Load(c[cRow1Start+j+lanes:])
			hwy.Store(hwy.Add(vC, acc11), c[cRow1Start+j+lanes:])
			vC = hwy.Load(c[cRow1Start+j+2*lanes:])
			hwy.Store(hwy.Add(vC, acc12), c[cRow1Start+j+2*lanes:])
			vC = hwy.Load(c[cRow1Start+j+3*lanes:])
			hwy.Store(hwy.Add(vC, acc13), c[cRow1Start+j+3*lanes:])
		}

		// Remainder: single accumulator per remaining vector strip
		for ; j+lanes <= blockDim; j += lanes {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			for k := range blockDim {
				aTRowK := k * blockDim
				vA0 := hwy.Set(aT[aTRowK+i])
				vA1 := hwy.Set(aT[aTRowK+i+1])
				vB := hwy.Load(b[k*blockDim+j:])
				acc0 = hwy.MulAdd(vA0, vB, acc0)
				acc1 = hwy.MulAdd(vA1, vB, acc1)
			}
			vC := hwy.Load(c[cRow0Start+j:])
			hwy.Store(hwy.Add(vC, acc0), c[cRow0Start+j:])
			vC = hwy.Load(c[cRow1Start+j:])
			hwy.Store(hwy.Add(vC, acc1), c[cRow1Start+j:])
		}

		// Scalar tail
		for ; j < blockDim; j++ {
			var sum0, sum1 T
			for k := range blockDim {
				aTRowK := k * blockDim
				bkj := b[k*blockDim+j]
				sum0 += aT[aTRowK+i] * bkj
				sum1 += aT[aTRowK+i+1] * bkj
			}
			c[cRow0Start+j] += sum0
			c[cRow1Start+j] += sum1
		}
	}

	// Handle odd row if blockDim is odd
	if i < blockDim {
		cRowStart := i * blockDim
		var j int
		for j = 0; j+tileJ <= blockDim; j += tileJ {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()
			for k := range blockDim {
				aik := aT[k*blockDim+i]
				vA := hwy.Set(aik)
				bRowStart := k * blockDim
				acc0 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j:]), acc0)
				acc1 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j+lanes:]), acc1)
				acc2 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j+2*lanes:]), acc2)
				acc3 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j+3*lanes:]), acc3)
			}
			vC := hwy.Load(c[cRowStart+j:])
			hwy.Store(hwy.Add(vC, acc0), c[cRowStart+j:])
			vC = hwy.Load(c[cRowStart+j+lanes:])
			hwy.Store(hwy.Add(vC, acc1), c[cRowStart+j+lanes:])
			vC = hwy.Load(c[cRowStart+j+2*lanes:])
			hwy.Store(hwy.Add(vC, acc2), c[cRowStart+j+2*lanes:])
			vC = hwy.Load(c[cRowStart+j+3*lanes:])
			hwy.Store(hwy.Add(vC, acc3), c[cRowStart+j+3*lanes:])
		}
		for ; j+lanes <= blockDim; j += lanes {
			acc := hwy.Zero[T]()
			for k := range blockDim {
				acc = hwy.MulAdd(hwy.Set(aT[k*blockDim+i]), hwy.Load(b[k*blockDim+j:]), acc)
			}
			vC := hwy.Load(c[cRowStart+j:])
			hwy.Store(hwy.Add(vC, acc), c[cRowStart+j:])
		}
		for ; j < blockDim; j++ {
			var sum T
			for k := range blockDim {
				sum += aT[k*blockDim+i] * b[k*blockDim+j]
			}
			c[cRowStart+j] += sum
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
// Uses register-blocked accumulators with 4-way row unrolling
// (4 rows × 4 column strips = 16 accumulators). Reuses B vector loads
// across all 4 rows for maximum throughput.
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
	tileJ := 4 * lanes

	// Process 4 rows of C at a time
	var i int
	for i = 0; i+3 < blockDim; i += 4 {
		cRow0 := i * blockDim
		cRow1 := (i + 1) * blockDim
		cRow2 := (i + 2) * blockDim
		cRow3 := (i + 3) * blockDim

		// Tiled J loop — 16 accumulators (4 rows × 4 strips)
		var j int
		for j = 0; j+tileJ <= blockDim; j += tileJ {
			// 16 accumulators: row r, strip s = acc_rs
			acc00 := hwy.Zero[T]()
			acc01 := hwy.Zero[T]()
			acc02 := hwy.Zero[T]()
			acc03 := hwy.Zero[T]()
			acc10 := hwy.Zero[T]()
			acc11 := hwy.Zero[T]()
			acc12 := hwy.Zero[T]()
			acc13 := hwy.Zero[T]()
			acc20 := hwy.Zero[T]()
			acc21 := hwy.Zero[T]()
			acc22 := hwy.Zero[T]()
			acc23 := hwy.Zero[T]()
			acc30 := hwy.Zero[T]()
			acc31 := hwy.Zero[T]()
			acc32 := hwy.Zero[T]()
			acc33 := hwy.Zero[T]()

			for k := range blockDim {
				aTRowK := k * blockDim
				vA0 := hwy.Set(aT[aTRowK+i])
				vA1 := hwy.Set(aT[aTRowK+i+1])
				vA2 := hwy.Set(aT[aTRowK+i+2])
				vA3 := hwy.Set(aT[aTRowK+i+3])
				bRowStart := k * blockDim
				vB0 := hwy.Load(b[bRowStart+j:])
				vB1 := hwy.Load(b[bRowStart+j+lanes:])
				vB2 := hwy.Load(b[bRowStart+j+2*lanes:])
				vB3 := hwy.Load(b[bRowStart+j+3*lanes:])
				acc00 = hwy.MulAdd(vA0, vB0, acc00)
				acc01 = hwy.MulAdd(vA0, vB1, acc01)
				acc02 = hwy.MulAdd(vA0, vB2, acc02)
				acc03 = hwy.MulAdd(vA0, vB3, acc03)
				acc10 = hwy.MulAdd(vA1, vB0, acc10)
				acc11 = hwy.MulAdd(vA1, vB1, acc11)
				acc12 = hwy.MulAdd(vA1, vB2, acc12)
				acc13 = hwy.MulAdd(vA1, vB3, acc13)
				acc20 = hwy.MulAdd(vA2, vB0, acc20)
				acc21 = hwy.MulAdd(vA2, vB1, acc21)
				acc22 = hwy.MulAdd(vA2, vB2, acc22)
				acc23 = hwy.MulAdd(vA2, vB3, acc23)
				acc30 = hwy.MulAdd(vA3, vB0, acc30)
				acc31 = hwy.MulAdd(vA3, vB1, acc31)
				acc32 = hwy.MulAdd(vA3, vB2, acc32)
				acc33 = hwy.MulAdd(vA3, vB3, acc33)
			}

			// Add accumulators to existing C
			vC := hwy.Load(c[cRow0+j:])
			hwy.Store(hwy.Add(vC, acc00), c[cRow0+j:])
			vC = hwy.Load(c[cRow0+j+lanes:])
			hwy.Store(hwy.Add(vC, acc01), c[cRow0+j+lanes:])
			vC = hwy.Load(c[cRow0+j+2*lanes:])
			hwy.Store(hwy.Add(vC, acc02), c[cRow0+j+2*lanes:])
			vC = hwy.Load(c[cRow0+j+3*lanes:])
			hwy.Store(hwy.Add(vC, acc03), c[cRow0+j+3*lanes:])

			vC = hwy.Load(c[cRow1+j:])
			hwy.Store(hwy.Add(vC, acc10), c[cRow1+j:])
			vC = hwy.Load(c[cRow1+j+lanes:])
			hwy.Store(hwy.Add(vC, acc11), c[cRow1+j+lanes:])
			vC = hwy.Load(c[cRow1+j+2*lanes:])
			hwy.Store(hwy.Add(vC, acc12), c[cRow1+j+2*lanes:])
			vC = hwy.Load(c[cRow1+j+3*lanes:])
			hwy.Store(hwy.Add(vC, acc13), c[cRow1+j+3*lanes:])

			vC = hwy.Load(c[cRow2+j:])
			hwy.Store(hwy.Add(vC, acc20), c[cRow2+j:])
			vC = hwy.Load(c[cRow2+j+lanes:])
			hwy.Store(hwy.Add(vC, acc21), c[cRow2+j+lanes:])
			vC = hwy.Load(c[cRow2+j+2*lanes:])
			hwy.Store(hwy.Add(vC, acc22), c[cRow2+j+2*lanes:])
			vC = hwy.Load(c[cRow2+j+3*lanes:])
			hwy.Store(hwy.Add(vC, acc23), c[cRow2+j+3*lanes:])

			vC = hwy.Load(c[cRow3+j:])
			hwy.Store(hwy.Add(vC, acc30), c[cRow3+j:])
			vC = hwy.Load(c[cRow3+j+lanes:])
			hwy.Store(hwy.Add(vC, acc31), c[cRow3+j+lanes:])
			vC = hwy.Load(c[cRow3+j+2*lanes:])
			hwy.Store(hwy.Add(vC, acc32), c[cRow3+j+2*lanes:])
			vC = hwy.Load(c[cRow3+j+3*lanes:])
			hwy.Store(hwy.Add(vC, acc33), c[cRow3+j+3*lanes:])
		}

		// Remainder: single vector strip, 4 rows
		for ; j+lanes <= blockDim; j += lanes {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()
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
			hwy.Store(hwy.Add(vC, acc0), c[cRow0+j:])
			vC = hwy.Load(c[cRow1+j:])
			hwy.Store(hwy.Add(vC, acc1), c[cRow1+j:])
			vC = hwy.Load(c[cRow2+j:])
			hwy.Store(hwy.Add(vC, acc2), c[cRow2+j:])
			vC = hwy.Load(c[cRow3+j:])
			hwy.Store(hwy.Add(vC, acc3), c[cRow3+j:])
		}

		// Scalar tail
		for ; j < blockDim; j++ {
			var sum0, sum1, sum2, sum3 T
			for k := range blockDim {
				aTRowK := k * blockDim
				bkj := b[k*blockDim+j]
				sum0 += aT[aTRowK+i] * bkj
				sum1 += aT[aTRowK+i+1] * bkj
				sum2 += aT[aTRowK+i+2] * bkj
				sum3 += aT[aTRowK+i+3] * bkj
			}
			c[cRow0+j] += sum0
			c[cRow1+j] += sum1
			c[cRow2+j] += sum2
			c[cRow3+j] += sum3
		}
	}

	// Handle remaining rows (0-3 rows)
	for ; i < blockDim; i++ {
		cRowStart := i * blockDim
		var j int
		for j = 0; j+tileJ <= blockDim; j += tileJ {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()
			for k := range blockDim {
				aik := aT[k*blockDim+i]
				vA := hwy.Set(aik)
				bRowStart := k * blockDim
				acc0 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j:]), acc0)
				acc1 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j+lanes:]), acc1)
				acc2 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j+2*lanes:]), acc2)
				acc3 = hwy.MulAdd(vA, hwy.Load(b[bRowStart+j+3*lanes:]), acc3)
			}
			vC := hwy.Load(c[cRowStart+j:])
			hwy.Store(hwy.Add(vC, acc0), c[cRowStart+j:])
			vC = hwy.Load(c[cRowStart+j+lanes:])
			hwy.Store(hwy.Add(vC, acc1), c[cRowStart+j+lanes:])
			vC = hwy.Load(c[cRowStart+j+2*lanes:])
			hwy.Store(hwy.Add(vC, acc2), c[cRowStart+j+2*lanes:])
			vC = hwy.Load(c[cRowStart+j+3*lanes:])
			hwy.Store(hwy.Add(vC, acc3), c[cRowStart+j+3*lanes:])
		}
		for ; j+lanes <= blockDim; j += lanes {
			acc := hwy.Zero[T]()
			for k := range blockDim {
				acc = hwy.MulAdd(hwy.Set(aT[k*blockDim+i]), hwy.Load(b[k*blockDim+j:]), acc)
			}
			vC := hwy.Load(c[cRowStart+j:])
			hwy.Store(hwy.Add(vC, acc), c[cRowStart+j:])
		}
		for ; j < blockDim; j++ {
			var sum T
			for k := range blockDim {
				sum += aT[k*blockDim+i] * b[k*blockDim+j]
			}
			c[cRowStart+j] += sum
		}
	}
}
