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

//go:generate go run ../../../cmd/hwygen -input packed_kernel.go -dispatch package_kernel -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

// BasePackedMicroKernel computes C[ir:ir+Mr, jr:jr+Nr] += packedA * packedB
// where packedA and packedB are in the packed layout from BasePackLHS/BasePackRHS.
//
// This is the innermost kernel of the GotoBLAS 5-loop algorithm. It operates on
// pre-packed data to achieve maximum memory bandwidth utilization:
//
//   - packedA: Kc values for Mr rows, laid out as [Kc, Mr] (K-first)
//   - packedB: Kc values for Nr cols, laid out as [Kc, Nr] (K-first)
//
// The kernel uses a 4×2-vector accumulator pattern:
//   - 4 rows (Mr=4) × 2 vector widths (Nr=2*lanes)
//   - 8 FMA operations per K iteration
//   - Accumulators held in registers across entire Kc loop
//
// Parameters:
//   - packedA: Packed A micro-panel, size Kc * Mr
//   - packedB: Packed B micro-panel, size Kc * Nr
//   - c: Output matrix C in row-major order
//   - n: Leading dimension of C (number of columns)
//   - ir: Starting row in C
//   - jr: Starting column in C
//   - kc: K-dimension of the packed panels
//   - mr: Number of rows (must be 4 for this kernel)
//   - nr: Number of columns (must be 2*lanes for this kernel)
func BasePackedMicroKernel[T hwy.Floats](packedA, packedB []T, c []T, n, ir, jr, kc, mr, nr int) {
	lanes := hwy.Zero[T]().NumLanes()

	// Verify dimensions match expected micro-tile size
	if mr != 4 || nr != lanes*2 {
		// Fall back to general implementation
		basePackedMicroKernelGeneral(packedA, packedB, c, n, ir, jr, kc, mr, nr)
		return
	}

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

	// K-loop: iterate through packed panels
	// packedA layout: [Kc, Mr] - consecutive Mr values for each k
	// packedB layout: [Kc, Nr] - consecutive Nr values for each k
	aIdx := 0
	bIdx := 0
	for range kc {
		// Load Mr values from packed A (contiguous)
		a0 := packedA[aIdx]
		a1 := packedA[aIdx+1]
		a2 := packedA[aIdx+2]
		a3 := packedA[aIdx+3]
		aIdx += 4

		vA0 := hwy.Set(a0)
		vA1 := hwy.Set(a1)
		vA2 := hwy.Set(a2)
		vA3 := hwy.Set(a3)

		// Load Nr values from packed B (2 vectors, contiguous)
		// Use Load for pointer-based access without bounds checking
		vB0 := hwy.Load(packedB[bIdx:])
		vB1 := hwy.Load(packedB[bIdx+lanes:])
		bIdx += nr

		// 8 FMA operations
		acc00 = hwy.MulAdd(vA0, vB0, acc00)
		acc01 = hwy.MulAdd(vA0, vB1, acc01)
		acc10 = hwy.MulAdd(vA1, vB0, acc10)
		acc11 = hwy.MulAdd(vA1, vB1, acc11)
		acc20 = hwy.MulAdd(vA2, vB0, acc20)
		acc21 = hwy.MulAdd(vA2, vB1, acc21)
		acc30 = hwy.MulAdd(vA3, vB0, acc30)
		acc31 = hwy.MulAdd(vA3, vB1, acc31)
	}

	// Write back: accumulate into C
	cRow0 := ir * n
	cRow1 := (ir + 1) * n
	cRow2 := (ir + 2) * n
	cRow3 := (ir + 3) * n

	// Load existing C values, add accumulators, store back
	// Use Load/Store for pointer-based access without bounds checking
	vC := hwy.Load(c[cRow0+jr:])
	vC = hwy.Add(vC, acc00)
	hwy.Store(vC, c[cRow0+jr:])

	vC = hwy.Load(c[cRow0+jr+lanes:])
	vC = hwy.Add(vC, acc01)
	hwy.Store(vC, c[cRow0+jr+lanes:])

	vC = hwy.Load(c[cRow1+jr:])
	vC = hwy.Add(vC, acc10)
	hwy.Store(vC, c[cRow1+jr:])

	vC = hwy.Load(c[cRow1+jr+lanes:])
	vC = hwy.Add(vC, acc11)
	hwy.Store(vC, c[cRow1+jr+lanes:])

	vC = hwy.Load(c[cRow2+jr:])
	vC = hwy.Add(vC, acc20)
	hwy.Store(vC, c[cRow2+jr:])

	vC = hwy.Load(c[cRow2+jr+lanes:])
	vC = hwy.Add(vC, acc21)
	hwy.Store(vC, c[cRow2+jr+lanes:])

	vC = hwy.Load(c[cRow3+jr:])
	vC = hwy.Add(vC, acc30)
	hwy.Store(vC, c[cRow3+jr:])

	vC = hwy.Load(c[cRow3+jr+lanes:])
	vC = hwy.Add(vC, acc31)
	hwy.Store(vC, c[cRow3+jr+lanes:])
}

// basePackedMicroKernelGeneral handles arbitrary micro-tile sizes.
// Used as fallback when Mr != 4 or Nr != 2*lanes.
func basePackedMicroKernelGeneral[T hwy.Floats](packedA, packedB []T, c []T, n, ir, jr, kc, mr, nr int) {
	lanes := hwy.Zero[T]().NumLanes()

	// Process rows one at a time
	for r := range mr {
		cRowStart := (ir + r) * n

		// Process columns in vector-width chunks
		var col int
		for col = 0; col+lanes <= nr; col += lanes {
			acc := hwy.Zero[T]()

			// K-loop
			for p := range kc {
				aVal := packedA[p*mr+r]
				vA := hwy.Set(aVal)
				vB := hwy.Load(packedB[p*nr+col:])
				acc = hwy.MulAdd(vA, vB, acc)
			}

			// Accumulate into C
			vC := hwy.Load(c[cRowStart+jr+col:])
			vC = hwy.Add(vC, acc)
			hwy.Store(vC, c[cRowStart+jr+col:])
		}

		// Scalar tail for remaining columns
		for ; col < nr; col++ {
			var sum T
			for p := range kc {
				sum += packedA[p*mr+r] * packedB[p*nr+col]
			}
			c[cRowStart+jr+col] += sum
		}
	}
}

// BasePackedMicroKernelPartial handles edge cases where the micro-tile
// extends beyond the matrix bounds.
//
// Parameters:
//   - activeRows: Actual number of valid rows (may be < Mr)
//   - activeCols: Actual number of valid columns (may be < Nr)
//
// The packed data is still Mr × Nr with zero padding, but we only
// write back the active portion to C.
func BasePackedMicroKernelPartial[T hwy.Floats](packedA, packedB []T, c []T, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	lanes := hwy.Zero[T]().NumLanes()

	// For each active row
	for r := range activeRows {
		cRowStart := (ir + r) * n

		// Process columns in vector-width chunks
		var col int
		for col = 0; col+lanes <= activeCols; col += lanes {
			acc := hwy.Zero[T]()

			// K-loop
			for p := range kc {
				aVal := packedA[p*mr+r]
				vA := hwy.Set(aVal)
				vB := hwy.Load(packedB[p*nr+col:])
				acc = hwy.MulAdd(vA, vB, acc)
			}

			// Accumulate into C
			vC := hwy.Load(c[cRowStart+jr+col:])
			vC = hwy.Add(vC, acc)
			hwy.Store(vC, c[cRowStart+jr+col:])
		}

		// Scalar tail for remaining columns
		for ; col < activeCols; col++ {
			var sum T
			for p := range kc {
				sum += packedA[p*mr+r] * packedB[p*nr+col]
			}
			c[cRowStart+jr+col] += sum
		}
	}
}
