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

//go:generate go run ../../../cmd/hwygen -input packing.go -dispatch packing -output . -targets avx2,avx512,neon:asm,fallback

import "github.com/ajroetker/go-highway/hwy"

// BasePackLHS packs a panel of the LHS matrix (A) into a cache-friendly layout.
//
// Input A is M x K in row-major order. This function packs a panel of rows
// [rowStart, rowStart+panelRows) and columns [colStart, colStart+panelK).
//
// The packed layout is organized as micro-panels of Mr rows each:
//   - For each micro-panel i (rows i*Mr to (i+1)*Mr):
//   - For each k in [0, panelK):
//   - Store A[rowStart+i*Mr+0, colStart+k], ..., A[rowStart+i*Mr+Mr-1, colStart+k]
//
// This gives memory layout: [num_micro_panels, panelK, Mr]
// where num_micro_panels = ceil(panelRows / Mr)
//
// The K-first layout within micro-panels optimizes for the inner loop
// which iterates over K and needs contiguous A values for each k.
//
// Parameters:
//   - a: Input matrix A in row-major order
//   - packed: Output buffer, must have size >= ceil(panelRows/Mr) * panelK * Mr
//   - m, k: Dimensions of the full A matrix
//   - rowStart: Starting row of the panel to pack
//   - colStart: Starting column of the panel to pack (K-dimension offset)
//   - panelRows: Number of rows to pack
//   - panelK: Number of columns to pack (K dimension)
//   - mr: Micro-tile row dimension
//
// Returns the number of active rows in the last micro-panel (may be < Mr).
func BasePackLHS[T hwy.Floats](a, packed []T, m, k, rowStart, colStart, panelRows, panelK, mr int) int {
	numMicroPanels := (panelRows + mr - 1) / mr
	activeRowsLast := panelRows - (numMicroPanels-1)*mr

	// Pack complete micro-panels
	fullPanels := numMicroPanels
	if activeRowsLast < mr {
		fullPanels--
	}

	packIdx := 0
	for panel := 0; panel < fullPanels; panel++ {
		baseRow := rowStart + panel*mr
		for kk := range panelK {
			for r := range mr {
				packed[packIdx] = a[(baseRow+r)*k+colStart+kk]
				packIdx++
			}
		}
	}

	// Pack partial last micro-panel (if any)
	if activeRowsLast < mr && activeRowsLast > 0 {
		baseRow := rowStart + fullPanels*mr
		for kk := range panelK {
			// Pack active rows
			for r := range activeRowsLast {
				packed[packIdx] = a[(baseRow+r)*k+colStart+kk]
				packIdx++
			}
			// Zero-pad remaining rows in micro-panel
			for r := activeRowsLast; r < mr; r++ {
				packed[packIdx] = 0
				packIdx++
			}
		}
	}

	return activeRowsLast
}

// BasePackLHSVec packs LHS using SIMD butterfly transpose for mr=4.
//
// Instead of gathering one element per row per k-value (strided access),
// this loads `lanes` contiguous k-values from each of the 4 rows and
// transposes them in-register using a 2-stage butterfly pattern
// (InterleaveLower/InterleaveUpper), producing the packed [panelK, mr]
// layout directly.
func BasePackLHSVec[T hwy.Floats](a, packed []T, m, k, rowStart, colStart, panelRows, panelK, mr int) int {
	if mr != 4 {
		return BasePackLHS(a, packed, m, k, rowStart, colStart, panelRows, panelK, mr)
	}

	lanes := hwy.Zero[T]().NumLanes()
	numMicroPanels := (panelRows + mr - 1) / mr
	activeRowsLast := panelRows - (numMicroPanels-1)*mr

	fullPanels := numMicroPanels
	if activeRowsLast < mr {
		fullPanels--
	}

	packIdx := 0

	for panel := 0; panel < fullPanels; panel++ {
		baseRow := rowStart + panel*mr
		row0 := baseRow*k + colStart
		row1 := (baseRow+1)*k + colStart
		row2 := (baseRow+2)*k + colStart
		row3 := (baseRow+3)*k + colStart

		// SIMD path: process lanes k-values at a time with butterfly transpose
		var kk int
		for kk = 0; kk+lanes <= panelK; kk += lanes {
			// Load lanes contiguous k-values from each of 4 rows
			r0 := hwy.Load(a[row0+kk:])
			r1 := hwy.Load(a[row1+kk:])
			r2 := hwy.Load(a[row2+kk:])
			r3 := hwy.Load(a[row3+kk:])

			// 2-stage butterfly transpose (log2(4) = 2 stages)
			// Stage 1: stride=2, pair rows (0,2) and (1,3)
			t0 := hwy.InterleaveLower(r0, r2)
			t2 := hwy.InterleaveUpper(r0, r2)
			t1 := hwy.InterleaveLower(r1, r3)
			t3 := hwy.InterleaveUpper(r1, r3)

			// Stage 2: stride=1
			c0 := hwy.InterleaveLower(t0, t1)
			c1 := hwy.InterleaveUpper(t0, t1)
			c2 := hwy.InterleaveLower(t2, t3)
			c3 := hwy.InterleaveUpper(t2, t3)

			// Store: lanes k-groups of mr=4 elements = lanes*4 elements
			hwy.Store(c0, packed[packIdx:])
			hwy.Store(c1, packed[packIdx+lanes:])
			hwy.Store(c2, packed[packIdx+2*lanes:])
			hwy.Store(c3, packed[packIdx+3*lanes:])
			packIdx += lanes * mr
		}

		// Scalar tail for remaining k-values
		for ; kk < panelK; kk++ {
			packed[packIdx] = a[row0+kk]
			packed[packIdx+1] = a[row1+kk]
			packed[packIdx+2] = a[row2+kk]
			packed[packIdx+3] = a[row3+kk]
			packIdx += mr
		}
	}

	// Pack partial last micro-panel with scalar code (zero-padding)
	if activeRowsLast < mr && activeRowsLast > 0 {
		baseRow := rowStart + fullPanels*mr
		for kk := range panelK {
			for r := range activeRowsLast {
				packed[packIdx] = a[(baseRow+r)*k+colStart+kk]
				packIdx++
			}
			for r := activeRowsLast; r < mr; r++ {
				packed[packIdx] = 0
				packIdx++
			}
		}
	}

	return activeRowsLast
}

// BasePackRHSVec packs a panel of the RHS matrix (B) into a cache-friendly layout
// using SIMD loads for full micro-panels and scalar code for partial ones.
//
// Input B is K x N in row-major order. This function packs a panel of rows
// [rowStart, rowStart+panelK) and columns [colStart, colStart+panelCols).
//
// The packed layout is organized as micro-panels of Nr columns each:
//   - For each micro-panel j (cols j*Nr to (j+1)*Nr):
//   - For each k in [0, panelK):
//   - Store B[rowStart+k, colStart+j*Nr+0], ..., B[rowStart+k, colStart+j*Nr+Nr-1]
//
// This gives memory layout: [num_micro_panels, panelK, Nr]
//
// Parameters:
//   - b: Input matrix B in row-major order
//   - packed: Output buffer
//   - n: Number of columns in B (row stride)
//   - rowStart: Starting row of the panel to pack (K-dimension offset)
//   - colStart: Starting column of the panel to pack
//   - panelK: Number of rows to pack (K dimension)
//   - panelCols: Number of columns to pack
//   - nr: Micro-tile column dimension
//
// Returns the number of active columns in the last micro-panel (may be < Nr).
func BasePackRHSVec[T hwy.Floats](b, packed []T, n, rowStart, colStart, panelK, panelCols, nr int) int {
	lanes := hwy.Zero[T]().NumLanes()
	numMicroPanels := (panelCols + nr - 1) / nr
	activeColsLast := panelCols - (numMicroPanels-1)*nr

	dstIdx := 0
	for strip := 0; strip < panelCols; strip += nr {
		validCols := min(nr, panelCols-strip)
		baseCol := colStart + strip

		// SIMD fast path: full strip with nr aligned to vector width
		if validCols == nr && nr >= lanes && nr%lanes == 0 {
			for kk := range panelK {
				srcRow := (rowStart + kk) * n
				for c := 0; c < nr; c += lanes {
					v := hwy.Load(b[srcRow+baseCol+c:])
					hwy.Store(v, packed[dstIdx+c:])
				}
				dstIdx += nr
			}
			continue
		}

		// Scalar path: partial strip with zero-padding
		for kk := range panelK {
			srcRow := (rowStart + kk) * n
			for c := range validCols {
				packed[dstIdx] = b[srcRow+baseCol+c]
				dstIdx++
			}
			for c := validCols; c < nr; c++ {
				packed[dstIdx] = 0
				dstIdx++
			}
		}
	}

	_ = numMicroPanels
	return activeColsLast
}
