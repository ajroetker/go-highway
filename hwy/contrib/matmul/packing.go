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

//go:generate go run ../../../cmd/hwygen -input packing.go -dispatch packing -output . -targets avx2,avx512,neon,fallback

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

// BasePackRHS packs a panel of the RHS matrix (B) into a cache-friendly layout.
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
// where num_micro_panels = ceil(panelCols / Nr)
//
// The K-first layout within micro-panels ensures sequential access
// when iterating over K in the inner loop.
//
// Parameters:
//   - b: Input matrix B in row-major order
//   - packed: Output buffer, must have size >= ceil(panelCols/Nr) * panelK * Nr
//   - k, n: Dimensions of the full B matrix
//   - rowStart: Starting row of the panel to pack (K-dimension offset)
//   - colStart: Starting column of the panel to pack
//   - panelK: Number of rows to pack (K dimension)
//   - panelCols: Number of columns to pack
//   - nr: Micro-tile column dimension
//
// Returns the number of active columns in the last micro-panel (may be < Nr).
func BasePackRHS[T hwy.Floats](b, packed []T, k, n, rowStart, colStart, panelK, panelCols, nr int) int {
	numMicroPanels := (panelCols + nr - 1) / nr
	activeColsLast := panelCols - (numMicroPanels-1)*nr

	// Pack complete micro-panels
	fullPanels := numMicroPanels
	if activeColsLast < nr {
		fullPanels--
	}

	packIdx := 0
	for panel := 0; panel < fullPanels; panel++ {
		baseCol := colStart + panel*nr
		for kk := range panelK {
			bRowStart := (rowStart + kk) * n
			for c := range nr {
				packed[packIdx] = b[bRowStart+baseCol+c]
				packIdx++
			}
		}
	}

	// Pack partial last micro-panel (if any)
	if activeColsLast < nr && activeColsLast > 0 {
		baseCol := colStart + fullPanels*nr
		for kk := range panelK {
			bRowStart := (rowStart + kk) * n
			// Pack active columns
			for c := range activeColsLast {
				packed[packIdx] = b[bRowStart+baseCol+c]
				packIdx++
			}
			// Zero-pad remaining columns in micro-panel
			for c := activeColsLast; c < nr; c++ {
				packed[packIdx] = 0
				packIdx++
			}
		}
	}

	return activeColsLast
}

// BasePackLHSVec packs LHS using SIMD when Mr aligns with vector width.
// This is a vectorized version of BasePackLHS for better performance.
func BasePackLHSVec[T hwy.Floats](a, packed []T, m, k, rowStart, colStart, panelRows, panelK, mr int) int {
	// For now, fall back to scalar implementation.
	// Future optimization: use SIMD gather or interleaved loads when beneficial.
	return BasePackLHS(a, packed, m, k, rowStart, colStart, panelRows, panelK, mr)
}

// BasePackRHSVec packs RHS using SIMD loads for contiguous data.
// This is a vectorized version of BasePackRHS for better performance.
func BasePackRHSVec[T hwy.Floats](b, packed []T, k, n, rowStart, colStart, panelK, panelCols, nr int) int {
	lanes := hwy.Zero[T]().NumLanes()

	// If nr is a multiple of lanes and cols are contiguous, use SIMD loads
	if nr >= lanes && nr%lanes == 0 {
		numMicroPanels := (panelCols + nr - 1) / nr
		activeColsLast := panelCols - (numMicroPanels-1)*nr

		fullPanels := numMicroPanels
		if activeColsLast < nr {
			fullPanels--
		}

		packIdx := 0
		for panel := 0; panel < fullPanels; panel++ {
			baseCol := colStart + panel*nr
			for kk := range panelK {
				bRowStart := (rowStart + kk) * n
				// SIMD copy of nr elements (nr/lanes vectors)
				for c := 0; c < nr; c += lanes {
					v := hwy.Load(b[bRowStart+baseCol+c:])
					hwy.Store(v, packed[packIdx+c:])
				}
				packIdx += nr
			}
		}

		// Handle partial panel with scalar code
		if activeColsLast < nr && activeColsLast > 0 {
			baseCol := colStart + fullPanels*nr
			for kk := range panelK {
				bRowStart := (rowStart + kk) * n
				for c := range activeColsLast {
					packed[packIdx] = b[bRowStart+baseCol+c]
					packIdx++
				}
				for c := activeColsLast; c < nr; c++ {
					packed[packIdx] = 0
					packIdx++
				}
			}
		}

		return activeColsLast
	}

	// Fall back to scalar for non-aligned cases
	return BasePackRHS(b, packed, k, n, rowStart, colStart, panelK, panelCols, nr)
}
