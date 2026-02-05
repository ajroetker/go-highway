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

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// Parallel tuning parameters for packed matmul
const (
	// MinPackedParallelOps is the minimum number of operations before parallelizing.
	// Packed matmul has higher overhead, so we need larger matrices to benefit.
	MinPackedParallelOps = 256 * 256 * 256

	// PackedRowsPerStrip defines how many rows each worker processes at a time.
	// Should be a multiple of Mc for best cache utilization.
	PackedRowsPerStrip = 256
)

// ParallelPackedMatMul computes C = A * B using parallel execution with
// the GotoBLAS-style 5-loop algorithm.
//
// Work is divided into horizontal strips along the M dimension. Each worker
// has its own packing buffers to avoid contention.
//
// For small matrices or nil pool, falls back to single-threaded PackedMatMul.
//
// Parameters:
//   - pool: Persistent worker pool for parallel execution
//   - a: Input matrix A in row-major order (M × K)
//   - b: Input matrix B in row-major order (K × N)
//   - c: Output matrix C in row-major order (M × N), will be zeroed
//   - m, n, k: Matrix dimensions
func ParallelPackedMatMul[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, m, n, k int) {
	totalOps := m * n * k

	// For small matrices or nil pool, use single-threaded version
	if pool == nil || totalOps < MinPackedParallelOps {
		PackedMatMul(a, b, c, m, n, k)
		return
	}

	params := getCacheParams[T]()

	// Calculate number of row strips
	// Use cache-aligned strip size for better performance
	stripSize := max(params.Mc, PackedRowsPerStrip)
	numStrips := (m + stripSize - 1) / stripSize

	// Zero the output matrix once (shared across all workers)
	zeroMatrix(c, m*n)

	// Workers process strips via atomic work stealing
	pool.ParallelForAtomic(numStrips, func(strip int) {
		// Each worker has its own packing buffers
		packedA := make([]T, params.PackedASize())
		packedB := make([]T, params.PackedBSize())

		rowStart := strip * stripSize
		rowEnd := min(rowStart+stripSize, m)

		// Process this strip using the worker's buffers
		processStripPacked(a, b, c, m, n, k, rowStart, rowEnd, packedA, packedB, params)
	})
}

// processStripPacked computes a horizontal strip of C using packed matmul.
// C[rowStart:rowEnd, :] += A[rowStart:rowEnd, :] * B
//
// Note: Assumes C is already zeroed.
func processStripPacked[T hwy.Floats](a, b, c []T, m, n, k, rowStart, rowEnd int, packedA, packedB []T, params CacheParams) {
	mr, nr := params.Mr, params.Nr
	kc, mc, nc := params.Kc, params.Mc, params.Nc

	// Loop 5: B panels (L3 blocking)
	for jc := 0; jc < n; jc += nc {
		jcEnd := min(jc+nc, n)
		panelCols := jcEnd - jc

		// Loop 4: K blocking (L1)
		for pc := 0; pc < k; pc += kc {
			pcEnd := min(pc+kc, k)
			panelK := pcEnd - pc

			// Pack B panel
			PackRHS(b, packedB, k, n, pc, jc, panelK, panelCols, nr)

			// Loop 3: A panels within this strip (L2 blocking)
			for ic := rowStart; ic < rowEnd; ic += mc {
				icEnd := min(ic+mc, rowEnd)
				panelRows := icEnd - ic

				// Pack A panel from this strip
				activeRowsLast := PackLHS(a, packedA, m, k, ic, pc, panelRows, panelK, mr)

				// GEBP for this panel
				gebp(packedA, packedB, c, n, ic, jc, panelRows, panelCols, panelK, mr, nr, activeRowsLast)
			}
		}
	}
}

// ParallelPackedMatMulSharedB is an optimized parallel version that packs B
// once and shares it across all workers.
//
// This is more efficient when M >> N, as B packing overhead is amortized.
// However, it requires more memory for the shared packed B buffer.
func ParallelPackedMatMulSharedB[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, m, n, k int) {
	totalOps := m * n * k

	// For small matrices or nil pool, use single-threaded version
	if pool == nil || totalOps < MinPackedParallelOps {
		PackedMatMul(a, b, c, m, n, k)
		return
	}

	params := getCacheParams[T]()

	// Calculate number of row strips
	stripSize := max(params.Mc, PackedRowsPerStrip)
	numStrips := (m + stripSize - 1) / stripSize

	// Allocate shared packed B buffer (larger, for entire B)
	// Layout: [ceil(N/Nr), K, Nr]
	numBPanels := (n + params.Nr - 1) / params.Nr
	sharedPackedBSize := numBPanels * k * params.Nr
	sharedPackedB := make([]T, sharedPackedBSize)

	// Pack entire B matrix (single-threaded, done once)
	packEntireRHS(b, sharedPackedB, k, n, params.Nr)

	// Zero the output matrix
	zeroMatrix(c, m*n)

	// Workers process strips using shared packed B
	pool.ParallelFor(numStrips, func(start, end int) {
		// Each worker only needs packed A buffer
		packedA := make([]T, params.PackedASize())

		for strip := start; strip < end; strip++ {
			rowStart := strip * stripSize
			rowEnd := min(rowStart+stripSize, m)

			processStripWithSharedB(a, sharedPackedB, c, m, n, k, rowStart, rowEnd, packedA, params)
		}
	})
}

// packEntireRHS packs the entire RHS matrix B for shared access.
// This packs all K rows and all N columns.
func packEntireRHS[T hwy.Floats](b, packedB []T, k, n, nr int) {
	// Pack entire B: all K rows, all N columns
	PackRHS(b, packedB, k, n, 0, 0, k, n, nr)
}

// processStripWithSharedB computes a strip using pre-packed B.
func processStripWithSharedB[T hwy.Floats](a, sharedPackedB, c []T, m, n, k, rowStart, rowEnd int, packedA []T, params CacheParams) {
	mr, nr := params.Mr, params.Nr
	mc := params.Mc

	numBPanels := (n + nr - 1) / nr

	// Process all B micro-panels
	for jPanel := range numBPanels {
		jr := jPanel * nr
		bPanelOffset := jPanel * k * nr

		// Determine active columns
		activeCols := min(nr, n-jr)

		// Loop over A panels within this strip
		for ic := rowStart; ic < rowEnd; ic += mc {
			icEnd := min(ic+mc, rowEnd)
			panelRows := icEnd - ic

			// Pack A panel (full K dimension, colStart=0)
			activeRowsLast := PackLHS(a, packedA, m, k, ic, 0, panelRows, k, mr)

			// Process micro-tiles
			numMicroPanelsA := (panelRows + mr - 1) / mr
			for iPanel := range numMicroPanelsA {
				ir := ic + iPanel*mr
				aPanelOffset := iPanel * k * mr

				activeRows := mr
				if iPanel == numMicroPanelsA-1 {
					activeRows = activeRowsLast
				}

				if activeRows == mr && activeCols == nr {
					PackedMicroKernel(packedA[aPanelOffset:], sharedPackedB[bPanelOffset:], c, n, ir, jr, k, mr, nr)
				} else {
					PackedMicroKernelPartial(packedA[aPanelOffset:], sharedPackedB[bPanelOffset:], c, n, ir, jr, k, mr, nr, activeRows, activeCols)
				}
			}
		}
	}
}
