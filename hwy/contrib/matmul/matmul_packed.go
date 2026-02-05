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

//go:generate go run ../../../cmd/hwygen -input matmul_packed.go -dispatch packedmatmul -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

// BasePackedMatMul computes C = A * B using the GotoBLAS-style 5-loop algorithm
// with matrix packing for optimal cache utilization.
//
// The algorithm structure (GEBP - GEneral Block Panel multiplication):
//
//	for jc := 0; jc < n; jc += Nc:       // Loop 5: B panels (L3 cache)
//	  for pc := 0; pc < k; pc += Kc:     // Loop 4: K blocking (L1 cache)
//	    PackRHS(B[pc:pc+Kc, jc:jc+Nc])   // Pack B panel once per (jc, pc)
//	    for ic := 0; ic < m; ic += Mc:   // Loop 3: A panels (L2 cache)
//	      PackLHS(A[ic:ic+Mc, pc:pc+Kc]) // Pack A panel once per (jc, pc, ic)
//	      for jr := 0; jr < Nc; jr += Nr:   // Loop 2: micro-tile columns
//	        for ir := 0; ir < Mc; ir += Mr: // Loop 1: micro-tile rows
//	          PackedMicroKernel(...)        // Mr × Nr micro-tile
//
// Key benefits over streaming matmul:
//   - K-dimension blocking prevents L1 cache thrashing
//   - Packed layout enables sequential memory access in innermost loops
//   - Accumulators stay in registers across entire Kc loop
//   - B panel reused across all A panels (L3 blocking)
//   - A panel reused across all micro-columns (L2 blocking)
//
// Parameters:
//   - a: Input matrix A in row-major order (M × K)
//   - b: Input matrix B in row-major order (K × N)
//   - c: Output matrix C in row-major order (M × N), will be zeroed
//   - m, n, k: Matrix dimensions
func BasePackedMatMul[T hwy.Floats](a, b, c []T, m, n, k int) {
	if len(a) < m*k {
		panic("packedmatmul: A slice too short")
	}
	if len(b) < k*n {
		panic("packedmatmul: B slice too short")
	}
	if len(c) < m*n {
		panic("packedmatmul: C slice too short")
	}

	// Get architecture-specific cache parameters
	params := getCacheParams[T]()
	mr, nr := params.Mr, params.Nr
	kc, mc, nc := params.Kc, params.Mc, params.Nc

	// Allocate packing buffers
	packedASize := params.PackedASize()
	packedBSize := params.PackedBSize()
	packedA := make([]T, packedASize)
	packedB := make([]T, packedBSize)

	// Zero output matrix
	zeroMatrix(c, m*n)

	// Loop 5: B panels (L3 blocking)
	for jc := 0; jc < n; jc += nc {
		jcEnd := min(jc+nc, n)
		panelCols := jcEnd - jc

		// Loop 4: K blocking (L1)
		for pc := 0; pc < k; pc += kc {
			pcEnd := min(pc+kc, k)
			panelK := pcEnd - pc

			// Pack B panel: B[pc:pcEnd, jc:jcEnd] -> packedB
			PackRHS(b, packedB, k, n, pc, jc, panelK, panelCols, nr)

			// Loop 3: A panels (L2 blocking)
			for ic := 0; ic < m; ic += mc {
				icEnd := min(ic+mc, m)
				panelRows := icEnd - ic

				// Pack A panel: A[ic:icEnd, pc:pcEnd] -> packedA
				activeRowsLast := PackLHS(a, packedA, m, k, ic, pc, panelRows, panelK, mr)

				// GEBP: multiply packed A panel with packed B panel
				gebp(packedA, packedB, c, n, ic, jc, panelRows, panelCols, panelK, mr, nr, activeRowsLast)
			}
		}
	}
}

// gebp performs the GEBP (GEneral Block Panel) multiplication:
// C[ic:ic+panelRows, jc:jc+panelCols] += packedA * packedB
func gebp[T hwy.Floats](packedA, packedB []T, c []T, n, ic, jc, panelRows, panelCols, panelK, mr, nr, activeRowsLast int) {
	numMicroPanelsA := (panelRows + mr - 1) / mr
	numMicroPanelsB := (panelCols + nr - 1) / nr

	// Compute active columns in last B micro-panel
	activeColsLast := panelCols - (numMicroPanelsB-1)*nr
	if activeColsLast <= 0 {
		activeColsLast = nr
	}

	// Loop 2: micro-tile columns (jr)
	for jPanel := range numMicroPanelsB {
		jr := jc + jPanel*nr
		bPanelOffset := jPanel * panelK * nr

		// Determine active columns for this micro-panel
		activeCols := nr
		if jPanel == numMicroPanelsB-1 {
			activeCols = activeColsLast
		}

		// Loop 1: micro-tile rows (ir)
		for iPanel := range numMicroPanelsA {
			ir := ic + iPanel*mr
			aPanelOffset := iPanel * panelK * mr

			// Determine active rows for this micro-panel
			activeRows := mr
			if iPanel == numMicroPanelsA-1 {
				activeRows = activeRowsLast
			}

			// Call micro-kernel
			if activeRows == mr && activeCols == nr {
				// Full micro-tile
				PackedMicroKernel(packedA[aPanelOffset:], packedB[bPanelOffset:], c, n, ir, jr, panelK, mr, nr)
			} else {
				// Partial micro-tile (edge case)
				PackedMicroKernelPartial(packedA[aPanelOffset:], packedB[bPanelOffset:], c, n, ir, jr, panelK, mr, nr, activeRows, activeCols)
			}
		}
	}
}

// getCacheParams returns architecture-appropriate cache parameters.
// The function is specialized by hwygen for each target.
func getCacheParams[T hwy.Floats]() CacheParams {
	lanes := hwy.Zero[T]().NumLanes()

	// Detect element size from lanes and use appropriate params
	// For float32 on AVX-512: lanes=16, for float64: lanes=8
	// We use a simple heuristic based on vector width

	switch lanes {
	case 16: // AVX-512 float32 or AVX2 float64
		var zero T
		if isFloat64(zero) {
			return CacheParamsFloat64AVX2()
		}
		return CacheParamsAVX512()
	case 8: // AVX2 float32, AVX-512 float64, or NEON float64
		var zero T
		if isFloat64(zero) {
			return CacheParamsFloat64AVX512()
		}
		return CacheParamsAVX2()
	case 4: // NEON float32 or fallback float64
		var zero T
		if isFloat64(zero) {
			return CacheParamsFloat64NEON()
		}
		return CacheParamsNEON()
	case 2: // NEON float64
		return CacheParamsFloat64NEON()
	default:
		return CacheParamsFallback()
	}
}

// isFloat64 returns true if T is float64
func isFloat64[T hwy.Floats](v T) bool {
	_, ok := any(v).(float64)
	return ok
}

// zeroMatrix zeros all elements of a slice using SIMD.
func zeroMatrix[T hwy.Floats](c []T, total int) {
	vZero := hwy.Zero[T]()
	lanes := vZero.NumLanes()

	var idx int
	for idx = 0; idx+lanes <= total; idx += lanes {
		hwy.Store(vZero, c[idx:])
	}
	for ; idx < total; idx++ {
		c[idx] = 0
	}
}

// BasePackedMatMulWithBuffers is like BasePackedMatMul but uses pre-allocated buffers.
// This is useful for parallel execution where each worker has its own buffers.
func BasePackedMatMulWithBuffers[T hwy.Floats](a, b, c []T, m, n, k int, packedA, packedB []T, params CacheParams) {
	if len(a) < m*k {
		panic("packedmatmul: A slice too short")
	}
	if len(b) < k*n {
		panic("packedmatmul: B slice too short")
	}
	if len(c) < m*n {
		panic("packedmatmul: C slice too short")
	}

	mr, nr := params.Mr, params.Nr
	kc, mc, nc := params.Kc, params.Mc, params.Nc

	// Zero output matrix
	zeroMatrix(c, m*n)

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

			// Loop 3: A panels (L2 blocking)
			for ic := 0; ic < m; ic += mc {
				icEnd := min(ic+mc, m)
				panelRows := icEnd - ic

				// Pack A panel
				activeRowsLast := PackLHS(a, packedA, m, k, ic, pc, panelRows, panelK, mr)

				// GEBP
				gebp(packedA, packedB, c, n, ic, jc, panelRows, panelCols, panelK, mr, nr, activeRowsLast)
			}
		}
	}
}

// BasePackedMatMulStrip computes a horizontal strip of C = A * B.
// Used by parallel implementation to divide work across workers.
//
// Computes: C[rowStart:rowEnd, :] = A[rowStart:rowEnd, :] * B
//
// Parameters:
//   - rowStart, rowEnd: Row range to compute (0-indexed)
//   - packedA, packedB: Pre-allocated packing buffers
//   - params: Cache blocking parameters
func BasePackedMatMulStrip[T hwy.Floats](a, b, c []T, m, n, k, rowStart, rowEnd int, packedA, packedB []T, params CacheParams) {
	mr, nr := params.Mr, params.Nr
	kc, mc, nc := params.Kc, params.Mc, params.Nc

	stripM := rowEnd - rowStart

	// Zero output strip
	zeroMatrix(c[rowStart*n:rowEnd*n], stripM*n)

	// Loop 5: B panels (L3 blocking)
	for jc := 0; jc < n; jc += nc {
		jcEnd := min(jc+nc, n)
		panelCols := jcEnd - jc

		// Loop 4: K blocking (L1)
		for pc := 0; pc < k; pc += kc {
			pcEnd := min(pc+kc, k)
			panelK := pcEnd - pc

			// Pack B panel (shared across all row strips)
			PackRHS(b, packedB, k, n, pc, jc, panelK, panelCols, nr)

			// Loop 3: A panels within this strip (L2 blocking)
			for ic := rowStart; ic < rowEnd; ic += mc {
				icEnd := min(ic+mc, rowEnd)
				panelRows := icEnd - ic

				// Pack A panel from this strip
				activeRowsLast := PackLHS(a, packedA, m, k, ic, pc, panelRows, panelK, mr)

				// GEBP for this strip
				gebp(packedA, packedB, c, n, ic, jc, panelRows, panelCols, panelK, mr, nr, activeRowsLast)
			}
		}
	}
}
