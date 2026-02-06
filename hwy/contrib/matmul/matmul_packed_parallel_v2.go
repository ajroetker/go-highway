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
	"runtime"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// getCacheParamsV2 returns V2-optimized cache parameters for the current architecture.
// V2 uses smaller Mc/Nc for the packed output buffer pattern.
func getCacheParamsV2[T hwy.Floats]() CacheParams {
	lanes := hwy.Zero[T]().NumLanes()

	// Detect architecture from vector width
	switch lanes {
	case 16: // AVX-512 float32
		var zero T
		if isFloat64(zero) {
			// This would be unusual (AVX2 doesn't have 16-lane float64)
			return CacheParamsV2AVX2()
		}
		return CacheParamsV2AVX512()
	case 8: // AVX2 float32, AVX-512 float64, or NEON float64
		var zero T
		if isFloat64(zero) {
			// AVX-512 float64 or NEON float64
			if runtime.GOARCH == "arm64" {
				return CacheParamsV2NEON()
			}
			return CacheParamsV2AVX512()
		}
		// AVX2 float32
		return CacheParamsV2AVX2()
	case 4: // NEON float32 or fallback float64
		if runtime.GOARCH == "arm64" {
			return CacheParamsV2NEON()
		}
		return CacheParamsV2Fallback()
	case 2: // NEON float64
		return CacheParamsV2NEON()
	default:
		return CacheParamsV2Fallback()
	}
}

// ParallelPackedMatMulV2 computes C = A * B using the optimized parallel
// algorithm inspired by gomlx's packgemm-simd-large-opt.
//
// Key optimizations over V1:
//   - Persistent worker pool for efficient worker management
//   - Intelligent work distribution via generateWorkItems
//   - Packed output buffer for faster micro-kernel writes
//   - SIMD-optimized output application
//
// For small matrices, falls back to single-threaded PackedMatMul.
//
// Parameters:
//   - pool: Persistent worker pool for parallel execution
//   - a: Input matrix A in row-major order (M × K)
//   - b: Input matrix B in row-major order (K × N)
//   - c: Output matrix C in row-major order (M × N), will be zeroed
//   - m, n, k: Matrix dimensions
func ParallelPackedMatMulV2[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, m, n, k int) {
	totalOps := m * n * k

	// For small matrices, use single-threaded version
	if totalOps < MinPackedParallelOps {
		PackedMatMul(a, b, c, m, n, k)
		return
	}

	params := getCacheParamsV2[T]()
	maxWorkers := pool.NumWorkers()

	// Zero the output matrix once (shared across all workers)
	zeroMatrix(c, m*n)

	// For single-worker case, run without parallelization overhead
	if maxWorkers <= 1 {
		packedA := make([]T, params.PackedASize())
		packedB := make([]T, params.PackedBSize())
		packedOut := make([]T, params.PackedOutputSize())
		processGEMMSliceV2(a, b, c, m, n, k, 0, m, 0, n, packedA, packedB, packedOut, params)
		return
	}

	// Pre-compute work items
	items := generateWorkItems(1, m, n, params, maxWorkers)

	// Use ParallelForAtomic to process work items with work stealing
	pool.ParallelForAtomic(len(items), func(idx int) {
		item := items[idx]
		// Each invocation gets its own buffers
		packedA := make([]T, params.PackedASize())
		packedB := make([]T, params.PackedBSize())
		packedOut := make([]T, params.PackedOutputSize())
		processGEMMSliceV2(
			a, b, c, m, n, k,
			item.lhsRowStart, item.lhsRowEnd,
			item.rhsColStart, item.rhsColEnd,
			packedA, packedB, packedOut, params,
		)
	})
}

// processGEMMSliceV2 computes a slice of C using the packed output buffer pattern.
// C[lhsRowStart:lhsRowEnd, rhsColStart:rhsColEnd] += A[lhsRowStart:lhsRowEnd, :] * B[:, rhsColStart:rhsColEnd]
//
// This function uses an intermediate packed output buffer for faster writes,
// then applies the result to the final output with SIMD.
func processGEMMSliceV2[T hwy.Floats](
	a, b, c []T, m, n, k int,
	lhsRowStart, lhsRowEnd int,
	rhsColStart, rhsColEnd int,
	packedA, packedB, packedOut []T,
	params CacheParams,
) {
	mr, nr := params.Mr, params.Nr
	kc, mc, nc := params.Kc, params.Mc, params.Nc

	sliceM := lhsRowEnd - lhsRowStart
	sliceN := rhsColEnd - rhsColStart

	// Loop 5: B panels (L3 blocking) - within slice N range
	for jc := 0; jc < sliceN; jc += nc {
		jcEnd := min(jc+nc, sliceN)
		panelCols := jcEnd - jc
		globalJC := rhsColStart + jc

		// Loop 4: K blocking (L1)
		for pc := 0; pc < k; pc += kc {
			pcEnd := min(pc+kc, k)
			panelK := pcEnd - pc

			// Pack B panel: B[pc:pcEnd, globalJC:globalJC+panelCols]
			PackRHSFast(b, packedB, n, pc, globalJC, panelK, panelCols, nr)

			// Loop 3: A panels (L2 blocking) - within slice M range
			for ic := 0; ic < sliceM; ic += mc {
				icEnd := min(ic+mc, sliceM)
				panelRows := icEnd - ic
				globalIC := lhsRowStart + ic

				// Pack A panel: A[globalIC:globalIC+panelRows, pc:pcEnd]
				PackLHS(a, packedA, m, k, globalIC, pc, panelRows, panelK, mr)

				// GEBP with packed output buffer
				gebpWithPackedOutput(
					packedA, packedB, packedOut, c,
					n, globalIC, globalJC,
					panelRows, panelCols, panelK,
					mr, nr, nc,
					pc > 0, // accumulate if not first K panel
				)
			}
		}
	}
}

// gebpWithPackedOutput performs GEBP using an intermediate packed output buffer.
// The micro-kernel writes to packedOut, then we apply to c with alpha/beta.
//
// This allows the micro-kernel to write full tiles without bounds checking,
// improving performance.
func gebpWithPackedOutput[T hwy.Floats](
	packedA, packedB, packedOut, c []T,
	cStride int,
	outputRowStart, outputColStart int,
	panelRows, panelCols, panelK int,
	mr, nr, ncStride int,
	accumulate bool,
) {
	numMicroPanelsA := (panelRows + mr - 1) / mr
	numMicroPanelsB := (panelCols + nr - 1) / nr

	// Compute active rows/cols in last micro-panels
	activeRowsLast := panelRows - (numMicroPanelsA-1)*mr
	if activeRowsLast <= 0 {
		activeRowsLast = mr
	}
	activeColsLast := panelCols - (numMicroPanelsB-1)*nr
	if activeColsLast <= 0 {
		activeColsLast = nr
	}

	// Zero the packed output buffer for this panel
	ZeroSlice(packedOut, panelRows*ncStride)

	// Loop 2: micro-tile columns (jr)
	for jPanel := range numMicroPanelsB {
		bPanelOffset := jPanel * panelK * nr

		// Loop 1: micro-tile rows (ir)
		for iPanel := range numMicroPanelsA {
			aPanelOffset := iPanel * panelK * mr

			// Compute output position in packed buffer
			outRowStart := iPanel * mr
			outColStart := jPanel * nr

			// Call micro-kernel to write to packed output
			packedMicroKernelToBuffer(
				packedA[aPanelOffset:],
				packedB[bPanelOffset:],
				packedOut,
				ncStride, // stride in packed output
				outRowStart, outColStart,
				panelK, mr, nr,
			)
		}
	}

	// Apply packed output to final output matrix with proper active region handling
	if accumulate {
		// Accumulate: c += packedOut (for K panels after the first)
		ApplyPackedOutputAccum(
			packedOut, c,
			ncStride,
			outputRowStart, outputColStart,
			cStride,
			panelRows, panelCols,
		)
	} else {
		// First K panel: just copy (alpha=1, beta=0)
		ApplyPackedOutputSimple(
			packedOut, c,
			ncStride,
			outputRowStart, outputColStart,
			cStride,
			panelRows, panelCols,
		)
	}
}

// packedMicroKernelToBuffer computes a micro-tile and writes to a buffer.
// This writes a full Mr x Nr tile without bounds checking.
//
// Optimized for mr=4 and nr=2*lanes (8 accumulators).
// Uses the generated dispatch function PackedMicroKernel4x2 which is optimized
// for each architecture (AVX2, AVX-512, NEON). Falls back to packedMicroKernelGenericImpl
// for non-standard configurations.
func packedMicroKernelToBuffer[T hwy.Floats](
	packedA, packedB []T,
	output []T,
	outputStride int,
	outRowStart, outColStart int,
	panelK, mr, nr int,
) {
	lanes := getLanes[T]()
	numBVecs := nr / lanes

	// For typical 4x(2*lanes) config, we have 8 accumulators.
	// Use the generated dispatch function which calls the architecture-specific kernel.
	if mr == 4 && numBVecs == 2 {
		PackedMicroKernel4x2(packedA, packedB, output, outputStride, outRowStart, outColStart, panelK, lanes)
		return
	}

	// Generic fallback for non-standard configurations (uses hwy.* calls which may allocate).
	packedMicroKernelGenericImpl(packedA, packedB, output, outputStride, outRowStart, outColStart, panelK, mr, nr, lanes)
}

// getLanes returns the vector width for type T.
// This returns the lanes appropriate for the SIMD implementation being used,
// NOT the global currentWidth (which may be SME 512-bit on M4).
//
// The generated NEON kernels use Float32x4/Float64x2 intrinsics, so we must
// return NEON-appropriate lanes regardless of SME detection.
func getLanes[T hwy.Floats]() int {
	var zero T
	switch any(zero).(type) {
	case float32:
		return getLanesFloat32()
	case float64:
		return getLanesFloat64()
	default:
		// Fallback for other float types - use actual vector width
		return hwy.Zero[T]().NumLanes()
	}
}

var lanesFloat32 int
var lanesFloat64 int

func init() {
	// Get the actual lanes from the SIMD implementation.
	// On ARM64, even if SME is detected (512-bit), the generated NEON kernels
	// use Float32x4 (4 lanes) and Float64x2 (2 lanes) intrinsics.
	// Use getKernelLanes() to get the implementation-appropriate lanes.
	lanesFloat32 = getKernelLanesFloat32()
	lanesFloat64 = getKernelLanesFloat64()
}

func getLanesFloat32() int { return lanesFloat32 }
func getLanesFloat64() int { return lanesFloat64 }

// ParallelPackedMatMulV2Float32 is the non-generic version for float32.
func ParallelPackedMatMulV2Float32(pool *workerpool.Pool, a, b, c []float32, m, n, k int) {
	ParallelPackedMatMulV2(pool, a, b, c, m, n, k)
}

// ParallelPackedMatMulV2Float64 is the non-generic version for float64.
func ParallelPackedMatMulV2Float64(pool *workerpool.Pool, a, b, c []float64, m, n, k int) {
	ParallelPackedMatMulV2(pool, a, b, c, m, n, k)
}

// BatchParallelPackedMatMulV2 computes batched C = A * B using the optimized
// parallel algorithm.
//
// Parameters:
//   - pool: Persistent worker pool for parallel execution
//   - a: Batched input matrix A [batchSize, M, K] in row-major order
//   - b: Batched input matrix B [batchSize, K, N] in row-major order
//   - c: Batched output matrix C [batchSize, M, N] in row-major order
//   - batchSize, m, n, k: Dimensions
func BatchParallelPackedMatMulV2[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, batchSize, m, n, k int) {
	totalOps := batchSize * m * n * k

	// For small total work, use single-threaded version
	if totalOps < MinPackedParallelOps {
		lhsStride := m * k
		rhsStride := k * n
		outStride := m * n
		for batch := range batchSize {
			PackedMatMul(
				a[batch*lhsStride:(batch+1)*lhsStride],
				b[batch*rhsStride:(batch+1)*rhsStride],
				c[batch*outStride:(batch+1)*outStride],
				m, n, k,
			)
		}
		return
	}

	params := getCacheParamsV2[T]()
	maxWorkers := pool.NumWorkers()

	lhsStride := m * k
	rhsStride := k * n
	outStride := m * n

	// Zero all output matrices
	zeroMatrix(c, batchSize*m*n)

	// Pre-compute work items
	items := generateWorkItems(batchSize, m, n, params, maxWorkers)

	// Use ParallelForAtomic to process work items with work stealing
	pool.ParallelForAtomic(len(items), func(idx int) {
		item := items[idx]
		packedA := make([]T, params.PackedASize())
		packedB := make([]T, params.PackedBSize())
		packedOut := make([]T, params.PackedOutputSize())

		for batch := item.batchStart; batch < item.batchEnd; batch++ {
			batchA := a[batch*lhsStride : (batch+1)*lhsStride]
			batchB := b[batch*rhsStride : (batch+1)*rhsStride]
			batchC := c[batch*outStride : (batch+1)*outStride]

			processGEMMSliceV2(
				batchA, batchB, batchC, m, n, k,
				item.lhsRowStart, item.lhsRowEnd,
				item.rhsColStart, item.rhsColEnd,
				packedA, packedB, packedOut, params,
			)
		}
	})
}

// workItem represents a chunk of work for parallel GEMM.
type workItem struct {
	batchStart, batchEnd   int
	lhsRowStart, lhsRowEnd int
	rhsColStart, rhsColEnd int
}

// generateWorkItems creates work items for parallel GEMM, distributing work
// intelligently across workers. It prioritizes batch splitting, then splits
// on LHS or RHS dimension.
//
// This implements the intelligent work splitting from gomlx's packgemm-simd-large-opt.
func generateWorkItems(
	batchSize, lhsCrossSize, rhsCrossSize int,
	params CacheParams,
	maxWorkers int,
) []workItem {
	if maxWorkers <= 0 {
		maxWorkers = 1
	}

	var items []workItem

	// If batch size is large enough, split only on batch dimension
	if batchSize >= 2*maxWorkers {
		batchStep := batchSize / maxWorkers
		for batchIdx := 0; batchIdx < batchSize; batchIdx += batchStep {
			items = append(items, workItem{
				batchStart:  batchIdx,
				batchEnd:    batchIdx + min(batchStep, batchSize-batchIdx),
				lhsRowStart: 0,
				lhsRowEnd:   lhsCrossSize,
				rhsColStart: 0,
				rhsColEnd:   rhsCrossSize,
			})
		}
		return items
	}

	// First handle batches one at a time up to maxWorkers
	batchIdx := 0
	if batchSize >= maxWorkers {
		for ; batchIdx < maxWorkers; batchIdx++ {
			items = append(items, workItem{
				batchStart:  batchIdx,
				batchEnd:    batchIdx + 1,
				lhsRowStart: 0,
				lhsRowEnd:   lhsCrossSize,
				rhsColStart: 0,
				rhsColEnd:   rhsCrossSize,
			})
		}
	}

	// Split remaining work on LHS or RHS dimension
	batchCountRemaining := batchSize - batchIdx
	if batchCountRemaining == 0 {
		return items
	}

	splitFactor := (maxWorkers + batchCountRemaining - 1) / batchCountRemaining

	if lhsCrossSize > rhsCrossSize {
		// Split on LHS dimension (aligned to Mc)
		lhsSplitSize := (lhsCrossSize + splitFactor - 1) / splitFactor
		lhsSplitSize = max(1, lhsSplitSize/params.Mc) * params.Mc

		batchStart := batchIdx
		for lhsRowIdx := 0; lhsRowIdx < lhsCrossSize; lhsRowIdx += lhsSplitSize {
			for bi := batchStart; bi < batchSize; bi++ {
				items = append(items, workItem{
					batchStart:  bi,
					batchEnd:    bi + 1,
					lhsRowStart: lhsRowIdx,
					lhsRowEnd:   lhsRowIdx + min(lhsSplitSize, lhsCrossSize-lhsRowIdx),
					rhsColStart: 0,
					rhsColEnd:   rhsCrossSize,
				})
			}
		}
	} else {
		// Split on RHS dimension (aligned to Nc)
		rhsSplitSize := (rhsCrossSize + splitFactor - 1) / splitFactor
		rhsSplitSize = max(1, rhsSplitSize/params.Nc) * params.Nc

		batchStart := batchIdx
		for rhsColIdx := 0; rhsColIdx < rhsCrossSize; rhsColIdx += rhsSplitSize {
			for bi := batchStart; bi < batchSize; bi++ {
				items = append(items, workItem{
					batchStart:  bi,
					batchEnd:    bi + 1,
					lhsRowStart: 0,
					lhsRowEnd:   lhsCrossSize,
					rhsColStart: rhsColIdx,
					rhsColEnd:   rhsColIdx + min(rhsSplitSize, rhsCrossSize-rhsColIdx),
				})
			}
		}
	}

	return items
}
