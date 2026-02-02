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

//go:build !noasm && arm64

// NOTE: This file is named "z_matmul_arm64.go" (starting with 'z')
// to ensure its init() runs AFTER the generated dispatch files.
// Go executes init() functions in lexicographic filename order within a package.
// The generated dispatch sets MatMul*, BlockedMatMul*, MatMulKLast*, etc. to
// fallback implementations; this file's init() must run afterward to override
// with optimized NEON and SME implementations when available.

package matmul

import (
	"runtime"
	"sync"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// =============================================================================
// Constants
// =============================================================================

// Minimum dimensions to use NEON vectorization
const minDimForNEON = 16

// Minimum dimensions to use SME FMOPA MatMul
// SME is 3x+ faster than NEON even at small sizes (32x32).
// Only use NEON for very small matrices where streaming mode overhead dominates.
const minDimForSME = 32

// minOpsForBlockedSME is the minimum padded total ops (paddedM*paddedN*paddedK)
// before SME FMOPA with padding/transpose is faster than NEON streaming.
// SME overhead is ~2µs (SMEGuard + pad + transpose + extract). NEON throughput
// is ~2 GFLOPS for blocked matmul. Crossover is around 64K ops.
// Benchmarks on Apple M4 Max:
//   1x32x32   (32K ops padded): SME 22x slower (overhead dominates)
//   4x128x128 (64K ops padded): SME 1.6x faster
//   1x512x512 (512K ops padded): SME 1.5x faster
//   16x64x64  (64K ops padded): SME 4x faster
//   8x512x512 (4M ops padded): SME 12.8x faster
const minOpsForBlockedSME = 64 * 1024

// Minimum dimensions to use NEON KLast vectorization
const minDimForNEONKLast = 16

// Minimum dimensions to use SME FMOPA for MatMulKLast
// SME with transpose is faster than NEON dot-product even at small sizes
// (2.2x faster at 64x64, 3x+ faster at larger sizes).
// Only use NEON for very small matrices where transpose overhead dominates.
const minDimForSMEKLast = 32

// MinFusedParallelTiles is the minimum number of N-tiles before parallelizing fused NF4/Int4 matmul
const MinFusedParallelTiles = 4 // N >= 64

// =============================================================================
// Transpose buffer pools to avoid allocations
// =============================================================================

var transposePool32 = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var transposePool64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var transposePoolF16 = sync.Pool{
	New: func() any {
		return make([]hwy.Float16, 0, 256*256)
	},
}

var transposePoolBF16 = sync.Pool{
	New: func() any {
		return make([]hwy.BFloat16, 0, 256*256)
	},
}

// Buffer pools for MatMulKLast transpose operations
// These are separate from the regular matmul pools since MatMulKLast
// transposes both A and B, potentially needing different sizes.
var klastTransposePoolA32 = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var klastTransposePoolB32 = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var klastTransposePoolA64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var klastTransposePoolB64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var klastTransposePoolAF16 = sync.Pool{
	New: func() any {
		return make([]hwy.Float16, 0, 256*256)
	},
}

var klastTransposePoolBF16 = sync.Pool{
	New: func() any {
		return make([]hwy.Float16, 0, 256*256)
	},
}

var klastTransposePoolABF16 = sync.Pool{
	New: func() any {
		return make([]hwy.BFloat16, 0, 256*256)
	},
}

var klastTransposePoolBBF16 = sync.Pool{
	New: func() any {
		return make([]hwy.BFloat16, 0, 256*256)
	},
}

// Fused NF4/Int4 tile buffer pools to reduce allocations (SME-specific)
var fusedTilePool = sync.Pool{
	New: func() any {
		// Max tile size: K * 16 floats for SME tile width (K up to 4096)
		return make([]float32, 0, 4096*16)
	},
}

// =============================================================================
// M-padding buffer pools for SME FMOPA
// =============================================================================
// When M is not tile-aligned, we pad A and use a padded C buffer internally.
// These pools avoid repeated allocations for the padding buffers.

var paddedAPool32 = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var paddedCPool32 = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var paddedAPool64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var paddedCPool64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var paddedAPoolF16 = sync.Pool{
	New: func() any {
		return make([]hwy.Float16, 0, 256*256)
	},
}

var paddedCPoolF16 = sync.Pool{
	New: func() any {
		return make([]hwy.Float16, 0, 256*256)
	},
}

var paddedAPoolBF16 = sync.Pool{
	New: func() any {
		return make([]hwy.BFloat16, 0, 256*256)
	},
}

var paddedCPoolBF16 = sync.Pool{
	New: func() any {
		return make([]hwy.BFloat16, 0, 256*256)
	},
}

// =============================================================================
// B-padding buffer pools for SME FMOPA (N/K dimension alignment)
// =============================================================================
// When N or K is not tile-aligned, we pad B into these buffers.

var paddedBPool32 = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var paddedBPool64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var paddedBPoolF16 = sync.Pool{
	New: func() any {
		return make([]hwy.Float16, 0, 256*256)
	},
}

var paddedBPoolBF16 = sync.Pool{
	New: func() any {
		return make([]hwy.BFloat16, 0, 256*256)
	},
}

// =============================================================================
// Helper functions
// =============================================================================

// AlignUp rounds m up to the next multiple of tileSize.
// Public so callers (e.g., nn package) can pre-align dimensions if needed.
func AlignUp(m, tileSize int) int {
	return (m + tileSize - 1) / tileSize * tileSize
}

// PadMatrix2D pads a [rows, cols] row-major matrix to [paddedRows, paddedCols].
// dst must have length >= paddedRows * paddedCols.
// If cols == paddedCols, uses efficient contiguous copy; otherwise re-strides row by row.
// Zero-fills all padding regions (right columns and bottom rows).
func PadMatrix2D[T hwy.Floats](dst []T, src []T, rows, cols, paddedRows, paddedCols int) {
	if cols == paddedCols {
		// Only row padding needed — contiguous copy + zero trailing rows
		copy(dst[:rows*cols], src[:rows*cols])
		if paddedRows > rows {
			clear(dst[rows*cols : paddedRows*cols])
		}
	} else {
		// Re-stride: copy each row, zero-pad right columns, then zero extra rows
		for i := range rows {
			copy(dst[i*paddedCols:i*paddedCols+cols], src[i*cols:i*cols+cols])
			clear(dst[i*paddedCols+cols : (i+1)*paddedCols])
		}
		if paddedRows > rows {
			clear(dst[rows*paddedCols : paddedRows*paddedCols])
		}
	}
}

// ExtractMatrix2D copies [rows, cols] from a [_, paddedCols] padded matrix into dst.
// If cols == paddedCols, uses efficient contiguous copy; otherwise extracts row by row.
func ExtractMatrix2D[T hwy.Floats](dst []T, src []T, rows, cols, paddedCols int) {
	if cols == paddedCols {
		copy(dst[:rows*cols], src[:rows*cols])
	} else {
		for i := range rows {
			copy(dst[i*cols:i*cols+cols], src[i*paddedCols:i*paddedCols+cols])
		}
	}
}

// transposeMatrix transposes M×K matrix A into K×M matrix AT (row-major to column-major)
// AT[k,i] = A[i,k]
// Dispatches to SIMD implementation (NEON or SME depending on size).
func transposeMatrix[T hwy.Floats](a []T, m, k int, at []T) {
	Transpose2D(a, m, k, at)
}

// =============================================================================
// NEON MatMul implementations
// =============================================================================

// matmulNEON uses ARM NEON FMLA instructions for matrix multiplication.
// Falls back to scalar for small matrices.
func matmulNEON(a, b, c []float32, m, n, k int) {
	// Streaming algorithm works for any M size - it processes one row at a time
	// with full vectorization across N. Only need N and K large enough for
	// SIMD benefit.
	if n < minDimForNEON || k < minDimForNEON {
		matmulScalar(a, b, c, m, n, k)
		return
	}

	asm.MatMulNEONF32(a, b, c, m, n, k)
}

// matmulNEONF16 uses ARM NEON for float16 matrix multiplication.
// Uses hand-written assembly with FMLA f16 instructions.
func matmulNEONF16(a, b, c []hwy.Float16, m, n, k int) {
	// Streaming algorithm works for any M size
	if n < minDimForNEON || k < minDimForNEON {
		BaseMatMul_neon_Float16(a, b, c, m, n, k)
		return
	}
	asm.MatMulNEONF16(a, b, c, m, n, k)
}

// matmulNEONBF16 uses ARM NEON for bfloat16 matrix multiplication.
// Uses hand-written assembly with BFDOT bf16 instructions.
func matmulNEONBF16(a, b, c []hwy.BFloat16, m, n, k int) {
	// Streaming algorithm works for any M size
	if n < minDimForNEON || k < minDimForNEON {
		BaseMatMul_neon_BFloat16(a, b, c, m, n, k)
		return
	}
	asm.MatMulNEONBF16(a, b, c, m, n, k)
}

// =============================================================================
// SME FMOPA MatMul implementations
// =============================================================================

// matmulFMOPA uses ARM SME FMOPA instruction for matrix multiplication.
// Uses outer product accumulate with ZA tiles - confirmed working on Apple M4!
// Processes matrices in 16x16 tiles using the ZA accumulator.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPA(a, b, c []float32, m, n, k int) {
	const tileSize = 16
	paddedM := AlignUp(m, tileSize)
	paddedN := AlignUp(n, tileSize)
	paddedK := AlignUp(k, tileSize)

	// For small matrices, NEON is faster (streaming mode has overhead)
	if paddedM < minDimForSME || paddedN < minDimForSME || paddedK < minDimForSME {
		matmulNEON(a, b, c, m, n, k)
		return
	}

	needsPadM := paddedM != m
	needsPadK := paddedK != k
	needsPadN := paddedN != n

	// Prepare A: [M, K] → [paddedM, paddedK]
	fmopaA := a
	fmopaK := k
	if needsPadM || needsPadK {
		paSize := paddedM * paddedK
		paBuf := paddedAPool32.Get().([]float32)
		if cap(paBuf) < paSize {
			paBuf = make([]float32, paSize)
		} else {
			paBuf = paBuf[:paSize]
		}
		PadMatrix2D(paBuf, a, m, k, paddedM, paddedK)
		fmopaA = paBuf
		fmopaK = paddedK
		defer paddedAPool32.Put(paBuf)
	}

	// Prepare B: [K, N] → [paddedK, paddedN]
	fmopaB := b
	fmopaN := n
	if needsPadK || needsPadN {
		pbSize := paddedK * paddedN
		pbBuf := paddedBPool32.Get().([]float32)
		if cap(pbBuf) < pbSize {
			pbBuf = make([]float32, pbSize)
		} else {
			pbBuf = pbBuf[:pbSize]
		}
		PadMatrix2D(pbBuf, b, k, n, paddedK, paddedN)
		fmopaB = pbBuf
		fmopaN = paddedN
		defer paddedBPool32.Put(pbBuf)
	}

	fmopaM := paddedM

	// Transpose A [paddedM, paddedK] → AT [paddedK, paddedM]
	atSize := fmopaK * fmopaM
	atBuf := transposePool32.Get().([]float32)
	if cap(atBuf) < atSize {
		atBuf = make([]float32, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	transposeMatrix(fmopaA, fmopaM, fmopaK, atBuf)
	defer transposePool32.Put(atBuf)

	// Call FMOPA; use padded C if any output dimension changed
	if needsPadM || needsPadN {
		pcSize := fmopaM * fmopaN
		paddedC := paddedCPool32.Get().([]float32)
		if cap(paddedC) < pcSize {
			paddedC = make([]float32, pcSize)
		} else {
			paddedC = paddedC[:pcSize]
		}
		clear(paddedC)
		asm.MultiTileMatMulFMOPAF32(atBuf, fmopaB, paddedC, fmopaM, fmopaN, fmopaK)
		ExtractMatrix2D(c, paddedC, m, n, fmopaN)
		paddedCPool32.Put(paddedC)
	} else {
		asm.MultiTileMatMulFMOPAF32(atBuf, fmopaB, c, fmopaM, fmopaN, fmopaK)
	}
}

// matmulFMOPA64 uses ARM SME FMOPA instruction for float64 matrix multiplication.
// Uses outer product accumulate with ZA tiles - 8×8 tiles for float64.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPA64(a, b, c []float64, m, n, k int) {
	const tileSize = 8
	paddedM := AlignUp(m, tileSize)
	paddedN := AlignUp(n, tileSize)
	paddedK := AlignUp(k, tileSize)

	// For small matrices, scalar is faster (streaming mode has overhead)
	if paddedM < minDimForSME || paddedN < minDimForSME || paddedK < minDimForSME {
		matmulScalar64(a, b, c, m, n, k)
		return
	}

	needsPadM := paddedM != m
	needsPadK := paddedK != k
	needsPadN := paddedN != n

	// Prepare A: [M, K] → [paddedM, paddedK]
	fmopaA := a
	fmopaK := k
	if needsPadM || needsPadK {
		paSize := paddedM * paddedK
		paBuf := paddedAPool64.Get().([]float64)
		if cap(paBuf) < paSize {
			paBuf = make([]float64, paSize)
		} else {
			paBuf = paBuf[:paSize]
		}
		PadMatrix2D(paBuf, a, m, k, paddedM, paddedK)
		fmopaA = paBuf
		fmopaK = paddedK
		defer paddedAPool64.Put(paBuf)
	}

	// Prepare B: [K, N] → [paddedK, paddedN]
	fmopaB := b
	fmopaN := n
	if needsPadK || needsPadN {
		pbSize := paddedK * paddedN
		pbBuf := paddedBPool64.Get().([]float64)
		if cap(pbBuf) < pbSize {
			pbBuf = make([]float64, pbSize)
		} else {
			pbBuf = pbBuf[:pbSize]
		}
		PadMatrix2D(pbBuf, b, k, n, paddedK, paddedN)
		fmopaB = pbBuf
		fmopaN = paddedN
		defer paddedBPool64.Put(pbBuf)
	}

	fmopaM := paddedM

	// Transpose A [paddedM, paddedK] → AT [paddedK, paddedM]
	atSize := fmopaK * fmopaM
	atBuf := transposePool64.Get().([]float64)
	if cap(atBuf) < atSize {
		atBuf = make([]float64, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	transposeMatrix(fmopaA, fmopaM, fmopaK, atBuf)
	defer transposePool64.Put(atBuf)

	// Call FMOPA; use padded C if any output dimension changed
	if needsPadM || needsPadN {
		pcSize := fmopaM * fmopaN
		paddedC := paddedCPool64.Get().([]float64)
		if cap(paddedC) < pcSize {
			paddedC = make([]float64, pcSize)
		} else {
			paddedC = paddedC[:pcSize]
		}
		clear(paddedC)
		asm.MultiTileMatMulFMOPAF64(atBuf, fmopaB, paddedC, fmopaM, fmopaN, fmopaK)
		ExtractMatrix2D(c, paddedC, m, n, fmopaN)
		paddedCPool64.Put(paddedC)
	} else {
		asm.MultiTileMatMulFMOPAF64(atBuf, fmopaB, c, fmopaM, fmopaN, fmopaK)
	}
}

// matmulFMOPAF16 uses ARM SME FMOPA instruction for float16 matrix multiplication.
// Uses widening: f16 -> f32 FMOPA -> f16, with 16×16 tiles (f32 accumulator).
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPAF16(a, b, c []hwy.Float16, m, n, k int) {
	const tileSize = 16
	paddedM := AlignUp(m, tileSize)
	paddedN := AlignUp(n, tileSize)
	paddedK := AlignUp(k, tileSize)

	// For small matrices, NEON is faster (streaming mode has overhead)
	if paddedM < minDimForSME || paddedN < minDimForSME || paddedK < minDimForSME {
		matmulNEONF16(a, b, c, m, n, k)
		return
	}

	needsPadM := paddedM != m
	needsPadK := paddedK != k
	needsPadN := paddedN != n

	fmopaA := a
	fmopaK := k
	if needsPadM || needsPadK {
		paSize := paddedM * paddedK
		paBuf := paddedAPoolF16.Get().([]hwy.Float16)
		if cap(paBuf) < paSize {
			paBuf = make([]hwy.Float16, paSize)
		} else {
			paBuf = paBuf[:paSize]
		}
		PadMatrix2D(paBuf, a, m, k, paddedM, paddedK)
		fmopaA = paBuf
		fmopaK = paddedK
		defer paddedAPoolF16.Put(paBuf)
	}

	fmopaB := b
	fmopaN := n
	if needsPadK || needsPadN {
		pbSize := paddedK * paddedN
		pbBuf := paddedBPoolF16.Get().([]hwy.Float16)
		if cap(pbBuf) < pbSize {
			pbBuf = make([]hwy.Float16, pbSize)
		} else {
			pbBuf = pbBuf[:pbSize]
		}
		PadMatrix2D(pbBuf, b, k, n, paddedK, paddedN)
		fmopaB = pbBuf
		fmopaN = paddedN
		defer paddedBPoolF16.Put(pbBuf)
	}

	fmopaM := paddedM

	atSize := fmopaK * fmopaM
	atBuf := transposePoolF16.Get().([]hwy.Float16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.Float16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	transposeMatrix(fmopaA, fmopaM, fmopaK, atBuf)
	defer transposePoolF16.Put(atBuf)

	if needsPadM || needsPadN {
		pcSize := fmopaM * fmopaN
		paddedC := paddedCPoolF16.Get().([]hwy.Float16)
		if cap(paddedC) < pcSize {
			paddedC = make([]hwy.Float16, pcSize)
		} else {
			paddedC = paddedC[:pcSize]
		}
		clear(paddedC)
		asm.MultiTileMatMulFMOPAF16(atBuf, fmopaB, paddedC, fmopaM, fmopaN, fmopaK)
		ExtractMatrix2D(c, paddedC, m, n, fmopaN)
		paddedCPoolF16.Put(paddedC)
	} else {
		asm.MultiTileMatMulFMOPAF16(atBuf, fmopaB, c, fmopaM, fmopaN, fmopaK)
	}
}

// matmulFMOPABF16 uses ARM SME BFMOPA instruction for bfloat16 matrix multiplication.
// Uses widening: bf16 -> f32 FMOPA -> bf16, with 16×16 tiles (f32 accumulator).
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPABF16(a, b, c []hwy.BFloat16, m, n, k int) {
	const tileSize = 16
	paddedM := AlignUp(m, tileSize)
	paddedN := AlignUp(n, tileSize)
	paddedK := AlignUp(k, tileSize)

	// For small matrices, NEON is faster (streaming mode has overhead)
	if paddedM < minDimForSME || paddedN < minDimForSME || paddedK < minDimForSME {
		matmulNEONBF16(a, b, c, m, n, k)
		return
	}

	needsPadM := paddedM != m
	needsPadK := paddedK != k
	needsPadN := paddedN != n

	fmopaA := a
	fmopaK := k
	if needsPadM || needsPadK {
		paSize := paddedM * paddedK
		paBuf := paddedAPoolBF16.Get().([]hwy.BFloat16)
		if cap(paBuf) < paSize {
			paBuf = make([]hwy.BFloat16, paSize)
		} else {
			paBuf = paBuf[:paSize]
		}
		PadMatrix2D(paBuf, a, m, k, paddedM, paddedK)
		fmopaA = paBuf
		fmopaK = paddedK
		defer paddedAPoolBF16.Put(paBuf)
	}

	fmopaB := b
	fmopaN := n
	if needsPadK || needsPadN {
		pbSize := paddedK * paddedN
		pbBuf := paddedBPoolBF16.Get().([]hwy.BFloat16)
		if cap(pbBuf) < pbSize {
			pbBuf = make([]hwy.BFloat16, pbSize)
		} else {
			pbBuf = pbBuf[:pbSize]
		}
		PadMatrix2D(pbBuf, b, k, n, paddedK, paddedN)
		fmopaB = pbBuf
		fmopaN = paddedN
		defer paddedBPoolBF16.Put(pbBuf)
	}

	fmopaM := paddedM

	atSize := fmopaK * fmopaM
	atBuf := transposePoolBF16.Get().([]hwy.BFloat16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.BFloat16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	transposeMatrix(fmopaA, fmopaM, fmopaK, atBuf)
	defer transposePoolBF16.Put(atBuf)

	if needsPadM || needsPadN {
		pcSize := fmopaM * fmopaN
		paddedC := paddedCPoolBF16.Get().([]hwy.BFloat16)
		if cap(paddedC) < pcSize {
			paddedC = make([]hwy.BFloat16, pcSize)
		} else {
			paddedC = paddedC[:pcSize]
		}
		clear(paddedC)
		asm.MultiTileMatMulFMOPABF16(atBuf, fmopaB, paddedC, fmopaM, fmopaN, fmopaK)
		ExtractMatrix2D(c, paddedC, m, n, fmopaN)
		paddedCPoolBF16.Put(paddedC)
	} else {
		asm.MultiTileMatMulFMOPABF16(atBuf, fmopaB, c, fmopaM, fmopaN, fmopaK)
	}
}

// =============================================================================
// Block Kernel SME wrappers
// =============================================================================

// blockMulAddFMOPAWrapper wraps the FMOPA implementation with dimension checks.
// Falls back to NEON for non-aligned dimensions or small blocks.
func blockMulAddFMOPAWrapper(aT, b, c []float32, blockDim int) {
	// FMOPA requires blockDim to be multiple of 16 (tile size for f32)
	if blockDim%16 != 0 || blockDim < 16 {
		asm.BlockMulAddNEONF32(aT, b, c, blockDim)
		return
	}
	asm.BlockMulAddFMOPAF32(aT, b, c, blockDim)
}

// blockMulAddFMOPAWrapper64 wraps the FMOPA implementation for float64.
// Falls back to NEON for non-aligned dimensions or small blocks.
func blockMulAddFMOPAWrapper64(aT, b, c []float64, blockDim int) {
	// FMOPA f64 requires blockDim to be multiple of 8 (tile size for f64)
	if blockDim%8 != 0 || blockDim < 8 {
		asm.BlockMulAddNEONF64(aT, b, c, blockDim)
		return
	}
	asm.BlockMulAddFMOPAF64(aT, b, c, blockDim)
}

// =============================================================================
// Blocked MatMul SME implementations
// =============================================================================

// blockedMatMulFMOPA uses ARM SME FMOPA for blocked matrix multiplication (f32).
// Uses outer product accumulate with ZA tiles and cache-tiled blocking.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func blockedMatMulFMOPA(a, b, c []float32, m, n, k int) {
	const tileSize = 16
	paddedM := AlignUp(m, tileSize)
	paddedN := AlignUp(n, tileSize)
	paddedK := AlignUp(k, tileSize)

	// For small matrices, use streaming NEON (SME streaming mode has overhead).
	// Check total padded ops rather than individual dimensions — SME with padding
	// is faster than NEON even at M=1 when N*K is large enough (e.g. 512x512).
	if paddedM*paddedN*paddedK < minOpsForBlockedSME {
		asm.MatMulNEONF32(a, b, c, m, n, k)
		return
	}

	// Pin goroutine to OS thread and block SIGURG to prevent async preemption
	// from corrupting ZA register state during SME streaming mode.
	defer hwy.SMEGuard()()

	needsPadM := paddedM != m
	needsPadK := paddedK != k
	needsPadN := paddedN != n

	// Prepare A: [M, K] → [paddedM, paddedK]
	fmopaA := a
	fmopaK := k
	if needsPadM || needsPadK {
		paSize := paddedM * paddedK
		paBuf := paddedAPool32.Get().([]float32)
		if cap(paBuf) < paSize {
			paBuf = make([]float32, paSize)
		} else {
			paBuf = paBuf[:paSize]
		}
		PadMatrix2D(paBuf, a, m, k, paddedM, paddedK)
		fmopaA = paBuf
		fmopaK = paddedK
		defer paddedAPool32.Put(paBuf)
	}

	// Prepare B: [K, N] → [paddedK, paddedN]
	fmopaB := b
	fmopaN := n
	if needsPadK || needsPadN {
		pbSize := paddedK * paddedN
		pbBuf := paddedBPool32.Get().([]float32)
		if cap(pbBuf) < pbSize {
			pbBuf = make([]float32, pbSize)
		} else {
			pbBuf = pbBuf[:pbSize]
		}
		PadMatrix2D(pbBuf, b, k, n, paddedK, paddedN)
		fmopaB = pbBuf
		fmopaN = paddedN
		defer paddedBPool32.Put(pbBuf)
	}

	fmopaM := paddedM

	// Transpose A [paddedM, paddedK] → AT [paddedK, paddedM]
	atSize := fmopaK * fmopaM
	atBuf := transposePool32.Get().([]float32)
	if cap(atBuf) < atSize {
		atBuf = make([]float32, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	transposeMatrix(fmopaA, fmopaM, fmopaK, atBuf)
	defer func() {
		clear(atBuf)
		transposePool32.Put(atBuf)
	}()

	// Call FMOPA; use padded C if any output dimension changed
	if needsPadM || needsPadN {
		pcSize := fmopaM * fmopaN
		paddedC := paddedCPool32.Get().([]float32)
		if cap(paddedC) < pcSize {
			paddedC = make([]float32, pcSize)
		} else {
			paddedC = paddedC[:pcSize]
		}
		clear(paddedC)
		asm.MultiTileMatMulFMOPAF32(atBuf, fmopaB, paddedC, fmopaM, fmopaN, fmopaK)
		ExtractMatrix2D(c, paddedC, m, n, fmopaN)
		paddedCPool32.Put(paddedC)
	} else {
		asm.MultiTileMatMulFMOPAF32(atBuf, fmopaB, c, fmopaM, fmopaN, fmopaK)
	}
}

// blockedMatMulFMOPA64 uses ARM SME FMOPA for blocked matrix multiplication (f64).
// Uses outer product accumulate with ZA tiles (8×8 for f64) and cache-tiled blocking.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func blockedMatMulFMOPA64(a, b, c []float64, m, n, k int) {
	const tileSize = 8
	paddedM := AlignUp(m, tileSize)
	paddedN := AlignUp(n, tileSize)
	paddedK := AlignUp(k, tileSize)

	// For small matrices, use streaming NEON (SME streaming mode has overhead).
	// See minOpsForBlockedSME comment in blockedMatMulFMOPA.
	if paddedM*paddedN*paddedK < minOpsForBlockedSME {
		asm.MatMulNEONF64(a, b, c, m, n, k)
		return
	}

	// Pin goroutine to OS thread and block SIGURG to prevent async preemption
	// from corrupting ZA register state during SME streaming mode.
	defer hwy.SMEGuard()()

	needsPadM := paddedM != m
	needsPadK := paddedK != k
	needsPadN := paddedN != n

	fmopaA := a
	fmopaK := k
	if needsPadM || needsPadK {
		paSize := paddedM * paddedK
		paBuf := paddedAPool64.Get().([]float64)
		if cap(paBuf) < paSize {
			paBuf = make([]float64, paSize)
		} else {
			paBuf = paBuf[:paSize]
		}
		PadMatrix2D(paBuf, a, m, k, paddedM, paddedK)
		fmopaA = paBuf
		fmopaK = paddedK
		defer paddedAPool64.Put(paBuf)
	}

	fmopaB := b
	fmopaN := n
	if needsPadK || needsPadN {
		pbSize := paddedK * paddedN
		pbBuf := paddedBPool64.Get().([]float64)
		if cap(pbBuf) < pbSize {
			pbBuf = make([]float64, pbSize)
		} else {
			pbBuf = pbBuf[:pbSize]
		}
		PadMatrix2D(pbBuf, b, k, n, paddedK, paddedN)
		fmopaB = pbBuf
		fmopaN = paddedN
		defer paddedBPool64.Put(pbBuf)
	}

	fmopaM := paddedM

	atSize := fmopaK * fmopaM
	atBuf := transposePool64.Get().([]float64)
	if cap(atBuf) < atSize {
		atBuf = make([]float64, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	transposeMatrix(fmopaA, fmopaM, fmopaK, atBuf)
	defer func() {
		clear(atBuf)
		transposePool64.Put(atBuf)
	}()

	if needsPadM || needsPadN {
		pcSize := fmopaM * fmopaN
		paddedC := paddedCPool64.Get().([]float64)
		if cap(paddedC) < pcSize {
			paddedC = make([]float64, pcSize)
		} else {
			paddedC = paddedC[:pcSize]
		}
		clear(paddedC)
		asm.MultiTileMatMulFMOPAF64(atBuf, fmopaB, paddedC, fmopaM, fmopaN, fmopaK)
		ExtractMatrix2D(c, paddedC, m, n, fmopaN)
		paddedCPool64.Put(paddedC)
	} else {
		asm.MultiTileMatMulFMOPAF64(atBuf, fmopaB, c, fmopaM, fmopaN, fmopaK)
	}
}

// blockedMatMulNEON uses GOAT-generated NEON for blocked matrix multiplication (f32).
// Used on non-SME hardware. For small matrices, streaming NEON is faster.
// For large matrices, blocked NEON has better cache behavior.
func blockedMatMulNEON(a, b, c []float32, m, n, k int) {
	totalOps := m * n * k
	// Below this threshold, streaming NEON is faster (~75 GFLOPS vs ~25 GFLOPS blocked)
	// Above this, blocked NEON's cache efficiency helps
	const blockedThreshold = 128 * 128 * 128 // 2M ops

	// The blocked NEON assembly crashes on some ARM64 CPUs (e.g., Ampere Altra)
	// when M is small (< BlockSize). Use streaming NEON for small M regardless
	// of total ops - blocking overhead isn't beneficial anyway for small M.
	const minMForBlocked = 48 // BlockSize

	if totalOps < blockedThreshold || m < minMForBlocked {
		asm.MatMulNEONF32(a, b, c, m, n, k)
	} else {
		asm.BlockedMatMulNEONF32(a, b, c, m, n, k)
	}
}

// blockedMatMulNEON64 uses GOAT-generated NEON for blocked matrix multiplication (f64).
func blockedMatMulNEON64(a, b, c []float64, m, n, k int) {
	totalOps := m * n * k
	const blockedThreshold = 128 * 128 * 128 // 2M ops
	const minMForBlocked = 48                // BlockSize

	if totalOps < blockedThreshold || m < minMForBlocked {
		asm.MatMulNEONF64(a, b, c, m, n, k)
	} else {
		asm.BlockedMatMulNEONF64(a, b, c, m, n, k)
	}
}

// blockedMatMulNEONF16 uses NEON for blocked float16 matmul.
func blockedMatMulNEONF16(a, b, c []hwy.Float16, m, n, k int) {
	totalOps := m * n * k
	const blockedThreshold = 128 * 128 * 128 // 2M ops
	const minMForBlocked = 48                // BlockSize

	if totalOps < blockedThreshold || m < minMForBlocked {
		asm.MatMulNEONF16(a, b, c, m, n, k)
	} else {
		asm.BlockedMatMulNEONF16(a, b, c, m, n, k)
	}
}

// blockedMatMulNEONBF16 uses NEON for blocked bfloat16 matmul.
func blockedMatMulNEONBF16(a, b, c []hwy.BFloat16, m, n, k int) {
	totalOps := m * n * k
	const blockedThreshold = 128 * 128 * 128 // 2M ops
	const minMForBlocked = 48                // BlockSize

	if totalOps < blockedThreshold || m < minMForBlocked {
		asm.MatMulNEONBF16(a, b, c, m, n, k)
	} else {
		asm.BlockedMatMulNEONBF16(a, b, c, m, n, k)
	}
}

// blockedMatMulFMOPAF16 uses SME FMOPA for blocked float16 matmul.
// This is used by ParallelMatMul for large matrices.
func blockedMatMulFMOPAF16(a, b, c []hwy.Float16, m, n, k int) {
	// The FMOPA implementation handles blocking internally with 16x16 tiles
	matmulFMOPAF16(a, b, c, m, n, k)
}

// blockedMatMulFMOPABF16 uses SME BFMOPA for blocked bfloat16 matmul.
// This is used by ParallelMatMul for large matrices.
func blockedMatMulFMOPABF16(a, b, c []hwy.BFloat16, m, n, k int) {
	// The BFMOPA implementation handles blocking internally with 16x16 tiles
	matmulFMOPABF16(a, b, c, m, n, k)
}

// =============================================================================
// NEON MatMulKLast implementations
// =============================================================================

// matmulKLastNEON uses ARM NEON for KLast matrix multiplication.
// Uses optimized tiled dot-product algorithm via GOAT-generated assembly.
// C = A * B^T where A is [M,K] and B is [N,K] (K-last layout).
func matmulKLastNEON(a, b, c []float32, m, n, k int) {
	// Fall back to scalar for small matrices
	if m < minDimForNEONKLast || n < minDimForNEONKLast || k < minDimForNEONKLast {
		BaseMatMulKLast(a, b, c, m, n, k)
		return
	}
	asm.MatMulKLastNEONF32(a, b, c, m, n, k)
}

// matmulKLastNEONF64 uses ARM NEON for float64 KLast matrix multiplication.
func matmulKLastNEONF64(a, b, c []float64, m, n, k int) {
	if m < minDimForNEONKLast || n < minDimForNEONKLast || k < minDimForNEONKLast {
		BaseMatMulKLast(a, b, c, m, n, k)
		return
	}
	asm.MatMulKLastNEONF64(a, b, c, m, n, k)
}

// matmulKLastNEONF16 uses ARM NEON for float16 KLast matrix multiplication.
// Uses f32 accumulation for precision.
func matmulKLastNEONF16(a, b, c []hwy.Float16, m, n, k int) {
	if m < minDimForNEONKLast || n < minDimForNEONKLast || k < minDimForNEONKLast {
		BaseMatMulKLast(a, b, c, m, n, k)
		return
	}
	asm.MatMulKLastNEONF16(a, b, c, m, n, k)
}

// matmulKLastNEONBF16 uses ARM NEON for bfloat16 KLast matrix multiplication.
// Uses BFDOT for computation with f32 accumulation.
func matmulKLastNEONBF16(a, b, c []hwy.BFloat16, m, n, k int) {
	if m < minDimForNEONKLast || n < minDimForNEONKLast || k < minDimForNEONKLast {
		BaseMatMulKLast(a, b, c, m, n, k)
		return
	}
	asm.MatMulKLastNEONBF16(a, b, c, m, n, k)
}

// =============================================================================
// SME FMOPA MatMulKLast implementations
// =============================================================================

// klastStripN is the strip width for incremental B transpose in MatMulKLast.
// Must be a multiple of 16 (f32 tile width). Chosen to balance cache pressure
// against streaming mode enter/exit overhead per strip.
const klastStripN = 48

// matmulKLastFMOPA uses ARM SME FMOPA for MatMulKLast with incremental B transpose.
//
// MatMulKLast computes C = A @ B^T where:
//   - A is M x K (row-major)
//   - B is N x K (row-major)
//   - C is M x N (row-major)
//
// Instead of transposing all of B upfront (O(K*N) buffer), B is transposed
// in strips of klastStripN columns. The strided FMOPA kernel writes each
// strip's output directly into the correct columns of C, avoiding any
// scatter copy.
func matmulKLastFMOPA(a, b, c []float32, m, n, k int) {
	const tileSize = 16
	paddedM := AlignUp(m, tileSize)
	paddedN := AlignUp(n, tileSize)
	paddedK := AlignUp(k, tileSize)

	// For small matrices, NEON is faster (transpose + streaming mode overhead)
	if paddedM < minDimForSMEKLast || paddedN < minDimForSMEKLast || paddedK < minDimForSMEKLast {
		asm.MatMulKLastNEONF32(a, b, c, m, n, k)
		return
	}

	// Pin goroutine to OS thread and block SIGURG to prevent async preemption
	// from corrupting ZA register state during SME streaming mode.
	defer hwy.SMEGuard()()

	needsPadM := paddedM != m
	needsPadK := paddedK != k
	needsPadN := paddedN != n

	// Prepare A: [M, K] → [paddedM, paddedK]
	fmopaA := a
	fmopaK := k
	if needsPadM || needsPadK {
		paSize := paddedM * paddedK
		paBuf := paddedAPool32.Get().([]float32)
		if cap(paBuf) < paSize {
			paBuf = make([]float32, paSize)
		} else {
			paBuf = paBuf[:paSize]
		}
		PadMatrix2D(paBuf, a, m, k, paddedM, paddedK)
		fmopaA = paBuf
		fmopaK = paddedK
		defer paddedAPool32.Put(paBuf)
	}

	fmopaM := paddedM

	// Transpose A upfront (reused across all strips)
	atSize := fmopaK * fmopaM
	atBuf := klastTransposePoolA32.Get().([]float32)
	if cap(atBuf) < atSize {
		atBuf = make([]float32, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	transposeMatrix(fmopaA, fmopaM, fmopaK, atBuf)
	defer klastTransposePoolA32.Put(atBuf)

	// B strip transpose buffer: sized for paddedK * stripN (max strip width)
	stripN := min(klastStripN, paddedN)
	btStripSize := fmopaK * stripN
	btStrip := klastTransposePoolB32.Get().([]float32)
	if cap(btStrip) < btStripSize {
		btStrip = make([]float32, btStripSize)
	} else {
		btStrip = btStrip[:btStripSize]
	}
	defer klastTransposePoolB32.Put(btStrip)

	// Output buffer: use paddedN stride if any output dimension needs padding
	var outputC []float32
	ldc := n
	if needsPadM || needsPadN {
		ldc = paddedN
		pcSize := fmopaM * paddedN
		paddedC := paddedCPool32.Get().([]float32)
		if cap(paddedC) < pcSize {
			paddedC = make([]float32, pcSize)
		} else {
			paddedC = paddedC[:pcSize]
		}
		clear(paddedC)
		outputC = paddedC
		defer func() {
			ExtractMatrix2D(c, paddedC, m, n, paddedN)
			paddedCPool32.Put(paddedC)
		}()
	} else {
		outputC = c
	}

	// Process B in strips
	if needsPadK || needsPadN {
		// Padded path: pad each B strip to aligned dimensions
		bPadSize := stripN * fmopaK
		bPadBuf := paddedBPool32.Get().([]float32)
		if cap(bPadBuf) < bPadSize {
			bPadBuf = make([]float32, bPadSize)
		} else {
			bPadBuf = bPadBuf[:bPadSize]
		}
		defer paddedBPool32.Put(bPadBuf)

		for j := 0; j < n; j += stripN {
			sn := min(stripN, n-j)
			// paddedSn is tile-aligned since paddedN is tile-aligned and stripN is a multiple of tileSize
			paddedSn := min(stripN, paddedN-j)

			// Pad B strip: [sn, k] → [paddedSn, paddedK]
			PadMatrix2D(bPadBuf[:paddedSn*fmopaK], b[j*k:(j+sn)*k], sn, k, paddedSn, fmopaK)

			// Transpose: [paddedSn, paddedK] → [paddedK, paddedSn]
			Transpose2D(bPadBuf[:paddedSn*fmopaK], paddedSn, fmopaK, btStrip[:fmopaK*paddedSn])

			asm.MultiTileMatMulFMOPAF32Strided(atBuf, btStrip[:fmopaK*paddedSn], outputC, fmopaM, paddedSn, fmopaK, ldc, j)
		}
	} else {
		// Fast path: N and K already aligned
		for j := 0; j < n; j += stripN {
			sn := min(stripN, n-j)
			Transpose2D(b[j*k:(j+sn)*k], sn, k, btStrip[:k*sn])
			asm.MultiTileMatMulFMOPAF32Strided(atBuf, btStrip[:k*sn], outputC, fmopaM, sn, k, ldc, j)
		}
	}
}

// matmulKLastFMOPA64 uses ARM SME FMOPA for float64 MatMulKLast with incremental B transpose.
func matmulKLastFMOPA64(a, b, c []float64, m, n, k int) {
	const tileSize = 8
	paddedM := AlignUp(m, tileSize)
	paddedN := AlignUp(n, tileSize)
	paddedK := AlignUp(k, tileSize)

	if paddedM < minDimForSMEKLast || paddedN < minDimForSMEKLast || paddedK < minDimForSMEKLast {
		asm.MatMulKLastNEONF64(a, b, c, m, n, k)
		return
	}

	defer hwy.SMEGuard()()

	needsPadM := paddedM != m
	needsPadK := paddedK != k
	needsPadN := paddedN != n

	fmopaA := a
	fmopaK := k
	if needsPadM || needsPadK {
		paSize := paddedM * paddedK
		paBuf := paddedAPool64.Get().([]float64)
		if cap(paBuf) < paSize {
			paBuf = make([]float64, paSize)
		} else {
			paBuf = paBuf[:paSize]
		}
		PadMatrix2D(paBuf, a, m, k, paddedM, paddedK)
		fmopaA = paBuf
		fmopaK = paddedK
		defer paddedAPool64.Put(paBuf)
	}

	fmopaM := paddedM

	atSize := fmopaK * fmopaM
	atBuf := klastTransposePoolA64.Get().([]float64)
	if cap(atBuf) < atSize {
		atBuf = make([]float64, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	transposeMatrix(fmopaA, fmopaM, fmopaK, atBuf)
	defer klastTransposePoolA64.Put(atBuf)

	stripN := min(klastStripN, paddedN)
	btStripSize := fmopaK * stripN
	btStrip := klastTransposePoolB64.Get().([]float64)
	if cap(btStrip) < btStripSize {
		btStrip = make([]float64, btStripSize)
	} else {
		btStrip = btStrip[:btStripSize]
	}
	defer klastTransposePoolB64.Put(btStrip)

	var outputC []float64
	ldc := n
	if needsPadM || needsPadN {
		ldc = paddedN
		pcSize := fmopaM * paddedN
		paddedC := paddedCPool64.Get().([]float64)
		if cap(paddedC) < pcSize {
			paddedC = make([]float64, pcSize)
		} else {
			paddedC = paddedC[:pcSize]
		}
		clear(paddedC)
		outputC = paddedC
		defer func() {
			ExtractMatrix2D(c, paddedC, m, n, paddedN)
			paddedCPool64.Put(paddedC)
		}()
	} else {
		outputC = c
	}

	if needsPadK || needsPadN {
		bPadSize := stripN * fmopaK
		bPadBuf := paddedBPool64.Get().([]float64)
		if cap(bPadBuf) < bPadSize {
			bPadBuf = make([]float64, bPadSize)
		} else {
			bPadBuf = bPadBuf[:bPadSize]
		}
		defer paddedBPool64.Put(bPadBuf)

		for j := 0; j < n; j += stripN {
			sn := min(stripN, n-j)
			paddedSn := min(stripN, paddedN-j)
			PadMatrix2D(bPadBuf[:paddedSn*fmopaK], b[j*k:(j+sn)*k], sn, k, paddedSn, fmopaK)
			Transpose2D(bPadBuf[:paddedSn*fmopaK], paddedSn, fmopaK, btStrip[:fmopaK*paddedSn])
			asm.MultiTileMatMulFMOPAF64Strided(atBuf, btStrip[:fmopaK*paddedSn], outputC, fmopaM, paddedSn, fmopaK, ldc, j)
		}
	} else {
		for j := 0; j < n; j += stripN {
			sn := min(stripN, n-j)
			Transpose2D(b[j*k:(j+sn)*k], sn, k, btStrip[:k*sn])
			asm.MultiTileMatMulFMOPAF64Strided(atBuf, btStrip[:k*sn], outputC, fmopaM, sn, k, ldc, j)
		}
	}
}

// matmulKLastFMOPAF16 uses ARM SME FMOPA for float16 MatMulKLast with incremental B transpose.
func matmulKLastFMOPAF16(a, b, c []hwy.Float16, m, n, k int) {
	const tileSize = 16
	paddedM := AlignUp(m, tileSize)
	paddedN := AlignUp(n, tileSize)
	paddedK := AlignUp(k, tileSize)

	if paddedM < minDimForSMEKLast || paddedN < minDimForSMEKLast || paddedK < minDimForSMEKLast {
		asm.MatMulKLastNEONF16(a, b, c, m, n, k)
		return
	}

	defer hwy.SMEGuard()()

	needsPadM := paddedM != m
	needsPadK := paddedK != k
	needsPadN := paddedN != n

	fmopaA := a
	fmopaK := k
	if needsPadM || needsPadK {
		paSize := paddedM * paddedK
		paBuf := paddedAPoolF16.Get().([]hwy.Float16)
		if cap(paBuf) < paSize {
			paBuf = make([]hwy.Float16, paSize)
		} else {
			paBuf = paBuf[:paSize]
		}
		PadMatrix2D(paBuf, a, m, k, paddedM, paddedK)
		fmopaA = paBuf
		fmopaK = paddedK
		defer paddedAPoolF16.Put(paBuf)
	}

	fmopaM := paddedM

	atSize := fmopaK * fmopaM
	atBuf := klastTransposePoolAF16.Get().([]hwy.Float16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.Float16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	transposeMatrix(fmopaA, fmopaM, fmopaK, atBuf)
	defer klastTransposePoolAF16.Put(atBuf)

	stripN := min(klastStripN, paddedN)
	btStripSize := fmopaK * stripN
	btStrip := klastTransposePoolBF16.Get().([]hwy.Float16)
	if cap(btStrip) < btStripSize {
		btStrip = make([]hwy.Float16, btStripSize)
	} else {
		btStrip = btStrip[:btStripSize]
	}
	defer klastTransposePoolBF16.Put(btStrip)

	var outputC []hwy.Float16
	ldc := n
	if needsPadM || needsPadN {
		ldc = paddedN
		pcSize := fmopaM * paddedN
		paddedC := paddedCPoolF16.Get().([]hwy.Float16)
		if cap(paddedC) < pcSize {
			paddedC = make([]hwy.Float16, pcSize)
		} else {
			paddedC = paddedC[:pcSize]
		}
		clear(paddedC)
		outputC = paddedC
		defer func() {
			ExtractMatrix2D(c, paddedC, m, n, paddedN)
			paddedCPoolF16.Put(paddedC)
		}()
	} else {
		outputC = c
	}

	if needsPadK || needsPadN {
		bPadSize := stripN * fmopaK
		bPadBuf := paddedBPoolF16.Get().([]hwy.Float16)
		if cap(bPadBuf) < bPadSize {
			bPadBuf = make([]hwy.Float16, bPadSize)
		} else {
			bPadBuf = bPadBuf[:bPadSize]
		}
		defer paddedBPoolF16.Put(bPadBuf)

		for j := 0; j < n; j += stripN {
			sn := min(stripN, n-j)
			paddedSn := min(stripN, paddedN-j)
			PadMatrix2D(bPadBuf[:paddedSn*fmopaK], b[j*k:(j+sn)*k], sn, k, paddedSn, fmopaK)
			Transpose2D(bPadBuf[:paddedSn*fmopaK], paddedSn, fmopaK, btStrip[:fmopaK*paddedSn])
			asm.MultiTileMatMulFMOPAF16Strided(atBuf, btStrip[:fmopaK*paddedSn], outputC, fmopaM, paddedSn, fmopaK, ldc, j)
		}
	} else {
		for j := 0; j < n; j += stripN {
			sn := min(stripN, n-j)
			Transpose2D(b[j*k:(j+sn)*k], sn, k, btStrip[:k*sn])
			asm.MultiTileMatMulFMOPAF16Strided(atBuf, btStrip[:k*sn], outputC, fmopaM, sn, k, ldc, j)
		}
	}
}

// matmulKLastFMOPABF16 uses ARM SME BFMOPA for bfloat16 MatMulKLast with incremental B transpose.
func matmulKLastFMOPABF16(a, b, c []hwy.BFloat16, m, n, k int) {
	const tileSize = 16
	paddedM := AlignUp(m, tileSize)
	paddedN := AlignUp(n, tileSize)
	paddedK := AlignUp(k, tileSize)

	if paddedM < minDimForSMEKLast || paddedN < minDimForSMEKLast || paddedK < minDimForSMEKLast {
		asm.MatMulKLastNEONBF16(a, b, c, m, n, k)
		return
	}

	defer hwy.SMEGuard()()

	needsPadM := paddedM != m
	needsPadK := paddedK != k
	needsPadN := paddedN != n

	fmopaA := a
	fmopaK := k
	if needsPadM || needsPadK {
		paSize := paddedM * paddedK
		paBuf := paddedAPoolBF16.Get().([]hwy.BFloat16)
		if cap(paBuf) < paSize {
			paBuf = make([]hwy.BFloat16, paSize)
		} else {
			paBuf = paBuf[:paSize]
		}
		PadMatrix2D(paBuf, a, m, k, paddedM, paddedK)
		fmopaA = paBuf
		fmopaK = paddedK
		defer paddedAPoolBF16.Put(paBuf)
	}

	fmopaM := paddedM

	atSize := fmopaK * fmopaM
	atBuf := klastTransposePoolABF16.Get().([]hwy.BFloat16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.BFloat16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	transposeMatrix(fmopaA, fmopaM, fmopaK, atBuf)
	defer klastTransposePoolABF16.Put(atBuf)

	stripN := min(klastStripN, paddedN)
	btStripSize := fmopaK * stripN
	btStrip := klastTransposePoolBBF16.Get().([]hwy.BFloat16)
	if cap(btStrip) < btStripSize {
		btStrip = make([]hwy.BFloat16, btStripSize)
	} else {
		btStrip = btStrip[:btStripSize]
	}
	defer klastTransposePoolBBF16.Put(btStrip)

	var outputC []hwy.BFloat16
	ldc := n
	if needsPadM || needsPadN {
		ldc = paddedN
		pcSize := fmopaM * paddedN
		paddedC := paddedCPoolBF16.Get().([]hwy.BFloat16)
		if cap(paddedC) < pcSize {
			paddedC = make([]hwy.BFloat16, pcSize)
		} else {
			paddedC = paddedC[:pcSize]
		}
		clear(paddedC)
		outputC = paddedC
		defer func() {
			ExtractMatrix2D(c, paddedC, m, n, paddedN)
			paddedCPoolBF16.Put(paddedC)
		}()
	} else {
		outputC = c
	}

	if needsPadK || needsPadN {
		bPadSize := stripN * fmopaK
		bPadBuf := paddedBPoolBF16.Get().([]hwy.BFloat16)
		if cap(bPadBuf) < bPadSize {
			bPadBuf = make([]hwy.BFloat16, bPadSize)
		} else {
			bPadBuf = bPadBuf[:bPadSize]
		}
		defer paddedBPoolBF16.Put(bPadBuf)

		for j := 0; j < n; j += stripN {
			sn := min(stripN, n-j)
			paddedSn := min(stripN, paddedN-j)
			PadMatrix2D(bPadBuf[:paddedSn*fmopaK], b[j*k:(j+sn)*k], sn, k, paddedSn, fmopaK)
			Transpose2D(bPadBuf[:paddedSn*fmopaK], paddedSn, fmopaK, btStrip[:fmopaK*paddedSn])
			asm.MultiTileMatMulFMOPABF16Strided(atBuf, btStrip[:fmopaK*paddedSn], outputC, fmopaM, paddedSn, fmopaK, ldc, j)
		}
	} else {
		for j := 0; j < n; j += stripN {
			sn := min(stripN, n-j)
			Transpose2D(b[j*k:(j+sn)*k], sn, k, btStrip[:k*sn])
			asm.MultiTileMatMulFMOPABF16Strided(atBuf, btStrip[:k*sn], outputC, fmopaM, sn, k, ldc, j)
		}
	}
}

// =============================================================================
// Packed Micro-Kernel NEON implementations
// =============================================================================

// packedMicroKernelNEONF32 wraps the GOAT-generated NEON micro-kernel.
// It adapts the signature to match the dispatched interface.
func packedMicroKernelNEONF32(packedA []float32, packedB []float32, c []float32, n, ir, jr, kc, mr, nr int) {
	cOffset := ir*n + jr
	asm.PackedMicroKernelNEONF32(packedA, packedB, c[cOffset:], kc, n, mr, nr)
}

// packedMicroKernelPartialNEONF32 handles edge micro-tiles with partial rows/columns.
func packedMicroKernelPartialNEONF32(packedA []float32, packedB []float32, c []float32, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	cOffset := ir*n + jr
	asm.PackedMicroKernelNEONF32(packedA, packedB, c[cOffset:], kc, n, activeRows, activeCols)
}

// packedMicroKernelNEONF64 wraps the GOAT-generated NEON micro-kernel for float64.
func packedMicroKernelNEONF64(packedA []float64, packedB []float64, c []float64, n, ir, jr, kc, mr, nr int) {
	cOffset := ir*n + jr
	asm.PackedMicroKernelNEONF64(packedA, packedB, c[cOffset:], kc, n, mr, nr)
}

func packedMicroKernelPartialNEONF64(packedA []float64, packedB []float64, c []float64, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	cOffset := ir*n + jr
	asm.PackedMicroKernelNEONF64(packedA, packedB, c[cOffset:], kc, n, activeRows, activeCols)
}

// packedMicroKernelNEONF16 wraps the GOAT-generated NEON FP16 micro-kernel.
func packedMicroKernelNEONF16(packedA []hwy.Float16, packedB []hwy.Float16, c []hwy.Float16, n, ir, jr, kc, mr, nr int) {
	cOffset := ir*n + jr
	asm.PackedMicroKernelNEONF16(packedA, packedB, c[cOffset:], kc, n, mr, nr)
}

func packedMicroKernelPartialNEONF16(packedA []hwy.Float16, packedB []hwy.Float16, c []hwy.Float16, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	cOffset := ir*n + jr
	asm.PackedMicroKernelNEONF16(packedA, packedB, c[cOffset:], kc, n, activeRows, activeCols)
}

// packedMicroKernelNEONBF16 wraps the GOAT-generated NEON BF16 micro-kernel.
func packedMicroKernelNEONBF16(packedA []hwy.BFloat16, packedB []hwy.BFloat16, c []hwy.BFloat16, n, ir, jr, kc, mr, nr int) {
	cOffset := ir*n + jr
	asm.PackedMicroKernelNEONBF16(packedA, packedB, c[cOffset:], kc, n, mr, nr)
}

func packedMicroKernelPartialNEONBF16(packedA []hwy.BFloat16, packedB []hwy.BFloat16, c []hwy.BFloat16, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	cOffset := ir*n + jr
	asm.PackedMicroKernelNEONBF16(packedA, packedB, c[cOffset:], kc, n, activeRows, activeCols)
}

// =============================================================================
// Fused NF4/Int4 SME implementations
// =============================================================================

// fusedNF4MatMulSME performs fused NF4 dequantization + matrix multiplication using SME.
// This is optimized for Apple M4 SME, dequantizing tiles on-the-fly.
//
// Memory usage: O(K * 16) for tile buffer instead of O(K * N) for full dequant
func fusedNF4MatMulSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if !hwy.HasSME() {
		// Fall back to scalar implementation
		BaseFusedNF4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	// Check alignment for SME (16x16 tiles)
	if K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		BaseFusedNF4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	// Pin goroutine to OS thread and block SIGURG to prevent async preemption
	// from corrupting ZA register state during SME streaming mode.
	defer hwy.SMEGuard()()

	numGroups := (N + groupSize - 1) / groupSize

	// Get tile buffer from pool
	tileBuf := fusedTilePool.Get().([]float32)
	tileSize := K * 16
	if cap(tileBuf) < tileSize {
		tileBuf = make([]float32, tileSize)
	} else {
		tileBuf = tileBuf[:tileSize]
	}
	defer fusedTilePool.Put(tileBuf)

	// Transpose buffer for input (needed for FMOPA)
	inputT := transposePool32.Get().([]float32)
	inputTSize := M * K
	if cap(inputT) < inputTSize {
		inputT = make([]float32, inputTSize)
	} else {
		inputT = inputT[:inputTSize]
	}
	defer transposePool32.Put(inputT)

	// Transpose input: [M, K] -> [K, M]
	transposeMatrix(input, M, K, inputT)

	// Zero output (strided kernel writes to sub-columns, must start from zero)
	clear(output[:M*N])

	// Process N in 16-column tiles using strided kernel to write directly to output
	for nTile := 0; nTile < N; nTile += 16 {
		nEnd := min(nTile+16, N)
		tileN := nEnd - nTile

		// Dequantize weight tile: [K, 16] from packed [K, N/2]
		dequantizeNF4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

		// Strided FMOPA: writes directly to output with stride N at column offset nTile
		asm.MultiTileMatMulFMOPAF32Strided(inputT, tileBuf[:K*tileN], output, M, tileN, K, N, nTile)
	}
}

// dequantizeNF4Tile dequantizes a K×tileN tile of NF4 weights.
// Output is row-major: tile[k*tileN + j] = weight[k, nTile+j]
func dequantizeNF4Tile(
	packed []uint8,
	scales []float32,
	tile []float32,
	nTile, K, N, tileN, numGroups, groupSize int,
) {
	for k := 0; k < K; k++ {
		for j := 0; j < tileN; j++ {
			n := nTile + j
			weightIdx := k*N + n
			packedIdx := weightIdx / 2

			var quantIdx int
			if weightIdx%2 == 0 {
				quantIdx = int(packed[packedIdx] & 0x0F)
			} else {
				quantIdx = int((packed[packedIdx] >> 4) & 0x0F)
			}

			groupIdx := n / groupSize
			scale := scales[k*numGroups+groupIdx]
			tile[k*tileN+j] = nf4LookupTable[quantIdx] * scale
		}
	}
}

// fusedInt4MatMulSME performs fused Int4 dequantization + matrix multiplication using SME.
// Similar to fusedNF4MatMulSME but for symmetric Int4 quantization.
func fusedInt4MatMulSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if !hwy.HasSME() || K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		BaseFusedInt4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	// Pin goroutine to OS thread and block SIGURG to prevent async preemption
	// from corrupting ZA register state during SME streaming mode.
	defer hwy.SMEGuard()()

	numGroups := (N + groupSize - 1) / groupSize

	tileBuf := fusedTilePool.Get().([]float32)
	tileSize := K * 16
	if cap(tileBuf) < tileSize {
		tileBuf = make([]float32, tileSize)
	} else {
		tileBuf = tileBuf[:tileSize]
	}
	defer fusedTilePool.Put(tileBuf)

	inputT := transposePool32.Get().([]float32)
	inputTSize := M * K
	if cap(inputT) < inputTSize {
		inputT = make([]float32, inputTSize)
	} else {
		inputT = inputT[:inputTSize]
	}
	defer transposePool32.Put(inputT)

	transposeMatrix(input, M, K, inputT)

	// Zero output (strided kernel writes to sub-columns, must start from zero)
	clear(output[:M*N])

	for nTile := 0; nTile < N; nTile += 16 {
		nEnd := min(nTile+16, N)
		tileN := nEnd - nTile

		dequantizeInt4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

		// Strided FMOPA: writes directly to output with stride N at column offset nTile
		asm.MultiTileMatMulFMOPAF32Strided(inputT, tileBuf[:K*tileN], output, M, tileN, K, N, nTile)
	}
}

// dequantizeInt4Tile dequantizes a K×tileN tile of Int4 weights.
// Int4 uses symmetric quantization: values in [0,15] map to [-8,7].
func dequantizeInt4Tile(
	packed []uint8,
	scales []float32,
	tile []float32,
	nTile, K, N, tileN, numGroups, groupSize int,
) {
	for k := 0; k < K; k++ {
		for j := 0; j < tileN; j++ {
			n := nTile + j
			weightIdx := k*N + n
			packedIdx := weightIdx / 2

			var unsignedVal int
			if weightIdx%2 == 0 {
				unsignedVal = int(packed[packedIdx] & 0x0F)
			} else {
				unsignedVal = int((packed[packedIdx] >> 4) & 0x0F)
			}

			groupIdx := n / groupSize
			scale := scales[k*numGroups+groupIdx]
			tile[k*tileN+j] = float32(unsignedVal-8) * scale
		}
	}
}

// processFusedNF4Tile processes a single N-tile for NF4 matmul.
// inputT is the transposed input [K, M], packed is NF4 weights, output is [M, N].
// Uses strided FMOPA to write directly to the correct columns of output.
func processFusedNF4Tile(
	inputT []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	tileBuf []float32,
	nTile, M, K, N, numGroups, groupSize int,
) {
	nEnd := min(nTile+16, N)
	tileN := nEnd - nTile

	// Dequantize weight tile: [K, tileN] from packed [K, N/2]
	dequantizeNF4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

	// Strided FMOPA: writes directly to output with stride N at column offset nTile
	asm.MultiTileMatMulFMOPAF32Strided(inputT, tileBuf[:K*tileN], output, M, tileN, K, N, nTile)
}

// processFusedInt4Tile processes a single N-tile for Int4 matmul.
// Uses strided FMOPA to write directly to the correct columns of output.
func processFusedInt4Tile(
	inputT []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	tileBuf []float32,
	nTile, M, K, N, numGroups, groupSize int,
) {
	nEnd := min(nTile+16, N)
	tileN := nEnd - nTile

	dequantizeInt4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

	// Strided FMOPA: writes directly to output with stride N at column offset nTile
	asm.MultiTileMatMulFMOPAF32Strided(inputT, tileBuf[:K*tileN], output, M, tileN, K, N, nTile)
}

// parallelFusedNF4MatMulSME performs fused NF4 matmul with parallel N-tile processing.
// Shares the transposed input across workers; each worker processes independent tiles.
func parallelFusedNF4MatMulSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if !hwy.HasSME() {
		BaseFusedNF4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	// Check alignment for SME (16x16 tiles)
	if K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		BaseFusedNF4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	numTiles := (N + 15) / 16
	numGroups := (N + groupSize - 1) / groupSize

	// Fall back to sequential if too few tiles
	if numTiles < MinFusedParallelTiles {
		fusedNF4MatMulSME(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	// Transpose input once (shared across workers, read-only)
	inputT := transposePool32.Get().([]float32)
	inputTSize := M * K
	if cap(inputT) < inputTSize {
		inputT = make([]float32, inputTSize)
	} else {
		inputT = inputT[:inputTSize]
	}
	transposeMatrix(input, M, K, inputT)
	defer transposePool32.Put(inputT)

	// Zero output (strided kernel writes to sub-columns, each tile writes independent columns)
	clear(output[:M*N])

	// Setup work queue of N-tile indices
	work := make(chan int, numTiles)
	for nTile := 0; nTile < N; nTile += 16 {
		work <- nTile
	}
	close(work)

	// Launch workers
	numWorkers := min(runtime.GOMAXPROCS(0), numTiles)
	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			// Pin goroutine to OS thread for SME streaming mode safety
			defer hwy.SMEGuard()()

			// Get thread-local tile buffer from pool
			tileBuf := fusedTilePool.Get().([]float32)
			tileSize := K * 16
			if cap(tileBuf) < tileSize {
				tileBuf = make([]float32, tileSize)
			} else {
				tileBuf = tileBuf[:tileSize]
			}
			clear(tileBuf)
			defer fusedTilePool.Put(tileBuf)

			for nTile := range work {
				processFusedNF4Tile(inputT, packed, scales, output, tileBuf,
					nTile, M, K, N, numGroups, groupSize)
			}
		})
	}
	wg.Wait()
}

// parallelFusedInt4MatMulSME performs fused Int4 matmul with parallel N-tile processing.
func parallelFusedInt4MatMulSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if !hwy.HasSME() {
		BaseFusedInt4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	if K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		BaseFusedInt4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	numTiles := (N + 15) / 16
	numGroups := (N + groupSize - 1) / groupSize

	if numTiles < MinFusedParallelTiles {
		fusedInt4MatMulSME(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	// Transpose input once (shared across workers, read-only)
	inputT := transposePool32.Get().([]float32)
	inputTSize := M * K
	if cap(inputT) < inputTSize {
		inputT = make([]float32, inputTSize)
	} else {
		inputT = inputT[:inputTSize]
	}
	transposeMatrix(input, M, K, inputT)
	defer transposePool32.Put(inputT)

	// Zero output (strided kernel writes to sub-columns, each tile writes independent columns)
	clear(output[:M*N])

	// Setup work queue of N-tile indices
	work := make(chan int, numTiles)
	for nTile := 0; nTile < N; nTile += 16 {
		work <- nTile
	}
	close(work)

	numWorkers := min(runtime.GOMAXPROCS(0), numTiles)
	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			// Pin goroutine to OS thread for SME streaming mode safety
			defer hwy.SMEGuard()()

			// Get thread-local tile buffer from pool
			tileBuf := fusedTilePool.Get().([]float32)
			tileSize := K * 16
			if cap(tileBuf) < tileSize {
				tileBuf = make([]float32, tileSize)
			} else {
				tileBuf = tileBuf[:tileSize]
			}
			clear(tileBuf)
			defer fusedTilePool.Put(tileBuf)

			for nTile := range work {
				processFusedInt4Tile(inputT, packed, scales, output, tileBuf,
					nTile, M, K, N, numGroups, groupSize)
			}
		})
	}
	wg.Wait()
}

// =============================================================================
// init() - Dispatch setup
// =============================================================================

func init() {
	// Skip NEON assembly if HWY_NO_SIMD is set - use pure Go fallback instead.
	if hwy.NoSimdEnv() {
		return
	}

	// Check for NEON capability
	lanesF32 := hwy.Zero[float32]().NumLanes()
	hasNEON := lanesF32 >= 4

	// ==========================================================================
	// Float32 MatMul dispatch
	// ==========================================================================
	if hwy.HasSME() {
		// Use FMOPA implementation which works on Apple M4
		MatMulFloat32 = matmulFMOPA
		MatMulFloat64 = matmulFMOPA64

		// Fused NF4/Int4 SME implementations
		FusedNF4MatMul = fusedNF4MatMulSME
		FusedInt4MatMul = fusedInt4MatMulSME
		ParallelFusedNF4MatMul = parallelFusedNF4MatMulSME
		ParallelFusedInt4MatMul = parallelFusedInt4MatMulSME
	} else {
		// Use hand-written NEON implementation on arm64
		MatMulFloat32 = matmulNEON
	}

	// ==========================================================================
	// Float32/Float64 BlockedMatMul dispatch
	// ==========================================================================
	if hwy.HasSME() {
		// Use blocked FMOPA implementation which works on Apple M4
		BlockedMatMulFloat32 = blockedMatMulFMOPA
		BlockedMatMulFloat64 = blockedMatMulFMOPA64

		// Override dispatch to use FMOPA for aligned dimensions
		BlockMulAddFloat32 = blockMulAddFMOPAWrapper
		BlockMulAddFloat64 = blockMulAddFMOPAWrapper64
	} else {
		// Use GOAT-generated NEON (13x faster than hwygen: 25 GFLOPS vs 2 GFLOPS)
		// with streaming NEON fallback for small sizes
		BlockedMatMulFloat32 = blockedMatMulNEON
		BlockedMatMulFloat64 = blockedMatMulNEON64
	}

	// ==========================================================================
	// Float16/BFloat16 MatMul dispatch based on CPU feature detection
	// ==========================================================================
	if hwy.HasSME() {
		// Use SME FMOPA for F16/BF16 when available
		if hwy.HasARMFP16() {
			MatMulFloat16 = matmulFMOPAF16
			BlockedMatMulFloat16 = blockedMatMulFMOPAF16
		}
		if hwy.HasARMBF16() {
			MatMulBFloat16 = matmulFMOPABF16
			BlockedMatMulBFloat16 = blockedMatMulFMOPABF16
		}
	} else {
		// Use optimized NEON path if CPU supports FP16
		if hwy.HasARMFP16() {
			MatMulFloat16 = matmulNEONF16
			BlockedMatMulFloat16 = blockedMatMulNEONF16
		} else {
			MatMulFloat16 = BaseMatMul_fallback_Float16
			BlockedMatMulFloat16 = BaseBlockedMatMul_fallback_Float16
		}

		// Use optimized NEON path if CPU supports BF16
		if hwy.HasARMBF16() {
			MatMulBFloat16 = matmulNEONBF16
			BlockedMatMulBFloat16 = blockedMatMulNEONBF16
		} else {
			MatMulBFloat16 = BaseMatMul_fallback_BFloat16
			BlockedMatMulBFloat16 = BaseBlockedMatMul_fallback_BFloat16
		}
	}

	// ==========================================================================
	// MatMulKLast dispatch
	// ==========================================================================
	if hwy.HasSME() {
		// Use FMOPA implementation for large aligned matrices
		MatMulKLastFloat32 = matmulKLastFMOPA
		MatMulKLastFloat64 = matmulKLastFMOPA64
		MatMulKLastFloat16 = matmulKLastFMOPAF16
		MatMulKLastBFloat16 = matmulKLastFMOPABF16

		// Blocked versions use the same approach
		MatMulKLastBlockedFloat32 = matmulKLastFMOPA
		MatMulKLastBlockedFloat64 = matmulKLastFMOPA64
		MatMulKLastBlockedFloat16 = matmulKLastFMOPAF16
		MatMulKLastBlockedBFloat16 = matmulKLastFMOPABF16
	} else {
		// Use GOAT-generated NEON assembly for arm64
		MatMulKLastFloat32 = matmulKLastNEON
		MatMulKLastFloat64 = matmulKLastNEONF64

		// Blocked versions use the same NEON implementations
		MatMulKLastBlockedFloat32 = matmulKLastNEON
		MatMulKLastBlockedFloat64 = matmulKLastNEONF64

		// FP16/BF16 require ARMv8.2+ extensions
		if hwy.HasARMFP16() {
			MatMulKLastFloat16 = matmulKLastNEONF16
			MatMulKLastBlockedFloat16 = matmulKLastNEONF16
		}
		if hwy.HasARMBF16() {
			MatMulKLastBFloat16 = matmulKLastNEONBF16
			MatMulKLastBlockedBFloat16 = matmulKLastNEONBF16
		}
	}

	// ==========================================================================
	// Packed Micro-Kernel dispatch (for GEBP algorithm)
	// ==========================================================================
	if hasNEON && !hwy.HasSME() {
		// Float32
		PackedMicroKernelFloat32 = packedMicroKernelNEONF32
		PackedMicroKernelPartialFloat32 = packedMicroKernelPartialNEONF32

		// Float64
		PackedMicroKernelFloat64 = packedMicroKernelNEONF64
		PackedMicroKernelPartialFloat64 = packedMicroKernelPartialNEONF64
	}

	// F16: Requires ARMv8.2-A FP16 extension
	if hasNEON && hwy.HasARMFP16() && !hwy.HasSME() {
		PackedMicroKernelFloat16 = packedMicroKernelNEONF16
		PackedMicroKernelPartialFloat16 = packedMicroKernelPartialNEONF16
	}

	// BF16: Requires ARMv8.6-A BF16 extension
	if hasNEON && hwy.HasARMBF16() && !hwy.HasSME() {
		PackedMicroKernelBFloat16 = packedMicroKernelNEONBF16
		PackedMicroKernelPartialBFloat16 = packedMicroKernelPartialNEONBF16
	}
}
