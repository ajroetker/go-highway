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
	"unsafe"

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

// Minimum dimensions to use SME blocked FMOPA.
// SME with transpose is 160x+ faster than hwygen-generated NEON blocked even at 32×32.
// Benchmarks on Apple M4: 32×32: SME 323 GFLOPS vs NEON 2 GFLOPS.
const minDimForBlockedSME = 32

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

// Output tile pool: M * 16 floats for SME tile width (M up to 4096)
var fusedOutputTilePool = sync.Pool{
	New: func() any {
		return make([]float32, 0, 4096*16)
	},
}

// =============================================================================
// Helper functions
// =============================================================================

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
	// For non-aligned sizes, fall back to generated NEON
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		matmulNEON(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEON(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool32.Get().([]float32)
	if cap(atBuf) < atSize {
		atBuf = make([]float32, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Call FMOPA with transposed A
	asm.MatMulFMOPAF32(atBuf, b, c, m, n, k)

	// Return buffer to pool
	transposePool32.Put(atBuf)
}

// matmulFMOPA64 uses ARM SME FMOPA instruction for float64 matrix multiplication.
// Uses outer product accumulate with ZA tiles - 8×8 tiles for float64.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPA64(a, b, c []float64, m, n, k int) {
	// For non-aligned sizes (8×8 tiles for float64), fall back to scalar
	if m%8 != 0 || n%8 != 0 || k%8 != 0 {
		matmulScalar64(a, b, c, m, n, k)
		return
	}

	// For small matrices, scalar is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulScalar64(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool64.Get().([]float64)
	if cap(atBuf) < atSize {
		atBuf = make([]float64, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Call FMOPA with transposed A
	asm.MatMulFMOPAF64(atBuf, b, c, m, n, k)

	// Return buffer to pool
	transposePool64.Put(atBuf)
}

// matmulFMOPAF16 uses ARM SME FMOPA instruction for float16 matrix multiplication.
// Uses widening: f16 -> f32 FMOPA -> f16, with 16×16 tiles (f32 accumulator).
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPAF16(a, b, c []hwy.Float16, m, n, k int) {
	// For non-aligned sizes (16×16 tiles), fall back to NEON
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		matmulNEONF16(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEONF16(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePoolF16.Get().([]hwy.Float16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.Float16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Scratch buffer for f32->f16 conversion
	var scratch [16]float32
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	// Call FMOPA with transposed A (from asm package)
	asm.MatmulFmopaAtF16(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)

	// Return buffer to pool
	transposePoolF16.Put(atBuf)
}

// matmulFMOPABF16 uses ARM SME BFMOPA instruction for bfloat16 matrix multiplication.
// Uses widening: bf16 -> f32 FMOPA -> bf16, with 16×16 tiles (f32 accumulator).
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPABF16(a, b, c []hwy.BFloat16, m, n, k int) {
	// For non-aligned sizes (16×16 tiles), fall back to NEON
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		matmulNEONBF16(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEONBF16(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePoolBF16.Get().([]hwy.BFloat16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.BFloat16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Scratch buffer for f32->bf16 conversion
	var scratch [16]float32
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	// Call BFMOPA with transposed A (from asm package)
	asm.MatmulBfmopaAtBF16(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)

	// Return buffer to pool
	transposePoolBF16.Put(atBuf)
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
	// For non-aligned sizes (16×16 tiles for f32), fall back to GOAT NEON streaming
	// (hwygen-generated blocked NEON is only ~2 GFLOPS, streaming NEON is ~75 GFLOPS)
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		asm.MatMulNEONF32(a, b, c, m, n, k)
		return
	}

	// For small matrices, use streaming NEON (SME streaming mode has overhead)
	if m < minDimForBlockedSME || n < minDimForBlockedSME || k < minDimForBlockedSME {
		asm.MatMulNEONF32(a, b, c, m, n, k)
		return
	}

	// Pin goroutine to OS thread and block SIGURG to prevent async preemption
	// from corrupting ZA register state during SME streaming mode.
	defer asm.SMEGuard()()

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool32.Get().([]float32)
	if cap(atBuf) < atSize {
		atBuf = make([]float32, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	clear(atBuf)

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Call blocked FMOPA with transposed A
	asm.BlockedMatMulFMOPAF32(atBuf, b, c, m, n, k)

	// Clear buffer before returning to pool to avoid stale data
	clear(atBuf)
	transposePool32.Put(atBuf)
}

// blockedMatMulFMOPA64 uses ARM SME FMOPA for blocked matrix multiplication (f64).
// Uses outer product accumulate with ZA tiles (8×8 for f64) and cache-tiled blocking.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func blockedMatMulFMOPA64(a, b, c []float64, m, n, k int) {
	// For non-aligned sizes (8×8 tiles for f64), fall back to GOAT NEON streaming
	// (hwygen-generated blocked NEON is only ~2 GFLOPS, streaming NEON is ~75 GFLOPS)
	if m%8 != 0 || n%8 != 0 || k%8 != 0 {
		asm.MatMulNEONF64(a, b, c, m, n, k)
		return
	}

	// For small matrices, use streaming NEON (SME streaming mode has overhead)
	if m < minDimForBlockedSME || n < minDimForBlockedSME || k < minDimForBlockedSME {
		asm.MatMulNEONF64(a, b, c, m, n, k)
		return
	}

	// Pin goroutine to OS thread and block SIGURG to prevent async preemption
	// from corrupting ZA register state during SME streaming mode.
	defer asm.SMEGuard()()

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool64.Get().([]float64)
	if cap(atBuf) < atSize {
		atBuf = make([]float64, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}
	clear(atBuf)

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Call blocked FMOPA with transposed A
	asm.BlockedMatMulFMOPAF64(atBuf, b, c, m, n, k)

	// Clear buffer before returning to pool to avoid stale data
	clear(atBuf)
	transposePool64.Put(atBuf)
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

// matmulKLastFMOPA uses ARM SME FMOPA for MatMulKLast.
//
// MatMulKLast computes C = A @ B^T where:
//   - A is M x K (row-major)
//   - B is N x K (row-major)
//   - C is M x N (row-major)
//
// The existing FMOPA kernel expects:
//   - AT: K x M (transposed A)
//   - B:  K x N (normal B)
//   - Computes: (AT)^T @ B = A @ B
//
// For MatMulKLast (C = A @ B^T):
//   - Transpose A to get AT [K, M]
//   - Transpose B to get BT [K, N]
//   - Call FMOPA(AT, BT) which computes (AT)^T @ BT = A @ B^T ✓
func matmulKLastFMOPA(a, b, c []float32, m, n, k int) {
	// For non-aligned sizes, fall back to NEON dot-product assembly
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		asm.MatMulKLastNEONF32(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (transpose + streaming mode overhead)
	if m < minDimForSMEKLast || n < minDimForSMEKLast || k < minDimForSMEKLast {
		asm.MatMulKLastNEONF32(a, b, c, m, n, k)
		return
	}

	// Get transpose buffers from pool
	atSize := k * m
	btSize := k * n

	atBuf := klastTransposePoolA32.Get().([]float32)
	if cap(atBuf) < atSize {
		atBuf = make([]float32, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	btBuf := klastTransposePoolB32.Get().([]float32)
	if cap(btBuf) < btSize {
		btBuf = make([]float32, btSize)
	} else {
		btBuf = btBuf[:btSize]
	}

	// Transpose A (M×K) to AT (K×M)
	transposeMatrix(a, m, k, atBuf)

	// Transpose B (N×K) to BT (K×N)
	transposeMatrix(b, n, k, btBuf)

	// Call FMOPA: (AT)^T @ BT = A @ B^T
	asm.MatMulFMOPAF32(atBuf, btBuf, c, m, n, k)

	// Return buffers to pool
	klastTransposePoolA32.Put(atBuf)
	klastTransposePoolB32.Put(btBuf)
}

// matmulKLastFMOPA64 uses ARM SME FMOPA for float64 MatMulKLast.
func matmulKLastFMOPA64(a, b, c []float64, m, n, k int) {
	// For non-aligned sizes (8×8 tiles for float64), fall back to NEON assembly
	if m%8 != 0 || n%8 != 0 || k%8 != 0 {
		asm.MatMulKLastNEONF64(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster
	if m < minDimForSMEKLast || n < minDimForSMEKLast || k < minDimForSMEKLast {
		asm.MatMulKLastNEONF64(a, b, c, m, n, k)
		return
	}

	// Get transpose buffers from pool
	atSize := k * m
	btSize := k * n

	atBuf := klastTransposePoolA64.Get().([]float64)
	if cap(atBuf) < atSize {
		atBuf = make([]float64, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	btBuf := klastTransposePoolB64.Get().([]float64)
	if cap(btBuf) < btSize {
		btBuf = make([]float64, btSize)
	} else {
		btBuf = btBuf[:btSize]
	}

	// Transpose A (M×K) to AT (K×M)
	transposeMatrix(a, m, k, atBuf)

	// Transpose B (N×K) to BT (K×N)
	transposeMatrix(b, n, k, btBuf)

	// Call FMOPA: (AT)^T @ BT = A @ B^T
	asm.MatMulFMOPAF64(atBuf, btBuf, c, m, n, k)

	// Return buffers to pool
	klastTransposePoolA64.Put(atBuf)
	klastTransposePoolB64.Put(btBuf)
}

// matmulKLastFMOPAF16 uses ARM SME FMOPA for float16 MatMulKLast.
func matmulKLastFMOPAF16(a, b, c []hwy.Float16, m, n, k int) {
	// For non-aligned sizes, fall back to NEON assembly
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		asm.MatMulKLastNEONF16(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster
	if m < minDimForSMEKLast || n < minDimForSMEKLast || k < minDimForSMEKLast {
		asm.MatMulKLastNEONF16(a, b, c, m, n, k)
		return
	}

	// Get transpose buffers from pool
	atSize := k * m
	btSize := k * n

	atBuf := klastTransposePoolAF16.Get().([]hwy.Float16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.Float16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	btBuf := klastTransposePoolBF16.Get().([]hwy.Float16)
	if cap(btBuf) < btSize {
		btBuf = make([]hwy.Float16, btSize)
	} else {
		btBuf = btBuf[:btSize]
	}

	// Transpose A (M×K) to AT (K×M)
	transposeMatrix(a, m, k, atBuf)

	// Transpose B (N×K) to BT (K×N)
	transposeMatrix(b, n, k, btBuf)

	// Scratch buffer for f32->f16 conversion
	var scratch [16]float32
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	// Call FMOPA: (AT)^T @ BT = A @ B^T
	asm.MatmulFmopaAtF16(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(btBuf)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)

	// Return buffers to pool
	klastTransposePoolAF16.Put(atBuf)
	klastTransposePoolBF16.Put(btBuf)
}

// matmulKLastFMOPABF16 uses ARM SME BFMOPA for bfloat16 MatMulKLast.
func matmulKLastFMOPABF16(a, b, c []hwy.BFloat16, m, n, k int) {
	// For non-aligned sizes, fall back to NEON assembly
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		asm.MatMulKLastNEONBF16(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster
	if m < minDimForSMEKLast || n < minDimForSMEKLast || k < minDimForSMEKLast {
		asm.MatMulKLastNEONBF16(a, b, c, m, n, k)
		return
	}

	// Get transpose buffers from pool
	atSize := k * m
	btSize := k * n

	atBuf := klastTransposePoolABF16.Get().([]hwy.BFloat16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.BFloat16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	btBuf := klastTransposePoolBBF16.Get().([]hwy.BFloat16)
	if cap(btBuf) < btSize {
		btBuf = make([]hwy.BFloat16, btSize)
	} else {
		btBuf = btBuf[:btSize]
	}

	// Transpose A (M×K) to AT (K×M)
	transposeMatrix(a, m, k, atBuf)

	// Transpose B (N×K) to BT (K×N)
	transposeMatrix(b, n, k, btBuf)

	// Scratch buffer for f32->bf16 conversion
	var scratch [16]float32
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	// Call BFMOPA: (AT)^T @ BT = A @ B^T
	asm.MatmulBfmopaAtBF16(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(btBuf)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)

	// Return buffers to pool
	klastTransposePoolABF16.Put(atBuf)
	klastTransposePoolBBF16.Put(btBuf)
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

	// Output tile buffer from pool
	outputTileSize := M * 16
	outputTile := fusedOutputTilePool.Get().([]float32)
	if cap(outputTile) < outputTileSize {
		outputTile = make([]float32, outputTileSize)
	} else {
		outputTile = outputTile[:outputTileSize]
	}
	defer fusedOutputTilePool.Put(outputTile)

	// Process N in 16-column tiles
	for nTile := 0; nTile < N; nTile += 16 {
		nEnd := min(nTile+16, N)
		tileN := nEnd - nTile

		// Dequantize weight tile: [K, 16] from packed [K, N/2]
		dequantizeNF4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

		// SME matmul: inputT[K, M]^T @ tileBuf[K, tileN] = input[M, K] @ tile[K, tileN]
		// Use the optimized FMOPA kernel
		asm.MatMulFMOPAF32(inputT, tileBuf[:K*tileN], outputTile[:M*tileN], M, tileN, K)

		// Copy output tile to final output (scatter to correct columns)
		for m := 0; m < M; m++ {
			for j := 0; j < tileN; j++ {
				output[m*N+nTile+j] = outputTile[m*tileN+j]
			}
		}
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

	// Output tile buffer from pool
	outputTileSize := M * 16
	outputTile := fusedOutputTilePool.Get().([]float32)
	if cap(outputTile) < outputTileSize {
		outputTile = make([]float32, outputTileSize)
	} else {
		outputTile = outputTile[:outputTileSize]
	}
	defer fusedOutputTilePool.Put(outputTile)

	for nTile := 0; nTile < N; nTile += 16 {
		nEnd := min(nTile+16, N)
		tileN := nEnd - nTile

		dequantizeInt4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

		asm.MatMulFMOPAF32(inputT, tileBuf[:K*tileN], outputTile[:M*tileN], M, tileN, K)

		for m := 0; m < M; m++ {
			for j := 0; j < tileN; j++ {
				output[m*N+nTile+j] = outputTile[m*tileN+j]
			}
		}
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
func processFusedNF4Tile(
	inputT []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	tileBuf []float32,
	outputTile []float32,
	nTile, M, K, N, numGroups, groupSize int,
) {
	nEnd := min(nTile+16, N)
	tileN := nEnd - nTile

	// Dequantize weight tile: [K, tileN] from packed [K, N/2]
	dequantizeNF4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

	// SME matmul: inputT[K, M]^T @ tileBuf[K, tileN] = input[M, K] @ tile[K, tileN]
	asm.MatMulFMOPAF32(inputT, tileBuf[:K*tileN], outputTile[:M*tileN], M, tileN, K)

	// Copy output tile to final output (scatter to correct columns)
	for m := 0; m < M; m++ {
		for j := 0; j < tileN; j++ {
			output[m*N+nTile+j] = outputTile[m*tileN+j]
		}
	}
}

// processFusedInt4Tile processes a single N-tile for Int4 matmul.
func processFusedInt4Tile(
	inputT []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	tileBuf []float32,
	outputTile []float32,
	nTile, M, K, N, numGroups, groupSize int,
) {
	nEnd := min(nTile+16, N)
	tileN := nEnd - nTile

	dequantizeInt4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

	asm.MatMulFMOPAF32(inputT, tileBuf[:K*tileN], outputTile[:M*tileN], M, tileN, K)

	for m := 0; m < M; m++ {
		for j := 0; j < tileN; j++ {
			output[m*N+nTile+j] = outputTile[m*tileN+j]
		}
	}
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
			// Get thread-local buffers from pool
			tileBuf := fusedTilePool.Get().([]float32)
			tileSize := K * 16
			if cap(tileBuf) < tileSize {
				tileBuf = make([]float32, tileSize)
			} else {
				tileBuf = tileBuf[:tileSize]
			}
			// Clear AFTER getting from pool to ensure no stale data
			// (sync.Pool may return buffers from different threads' caches)
			clear(tileBuf)
			defer fusedTilePool.Put(tileBuf)

			outputTile := fusedOutputTilePool.Get().([]float32)
			outputTileSize := M * 16
			if cap(outputTile) < outputTileSize {
				outputTile = make([]float32, outputTileSize)
			} else {
				outputTile = outputTile[:outputTileSize]
			}
			// Clear AFTER getting from pool to ensure no stale data
			clear(outputTile)
			defer fusedOutputTilePool.Put(outputTile)

			for nTile := range work {
				processFusedNF4Tile(inputT, packed, scales, output, tileBuf, outputTile,
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
			// Get thread-local buffers from pool
			tileBuf := fusedTilePool.Get().([]float32)
			tileSize := K * 16
			if cap(tileBuf) < tileSize {
				tileBuf = make([]float32, tileSize)
			} else {
				tileBuf = tileBuf[:tileSize]
			}
			// Clear AFTER getting from pool to ensure no stale data
			clear(tileBuf)
			defer fusedTilePool.Put(tileBuf)

			outputTile := fusedOutputTilePool.Get().([]float32)
			outputTileSize := M * 16
			if cap(outputTile) < outputTileSize {
				outputTile = make([]float32, outputTileSize)
			} else {
				outputTile = outputTile[:outputTileSize]
			}
			// Clear AFTER getting from pool to ensure no stale data
			clear(outputTile)
			defer fusedOutputTilePool.Put(outputTile)

			for nTile := range work {
				processFusedInt4Tile(inputT, packed, scales, output, tileBuf, outputTile,
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
