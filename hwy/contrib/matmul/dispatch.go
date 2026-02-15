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

// Size-based dispatch thresholds.
// Tuned empirically - adjust based on benchmarks on target hardware.
const (
	// Below this total ops count, streaming is faster (less overhead)
	SmallMatrixThreshold = 64 * 64 * 64 // 262144 ops

	// Above this total ops count, use V2 packed matmul on AMD64 for best cache efficiency
	// 1024^3 = 1B ops, where K-blocking benefit outweighs V2 overhead
	// Benchmarks on AMD EPYC 7763 (AVX2) show V2 is slower until ~1024x1024:
	//   256x256: V2 +8% slower, 512x512: V2 +32% slower, 1024x1024: V2 -8% faster
	LargeMatrixThreshold = 1024 * 1024 * 1024 // 1073741824 ops

	// When K/N ratio exceeds this, blocking helps reduce C traffic
	DeepKRatio = 4

	// MinParallelStrips is the minimum number of RowsPerStrip-sized strips
	// required for coarse-grained parallelism to overcome dispatch overhead.
	// Benchmarks on M4 Max (ARM64 SME) show:
	//   2 strips (96x96x96, 128x128x128): Parallel 14-33% SLOWER than Blocked
	//   3 strips (192x192x192): Parallel 28% faster
	//   4+ strips: Parallel consistently faster (up to 2.6x at 16 strips)
	MinParallelStrips = 3
)

// MatMulAuto automatically selects the best algorithm based on matrix dimensions.
// Requires a persistent worker pool for parallel execution.
//
// Algorithm selection:
//
//  1. Small matrices (M*N*K < 64^3): Streaming MatMul — lowest overhead
//  2. Small M on AMD64 (M < RowsPerStrip): Fine-grained row parallelism —
//     each row dispatched via atomic work stealing.
//  3. Few strips (M/RowsPerStrip < 3): Sequential BlockedMatMul — parallel
//     dispatch overhead exceeds benefit with <3 strips.
//  4. Large (AMD64 only, M*N*K >= 1024^3): ParallelPackedMatMulV2 with K-blocking.
//  5. Default: ParallelMatMul with 64-row strips.
//
// On ARM64 with SME, BlockedMatMul uses FMOPA outer products with padding for
// any size where total padded ops >= 64K (including M=1). SME with padding is
// 1.5-92x faster than NEON even at small M. Fine-grained per-row dispatch is
// not used on ARM64 because splitting rows forces each sub-call through NEON
// (individual rows can't reach the SME ops threshold).
//
// Usage:
//
//	pool := workerpool.New(runtime.GOMAXPROCS(0))
//	defer pool.Close()
//
//	for _, layer := range layers {
//	    matmul.MatMulAuto(pool, a, b, c, m, n, k)
//	}
func MatMulAuto[T hwy.Floats](pool workerpool.Executor, a, b, c []T, m, n, k int) {
	totalOps := m * n * k

	// Very small matrices: streaming is fastest (fits in cache, no overhead).
	if totalOps < SmallMatrixThreshold {
		MatMul(a, b, c, m, n, k)
		return
	}

	// For small M with large N*K on AMD64, use fine-grained row parallelism.
	// Each row is dispatched independently via atomic work stealing.
	//
	// On ARM64, this path is skipped. BlockedMatMul now uses SME FMOPA with
	// padding even for M=1 (total ops guard instead of per-dimension guard),
	// so sequential BlockedMatMul is already fast. Per-row FineGrained dispatch
	// would force each row through NEON since M=1 per-row calls can't reach
	// the SME ops threshold. Benchmarks on M4 Max (SME with padding):
	//   BlockedMatMul(1, 1024, 1024):  ~93µs  (SME, padded to 16×1024×1024)
	//   BlockedMatMul(16, 1024, 1024): ~88µs  (SME, padded to 16×1024×1024)
	//   BlockedMatMul(32, 1024, 1024): ~101µs (SME, no padding needed)
	if runtime.GOARCH != "arm64" && m < RowsPerStrip {
		ParallelMatMulFineGrained(pool, a, b, c, m, n, k)
		return
	}

	// Coarse parallelism requires enough strips for load balancing.
	// With <3 strips, dispatch overhead exceeds the parallelism benefit.
	// Benchmarks on M4 Max:
	//   96x96x96  (2 strips): Parallel 33% slower than Blocked
	//   128x128x128 (2 strips): Parallel 14% slower
	//   192x192x192 (3 strips): Parallel 28% faster
	numStrips := (m + RowsPerStrip - 1) / RowsPerStrip
	if numStrips < MinParallelStrips {
		BlockedMatMul(a, b, c, m, n, k)
		return
	}

	if totalOps >= LargeMatrixThreshold && runtime.GOARCH != "arm64" {
		// Use optimized V2 packed GEBP with K-blocking on AMD64.
		ParallelPackedMatMulV2(pool, a, b, c, m, n, k)
	} else {
		ParallelMatMul(pool, a, b, c, m, n, k)
	}
}

// MatMulKLastAuto automatically selects the best algorithm for K-last layout.
// Requires a persistent worker pool for parallel execution.
//
// K-last layout: A is [M,K], B is [N,K] (both with K as last dimension).
// Computes C = A @ B^T where C is [M,N].
//
// Algorithm selection mirrors MatMulAuto:
//  1. Small matrices (M*N*K < 64^3): Streaming MatMulKLast
//  2. Small M on AMD64 (M < RowsPerStrip): Fine-grained row parallelism
//  3. Few strips (< 3): Sequential MatMulKLastBlocked
//  4. Default: ParallelMatMulKLast with coarse row striping
//
// On ARM64 with SME, FMOPA with padding handles small M directly.
func MatMulKLastAuto[T hwy.Floats](pool workerpool.Executor, a, b, c []T, m, n, k int) {
	totalOps := m * n * k

	if totalOps < SmallMatrixThreshold {
		MatMulKLast(a, b, c, m, n, k)
		return
	}

	// Fine-grained row parallelism for small M on AMD64.
	// On ARM64, BlockedMatMul handles small M via SME with padding.
	// See MatMulAuto comments for full rationale.
	if runtime.GOARCH != "arm64" && m < RowsPerStrip {
		ParallelMatMulKLastFineGrained(pool, a, b, c, m, n, k)
		return
	}

	// Need enough strips for coarse parallelism to overcome overhead.
	numStrips := (m + RowsPerStrip - 1) / RowsPerStrip
	if numStrips < MinParallelStrips {
		MatMulKLastBlocked(a, b, c, m, n, k)
		return
	}

	ParallelMatMulKLast(pool, a, b, c, m, n, k)
}

// =============================================================================
// Parallel Fused MatMul dispatch
// =============================================================================

// ParallelFusedNF4MatMul performs fused NF4 dequantization + matrix multiplication
// with parallel execution for large matrices.
// Dispatches to the best available implementation for the current platform.
// On platforms with SME, this uses tiled parallel execution.
// On other platforms, this falls back to the serial implementation.
var ParallelFusedNF4MatMul func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int)

// ParallelFusedInt4MatMul performs fused Int4 dequantization + matrix multiplication
// with parallel execution for large matrices.
// Dispatches to the best available implementation for the current platform.
// On platforms with SME, this uses tiled parallel execution.
// On other platforms, this falls back to the serial implementation.
var ParallelFusedInt4MatMul func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int)

// ParallelFusedInt8MatMul performs fused Int8 dequantization + matrix multiplication
// with parallel execution for large matrices.
// On platforms with SME, this uses tiled parallel execution.
// On other platforms, this falls back to the serial implementation.
var ParallelFusedInt8MatMul func(pool workerpool.Executor, input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int)

// FusedNF4MatMulAct performs fused NF4 dequantization + matmul + optional bias + activation.
// Dispatches to the best available implementation for the current platform.
// On non-SME platforms, this switches through the individual activation-specific functions.
// On SME platforms, this is overridden to use the unified SME implementation directly.
var FusedNF4MatMulAct func(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType)

// FusedInt4MatMulAct performs fused Int4 dequantization + matmul + optional bias + activation.
// Dispatches to the best available implementation for the current platform.
var FusedInt4MatMulAct func(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType)

// ParallelFusedNF4MatMulAct performs parallel fused NF4 + optional bias + activation for large matrices.
// On SME platforms, this uses tiled parallel execution.
// On other platforms, this falls back to the serial FusedNF4MatMulAct.
var ParallelFusedNF4MatMulAct func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType)

// ParallelFusedInt4MatMulAct performs parallel fused Int4 + optional bias + activation for large matrices.
// On SME platforms, this uses tiled parallel execution.
// On other platforms, this falls back to the serial FusedInt4MatMulAct.
var ParallelFusedInt4MatMulAct func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType)

// FusedInt8MatMulAct performs fused Int8 dequantization + matmul + optional bias + activation.
// Dispatches to the best available implementation for the current platform.
var FusedInt8MatMulAct func(input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType)

// ParallelFusedInt8MatMulAct performs parallel fused Int8 + optional bias + activation for large matrices.
// On SME platforms, this uses tiled parallel execution.
// On other platforms, this falls back to the serial FusedInt8MatMulAct.
var ParallelFusedInt8MatMulAct func(pool workerpool.Executor, input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType)

// ParallelFusedNF4MatMulSiLU performs parallel fused NF4 + bias + SiLU for large matrices.
var ParallelFusedNF4MatMulSiLU func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int)

// ParallelFusedNF4MatMulGELU performs parallel fused NF4 + bias + GELU for large matrices.
var ParallelFusedNF4MatMulGELU func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int)

// ParallelFusedInt4MatMulSiLU performs parallel fused Int4 + bias + SiLU for large matrices.
var ParallelFusedInt4MatMulSiLU func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int)

// ParallelFusedInt4MatMulGELU performs parallel fused Int4 + bias + GELU for large matrices.
var ParallelFusedInt4MatMulGELU func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int)

func init() {
	// Default Act implementations dispatch through individual activation-specific
	// function variables. These are set by hwygen-generated init() functions to the
	// best available arch-specific implementations (AVX2, NEON, fallback).
	// SME-enabled platforms override the Act variants directly in z_matmul_arm64.go init().
	FusedNF4MatMulAct = func(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType) {
		switch act {
		case ActSiLU:
			FusedNF4MatMulSiLU(input, packed, scales, bias, output, M, K, N, groupSize)
		case ActGELU:
			FusedNF4MatMulGELU(input, packed, scales, bias, output, M, K, N, groupSize)
		case ActGELUApprox:
			FusedNF4MatMulGELUApprox(input, packed, scales, bias, output, M, K, N, groupSize)
		case ActReLU:
			FusedNF4MatMulReLU(input, packed, scales, bias, output, M, K, N, groupSize)
		default:
			FusedNF4MatMul(input, packed, scales, bias, output, M, K, N, groupSize)
		}
	}
	FusedInt4MatMulAct = func(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType) {
		switch act {
		case ActSiLU:
			FusedInt4MatMulSiLU(input, packed, scales, bias, output, M, K, N, groupSize)
		case ActGELU:
			FusedInt4MatMulGELU(input, packed, scales, bias, output, M, K, N, groupSize)
		case ActGELUApprox:
			FusedInt4MatMulGELUApprox(input, packed, scales, bias, output, M, K, N, groupSize)
		case ActReLU:
			FusedInt4MatMulReLU(input, packed, scales, bias, output, M, K, N, groupSize)
		default:
			FusedInt4MatMul(input, packed, scales, bias, output, M, K, N, groupSize)
		}
	}
	FusedInt8MatMulAct = func(input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType) {
		switch act {
		case ActSiLU:
			FusedInt8MatMulSiLU(input, weights, scales, bias, output, M, K, N, groupSize)
		case ActGELU:
			FusedInt8MatMulGELU(input, weights, scales, bias, output, M, K, N, groupSize)
		case ActGELUApprox:
			FusedInt8MatMulGELUApprox(input, weights, scales, bias, output, M, K, N, groupSize)
		case ActReLU:
			FusedInt8MatMulReLU(input, weights, scales, bias, output, M, K, N, groupSize)
		default:
			FusedInt8MatMul(input, weights, scales, bias, output, M, K, N, groupSize)
		}
	}

	// Default parallel implementations use pool.ParallelFor across M rows.
	// SME platforms override these in z_matmul_arm64.go init() with N-tile parallelism.
	ParallelFusedNF4MatMulAct = func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType) {
		pool.ParallelFor(M, func(mStart, mEnd int) {
			rows := mEnd - mStart
			FusedNF4MatMulAct(input[mStart*K:mEnd*K], packed, scales, bias, output[mStart*N:mEnd*N], rows, K, N, groupSize, act)
		})
	}
	ParallelFusedInt4MatMulAct = func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType) {
		pool.ParallelFor(M, func(mStart, mEnd int) {
			rows := mEnd - mStart
			FusedInt4MatMulAct(input[mStart*K:mEnd*K], packed, scales, bias, output[mStart*N:mEnd*N], rows, K, N, groupSize, act)
		})
	}
	ParallelFusedInt8MatMulAct = func(pool workerpool.Executor, input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType) {
		pool.ParallelFor(M, func(mStart, mEnd int) {
			rows := mEnd - mStart
			FusedInt8MatMulAct(input[mStart*K:mEnd*K], weights, scales, bias, output[mStart*N:mEnd*N], rows, K, N, groupSize, act)
		})
	}

	// Default non-activation parallel implementations use pool.ParallelFor across M rows.
	ParallelFusedNF4MatMul = func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
		pool.ParallelFor(M, func(mStart, mEnd int) {
			rows := mEnd - mStart
			FusedNF4MatMul(input[mStart*K:mEnd*K], packed, scales, bias, output[mStart*N:mEnd*N], rows, K, N, groupSize)
		})
	}
	ParallelFusedInt4MatMul = func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
		pool.ParallelFor(M, func(mStart, mEnd int) {
			rows := mEnd - mStart
			FusedInt4MatMul(input[mStart*K:mEnd*K], packed, scales, bias, output[mStart*N:mEnd*N], rows, K, N, groupSize)
		})
	}
	ParallelFusedInt8MatMul = func(pool workerpool.Executor, input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
		pool.ParallelFor(M, func(mStart, mEnd int) {
			rows := mEnd - mStart
			FusedInt8MatMul(input[mStart*K:mEnd*K], weights, scales, bias, output[mStart*N:mEnd*N], rows, K, N, groupSize)
		})
	}

	// Convenience wrappers route through Act variants.
	// On SME, ParallelFused*MatMulAct is overridden, so these automatically get SME.
	ParallelFusedNF4MatMulSiLU = func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
		ParallelFusedNF4MatMulAct(pool, input, packed, scales, bias, output, M, K, N, groupSize, ActSiLU)
	}
	ParallelFusedNF4MatMulGELU = func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
		ParallelFusedNF4MatMulAct(pool, input, packed, scales, bias, output, M, K, N, groupSize, ActGELU)
	}
	ParallelFusedInt4MatMulSiLU = func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
		ParallelFusedInt4MatMulAct(pool, input, packed, scales, bias, output, M, K, N, groupSize, ActSiLU)
	}
	ParallelFusedInt4MatMulGELU = func(pool workerpool.Executor, input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
		ParallelFusedInt4MatMulAct(pool, input, packed, scales, bias, output, M, K, N, groupSize, ActGELU)
	}
}
