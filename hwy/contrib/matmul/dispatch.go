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
)

// MatMulAuto automatically selects the best algorithm based on matrix dimensions.
//
// Algorithm selection based on matrix size (total ops = M * N * K):
//   - Small (<64^3): Streaming MatMul - lowest overhead, fits in cache
//   - Medium/Large: Parallel BlockedMatMul - cache tiling + parallelism
//
// On AMD64, large matrices (>=256^3) use ParallelPackedMatMulV2 with K-blocking.
// The V2 algorithm uses:
//   - WorkersPool with intelligent work distribution (batch-first, then LHS/RHS split)
//   - Small packed output buffer (Mc=4) for better cache locality
//   - 4x K-loop unrolling with BCE hints for better ILP
//   - SIMD-optimized output application
// Benchmarks show V2 is 11-27% faster than V1 on AMD64.
//
// On ARM64, SME FMOPA outer product instructions are 100-1000x faster than
// any pure-Go SIMD implementation, so we use ParallelMatMul which leverages
// the hardware-accelerated blocked implementation.
func MatMulAuto[T hwy.Floats](a, b, c []T, m, n, k int) {
	totalOps := m * n * k

	// For small M with large N*K, we need row-parallel with 1-row strips
	// to achieve parallelism. Standard RowsPerStrip=64 would give only 1 strip
	// for M<64, meaning no parallelism.
	//
	// Benchmarks on M4 Max show 4.3x speedup for M=11, N=1024, K=1024:
	//   - Streaming single-threaded: 2.78ms
	//   - Row-parallel 1-row strips: 0.65ms
	const SmallMThreshold = 64 // Use fine-grained parallelism when M < RowsPerStrip
	if m < SmallMThreshold && totalOps >= SmallMatrixThreshold {
		ParallelMatMulFineGrained(a, b, c, m, n, k)
		return
	}

	if totalOps < SmallMatrixThreshold {
		// Small matrices: streaming is faster (no blocking overhead)
		MatMul(a, b, c, m, n, k)
	} else if totalOps < LargeMatrixThreshold {
		// Medium matrices: parallel blocked
		ParallelMatMul(a, b, c, m, n, k)
	} else {
		// Large matrices: architecture-dependent
		if runtime.GOARCH == "arm64" {
			// Use SME FMOPA / NEON blocked - hardware outer product is 100-1000x faster
			ParallelMatMul(a, b, c, m, n, k)
		} else {
			// Use optimized V2 packed GEBP with K-blocking on AMD64
			// V2 uses smaller panels (Mc=4, Nc=512) for better cache locality
			ParallelPackedMatMulV2(a, b, c, m, n, k)
		}
	}
}

// MatMulAutoFloat32 is the non-generic version for float32.
func MatMulAutoFloat32(a, b, c []float32, m, n, k int) {
	MatMulAuto(a, b, c, m, n, k)
}

// MatMulAutoFloat64 is the non-generic version for float64.
func MatMulAutoFloat64(a, b, c []float64, m, n, k int) {
	MatMulAuto(a, b, c, m, n, k)
}

// MatMulAutoWithPool is like MatMulAuto but uses a persistent worker pool.
// This avoids per-call goroutine spawn overhead, critical for transformer
// inference with ~50+ matmul ops per forward pass.
//
// Usage:
//
//	pool := workerpool.New(runtime.GOMAXPROCS(0))
//	defer pool.Close()
//
//	for _, layer := range layers {
//	    matmul.MatMulAutoWithPool(pool, a, b, c, m, n, k)
//	}
func MatMulAutoWithPool[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, m, n, k int) {
	if pool == nil {
		MatMulAuto(a, b, c, m, n, k)
		return
	}

	totalOps := m * n * k

	const SmallMThreshold = 64
	if m < SmallMThreshold && totalOps >= SmallMatrixThreshold {
		ParallelMatMulFineGrainedWithPool(pool, a, b, c, m, n, k)
		return
	}

	if totalOps < SmallMatrixThreshold {
		MatMul(a, b, c, m, n, k)
	} else if totalOps < LargeMatrixThreshold {
		ParallelMatMulWithPool(pool, a, b, c, m, n, k)
	} else {
		if runtime.GOARCH == "arm64" {
			ParallelMatMulWithPool(pool, a, b, c, m, n, k)
		} else {
			// V2 has its own parallelism, fall back to non-pool version
			ParallelPackedMatMulV2(a, b, c, m, n, k)
		}
	}
}

// MatMulAutoWithPoolFloat32 is the non-generic version for float32.
func MatMulAutoWithPoolFloat32(pool *workerpool.Pool, a, b, c []float32, m, n, k int) {
	MatMulAutoWithPool(pool, a, b, c, m, n, k)
}

// MatMulAutoWithPoolFloat64 is the non-generic version for float64.
func MatMulAutoWithPoolFloat64(pool *workerpool.Pool, a, b, c []float64, m, n, k int) {
	MatMulAutoWithPool(pool, a, b, c, m, n, k)
}

// MatMulKLastAuto automatically selects the best algorithm for K-last layout.
//
// K-last layout: A is [M,K], B is [N,K] (both with K as last dimension).
// Computes C = A @ B^T where C is [M,N].
//
// Algorithm selection based on matrix size (total ops = M * N * K):
//   - Small (<64^3): Streaming MatMulKLast - lowest overhead
//   - Medium/Large: ParallelMatMulKLast - parallel row striping + blocked
//
// ParallelMatMulKLast enables intra-example parallelism: a single large matrix
// multiplication can utilize all CPU cores by processing independent row strips
// concurrently. This is critical for patterns like multi-cross (bsi,oi->bso)
// where batchSize=1 but M,N,K are large.
//
// On ARM64 with SME, the dispatch already uses FMOPA with transpose
// for sizes >= 32 (when 16-aligned), which is 2-4x faster than NEON.
func MatMulKLastAuto[T hwy.Floats](a, b, c []T, m, n, k int) {
	totalOps := m * n * k

	// For small M with large N*K, use fine-grained row parallelism.
	const SmallMThreshold = 64
	if m < SmallMThreshold && totalOps >= SmallMatrixThreshold {
		ParallelMatMulKLastFineGrained(a, b, c, m, n, k)
		return
	}

	if totalOps < SmallMatrixThreshold {
		// Small matrices: streaming is faster (no blocking overhead)
		MatMulKLast(a, b, c, m, n, k)
	} else {
		// Medium/large matrices: parallel row striping + blocked
		ParallelMatMulKLast(a, b, c, m, n, k)
	}
}

// MatMulKLastAutoFloat32 is the non-generic version for float32.
func MatMulKLastAutoFloat32(a, b, c []float32, m, n, k int) {
	MatMulKLastAuto(a, b, c, m, n, k)
}

// MatMulKLastAutoFloat64 is the non-generic version for float64.
func MatMulKLastAutoFloat64(a, b, c []float64, m, n, k int) {
	MatMulKLastAuto(a, b, c, m, n, k)
}

// MatMulKLastAutoFloat16 is the non-generic version for Float16.
func MatMulKLastAutoFloat16(a, b, c []hwy.Float16, m, n, k int) {
	MatMulKLastAuto(a, b, c, m, n, k)
}

// MatMulKLastAutoBFloat16 is the non-generic version for BFloat16.
func MatMulKLastAutoBFloat16(a, b, c []hwy.BFloat16, m, n, k int) {
	MatMulKLastAuto(a, b, c, m, n, k)
}

// MatMulKLastAutoWithPool is like MatMulKLastAuto but uses a persistent worker pool.
// This avoids per-call goroutine spawn overhead, critical for transformer
// inference with ~50+ matmul ops per forward pass.
//
// K-last layout: A is [M,K], B is [N,K] (both with K as last dimension).
// Computes C = A @ B^T where C is [M,N].
func MatMulKLastAutoWithPool[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, m, n, k int) {
	if pool == nil {
		MatMulKLastAuto(a, b, c, m, n, k)
		return
	}

	totalOps := m * n * k

	const SmallMThreshold = 64
	if m < SmallMThreshold && totalOps >= SmallMatrixThreshold {
		ParallelMatMulKLastFineGrainedWithPool(pool, a, b, c, m, n, k)
		return
	}

	if totalOps < SmallMatrixThreshold {
		MatMulKLast(a, b, c, m, n, k)
	} else {
		ParallelMatMulKLastWithPool(pool, a, b, c, m, n, k)
	}
}

// MatMulKLastAutoWithPoolFloat32 is the non-generic version for float32.
func MatMulKLastAutoWithPoolFloat32(pool *workerpool.Pool, a, b, c []float32, m, n, k int) {
	MatMulKLastAutoWithPool(pool, a, b, c, m, n, k)
}

// MatMulKLastAutoWithPoolFloat64 is the non-generic version for float64.
func MatMulKLastAutoWithPoolFloat64(pool *workerpool.Pool, a, b, c []float64, m, n, k int) {
	MatMulKLastAutoWithPool(pool, a, b, c, m, n, k)
}

// MatMulKLastAutoWithPoolFloat16 is the non-generic version for Float16.
func MatMulKLastAutoWithPoolFloat16(pool *workerpool.Pool, a, b, c []hwy.Float16, m, n, k int) {
	MatMulKLastAutoWithPool(pool, a, b, c, m, n, k)
}

// MatMulKLastAutoWithPoolBFloat16 is the non-generic version for BFloat16.
func MatMulKLastAutoWithPoolBFloat16(pool *workerpool.Pool, a, b, c []hwy.BFloat16, m, n, k int) {
	MatMulKLastAutoWithPool(pool, a, b, c, m, n, k)
}

// =============================================================================
// Parallel Fused NF4/Int4 MatMul dispatch
// =============================================================================

// ParallelFusedNF4MatMul performs fused NF4 dequantization + matrix multiplication
// with parallel execution for large matrices.
// Dispatches to the best available implementation for the current platform.
// On platforms with SME, this uses tiled parallel execution.
// On other platforms, this falls back to the serial implementation.
var ParallelFusedNF4MatMul func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// ParallelFusedInt4MatMul performs fused Int4 dequantization + matrix multiplication
// with parallel execution for large matrices.
// Dispatches to the best available implementation for the current platform.
// On platforms with SME, this uses tiled parallel execution.
// On other platforms, this falls back to the serial implementation.
var ParallelFusedInt4MatMul func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

func init() {
	// Default parallel implementations just call the serial versions.
	// SME-enabled platforms override these in matmul_fused_nf4_sme.go init().
	ParallelFusedNF4MatMul = func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
		FusedNF4MatMul(input, packed, scales, output, M, K, N, groupSize)
	}
	ParallelFusedInt4MatMul = func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
		FusedInt4MatMul(input, packed, scales, output, M, K, N, groupSize)
	}
}
