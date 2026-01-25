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
