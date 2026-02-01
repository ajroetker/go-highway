// Copyright 2024 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// Parallel tuning parameters
const (
	// MinParallelOps is the minimum number of operations before parallelizing
	MinParallelOps = 64 * 64 * 64

	// RowsPerStrip defines how many rows each worker processes at a time.
	// Tuned for good load balancing while keeping strips large enough for cache efficiency.
	RowsPerStrip = 64
)

// ParallelMatMul computes C = A * B using a persistent worker pool.
// Divides work into horizontal strips and uses the optimized BlockedMatMul for each strip.
//
//   - A is M x K (row-major)
//   - B is K x N (row-major)
//   - C is M x N (row-major)
func ParallelMatMul[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, m, n, k int) {
	if m*n*k < MinParallelOps {
		BlockedMatMul(a, b, c, m, n, k)
		return
	}

	numStrips := (m + RowsPerStrip - 1) / RowsPerStrip

	pool.ParallelFor(numStrips, func(start, end int) {
		for strip := start; strip < end; strip++ {
			rowStart := strip * RowsPerStrip
			rowEnd := min(rowStart+RowsPerStrip, m)
			stripM := rowEnd - rowStart

			aStrip := a[rowStart*k : rowEnd*k]
			cStrip := c[rowStart*n : rowEnd*n]

			BlockedMatMul(aStrip, b, cStrip, stripM, n, k)
		}
	})
}

// ParallelMatMulFineGrained computes C = A * B using fine-grained parallelism
// with a persistent worker pool. Uses atomic work stealing for load balancing.
// Uses 1-row strips to maximize parallelism when M is small.
// This is critical for cases like M=11, N=1024, K=1024 where RowsPerStrip=64
// would result in only 1 strip (no parallelism).
//
// Benchmarks on M4 Max show 4.3x speedup for M=11, N=1024, K=1024.
func ParallelMatMulFineGrained[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, m, n, k int) {
	if m*n*k < MinParallelOps {
		BlockedMatMul(a, b, c, m, n, k)
		return
	}

	pool.ParallelForAtomic(m, func(row int) {
		aRow := a[row*k : (row+1)*k]
		cRow := c[row*n : (row+1)*n]
		BlockedMatMul(aRow, b, cRow, 1, n, k)
	})
}

// ParallelMatMulFloat32 is the non-generic version for float32.
func ParallelMatMulFloat32(pool *workerpool.Pool, a, b, c []float32, m, n, k int) {
	ParallelMatMul(pool, a, b, c, m, n, k)
}

// ParallelMatMulFloat64 is the non-generic version for float64.
func ParallelMatMulFloat64(pool *workerpool.Pool, a, b, c []float64, m, n, k int) {
	ParallelMatMul(pool, a, b, c, m, n, k)
}

// ParallelMatMulFineGrainedFloat32 is the non-generic version for float32.
func ParallelMatMulFineGrainedFloat32(pool *workerpool.Pool, a, b, c []float32, m, n, k int) {
	ParallelMatMulFineGrained(pool, a, b, c, m, n, k)
}

// ParallelMatMulFineGrainedFloat64 is the non-generic version for float64.
func ParallelMatMulFineGrainedFloat64(pool *workerpool.Pool, a, b, c []float64, m, n, k int) {
	ParallelMatMulFineGrained(pool, a, b, c, m, n, k)
}
