// Copyright 2024 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"runtime"
	"sync"
	"sync/atomic"

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

// ParallelMatMul computes C = A * B using parallel execution.
// Divides work into horizontal strips and uses the optimized BlockedMatMul for each strip.
//
//   - A is M x K (row-major)
//   - B is K x N (row-major)
//   - C is M x N (row-major)
func ParallelMatMul[T hwy.Floats](a, b, c []T, m, n, k int) {
	// For small matrices, use single-threaded version
	if m*n*k < MinParallelOps {
		BlockedMatMul(a, b, c, m, n, k)
		return
	}

	numWorkers := runtime.GOMAXPROCS(0)
	numStrips := (m + RowsPerStrip - 1) / RowsPerStrip
	if numWorkers > numStrips {
		numWorkers = numStrips
	}

	var nextStrip atomic.Int32
	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			for {
				strip := int(nextStrip.Add(1)) - 1
				if strip >= numStrips {
					return
				}

				rowStart := strip * RowsPerStrip
				rowEnd := min(rowStart+RowsPerStrip, m)
				stripM := rowEnd - rowStart

				aStrip := a[rowStart*k : rowEnd*k]
				cStrip := c[rowStart*n : rowEnd*n]

				BlockedMatMul(aStrip, b, cStrip, stripM, n, k)
			}
		})
	}
	wg.Wait()
}

// ParallelMatMulFineGrained computes C = A * B using fine-grained parallelism.
// Uses 1-row strips to maximize parallelism when M is small.
// This is critical for cases like M=11, N=1024, K=1024 where RowsPerStrip=64
// would result in only 1 strip (no parallelism).
//
// Benchmarks on M4 Max show 4.3x speedup for M=11, N=1024, K=1024.
func ParallelMatMulFineGrained[T hwy.Floats](a, b, c []T, m, n, k int) {
	// For very small matrices, single-threaded is faster
	if m*n*k < MinParallelOps {
		BlockedMatMul(a, b, c, m, n, k)
		return
	}

	numWorkers := min(runtime.GOMAXPROCS(0), m)

	var nextRow atomic.Int32
	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			for {
				row := int(nextRow.Add(1)) - 1
				if row >= m {
					return
				}
				aRow := a[row*k : (row+1)*k]
				cRow := c[row*n : (row+1)*n]
				BlockedMatMul(aRow, b, cRow, 1, n, k)
			}
		})
	}
	wg.Wait()
}

// ParallelMatMulWithPool computes C = A * B using a persistent worker pool.
// This avoids per-call goroutine spawn overhead, critical for transformer
// inference with ~50+ matmul ops per forward pass.
//
//   - A is M x K (row-major)
//   - B is K x N (row-major)
//   - C is M x N (row-major)
func ParallelMatMulWithPool[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, m, n, k int) {
	if m*n*k < MinParallelOps || pool == nil {
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

// ParallelMatMulFineGrainedWithPool computes C = A * B using fine-grained parallelism
// with a persistent worker pool. Uses atomic work stealing for load balancing.
func ParallelMatMulFineGrainedWithPool[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, m, n, k int) {
	if m*n*k < MinParallelOps || pool == nil {
		BlockedMatMul(a, b, c, m, n, k)
		return
	}

	pool.ParallelForAtomic(m, func(row int) {
		aRow := a[row*k : (row+1)*k]
		cRow := c[row*n : (row+1)*n]
		BlockedMatMul(aRow, b, cRow, 1, n, k)
	})
}

// ParallelMatMulFineGrainedFloat32 is the non-generic version for float32.
func ParallelMatMulFineGrainedFloat32(a, b, c []float32, m, n, k int) {
	ParallelMatMulFineGrained(a, b, c, m, n, k)
}

// ParallelMatMulFineGrainedFloat64 is the non-generic version for float64.
func ParallelMatMulFineGrainedFloat64(a, b, c []float64, m, n, k int) {
	ParallelMatMulFineGrained(a, b, c, m, n, k)
}

// ParallelMatMulFloat32 is the non-generic version for float32.
func ParallelMatMulFloat32(a, b, c []float32, m, n, k int) {
	ParallelMatMul(a, b, c, m, n, k)
}

// ParallelMatMulFloat64 is the non-generic version for float64.
func ParallelMatMulFloat64(a, b, c []float64, m, n, k int) {
	ParallelMatMul(a, b, c, m, n, k)
}

// ParallelMatMulWithPoolFloat32 is the non-generic version for float32.
func ParallelMatMulWithPoolFloat32(pool *workerpool.Pool, a, b, c []float32, m, n, k int) {
	ParallelMatMulWithPool(pool, a, b, c, m, n, k)
}

// ParallelMatMulWithPoolFloat64 is the non-generic version for float64.
func ParallelMatMulWithPoolFloat64(pool *workerpool.Pool, a, b, c []float64, m, n, k int) {
	ParallelMatMulWithPool(pool, a, b, c, m, n, k)
}

// ParallelMatMulFineGrainedWithPoolFloat32 is the non-generic version for float32.
func ParallelMatMulFineGrainedWithPoolFloat32(pool *workerpool.Pool, a, b, c []float32, m, n, k int) {
	ParallelMatMulFineGrainedWithPool(pool, a, b, c, m, n, k)
}

// ParallelMatMulFineGrainedWithPoolFloat64 is the non-generic version for float64.
func ParallelMatMulFineGrainedWithPoolFloat64(pool *workerpool.Pool, a, b, c []float64, m, n, k int) {
	ParallelMatMulFineGrainedWithPool(pool, a, b, c, m, n, k)
}
