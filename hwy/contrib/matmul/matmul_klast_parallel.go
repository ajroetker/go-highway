// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// ParallelMatMulKLast computes C = A * B^T using parallel execution.
// Divides work into horizontal strips and uses the optimized MatMulKLastBlocked for each strip.
//
//   - A is M x K (row-major, K last)
//   - B is N x K (row-major, K last - PyTorch weight format)
//   - C is M x N (row-major)
//
// This enables intra-example parallelism: a single large matrix multiplication
// can utilize all CPU cores by processing independent row strips concurrently.
func ParallelMatMulKLast[T hwy.Floats](a, b, c []T, m, n, k int) {
	// For small matrices, use single-threaded version
	if m*n*k < MinParallelOps {
		MatMulKLastBlocked(a, b, c, m, n, k)
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

				// A: rows [rowStart:rowEnd] with K columns each
				aStrip := a[rowStart*k : rowEnd*k]
				// C: rows [rowStart:rowEnd] with N columns each
				cStrip := c[rowStart*n : rowEnd*n]

				// B is shared across all strips (N x K)
				MatMulKLastBlocked(aStrip, b, cStrip, stripM, n, k)
			}
		})
	}
	wg.Wait()
}

// ParallelMatMulKLastFineGrained computes C = A * B^T using fine-grained parallelism.
// Uses 1-row strips to maximize parallelism when M is small.
func ParallelMatMulKLastFineGrained[T hwy.Floats](a, b, c []T, m, n, k int) {
	if m*n*k < MinParallelOps {
		MatMulKLastBlocked(a, b, c, m, n, k)
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
				MatMulKLastBlocked(aRow, b, cRow, 1, n, k)
			}
		})
	}
	wg.Wait()
}

// ParallelMatMulKLastWithPool computes C = A * B^T using a persistent worker pool.
// This avoids per-call goroutine spawn overhead, critical for transformer
// inference with ~50+ matmul ops per forward pass.
//
//   - A is M x K (row-major, K last)
//   - B is N x K (row-major, K last - PyTorch weight format)
//   - C is M x N (row-major)
func ParallelMatMulKLastWithPool[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, m, n, k int) {
	if m*n*k < MinParallelOps || pool == nil {
		MatMulKLastBlocked(a, b, c, m, n, k)
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

			MatMulKLastBlocked(aStrip, b, cStrip, stripM, n, k)
		}
	})
}

// ParallelMatMulKLastFineGrainedWithPool computes C = A * B^T using fine-grained
// parallelism with a persistent worker pool. Uses atomic work stealing for load balancing.
func ParallelMatMulKLastFineGrainedWithPool[T hwy.Floats](pool *workerpool.Pool, a, b, c []T, m, n, k int) {
	if m*n*k < MinParallelOps || pool == nil {
		MatMulKLastBlocked(a, b, c, m, n, k)
		return
	}

	pool.ParallelForAtomic(m, func(row int) {
		aRow := a[row*k : (row+1)*k]
		cRow := c[row*n : (row+1)*n]
		MatMulKLastBlocked(aRow, b, cRow, 1, n, k)
	})
}

// ParallelMatMulKLastFineGrainedFloat32 is the non-generic version for float32.
func ParallelMatMulKLastFineGrainedFloat32(a, b, c []float32, m, n, k int) {
	ParallelMatMulKLastFineGrained(a, b, c, m, n, k)
}

// ParallelMatMulKLastFineGrainedFloat64 is the non-generic version for float64.
func ParallelMatMulKLastFineGrainedFloat64(a, b, c []float64, m, n, k int) {
	ParallelMatMulKLastFineGrained(a, b, c, m, n, k)
}

// ParallelMatMulKLastFloat32 is the non-generic version for float32.
func ParallelMatMulKLastFloat32(a, b, c []float32, m, n, k int) {
	ParallelMatMulKLast(a, b, c, m, n, k)
}

// ParallelMatMulKLastFloat64 is the non-generic version for float64.
func ParallelMatMulKLastFloat64(a, b, c []float64, m, n, k int) {
	ParallelMatMulKLast(a, b, c, m, n, k)
}

// ParallelMatMulKLastFloat16 is the non-generic version for Float16.
func ParallelMatMulKLastFloat16(a, b, c []hwy.Float16, m, n, k int) {
	ParallelMatMulKLast(a, b, c, m, n, k)
}

// ParallelMatMulKLastBFloat16 is the non-generic version for BFloat16.
func ParallelMatMulKLastBFloat16(a, b, c []hwy.BFloat16, m, n, k int) {
	ParallelMatMulKLast(a, b, c, m, n, k)
}

// ParallelMatMulKLastWithPoolFloat32 is the non-generic version for float32.
func ParallelMatMulKLastWithPoolFloat32(pool *workerpool.Pool, a, b, c []float32, m, n, k int) {
	ParallelMatMulKLastWithPool(pool, a, b, c, m, n, k)
}

// ParallelMatMulKLastWithPoolFloat64 is the non-generic version for float64.
func ParallelMatMulKLastWithPoolFloat64(pool *workerpool.Pool, a, b, c []float64, m, n, k int) {
	ParallelMatMulKLastWithPool(pool, a, b, c, m, n, k)
}

// ParallelMatMulKLastWithPoolFloat16 is the non-generic version for Float16.
func ParallelMatMulKLastWithPoolFloat16(pool *workerpool.Pool, a, b, c []hwy.Float16, m, n, k int) {
	ParallelMatMulKLastWithPool(pool, a, b, c, m, n, k)
}

// ParallelMatMulKLastWithPoolBFloat16 is the non-generic version for BFloat16.
func ParallelMatMulKLastWithPoolBFloat16(pool *workerpool.Pool, a, b, c []hwy.BFloat16, m, n, k int) {
	ParallelMatMulKLastWithPool(pool, a, b, c, m, n, k)
}

// ParallelMatMulKLastFineGrainedWithPoolFloat32 is the non-generic version for float32.
func ParallelMatMulKLastFineGrainedWithPoolFloat32(pool *workerpool.Pool, a, b, c []float32, m, n, k int) {
	ParallelMatMulKLastFineGrainedWithPool(pool, a, b, c, m, n, k)
}

// ParallelMatMulKLastFineGrainedWithPoolFloat64 is the non-generic version for float64.
func ParallelMatMulKLastFineGrainedWithPoolFloat64(pool *workerpool.Pool, a, b, c []float64, m, n, k int) {
	ParallelMatMulKLastFineGrainedWithPool(pool, a, b, c, m, n, k)
}

// ParallelMatMulKLastFineGrainedWithPoolFloat16 is the non-generic version for Float16.
func ParallelMatMulKLastFineGrainedWithPoolFloat16(pool *workerpool.Pool, a, b, c []hwy.Float16, m, n, k int) {
	ParallelMatMulKLastFineGrainedWithPool(pool, a, b, c, m, n, k)
}

// ParallelMatMulKLastFineGrainedWithPoolBFloat16 is the non-generic version for BFloat16.
func ParallelMatMulKLastFineGrainedWithPoolBFloat16(pool *workerpool.Pool, a, b, c []hwy.BFloat16, m, n, k int) {
	ParallelMatMulKLastFineGrainedWithPool(pool, a, b, c, m, n, k)
}
