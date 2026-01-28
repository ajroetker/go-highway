// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"runtime"
	"sync"

	"github.com/ajroetker/go-highway/hwy"
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

	// Calculate number of row strips
	numStrips := (m + RowsPerStrip - 1) / RowsPerStrip

	// Work queue of row strips
	work := make(chan int, numStrips)
	for strip := range numStrips {
		work <- strip
	}
	close(work)

	// Workers grab strips from queue and use MatMulKLastBlocked for each
	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			for strip := range work {
				rowStart := strip * RowsPerStrip
				rowEnd := min(rowStart+RowsPerStrip, m)
				stripM := rowEnd - rowStart

				// Get slices for this strip
				// A: rows [rowStart:rowEnd] with K columns each
				aStrip := a[rowStart*k : rowEnd*k]
				// C: rows [rowStart:rowEnd] with N columns each
				cStrip := c[rowStart*n : rowEnd*n]

				// B is shared across all strips (N x K)
				// Use optimized blocked K-last matmul for this strip
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

	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers > m {
		numWorkers = m
	}

	work := make(chan int, m)
	for row := range m {
		work <- row
	}
	close(work)

	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			for row := range work {
				aRow := a[row*k : (row+1)*k]
				cRow := c[row*n : (row+1)*n]
				MatMulKLastBlocked(aRow, b, cRow, 1, n, k)
			}
		})
	}
	wg.Wait()
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
