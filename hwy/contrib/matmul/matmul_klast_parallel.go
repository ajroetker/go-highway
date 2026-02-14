// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// ParallelMatMulKLast computes C = A * B^T using a persistent worker pool.
// Divides work into horizontal strips and uses the optimized MatMulKLastBlocked for each strip.
//
//   - A is M x K (row-major, K last)
//   - B is N x K (row-major, K last - PyTorch weight format)
//   - C is M x N (row-major)
//
// This enables intra-example parallelism: a single large matrix multiplication
// can utilize all CPU cores by processing independent row strips concurrently.
func ParallelMatMulKLast[T hwy.Floats](pool workerpool.Executor, a, b, c []T, m, n, k int) {
	if m*n*k < MinParallelOps {
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

// ParallelMatMulKLastFineGrained computes C = A * B^T using fine-grained
// parallelism with a persistent worker pool. Uses atomic work stealing for load balancing.
func ParallelMatMulKLastFineGrained[T hwy.Floats](pool workerpool.Executor, a, b, c []T, m, n, k int) {
	if m*n*k < MinParallelOps {
		MatMulKLastBlocked(a, b, c, m, n, k)
		return
	}

	pool.ParallelForAtomic(m, func(row int) {
		aRow := a[row*k : (row+1)*k]
		cRow := c[row*n : (row+1)*n]
		MatMulKLastBlocked(aRow, b, cRow, 1, n, k)
	})
}

