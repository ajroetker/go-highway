// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// Transpose tuning parameters
const (
	// MinTransposeParallelOps is the minimum elements before parallelizing transpose
	MinTransposeParallelOps = 64 * 64

	// TransposeRowsPerStrip defines how many rows each worker processes
	TransposeRowsPerStrip = 64
)

// ParallelTranspose2D transposes an M×K row-major matrix to K×M using parallel execution.
// Divides work into horizontal strips processed concurrently.
//
// For large matrices, this is faster than serial SIMD transpose because transpose
// is memory-bandwidth bound and parallelism helps saturate memory bandwidth.
func ParallelTranspose2D[T hwy.Floats](src []T, m, k int, dst []T) {
	if len(src) < m*k || len(dst) < k*m {
		return
	}

	// For small matrices, use optimized SIMD version
	if m*k < MinTransposeParallelOps {
		Transpose2D(src, m, k, dst)
		return
	}

	numWorkers := runtime.GOMAXPROCS(0)
	numStrips := (m + TransposeRowsPerStrip - 1) / TransposeRowsPerStrip
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

				rowStart := strip * TransposeRowsPerStrip
				rowEnd := min(rowStart+TransposeRowsPerStrip, m)

				// Use strided SIMD transpose for this strip
				// dstM = m (full destination stride)
				Transpose2DStrided(src, rowStart, rowEnd, k, m, dst)
			}
		})
	}
	wg.Wait()
}

// ParallelTranspose2DWithPool transposes using a persistent worker pool.
func ParallelTranspose2DWithPool[T hwy.Floats](pool *workerpool.Pool, src []T, m, k int, dst []T) {
	if len(src) < m*k || len(dst) < k*m {
		return
	}

	if m*k < MinTransposeParallelOps || pool == nil {
		Transpose2D(src, m, k, dst)
		return
	}

	numStrips := (m + TransposeRowsPerStrip - 1) / TransposeRowsPerStrip

	pool.ParallelFor(numStrips, func(start, end int) {
		for strip := start; strip < end; strip++ {
			rowStart := strip * TransposeRowsPerStrip
			rowEnd := min(rowStart+TransposeRowsPerStrip, m)

			// Use strided SIMD transpose for this strip
			Transpose2DStrided(src, rowStart, rowEnd, k, m, dst)
		}
	})
}

// TransposeAuto automatically selects the best transpose algorithm.
func TransposeAuto[T hwy.Floats](src []T, m, k int, dst []T) {
	if m*k < MinTransposeParallelOps {
		Transpose2D(src, m, k, dst)
	} else {
		ParallelTranspose2D(src, m, k, dst)
	}
}

// TransposeAutoWithPool uses automatic selection with a persistent worker pool.
func TransposeAutoWithPool[T hwy.Floats](pool *workerpool.Pool, src []T, m, k int, dst []T) {
	if pool == nil {
		TransposeAuto(src, m, k, dst)
		return
	}

	if m*k < MinTransposeParallelOps {
		Transpose2D(src, m, k, dst)
	} else {
		ParallelTranspose2DWithPool(pool, src, m, k, dst)
	}
}

// ParallelTranspose2DFloat32 is the non-generic version for float32.
func ParallelTranspose2DFloat32(src []float32, m, k int, dst []float32) {
	ParallelTranspose2D(src, m, k, dst)
}

// ParallelTranspose2DFloat64 is the non-generic version for float64.
func ParallelTranspose2DFloat64(src []float64, m, k int, dst []float64) {
	ParallelTranspose2D(src, m, k, dst)
}

// ParallelTranspose2DWithPoolFloat32 is the non-generic version for float32.
func ParallelTranspose2DWithPoolFloat32(pool *workerpool.Pool, src []float32, m, k int, dst []float32) {
	ParallelTranspose2DWithPool(pool, src, m, k, dst)
}

// ParallelTranspose2DWithPoolFloat64 is the non-generic version for float64.
func ParallelTranspose2DWithPoolFloat64(pool *workerpool.Pool, src []float64, m, k int, dst []float64) {
	ParallelTranspose2DWithPool(pool, src, m, k, dst)
}

// TransposeAutoFloat32 is the non-generic version for float32.
func TransposeAutoFloat32(src []float32, m, k int, dst []float32) {
	TransposeAuto(src, m, k, dst)
}

// TransposeAutoFloat64 is the non-generic version for float64.
func TransposeAutoFloat64(src []float64, m, k int, dst []float64) {
	TransposeAuto(src, m, k, dst)
}

// TransposeAutoWithPoolFloat32 is the non-generic version for float32.
func TransposeAutoWithPoolFloat32(pool *workerpool.Pool, src []float32, m, k int, dst []float32) {
	TransposeAutoWithPool(pool, src, m, k, dst)
}

// TransposeAutoWithPoolFloat64 is the non-generic version for float64.
func TransposeAutoWithPoolFloat64(pool *workerpool.Pool, src []float64, m, k int, dst []float64) {
	TransposeAutoWithPool(pool, src, m, k, dst)
}
