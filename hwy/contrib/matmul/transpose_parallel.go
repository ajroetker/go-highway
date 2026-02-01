// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
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

// ParallelTranspose2D transposes an M×K row-major matrix to K×M using a persistent worker pool.
// Divides work into horizontal strips processed concurrently.
//
// For large matrices, this is faster than serial SIMD transpose because transpose
// is memory-bandwidth bound and parallelism helps saturate memory bandwidth.
func ParallelTranspose2D[T hwy.Floats](pool *workerpool.Pool, src []T, m, k int, dst []T) {
	if len(src) < m*k || len(dst) < k*m {
		return
	}

	if m*k < MinTransposeParallelOps {
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
func TransposeAuto[T hwy.Floats](pool *workerpool.Pool, src []T, m, k int, dst []T) {
	if m*k < MinTransposeParallelOps {
		Transpose2D(src, m, k, dst)
	} else {
		ParallelTranspose2D(pool, src, m, k, dst)
	}
}

// ParallelTranspose2DFloat32 is the non-generic version for float32.
func ParallelTranspose2DFloat32(pool *workerpool.Pool, src []float32, m, k int, dst []float32) {
	ParallelTranspose2D(pool, src, m, k, dst)
}

// ParallelTranspose2DFloat64 is the non-generic version for float64.
func ParallelTranspose2DFloat64(pool *workerpool.Pool, src []float64, m, k int, dst []float64) {
	ParallelTranspose2D(pool, src, m, k, dst)
}

// TransposeAutoFloat32 is the non-generic version for float32.
func TransposeAutoFloat32(pool *workerpool.Pool, src []float32, m, k int, dst []float32) {
	TransposeAuto(pool, src, m, k, dst)
}

// TransposeAutoFloat64 is the non-generic version for float64.
func TransposeAutoFloat64(pool *workerpool.Pool, src []float64, m, k int, dst []float64) {
	TransposeAuto(pool, src, m, k, dst)
}
