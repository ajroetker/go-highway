// Copyright 2024 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"runtime"
	"sync"

	"github.com/ajroetker/go-highway/hwy"
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

	// Calculate number of row strips
	numStrips := (m + RowsPerStrip - 1) / RowsPerStrip

	// Work queue of row strips
	work := make(chan int, numStrips)
	for strip := 0; strip < numStrips; strip++ {
		work <- strip
	}
	close(work)

	// Workers grab strips from queue and use BlockedMatMul for each
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for strip := range work {
				rowStart := strip * RowsPerStrip
				rowEnd := min(rowStart+RowsPerStrip, m)
				stripM := rowEnd - rowStart

				// Get slices for this strip
				aStrip := a[rowStart*k : rowEnd*k]
				cStrip := c[rowStart*n : rowEnd*n]

				// Use optimized blocked matmul for this strip
				BlockedMatMul(aStrip, b, cStrip, stripM, n, k)
			}
		}()
	}
	wg.Wait()
}

// ParallelMatMulFloat32 is the non-generic version for float32.
func ParallelMatMulFloat32(a, b, c []float32, m, n, k int) {
	ParallelMatMul(a, b, c, m, n, k)
}

// ParallelMatMulFloat64 is the non-generic version for float64.
func ParallelMatMulFloat64(a, b, c []float64, m, n, k int) {
	ParallelMatMul(a, b, c, m, n, k)
}
