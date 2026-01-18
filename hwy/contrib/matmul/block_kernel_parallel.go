// Copyright 2024 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"runtime"
	"sync"

	"github.com/ajroetker/go-highway/hwy"
)

// Parallel BlockMulAdd tuning parameters
const (
	// MinBlocksForParallel is the minimum number of blocks before parallelizing
	MinBlocksForParallel = 4
)

// ParallelBlockMulAdd computes C += A^T * B for multiple independent blocks in parallel.
// Each block is blockDim × blockDim. The blocks are processed concurrently using goroutines.
//
// Parameters:
//   - aTs: slice of pre-transposed A blocks (each blockDim × blockDim)
//   - bs: slice of B blocks (each blockDim × blockDim)
//   - cs: slice of C blocks to accumulate into (each blockDim × blockDim)
//   - blockDim: dimension of each square block
//
// All slices must have the same length (number of blocks to process).
// Uses the best available SIMD implementation (FMOPA on SME, NEON otherwise).
func ParallelBlockMulAdd[T hwy.Floats](aTs, bs, cs [][]T, blockDim int) {
	numBlocks := len(aTs)
	if numBlocks == 0 {
		return
	}
	if len(bs) != numBlocks || len(cs) != numBlocks {
		panic("ParallelBlockMulAdd: mismatched slice lengths")
	}

	// For small number of blocks, process sequentially
	if numBlocks < MinBlocksForParallel {
		for i := range numBlocks {
			BlockMulAdd(aTs[i], bs[i], cs[i], blockDim)
		}
		return
	}

	numWorkers := min(runtime.GOMAXPROCS(0), numBlocks)

	// Work queue of block indices
	work := make(chan int, numBlocks)
	for i := range numBlocks {
		work <- i
	}
	close(work)

	// Workers process blocks in parallel
	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			for idx := range work {
				BlockMulAdd(aTs[idx], bs[idx], cs[idx], blockDim)
			}
		})
	}
	wg.Wait()
}

// ParallelBlockMulAddFloat32 is the non-generic version for float32.
func ParallelBlockMulAddFloat32(aTs, bs, cs [][]float32, blockDim int) {
	ParallelBlockMulAdd(aTs, bs, cs, blockDim)
}

// ParallelBlockMulAddFloat64 is the non-generic version for float64.
func ParallelBlockMulAddFloat64(aTs, bs, cs [][]float64, blockDim int) {
	ParallelBlockMulAdd(aTs, bs, cs, blockDim)
}
