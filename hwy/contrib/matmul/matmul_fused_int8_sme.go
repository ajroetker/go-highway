//go:build !noasm && darwin && arm64

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

// NOTE: This file is named "matmul_fused_int8_sme.go" (starting with 'm')
// to ensure its init() runs AFTER "fusedint8matmul_arm64.gen.go" (starting with 'f').
// Go executes init() functions in lexicographic filename order within a package.
// The generated dispatch sets FusedInt8MatMul to the base implementation; this file's init()
// must run afterward to override with the SME implementation when available.

package matmul

import (
	"runtime"
	"sync"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// Int8 tile buffer pool to reduce allocations
var fusedInt8TilePool = sync.Pool{
	New: func() any {
		// Max tile size: K * 16 floats for SME tile width (K up to 4096)
		return make([]float32, 0, 4096*16)
	},
}

// fusedInt8MatMulSME performs fused Int8 dequantization + matrix multiplication using SME.
// This is optimized for Apple M4 SME, dequantizing tiles on-the-fly.
//
// Memory usage: O(K * 16) for tile buffer instead of O(K * N) for full dequant
func fusedInt8MatMulSME(
	input []float32,
	weights []int8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if !hwy.HasSME() {
		// Fall back to base implementation
		BaseFusedInt8MatMul_fallback(input, weights, scales, output, M, K, N, groupSize)
		return
	}

	// Check alignment for SME (16x16 tiles)
	if K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		BaseFusedInt8MatMul_fallback(input, weights, scales, output, M, K, N, groupSize)
		return
	}

	// SMEGuard pins goroutine to OS thread and blocks SIGURG to prevent
	// Go's async preemption from corrupting ZA register state during SME ops.
	defer hwy.SMEGuard()()

	numGroups := (N + groupSize - 1) / groupSize

	// Get tile buffer from pool
	tileBuf := fusedInt8TilePool.Get().([]float32)
	tileSize := K * 16
	if cap(tileBuf) < tileSize {
		tileBuf = make([]float32, tileSize)
	} else {
		tileBuf = tileBuf[:tileSize]
	}
	defer fusedInt8TilePool.Put(tileBuf)

	// Transpose buffer for input (needed for FMOPA)
	inputT := transposePool32.Get().([]float32)
	inputTSize := M * K
	if cap(inputT) < inputTSize {
		inputT = make([]float32, inputTSize)
	} else {
		inputT = inputT[:inputTSize]
	}
	defer transposePool32.Put(inputT)

	// Transpose input: [M, K] -> [K, M]
	transposeMatrix(input, M, K, inputT)

	// Output tile buffer from pool
	outputTileSize := M * 16
	outputTile := fusedOutputTilePool.Get().([]float32)
	if cap(outputTile) < outputTileSize {
		outputTile = make([]float32, outputTileSize)
	} else {
		outputTile = outputTile[:outputTileSize]
	}
	defer fusedOutputTilePool.Put(outputTile)

	// Process N in 16-column tiles
	for nTile := 0; nTile < N; nTile += 16 {
		nEnd := min(nTile+16, N)
		tileN := nEnd - nTile

		// Dequantize Int8 weight tile: [K, 16] from weights [K, N]
		dequantizeInt8Tile(weights, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

		// SME matmul: inputT[K, M]^T @ tileBuf[K, tileN] = input[M, K] @ tile[K, tileN]
		asm.MultiTileMatMulFMOPAF32(inputT, tileBuf[:K*tileN], outputTile[:M*tileN], M, tileN, K)

		// Copy output tile to final output (scatter to correct columns)
		for m := 0; m < M; m++ {
			for j := 0; j < tileN; j++ {
				output[m*N+nTile+j] = outputTile[m*tileN+j]
			}
		}
	}
}

// dequantizeInt8Tile dequantizes a K×tileN tile of Int8 weights.
// Output is row-major: tile[k*tileN + j] = weight[k, nTile+j] * scale
func dequantizeInt8Tile(
	weights []int8,
	scales []float32,
	tile []float32,
	nTile, K, N, tileN, numGroups, groupSize int,
) {
	for k := 0; k < K; k++ {
		for j := 0; j < tileN; j++ {
			n := nTile + j
			weightIdx := k*N + n
			val := float32(weights[weightIdx])

			groupIdx := n / groupSize
			scale := scales[k*numGroups+groupIdx]
			tile[k*tileN+j] = val * scale
		}
	}
}

// processFusedInt8Tile processes a single N-tile for Int8 matmul.
// inputT is the transposed input [K, M], weights is Int8 [K, N], output is [M, N].
func processFusedInt8Tile(
	inputT []float32,
	weights []int8,
	scales []float32,
	output []float32,
	tileBuf []float32,
	outputTile []float32,
	nTile, M, K, N, numGroups, groupSize int,
) {
	nEnd := min(nTile+16, N)
	tileN := nEnd - nTile

	// Dequantize weight tile: [K, tileN]
	dequantizeInt8Tile(weights, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

	// SME matmul: inputT[K, M]^T @ tileBuf[K, tileN] = input[M, K] @ tile[K, tileN]
	asm.MultiTileMatMulFMOPAF32(inputT, tileBuf[:K*tileN], outputTile[:M*tileN], M, tileN, K)

	// Copy output tile to final output (scatter to correct columns)
	for m := 0; m < M; m++ {
		for j := 0; j < tileN; j++ {
			output[m*N+nTile+j] = outputTile[m*tileN+j]
		}
	}
}

// parallelFusedInt8MatMulSME performs fused Int8 matmul with parallel N-tile processing.
// Shares the transposed input across workers; each worker processes independent tiles.
func parallelFusedInt8MatMulSME(
	input []float32,
	weights []int8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if !hwy.HasSME() {
		BaseFusedInt8MatMul_fallback(input, weights, scales, output, M, K, N, groupSize)
		return
	}

	// Check alignment for SME (16x16 tiles)
	if K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		BaseFusedInt8MatMul_fallback(input, weights, scales, output, M, K, N, groupSize)
		return
	}

	numTiles := (N + 15) / 16
	numGroups := (N + groupSize - 1) / groupSize

	// Fall back to sequential if too few tiles
	if numTiles < MinFusedParallelTiles {
		fusedInt8MatMulSME(input, weights, scales, output, M, K, N, groupSize)
		return
	}

	// Transpose input once (shared across workers, read-only)
	inputT := transposePool32.Get().([]float32)
	inputTSize := M * K
	if cap(inputT) < inputTSize {
		inputT = make([]float32, inputTSize)
	} else {
		inputT = inputT[:inputTSize]
	}
	transposeMatrix(input, M, K, inputT)
	defer transposePool32.Put(inputT)

	// Setup work queue of N-tile indices
	work := make(chan int, numTiles)
	for nTile := 0; nTile < N; nTile += 16 {
		work <- nTile
	}
	close(work)

	// Launch workers
	numWorkers := min(runtime.GOMAXPROCS(0), numTiles)
	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			// SMEGuard pins goroutine to OS thread and blocks SIGURG to prevent
			// Go's async preemption from corrupting ZA register state during SME ops.
			defer hwy.SMEGuard()()

			// Get thread-local buffers from pool
			tileBuf := fusedInt8TilePool.Get().([]float32)
			tileSize := K * 16
			if cap(tileBuf) < tileSize {
				tileBuf = make([]float32, tileSize)
			} else {
				tileBuf = tileBuf[:tileSize]
			}
			defer fusedInt8TilePool.Put(tileBuf)

			outputTile := fusedOutputTilePool.Get().([]float32)
			outputTileSize := M * 16
			if cap(outputTile) < outputTileSize {
				outputTile = make([]float32, outputTileSize)
			} else {
				outputTile = outputTile[:outputTileSize]
			}
			defer fusedOutputTilePool.Put(outputTile)

			for nTile := range work {
				processFusedInt8Tile(inputT, weights, scales, output, tileBuf, outputTile,
					nTile, M, K, N, numGroups, groupSize)
			}
		})
	}
	wg.Wait()
}

func init() {
	if hwy.HasSME() {
		// Override dispatch with SME-optimized implementations
		FusedInt8MatMul = fusedInt8MatMulSME
		ParallelFusedInt8MatMul = parallelFusedInt8MatMulSME
	}
}
