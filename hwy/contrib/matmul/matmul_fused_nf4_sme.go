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

// NOTE: This file is named "matmul_fused_nf4_sme.go" (starting with 'm')
// to ensure its init() runs AFTER "dispatch_fused_nf4_arm64.gen.go" (starting with 'd').
// Go executes init() functions in lexicographic filename order within a package.
// The generated dispatch sets FusedNF4MatMul/FusedInt4MatMul to fallback; this file's init()
// must run afterward to override with the SME implementation when available.

package matmul

import (
	"runtime"
	"sync"
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// Parallel tuning parameters for fused matmul
const (
	// MinFusedParallelTiles is the minimum number of N-tiles before parallelizing
	MinFusedParallelTiles = 4 // N >= 64
)

// Tile buffer pools to reduce allocations (SME-specific)
var fusedTilePool = sync.Pool{
	New: func() any {
		// Max tile size: K * 16 floats for SME tile width (K up to 4096)
		return make([]float32, 0, 4096*16)
	},
}

// Output tile pool: M * 16 floats for SME tile width (M up to 4096)
var fusedOutputTilePool = sync.Pool{
	New: func() any {
		return make([]float32, 0, 4096*16)
	},
}

// fusedNF4MatMulSME performs fused NF4 dequantization + matrix multiplication using SME.
// This is optimized for Apple M4 SME, dequantizing tiles on-the-fly.
//
// Memory usage: O(K * 16) for tile buffer instead of O(K * N) for full dequant
func fusedNF4MatMulSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if !hwy.HasSME() {
		// Fall back to scalar implementation
		BaseFusedNF4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	// Check alignment for SME (16x16 tiles)
	if K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		BaseFusedNF4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	numGroups := (N + groupSize - 1) / groupSize

	// Get tile buffer from pool
	tileBuf := fusedTilePool.Get().([]float32)
	tileSize := K * 16
	if cap(tileBuf) < tileSize {
		tileBuf = make([]float32, tileSize)
	} else {
		tileBuf = tileBuf[:tileSize]
	}
	defer fusedTilePool.Put(tileBuf)

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

		// Dequantize weight tile: [K, 16] from packed [K, N/2]
		dequantizeNF4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

		// SME matmul: inputT[K, M]^T @ tileBuf[K, tileN] = input[M, K] @ tile[K, tileN]
		// Use the optimized FMOPA kernel
		matmul_fmopa_at_f32(
			unsafe.Pointer(unsafe.SliceData(inputT)),
			unsafe.Pointer(unsafe.SliceData(tileBuf[:K*tileN])),
			unsafe.Pointer(unsafe.SliceData(outputTile[:M*tileN])),
			int64(M),
			int64(tileN),
			int64(K),
		)

		// Copy output tile to final output (scatter to correct columns)
		for m := 0; m < M; m++ {
			for j := 0; j < tileN; j++ {
				output[m*N+nTile+j] = outputTile[m*tileN+j]
			}
		}
	}
}

// dequantizeNF4Tile dequantizes a K×tileN tile of NF4 weights.
// Output is row-major: tile[k*tileN + j] = weight[k, nTile+j]
func dequantizeNF4Tile(
	packed []uint8,
	scales []float32,
	tile []float32,
	nTile, K, N, tileN, numGroups, groupSize int,
) {
	for k := 0; k < K; k++ {
		for j := 0; j < tileN; j++ {
			n := nTile + j
			weightIdx := k*N + n
			packedIdx := weightIdx / 2

			var quantIdx int
			if weightIdx%2 == 0 {
				quantIdx = int(packed[packedIdx] & 0x0F)
			} else {
				quantIdx = int((packed[packedIdx] >> 4) & 0x0F)
			}

			groupIdx := n / groupSize
			scale := scales[k*numGroups+groupIdx]
			tile[k*tileN+j] = nf4LookupTable[quantIdx] * scale
		}
	}
}

// fusedInt4MatMulSME performs fused Int4 dequantization + matrix multiplication using SME.
// Similar to fusedNF4MatMulSME but for symmetric Int4 quantization.
func fusedInt4MatMulSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if !hwy.HasSME() || K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		BaseFusedInt4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	numGroups := (N + groupSize - 1) / groupSize

	tileBuf := fusedTilePool.Get().([]float32)
	tileSize := K * 16
	if cap(tileBuf) < tileSize {
		tileBuf = make([]float32, tileSize)
	} else {
		tileBuf = tileBuf[:tileSize]
	}
	defer fusedTilePool.Put(tileBuf)

	inputT := transposePool32.Get().([]float32)
	inputTSize := M * K
	if cap(inputT) < inputTSize {
		inputT = make([]float32, inputTSize)
	} else {
		inputT = inputT[:inputTSize]
	}
	defer transposePool32.Put(inputT)

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

	for nTile := 0; nTile < N; nTile += 16 {
		nEnd := min(nTile+16, N)
		tileN := nEnd - nTile

		dequantizeInt4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

		matmul_fmopa_at_f32(
			unsafe.Pointer(unsafe.SliceData(inputT)),
			unsafe.Pointer(unsafe.SliceData(tileBuf[:K*tileN])),
			unsafe.Pointer(unsafe.SliceData(outputTile[:M*tileN])),
			int64(M),
			int64(tileN),
			int64(K),
		)

		for m := 0; m < M; m++ {
			for j := 0; j < tileN; j++ {
				output[m*N+nTile+j] = outputTile[m*tileN+j]
			}
		}
	}
}

// dequantizeInt4Tile dequantizes a K×tileN tile of Int4 weights.
// Int4 uses symmetric quantization: values in [0,15] map to [-8,7].
func dequantizeInt4Tile(
	packed []uint8,
	scales []float32,
	tile []float32,
	nTile, K, N, tileN, numGroups, groupSize int,
) {
	for k := 0; k < K; k++ {
		for j := 0; j < tileN; j++ {
			n := nTile + j
			weightIdx := k*N + n
			packedIdx := weightIdx / 2

			var unsignedVal int
			if weightIdx%2 == 0 {
				unsignedVal = int(packed[packedIdx] & 0x0F)
			} else {
				unsignedVal = int((packed[packedIdx] >> 4) & 0x0F)
			}

			groupIdx := n / groupSize
			scale := scales[k*numGroups+groupIdx]
			tile[k*tileN+j] = float32(unsignedVal-8) * scale
		}
	}
}

// processFusedNF4Tile processes a single N-tile for NF4 matmul.
// inputT is the transposed input [K, M], packed is NF4 weights, output is [M, N].
func processFusedNF4Tile(
	inputT []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	tileBuf []float32,
	outputTile []float32,
	nTile, M, K, N, numGroups, groupSize int,
) {
	nEnd := min(nTile+16, N)
	tileN := nEnd - nTile

	// Dequantize weight tile: [K, tileN] from packed [K, N/2]
	dequantizeNF4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

	// SME matmul: inputT[K, M]^T @ tileBuf[K, tileN] = input[M, K] @ tile[K, tileN]
	matmul_fmopa_at_f32(
		unsafe.Pointer(unsafe.SliceData(inputT)),
		unsafe.Pointer(unsafe.SliceData(tileBuf[:K*tileN])),
		unsafe.Pointer(unsafe.SliceData(outputTile[:M*tileN])),
		int64(M),
		int64(tileN),
		int64(K),
	)

	// Copy output tile to final output (scatter to correct columns)
	for m := 0; m < M; m++ {
		for j := 0; j < tileN; j++ {
			output[m*N+nTile+j] = outputTile[m*tileN+j]
		}
	}
}

// processFusedInt4Tile processes a single N-tile for Int4 matmul.
func processFusedInt4Tile(
	inputT []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	tileBuf []float32,
	outputTile []float32,
	nTile, M, K, N, numGroups, groupSize int,
) {
	nEnd := min(nTile+16, N)
	tileN := nEnd - nTile

	dequantizeInt4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

	matmul_fmopa_at_f32(
		unsafe.Pointer(unsafe.SliceData(inputT)),
		unsafe.Pointer(unsafe.SliceData(tileBuf[:K*tileN])),
		unsafe.Pointer(unsafe.SliceData(outputTile[:M*tileN])),
		int64(M),
		int64(tileN),
		int64(K),
	)

	for m := 0; m < M; m++ {
		for j := 0; j < tileN; j++ {
			output[m*N+nTile+j] = outputTile[m*tileN+j]
		}
	}
}

// parallelFusedNF4MatMulSME performs fused NF4 matmul with parallel N-tile processing.
// Shares the transposed input across workers; each worker processes independent tiles.
func parallelFusedNF4MatMulSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if !hwy.HasSME() {
		BaseFusedNF4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	// Check alignment for SME (16x16 tiles)
	if K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		BaseFusedNF4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	numTiles := (N + 15) / 16
	numGroups := (N + groupSize - 1) / groupSize

	// Fall back to sequential if too few tiles
	if numTiles < MinFusedParallelTiles {
		fusedNF4MatMulSME(input, packed, scales, output, M, K, N, groupSize)
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
			// Get thread-local buffers from pool
			tileBuf := fusedTilePool.Get().([]float32)
			tileSize := K * 16
			if cap(tileBuf) < tileSize {
				tileBuf = make([]float32, tileSize)
			} else {
				tileBuf = tileBuf[:tileSize]
			}
			defer fusedTilePool.Put(tileBuf)

			outputTile := fusedOutputTilePool.Get().([]float32)
			outputTileSize := M * 16
			if cap(outputTile) < outputTileSize {
				outputTile = make([]float32, outputTileSize)
			} else {
				outputTile = outputTile[:outputTileSize]
			}
			defer fusedOutputTilePool.Put(outputTile)

			for nTile := range work {
				processFusedNF4Tile(inputT, packed, scales, output, tileBuf, outputTile,
					nTile, M, K, N, numGroups, groupSize)
			}
		})
	}
	wg.Wait()
}

// parallelFusedInt4MatMulSME performs fused Int4 matmul with parallel N-tile processing.
func parallelFusedInt4MatMulSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if !hwy.HasSME() {
		BaseFusedInt4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	if K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		BaseFusedInt4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)
		return
	}

	numTiles := (N + 15) / 16
	numGroups := (N + groupSize - 1) / groupSize

	if numTiles < MinFusedParallelTiles {
		fusedInt4MatMulSME(input, packed, scales, output, M, K, N, groupSize)
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

	numWorkers := min(runtime.GOMAXPROCS(0), numTiles)
	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			tileBuf := fusedTilePool.Get().([]float32)
			tileSize := K * 16
			if cap(tileBuf) < tileSize {
				tileBuf = make([]float32, tileSize)
			} else {
				tileBuf = tileBuf[:tileSize]
			}
			defer fusedTilePool.Put(tileBuf)

			outputTile := fusedOutputTilePool.Get().([]float32)
			outputTileSize := M * 16
			if cap(outputTile) < outputTileSize {
				outputTile = make([]float32, outputTileSize)
			} else {
				outputTile = outputTile[:outputTileSize]
			}
			defer fusedOutputTilePool.Put(outputTile)

			for nTile := range work {
				processFusedInt4Tile(inputT, packed, scales, output, tileBuf, outputTile,
					nTile, M, K, N, numGroups, groupSize)
			}
		})
	}
	wg.Wait()
}

func init() {
	if hwy.HasSME() {
		// Override dispatch with SME-optimized implementations
		FusedNF4MatMul = fusedNF4MatMulSME
		FusedInt4MatMul = fusedInt4MatMulSME
		ParallelFusedNF4MatMul = parallelFusedNF4MatMulSME
		ParallelFusedInt4MatMul = parallelFusedInt4MatMulSME
	}
}
