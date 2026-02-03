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

// NOTE: This file is named "matmul_fused_nf4_act_sme.go" (starting with 'm')
// to ensure its init() runs AFTER dispatch files (starting with 'd' or 'f').
// Go executes init() functions in lexicographic filename order within a package.

package matmul

import (
	"runtime"
	"sync"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// =============================================================================
// Scalar Activation Functions for SME Output Tiles
// =============================================================================

// applySiLUScalar applies SiLU activation: x * sigmoid(x)
func applySiLUScalar(x float32) float32 {
	return x * sigmoidf32(x)
}

// applyGELUScalar applies exact GELU: x * 0.5 * (1 + erf(x/sqrt(2)))
func applyGELUScalar(x float32) float32 {
	return x * 0.5 * (1.0 + erff32(x*0.7071067811865476))
}

// =============================================================================
// Tile Activation Functions
// =============================================================================

// applyActivationToTile applies activation function to output tile region.
// The tile region is: output[m*stride+colOffset : m*stride+colOffset+tileN] for all m.
func applyActivationToTile(output []float32, M, tileN, stride, colOffset int, act ActivationType) {
	switch act {
	case ActSiLU:
		for m := 0; m < M; m++ {
			rowStart := m*stride + colOffset
			for j := 0; j < tileN; j++ {
				idx := rowStart + j
				output[idx] = applySiLUScalar(output[idx])
			}
		}
	case ActGELU:
		for m := 0; m < M; m++ {
			rowStart := m*stride + colOffset
			for j := 0; j < tileN; j++ {
				idx := rowStart + j
				output[idx] = applyGELUScalar(output[idx])
			}
		}
	case ActGELUApprox:
		for m := 0; m < M; m++ {
			rowStart := m*stride + colOffset
			for j := 0; j < tileN; j++ {
				idx := rowStart + j
				x := output[idx]
				output[idx] = x * sigmoidf32(1.702*x)
			}
		}
	case ActReLU:
		for m := 0; m < M; m++ {
			rowStart := m*stride + colOffset
			for j := 0; j < tileN; j++ {
				idx := rowStart + j
				if output[idx] < 0 {
					output[idx] = 0
				}
			}
		}
	// ActNone: do nothing
	}
}

// =============================================================================
// Fused NF4 + Activation SME Implementations
// =============================================================================

// fusedNF4MatMulSiLUSME performs fused NF4 dequant + matmul + SiLU using SME.
func fusedNF4MatMulSiLUSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	fusedNF4MatMulActSME(input, packed, scales, output, M, K, N, groupSize, ActSiLU)
}

// fusedNF4MatMulGELUSME performs fused NF4 dequant + matmul + GELU using SME.
func fusedNF4MatMulGELUSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	fusedNF4MatMulActSME(input, packed, scales, output, M, K, N, groupSize, ActGELU)
}

// fusedNF4MatMulActSME is the core SME implementation for fused NF4 + activation.
func fusedNF4MatMulActSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
	act ActivationType,
) {
	if !hwy.HasSME() {
		// Fall back to base implementation
		baseFusedNF4MatMulAct(input, packed, scales, output, M, K, N, groupSize, act)
		return
	}

	// Check alignment for SME (16x16 tiles)
	if K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		baseFusedNF4MatMulAct(input, packed, scales, output, M, K, N, groupSize, act)
		return
	}

	defer hwy.SMEGuard()()

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

	// Transpose buffer for input
	inputT := transposePool32.Get().([]float32)
	inputTSize := M * K
	if cap(inputT) < inputTSize {
		inputT = make([]float32, inputTSize)
	} else {
		inputT = inputT[:inputTSize]
	}
	defer transposePool32.Put(inputT)

	transposeMatrix(input, M, K, inputT)

	// Zero output
	clear(output[:M*N])

	// Process N in 16-column tiles
	for nTile := 0; nTile < N; nTile += 16 {
		nEnd := min(nTile+16, N)
		tileN := nEnd - nTile

		// Dequantize weight tile
		dequantizeNF4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

		// Strided FMOPA: writes directly to output
		asm.MultiTileMatMulFMOPAF32Strided(inputT, tileBuf[:K*tileN], output, M, tileN, K, N, nTile)

		// Apply activation to this tile's columns
		applyActivationToTile(output, M, tileN, N, nTile, act)
	}
}

// =============================================================================
// Fused Int4 + Activation SME Implementations
// =============================================================================

// fusedInt4MatMulSiLUSME performs fused Int4 dequant + matmul + SiLU using SME.
func fusedInt4MatMulSiLUSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	fusedInt4MatMulActSME(input, packed, scales, output, M, K, N, groupSize, ActSiLU)
}

// fusedInt4MatMulGELUSME performs fused Int4 dequant + matmul + GELU using SME.
func fusedInt4MatMulGELUSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	fusedInt4MatMulActSME(input, packed, scales, output, M, K, N, groupSize, ActGELU)
}

// fusedInt4MatMulActSME is the core SME implementation for fused Int4 + activation.
func fusedInt4MatMulActSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
	act ActivationType,
) {
	if !hwy.HasSME() || K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		baseFusedInt4MatMulAct(input, packed, scales, output, M, K, N, groupSize, act)
		return
	}

	defer hwy.SMEGuard()()

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

	clear(output[:M*N])

	for nTile := 0; nTile < N; nTile += 16 {
		nEnd := min(nTile+16, N)
		tileN := nEnd - nTile

		dequantizeInt4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)

		asm.MultiTileMatMulFMOPAF32Strided(inputT, tileBuf[:K*tileN], output, M, tileN, K, N, nTile)

		applyActivationToTile(output, M, tileN, N, nTile, act)
	}
}

// =============================================================================
// Parallel SME Implementations with Activation
// =============================================================================

// processFusedNF4TileWithAct processes a single N-tile with activation.
func processFusedNF4TileWithAct(
	inputT []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	tileBuf []float32,
	nTile, M, K, N, numGroups, groupSize int,
	act ActivationType,
) {
	nEnd := min(nTile+16, N)
	tileN := nEnd - nTile

	dequantizeNF4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)
	asm.MultiTileMatMulFMOPAF32Strided(inputT, tileBuf[:K*tileN], output, M, tileN, K, N, nTile)
	applyActivationToTile(output, M, tileN, N, nTile, act)
}

// processFusedInt4TileWithAct processes a single N-tile with activation.
func processFusedInt4TileWithAct(
	inputT []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	tileBuf []float32,
	nTile, M, K, N, numGroups, groupSize int,
	act ActivationType,
) {
	nEnd := min(nTile+16, N)
	tileN := nEnd - nTile

	dequantizeInt4Tile(packed, scales, tileBuf, nTile, K, N, tileN, numGroups, groupSize)
	asm.MultiTileMatMulFMOPAF32Strided(inputT, tileBuf[:K*tileN], output, M, tileN, K, N, nTile)
	applyActivationToTile(output, M, tileN, N, nTile, act)
}

// parallelFusedNF4MatMulSiLUSME performs parallel fused NF4 + SiLU using SME.
func parallelFusedNF4MatMulSiLUSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	parallelFusedNF4MatMulActSME(input, packed, scales, output, M, K, N, groupSize, ActSiLU)
}

// parallelFusedNF4MatMulGELUSME performs parallel fused NF4 + GELU using SME.
func parallelFusedNF4MatMulGELUSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	parallelFusedNF4MatMulActSME(input, packed, scales, output, M, K, N, groupSize, ActGELU)
}

// parallelFusedNF4MatMulActSME performs parallel fused NF4 + activation using SME.
func parallelFusedNF4MatMulActSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
	act ActivationType,
) {
	if !hwy.HasSME() {
		baseFusedNF4MatMulAct(input, packed, scales, output, M, K, N, groupSize, act)
		return
	}

	if K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		baseFusedNF4MatMulAct(input, packed, scales, output, M, K, N, groupSize, act)
		return
	}

	numTiles := (N + 15) / 16
	numGroups := (N + groupSize - 1) / groupSize

	if numTiles < MinFusedParallelTiles {
		fusedNF4MatMulActSME(input, packed, scales, output, M, K, N, groupSize, act)
		return
	}

	// Transpose input once (shared across workers)
	inputT := transposePool32.Get().([]float32)
	inputTSize := M * K
	if cap(inputT) < inputTSize {
		inputT = make([]float32, inputTSize)
	} else {
		inputT = inputT[:inputTSize]
	}
	transposeMatrix(input, M, K, inputT)
	defer transposePool32.Put(inputT)

	clear(output[:M*N])

	work := make(chan int, numTiles)
	for nTile := 0; nTile < N; nTile += 16 {
		work <- nTile
	}
	close(work)

	numWorkers := min(runtime.GOMAXPROCS(0), numTiles)
	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			defer hwy.SMEGuard()()

			tileBuf := fusedTilePool.Get().([]float32)
			tileSize := K * 16
			if cap(tileBuf) < tileSize {
				tileBuf = make([]float32, tileSize)
			} else {
				tileBuf = tileBuf[:tileSize]
			}
			clear(tileBuf)
			defer fusedTilePool.Put(tileBuf)

			for nTile := range work {
				processFusedNF4TileWithAct(inputT, packed, scales, output, tileBuf,
					nTile, M, K, N, numGroups, groupSize, act)
			}
		})
	}
	wg.Wait()
}

// parallelFusedInt4MatMulSiLUSME performs parallel fused Int4 + SiLU using SME.
func parallelFusedInt4MatMulSiLUSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	parallelFusedInt4MatMulActSME(input, packed, scales, output, M, K, N, groupSize, ActSiLU)
}

// parallelFusedInt4MatMulGELUSME performs parallel fused Int4 + GELU using SME.
func parallelFusedInt4MatMulGELUSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	parallelFusedInt4MatMulActSME(input, packed, scales, output, M, K, N, groupSize, ActGELU)
}

// parallelFusedInt4MatMulActSME performs parallel fused Int4 + activation using SME.
func parallelFusedInt4MatMulActSME(
	input []float32,
	packed []uint8,
	scales []float32,
	output []float32,
	M, K, N, groupSize int,
	act ActivationType,
) {
	if !hwy.HasSME() || K%16 != 0 || N%16 != 0 || M < 64 || K < 64 || N < 64 {
		baseFusedInt4MatMulAct(input, packed, scales, output, M, K, N, groupSize, act)
		return
	}

	numTiles := (N + 15) / 16
	numGroups := (N + groupSize - 1) / groupSize

	if numTiles < MinFusedParallelTiles {
		fusedInt4MatMulActSME(input, packed, scales, output, M, K, N, groupSize, act)
		return
	}

	inputT := transposePool32.Get().([]float32)
	inputTSize := M * K
	if cap(inputT) < inputTSize {
		inputT = make([]float32, inputTSize)
	} else {
		inputT = inputT[:inputTSize]
	}
	transposeMatrix(input, M, K, inputT)
	defer transposePool32.Put(inputT)

	clear(output[:M*N])

	work := make(chan int, numTiles)
	for nTile := 0; nTile < N; nTile += 16 {
		work <- nTile
	}
	close(work)

	numWorkers := min(runtime.GOMAXPROCS(0), numTiles)
	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			defer hwy.SMEGuard()()

			tileBuf := fusedTilePool.Get().([]float32)
			tileSize := K * 16
			if cap(tileBuf) < tileSize {
				tileBuf = make([]float32, tileSize)
			} else {
				tileBuf = tileBuf[:tileSize]
			}
			clear(tileBuf)
			defer fusedTilePool.Put(tileBuf)

			for nTile := range work {
				processFusedInt4TileWithAct(inputT, packed, scales, output, tileBuf,
					nTile, M, K, N, numGroups, groupSize, act)
			}
		})
	}
	wg.Wait()
}

// =============================================================================
// init() - Override dispatch with SME implementations
// =============================================================================

func init() {
	if !hwy.HasSME() {
		return
	}

	// Override fused NF4 + activation dispatch
	FusedNF4MatMulSiLU = fusedNF4MatMulSiLUSME
	FusedNF4MatMulGELU = fusedNF4MatMulGELUSME
	ParallelFusedNF4MatMulSiLU = parallelFusedNF4MatMulSiLUSME
	ParallelFusedNF4MatMulGELU = parallelFusedNF4MatMulGELUSME

	// Override fused Int4 + activation dispatch
	FusedInt4MatMulSiLU = fusedInt4MatMulSiLUSME
	FusedInt4MatMulGELU = fusedInt4MatMulGELUSME
	ParallelFusedInt4MatMulSiLU = parallelFusedInt4MatMulSiLUSME
	ParallelFusedInt4MatMulGELU = parallelFusedInt4MatMulGELUSME
}
