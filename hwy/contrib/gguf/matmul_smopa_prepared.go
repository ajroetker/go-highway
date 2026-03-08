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

//go:build !noasm && arm64

package gguf

import (
	"sync"
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/gguf/asm"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// GoAT compilation directives for 4-tile SUMOPA/SMOPA prepacked kernels.
//go:generate go tool goat c/multitile_sumopa_prepacked_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv9-a+sme+sme-i16i64" -o asm/
//go:generate go tool goat c/multitile_smopa_prepacked_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv9-a+sme+sme-i16i64" -o asm/

// Buffer pool for 4-tile int32 output: 4 x 16x16 = 1024 int32s.
var smopaTile4Pool = sync.Pool{
	New: func() any { return make([]int32, 1024) },
}

// Buffer pool for 4-tile float32 accumulator: paddedM x 64.
var smopaAcc4Pool = sync.Pool{
	New: func() any { return make([]float32, 0, 4096*64) },
}

const nColsPerGroup = nTilesPerGroup * smeTileSize // 64

// tileScales holds pre-computed activation scales for one M-tile (16 rows).
// dA[row] is the Q8_K block scale for row m = mTile + row.
// dABsum[row] is dA * bsum for unsigned quant types (zero for signed).
type tileScales struct {
	dA     [smeTileSize]float32
	dABsum [smeTileSize]float32
}

// smePreparedGGUFMatMul computes output = input @ weights^T using pre-packed
// weights and the 4-tile SMOPA/SUMOPA kernel. B-panel packing overhead is
// eliminated since panels are pre-packed in PreparedWeights.
//
// Activation panels and dA/bsum values are pre-computed once for all
// (kb, j, mTile) combinations, then reused across N-tile-groups. This
// eliminates O(NTileGroups) redundant packActivationPanel and activationBsum
// calls.
func smePreparedGGUFMatMul(input []float32, pw *PreparedWeights,
	output []float32, M int) {

	K, N := pw.K, pw.N
	nblocks := K / QK_K
	numSubBlocks := pw.NumSubBlocks
	aRowBytes := nblocks * BlockSizeQ8K

	// Step 1: Quantize all activations to Q8_K.
	aDataSize := M * aRowBytes
	aData := getPoolSliceU8(&smopaActivationPool, aDataSize)
	defer smopaActivationPool.Put(aData)
	for m := range M {
		QuantizeQ8_K(input[m*K:(m+1)*K], aData[m*aRowBytes:(m+1)*aRowBytes])
	}

	paddedM := alignUp(M, smeTileSize)
	nMTiles := paddedM / smeTileSize

	// Step 2: Pre-compute activation panels and dA/bsum values.
	// These depend on (kb, j, mTile) but not on the N-tile-group,
	// so computing them once saves NTileGroups-fold redundant work.
	panelSize := pw.KGroups * 64
	nEntries := nblocks * numSubBlocks * nMTiles
	cachedPanels := make([]int8, nEntries*panelSize)
	cachedScales := make([]tileScales, nEntries)

	for kb := range nblocks {
		for j := range numSubBlocks {
			for mt := range nMTiles {
				mTile := mt * smeTileSize
				mRows := min(smeTileSize, M-mTile)
				idx := kb*numSubBlocks*nMTiles + j*nMTiles + mt

				// Pack activation panel once.
				panel := cachedPanels[idx*panelSize : (idx+1)*panelSize]
				packActivationPanel(aData, aRowBytes, kb, j, pw.SubBlockSize,
					mTile, mRows, panel)

				// Pre-compute dA and dABsum for each row.
				ts := &cachedScales[idx]
				for row := range mRows {
					m := mTile + row
					aBlockOff := m*aRowBytes + kb*BlockSizeQ8K
					ts.dA[row] = f32LE(aData[aBlockOff], aData[aBlockOff+1],
						aData[aBlockOff+2], aData[aBlockOff+3])
					if !pw.Signed {
						bsumsOff := aBlockOff + 4 + QK_K
						ts.dABsum[row] = ts.dA[row] * float32(
							activationBsum(aData, bsumsOff, j, pw.SubBlockSize))
					}
				}
			}
		}
	}

	// 4-tile int32 output buffer.
	tileI32 := getPoolSliceI32(&smopaTile4Pool, 1024)
	defer smopaTile4Pool.Put(tileI32)

	// Float32 accumulator for one N-tile-group (paddedM x 64).
	accTile := getPoolSliceF32(&smopaAcc4Pool, paddedM*nColsPerGroup)
	defer smopaAcc4Pool.Put(accTile)

	defer hwy.SMEGuard()()

	for ng := range pw.NTileGroups {
		nBase := ng * nColsPerGroup
		nCols := min(nColsPerGroup, N-nBase)

		clear(accTile[:paddedM*nColsPerGroup])

		for kb := range nblocks {
			scaleKBOff := pw.ScaleOffset(ng, kb)
			var minKBOff int
			if !pw.Signed {
				minKBOff = pw.MinOffset(ng, kb)
			}

			for j := range numSubBlocks {
				panelOff := pw.PanelOffset(ng, kb, j)
				scaleSubOff := scaleKBOff + j*nColsPerGroup
				var minSubOff int
				if !pw.Signed {
					minSubOff = minKBOff + j*nColsPerGroup
				}

				for mt := range nMTiles {
					mTile := mt * smeTileSize
					idx := kb*numSubBlocks*nMTiles + j*nMTiles + mt

					aPtr := unsafe.Pointer(&cachedPanels[idx*panelSize])
					bPtr := unsafe.Pointer(&pw.Panels[panelOff])
					tPtr := unsafe.Pointer(&tileI32[0])

					if pw.Signed {
						asm.MultiTileSMOPAPrepacked(aPtr, bPtr, tPtr, int64(pw.KGroups))
					} else {
						asm.MultiTileSUMOPAPrepacked(aPtr, bPtr, tPtr, int64(pw.KGroups))
					}

					ts := &cachedScales[idx]
					nRows := min(smeTileSize, M-mTile)
					if pw.Signed {
						AccumulateTilesSigned(accTile, tileI32,
							pw.Scales[scaleSubOff:scaleSubOff+nColsPerGroup],
							ts.dA[:nRows],
							mTile, nRows)
					} else {
						AccumulateTilesUnsigned(accTile, tileI32,
							pw.Scales[scaleSubOff:scaleSubOff+nColsPerGroup],
							pw.Scales[minSubOff:minSubOff+nColsPerGroup],
							ts.dA[:nRows], ts.dABsum[:nRows],
							mTile, nRows)
					}
				}
			}
		}

		// Write accumulated results to output.
		for m := range M {
			for col := range nCols {
				output[m*N+nBase+col] = accTile[m*nColsPerGroup+col]
			}
		}
	}
}

// accumulatePreparedTiles applies pre-computed scales to the 4-tile int32
// output and adds to the float32 accumulator.
//
// Tile layout: tileI32[tile*256 + row*16 + col] for tile in [0,4).
// Scale layout: scales[scaleSubOff + tile*16 + col].
// Min layout (unsigned only): scales[minSubOff + tile*16 + col].
func accumulatePreparedTiles(accTile []float32, tileI32 []int32,
	scales []float32, scaleSubOff, minSubOff int,
	aData []uint8, aRowBytes, kb, mTile, M, nCols, j,
	subBlockSize int, signed bool) {

	for row := range smeTileSize {
		m := mTile + row
		if m >= M {
			break
		}

		// Activation d for this row's Q8_K block.
		aBlockOff := m*aRowBytes + kb*BlockSizeQ8K
		dA := f32LE(aData[aBlockOff], aData[aBlockOff+1], aData[aBlockOff+2], aData[aBlockOff+3])

		// For unsigned types, compute bsum once per (row, j).
		var dABsum float32
		if !signed {
			bsumsOff := aBlockOff + 4 + QK_K
			dABsum = dA * float32(activationBsum(aData, bsumsOff, j, subBlockSize))
		}

		for tile := range nTilesPerGroup {
			tileNCols := min(smeTileSize, nCols-tile*smeTileSize)
			if tileNCols <= 0 {
				break
			}

			tileBase := tile * 256
			scaleBase := scaleSubOff + tile*smeTileSize
			accBase := m*nColsPerGroup + tile*smeTileSize

			for col := range tileNCols {
				raw := float32(tileI32[tileBase+row*smeTileSize+col])
				accTile[accBase+col] += raw * scales[scaleBase+col] * dA
			}

			if !signed {
				minBase := minSubOff + tile*smeTileSize
				for col := range tileNCols {
					accTile[accBase+col] -= scales[minBase+col] * dABsum
				}
			}
		}
	}
}

// accumulatePreparedTilesV2 is the optimized accumulation function that takes
// pre-computed dA and dABsum values. The body is kept free of aData/f32LE/
// activationBsum code to allow the Go compiler to generate optimal code for
// the inner float accumulation loops.
func accumulatePreparedTilesV2(accTile []float32, tileI32 []int32,
	scales []float32, scaleSubOff, minSubOff int,
	dA *[smeTileSize]float32, dABsum *[smeTileSize]float32,
	mTile, M, nCols int, signed bool) {

	nRows := min(smeTileSize, M-mTile)
	for row := range nRows {
		m := mTile + row
		d := dA[row]

		for tile := range nTilesPerGroup {
			tileNCols := min(smeTileSize, nCols-tile*smeTileSize)
			if tileNCols <= 0 {
				break
			}

			tileBase := tile * 256
			scaleBase := scaleSubOff + tile*smeTileSize
			accBase := m*nColsPerGroup + tile*smeTileSize

			for col := range tileNCols {
				raw := float32(tileI32[tileBase+row*smeTileSize+col])
				accTile[accBase+col] += raw * scales[scaleBase+col] * d
			}

			if !signed {
				bs := dABsum[row]
				minBase := minSubOff + tile*smeTileSize
				for col := range tileNCols {
					accTile[accBase+col] -= scales[minBase+col] * bs
				}
			}
		}
	}
}

// parallelSMEPreparedGGUFMatMul distributes N-tile-groups across workers.
// Activations are quantized once (shared); each worker processes its own
// N-tile-group range using pre-packed B panels and pre-computed activation
// scales from the shared caches.
func parallelSMEPreparedGGUFMatMul(pool workerpool.Executor, input []float32,
	pw *PreparedWeights, output []float32, M int) {

	K, N := pw.K, pw.N
	nblocks := K / QK_K
	numSubBlocks := pw.NumSubBlocks
	aRowBytes := nblocks * BlockSizeQ8K

	// Quantize activations (shared across workers).
	aDataSize := M * aRowBytes
	aData := getPoolSliceU8(&smopaActivationPool, aDataSize)
	defer smopaActivationPool.Put(aData)
	for m := range M {
		QuantizeQ8_K(input[m*K:(m+1)*K], aData[m*aRowBytes:(m+1)*aRowBytes])
	}

	paddedM := alignUp(M, smeTileSize)
	nMTiles := paddedM / smeTileSize

	// Pre-compute activation panels and dA/bsum values (shared read-only).
	panelSize := pw.KGroups * 64
	nEntries := nblocks * numSubBlocks * nMTiles
	cachedPanels := make([]int8, nEntries*panelSize)
	cachedScales := make([]tileScales, nEntries)

	for kb := range nblocks {
		for j := range numSubBlocks {
			for mt := range nMTiles {
				mTile := mt * smeTileSize
				mRows := min(smeTileSize, M-mTile)
				idx := kb*numSubBlocks*nMTiles + j*nMTiles + mt

				panel := cachedPanels[idx*panelSize : (idx+1)*panelSize]
				packActivationPanel(aData, aRowBytes, kb, j, pw.SubBlockSize,
					mTile, mRows, panel)

				ts := &cachedScales[idx]
				for row := range mRows {
					m := mTile + row
					aBlockOff := m*aRowBytes + kb*BlockSizeQ8K
					ts.dA[row] = f32LE(aData[aBlockOff], aData[aBlockOff+1],
						aData[aBlockOff+2], aData[aBlockOff+3])
					if !pw.Signed {
						bsumsOff := aBlockOff + 4 + QK_K
						ts.dABsum[row] = ts.dA[row] * float32(
							activationBsum(aData, bsumsOff, j, pw.SubBlockSize))
					}
				}
			}
		}
	}

	pool.ParallelFor(pw.NTileGroups, func(ngStart, ngEnd int) {
		tileI32 := getPoolSliceI32(&smopaTile4Pool, 1024)
		defer smopaTile4Pool.Put(tileI32)

		accTile := getPoolSliceF32(&smopaAcc4Pool, paddedM*nColsPerGroup)
		defer smopaAcc4Pool.Put(accTile)

		defer hwy.SMEGuard()()

		for ng := ngStart; ng < ngEnd; ng++ {
			nBase := ng * nColsPerGroup
			nCols := min(nColsPerGroup, N-nBase)

			clear(accTile[:paddedM*nColsPerGroup])

			for kb := range nblocks {
				scaleKBOff := pw.ScaleOffset(ng, kb)
				var minKBOff int
				if !pw.Signed {
					minKBOff = pw.MinOffset(ng, kb)
				}

				for j := range numSubBlocks {
					panelOff := pw.PanelOffset(ng, kb, j)
					scaleSubOff := scaleKBOff + j*nColsPerGroup
					var minSubOff int
					if !pw.Signed {
						minSubOff = minKBOff + j*nColsPerGroup
					}

					for mt := range nMTiles {
						mTile := mt * smeTileSize
						idx := kb*numSubBlocks*nMTiles + j*nMTiles + mt

						aPtr := unsafe.Pointer(&cachedPanels[idx*panelSize])
						bPtr := unsafe.Pointer(&pw.Panels[panelOff])
						tPtr := unsafe.Pointer(&tileI32[0])

						if pw.Signed {
							asm.MultiTileSMOPAPrepacked(aPtr, bPtr, tPtr, int64(pw.KGroups))
						} else {
							asm.MultiTileSUMOPAPrepacked(aPtr, bPtr, tPtr, int64(pw.KGroups))
						}

						ts := &cachedScales[idx]
						nRows := min(smeTileSize, M-mTile)
						if pw.Signed {
							AccumulateTilesSigned(accTile, tileI32,
								pw.Scales[scaleSubOff:scaleSubOff+nColsPerGroup],
								ts.dA[:nRows],
								mTile, nRows)
						} else {
							AccumulateTilesUnsigned(accTile, tileI32,
								pw.Scales[scaleSubOff:scaleSubOff+nColsPerGroup],
								pw.Scales[minSubOff:minSubOff+nColsPerGroup],
								ts.dA[:nRows], ts.dABsum[:nRows],
								mTile, nRows)
						}
					}
				}
			}

			for m := range M {
				for col := range nCols {
					output[m*N+nBase+col] = accTile[m*nColsPerGroup+col]
				}
			}
		}
	})
}
