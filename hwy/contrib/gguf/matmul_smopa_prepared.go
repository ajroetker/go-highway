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

// GoAT compilation directive for SDOT/TBL vecdot kernels (Tier 1 types).
//go:generate go tool goat c/vecdot_sdot_neon_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv8.2-a+dotprod+simd+fp" -o asm/

// GoAT compilation directive for fused quantize+pack activation kernels.
//go:generate go tool goat c/fused_quantize_pack_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv8.2-a+dotprod+simd+fp" -o asm/

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
// Activation panels are prepared via fused quantize+pack: float32 input is
// quantized to int8 and packed directly into A-panel layout in a single pass,
// eliminating the intermediate Q8_K buffer. Block scales (absmax/127) are
// precomputed once, then reused for all sub-blocks within each Q8_K block.
func smePreparedGGUFMatMul(input []float32, pw *PreparedWeights,
	output []float32, M int) {

	K, N := pw.K, pw.N
	nblocks := K / QK_K
	numSubBlocks := pw.NumSubBlocks

	// Step 1: Precompute block scales (absmax/127 per block per row).
	// This replaces the full Q8_K quantization, eliminating the M×nblocks×292
	// byte intermediate buffer. dAScales uses only M×nblocks×4 bytes.
	dAScales := make([]float32, M*nblocks)
	for m := range M {
		for kb := range nblocks {
			off := m*K + kb*QK_K
			amax := asm.ComputeAbsmax(unsafe.Pointer(&input[off]), int64(QK_K))
			dAScales[m*nblocks+kb] = amax / 127.0
		}
	}

	paddedM := alignUp(M, smeTileSize)
	nMTiles := paddedM / smeTileSize

	// Step 2: Fused quantize+pack activation panels directly from float32 input.
	// For each (kb, j, mTile), quantize the sub-block values and pack into
	// A-panel layout in a single pass, computing dA and dABsum as byproducts.
	panelSize := pw.KGroups * 64
	nEntries := nblocks * numSubBlocks * nMTiles
	cachedPanels := make([]int8, nEntries*panelSize)
	cachedScales := make([]tileScales, nEntries)

	var invScaleBuf [smeTileSize]float32
	var bsumBuf [smeTileSize]int64

	for kb := range nblocks {
		for j := range numSubBlocks {
			for mt := range nMTiles {
				mTile := mt * smeTileSize
				mRows := min(smeTileSize, M-mTile)
				idx := kb*numSubBlocks*nMTiles + j*nMTiles + mt

				// Compute inverse scales for each row in this M-tile.
				ts := &cachedScales[idx]
				for row := range mRows {
					m := mTile + row
					d := dAScales[m*nblocks+kb]
					ts.dA[row] = d
					if d > 0 {
						invScaleBuf[row] = 1.0 / d
					} else {
						invScaleBuf[row] = 0
					}
				}

				// Fused quantize + pack into A-panel.
				panel := cachedPanels[idx*panelSize : (idx+1)*panelSize]
				inputOff := mTile*K + kb*QK_K + j*pw.SubBlockSize
				if !pw.Signed {
					asm.FusedQuantizePackBsum(
						unsafe.Pointer(&input[inputOff]),
						int64(K),
						unsafe.Pointer(&invScaleBuf[0]),
						int64(pw.SubBlockSize),
						int64(mRows),
						unsafe.Pointer(&panel[0]),
						unsafe.Pointer(&bsumBuf[0]),
					)
					for row := range mRows {
						ts.dABsum[row] = ts.dA[row] * float32(bsumBuf[row])
					}
				} else {
					asm.FusedQuantizePack(
						unsafe.Pointer(&input[inputOff]),
						int64(K),
						unsafe.Pointer(&invScaleBuf[0]),
						int64(pw.SubBlockSize),
						int64(mRows),
						unsafe.Pointer(&panel[0]),
					)
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

// accumulatePreparedTilesV2 is the optimized accumulation function that takes
// pre-computed dA and dABsum values, enabling the Go compiler to generate
// optimal code for the inner float accumulation loops.
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

	// Precompute block scales (shared across workers).
	dAScales := make([]float32, M*nblocks)
	for m := range M {
		for kb := range nblocks {
			off := m*K + kb*QK_K
			amax := asm.ComputeAbsmax(unsafe.Pointer(&input[off]), int64(QK_K))
			dAScales[m*nblocks+kb] = amax / 127.0
		}
	}

	paddedM := alignUp(M, smeTileSize)
	nMTiles := paddedM / smeTileSize

	// Fused quantize+pack activation panels (shared read-only).
	panelSize := pw.KGroups * 64
	nEntries := nblocks * numSubBlocks * nMTiles
	cachedPanels := make([]int8, nEntries*panelSize)
	cachedScales := make([]tileScales, nEntries)

	var invScaleBuf [smeTileSize]float32
	var bsumBuf [smeTileSize]int64

	for kb := range nblocks {
		for j := range numSubBlocks {
			for mt := range nMTiles {
				mTile := mt * smeTileSize
				mRows := min(smeTileSize, M-mTile)
				idx := kb*numSubBlocks*nMTiles + j*nMTiles + mt

				ts := &cachedScales[idx]
				for row := range mRows {
					m := mTile + row
					d := dAScales[m*nblocks+kb]
					ts.dA[row] = d
					if d > 0 {
						invScaleBuf[row] = 1.0 / d
					} else {
						invScaleBuf[row] = 0
					}
				}

				panel := cachedPanels[idx*panelSize : (idx+1)*panelSize]
				inputOff := mTile*K + kb*QK_K + j*pw.SubBlockSize
				if !pw.Signed {
					asm.FusedQuantizePackBsum(
						unsafe.Pointer(&input[inputOff]),
						int64(K),
						unsafe.Pointer(&invScaleBuf[0]),
						int64(pw.SubBlockSize),
						int64(mRows),
						unsafe.Pointer(&panel[0]),
						unsafe.Pointer(&bsumBuf[0]),
					)
					for row := range mRows {
						ts.dABsum[row] = ts.dA[row] * float32(bsumBuf[row])
					}
				} else {
					asm.FusedQuantizePack(
						unsafe.Pointer(&input[inputOff]),
						int64(K),
						unsafe.Pointer(&invScaleBuf[0]),
						int64(pw.SubBlockSize),
						int64(mRows),
						unsafe.Pointer(&panel[0]),
					)
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
