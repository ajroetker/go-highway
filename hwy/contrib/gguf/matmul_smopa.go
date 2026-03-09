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
	ggufAsm "github.com/ajroetker/go-highway/hwy/contrib/gguf/asm"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// Buffer pools for SMOPA matmul to reduce allocations.
// Accumulator pool: paddedM * 16, typically up to 4096 * 16 = 64K floats.
var smopaAccPool = sync.Pool{
	New: func() any { return make([]float32, 0, 4096*smeTileSize) },
}

// Panel pool: up to 8 kGroups * 64 = 512 bytes (Q4_K max).
// Shared for both signed and unsigned panels.
var smopaPanelPool = sync.Pool{
	New: func() any { return make([]byte, 0, 512) },
}

// Int32 tile pool: 16*16 = 256 int32s.
var smopaTilePool = sync.Pool{
	New: func() any { return make([]int32, 256) },
}

// getPoolSlice retrieves a byte slice from the pool, resized to n.
func getPoolSlice(pool *sync.Pool, n int) []byte {
	buf := pool.Get().([]byte)
	if cap(buf) < n {
		buf = make([]byte, n)
	} else {
		buf = buf[:n]
	}
	return buf
}

// getPoolSliceF32 retrieves a float32 slice from the pool, resized to n.
func getPoolSliceF32(pool *sync.Pool, n int) []float32 {
	buf := pool.Get().([]float32)
	if cap(buf) < n {
		buf = make([]float32, n)
	} else {
		buf = buf[:n]
	}
	return buf
}

// getPoolSliceI32 retrieves an int32 slice from the pool, resized to n.
func getPoolSliceI32(pool *sync.Pool, n int) []int32 {
	buf := pool.Get().([]int32)
	if cap(buf) < n {
		buf = make([]int32, n)
	} else {
		buf = buf[:n]
	}
	return buf
}

// alignUp rounds m up to the nearest multiple of tileSize.
func alignUp(m, tileSize int) int {
	return (m + tileSize - 1) / tileSize * tileSize
}

const smeTileSize = 16

// smeGGUFMatMul computes output = input @ weights^T using SME SMOPA/SUMOPA.
//
// For each 16×16 output tile, it iterates over K-blocks (256 values each),
// and within each K-block iterates over sub-blocks. Each sub-block's integer
// quants are packed into SMOPA panels, the outer product is computed, and the
// result is scaled by the per-sub-block scale factors.
func smeGGUFMatMul(input []float32, weights []uint8, output []float32,
	M, K, N int, qt QuantType) {

	info := GetSubBlockInfo(qt)
	if info.NumSubBlocks == 0 {
		// Unsupported type, fall back.
		vecdotGGUFMatMul(input, weights, output, M, K, N, qt)
		return
	}

	nblocks := K / QK_K
	wBlockBytes := BytesPerBlock(qt)
	wRowBytes := nblocks * wBlockBytes

	// Precompute block scales, eliminating the intermediate Q8_K buffer.
	dAScales := make([]float32, M*nblocks)
	for m := range M {
		for kb := range nblocks {
			off := m*K + kb*QK_K
			amax := ggufAsm.ComputeAbsmax(unsafe.Pointer(&input[off]), int64(QK_K))
			dAScales[m*nblocks+kb] = amax / 127.0
		}
	}

	// Pad M to tile boundary.
	paddedM := alignUp(M, smeTileSize)

	// Pre-allocate panel and tile buffers from pools.
	subBlockSize := info.SubBlockSize
	kGroups := subBlockSize / 4
	panelSize := kGroups * 64
	aPanel := make([]int8, panelSize)

	tileI32 := getPoolSliceI32(&smopaTilePool, smeTileSize*smeTileSize)
	defer smopaTilePool.Put(tileI32)

	// Float32 accumulator for one N-tile column (paddedM × smeTileSize).
	accTile := getPoolSliceF32(&smopaAccPool, paddedM*smeTileSize)
	defer smopaAccPool.Put(accTile)

	// Pre-allocate B panel buffers outside the inner loop.
	bPanelBuf := getPoolSlice(&smopaPanelPool, panelSize)
	defer smopaPanelPool.Put(bPanelBuf)

	// Scratch buffers for sub-block quant extraction per weight row.
	var signedBuf [32]int8
	var unsignedBuf [32]uint8

	// Per-M-tile buffers for fused quantize+pack.
	var invScaleBuf [smeTileSize]float32
	var bsumBuf [smeTileSize]int64

	defer hwy.SMEGuard()()

	for nTile := 0; nTile < N; nTile += smeTileSize {
		nCols := min(smeTileSize, N-nTile)

		clear(accTile)

		for kb := range nblocks {
			var metas [smeTileSize]blockMeta

			for col := range nCols {
				n := nTile + col
				wBlock := weights[n*wRowBytes+kb*wBlockBytes : n*wRowBytes+(kb+1)*wBlockBytes]
				parseBlockMeta(qt, wBlock, &metas[col])
			}

			for j := range info.NumSubBlocks {
				clear(bPanelBuf)

				if info.Signed {
					bPanelSigned := byteSliceAsInt8(bPanelBuf)

					for col := range nCols {
						n := nTile + col
						wBlock := weights[n*wRowBytes+kb*wBlockBytes : n*wRowBytes+(kb+1)*wBlockBytes]
						extractSignedSubBlock(qt, wBlock, j, signedBuf[:subBlockSize])
						for k4 := range kGroups {
							for g := range 4 {
								bPanelSigned[k4*64+col*4+g] = signedBuf[k4*4+g]
							}
						}
					}

					for mTile := 0; mTile < paddedM; mTile += smeTileSize {
						mRows := min(smeTileSize, M-mTile)

						// Compute inverse scales for fused quantize+pack.
						for row := range mRows {
							m := mTile + row
							d := dAScales[m*nblocks+kb]
							if d > 0 {
								invScaleBuf[row] = 1.0 / d
							} else {
								invScaleBuf[row] = 0
							}
						}

						inputOff := mTile*K + kb*QK_K + j*subBlockSize
						ggufAsm.FusedQuantizePack(
							unsafe.Pointer(&input[inputOff]),
							int64(K),
							unsafe.Pointer(&invScaleBuf[0]),
							int64(subBlockSize),
							int64(mRows),
							unsafe.Pointer(&aPanel[0]),
						)

						asm.TileSMOPAS8(aPanel, bPanelSigned, tileI32, kGroups)

						accumulateTileSigned(accTile, tileI32, metas[:], dAScales, nblocks,
							kb, mTile, M, nCols, j)
					}
				} else {
					bPanelUnsigned := bPanelBuf

					for col := range nCols {
						n := nTile + col
						wBlock := weights[n*wRowBytes+kb*wBlockBytes : n*wRowBytes+(kb+1)*wBlockBytes]
						extractUnsignedSubBlock(qt, wBlock, j, unsignedBuf[:subBlockSize])
						for k4 := range kGroups {
							for g := range 4 {
								bPanelUnsigned[k4*64+col*4+g] = unsignedBuf[k4*4+g]
							}
						}
					}

					for mTile := 0; mTile < paddedM; mTile += smeTileSize {
						mRows := min(smeTileSize, M-mTile)

						for row := range mRows {
							m := mTile + row
							d := dAScales[m*nblocks+kb]
							if d > 0 {
								invScaleBuf[row] = 1.0 / d
							} else {
								invScaleBuf[row] = 0
							}
						}

						inputOff := mTile*K + kb*QK_K + j*subBlockSize
						ggufAsm.FusedQuantizePackBsum(
							unsafe.Pointer(&input[inputOff]),
							int64(K),
							unsafe.Pointer(&invScaleBuf[0]),
							int64(subBlockSize),
							int64(mRows),
							unsafe.Pointer(&aPanel[0]),
							unsafe.Pointer(&bsumBuf[0]),
						)

						asm.TileSUMOPAS8U8(aPanel, bPanelUnsigned, tileI32, kGroups)

						accumulateTileUnsigned(accTile, tileI32, metas[:], dAScales, nblocks,
							kb, mTile, M, nCols, j, &bsumBuf)
					}
				}
			}
		}

		for m := range M {
			for col := range nCols {
				output[m*N+nTile+col] = accTile[m*smeTileSize+col]
			}
		}
	}
}

// byteSliceAsInt8 reinterprets a []byte as []int8 with the same backing array.
func byteSliceAsInt8(b []byte) []int8 {
	return unsafe.Slice((*int8)(unsafe.Pointer(unsafe.SliceData(b))), len(b))
}

// parseBlockMeta extracts d, dmin, scales, and mins from a GGUF weight block.
func parseBlockMeta(qt QuantType, block []uint8, meta *blockMeta) {
	switch qt {
	case TypeQ4_K:
		meta.d = fp16LE(block[0], block[1])
		meta.dmin = fp16LE(block[2], block[3])
		var scs, mns [8]float32
		unpackQ4KScaleMins(block[4:16], &scs, &mns)
		for i := range 8 {
			meta.scs[i] = scs[i]
			meta.mns[i] = mns[i]
		}
	case TypeQ5_K:
		meta.d = fp16LE(block[0], block[1])
		meta.dmin = fp16LE(block[2], block[3])
		var scs, mns [8]float32
		unpackQ4KScaleMins(block[4:16], &scs, &mns)
		for i := range 8 {
			meta.scs[i] = scs[i]
			meta.mns[i] = mns[i]
		}
	case TypeQ6_K:
		meta.d = fp16LE(block[208], block[209])
		sc := block[192:208]
		for i := range 16 {
			meta.scs[i] = float32(int8(sc[i]))
		}
	case TypeQ2_K:
		meta.d = fp16LE(block[16], block[17])
		meta.dmin = fp16LE(block[18], block[19])
		scalesRaw := block[0:16]
		for i := range 16 {
			meta.scs[i] = float32(scalesRaw[i] & 0x0F)
			meta.mns[i] = float32(scalesRaw[i] >> 4)
		}
	case TypeQ3_K:
		meta.d = fp16LE(block[108], block[109])
		scaleData := block[96:108]
		var rawScales [16]int
		unpackQ3KScales(scaleData, &rawScales)
		for i := range 16 {
			meta.scs[i] = float32(rawScales[i] - 32)
		}
	}
}

type blockMeta struct {
	d    float32
	dmin float32
	scs  [16]float32
	mns  [16]float32
}

// extractSignedSubBlock extracts signed quants for Q6_K or Q3_K sub-block j.
func extractSignedSubBlock(qt QuantType, block []uint8, j int, dst []int8) {
	switch qt {
	case TypeQ6_K:
		extractQ6KSubBlock(block, j, dst)
	case TypeQ3_K:
		extractQ3KSubBlock(block, j, dst)
	}
}

// extractUnsignedSubBlock extracts unsigned quants for Q4_K, Q5_K, or Q2_K sub-block j.
func extractUnsignedSubBlock(qt QuantType, block []uint8, j int, dst []uint8) {
	switch qt {
	case TypeQ4_K:
		extractQ4KSubBlock(block, j, dst)
	case TypeQ5_K:
		extractQ5KSubBlock(block, j, dst)
	case TypeQ2_K:
		extractQ2KSubBlock(block, j, dst)
	}
}

// accumulateTileSigned applies per-sub-block scale to the SMOPA int32 tile
// and adds to the float32 accumulator. Used for Q6_K and Q3_K (signed quants).
//
// For Q6_K: accTile[m][n] += float32(tileI32[m][n]) * d_w[n] * sc[n][j] * d_a[m]
// For Q3_K: same formula (sc already has -32 applied).
func accumulateTileSigned(accTile []float32, tileI32 []int32, metas []blockMeta,
	dAScales []float32, nblocks, kb, mTile, M, nCols, j int) {

	for row := range smeTileSize {
		m := mTile + row
		if m >= M {
			break
		}
		dA := dAScales[m*nblocks+kb]

		for col := range nCols {
			raw := float32(tileI32[row*smeTileSize+col])
			scale := metas[col].d * metas[col].scs[j] * dA
			accTile[m*smeTileSize+col] += raw * scale
		}
	}
}

// accumulateTileUnsigned applies per-sub-block scale and min correction to the
// SUMOPA int32 tile for unsigned quant types (Q4_K, Q5_K, Q2_K).
//
// accTile[m][n] += float32(tileI32[m][n]) * d_w[n] * sc[n][j] * d_a[m]
// accTile[m][n] -= dmin_w[n] * mn[n][j] * d_a[m] * float32(bsum[m])
func accumulateTileUnsigned(accTile []float32, tileI32 []int32, metas []blockMeta,
	dAScales []float32, nblocks, kb, mTile, M, nCols, j int,
	bsumBuf *[smeTileSize]int64) {

	for row := range smeTileSize {
		m := mTile + row
		if m >= M {
			break
		}
		dA := dAScales[m*nblocks+kb]
		bsum := int32(bsumBuf[row])

		for col := range nCols {
			raw := float32(tileI32[row*smeTileSize+col])
			scale := metas[col].d * metas[col].scs[j] * dA
			accTile[m*smeTileSize+col] += raw * scale

			// Min correction.
			if metas[col].dmin != 0 && metas[col].mns[j] != 0 {
				minCorr := metas[col].dmin * metas[col].mns[j] * dA * float32(bsum)
				accTile[m*smeTileSize+col] -= minCorr
			}
		}
	}
}

// vecdotGGUFMatMul is the fallback vecdot-based matmul (same as GGUFMatMul
// but callable from the SME path when dimensions are too small).
func vecdotGGUFMatMul(input []float32, weights []uint8, output []float32,
	M, K, N int, qt QuantType) {

	vecDot := getVecDot(qt)
	quantize := getQuantize(qt)
	if vecDot == nil || quantize == nil {
		return
	}

	nblocks := K / ValuesPerBlock(qt)
	wBlockBytes := BytesPerBlock(qt)
	aBlockBytes := ActivationBlockSize(qt)
	wRowBytes := nblocks * wBlockBytes
	aRowBytes := nblocks * aBlockBytes

	qRow := make([]uint8, aRowBytes)
	for m := range M {
		quantize(input[m*K:(m+1)*K], qRow)
		for n := range N {
			wRow := weights[n*wRowBytes : (n+1)*wRowBytes]
			output[m*N+n] = vecDot(wRow, qRow, nblocks)
		}
	}
}

// parallelSMEGGUFMatMul distributes N-tiles across workers.
// Block scales are precomputed once (shared); each worker fuses quantize+pack.
func parallelSMEGGUFMatMul(pool workerpool.Executor, input []float32,
	weights []uint8, output []float32, M, K, N int, qt QuantType) {

	info := GetSubBlockInfo(qt)
	if info.NumSubBlocks == 0 {
		// Unsupported type, fall back.
		ParallelGGUFMatMul(pool, input, weights, output, M, K, N, qt)
		return
	}

	nblocks := K / QK_K
	wBlockBytes := BytesPerBlock(qt)
	wRowBytes := nblocks * wBlockBytes

	// Precompute block scales (shared across workers).
	dAScales := make([]float32, M*nblocks)
	for m := range M {
		for kb := range nblocks {
			off := m*K + kb*QK_K
			amax := ggufAsm.ComputeAbsmax(unsafe.Pointer(&input[off]), int64(QK_K))
			dAScales[m*nblocks+kb] = amax / 127.0
		}
	}

	paddedM := alignUp(M, smeTileSize)
	nTiles := alignUp(N, smeTileSize) / smeTileSize

	pool.ParallelFor(nTiles, func(tStart, tEnd int) {
		subBlockSize := info.SubBlockSize
		kGroups := subBlockSize / 4
		panelSize := kGroups * 64
		aPanel := make([]int8, panelSize)

		tileI32 := getPoolSliceI32(&smopaTilePool, smeTileSize*smeTileSize)
		defer smopaTilePool.Put(tileI32)

		accTile := getPoolSliceF32(&smopaAccPool, paddedM*smeTileSize)
		defer smopaAccPool.Put(accTile)

		bPanelBuf := getPoolSlice(&smopaPanelPool, panelSize)
		defer smopaPanelPool.Put(bPanelBuf)

		var signedBuf [32]int8
		var unsignedBuf [32]uint8
		var invScaleBuf [smeTileSize]float32
		var bsumBuf [smeTileSize]int64

		defer hwy.SMEGuard()()

		for t := tStart; t < tEnd; t++ {
			nTile := t * smeTileSize
			nCols := min(smeTileSize, N-nTile)

			clear(accTile)

			for kb := range nblocks {
				var metas [smeTileSize]blockMeta
				for col := range nCols {
					n := nTile + col
					wBlock := weights[n*wRowBytes+kb*wBlockBytes : n*wRowBytes+(kb+1)*wBlockBytes]
					parseBlockMeta(qt, wBlock, &metas[col])
				}

				for j := range info.NumSubBlocks {
					clear(bPanelBuf)

					if info.Signed {
						bPanelSigned := byteSliceAsInt8(bPanelBuf)

						for col := range nCols {
							n := nTile + col
							wBlock := weights[n*wRowBytes+kb*wBlockBytes : n*wRowBytes+(kb+1)*wBlockBytes]
							extractSignedSubBlock(qt, wBlock, j, signedBuf[:subBlockSize])
							for k4 := range kGroups {
								for g := range 4 {
									bPanelSigned[k4*64+col*4+g] = signedBuf[k4*4+g]
								}
							}
						}

						for mTile := 0; mTile < paddedM; mTile += smeTileSize {
							mRows := min(smeTileSize, M-mTile)

							for row := range mRows {
								m := mTile + row
								d := dAScales[m*nblocks+kb]
								if d > 0 {
									invScaleBuf[row] = 1.0 / d
								} else {
									invScaleBuf[row] = 0
								}
							}

							inputOff := mTile*K + kb*QK_K + j*subBlockSize
							ggufAsm.FusedQuantizePack(
								unsafe.Pointer(&input[inputOff]),
								int64(K),
								unsafe.Pointer(&invScaleBuf[0]),
								int64(subBlockSize),
								int64(mRows),
								unsafe.Pointer(&aPanel[0]),
							)

							asm.TileSMOPAS8(aPanel, bPanelSigned, tileI32, kGroups)
							accumulateTileSigned(accTile, tileI32, metas[:], dAScales, nblocks,
								kb, mTile, M, nCols, j)
						}
					} else {
						bPanelUnsigned := bPanelBuf

						for col := range nCols {
							n := nTile + col
							wBlock := weights[n*wRowBytes+kb*wBlockBytes : n*wRowBytes+(kb+1)*wBlockBytes]
							extractUnsignedSubBlock(qt, wBlock, j, unsignedBuf[:subBlockSize])
							for k4 := range kGroups {
								for g := range 4 {
									bPanelUnsigned[k4*64+col*4+g] = unsignedBuf[k4*4+g]
								}
							}
						}

						for mTile := 0; mTile < paddedM; mTile += smeTileSize {
							mRows := min(smeTileSize, M-mTile)

							for row := range mRows {
								m := mTile + row
								d := dAScales[m*nblocks+kb]
								if d > 0 {
									invScaleBuf[row] = 1.0 / d
								} else {
									invScaleBuf[row] = 0
								}
							}

							inputOff := mTile*K + kb*QK_K + j*subBlockSize
							ggufAsm.FusedQuantizePackBsum(
								unsafe.Pointer(&input[inputOff]),
								int64(K),
								unsafe.Pointer(&invScaleBuf[0]),
								int64(subBlockSize),
								int64(mRows),
								unsafe.Pointer(&aPanel[0]),
								unsafe.Pointer(&bsumBuf[0]),
							)

							asm.TileSUMOPAS8U8(aPanel, bPanelUnsigned, tileI32, kGroups)
							accumulateTileUnsigned(accTile, tileI32, metas[:], dAScales, nblocks,
								kb, mTile, M, nCols, j, &bsumBuf)
						}
					}
				}
			}

			for m := range M {
				for col := range nCols {
					output[m*N+nTile+col] = accTile[m*smeTileSize+col]
				}
			}
		}
	})
}

