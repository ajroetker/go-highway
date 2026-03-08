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

import "fmt"

// PreparedWeights holds GGUF K-quant weights pre-packed into SMOPA panel layout.
// Created once at model load time via PrepareWeights, then reused for all
// inference calls via PreparedGGUFMatMul. The pre-pack eliminates per-inference
// nibble extraction and scatter overhead.
//
// Memory expansion is ~2.2x for Q4_K (320 packed bytes vs 144 raw per column
// per K-block). For a 7B Q4_K model: ~7.7 GB packed vs ~3.5 GB raw.
type PreparedWeights struct {
	// Panels holds pre-packed B panels in SMOPA layout.
	// Organization: [nTileGroup][kBlock][subBlock][k4Group][tile(0-3) * 64 bytes]
	//
	// For each k4 group, 4 tiles' panels are contiguous (256 bytes total),
	// enabling svld1_vnum_u8(pg8, base, 0/1/2/3) group loads.
	//
	// Each 64-byte panel vector: panel[col * 4 + g] for col in [0,16), g in [0,4).
	Panels []byte

	// Scales holds pre-computed float32 scale and min products.
	//
	// For unsigned types (Q4_K, Q5_K, Q2_K):
	//   scaleVecs: [nTileGroup][kBlock][subBlock * 64] = d[col]*sc[col][j]
	//   minVecs:   [nTileGroup][kBlock][subBlock * 64] = dmin[col]*mn[col][j]
	//   (64 = 4 tiles * 16 cols, contiguous for svld1_vnum_f32 group loads)
	//   minVecs follow scaleVecs within each K-block.
	//
	// For signed types (Q6_K, Q3_K):
	//   scaleVecs only, no minVecs.
	Scales []float32

	K, N int
	QT   QuantType

	// Layout metadata.
	NTileGroups  int // ceil(N / 64)
	PaddedN      int // NTileGroups * 64 (N rounded up to multiple of 64)
	NumSubBlocks int // 8 (Q4_K/Q5_K) or 16 (Q6_K/Q2_K/Q3_K)
	SubBlockSize int // 32 (Q4_K/Q5_K) or 16 (Q6_K/Q2_K/Q3_K)
	KGroups      int // SubBlockSize / 4
	Signed       bool

	// Strides for indexing.
	PanelNGroupStride int // bytes between N-tile-groups in Panels
	PanelKBlockStride int // bytes between K-blocks within an N-tile-group
	PanelSubBlkStride int // bytes between sub-blocks within a K-block
	ScaleNGroupStride int // float32 offset between N-tile-groups in Scales
	ScaleKBlockStride int // float32 offset between K-blocks
}

const nTilesPerGroup = 4 // 4 N-tiles per group (64 columns)

// PrepareWeights converts raw GGUF K-quant weight data into pre-packed
// SMOPA panel layout for efficient 4-tile kernel execution.
//
// This is a one-time cost at model load time. The returned PreparedWeights
// can be reused for all inference calls via PreparedGGUFMatMul.
//
// Parameters:
//   - weights: [N, K] raw GGUF-quantized weight data
//   - K: hidden dimension (must be multiple of QK_K = 256)
//   - N: output dimension (number of weight rows)
//   - qt: quantization type (must be a K-quant type)
//
// Returns nil if qt is not a K-quant type.
func PrepareWeights(weights []uint8, K, N int, qt QuantType) *PreparedWeights {
	if !isKQuant(qt) {
		return nil
	}

	info := GetSubBlockInfo(qt)
	if info.NumSubBlocks == 0 {
		return nil
	}

	nblocks := K / QK_K
	wBlockBytes := BytesPerBlock(qt)
	wRowBytes := nblocks * wBlockBytes

	nTileGroups := (N + 63) / 64
	paddedN := nTileGroups * 64
	kGroups := info.SubBlockSize / 4

	// Panel layout: per k4 group, 4 tiles × 64 bytes = 256 bytes.
	panelSubBlkStride := kGroups * 256 // bytes per sub-block
	panelKBlockStride := info.NumSubBlocks * panelSubBlkStride
	panelNGroupStride := nblocks * panelKBlockStride
	totalPanelBytes := nTileGroups * panelNGroupStride

	// Scale layout: per sub-block, 4 tiles × 16 = 64 float32s.
	// For unsigned types: scaleVecs + minVecs = 2 * numSubBlocks * 64.
	// For signed types: scaleVecs only = numSubBlocks * 64.
	scalesPerSubBlock := 64 // 4 tiles * 16 cols
	var scaleKBlockStride int
	if info.Signed {
		scaleKBlockStride = info.NumSubBlocks * scalesPerSubBlock
	} else {
		scaleKBlockStride = 2 * info.NumSubBlocks * scalesPerSubBlock // scale + min
	}
	scaleNGroupStride := nblocks * scaleKBlockStride
	totalScaleF32s := nTileGroups * scaleNGroupStride

	pw := &PreparedWeights{
		Panels:            make([]byte, totalPanelBytes),
		Scales:            make([]float32, totalScaleF32s),
		K:                 K,
		N:                 N,
		QT:                qt,
		NTileGroups:       nTileGroups,
		PaddedN:           paddedN,
		NumSubBlocks:      info.NumSubBlocks,
		SubBlockSize:      info.SubBlockSize,
		KGroups:           kGroups,
		Signed:            info.Signed,
		PanelNGroupStride: panelNGroupStride,
		PanelKBlockStride: panelKBlockStride,
		PanelSubBlkStride: panelSubBlkStride,
		ScaleNGroupStride: scaleNGroupStride,
		ScaleKBlockStride: scaleKBlockStride,
	}

	// Pack panels and scales.
	var unsignedBuf [32]uint8
	var signedBuf [32]int8

	for ng := range nTileGroups {
		nBase := ng * 64 // first column in this N-tile-group

		for kb := range nblocks {
			panelKBOff := ng*panelNGroupStride + kb*panelKBlockStride
			scaleKBOff := ng*scaleNGroupStride + kb*scaleKBlockStride

			// For each of the 4 tiles in this group, parse block metadata.
			var metas [64]blockMeta // up to 64 columns
			for tile := range nTilesPerGroup {
				for col := range 16 {
					n := nBase + tile*16 + col
					if n >= N {
						continue
					}
					wBlock := weights[n*wRowBytes+kb*wBlockBytes : n*wRowBytes+(kb+1)*wBlockBytes]
					parseBlockMeta(qt, wBlock, &metas[tile*16+col])
				}
			}

			for j := range info.NumSubBlocks {
				panelSubOff := panelKBOff + j*panelSubBlkStride
				scaleSubOff := scaleKBOff + j*scalesPerSubBlock

				// Pack scales.
				for tile := range nTilesPerGroup {
					for col := range 16 {
						n := nBase + tile*16 + col
						idx := tile*16 + col
						if n < N {
							pw.Scales[scaleSubOff+idx] = metas[idx].d * metas[idx].scs[j]
						}
					}
				}

				// Pack mins for unsigned types.
				if !info.Signed {
					minSubOff := scaleKBOff + info.NumSubBlocks*scalesPerSubBlock + j*scalesPerSubBlock
					for tile := range nTilesPerGroup {
						for col := range 16 {
							n := nBase + tile*16 + col
							idx := tile*16 + col
							if n < N {
								pw.Scales[minSubOff+idx] = metas[idx].dmin * metas[idx].mns[j]
							}
						}
					}
				}

				// Pack B panels: extract quants and scatter into panel layout.
				for tile := range nTilesPerGroup {
					for col := range 16 {
						n := nBase + tile*16 + col
						if n >= N {
							continue
						}
						wBlock := weights[n*wRowBytes+kb*wBlockBytes : n*wRowBytes+(kb+1)*wBlockBytes]

						if info.Signed {
							extractSignedSubBlock(qt, wBlock, j, signedBuf[:info.SubBlockSize])
							for k4 := range kGroups {
								for g := range 4 {
									pw.Panels[panelSubOff+k4*256+tile*64+col*4+g] = byte(signedBuf[k4*4+g])
								}
							}
						} else {
							extractUnsignedSubBlock(qt, wBlock, j, unsignedBuf[:info.SubBlockSize])
							for k4 := range kGroups {
								for g := range 4 {
									pw.Panels[panelSubOff+k4*256+tile*64+col*4+g] = unsignedBuf[k4*4+g]
								}
							}
						}
					}
				}
			}
		}
	}

	return pw
}

// PanelOffset returns the byte offset into Panels for a given N-tile-group,
// K-block, sub-block, and k4 group.
func (pw *PreparedWeights) PanelOffset(nGroup, kb, j int) int {
	return nGroup*pw.PanelNGroupStride + kb*pw.PanelKBlockStride + j*pw.PanelSubBlkStride
}

// ScaleOffset returns the float32 offset into Scales for a given N-tile-group
// and K-block's scaleVecs start.
func (pw *PreparedWeights) ScaleOffset(nGroup, kb int) int {
	return nGroup*pw.ScaleNGroupStride + kb*pw.ScaleKBlockStride
}

// MinOffset returns the float32 offset into Scales for a given N-tile-group
// and K-block's minVecs start. Only valid for unsigned types.
func (pw *PreparedWeights) MinOffset(nGroup, kb int) int {
	if pw.Signed {
		panic(fmt.Sprintf("gguf: MinOffset called on signed type %d", pw.QT))
	}
	return nGroup*pw.ScaleNGroupStride + kb*pw.ScaleKBlockStride + pw.NumSubBlocks*64
}
