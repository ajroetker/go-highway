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
