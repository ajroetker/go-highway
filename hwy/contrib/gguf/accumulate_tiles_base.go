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

package gguf

//go:generate go run ../../../cmd/hwygen -input accumulate_tiles_base.go -output . -targets neon:asm,fallback -dispatch accumulatetiles

import (
	"github.com/ajroetker/go-highway/hwy"
)

// BaseAccumulateTilesUnsigned accumulates 4 tiles of int32 SMOPA output into
// a float32 accumulator with per-column scales and per-row activation scales.
// For unsigned quant types (Q4_K, Q5_K, Q2_K), also applies min correction.
//
// Tile layout: tiles[tile*256 + row*16 + col], 4 tiles of 16x16 int32s.
// Acc layout:  acc[(mTile+row)*64 + tile*16 + col], 64 columns per row.
// Scale layout: sc[tile*16 + col], 64 floats (4 tiles x 16 cols).
// Min layout:   mn[tile*16 + col], 64 floats (4 tiles x 16 cols).
//
// For each element: acc += float32(tile_int32) * sc * dA[row]
//                   acc -= mn * dABsum[row]
func BaseAccumulateTilesUnsigned(acc []float32, tiles []int32,
	sc []float32, mn []float32,
	dA []float32, dABsum []float32,
	mTile int, nRows int) {

	lanes := hwy.NumLanes[float32]()
	fbuf := make([]float32, lanes)

	for row := range nRows {
		dVec := hwy.Set[float32](dA[row])
		bsVec := hwy.Set[float32](dABsum[row])
		accRowOff := (mTile + row) * 64

		for tile := range 4 {
			tileOff := tile*256 + row*16
			off := accRowOff + tile*16
			scOff := tile * 16

			i := 0
			for ; i+lanes <= 16; i += lanes {
				// Convert int32 tile values to float32 via scalar cast.
				for j := range lanes {
					fbuf[j] = float32(tiles[tileOff+i+j])
				}
				raw := hwy.Load(fbuf)

				// Load scales and accumulator.
				scVec := hwy.Load(sc[scOff+i : scOff+i+lanes])
				aVec := hwy.Load(acc[off+i : off+i+lanes])

				// acc += raw * scale * d
				aVec = hwy.MulAdd(hwy.Mul(raw, scVec), dVec, aVec)

				// acc -= mn * bs
				mnVec := hwy.Load(mn[scOff+i : scOff+i+lanes])
				aVec = hwy.Sub(aVec, hwy.Mul(mnVec, bsVec))

				hwy.Store(aVec, acc[off+i:off+i+lanes])
			}
			// Scalar tail for remaining elements.
			for ; i < 16; i++ {
				raw := float32(tiles[tileOff+i])
				acc[off+i] += raw * sc[scOff+i] * dA[row]
				acc[off+i] -= mn[scOff+i] * dABsum[row]
			}
		}
	}
}

// BaseAccumulateTilesSigned accumulates 4 tiles of int32 SMOPA output into
// a float32 accumulator with per-column scales and per-row activation scales.
// For signed quant types (Q6_K, Q3_K), no min correction is needed.
//
// For each element: acc += float32(tile_int32) * sc * dA[row]
func BaseAccumulateTilesSigned(acc []float32, tiles []int32,
	sc []float32,
	dA []float32,
	mTile int, nRows int) {

	lanes := hwy.NumLanes[float32]()
	fbuf := make([]float32, lanes)

	for row := range nRows {
		dVec := hwy.Set[float32](dA[row])
		accRowOff := (mTile + row) * 64

		for tile := range 4 {
			tileOff := tile*256 + row*16
			off := accRowOff + tile*16
			scOff := tile * 16

			i := 0
			for ; i+lanes <= 16; i += lanes {
				for j := range lanes {
					fbuf[j] = float32(tiles[tileOff+i+j])
				}
				raw := hwy.Load(fbuf)

				scVec := hwy.Load(sc[scOff+i : scOff+i+lanes])
				aVec := hwy.Load(acc[off+i : off+i+lanes])

				aVec = hwy.MulAdd(hwy.Mul(raw, scVec), dVec, aVec)

				hwy.Store(aVec, acc[off+i:off+i+lanes])
			}
			for ; i < 16; i++ {
				raw := float32(tiles[tileOff+i])
				acc[off+i] += raw * sc[scOff+i] * dA[row]
			}
		}
	}
}
