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

import (
	"math"
	"testing"
)

// TestExtractQ4KSubBlock verifies that extracting nibbles and then
// computing d*sc*q - dmin*m matches DequantizeQ4K.
func TestExtractQ4KSubBlock(t *testing.T) {
	nblocks := 2
	wdata, wFloat := makeTestWeightData(TypeQ4_K, nblocks)

	for b := range nblocks {
		block := wdata[b*BlockSizeQ4K : (b+1)*BlockSizeQ4K]
		d := fp16LE(block[0], block[1])
		dmin := fp16LE(block[2], block[3])

		var scs, mns [8]float32
		unpackQ4KScaleMins(block[4:16], &scs, &mns)

		for j := range 8 {
			var dst [32]uint8
			extractQ4KSubBlock(block, j, dst[:])

			baseIdx := b*QK_K + j*32
			for i := range 32 {
				want := wFloat[baseIdx+i]
				got := d*scs[j]*float32(dst[i]) - dmin*mns[j]
				if math.Abs(float64(got-want)) > 1e-4 {
					t.Errorf("Q4_K block=%d sub=%d val=%d: got %f, want %f", b, j, i, got, want)
				}
			}
		}
	}
}

// TestExtractQ5KSubBlock verifies Q5_K extraction.
func TestExtractQ5KSubBlock(t *testing.T) {
	nblocks := 2
	wdata, wFloat := makeTestWeightData(TypeQ5_K, nblocks)

	for b := range nblocks {
		block := wdata[b*BlockSizeQ5K : (b+1)*BlockSizeQ5K]
		d := fp16LE(block[0], block[1])
		dmin := fp16LE(block[2], block[3])

		var scs, mns [8]float32
		unpackQ4KScaleMins(block[4:16], &scs, &mns)

		for j := range 8 {
			var dst [32]uint8
			extractQ5KSubBlock(block, j, dst[:])

			baseIdx := b*QK_K + j*32
			for i := range 32 {
				want := wFloat[baseIdx+i]
				got := d*scs[j]*float32(dst[i]) - dmin*mns[j]
				if math.Abs(float64(got-want)) > 1e-4 {
					t.Errorf("Q5_K block=%d sub=%d val=%d: got %f, want %f (q=%d)", b, j, i, got, want, dst[i])
				}
			}
		}
	}
}

// TestExtractQ2KSubBlock verifies Q2_K extraction.
func TestExtractQ2KSubBlock(t *testing.T) {
	nblocks := 2
	wdata, wFloat := makeTestWeightData(TypeQ2_K, nblocks)

	for b := range nblocks {
		block := wdata[b*BlockSizeQ2K : (b+1)*BlockSizeQ2K]
		d := fp16LE(block[16], block[17])
		dmin := fp16LE(block[18], block[19])

		for j := range 16 {
			var dst [16]uint8
			extractQ2KSubBlock(block, j, dst[:])

			sc, mn := extractQ2KScaleMin(block, j)

			baseIdx := b*QK_K + j*16
			for i := range 16 {
				want := wFloat[baseIdx+i]
				got := d*sc*float32(dst[i]) - dmin*mn
				if math.Abs(float64(got-want)) > 1e-4 {
					t.Errorf("Q2_K block=%d sub=%d val=%d: got %f, want %f", b, j, i, got, want)
				}
			}
		}
	}
}

// TestExtractQ6KSubBlock verifies Q6_K extraction.
func TestExtractQ6KSubBlock(t *testing.T) {
	nblocks := 2
	wdata, wFloat := makeTestWeightData(TypeQ6_K, nblocks)

	for b := range nblocks {
		block := wdata[b*BlockSizeQ6K : (b+1)*BlockSizeQ6K]
		d := fp16LE(block[208], block[209])
		sc := block[192:208]

		for j := range 16 {
			var dst [16]int8
			extractQ6KSubBlock(block, j, dst[:])

			scaleVal := d * float32(int8(sc[j]))

			baseIdx := b*QK_K + j*16
			for i := range 16 {
				want := wFloat[baseIdx+i]
				got := scaleVal * float32(dst[i])
				if math.Abs(float64(got-want)) > 1e-4 {
					t.Errorf("Q6_K block=%d sub=%d val=%d: got %f, want %f (q=%d)", b, j, i, got, want, dst[i])
				}
			}
		}
	}
}

// TestExtractQ3KSubBlock verifies Q3_K extraction.
func TestExtractQ3KSubBlock(t *testing.T) {
	nblocks := 2
	wdata, wFloat := makeTestWeightData(TypeQ3_K, nblocks)

	for b := range nblocks {
		block := wdata[b*BlockSizeQ3K : (b+1)*BlockSizeQ3K]
		d := fp16LE(block[108], block[109])
		scaleData := block[96:108]

		var rawScales [16]int
		unpackQ3KScales(scaleData, &rawScales)

		for j := range 16 {
			var dst [16]int8
			extractQ3KSubBlock(block, j, dst[:])

			scaleVal := d * float32(rawScales[j]-32)

			baseIdx := b*QK_K + j*16
			for i := range 16 {
				want := wFloat[baseIdx+i]
				got := scaleVal * float32(dst[i])
				if math.Abs(float64(got-want)) > 1e-4 {
					t.Errorf("Q3_K block=%d sub=%d val=%d: got %f, want %f (q=%d, scale=%f)", b, j, i, got, want, dst[i], scaleVal)
				}
			}
		}
	}
}
