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

package image

import (
	"math"
	"testing"
)

// Test widths to cover aligned, unaligned, and tail cases
var testWidths = []int{1, 7, 8, 15, 16, 17, 100, 1024}

func TestForwardRCT(t *testing.T) {
	for _, width := range testWidths {
		t.Run(widthName(width), func(t *testing.T) {
			height := 4
			r := NewImage[int32](width, height)
			g := NewImage[int32](width, height)
			b := NewImage[int32](width, height)
			outY := NewImage[int32](width, height)
			outCb := NewImage[int32](width, height)
			outCr := NewImage[int32](width, height)

			// Fill with test data
			for y := 0; y < height; y++ {
				rRow := r.Row(y)
				gRow := g.Row(y)
				bRow := b.Row(y)
				for x := 0; x < width; x++ {
					rRow[x] = int32((x + y*width) % 256)
					gRow[x] = int32((x + y*width + 85) % 256)
					bRow[x] = int32((x + y*width + 170) % 256)
				}
			}

			// Apply forward RCT
			ForwardRCT(r, g, b, outY, outCb, outCr)

			// Verify results manually
			for y := 0; y < height; y++ {
				rRow := r.Row(y)
				gRow := g.Row(y)
				bRow := b.Row(y)
				yRow := outY.Row(y)
				cbRow := outCb.Row(y)
				crRow := outCr.Row(y)
				for x := 0; x < width; x++ {
					expectedY := (rRow[x] + 2*gRow[x] + bRow[x]) >> 2
					expectedCb := bRow[x] - gRow[x]
					expectedCr := rRow[x] - gRow[x]

					if yRow[x] != expectedY {
						t.Errorf("Y at (%d,%d): got %d, want %d", x, y, yRow[x], expectedY)
					}
					if cbRow[x] != expectedCb {
						t.Errorf("Cb at (%d,%d): got %d, want %d", x, y, cbRow[x], expectedCb)
					}
					if crRow[x] != expectedCr {
						t.Errorf("Cr at (%d,%d): got %d, want %d", x, y, crRow[x], expectedCr)
					}
				}
			}
		})
	}
}

func TestInverseRCT(t *testing.T) {
	for _, width := range testWidths {
		t.Run(widthName(width), func(t *testing.T) {
			height := 4
			yImg := NewImage[int32](width, height)
			cb := NewImage[int32](width, height)
			cr := NewImage[int32](width, height)
			outR := NewImage[int32](width, height)
			outG := NewImage[int32](width, height)
			outB := NewImage[int32](width, height)

			// Fill with test data (YCbCr values)
			for y := 0; y < height; y++ {
				yRow := yImg.Row(y)
				cbRow := cb.Row(y)
				crRow := cr.Row(y)
				for x := 0; x < width; x++ {
					yRow[x] = int32(128 + (x+y*width)%64)
					cbRow[x] = int32((x + y*width) % 128 - 64)
					crRow[x] = int32((x + y*width + 32) % 128 - 64)
				}
			}

			// Apply inverse RCT
			InverseRCT(yImg, cb, cr, outR, outG, outB)

			// Verify results manually
			for y := 0; y < height; y++ {
				yRow := yImg.Row(y)
				cbRow := cb.Row(y)
				crRow := cr.Row(y)
				rRow := outR.Row(y)
				gRow := outG.Row(y)
				bRow := outB.Row(y)
				for x := 0; x < width; x++ {
					expectedG := yRow[x] - ((cbRow[x] + crRow[x]) >> 2)
					expectedR := crRow[x] + expectedG
					expectedB := cbRow[x] + expectedG

					if rRow[x] != expectedR {
						t.Errorf("R at (%d,%d): got %d, want %d", x, y, rRow[x], expectedR)
					}
					if gRow[x] != expectedG {
						t.Errorf("G at (%d,%d): got %d, want %d", x, y, gRow[x], expectedG)
					}
					if bRow[x] != expectedB {
						t.Errorf("B at (%d,%d): got %d, want %d", x, y, bRow[x], expectedB)
					}
				}
			}
		})
	}
}

func TestRCTRoundTrip(t *testing.T) {
	for _, width := range testWidths {
		t.Run(widthName(width), func(t *testing.T) {
			height := 4

			// Original RGB
			origR := NewImage[int32](width, height)
			origG := NewImage[int32](width, height)
			origB := NewImage[int32](width, height)

			// YCbCr intermediate
			yImg := NewImage[int32](width, height)
			cb := NewImage[int32](width, height)
			cr := NewImage[int32](width, height)

			// Reconstructed RGB
			recR := NewImage[int32](width, height)
			recG := NewImage[int32](width, height)
			recB := NewImage[int32](width, height)

			// Fill with test data
			for y := 0; y < height; y++ {
				rRow := origR.Row(y)
				gRow := origG.Row(y)
				bRow := origB.Row(y)
				for x := 0; x < width; x++ {
					rRow[x] = int32((x + y*width) % 256)
					gRow[x] = int32((x + y*width + 85) % 256)
					bRow[x] = int32((x + y*width + 170) % 256)
				}
			}

			// Forward + Inverse should give exact reconstruction
			ForwardRCT(origR, origG, origB, yImg, cb, cr)
			InverseRCT(yImg, cb, cr, recR, recG, recB)

			// Verify exact match (RCT is lossless)
			for y := 0; y < height; y++ {
				origRRow := origR.Row(y)
				origGRow := origG.Row(y)
				origBRow := origB.Row(y)
				recRRow := recR.Row(y)
				recGRow := recG.Row(y)
				recBRow := recB.Row(y)
				for x := 0; x < width; x++ {
					if origRRow[x] != recRRow[x] {
						t.Errorf("R at (%d,%d): orig %d != rec %d", x, y, origRRow[x], recRRow[x])
					}
					if origGRow[x] != recGRow[x] {
						t.Errorf("G at (%d,%d): orig %d != rec %d", x, y, origGRow[x], recGRow[x])
					}
					if origBRow[x] != recBRow[x] {
						t.Errorf("B at (%d,%d): orig %d != rec %d", x, y, origBRow[x], recBRow[x])
					}
				}
			}
		})
	}
}

func TestForwardICT_Float32(t *testing.T) {
	for _, width := range testWidths {
		t.Run(widthName(width), func(t *testing.T) {
			height := 4
			r := NewImage[float32](width, height)
			g := NewImage[float32](width, height)
			b := NewImage[float32](width, height)
			outY := NewImage[float32](width, height)
			outCb := NewImage[float32](width, height)
			outCr := NewImage[float32](width, height)

			// Fill with test data (normalized [0, 1])
			for y := 0; y < height; y++ {
				rRow := r.Row(y)
				gRow := g.Row(y)
				bRow := b.Row(y)
				for x := 0; x < width; x++ {
					rRow[x] = float32(x+y*width) / float32(width*height)
					gRow[x] = float32((x+y*width)+width*height/3) / float32(width*height*2)
					bRow[x] = float32((x+y*width)+width*height*2/3) / float32(width*height*2)
				}
			}

			// Apply forward ICT
			ForwardICT(r, g, b, outY, outCb, outCr)

			// Verify results manually
			for y := 0; y < height; y++ {
				rRow := r.Row(y)
				gRow := g.Row(y)
				bRow := b.Row(y)
				yRow := outY.Row(y)
				cbRow := outCb.Row(y)
				crRow := outCr.Row(y)
				for x := 0; x < width; x++ {
					expectedY := float32(ICT_RtoY)*rRow[x] + float32(ICT_GtoY)*gRow[x] + float32(ICT_BtoY)*bRow[x]
					expectedCb := float32(ICT_RtoCb)*rRow[x] + float32(ICT_GtoCb)*gRow[x] + float32(ICT_BtoCb)*bRow[x]
					expectedCr := float32(ICT_RtoCr)*rRow[x] + float32(ICT_GtoCr)*gRow[x] + float32(ICT_BtoCr)*bRow[x]

					if !almostEqual(yRow[x], expectedY, 1e-5) {
						t.Errorf("Y at (%d,%d): got %v, want %v", x, y, yRow[x], expectedY)
					}
					if !almostEqual(cbRow[x], expectedCb, 1e-5) {
						t.Errorf("Cb at (%d,%d): got %v, want %v", x, y, cbRow[x], expectedCb)
					}
					if !almostEqual(crRow[x], expectedCr, 1e-5) {
						t.Errorf("Cr at (%d,%d): got %v, want %v", x, y, crRow[x], expectedCr)
					}
				}
			}
		})
	}
}

func TestForwardICT_Float64(t *testing.T) {
	for _, width := range testWidths {
		t.Run(widthName(width), func(t *testing.T) {
			height := 4
			r := NewImage[float64](width, height)
			g := NewImage[float64](width, height)
			b := NewImage[float64](width, height)
			outY := NewImage[float64](width, height)
			outCb := NewImage[float64](width, height)
			outCr := NewImage[float64](width, height)

			// Fill with test data
			for y := 0; y < height; y++ {
				rRow := r.Row(y)
				gRow := g.Row(y)
				bRow := b.Row(y)
				for x := 0; x < width; x++ {
					rRow[x] = float64(x+y*width) / float64(width*height)
					gRow[x] = float64((x+y*width)+width*height/3) / float64(width*height*2)
					bRow[x] = float64((x+y*width)+width*height*2/3) / float64(width*height*2)
				}
			}

			// Apply forward ICT
			ForwardICT(r, g, b, outY, outCb, outCr)

			// Verify results manually
			for y := 0; y < height; y++ {
				rRow := r.Row(y)
				gRow := g.Row(y)
				bRow := b.Row(y)
				yRow := outY.Row(y)
				cbRow := outCb.Row(y)
				crRow := outCr.Row(y)
				for x := 0; x < width; x++ {
					expectedY := ICT_RtoY*rRow[x] + ICT_GtoY*gRow[x] + ICT_BtoY*bRow[x]
					expectedCb := ICT_RtoCb*rRow[x] + ICT_GtoCb*gRow[x] + ICT_BtoCb*bRow[x]
					expectedCr := ICT_RtoCr*rRow[x] + ICT_GtoCr*gRow[x] + ICT_BtoCr*bRow[x]

					if !almostEqualF64(yRow[x], expectedY, 1e-10) {
						t.Errorf("Y at (%d,%d): got %v, want %v", x, y, yRow[x], expectedY)
					}
					if !almostEqualF64(cbRow[x], expectedCb, 1e-10) {
						t.Errorf("Cb at (%d,%d): got %v, want %v", x, y, cbRow[x], expectedCb)
					}
					if !almostEqualF64(crRow[x], expectedCr, 1e-10) {
						t.Errorf("Cr at (%d,%d): got %v, want %v", x, y, crRow[x], expectedCr)
					}
				}
			}
		})
	}
}

func TestICTRoundTrip_Float32(t *testing.T) {
	for _, width := range testWidths {
		t.Run(widthName(width), func(t *testing.T) {
			height := 4

			// Original RGB
			origR := NewImage[float32](width, height)
			origG := NewImage[float32](width, height)
			origB := NewImage[float32](width, height)

			// YCbCr intermediate
			yImg := NewImage[float32](width, height)
			cb := NewImage[float32](width, height)
			cr := NewImage[float32](width, height)

			// Reconstructed RGB
			recR := NewImage[float32](width, height)
			recG := NewImage[float32](width, height)
			recB := NewImage[float32](width, height)

			// Fill with test data
			for y := 0; y < height; y++ {
				rRow := origR.Row(y)
				gRow := origG.Row(y)
				bRow := origB.Row(y)
				for x := 0; x < width; x++ {
					rRow[x] = float32(x+y*width) / float32(width*height)
					gRow[x] = float32((x+y*width)+width*height/3) / float32(width*height*2)
					bRow[x] = float32((x+y*width)+width*height*2/3) / float32(width*height*2)
				}
			}

			// Forward + Inverse
			ForwardICT(origR, origG, origB, yImg, cb, cr)
			InverseICT(yImg, cb, cr, recR, recG, recB)

			// Verify approximate match (ICT is lossy due to float precision)
			// float32 has ~7 decimal digits of precision, so use 1e-4 tolerance
			for y := 0; y < height; y++ {
				origRRow := origR.Row(y)
				origGRow := origG.Row(y)
				origBRow := origB.Row(y)
				recRRow := recR.Row(y)
				recGRow := recG.Row(y)
				recBRow := recB.Row(y)
				for x := 0; x < width; x++ {
					if !almostEqual(origRRow[x], recRRow[x], 1e-4) {
						t.Errorf("R at (%d,%d): orig %v != rec %v", x, y, origRRow[x], recRRow[x])
					}
					if !almostEqual(origGRow[x], recGRow[x], 1e-4) {
						t.Errorf("G at (%d,%d): orig %v != rec %v", x, y, origGRow[x], recGRow[x])
					}
					if !almostEqual(origBRow[x], recBRow[x], 1e-4) {
						t.Errorf("B at (%d,%d): orig %v != rec %v", x, y, origBRow[x], recBRow[x])
					}
				}
			}
		})
	}
}

func TestICTRoundTrip_Float64(t *testing.T) {
	for _, width := range testWidths {
		t.Run(widthName(width), func(t *testing.T) {
			height := 4

			// Original RGB
			origR := NewImage[float64](width, height)
			origG := NewImage[float64](width, height)
			origB := NewImage[float64](width, height)

			// YCbCr intermediate
			yImg := NewImage[float64](width, height)
			cb := NewImage[float64](width, height)
			cr := NewImage[float64](width, height)

			// Reconstructed RGB
			recR := NewImage[float64](width, height)
			recG := NewImage[float64](width, height)
			recB := NewImage[float64](width, height)

			// Fill with test data
			for y := 0; y < height; y++ {
				rRow := origR.Row(y)
				gRow := origG.Row(y)
				bRow := origB.Row(y)
				for x := 0; x < width; x++ {
					rRow[x] = float64(x+y*width) / float64(width*height)
					gRow[x] = float64((x+y*width)+width*height/3) / float64(width*height*2)
					bRow[x] = float64((x+y*width)+width*height*2/3) / float64(width*height*2)
				}
			}

			// Forward + Inverse
			ForwardICT(origR, origG, origB, yImg, cb, cr)
			InverseICT(yImg, cb, cr, recR, recG, recB)

			// Verify close match (float64 has higher precision, but ICT/inverse ICT
			// coefficients are not an exact inverse matrix, so there's inherent error)
			for y := 0; y < height; y++ {
				origRRow := origR.Row(y)
				origGRow := origG.Row(y)
				origBRow := origB.Row(y)
				recRRow := recR.Row(y)
				recGRow := recG.Row(y)
				recBRow := recB.Row(y)
				for x := 0; x < width; x++ {
					if !almostEqualF64(origRRow[x], recRRow[x], 1e-4) {
						t.Errorf("R at (%d,%d): orig %v != rec %v", x, y, origRRow[x], recRRow[x])
					}
					if !almostEqualF64(origGRow[x], recGRow[x], 1e-4) {
						t.Errorf("G at (%d,%d): orig %v != rec %v", x, y, origGRow[x], recGRow[x])
					}
					if !almostEqualF64(origBRow[x], recBRow[x], 1e-4) {
						t.Errorf("B at (%d,%d): orig %v != rec %v", x, y, origBRow[x], recBRow[x])
					}
				}
			}
		})
	}
}

func TestNilColorImages(t *testing.T) {
	// Test that nil inputs don't panic
	var nilImg *Image[int32]
	out := NewImage[int32](8, 8)

	// These should all return without panicking
	ForwardRCT(nilImg, out, out, out, out, out)
	ForwardRCT(out, nilImg, out, out, out, out)
	ForwardRCT(out, out, nilImg, out, out, out)
	ForwardRCT(out, out, out, nilImg, out, out)
	ForwardRCT(out, out, out, out, nilImg, out)
	ForwardRCT(out, out, out, out, out, nilImg)

	InverseRCT(nilImg, out, out, out, out, out)
	InverseRCT(out, nilImg, out, out, out, out)
	InverseRCT(out, out, nilImg, out, out, out)
	InverseRCT(out, out, out, nilImg, out, out)
	InverseRCT(out, out, out, out, nilImg, out)
	InverseRCT(out, out, out, out, out, nilImg)

	var nilImgF *Image[float32]
	outF := NewImage[float32](8, 8)

	ForwardICT(nilImgF, outF, outF, outF, outF, outF)
	ForwardICT(outF, nilImgF, outF, outF, outF, outF)
	ForwardICT(outF, outF, nilImgF, outF, outF, outF)
	ForwardICT(outF, outF, outF, nilImgF, outF, outF)
	ForwardICT(outF, outF, outF, outF, nilImgF, outF)
	ForwardICT(outF, outF, outF, outF, outF, nilImgF)

	InverseICT(nilImgF, outF, outF, outF, outF, outF)
	InverseICT(outF, nilImgF, outF, outF, outF, outF)
	InverseICT(outF, outF, nilImgF, outF, outF, outF)
	InverseICT(outF, outF, outF, nilImgF, outF, outF)
	InverseICT(outF, outF, outF, outF, nilImgF, outF)
	InverseICT(outF, outF, outF, outF, outF, nilImgF)
}

func TestRCTInt64(t *testing.T) {
	// Test that RCT works with int64 as well
	width, height := 32, 4
	r := NewImage[int64](width, height)
	g := NewImage[int64](width, height)
	b := NewImage[int64](width, height)
	yImg := NewImage[int64](width, height)
	cb := NewImage[int64](width, height)
	cr := NewImage[int64](width, height)
	recR := NewImage[int64](width, height)
	recG := NewImage[int64](width, height)
	recB := NewImage[int64](width, height)

	// Fill with test data
	for y := 0; y < height; y++ {
		rRow := r.Row(y)
		gRow := g.Row(y)
		bRow := b.Row(y)
		for x := 0; x < width; x++ {
			rRow[x] = int64((x + y*width) % 256)
			gRow[x] = int64((x + y*width + 85) % 256)
			bRow[x] = int64((x + y*width + 170) % 256)
		}
	}

	// Round-trip
	ForwardRCT(r, g, b, yImg, cb, cr)
	InverseRCT(yImg, cb, cr, recR, recG, recB)

	// Verify exact match
	for y := 0; y < height; y++ {
		rRow := r.Row(y)
		gRow := g.Row(y)
		bRow := b.Row(y)
		recRRow := recR.Row(y)
		recGRow := recG.Row(y)
		recBRow := recB.Row(y)
		for x := 0; x < width; x++ {
			if rRow[x] != recRRow[x] || gRow[x] != recGRow[x] || bRow[x] != recBRow[x] {
				t.Errorf("Mismatch at (%d,%d): orig (%d,%d,%d) != rec (%d,%d,%d)",
					x, y, rRow[x], gRow[x], bRow[x], recRRow[x], recGRow[x], recBRow[x])
			}
		}
	}
}

// Helper functions

func widthName(w int) string {
	switch {
	case w == 1:
		return "single"
	case w < 8:
		return "small"
	case w == 8 || w == 16:
		return "aligned"
	case w < 100:
		return "unaligned"
	default:
		return "large"
	}
}

func almostEqualSlice(got, want []float32, tol float32) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > float64(tol) {
			return false
		}
	}
	return true
}
