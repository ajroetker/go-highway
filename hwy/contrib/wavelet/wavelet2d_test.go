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

package wavelet

import (
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/image"
)

// zeroPhase always returns phase (0, 0)
func zeroPhase(level int) (int, int) {
	return 0, 0
}

func TestSynthesize2D_53_RoundTrip(t *testing.T) {
	sizes := []struct {
		width, height int
	}{
		{8, 8},
		{16, 16},
		{17, 15},
		{32, 32},
		{64, 64},
	}

	for _, size := range sizes {
		for levels := 1; levels <= 3; levels++ {
			t.Run(size2DString(size.width, size.height, levels), func(t *testing.T) {
				// Create original image
				img := image.NewImage[int32](size.width, size.height)
				original := make([]int32, size.width*size.height)

				for y := 0; y < size.height; y++ {
					row := img.Row(y)
					for x := 0; x < size.width; x++ {
						val := int32(x*y + x - y)
						row[x] = val
						original[y*size.width+x] = val
					}
				}

				// Forward then inverse
				Analyze2D_53(img, levels, zeroPhase)
				Synthesize2D_53(img, levels, zeroPhase)

				// Verify exact match (5/3 is lossless)
				for y := 0; y < size.height; y++ {
					row := img.Row(y)
					for x := 0; x < size.width; x++ {
						expected := original[y*size.width+x]
						if row[x] != expected {
							t.Errorf("at (%d,%d): got %d, want %d", x, y, row[x], expected)
						}
					}
				}
			})
		}
	}
}

func TestSynthesize2D_97_RoundTrip_Float32(t *testing.T) {
	sizes := []struct {
		width, height int
	}{
		{8, 8},
		{16, 16},
		{17, 15},
		{32, 32},
	}

	for _, size := range sizes {
		for levels := 1; levels <= 3; levels++ {
			t.Run(size2DString(size.width, size.height, levels), func(t *testing.T) {
				// Create original image
				img := image.NewImage[float32](size.width, size.height)
				original := make([]float32, size.width*size.height)

				for y := 0; y < size.height; y++ {
					row := img.Row(y)
					for x := 0; x < size.width; x++ {
						val := float32(x*y+x-y) / float32(size.width*size.height)
						row[x] = val
						original[y*size.width+x] = val
					}
				}

				// Forward then inverse
				Analyze2D_97(img, levels, zeroPhase)
				Synthesize2D_97(img, levels, zeroPhase)

				// Verify approximate match (float32 precision)
				for y := 0; y < size.height; y++ {
					row := img.Row(y)
					for x := 0; x < size.width; x++ {
						expected := original[y*size.width+x]
						if !almostEqualF32(row[x], expected, 1e-4) {
							t.Errorf("at (%d,%d): got %v, want %v", x, y, row[x], expected)
						}
					}
				}
			})
		}
	}
}

func TestSynthesize2D_97_RoundTrip_Float64(t *testing.T) {
	sizes := []struct {
		width, height int
	}{
		{8, 8},
		{16, 16},
		{17, 15},
		{32, 32},
	}

	for _, size := range sizes {
		for levels := 1; levels <= 3; levels++ {
			t.Run(size2DString(size.width, size.height, levels), func(t *testing.T) {
				// Create original image
				img := image.NewImage[float64](size.width, size.height)
				original := make([]float64, size.width*size.height)

				for y := 0; y < size.height; y++ {
					row := img.Row(y)
					for x := 0; x < size.width; x++ {
						val := float64(x*y+x-y) / float64(size.width*size.height)
						row[x] = val
						original[y*size.width+x] = val
					}
				}

				// Forward then inverse
				Analyze2D_97(img, levels, zeroPhase)
				Synthesize2D_97(img, levels, zeroPhase)

				// Verify very close match (float64 precision)
				for y := 0; y < size.height; y++ {
					row := img.Row(y)
					for x := 0; x < size.width; x++ {
						expected := original[y*size.width+x]
						if !almostEqualF64(row[x], expected, 1e-10) {
							t.Errorf("at (%d,%d): got %v, want %v", x, y, row[x], expected)
						}
					}
				}
			})
		}
	}
}

func TestNil2DImages(t *testing.T) {
	// Verify nil images don't panic
	Synthesize2D_53(nil, 3, zeroPhase)
	Analyze2D_53(nil, 3, zeroPhase)

	var nilImg *image.Image[float32]
	Synthesize2D_97(nilImg, 3, zeroPhase)
	Analyze2D_97(nilImg, 3, zeroPhase)

	// Empty images
	empty := image.NewImage[int32](0, 0)
	Synthesize2D_53(empty, 3, zeroPhase)
	Analyze2D_53(empty, 3, zeroPhase)
}

func TestLevelDim(t *testing.T) {
	tests := []struct {
		dim, level int
		expected   int
	}{
		{64, 0, 64},
		{64, 1, 32},
		{64, 2, 16},
		{64, 3, 8},
		{63, 1, 32},
		{63, 2, 16},
		{17, 1, 9},
		{17, 2, 5},
		{1, 1, 1},
	}

	for _, tc := range tests {
		result := levelDim(tc.dim, tc.level)
		if result != tc.expected {
			t.Errorf("levelDim(%d, %d) = %d, want %d", tc.dim, tc.level, result, tc.expected)
		}
	}
}

func size2DString(width, height, levels int) string {
	switch {
	case width <= 16:
		return "small"
	case width <= 32:
		return "medium"
	default:
		return "large"
	}
}
