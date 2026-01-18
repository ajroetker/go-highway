package image

import (
	"math"
	"testing"
)

const tolerance = 1e-5

func almostEqual(a, b, tol float32) bool {
	return math.Abs(float64(a-b)) < float64(tol)
}

func almostEqualF64(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

func TestBrightnessContrast(t *testing.T) {
	tests := []struct {
		name   string
		width  int
		height int
		scale  float32
		offset float32
	}{
		{"small", 8, 4, 1.5, 0.1},
		{"exact_vector", 16, 1, 2.0, -0.5},
		{"with_tail", 17, 3, 0.5, 0.2},
		{"large", 100, 100, 1.2, 0.05},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			img := NewImage[float32](tc.width, tc.height)
			out := NewImage[float32](tc.width, tc.height)

			// Fill with test data
			for y := 0; y < tc.height; y++ {
				row := img.Row(y)
				for x := 0; x < tc.width; x++ {
					row[x] = float32(x+y) / float32(tc.width+tc.height)
				}
			}

			// Apply BrightnessContrast
			BrightnessContrast(img, out, tc.scale, tc.offset)

			// Verify
			for y := 0; y < tc.height; y++ {
				inRow := img.Row(y)
				outRow := out.Row(y)
				for x := 0; x < tc.width; x++ {
					expected := inRow[x]*tc.scale + tc.offset
					if !almostEqual(outRow[x], expected, tolerance) {
						t.Errorf("at (%d,%d): got %v, want %v", x, y, outRow[x], expected)
					}
				}
			}
		})
	}
}

func TestBrightnessContrastFloat64(t *testing.T) {
	img := NewImage[float64](32, 8)
	out := NewImage[float64](32, 8)

	scale, offset := 2.0, 0.25
	for y := range 8 {
		row := img.Row(y)
		for x := range 32 {
			row[x] = float64(x) / 32.0
		}
	}

	BrightnessContrast(img, out, scale, offset)

	for y := range 8 {
		inRow := img.Row(y)
		outRow := out.Row(y)
		for x := range 32 {
			expected := inRow[x]*scale + offset
			if !almostEqualF64(outRow[x], expected, 1e-10) {
				t.Errorf("at (%d,%d): got %v, want %v", x, y, outRow[x], expected)
			}
		}
	}
}

func TestClampImage(t *testing.T) {
	img := NewImage[float32](20, 5)
	out := NewImage[float32](20, 5)

	// Fill with values outside [0, 1]
	for y := range 5 {
		row := img.Row(y)
		for x := range 20 {
			row[x] = float32(x-10) / 5.0 // Range: -2 to +1.8
		}
	}

	ClampImage(img, out, 0.0, 1.0)

	for y := range 5 {
		inRow := img.Row(y)
		outRow := out.Row(y)
		for x := range 20 {
			expected := inRow[x]
			if expected < 0 {
				expected = 0
			} else if expected > 1 {
				expected = 1
			}
			if !almostEqual(outRow[x], expected, tolerance) {
				t.Errorf("at (%d,%d): got %v, want %v", x, y, outRow[x], expected)
			}
		}
	}
}

func TestThreshold(t *testing.T) {
	img := NewImage[float32](16, 4)
	out := NewImage[float32](16, 4)

	thresh, below, above := float32(0.5), float32(0.0), float32(1.0)

	for y := range 4 {
		row := img.Row(y)
		for x := range 16 {
			row[x] = float32(x) / 15.0
		}
	}

	Threshold(img, out, thresh, below, above)

	for y := range 4 {
		inRow := img.Row(y)
		outRow := out.Row(y)
		for x := range 16 {
			var expected float32
			if inRow[x] >= thresh {
				expected = above
			} else {
				expected = below
			}
			if outRow[x] != expected {
				t.Errorf("at (%d,%d): input=%v, got %v, want %v", x, y, inRow[x], outRow[x], expected)
			}
		}
	}
}

func TestInvert(t *testing.T) {
	img := NewImage[float32](24, 3)
	out := NewImage[float32](24, 3)

	maxVal := float32(1.0)

	for y := range 3 {
		row := img.Row(y)
		for x := range 24 {
			row[x] = float32(x) / 24.0
		}
	}

	Invert(img, out, maxVal)

	for y := range 3 {
		inRow := img.Row(y)
		outRow := out.Row(y)
		for x := range 24 {
			expected := maxVal - inRow[x]
			if !almostEqual(outRow[x], expected, tolerance) {
				t.Errorf("at (%d,%d): got %v, want %v", x, y, outRow[x], expected)
			}
		}
	}
}

func TestAbs(t *testing.T) {
	img := NewImage[float32](32, 2)
	out := NewImage[float32](32, 2)

	for y := range 2 {
		row := img.Row(y)
		for x := range 32 {
			row[x] = float32(x-16) / 8.0 // Range: -2 to +1.875
		}
	}

	Abs(img, out)

	for y := range 2 {
		inRow := img.Row(y)
		outRow := out.Row(y)
		for x := range 32 {
			expected := inRow[x]
			if expected < 0 {
				expected = -expected
			}
			if !almostEqual(outRow[x], expected, tolerance) {
				t.Errorf("at (%d,%d): got %v, want %v", x, y, outRow[x], expected)
			}
		}
	}
}

func TestScale(t *testing.T) {
	img := NewImage[float32](20, 4)
	out := NewImage[float32](20, 4)

	scale := float32(2.5)

	for y := range 4 {
		row := img.Row(y)
		for x := range 20 {
			row[x] = float32(x) / 20.0
		}
	}

	Scale(img, out, scale)

	for y := range 4 {
		inRow := img.Row(y)
		outRow := out.Row(y)
		for x := range 20 {
			expected := inRow[x] * scale
			if !almostEqual(outRow[x], expected, tolerance) {
				t.Errorf("at (%d,%d): got %v, want %v", x, y, outRow[x], expected)
			}
		}
	}
}

func TestOffset(t *testing.T) {
	img := NewImage[float32](24, 3)
	out := NewImage[float32](24, 3)

	offset := float32(0.25)

	for y := range 3 {
		row := img.Row(y)
		for x := range 24 {
			row[x] = float32(x) / 24.0
		}
	}

	Offset(img, out, offset)

	for y := range 3 {
		inRow := img.Row(y)
		outRow := out.Row(y)
		for x := range 24 {
			expected := inRow[x] + offset
			if !almostEqual(outRow[x], expected, tolerance) {
				t.Errorf("at (%d,%d): got %v, want %v", x, y, outRow[x], expected)
			}
		}
	}
}

func TestGamma(t *testing.T) {
	img := NewImage[float32](16, 4)
	out := NewImage[float32](16, 4)

	gamma := float32(2.2)

	for y := range 4 {
		row := img.Row(y)
		for x := range 16 {
			row[x] = float32(x+1) / 17.0 // Avoid 0 for pow
		}
	}

	Gamma(img, out, gamma)

	for y := range 4 {
		inRow := img.Row(y)
		outRow := out.Row(y)
		for x := range 16 {
			expected := float32(math.Pow(float64(inRow[x]), float64(gamma)))
			if !almostEqual(outRow[x], expected, tolerance) {
				t.Errorf("at (%d,%d): got %v, want %v", x, y, outRow[x], expected)
			}
		}
	}
}

func TestMinImage(t *testing.T) {
	a := NewImage[float32](20, 5)
	b := NewImage[float32](20, 5)
	out := NewImage[float32](20, 5)

	for y := range 5 {
		rowA := a.Row(y)
		rowB := b.Row(y)
		for x := range 20 {
			rowA[x] = float32(x) / 20.0
			rowB[x] = float32(19-x) / 20.0
		}
	}

	MinImage(a, b, out)

	for y := range 5 {
		rowA := a.Row(y)
		rowB := b.Row(y)
		rowOut := out.Row(y)
		for x := range 20 {
			expected := rowA[x]
			if rowB[x] < expected {
				expected = rowB[x]
			}
			if !almostEqual(rowOut[x], expected, tolerance) {
				t.Errorf("at (%d,%d): got %v, want %v", x, y, rowOut[x], expected)
			}
		}
	}
}

func TestMaxImage(t *testing.T) {
	a := NewImage[float32](24, 4)
	b := NewImage[float32](24, 4)
	out := NewImage[float32](24, 4)

	for y := range 4 {
		rowA := a.Row(y)
		rowB := b.Row(y)
		for x := range 24 {
			rowA[x] = float32(x) / 24.0
			rowB[x] = float32(23-x) / 24.0
		}
	}

	MaxImage(a, b, out)

	for y := range 4 {
		rowA := a.Row(y)
		rowB := b.Row(y)
		rowOut := out.Row(y)
		for x := range 24 {
			expected := rowA[x]
			if rowB[x] > expected {
				expected = rowB[x]
			}
			if !almostEqual(rowOut[x], expected, tolerance) {
				t.Errorf("at (%d,%d): got %v, want %v", x, y, rowOut[x], expected)
			}
		}
	}
}

func TestNilImages(t *testing.T) {
	// Test that nil inputs don't panic
	var nilImg *Image[float32]
	out := NewImage[float32](8, 8)

	// These should all return without panicking
	BrightnessContrast(nilImg, out, 1.0, 0.0)
	ClampImage(nilImg, out, 0.0, 1.0)
	Threshold(nilImg, out, 0.5, 0.0, 1.0)
	Invert(nilImg, out, 1.0)
	Abs(nilImg, out)
	Scale(nilImg, out, 2.0)
	Offset(nilImg, out, 0.5)
	Gamma(nilImg, out, 2.2)
	MinImage(nilImg, out, out)
	MaxImage(nilImg, out, out)
}

func TestInPlace(t *testing.T) {
	// Test that operations work in-place (same input and output)
	img := NewImage[float32](16, 4)

	for y := range 4 {
		row := img.Row(y)
		for x := range 16 {
			row[x] = float32(x) / 16.0
		}
	}

	// Store original values
	original := make([]float32, 16)
	copy(original, img.Row(0))

	// Apply in-place
	Scale(img, img, 2.0)

	// Verify
	row := img.Row(0)
	for x := range 16 {
		expected := original[x] * 2.0
		if !almostEqual(row[x], expected, tolerance) {
			t.Errorf("at x=%d: got %v, want %v", x, row[x], expected)
		}
	}
}
