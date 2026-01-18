package image

import (
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

func TestNewImage(t *testing.T) {
	img := NewImage[float32](100, 50)

	if img.Width() != 100 {
		t.Errorf("Width: got %d, want 100", img.Width())
	}
	if img.Height() != 50 {
		t.Errorf("Height: got %d, want 50", img.Height())
	}

	// Stride should be >= width and aligned to vector width
	lanes := hwy.MaxLanes[float32]()
	if img.Stride() < 100 {
		t.Errorf("Stride: got %d, want >= 100", img.Stride())
	}
	if img.Stride()%lanes != 0 {
		t.Errorf("Stride not aligned: got %d, want multiple of %d", img.Stride(), lanes)
	}
}

func TestNewImage_ZeroDimensions(t *testing.T) {
	img := NewImage[float32](0, 0)
	if img.Width() != 0 || img.Height() != 0 {
		t.Errorf("Zero dimensions: got %dx%d, want 0x0", img.Width(), img.Height())
	}

	img = NewImage[float32](-1, 10)
	if img.Width() != 0 || img.Height() != 0 {
		t.Errorf("Negative width: got %dx%d, want 0x0", img.Width(), img.Height())
	}
}

func TestImage_Row(t *testing.T) {
	img := NewImage[float32](10, 5)

	// Set values in first row
	row0 := img.Row(0)
	for i := range 10 {
		row0[i] = float32(i)
	}

	// Read back
	for i := range 10 {
		if row0[i] != float32(i) {
			t.Errorf("Row[0][%d]: got %v, want %v", i, row0[i], float32(i))
		}
	}

	// Different row should be independent
	row1 := img.Row(1)
	row1[0] = 999
	if row0[0] == 999 {
		t.Error("Rows should be independent")
	}

	// Out of bounds
	if img.Row(-1) != nil {
		t.Error("Row(-1) should return nil")
	}
	if img.Row(5) != nil {
		t.Error("Row(5) should return nil")
	}
}

func TestImage_RowSlice(t *testing.T) {
	img := NewImage[float32](10, 5)

	rowSlice := img.RowSlice(0)
	if len(rowSlice) != 10 {
		t.Errorf("RowSlice length: got %d, want 10", len(rowSlice))
	}

	fullRow := img.Row(0)
	if len(fullRow) < 10 {
		t.Errorf("Row length: got %d, want >= 10", len(fullRow))
	}
}

func TestImage_AtSet(t *testing.T) {
	img := NewImage[float32](10, 10)

	// Set and get
	img.Set(5, 7, 42.0)
	if got := img.At(5, 7); got != 42.0 {
		t.Errorf("At(5,7): got %v, want 42.0", got)
	}

	// Out of bounds should return zero
	if got := img.At(-1, 0); got != 0 {
		t.Errorf("At(-1,0): got %v, want 0", got)
	}
	if got := img.At(0, -1); got != 0 {
		t.Errorf("At(0,-1): got %v, want 0", got)
	}
	if got := img.At(10, 0); got != 0 {
		t.Errorf("At(10,0): got %v, want 0", got)
	}

	// Set out of bounds should be no-op
	img.Set(-1, 0, 999)
	img.Set(10, 0, 999)
}

func TestImage_Clone(t *testing.T) {
	img := NewImage[float32](10, 10)
	img.Set(5, 5, 42.0)

	clone := img.Clone()

	// Should have same dimensions
	if clone.Width() != img.Width() || clone.Height() != img.Height() {
		t.Error("Clone dimensions differ")
	}

	// Should have same data
	if clone.At(5, 5) != 42.0 {
		t.Errorf("Clone data: got %v, want 42.0", clone.At(5, 5))
	}

	// Should be independent
	clone.Set(5, 5, 100.0)
	if img.At(5, 5) != 42.0 {
		t.Error("Clone should be independent")
	}
}

func TestImage_ClearFill(t *testing.T) {
	img := NewImage[float32](10, 10)

	// Fill
	img.Fill(42.0)
	for y := range 10 {
		for x := range 10 {
			if img.At(x, y) != 42.0 {
				t.Errorf("Fill: At(%d,%d) = %v, want 42.0", x, y, img.At(x, y))
			}
		}
	}

	// Clear
	img.Clear()
	for y := range 10 {
		for x := range 10 {
			if img.At(x, y) != 0 {
				t.Errorf("Clear: At(%d,%d) = %v, want 0", x, y, img.At(x, y))
			}
		}
	}
}

func TestImage_Bounds(t *testing.T) {
	img := NewImage[float32](100, 50)
	bounds := img.Bounds()

	if bounds.X0 != 0 || bounds.Y0 != 0 {
		t.Errorf("Bounds origin: got (%d,%d), want (0,0)", bounds.X0, bounds.Y0)
	}
	if bounds.X1 != 100 || bounds.Y1 != 50 {
		t.Errorf("Bounds size: got (%d,%d), want (100,50)", bounds.X1, bounds.Y1)
	}
	if bounds.Width() != 100 || bounds.Height() != 50 {
		t.Errorf("Bounds dimensions: got %dx%d, want 100x50", bounds.Width(), bounds.Height())
	}
}

func TestSameSize(t *testing.T) {
	a := NewImage[float32](100, 50)
	b := NewImage[float32](100, 50)
	c := NewImage[float32](50, 100)

	if !SameSize(a, b) {
		t.Error("SameSize should return true for equal dimensions")
	}
	if SameSize(a, c) {
		t.Error("SameSize should return false for different dimensions")
	}

	// Different types
	d := NewImage[int32](100, 50)
	if !SameSize(a, d) {
		t.Error("SameSize should work across different element types")
	}
}

func TestRect(t *testing.T) {
	r := Rect{X0: 10, Y0: 20, X1: 100, Y1: 80}

	if r.Width() != 90 {
		t.Errorf("Width: got %d, want 90", r.Width())
	}
	if r.Height() != 60 {
		t.Errorf("Height: got %d, want 60", r.Height())
	}
	if r.IsEmpty() {
		t.Error("IsEmpty should be false")
	}

	empty := Rect{X0: 10, Y0: 10, X1: 10, Y1: 10}
	if !empty.IsEmpty() {
		t.Error("Zero-area rect should be empty")
	}

	negative := Rect{X0: 10, Y0: 10, X1: 5, Y1: 5}
	if !negative.IsEmpty() {
		t.Error("Negative-area rect should be empty")
	}
}

func TestRect_Intersect(t *testing.T) {
	a := Rect{X0: 0, Y0: 0, X1: 100, Y1: 100}
	b := Rect{X0: 50, Y0: 50, X1: 150, Y1: 150}

	intersect := a.Intersect(b)

	if intersect.X0 != 50 || intersect.Y0 != 50 {
		t.Errorf("Intersect origin: got (%d,%d), want (50,50)", intersect.X0, intersect.Y0)
	}
	if intersect.X1 != 100 || intersect.Y1 != 100 {
		t.Errorf("Intersect end: got (%d,%d), want (100,100)", intersect.X1, intersect.Y1)
	}

	// Non-overlapping
	c := Rect{X0: 200, Y0: 200, X1: 300, Y1: 300}
	noIntersect := a.Intersect(c)
	if !noIntersect.IsEmpty() {
		t.Error("Non-overlapping rects should have empty intersection")
	}
}

func TestImage3(t *testing.T) {
	img := NewImage3[float32](100, 50)

	if img.Width() != 100 || img.Height() != 50 {
		t.Errorf("Image3 dimensions: got %dx%d, want 100x50", img.Width(), img.Height())
	}

	// Set values in each plane
	for p := range 3 {
		plane := img.Plane(p)
		if plane == nil {
			t.Errorf("Plane(%d) returned nil", p)
			continue
		}
		plane.Set(10, 20, float32(p*100))
	}

	// Verify each plane is independent
	for p := range 3 {
		got := img.Plane(p).At(10, 20)
		want := float32(p * 100)
		if got != want {
			t.Errorf("Plane(%d).At(10,20): got %v, want %v", p, got, want)
		}
	}

	// PlaneRow convenience method
	row := img.PlaneRow(1, 5)
	if row == nil {
		t.Error("PlaneRow returned nil")
	}

	// Invalid plane
	if img.Plane(-1) != nil || img.Plane(3) != nil {
		t.Error("Invalid plane index should return nil")
	}
}

func TestMirror(t *testing.T) {
	tests := []struct {
		index, size, want int
	}{
		{0, 10, 0},
		{5, 10, 5},
		{9, 10, 9},
		{10, 10, 9},  // Mirror at boundary
		{11, 10, 8},  // Mirror past boundary
		{-1, 10, 0},  // Mirror negative
		{-2, 10, 1},  // Mirror more negative
		{20, 10, 0},  // Double wrap
		{-10, 10, 9}, // Negative wrap
	}

	for _, tt := range tests {
		got := Mirror(tt.index, tt.size)
		if got != tt.want {
			t.Errorf("Mirror(%d, %d) = %d, want %d", tt.index, tt.size, got, tt.want)
		}
	}
}

func TestClamp(t *testing.T) {
	tests := []struct {
		index, size, want int
	}{
		{0, 10, 0},
		{5, 10, 5},
		{9, 10, 9},
		{10, 10, 9},  // Clamp high
		{100, 10, 9}, // Clamp very high
		{-1, 10, 0},  // Clamp low
		{-100, 10, 0},
	}

	for _, tt := range tests {
		got := Clamp(tt.index, tt.size)
		if got != tt.want {
			t.Errorf("Clamp(%d, %d) = %d, want %d", tt.index, tt.size, got, tt.want)
		}
	}
}

func TestWrap(t *testing.T) {
	tests := []struct {
		index, size, want int
	}{
		{0, 10, 0},
		{5, 10, 5},
		{9, 10, 9},
		{10, 10, 0}, // Wrap at boundary
		{11, 10, 1},
		{25, 10, 5},
		{-1, 10, 9}, // Wrap negative
		{-10, 10, 0},
		{-11, 10, 9},
	}

	for _, tt := range tests {
		got := Wrap(tt.index, tt.size)
		if got != tt.want {
			t.Errorf("Wrap(%d, %d) = %d, want %d", tt.index, tt.size, got, tt.want)
		}
	}
}

func TestImage_Int32(t *testing.T) {
	img := NewImage[int32](50, 50)
	img.Set(10, 10, 42)

	if got := img.At(10, 10); got != 42 {
		t.Errorf("Int32 image: got %v, want 42", got)
	}
}

func TestImage_Float64(t *testing.T) {
	img := NewImage[float64](50, 50)
	img.Set(10, 10, 3.14159)

	if got := img.At(10, 10); got != 3.14159 {
		t.Errorf("Float64 image: got %v, want 3.14159", got)
	}
}

// Benchmarks

func BenchmarkNewImage(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = NewImage[float32](1920, 1080)
	}
}

func BenchmarkImage_RowAccess(b *testing.B) {
	img := NewImage[float32](1920, 1080)

	b.ReportAllocs()

	for b.Loop() {
		for y := 0; y < img.Height(); y++ {
			row := img.Row(y)
			_ = row
		}
	}
}

func BenchmarkImage_Fill(b *testing.B) {
	img := NewImage[float32](1920, 1080)

	b.ReportAllocs()

	for b.Loop() {
		img.Fill(42.0)
	}
}

func BenchmarkImage_Clone(b *testing.B) {
	img := NewImage[float32](1920, 1080)
	img.Fill(42.0)

	b.ReportAllocs()

	for b.Loop() {
		_ = img.Clone()
	}
}
