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

// Package image provides SIMD-friendly 2D image types with aligned rows.
//
// The Image type stores single-channel 2D data with rows aligned for
// efficient SIMD processing. This eliminates the need for special handling
// of row boundaries during vectorized operations.
//
// Example usage:
//
//	img := image.NewImage[float32](640, 480)
//	for y := 0; y < img.Height(); y++ {
//	    row := img.Row(y)
//	    // Process row with SIMD operations
//	    algo.ExpTransform(row, row)
//	}
package image

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// Image is a single-channel 2D array with SIMD-aligned rows.
// Each row is padded to a multiple of the SIMD vector width,
// enabling efficient vectorized operations without bounds checking.
type Image[T hwy.Lanes] struct {
	data        []T
	width       int
	height      int
	stride      int // elements per row (includes padding)
	bytesPerRow int
}

// NewImage creates a new image with the specified dimensions.
// Rows are aligned to the SIMD vector width for optimal performance.
func NewImage[T hwy.Lanes](width, height int) *Image[T] {
	if width <= 0 || height <= 0 {
		return &Image[T]{
			data:   nil,
			width:  0,
			height: 0,
			stride: 0,
		}
	}

	lanes := hwy.MaxLanes[T]()

	// Calculate stride (elements per row, rounded up to vector width)
	stride := ((width + lanes - 1) / lanes) * lanes

	// Calculate bytes per row
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	bytesPerRow := stride * elemSize

	// Allocate aligned data
	data := make([]T, stride*height)

	return &Image[T]{
		data:        data,
		width:       width,
		height:      height,
		stride:      stride,
		bytesPerRow: bytesPerRow,
	}
}

// Width returns the image width in pixels.
func (img *Image[T]) Width() int {
	return img.width
}

// Height returns the image height in pixels.
func (img *Image[T]) Height() int {
	return img.height
}

// Stride returns the number of elements per row (including padding).
func (img *Image[T]) Stride() int {
	return img.stride
}

// BytesPerRow returns the number of bytes per row.
func (img *Image[T]) BytesPerRow() int {
	return img.bytesPerRow
}

// Row returns a mutable slice for the specified row.
// The slice includes padding elements beyond the image width.
// These can be safely read/written but are not part of the image.
func (img *Image[T]) Row(y int) []T {
	if y < 0 || y >= img.height || img.data == nil {
		return nil
	}
	start := y * img.stride
	return img.data[start : start+img.stride]
}

// dataPtr returns an unsafe pointer to the first element of the image data.
// Used internally for passing to assembly functions.
func (img *Image[T]) dataPtr() unsafe.Pointer {
	if img.data == nil || len(img.data) == 0 {
		return nil
	}
	return unsafe.Pointer(&img.data[0])
}

// ConstRow returns a read-only slice for the specified row.
// In Go, this is the same as Row since const references don't exist.
func (img *Image[T]) ConstRow(y int) []T {
	return img.Row(y)
}

// RowSlice returns a mutable slice for the specified row,
// limited to the actual image width (excluding padding).
func (img *Image[T]) RowSlice(y int) []T {
	if y < 0 || y >= img.height || img.data == nil {
		return nil
	}
	start := y * img.stride
	return img.data[start : start+img.width]
}

// At returns the value at position (x, y).
func (img *Image[T]) At(x, y int) T {
	if x < 0 || x >= img.width || y < 0 || y >= img.height || img.data == nil {
		var zero T
		return zero
	}
	return img.data[y*img.stride+x]
}

// Set sets the value at position (x, y).
func (img *Image[T]) Set(x, y int, value T) {
	if x < 0 || x >= img.width || y < 0 || y >= img.height || img.data == nil {
		return
	}
	img.data[y*img.stride+x] = value
}

// SameSize returns true if both images have the same dimensions.
func SameSize[T, U hwy.Lanes](a *Image[T], b *Image[U]) bool {
	return a.width == b.width && a.height == b.height
}

// Clone creates a deep copy of the image.
func (img *Image[T]) Clone() *Image[T] {
	if img.data == nil {
		return NewImage[T](0, 0)
	}

	clone := &Image[T]{
		data:        make([]T, len(img.data)),
		width:       img.width,
		height:      img.height,
		stride:      img.stride,
		bytesPerRow: img.bytesPerRow,
	}
	copy(clone.data, img.data)
	return clone
}

// Clear sets all pixels to zero.
func (img *Image[T]) Clear() {
	for i := range img.data {
		var zero T
		img.data[i] = zero
	}
}

// Fill sets all pixels to the specified value.
func (img *Image[T]) Fill(value T) {
	for i := range img.data {
		img.data[i] = value
	}
}

// Rect defines a rectangular region within an image.
type Rect struct {
	X0, Y0 int // Top-left corner (inclusive)
	X1, Y1 int // Bottom-right corner (exclusive)
}

// Width returns the rectangle width.
func (r Rect) Width() int {
	return r.X1 - r.X0
}

// Height returns the rectangle height.
func (r Rect) Height() int {
	return r.Y1 - r.Y0
}

// IsEmpty returns true if the rectangle has zero or negative area.
func (r Rect) IsEmpty() bool {
	return r.X1 <= r.X0 || r.Y1 <= r.Y0
}

// Intersect returns the intersection of two rectangles.
func (r Rect) Intersect(other Rect) Rect {
	x0 := max(r.X0, other.X0)
	y0 := max(r.Y0, other.Y0)
	x1 := min(r.X1, other.X1)
	y1 := min(r.Y1, other.Y1)
	return Rect{X0: x0, Y0: y0, X1: x1, Y1: y1}
}

// Bounds returns the bounding rectangle of the image.
func (img *Image[T]) Bounds() Rect {
	return Rect{X0: 0, Y0: 0, X1: img.width, Y1: img.height}
}

// Image3 bundles three same-sized Image instances.
// Commonly used for RGB or YUV color planes.
type Image3[T hwy.Lanes] struct {
	planes [3]*Image[T]
}

// NewImage3 creates a new 3-plane image with the specified dimensions.
func NewImage3[T hwy.Lanes](width, height int) *Image3[T] {
	return &Image3[T]{
		planes: [3]*Image[T]{
			NewImage[T](width, height),
			NewImage[T](width, height),
			NewImage[T](width, height),
		},
	}
}

// Plane returns the specified plane (0, 1, or 2).
func (img *Image3[T]) Plane(i int) *Image[T] {
	if i < 0 || i > 2 {
		return nil
	}
	return img.planes[i]
}

// PlaneRow returns a row from the specified plane.
func (img *Image3[T]) PlaneRow(plane, y int) []T {
	if plane < 0 || plane > 2 {
		return nil
	}
	return img.planes[plane].Row(y)
}

// Width returns the image width (all planes have the same size).
func (img *Image3[T]) Width() int {
	return img.planes[0].Width()
}

// Height returns the image height.
func (img *Image3[T]) Height() int {
	return img.planes[0].Height()
}

// Mirror returns the mirrored index for out-of-bounds coordinates.
// Used for edge handling in convolution operations.
// Given bounds [0, size), mirrors index to stay within bounds.
func Mirror(index, size int) int {
	if size <= 0 {
		return 0
	}
	if index < 0 {
		index = -index - 1
	}
	if index >= size {
		// Wrap around using modulo with mirroring
		period := 2 * size
		index = index % period
		if index >= size {
			index = period - index - 1
		}
	}
	return index
}

// Clamp returns index clamped to [0, size-1].
func Clamp(index, size int) int {
	if index < 0 {
		return 0
	}
	if index >= size {
		return size - 1
	}
	return index
}

// Wrap returns index wrapped to [0, size) using modulo.
func Wrap(index, size int) int {
	if size <= 0 {
		return 0
	}
	index = index % size
	if index < 0 {
		index += size
	}
	return index
}
