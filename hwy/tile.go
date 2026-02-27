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

package hwy

// Tile is a 2D matrix accumulator of size TileDim × TileDim.
// In base (scalar) mode, it wraps a flat slice. In SIMD modes, hwygen
// replaces this with target-specific tile types backed by arrays of vectors
// (AVX2/NEON) or hardware tile registers (SME ZA).
//
// Tile instances should be created with NewTile and zeroed with TileZero.
// The primary operation is OuterProductAdd which accumulates outer products
// into the tile, mapping to a single FMOPA instruction on SME.
type Tile[T Lanes] struct {
	data []T
	dim  int
}

// TileDim returns the tile dimension for type T with the current SIMD width.
// The tile is TileDim × TileDim elements. This equals MaxLanes[T]() for
// non-SME targets; on SME it matches the ZA tile size.
//
// For example, with AVX2 (256 bits):
//   - float32: 8 (8×8 tile)
//   - float64: 4 (4×4 tile)
func TileDim[T Lanes]() int {
	return MaxLanes[T]()
}

// NewTile creates a zero-initialized tile of size TileDim × TileDim.
func NewTile[T Lanes]() Tile[T] {
	dim := TileDim[T]()
	return Tile[T]{data: make([]T, dim*dim), dim: dim}
}

// TileZero zeroes all elements of the tile.
func TileZero[T Lanes](tile *Tile[T]) {
	for i := range tile.data {
		var zero T
		tile.data[i] = zero
	}
}

// OuterProductAdd accumulates an outer product into the tile:
//
//	tile[i][j] += row[i] * col[j]
//
// On SME, this maps to a single FMOPA instruction. On AVX2/NEON, it
// expands to N broadcast+FMA operations. On scalar fallback, it uses
// a nested loop.
func OuterProductAdd[T Floats](tile *Tile[T], row, col Vec[T]) {
	dim := tile.dim
	for i := range dim {
		ri := row.data[i]
		rowStart := i * dim
		for j := range dim {
			tile.data[rowStart+j] = fmaScalar(ri, col.data[j], tile.data[rowStart+j])
		}
	}
}

// OuterProductSub subtracts an outer product from the tile:
//
//	tile[i][j] -= row[i] * col[j]
//
// On SME, this maps to a single FMOPS instruction.
func OuterProductSub[T Floats](tile *Tile[T], row, col Vec[T]) {
	dim := tile.dim
	for i := range dim {
		ri := row.data[i]
		rowStart := i * dim
		for j := range dim {
			tile.data[rowStart+j] = fmsScalar(ri, col.data[j], tile.data[rowStart+j])
		}
	}
}

// TileStoreRow copies tile row rowIdx to dst.
// PRECONDITION: len(dst) >= TileDim[T]().
func TileStoreRow[T Lanes](tile *Tile[T], rowIdx int, dst []T) {
	dim := tile.dim
	copy(dst[:dim], tile.data[rowIdx*dim:(rowIdx+1)*dim])
}

// TileReadRow reads tile row rowIdx as a Vec.
func TileReadRow[T Lanes](tile *Tile[T], rowIdx int) Vec[T] {
	dim := tile.dim
	data := make([]T, dim)
	copy(data, tile.data[rowIdx*dim:(rowIdx+1)*dim])
	return Vec[T]{data: data}
}

// TileLoadCol loads a column into tile column colIdx from src.
// src[i] is placed into tile[i][colIdx] for each row i.
// PRECONDITION: len(src) >= TileDim[T]().
func TileLoadCol[T Lanes](tile *Tile[T], colIdx int, src []T) {
	dim := tile.dim
	for i := range dim {
		tile.data[i*dim+colIdx] = src[i]
	}
}

// fmaScalar computes a*b + c for a single element.
func fmaScalar[T Floats](a, b, c T) T {
	switch av := any(a).(type) {
	case Float16:
		bv := any(b).(Float16)
		cv := any(c).(Float16)
		return any(Float32ToFloat16(av.Float32()*bv.Float32() + cv.Float32())).(T)
	case BFloat16:
		bv := any(b).(BFloat16)
		cv := any(c).(BFloat16)
		return any(Float32ToBFloat16(av.Float32()*bv.Float32() + cv.Float32())).(T)
	case float32:
		bv := any(b).(float32)
		cv := any(c).(float32)
		return any(av*bv + cv).(T)
	case float64:
		bv := any(b).(float64)
		cv := any(c).(float64)
		return any(av*bv + cv).(T)
	default:
		return c
	}
}

// fmsScalar computes c - a*b for a single element.
func fmsScalar[T Floats](a, b, c T) T {
	switch av := any(a).(type) {
	case Float16:
		bv := any(b).(Float16)
		cv := any(c).(Float16)
		return any(Float32ToFloat16(cv.Float32() - av.Float32()*bv.Float32())).(T)
	case BFloat16:
		bv := any(b).(BFloat16)
		cv := any(c).(BFloat16)
		return any(Float32ToBFloat16(cv.Float32() - av.Float32()*bv.Float32())).(T)
	case float32:
		bv := any(b).(float32)
		cv := any(c).(float32)
		return any(cv - av*bv).(T)
	case float64:
		bv := any(b).(float64)
		cv := any(c).(float64)
		return any(cv - av*bv).(T)
	default:
		return c
	}
}
