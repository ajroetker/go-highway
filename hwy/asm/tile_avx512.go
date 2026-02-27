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

//go:build amd64 && goexperiment.simd

package asm

import "simd/archsimd"

// NewTileFloat32x16 returns a zero-initialized 16×16 tile accumulator.
func NewTileFloat32x16() TileFloat32x16 {
	return TileFloat32x16{}
}

// NewTileFloat64x8 returns a zero-initialized 8×8 tile accumulator.
func NewTileFloat64x8() TileFloat64x8 {
	return TileFloat64x8{}
}

// TileFloat32x16 represents a 16×16 tile accumulator as 16 AVX-512 Float32x16 row vectors.
// On SME, a 16×16 f32 tile maps directly to the ZA register. On AVX-512, this tile
// uses 16 ZMM registers with broadcast+FMA for outer products.
type TileFloat32x16 struct {
	Rows [16]archsimd.Float32x16
}

// Zero zeroes all elements of the tile.
func (t *TileFloat32x16) Zero() {
	z := archsimd.BroadcastFloat32x16(0)
	for i := range t.Rows {
		t.Rows[i] = z
	}
}

// OuterProductAdd accumulates an outer product: tile += outer(row, col).
// tile[i][j] += row[i] * col[j] for all i,j in [0,16).
func (t *TileFloat32x16) OuterProductAdd(row, col archsimd.Float32x16) {
	var rowArr [16]float32
	row.Store(&rowArr)
	for i := range 16 {
		bcast := archsimd.BroadcastFloat32x16(rowArr[i])
		t.Rows[i] = bcast.MulAdd(col, t.Rows[i])
	}
}

// OuterProductSub subtracts an outer product: tile -= outer(row, col).
func (t *TileFloat32x16) OuterProductSub(row, col archsimd.Float32x16) {
	var rowArr [16]float32
	row.Store(&rowArr)
	for i := range 16 {
		bcast := archsimd.BroadcastFloat32x16(rowArr[i])
		prod := bcast.Mul(col)
		t.Rows[i] = t.Rows[i].Sub(prod)
	}
}

// StoreRow copies tile row rowIdx to dst.
// PRECONDITION: len(dst) >= 16.
func (t *TileFloat32x16) StoreRow(idx int, dst []float32) {
	t.Rows[idx].StoreSlice(dst)
}

// ReadRow returns tile row rowIdx as a Float32x16 vector.
func (t *TileFloat32x16) ReadRow(idx int) archsimd.Float32x16 {
	return t.Rows[idx]
}

// LoadCol loads src into tile column colIdx.
// PRECONDITION: len(src) >= 16.
func (t *TileFloat32x16) LoadCol(idx int, src []float32) {
	for i := range 16 {
		var arr [16]float32
		t.Rows[i].Store(&arr)
		arr[idx] = src[i]
		t.Rows[i] = archsimd.LoadFloat32x16(&arr)
	}
}

// TileFloat64x8 represents an 8×8 tile accumulator as 8 AVX-512 Float64x8 row vectors.
type TileFloat64x8 struct {
	Rows [8]archsimd.Float64x8
}

// Zero zeroes all elements of the tile.
func (t *TileFloat64x8) Zero() {
	z := archsimd.BroadcastFloat64x8(0)
	for i := range t.Rows {
		t.Rows[i] = z
	}
}

// OuterProductAdd accumulates an outer product: tile += outer(row, col).
func (t *TileFloat64x8) OuterProductAdd(row, col archsimd.Float64x8) {
	var rowArr [8]float64
	row.Store(&rowArr)
	for i := range 8 {
		bcast := archsimd.BroadcastFloat64x8(rowArr[i])
		t.Rows[i] = bcast.MulAdd(col, t.Rows[i])
	}
}

// OuterProductSub subtracts an outer product: tile -= outer(row, col).
func (t *TileFloat64x8) OuterProductSub(row, col archsimd.Float64x8) {
	var rowArr [8]float64
	row.Store(&rowArr)
	for i := range 8 {
		bcast := archsimd.BroadcastFloat64x8(rowArr[i])
		prod := bcast.Mul(col)
		t.Rows[i] = t.Rows[i].Sub(prod)
	}
}

// StoreRow copies tile row rowIdx to dst.
func (t *TileFloat64x8) StoreRow(idx int, dst []float64) {
	t.Rows[idx].StoreSlice(dst)
}

// ReadRow returns tile row rowIdx as a Float64x8 vector.
func (t *TileFloat64x8) ReadRow(idx int) archsimd.Float64x8 {
	return t.Rows[idx]
}

// LoadCol loads src into tile column colIdx.
func (t *TileFloat64x8) LoadCol(idx int, src []float64) {
	for i := range 8 {
		var arr [8]float64
		t.Rows[i].Store(&arr)
		arr[idx] = src[i]
		t.Rows[i] = archsimd.LoadFloat64x8(&arr)
	}
}
