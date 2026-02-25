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

// NewTileFloat32x8 returns a zero-initialized 8×8 tile accumulator.
func NewTileFloat32x8() TileFloat32x8 {
	return TileFloat32x8{}
}

// NewTileFloat64x4 returns a zero-initialized 4×4 tile accumulator.
func NewTileFloat64x4() TileFloat64x4 {
	return TileFloat64x4{}
}

// TileFloat32x8 represents an 8×8 tile accumulator as 8 AVX2 Float32x8 row vectors.
// OuterProductAdd uses broadcast+FMA per row: for each lane i of the row vector,
// broadcast row[i] to all 8 lanes and FMA with the column vector into tile row i.
type TileFloat32x8 struct {
	Rows [8]archsimd.Float32x8
}

// Zero zeroes all elements of the tile.
func (t *TileFloat32x8) Zero() {
	z := archsimd.BroadcastFloat32x8(0)
	for i := range t.Rows {
		t.Rows[i] = z
	}
}

// OuterProductAdd accumulates an outer product: tile += outer(row, col).
// tile[i][j] += row[i] * col[j] for all i,j in [0,8).
func (t *TileFloat32x8) OuterProductAdd(row, col archsimd.Float32x8) {
	var rowArr [8]float32
	row.Store(&rowArr)
	for i := range 8 {
		bcast := archsimd.BroadcastFloat32x8(rowArr[i])
		t.Rows[i] = bcast.MulAdd(col, t.Rows[i])
	}
}

// OuterProductSub subtracts an outer product: tile -= outer(row, col).
// tile[i][j] -= row[i] * col[j] for all i,j in [0,8).
func (t *TileFloat32x8) OuterProductSub(row, col archsimd.Float32x8) {
	var rowArr [8]float32
	row.Store(&rowArr)
	for i := range 8 {
		bcast := archsimd.BroadcastFloat32x8(rowArr[i])
		prod := bcast.Mul(col)
		t.Rows[i] = t.Rows[i].Sub(prod)
	}
}

// StoreRow copies tile row rowIdx to dst.
// PRECONDITION: len(dst) >= 8.
func (t *TileFloat32x8) StoreRow(idx int, dst []float32) {
	t.Rows[idx].StoreSlice(dst)
}

// ReadRow returns tile row rowIdx as a Float32x8 vector.
func (t *TileFloat32x8) ReadRow(idx int) archsimd.Float32x8 {
	return t.Rows[idx]
}

// LoadCol loads src into tile column colIdx.
// src[i] is placed into tile[i][colIdx] for each row i.
// PRECONDITION: len(src) >= 8.
func (t *TileFloat32x8) LoadCol(idx int, src []float32) {
	for i := range 8 {
		var arr [8]float32
		t.Rows[i].Store(&arr)
		arr[idx] = src[i]
		t.Rows[i] = archsimd.LoadFloat32x8(&arr)
	}
}

// TileFloat64x4 represents a 4×4 tile accumulator as 4 AVX2 Float64x4 row vectors.
type TileFloat64x4 struct {
	Rows [4]archsimd.Float64x4
}

// Zero zeroes all elements of the tile.
func (t *TileFloat64x4) Zero() {
	z := archsimd.BroadcastFloat64x4(0)
	for i := range t.Rows {
		t.Rows[i] = z
	}
}

// OuterProductAdd accumulates an outer product: tile += outer(row, col).
func (t *TileFloat64x4) OuterProductAdd(row, col archsimd.Float64x4) {
	var rowArr [4]float64
	row.Store(&rowArr)
	for i := range 4 {
		bcast := archsimd.BroadcastFloat64x4(rowArr[i])
		t.Rows[i] = bcast.MulAdd(col, t.Rows[i])
	}
}

// OuterProductSub subtracts an outer product: tile -= outer(row, col).
func (t *TileFloat64x4) OuterProductSub(row, col archsimd.Float64x4) {
	var rowArr [4]float64
	row.Store(&rowArr)
	for i := range 4 {
		bcast := archsimd.BroadcastFloat64x4(rowArr[i])
		prod := bcast.Mul(col)
		t.Rows[i] = t.Rows[i].Sub(prod)
	}
}

// StoreRow copies tile row rowIdx to dst.
func (t *TileFloat64x4) StoreRow(idx int, dst []float64) {
	t.Rows[idx].StoreSlice(dst)
}

// ReadRow returns tile row rowIdx as a Float64x4 vector.
func (t *TileFloat64x4) ReadRow(idx int) archsimd.Float64x4 {
	return t.Rows[idx]
}

// LoadCol loads src into tile column colIdx.
func (t *TileFloat64x4) LoadCol(idx int, src []float64) {
	for i := range 4 {
		var arr [4]float64
		t.Rows[i].Store(&arr)
		arr[idx] = src[i]
		t.Rows[i] = archsimd.LoadFloat64x4(&arr)
	}
}
