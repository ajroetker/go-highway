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

//go:build !noasm && arm64

package asm

// NewTileFloat32x4 returns a zero-initialized 4×4 tile accumulator.
func NewTileFloat32x4() TileFloat32x4 {
	return TileFloat32x4{}
}

// NewTileFloat64x2 returns a zero-initialized 2×2 tile accumulator.
func NewTileFloat64x2() TileFloat64x2 {
	return TileFloat64x2{}
}

// TileFloat32x4 represents a 4×4 tile accumulator as 4 NEON Float32x4 row vectors.
// On ARM64, OuterProductAdd uses broadcast+FMA per row. On SME targets, the entire
// function using tile ops is compiled to assembly with FMOPA instructions instead.
type TileFloat32x4 struct {
	Rows [4]Float32x4
}

// Zero zeroes all elements of the tile.
func (t *TileFloat32x4) Zero() {
	t.Rows[0] = ZeroFloat32x4()
	t.Rows[1] = ZeroFloat32x4()
	t.Rows[2] = ZeroFloat32x4()
	t.Rows[3] = ZeroFloat32x4()
}

// OuterProductAdd accumulates an outer product: tile += outer(row, col).
// tile[i][j] += row[i] * col[j] for all i,j in [0,4).
//
// Uses 4 broadcast+FMA operations. For higher performance on NEON, the
// :asm path uses vfmaq_laneq_f32 which avoids explicit broadcast.
func (t *TileFloat32x4) OuterProductAdd(row, col Float32x4) {
	bcast0 := BroadcastFloat32x4(row.Get(0))
	bcast0.MulAddAcc(col, &t.Rows[0])
	bcast1 := BroadcastFloat32x4(row.Get(1))
	bcast1.MulAddAcc(col, &t.Rows[1])
	bcast2 := BroadcastFloat32x4(row.Get(2))
	bcast2.MulAddAcc(col, &t.Rows[2])
	bcast3 := BroadcastFloat32x4(row.Get(3))
	bcast3.MulAddAcc(col, &t.Rows[3])
}

// OuterProductSub subtracts an outer product: tile -= outer(row, col).
// tile[i][j] -= row[i] * col[j] for all i,j in [0,4).
func (t *TileFloat32x4) OuterProductSub(row, col Float32x4) {
	bcast0 := BroadcastFloat32x4(row.Get(0))
	prod0 := bcast0.Mul(col)
	t.Rows[0] = t.Rows[0].Sub(prod0)
	bcast1 := BroadcastFloat32x4(row.Get(1))
	prod1 := bcast1.Mul(col)
	t.Rows[1] = t.Rows[1].Sub(prod1)
	bcast2 := BroadcastFloat32x4(row.Get(2))
	prod2 := bcast2.Mul(col)
	t.Rows[2] = t.Rows[2].Sub(prod2)
	bcast3 := BroadcastFloat32x4(row.Get(3))
	prod3 := bcast3.Mul(col)
	t.Rows[3] = t.Rows[3].Sub(prod3)
}

// StoreRow copies tile row rowIdx to dst.
// PRECONDITION: len(dst) >= 4.
func (t *TileFloat32x4) StoreRow(idx int, dst []float32) {
	t.Rows[idx].StoreSlice(dst)
}

// ReadRow returns tile row rowIdx as a Float32x4 vector.
func (t *TileFloat32x4) ReadRow(idx int) Float32x4 {
	return t.Rows[idx]
}

// LoadCol loads src into tile column colIdx.
// src[i] is placed into tile[i][colIdx] for each row i.
// PRECONDITION: len(src) >= 4.
func (t *TileFloat32x4) LoadCol(idx int, src []float32) {
	t.Rows[0].Set(idx, src[0])
	t.Rows[1].Set(idx, src[1])
	t.Rows[2].Set(idx, src[2])
	t.Rows[3].Set(idx, src[3])
}

// TileFloat64x2 represents a 2×2 tile accumulator as 2 NEON Float64x2 row vectors.
type TileFloat64x2 struct {
	Rows [2]Float64x2
}

// Zero zeroes all elements of the tile.
func (t *TileFloat64x2) Zero() {
	t.Rows[0] = ZeroFloat64x2()
	t.Rows[1] = ZeroFloat64x2()
}

// OuterProductAdd accumulates an outer product: tile += outer(row, col).
func (t *TileFloat64x2) OuterProductAdd(row, col Float64x2) {
	bcast0 := BroadcastFloat64x2(row.Get(0))
	bcast0.MulAddAcc(col, &t.Rows[0])
	bcast1 := BroadcastFloat64x2(row.Get(1))
	bcast1.MulAddAcc(col, &t.Rows[1])
}

// OuterProductSub subtracts an outer product: tile -= outer(row, col).
func (t *TileFloat64x2) OuterProductSub(row, col Float64x2) {
	bcast0 := BroadcastFloat64x2(row.Get(0))
	prod0 := bcast0.Mul(col)
	t.Rows[0] = t.Rows[0].Sub(prod0)
	bcast1 := BroadcastFloat64x2(row.Get(1))
	prod1 := bcast1.Mul(col)
	t.Rows[1] = t.Rows[1].Sub(prod1)
}

// StoreRow copies tile row rowIdx to dst.
func (t *TileFloat64x2) StoreRow(idx int, dst []float64) {
	t.Rows[idx].StoreSlice(dst)
}

// ReadRow returns tile row rowIdx as a Float64x2 vector.
func (t *TileFloat64x2) ReadRow(idx int) Float64x2 {
	return t.Rows[idx]
}

// LoadCol loads src into tile column colIdx.
func (t *TileFloat64x2) LoadCol(idx int, src []float64) {
	t.Rows[0].Set(idx, src[0])
	t.Rows[1].Set(idx, src[1])
}
