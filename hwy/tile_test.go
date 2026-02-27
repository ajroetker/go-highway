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

import (
	"math"
	"testing"
)

func TestNewTile(t *testing.T) {
	tile := NewTile[float32]()
	dim := TileDim[float32]()
	if tile.dim != dim {
		t.Errorf("tile.dim = %d, want %d", tile.dim, dim)
	}
	if len(tile.data) != dim*dim {
		t.Errorf("len(tile.data) = %d, want %d", len(tile.data), dim*dim)
	}
	for i, v := range tile.data {
		if v != 0 {
			t.Errorf("tile.data[%d] = %f, want 0", i, v)
		}
	}
}

func TestTileZero(t *testing.T) {
	tile := NewTile[float32]()
	dim := tile.dim

	// Fill with non-zero values
	for i := range tile.data {
		tile.data[i] = float32(i + 1)
	}

	TileZero(&tile)

	for i := range tile.data {
		if tile.data[i] != 0 {
			t.Errorf("after TileZero: tile.data[%d] = %f, want 0", i, tile.data[i])
		}
	}
	_ = dim
}

func TestOuterProductAdd(t *testing.T) {
	dim := TileDim[float32]()

	// Create row = [1, 2, 3, ...] and col = [10, 20, 30, ...]
	rowData := make([]float32, dim)
	colData := make([]float32, dim)
	for i := range dim {
		rowData[i] = float32(i + 1)
		colData[i] = float32((i + 1) * 10)
	}
	row := Vec[float32]{data: rowData}
	col := Vec[float32]{data: colData}

	tile := NewTile[float32]()
	OuterProductAdd(&tile, row, col)

	// Verify: tile[i][j] should be row[i] * col[j]
	for i := range dim {
		for j := range dim {
			want := rowData[i] * colData[j]
			got := tile.data[i*dim+j]
			if math.Abs(float64(got-want)) > 1e-6 {
				t.Errorf("tile[%d][%d] = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestOuterProductAddAccumulates(t *testing.T) {
	dim := TileDim[float32]()

	row := Vec[float32]{data: make([]float32, dim)}
	col := Vec[float32]{data: make([]float32, dim)}
	for i := range dim {
		row.data[i] = 1
		col.data[i] = 1
	}

	tile := NewTile[float32]()

	// Accumulate 3 outer products
	OuterProductAdd(&tile, row, col)
	OuterProductAdd(&tile, row, col)
	OuterProductAdd(&tile, row, col)

	// Each element should be 3.0 (1*1 accumulated 3 times)
	for i := range dim {
		for j := range dim {
			got := tile.data[i*dim+j]
			if math.Abs(float64(got-3.0)) > 1e-6 {
				t.Errorf("tile[%d][%d] = %f, want 3.0", i, j, got)
			}
		}
	}
}

func TestOuterProductSub(t *testing.T) {
	dim := TileDim[float32]()

	rowData := make([]float32, dim)
	colData := make([]float32, dim)
	for i := range dim {
		rowData[i] = float32(i + 1)
		colData[i] = float32((i + 1) * 10)
	}
	row := Vec[float32]{data: rowData}
	col := Vec[float32]{data: colData}

	// Start with tile filled with 100
	tile := NewTile[float32]()
	for i := range tile.data {
		tile.data[i] = 100
	}

	OuterProductSub(&tile, row, col)

	// Verify: tile[i][j] should be 100 - row[i] * col[j]
	for i := range dim {
		for j := range dim {
			want := 100 - rowData[i]*colData[j]
			got := tile.data[i*dim+j]
			if math.Abs(float64(got-want)) > 1e-6 {
				t.Errorf("tile[%d][%d] = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestTileStoreRow(t *testing.T) {
	dim := TileDim[float32]()

	tile := NewTile[float32]()
	// Fill tile with known pattern: tile[i][j] = i*dim + j + 1
	for i := range dim {
		for j := range dim {
			tile.data[i*dim+j] = float32(i*dim + j + 1)
		}
	}

	// Store each row and verify
	for i := range dim {
		dst := make([]float32, dim)
		TileStoreRow(&tile, i, dst)
		for j := range dim {
			want := float32(i*dim + j + 1)
			if dst[j] != want {
				t.Errorf("row %d, col %d: got %f, want %f", i, j, dst[j], want)
			}
		}
	}
}

func TestTileReadRow(t *testing.T) {
	dim := TileDim[float32]()

	tile := NewTile[float32]()
	for i := range dim {
		for j := range dim {
			tile.data[i*dim+j] = float32(i*dim + j + 1)
		}
	}

	for i := range dim {
		vec := TileReadRow(&tile, i)
		if len(vec.data) != dim {
			t.Errorf("row %d: len = %d, want %d", i, len(vec.data), dim)
		}
		for j := range dim {
			want := float32(i*dim + j + 1)
			if vec.data[j] != want {
				t.Errorf("row %d, lane %d: got %f, want %f", i, j, vec.data[j], want)
			}
		}
	}
}

func TestTileLoadCol(t *testing.T) {
	dim := TileDim[float32]()

	tile := NewTile[float32]()
	src := make([]float32, dim)
	for i := range dim {
		src[i] = float32(i*10 + 1)
	}

	// Load into column 2
	colIdx := min(2, dim-1)
	TileLoadCol(&tile, colIdx, src)

	// Verify column
	for i := range dim {
		got := tile.data[i*dim+colIdx]
		if got != src[i] {
			t.Errorf("tile[%d][%d] = %f, want %f", i, colIdx, got, src[i])
		}
	}

	// Verify other columns are still zero
	for i := range dim {
		for j := range dim {
			if j == colIdx {
				continue
			}
			if tile.data[i*dim+j] != 0 {
				t.Errorf("tile[%d][%d] = %f, want 0 (non-loaded column)", i, j, tile.data[i*dim+j])
			}
		}
	}
}

func TestTileMatMul(t *testing.T) {
	// Verify that accumulating outer products over K computes C += A * B
	// where A is column-major packed and B is row-major.
	dim := TileDim[float32]()

	// Create small dim×dim matrices A and B
	a := make([]float32, dim*dim)
	b := make([]float32, dim*dim)
	for i := range dim {
		for j := range dim {
			a[i*dim+j] = float32(i + j + 1)
			b[i*dim+j] = float32(i*2 + j + 1)
		}
	}

	// Pack A column-major: aPacked[k*dim + i] = A[i][k]
	aPacked := make([]float32, dim*dim)
	for i := range dim {
		for k := range dim {
			aPacked[k*dim+i] = a[i*dim+k]
		}
	}

	// Compute C = A*B using tile outer products
	tile := NewTile[float32]()
	for k := range dim {
		aCol := Vec[float32]{data: aPacked[k*dim : (k+1)*dim]}
		bRow := Vec[float32]{data: b[k*dim : (k+1)*dim]}
		OuterProductAdd(&tile, aCol, bRow)
	}

	// Compute expected C = A*B via naive triple loop
	expected := make([]float32, dim*dim)
	for i := range dim {
		for j := range dim {
			var sum float32
			for k := range dim {
				sum += a[i*dim+k] * b[k*dim+j]
			}
			expected[i*dim+j] = sum
		}
	}

	// Compare
	for i := range dim {
		for j := range dim {
			got := tile.data[i*dim+j]
			want := expected[i*dim+j]
			if math.Abs(float64(got-want)) > 1e-3 {
				t.Errorf("C[%d][%d] = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestTileFloat64(t *testing.T) {
	dim := TileDim[float64]()
	if dim <= 0 {
		t.Fatal("TileDim[float64]() returned 0")
	}

	rowData := make([]float64, dim)
	colData := make([]float64, dim)
	for i := range dim {
		rowData[i] = float64(i + 1)
		colData[i] = float64((i + 1) * 10)
	}
	row := Vec[float64]{data: rowData}
	col := Vec[float64]{data: colData}

	tile := NewTile[float64]()
	OuterProductAdd(&tile, row, col)

	for i := range dim {
		for j := range dim {
			want := rowData[i] * colData[j]
			got := tile.data[i*dim+j]
			if math.Abs(got-want) > 1e-10 {
				t.Errorf("tile[%d][%d] = %f, want %f", i, j, got, want)
			}
		}
	}
}
