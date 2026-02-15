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

package matmul

import "testing"

func TestPadMatrix2DUint8(t *testing.T) {
	t.Run("no_padding_needed", func(t *testing.T) {
		src := []uint8{1, 2, 3, 4, 5, 6}
		dst := make([]uint8, 6)
		padMatrix2DUint8(dst, src, 2, 3, 2, 3)

		for i, v := range src {
			if dst[i] != v {
				t.Errorf("dst[%d] = %d, want %d", i, dst[i], v)
			}
		}
	})

	t.Run("row_padding_only", func(t *testing.T) {
		// 2x3 → 4x3 (same cols, extra rows)
		src := []uint8{1, 2, 3, 4, 5, 6}
		dst := make([]uint8, 12)
		padMatrix2DUint8(dst, src, 2, 3, 4, 3)

		// Original rows
		want := []uint8{1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0}
		for i, v := range want {
			if dst[i] != v {
				t.Errorf("dst[%d] = %d, want %d", i, dst[i], v)
			}
		}
	})

	t.Run("col_padding_only", func(t *testing.T) {
		// 2x3 → 2x4 (extra col, same rows)
		src := []uint8{1, 2, 3, 4, 5, 6}
		dst := make([]uint8, 8)
		padMatrix2DUint8(dst, src, 2, 3, 2, 4)

		want := []uint8{1, 2, 3, 0, 4, 5, 6, 0}
		for i, v := range want {
			if dst[i] != v {
				t.Errorf("dst[%d] = %d, want %d", i, dst[i], v)
			}
		}
	})

	t.Run("both_row_and_col_padding", func(t *testing.T) {
		// 2x3 → 4x4
		src := []uint8{1, 2, 3, 4, 5, 6}
		dst := make([]uint8, 16)
		padMatrix2DUint8(dst, src, 2, 3, 4, 4)

		want := []uint8{
			1, 2, 3, 0,
			4, 5, 6, 0,
			0, 0, 0, 0,
			0, 0, 0, 0,
		}
		for i, v := range want {
			if dst[i] != v {
				t.Errorf("dst[%d] = %d, want %d", i, dst[i], v)
			}
		}
	})

	t.Run("dirty_dst_is_zeroed", func(t *testing.T) {
		// Ensure padding regions are zeroed even if dst has stale data
		src := []uint8{10, 20}
		dst := []uint8{255, 255, 255, 255, 255, 255, 255, 255}
		padMatrix2DUint8(dst, src, 1, 2, 2, 4)

		want := []uint8{10, 20, 0, 0, 0, 0, 0, 0}
		for i, v := range want {
			if dst[i] != v {
				t.Errorf("dst[%d] = %d, want %d", i, dst[i], v)
			}
		}
	})
}

func TestExtractMatrix2DInt32(t *testing.T) {
	t.Run("no_padding", func(t *testing.T) {
		src := []int32{1, 2, 3, 4, 5, 6}
		dst := make([]int32, 6)
		extractMatrix2DInt32(dst, src, 2, 3, 3)

		for i, v := range src {
			if dst[i] != v {
				t.Errorf("dst[%d] = %d, want %d", i, dst[i], v)
			}
		}
	})

	t.Run("with_col_padding", func(t *testing.T) {
		// Extract 2x3 from padded 2x4
		src := []int32{1, 2, 3, 99, 4, 5, 6, 99}
		dst := make([]int32, 6)
		extractMatrix2DInt32(dst, src, 2, 3, 4)

		want := []int32{1, 2, 3, 4, 5, 6}
		for i, v := range want {
			if dst[i] != v {
				t.Errorf("dst[%d] = %d, want %d", i, dst[i], v)
			}
		}
	})

	t.Run("extract_from_large_padded", func(t *testing.T) {
		// Extract 2x2 from padded _x4
		src := []int32{10, 20, 0, 0, 30, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
		dst := make([]int32, 4)
		extractMatrix2DInt32(dst, src, 2, 2, 4)

		want := []int32{10, 20, 30, 40}
		for i, v := range want {
			if dst[i] != v {
				t.Errorf("dst[%d] = %d, want %d", i, dst[i], v)
			}
		}
	})
}
