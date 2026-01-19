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

//go:build (amd64 && goexperiment.simd) || arm64

package algo

import (
	"slices"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

func TestBasePrefixSum(t *testing.T) {
	tests := []struct {
		name     string
		input    []int64
		expected []int64
	}{
		{
			name:     "simple",
			input:    []int64{1, 2, 3, 4},
			expected: []int64{1, 3, 6, 10},
		},
		{
			name:     "single",
			input:    []int64{42},
			expected: []int64{42},
		},
		{
			name:     "zeros",
			input:    []int64{0, 0, 0, 0},
			expected: []int64{0, 0, 0, 0},
		},
		{
			name:     "vector_aligned",
			input:    []int64{1, 2, 3, 4, 5, 6, 7, 8},
			expected: []int64{1, 3, 6, 10, 15, 21, 28, 36},
		},
		{
			name:     "unaligned",
			input:    []int64{1, 2, 3, 4, 5},
			expected: []int64{1, 3, 6, 10, 15},
		},
		{
			name:     "large",
			input:    []int64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			expected: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := slices.Clone(tt.input)
			BasePrefixSum(data)

			if len(data) != len(tt.expected) {
				t.Fatalf("length mismatch: got %d, want %d", len(data), len(tt.expected))
			}

			for i := range data {
				if data[i] != tt.expected[i] {
					t.Errorf("BasePrefixSum[%d]: got %d, want %d", i, data[i], tt.expected[i])
				}
			}
		})
	}
}

func TestBasePrefixSum_Empty(t *testing.T) {
	data := []int64{}
	BasePrefixSum(data) // should not panic
	if len(data) != 0 {
		t.Errorf("expected empty slice, got %v", data)
	}
}

func TestBasePrefixSum_Float32(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	expected := []float32{1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0}

	BasePrefixSum(data)

	for i := range data {
		if data[i] != expected[i] {
			t.Errorf("BasePrefixSum[%d]: got %f, want %f", i, data[i], expected[i])
		}
	}
}

func TestBaseDeltaDecode(t *testing.T) {
	tests := []struct {
		name     string
		input    []uint64
		base     uint64
		expected []uint64
	}{
		{
			name:     "simple",
			input:    []uint64{3, 2, 5, 1},
			base:     10,
			expected: []uint64{13, 15, 20, 21},
		},
		{
			name:     "zero_base",
			input:    []uint64{1, 2, 3, 4},
			base:     0,
			expected: []uint64{1, 3, 6, 10},
		},
		{
			name:     "single",
			input:    []uint64{5},
			base:     100,
			expected: []uint64{105},
		},
		{
			name:     "posting_list_simulation",
			input:    []uint64{100, 5, 10, 2, 15, 3, 8, 1},
			base:     0,
			expected: []uint64{100, 105, 115, 117, 132, 135, 143, 144},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := slices.Clone(tt.input)
			BaseDeltaDecode(data, tt.base)

			if len(data) != len(tt.expected) {
				t.Fatalf("length mismatch: got %d, want %d", len(data), len(tt.expected))
			}

			for i := range data {
				if data[i] != tt.expected[i] {
					t.Errorf("BaseDeltaDecode[%d]: got %d, want %d", i, data[i], tt.expected[i])
				}
			}
		})
	}
}

func TestBaseDeltaDecode_Empty(t *testing.T) {
	data := []uint64{}
	BaseDeltaDecode(data, 10) // should not panic
	if len(data) != 0 {
		t.Errorf("expected empty slice, got %v", data)
	}
}

func TestBasePrefixSumVec(t *testing.T) {
	// Test the vector-level prefix sum directly
	input := []int64{1, 2, 3, 4}
	v := hwy.Load(input)
	result := BasePrefixSumVec(v)

	expected := []int64{1, 3, 6, 10}
	for i := 0; i < result.NumLanes() && i < len(expected); i++ {
		got := hwy.GetLane(result, i)
		if got != expected[i] {
			t.Errorf("BasePrefixSumVec[%d]: got %d, want %d", i, got, expected[i])
		}
	}
}

func TestBasePrefixSum_LargeSlice(t *testing.T) {
	// Test with a larger slice to exercise multiple vector iterations
	size := 1000
	data := make([]int64, size)
	expected := make([]int64, size)

	for i := range data {
		data[i] = 1
		expected[i] = int64(i + 1)
	}

	BasePrefixSum(data)

	for i := range data {
		if data[i] != expected[i] {
			t.Errorf("BasePrefixSum[%d]: got %d, want %d", i, data[i], expected[i])
		}
	}
}

// Benchmarks

func BenchmarkPrefixSum_Int64(b *testing.B) {
	template := make([]int64, benchSize)
	for i := range template {
		template[i] = int64(i % 100)
	}
	data := make([]int64, benchSize)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		copy(data, template)
		PrefixSum(data)
	}
}

func BenchmarkBasePrefixSum_Scalar(b *testing.B) {
	template := make([]int64, benchSize)
	for i := range template {
		template[i] = int64(i % 100)
	}
	data := make([]int64, benchSize)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		copy(data, template)
		acc := int64(0)
		for i, v := range data {
			acc += v
			data[i] = acc
		}
	}
}

func BenchmarkDeltaDecode_Uint64(b *testing.B) {
	template := make([]uint64, benchSize)
	for i := range template {
		template[i] = uint64(i % 10) // Small deltas typical of posting lists
	}
	data := make([]uint64, benchSize)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		copy(data, template)
		DeltaDecode(data, 0)
	}
}

func BenchmarkBaseDeltaDecode_Scalar(b *testing.B) {
	template := make([]uint64, benchSize)
	for i := range template {
		template[i] = uint64(i % 10)
	}
	data := make([]uint64, benchSize)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		copy(data, template)
		acc := uint64(0)
		for i, v := range data {
			acc += v
			data[i] = acc
		}
	}
}

func BenchmarkDeltaDecode_Int32(b *testing.B) {
	template := make([]int32, benchSize)
	for i := range template {
		template[i] = int32(i % 10)
	}
	data := make([]int32, benchSize)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		copy(data, template)
		DeltaDecode(data, 0)
	}
}

func BenchmarkDeltaDecode_Uint32(b *testing.B) {
	template := make([]uint32, benchSize)
	for i := range template {
		template[i] = uint32(i % 10)
	}
	data := make([]uint32, benchSize)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		copy(data, template)
		DeltaDecode(data, 0)
	}
}

func BenchmarkPrefixSum_Float32(b *testing.B) {
	template := make([]float32, benchSize)
	for i := range template {
		template[i] = float32(i % 100)
	}
	data := make([]float32, benchSize)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		copy(data, template)
		PrefixSum(data)
	}
}

func BenchmarkBasePrefixSum_Scalar_Float32(b *testing.B) {
	template := make([]float32, benchSize)
	for i := range template {
		template[i] = float32(i % 100)
	}
	data := make([]float32, benchSize)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		copy(data, template)
		acc := float32(0)
		for i, v := range data {
			acc += v
			data[i] = acc
		}
	}
}
