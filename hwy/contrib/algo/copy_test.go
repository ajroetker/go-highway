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
	"testing"
)

func TestFill(t *testing.T) {
	tests := []struct {
		name  string
		size  int
		value float32
	}{
		{"single", 1, 42.0},
		{"small", 3, 3.14},
		{"vector_aligned", 8, 2.71},
		{"unaligned", 15, 1.41},
		{"large", 100, 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.size)
			Fill(dst, tt.value)

			for i := range dst {
				if dst[i] != tt.value {
					t.Errorf("Fill[%d]: got %v, want %v", i, dst[i], tt.value)
				}
			}
		})
	}
}

func TestFill_Int32(t *testing.T) {
	dst := make([]int32, 17)
	Fill(dst, int32(123))

	for i := range dst {
		if dst[i] != 123 {
			t.Errorf("Fill[%d]: got %v, want 123", i, dst[i])
		}
	}
}

func TestFill_Empty(t *testing.T) {
	var dst []float32
	Fill(dst, 1.0) // Should not panic
}

func TestCopy(t *testing.T) {
	tests := []struct {
		name string
		size int
	}{
		{"single", 1},
		{"small", 3},
		{"vector_aligned", 8},
		{"unaligned", 15},
		{"large", 100},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := make([]float32, tt.size)
			dst := make([]float32, tt.size)

			for i := range src {
				src[i] = float32(i) * 1.5
			}

			copied := Copy(src, dst)

			if copied != tt.size {
				t.Errorf("Copy returned %d, want %d", copied, tt.size)
			}

			for i := range src {
				if dst[i] != src[i] {
					t.Errorf("Copy[%d]: got %v, want %v", i, dst[i], src[i])
				}
			}
		})
	}
}

func TestCopy_DifferentSizes(t *testing.T) {
	// dst smaller than src
	src := make([]float32, 20)
	dst := make([]float32, 10)

	for i := range src {
		src[i] = float32(i)
	}

	copied := Copy(src, dst)

	if copied != 10 {
		t.Errorf("Copy returned %d, want 10", copied)
	}

	for i := range 10 {
		if dst[i] != float32(i) {
			t.Errorf("Copy[%d]: got %v, want %v", i, dst[i], float32(i))
		}
	}

	// src smaller than dst
	src = make([]float32, 10)
	dst = make([]float32, 20)

	for i := range src {
		src[i] = float32(i)
	}

	copied = Copy(src, dst)

	if copied != 10 {
		t.Errorf("Copy returned %d, want 10", copied)
	}
}

func TestCopy_Empty(t *testing.T) {
	var src, dst []float32
	copied := Copy(src, dst)
	if copied != 0 {
		t.Errorf("Copy of empty slices returned %d, want 0", copied)
	}
}

// Benchmarks
// Note: benchSize is defined in transform_test.go

func BenchmarkFill(b *testing.B) {
	dst := make([]float32, benchSize)

	b.ReportAllocs()
	for b.Loop() {
		Fill(dst, 42.0)
	}
}

func BenchmarkFill_Stdlib(b *testing.B) {
	dst := make([]float32, benchSize)
	value := float32(42.0)

	b.ReportAllocs()
	for b.Loop() {
		for j := range dst {
			dst[j] = value
		}
	}
}

func BenchmarkCopy(b *testing.B) {
	src := make([]float32, benchSize)
	dst := make([]float32, benchSize)
	for i := range src {
		src[i] = float32(i)
	}

	b.ReportAllocs()
	for b.Loop() {
		Copy(src, dst)
	}
}

func BenchmarkCopy_Stdlib(b *testing.B) {
	src := make([]float32, benchSize)
	dst := make([]float32, benchSize)
	for i := range src {
		src[i] = float32(i)
	}

	b.ReportAllocs()
	for b.Loop() {
		copy(dst, src)
	}
}

