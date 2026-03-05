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

func TestFind(t *testing.T) {
	tests := []struct {
		name   string
		slice  []float32
		value  float32
		expect int
	}{
		{"first", []float32{1, 2, 3, 4, 5}, 1, 0},
		{"middle", []float32{1, 2, 3, 4, 5}, 3, 2},
		{"last", []float32{1, 2, 3, 4, 5}, 5, 4},
		{"not_found", []float32{1, 2, 3, 4, 5}, 6, -1},
		{"empty", []float32{}, 1, -1},
		{"single_found", []float32{42}, 42, 0},
		{"single_not_found", []float32{42}, 1, -1},
		{"duplicates", []float32{1, 2, 3, 2, 5}, 2, 1}, // Returns first occurrence
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Find(tt.slice, tt.value)
			if got != tt.expect {
				t.Errorf("Find(%v, %v) = %d, want %d", tt.slice, tt.value, got, tt.expect)
			}
		})
	}
}

func TestFind_Large(t *testing.T) {
	// Test with array larger than vector width
	sizes := []int{15, 16, 17, 31, 32, 33, 100}

	for _, size := range sizes {
		t.Run("size_"+string(rune('0'+size%10)), func(t *testing.T) {
			slice := make([]float32, size)
			for i := range slice {
				slice[i] = float32(i)
			}

			// Find element in tail
			got := Find(slice, float32(size-1))
			if got != size-1 {
				t.Errorf("Find last element: got %d, want %d", got, size-1)
			}

			// Find element in middle
			mid := size / 2
			got = Find(slice, float32(mid))
			if got != mid {
				t.Errorf("Find middle element: got %d, want %d", got, mid)
			}
		})
	}
}

func TestFind_Int32(t *testing.T) {
	slice := []int32{10, 20, 30, 40, 50}
	got := Find(slice, int32(30))
	if got != 2 {
		t.Errorf("Find int32: got %d, want 2", got)
	}

	got = Find(slice, int32(99))
	if got != -1 {
		t.Errorf("Find int32 not found: got %d, want -1", got)
	}
}

func TestCount(t *testing.T) {
	tests := []struct {
		name   string
		slice  []float32
		value  float32
		expect int
	}{
		{"single", []float32{1, 2, 3, 4, 5}, 3, 1},
		{"multiple", []float32{1, 2, 2, 2, 5}, 2, 3},
		{"none", []float32{1, 2, 3, 4, 5}, 6, 0},
		{"all", []float32{7, 7, 7, 7}, 7, 4},
		{"empty", []float32{}, 1, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Count(tt.slice, tt.value)
			if got != tt.expect {
				t.Errorf("Count(%v, %v) = %d, want %d", tt.slice, tt.value, got, tt.expect)
			}
		})
	}
}

func TestCount_Large(t *testing.T) {
	// Create array with known count
	slice := make([]float32, 100)
	for i := range slice {
		if i%3 == 0 {
			slice[i] = 42
		} else {
			slice[i] = float32(i)
		}
	}

	// Should find 34 occurrences of 42 (indices 0, 3, 6, ..., 99)
	got := Count(slice, 42)
	expected := (99 / 3) + 1 // 34
	if got != expected {
		t.Errorf("Count large: got %d, want %d", got, expected)
	}
}

func TestContains(t *testing.T) {
	slice := []float32{1, 2, 3, 4, 5}

	if !Contains(slice, 3) {
		t.Error("Contains should find 3")
	}

	if Contains(slice, 6) {
		t.Error("Contains should not find 6")
	}
}

// Benchmarks

func BenchmarkFind(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(i)
	}
	target := float32(benchSize - 1) // Worst case: last element

	b.ReportAllocs()
	for b.Loop() {
		Find(slice, target)
	}
}

func BenchmarkFind_Stdlib(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(i)
	}
	target := float32(benchSize - 1)

	b.ReportAllocs()
	for b.Loop() {
		for j, v := range slice {
			if v == target {
				_ = j
				break
			}
		}
	}
}

func BenchmarkFind_Early(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(i)
	}
	target := float32(10) // Best case: early element

	b.ReportAllocs()
	for b.Loop() {
		Find(slice, target)
	}
}

func BenchmarkCount(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		if i%10 == 0 {
			slice[i] = 42
		} else {
			slice[i] = float32(i)
		}
	}

	b.ReportAllocs()
	for b.Loop() {
		Count(slice, 42)
	}
}

func BenchmarkCount_Stdlib(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		if i%10 == 0 {
			slice[i] = 42
		} else {
			slice[i] = float32(i)
		}
	}
	target := float32(42)

	b.ReportAllocs()
	for b.Loop() {
		count := 0
		for _, v := range slice {
			if v == target {
				count++
			}
		}
		_ = count
	}
}
