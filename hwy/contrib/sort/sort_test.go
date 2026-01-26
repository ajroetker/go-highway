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

package sort

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// Helper to check if slice is sorted
func isSorted[T hwy.Lanes](data []T) bool {
	for i := 1; i < len(data); i++ {
		if data[i] < data[i-1] {
			return false
		}
	}
	return true
}

// TestSortEmpty tests sorting empty slices
func TestSortEmpty(t *testing.T) {
	var empty []float32
	Sort(empty)
	if len(empty) != 0 {
		t.Errorf("Sort(empty) should not modify empty slice")
	}
}

// TestSortSingle tests sorting single element slices
func TestSortSingle(t *testing.T) {
	data := []float32{42.0}
	Sort(data)
	if data[0] != 42.0 {
		t.Errorf("Sort([42]) = %v, want [42]", data)
	}
}

// TestSortAlreadySorted tests sorting already sorted data
func TestSortAlreadySorted(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	Sort(data)
	if !isSorted(data) {
		t.Errorf("Sort(sorted) produced unsorted result: %v", data)
	}
}

// TestSortReverse tests sorting reverse sorted data
func TestSortReverse(t *testing.T) {
	data := []float32{8, 7, 6, 5, 4, 3, 2, 1}
	Sort(data)
	if !isSorted(data) {
		t.Errorf("Sort(reverse) produced unsorted result: %v", data)
	}
}

// TestSortDuplicates tests sorting with duplicate elements
func TestSortDuplicates(t *testing.T) {
	data := []float32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5}
	Sort(data)
	if !isSorted(data) {
		t.Errorf("Sort(duplicates) produced unsorted result: %v", data)
	}
}

// TestSortAllSame tests sorting with all identical elements
func TestSortAllSame(t *testing.T) {
	data := []float32{5, 5, 5, 5, 5, 5, 5, 5}
	Sort(data)
	if !isSorted(data) {
		t.Errorf("Sort(allSame) produced unsorted result: %v", data)
	}
}

// TestSortRandomFloat32 tests sorting random float32 data
func TestSortRandomFloat32(t *testing.T) {
	sizes := []int{0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 100, 256, 1000}
	for _, n := range sizes {
		data := make([]float32, n)
		for i := range data {
			data[i] = rand.Float32() * 1000
		}
		Sort(data)
		if !isSorted(data) {
			t.Errorf("Sort(random float32, n=%d) produced unsorted result", n)
		}
	}
}

// TestSortRandomFloat64 tests sorting random float64 data
func TestSortRandomFloat64(t *testing.T) {
	sizes := []int{0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 100, 256, 1000}
	for _, n := range sizes {
		data := make([]float64, n)
		for i := range data {
			data[i] = rand.Float64() * 1000
		}
		Sort(data)
		if !isSorted(data) {
			t.Errorf("Sort(random float64, n=%d) produced unsorted result", n)
		}
	}
}

// TestSortRandomInt32 tests sorting random int32 data
func TestSortRandomInt32(t *testing.T) {
	sizes := []int{0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 100, 256, 1000}
	for _, n := range sizes {
		data := make([]int32, n)
		for i := range data {
			data[i] = rand.Int31n(10000) - 5000
		}
		Sort(data)
		if !isSorted(data) {
			t.Errorf("Sort(random int32, n=%d) produced unsorted result", n)
		}
	}
}

// TestSortRandomInt64 tests sorting random int64 data
func TestSortRandomInt64(t *testing.T) {
	sizes := []int{0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 100, 256, 1000}
	for _, n := range sizes {
		data := make([]int64, n)
		for i := range data {
			data[i] = rand.Int63n(10000) - 5000
		}
		Sort(data)
		if !isSorted(data) {
			t.Errorf("Sort(random int64, n=%d) produced unsorted result", n)
		}
	}
}

// TestSortMatchesStdlib verifies Sort produces same result as slices.Sort
func TestSortMatchesStdlib(t *testing.T) {
	rand.Seed(12345)
	sizes := []int{100, 256, 1000, 10000}
	for _, n := range sizes {
		// Create identical copies
		data1 := make([]float32, n)
		data2 := make([]float32, n)
		for i := range data1 {
			v := rand.Float32() * 1000
			data1[i] = v
			data2[i] = v
		}

		// Sort with both methods
		Sort(data1)
		slices.Sort(data2)

		// Compare
		for i := range data1 {
			if data1[i] != data2[i] {
				t.Errorf("Sort mismatch at index %d: got %v, want %v", i, data1[i], data2[i])
				break
			}
		}
	}
}

// TestPartition3Way tests 3-way partitioning
func TestPartition3Way(t *testing.T) {
	data := []float32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5}
	pivot := float32(5)

	lt, gt := Partition3Way(data, pivot)

	// Verify partitioning
	for i := range lt {
		if data[i] >= pivot {
			t.Errorf("data[%d]=%v should be < pivot %v", i, data[i], pivot)
		}
	}
	for i := lt; i < gt; i++ {
		if data[i] != pivot {
			t.Errorf("data[%d]=%v should be == pivot %v", i, data[i], pivot)
		}
	}
	for i := gt; i < len(data); i++ {
		if data[i] <= pivot {
			t.Errorf("data[%d]=%v should be > pivot %v", i, data[i], pivot)
		}
	}
}

// TestPartition tests 2-way partitioning
func TestPartition(t *testing.T) {
	data := []float32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5}
	pivot := float32(5)

	idx := Partition(data, pivot)

	// Verify partitioning: data[0:idx] < pivot, data[idx:n] >= pivot
	for i := range idx {
		if data[i] >= pivot {
			t.Errorf("data[%d]=%v should be < pivot %v", i, data[i], pivot)
		}
	}
	for i := idx; i < len(data); i++ {
		if data[i] < pivot {
			t.Errorf("data[%d]=%v should be >= pivot %v", i, data[i], pivot)
		}
	}
}

// TestNthElement tests partial sorting
func TestNthElement(t *testing.T) {
	// Create sorted reference
	ref := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	for k := range ref {
		// Shuffle data
		data := make([]float32, len(ref))
		copy(data, ref)
		rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })

		NthElement(data, k)

		if data[k] != ref[k] {
			t.Errorf("NthElement(k=%d): got %v, want %v", k, data[k], ref[k])
		}
	}
}

// TestIsSorted tests the IsSorted function
func TestIsSorted(t *testing.T) {
	tests := []struct {
		name string
		data []float32
		want bool
	}{
		{"empty", []float32{}, true},
		{"single", []float32{1}, true},
		{"sorted", []float32{1, 2, 3, 4, 5}, true},
		{"unsorted", []float32{1, 3, 2, 4, 5}, false},
		{"reverse", []float32{5, 4, 3, 2, 1}, false},
		{"equal", []float32{3, 3, 3, 3}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsSorted(tt.data)
			if got != tt.want {
				t.Errorf("IsSorted(%v) = %v, want %v", tt.data, got, tt.want)
			}
		})
	}
}

// TestSortSmall tests the sorting network for small arrays
func TestSortSmall(t *testing.T) {
	sizes := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 32}
	for _, n := range sizes {
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(n - i) // Reverse order
		}
		SortSmall(data)
		if !isSorted(data) {
			t.Errorf("SortSmall(n=%d) produced unsorted result: %v", n, data)
		}
	}
}

// TestPivotSampled tests pivot selection
func TestPivotSampled(t *testing.T) {
	// For sorted data, sampled pivot should be near median
	data := make([]float32, 100)
	for i := range data {
		data[i] = float32(i)
	}

	pivot := PivotSampled(data)
	// Pivot should be somewhere in the middle range
	if pivot < 20 || pivot > 80 {
		t.Errorf("PivotSampled(sorted) = %v, expected near 50", pivot)
	}
}

// TestCompressPartition3WayFloat32 tests compress-based partitioning
func TestCompressPartition3WayFloat32(t *testing.T) {
	tests := []struct {
		name  string
		data  []float32
		pivot float32
	}{
		{"empty", []float32{}, 0},
		{"single_less", []float32{1}, 5},
		{"single_equal", []float32{5}, 5},
		{"single_greater", []float32{9}, 5},
		{"all_less", []float32{1, 2, 3, 4}, 5},
		{"all_greater", []float32{6, 7, 8, 9}, 5},
		{"all_equal", []float32{5, 5, 5, 5}, 5},
		{"mixed", []float32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5}, 5},
		{"random_small", nil, 50},  // will be filled
		{"random_medium", nil, 50}, // will be filled
		{"random_large", nil, 50},  // will be filled
	}

	// Fill random test cases
	rand.Seed(42)
	for i := range tests {
		if tests[i].name == "random_small" {
			tests[i].data = make([]float32, 17)
			for j := range tests[i].data {
				tests[i].data[j] = float32(rand.Intn(100))
			}
		} else if tests[i].name == "random_medium" {
			tests[i].data = make([]float32, 100)
			for j := range tests[i].data {
				tests[i].data[j] = float32(rand.Intn(100))
			}
		} else if tests[i].name == "random_large" {
			tests[i].data = make([]float32, 1000)
			for j := range tests[i].data {
				tests[i].data[j] = float32(rand.Intn(100))
			}
		}
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy to preserve original
			data := make([]float32, len(tt.data))
			copy(data, tt.data)

			lt, gt := CompressPartition3WayFloat32(data, tt.pivot)

			// Verify partitioning
			for i := range lt {
				if data[i] >= tt.pivot {
					t.Errorf("data[%d]=%v should be < pivot %v", i, data[i], tt.pivot)
				}
			}
			for i := lt; i < gt; i++ {
				if data[i] != tt.pivot {
					t.Errorf("data[%d]=%v should be == pivot %v", i, data[i], tt.pivot)
				}
			}
			for i := gt; i < len(data); i++ {
				if data[i] <= tt.pivot {
					t.Errorf("data[%d]=%v should be > pivot %v", i, data[i], tt.pivot)
				}
			}

			// Verify all elements are preserved (same multiset)
			orig := make([]float32, len(tt.data))
			copy(orig, tt.data)
			slices.Sort(orig)
			sorted := make([]float32, len(data))
			copy(sorted, data)
			slices.Sort(sorted)
			for i := range orig {
				if orig[i] != sorted[i] {
					t.Errorf("element mismatch after partition: orig[%d]=%v, got[%d]=%v", i, orig[i], i, sorted[i])
					break
				}
			}
		})
	}
}

// TestRadixSortInt32 tests radix sort for int32
func TestRadixSortInt32(t *testing.T) {
	sizes := []int{0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 100, 256, 1000, 10000}
	for _, n := range sizes {
		data := make([]int32, n)
		for i := range data {
			data[i] = rand.Int31n(1000000) - 500000 // Include negative numbers
		}
		RadixSort(data)
		if !isSorted(data) {
			t.Errorf("RadixSort[int32](n=%d) produced unsorted result", n)
		}
	}
}

// TestRadixSortInt64 tests radix sort for int64
func TestRadixSortInt64(t *testing.T) {
	sizes := []int{0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 100, 256, 1000, 10000}
	for _, n := range sizes {
		data := make([]int64, n)
		for i := range data {
			data[i] = rand.Int63n(1000000) - 500000
		}
		RadixSort(data)
		if !isSorted(data) {
			t.Errorf("RadixSort[int64](n=%d) produced unsorted result", n)
		}
	}
}

// TestRadixSortInt32MatchesStdlib verifies RadixSort produces same result as slices.Sort
func TestRadixSortInt32MatchesStdlib(t *testing.T) {
	rand.Seed(54321)
	sizes := []int{100, 256, 1000, 10000}
	for _, n := range sizes {
		data1 := make([]int32, n)
		data2 := make([]int32, n)
		for i := range data1 {
			v := rand.Int31n(1000000) - 500000
			data1[i] = v
			data2[i] = v
		}

		RadixSort(data1)
		slices.Sort(data2)

		for i := range data1 {
			if data1[i] != data2[i] {
				t.Errorf("RadixSort[int32] mismatch at index %d: got %v, want %v", i, data1[i], data2[i])
				break
			}
		}
	}
}

// TestRadixSortInt32EdgeCases tests edge cases for radix sort
func TestRadixSortInt32EdgeCases(t *testing.T) {
	tests := []struct {
		name string
		data []int32
	}{
		{"all_zeros", []int32{0, 0, 0, 0, 0}},
		{"all_same", []int32{42, 42, 42, 42}},
		{"all_negative", []int32{-5, -3, -8, -1, -9}},
		{"all_positive", []int32{5, 3, 8, 1, 9}},
		{"mixed_signs", []int32{-5, 3, -8, 1, 0, -9, 7}},
		{"min_max", []int32{-2147483648, 2147483647, 0, -1, 1}},
		{"sorted", []int32{1, 2, 3, 4, 5}},
		{"reverse", []int32{5, 4, 3, 2, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := make([]int32, len(tt.data))
			copy(data, tt.data)
			RadixSort(data)
			if !isSorted(data) {
				t.Errorf("RadixSort[int32](%s) produced unsorted result: %v", tt.name, data)
			}
		})
	}
}

// TestCompressPartition3WayMatchesRegular verifies compress partition matches regular partition
func TestCompressPartition3WayMatchesRegular(t *testing.T) {
	rand.Seed(12345)
	sizes := []int{0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 100, 256, 1000}

	for _, n := range sizes {
		// Create data
		data1 := make([]float32, n)
		data2 := make([]float32, n)
		for i := range data1 {
			v := float32(rand.Intn(100))
			data1[i] = v
			data2[i] = v
		}

		pivot := float32(50)

		// Partition with both methods
		lt1, gt1 := Partition3Way(data1, pivot)
		lt2, gt2 := CompressPartition3WayFloat32(data2, pivot)

		// Counts should match
		if lt1 != lt2 || gt1 != gt2 {
			t.Errorf("n=%d: Partition3Way returned (%d,%d), CompressPartition3Way returned (%d,%d)", n, lt1, gt1, lt2, gt2)
		}

		// Both should be valid partitions (already verified in individual tests)
	}
}
