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

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input network_base.go -output . -targets avx2,avx512,neon,fallback -dispatch network

// BaseSortSmall sorts a small slice in-place using sorting networks.
// For slices up to 2*lanes elements, uses optimized sorting.
// For larger slices, falls back to insertion sort.
func BaseSortSmall[T hwy.Lanes](data []T) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Use insertion sort for very small arrays
	if n <= 4 {
		InsertionSortSmall(data)
		return
	}

	lanes := hwy.MaxLanes[T]()

	// Single vector case
	if n <= lanes {
		SortSingleVector(data)
		return
	}

	// Two vectors case
	if n <= lanes*2 {
		SortTwoVectors(data)
		return
	}

	// Larger arrays: use insertion sort
	InsertionSortSmall(data)
}

// BaseIsSorted checks if a slice is sorted in ascending order.
func BaseIsSorted[T hwy.Lanes](data []T) bool {
	n := len(data)
	if n <= 1 {
		return true
	}

	lanes := hwy.MaxLanes[T]()
	i := 0

	// Process full vectors: compare adjacent pairs
	for ; i+lanes < n; i += lanes {
		v1 := hwy.LoadFull(data[i:])
		v2 := hwy.LoadFull(data[i+1:])

		// Check if any element in v1 is greater than corresponding element in v2
		mask := hwy.GreaterThan(v1, v2)
		if hwy.FindFirstTrue(mask) >= 0 {
			return false
		}
	}

	// Handle tail
	for ; i < n-1; i++ {
		if data[i] > data[i+1] {
			return false
		}
	}

	return true
}
