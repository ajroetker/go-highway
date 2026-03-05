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

package algo

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input find_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch find

// BaseFind returns the index of the first element equal to value, or -1 if not found.
// Uses SIMD comparison for efficient searching.
func BaseFind[T hwy.Lanes](slice []T, value T) int {
	n := len(slice)
	if n == 0 {
		return -1
	}

	target := hwy.Set(value)
	lanes := hwy.MaxLanes[T]()
	i := 0

	// Process full vectors - compare lanes elements at once
	for ; i+lanes <= n; i += lanes {
		v := hwy.Load(slice[i:])
		mask := hwy.Equal(v, target)
		if idx := hwy.FindFirstTrue(mask); idx >= 0 {
			return i + idx
		}
	}

	// Handle tail elements
	for ; i < n; i++ {
		if slice[i] == value {
			return i
		}
	}

	return -1
}

// BaseCount returns the number of elements equal to target.
// Uses SIMD comparison and popcount for efficiency.
func BaseCount[T hwy.Lanes](slice []T, value T) int {
	n := len(slice)
	if n == 0 {
		return 0
	}

	target := hwy.Set(value)
	lanes := hwy.MaxLanes[T]()
	count := 0
	i := 0

	// Process full vectors
	for ; i+lanes <= n; i += lanes {
		v := hwy.Load(slice[i:])
		mask := hwy.Equal(v, target)
		count += hwy.CountTrue(mask)
	}

	// Handle tail elements
	for ; i < n; i++ {
		if slice[i] == value {
			count++
		}
	}

	return count
}

// BaseContains returns true if slice contains the specified value.
// This is a convenience wrapper around BaseFind.
func BaseContains[T hwy.Lanes](slice []T, value T) bool {
	return BaseFind(slice, value) >= 0
}

