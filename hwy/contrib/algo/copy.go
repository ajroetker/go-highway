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

// Fill sets all elements in dst to the specified value.
// Uses an efficient doubling pattern that leverages Go's optimized memmove.
func Fill[T hwy.Lanes](dst []T, value T) {
	n := len(dst)
	if n == 0 {
		return
	}

	// Set first element
	dst[0] = value

	// Double the filled region each iteration
	// This is O(log n) calls to copy, and copy is highly optimized
	for filled := 1; filled < n; filled *= 2 {
		copy(dst[filled:], dst[:filled])
	}
}

// Copy copies elements from src to dst.
// Uses the built-in copy which is already highly optimized (memmove).
// Returns the number of elements copied (min of len(src) and len(dst)).
func Copy[T hwy.Lanes](src, dst []T) int {
	return copy(dst, src)
}

