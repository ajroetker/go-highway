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

//go:generate go run ../../../cmd/hwygen -input find_base.go -output . -targets avx2,avx512,neon,fallback -dispatch find

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
		v := hwy.LoadFull(slice[i:])
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
		v := hwy.LoadFull(slice[i:])
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

// BaseAll returns true if pred returns true for all elements.
// Short-circuits on first false.
// The predicate P must implement Predicate[T] interface.
func BaseAll[T hwy.Lanes, P Predicate[T]](slice []T, pred P) bool {
	n := len(slice)
	if n == 0 {
		return true
	}

	lanes := hwy.MaxLanes[T]()
	i := 0

	// Process full vectors
	for ; i+lanes <= n; i += lanes {
		v := hwy.LoadFull(slice[i:])
		mask := pred.Apply(v)
		if !hwy.AllTrue(mask) {
			return false
		}
	}

	// Handle tail elements
	if remaining := n - i; remaining > 0 {
		buf := make([]T, lanes)
		copy(buf, slice[i:i+remaining])
		v := hwy.Load(buf)
		mask := pred.Apply(v)

		// Check only valid elements in the tail
		tailMask := hwy.FirstN[T](remaining)
		// For "all" we need: every valid element passes
		// That means: mask OR (NOT tailMask) should be all true
		// Or equivalently: NOT(tailMask AND NOT(mask)) should be all true
		inverted := hwy.MaskAndNot(mask, tailMask)
		if !hwy.AllFalse(inverted) {
			return false
		}
	}

	return true
}

// BaseAny returns true if pred returns true for any element.
// Short-circuits on first true.
// The predicate P must implement Predicate[T] interface.
func BaseAny[T hwy.Lanes, P Predicate[T]](slice []T, pred P) bool {
	n := len(slice)
	if n == 0 {
		return false
	}

	lanes := hwy.MaxLanes[T]()
	i := 0

	// Process full vectors
	for ; i+lanes <= n; i += lanes {
		v := hwy.LoadFull(slice[i:])
		mask := pred.Apply(v)
		if idx := hwy.FindFirstTrue(mask); idx >= 0 {
			return true
		}
	}

	// Handle tail elements
	if remaining := n - i; remaining > 0 {
		buf := make([]T, lanes)
		copy(buf, slice[i:i+remaining])
		v := hwy.Load(buf)
		mask := pred.Apply(v)

		tailMask := hwy.FirstN[T](remaining)
		mask = hwy.MaskAnd(mask, tailMask)
		if idx := hwy.FindFirstTrue(mask); idx >= 0 {
			return true
		}
	}

	return false
}

// BaseNone returns true if pred returns false for all elements.
// This is equivalent to !BaseAny(slice, pred).
// The predicate P must implement Predicate[T] interface.
func BaseNone[T hwy.Lanes, P Predicate[T]](slice []T, pred P) bool {
	return !BaseAny(slice, pred)
}

// BaseFindIf returns the index of the first element where pred returns true.
// Returns -1 if no element matches.
// The predicate P must implement Predicate[T] interface.
func BaseFindIf[T hwy.Lanes, P Predicate[T]](slice []T, pred P) int {
	n := len(slice)
	if n == 0 {
		return -1
	}

	lanes := hwy.MaxLanes[T]()
	i := 0

	// Process full vectors
	for ; i+lanes <= n; i += lanes {
		v := hwy.LoadFull(slice[i:])
		mask := pred.Apply(v)
		if idx := hwy.FindFirstTrue(mask); idx >= 0 {
			return i + idx
		}
	}

	// Handle tail elements
	if remaining := n - i; remaining > 0 {
		buf := make([]T, lanes)
		copy(buf, slice[i:i+remaining])
		v := hwy.Load(buf)
		mask := pred.Apply(v)

		// Only check valid indices in the tail
		if idx := hwy.FindFirstTrue(mask); idx >= 0 && idx < remaining {
			return i + idx
		}
	}

	return -1
}

// BaseCountIf returns the number of elements where pred returns true.
// The predicate P must implement Predicate[T] interface.
func BaseCountIf[T hwy.Lanes, P Predicate[T]](slice []T, pred P) int {
	n := len(slice)
	if n == 0 {
		return 0
	}

	lanes := hwy.MaxLanes[T]()
	count := 0
	i := 0

	// Process full vectors
	for ; i+lanes <= n; i += lanes {
		v := hwy.LoadFull(slice[i:])
		mask := pred.Apply(v)
		count += hwy.CountTrue(mask)
	}

	// Handle tail elements
	if remaining := n - i; remaining > 0 {
		buf := make([]T, lanes)
		copy(buf, slice[i:i+remaining])
		v := hwy.Load(buf)
		mask := pred.Apply(v)

		// Only count valid elements in the tail
		tailMask := hwy.FirstN[T](remaining)
		mask = hwy.MaskAnd(mask, tailMask)
		count += hwy.CountTrue(mask)
	}

	return count
}
