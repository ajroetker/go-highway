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

package vec

//go:generate go run ../../../cmd/hwygen -input argmax_base.go -output . -targets avx2,avx512,neon,fallback -dispatch argmax

import (
	"github.com/ajroetker/go-highway/hwy"
)

// BaseArgmax returns the index of the maximum value in a slice.
// If multiple elements have the maximum value, returns the first occurrence.
// NaN values are treated as less than all other values.
// Panics if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float16, bfloat16, float32, and float64 slices.
//
// Example:
//
//	data := []float32{3, 1, 4, 1, 5}
//	idx := Argmax(data)  // 4 (index of value 5)
func BaseArgmax[T hwy.Floats](v []T) int {
	if len(v) == 0 {
		panic("vec: Argmax called on empty slice")
	}

	lanes := hwy.MaxLanes[T]()

	// For small slices, use scalar implementation
	if len(v) < lanes {
		return scalarArgmax(v)
	}

	// Initialize with first vector
	maxVals := hwy.Load(v)
	maxIdxs := hwy.Iota[T]() // [0, 1, 2, ...]

	// Process remaining full vectors
	i := lanes
	for ; i+lanes <= len(v); i += lanes {
		vals := hwy.LoadFull(v[i:])
		// Current indices: base + iota
		curIdxs := hwy.Add(hwy.Set(T(i)), hwy.Iota[T]())

		// Create mask where new values > current max
		mask := hwy.GreaterThan(vals, maxVals)

		// Update max values and indices where mask is true
		maxVals = hwy.IfThenElse(mask, vals, maxVals)
		maxIdxs = hwy.IfThenElse(mask, curIdxs, maxIdxs)
	}

	// Reduce across lanes (NaN-safe: NaN != NaN skips NaN values)
	valsData := maxVals.Data()
	idxsData := maxIdxs.Data()

	bestIdx := 0
	var maxVal T
	foundValid := false
	for j := 0; j < lanes; j++ {
		val := valsData[j]
		if val != val {
			continue
		}
		idx := int(idxsData[j])
		if !foundValid || val > maxVal || (val == maxVal && idx < bestIdx) {
			maxVal = val
			bestIdx = idx
			foundValid = true
		}
	}

	// Handle tail elements
	for ; i < len(v); i++ {
		if v[i] != v[i] {
			continue
		}
		if !foundValid || v[i] > maxVal || (v[i] == maxVal && i < bestIdx) {
			maxVal = v[i]
			bestIdx = i
			foundValid = true
		}
	}

	return bestIdx
}

// BaseArgmin returns the index of the minimum value in a slice.
// If multiple elements have the minimum value, returns the first occurrence.
// NaN values are treated as greater than all other values.
// Panics if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float16, bfloat16, float32, and float64 slices.
//
// Example:
//
//	data := []float32{3, 1, 4, 1, 5}
//	idx := Argmin(data)  // 1 (index of first value 1)
func BaseArgmin[T hwy.Floats](v []T) int {
	if len(v) == 0 {
		panic("vec: Argmin called on empty slice")
	}

	lanes := hwy.MaxLanes[T]()

	// For small slices, use scalar implementation
	if len(v) < lanes {
		return scalarArgmin(v)
	}

	// Initialize with first vector
	minVals := hwy.Load(v)
	minIdxs := hwy.Iota[T]() // [0, 1, 2, ...]

	// Process remaining full vectors
	i := lanes
	for ; i+lanes <= len(v); i += lanes {
		vals := hwy.LoadFull(v[i:])
		// Current indices: base + iota
		curIdxs := hwy.Add(hwy.Set(T(i)), hwy.Iota[T]())

		// Create mask where new values < current min
		mask := hwy.LessThan(vals, minVals)

		// Update min values and indices where mask is true
		minVals = hwy.IfThenElse(mask, vals, minVals)
		minIdxs = hwy.IfThenElse(mask, curIdxs, minIdxs)
	}

	// Reduce across lanes (NaN-safe: NaN != NaN skips NaN values)
	valsData := minVals.Data()
	idxsData := minIdxs.Data()

	bestIdx := 0
	var minVal T
	foundValid := false
	for j := 0; j < lanes; j++ {
		val := valsData[j]
		if val != val {
			continue
		}
		idx := int(idxsData[j])
		if !foundValid || val < minVal || (val == minVal && idx < bestIdx) {
			minVal = val
			bestIdx = idx
			foundValid = true
		}
	}

	// Handle tail elements
	for ; i < len(v); i++ {
		if v[i] != v[i] {
			continue
		}
		if !foundValid || v[i] < minVal || (v[i] == minVal && i < bestIdx) {
			minVal = v[i]
			bestIdx = i
			foundValid = true
		}
	}

	return bestIdx
}

// scalarArgmax is the scalar fallback for small slices.
// NaN values are treated as less than all other values, matching the SIMD path.
func scalarArgmax[T hwy.Floats](v []T) int {
	bestIdx := 0
	var maxVal T
	foundValid := false
	for i := 0; i < len(v); i++ {
		if v[i] != v[i] {
			continue // skip NaN
		}
		if !foundValid || v[i] > maxVal || (v[i] == maxVal && i < bestIdx) {
			maxVal = v[i]
			bestIdx = i
			foundValid = true
		}
	}
	return bestIdx
}

// scalarArgmin is the scalar fallback for small slices.
// NaN values are treated as greater than all other values, matching the SIMD path.
func scalarArgmin[T hwy.Floats](v []T) int {
	bestIdx := 0
	var minVal T
	foundValid := false
	for i := 0; i < len(v); i++ {
		if v[i] != v[i] {
			continue // skip NaN
		}
		if !foundValid || v[i] < minVal || (v[i] == minVal && i < bestIdx) {
			minVal = v[i]
			bestIdx = i
			foundValid = true
		}
	}
	return bestIdx
}
