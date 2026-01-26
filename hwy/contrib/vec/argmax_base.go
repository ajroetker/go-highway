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

import (
	"math"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseArgmaxFloat32 returns the index of the maximum value in a float32 slice.
// If multiple elements have the maximum value, returns the first occurrence.
// NaN values are skipped.
// Panics if the slice is empty.
func BaseArgmaxFloat32(v []float32) int {
	if len(v) == 0 {
		panic("vec: Argmax called on empty slice")
	}

	// Find first non-NaN position
	start := 0
	for ; start < len(v); start++ {
		if !math.IsNaN(float64(v[start])) {
			break
		}
	}
	if start >= len(v) {
		return 0 // All NaN
	}

	maxIdx := start
	maxVal := v[start]
	for i := start + 1; i < len(v); i++ {
		if !math.IsNaN(float64(v[i])) && v[i] > maxVal {
			maxVal = v[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// BaseArgmaxFloat64 returns the index of the maximum value in a float64 slice.
func BaseArgmaxFloat64(v []float64) int {
	if len(v) == 0 {
		panic("vec: Argmax called on empty slice")
	}

	start := 0
	for ; start < len(v); start++ {
		if !math.IsNaN(v[start]) {
			break
		}
	}
	if start >= len(v) {
		return 0
	}

	maxIdx := start
	maxVal := v[start]
	for i := start + 1; i < len(v); i++ {
		if !math.IsNaN(v[i]) && v[i] > maxVal {
			maxVal = v[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// BaseArgmaxFloat16 returns the index of the maximum value in a Float16 slice.
func BaseArgmaxFloat16(v []hwy.Float16) int {
	if len(v) == 0 {
		panic("vec: Argmax called on empty slice")
	}

	start := 0
	for ; start < len(v); start++ {
		f32 := v[start].Float32()
		if !math.IsNaN(float64(f32)) {
			break
		}
	}
	if start >= len(v) {
		return 0
	}

	maxIdx := start
	maxVal := v[start].Float32()
	for i := start + 1; i < len(v); i++ {
		val := v[i].Float32()
		if !math.IsNaN(float64(val)) && val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}

// BaseArgmaxBFloat16 returns the index of the maximum value in a BFloat16 slice.
func BaseArgmaxBFloat16(v []hwy.BFloat16) int {
	if len(v) == 0 {
		panic("vec: Argmax called on empty slice")
	}

	start := 0
	for ; start < len(v); start++ {
		f32 := v[start].Float32()
		if !math.IsNaN(float64(f32)) {
			break
		}
	}
	if start >= len(v) {
		return 0
	}

	maxIdx := start
	maxVal := v[start].Float32()
	for i := start + 1; i < len(v); i++ {
		val := v[i].Float32()
		if !math.IsNaN(float64(val)) && val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}

// BaseArgminFloat32 returns the index of the minimum value in a float32 slice.
// If multiple elements have the minimum value, returns the first occurrence.
// NaN values are skipped.
// Panics if the slice is empty.
func BaseArgminFloat32(v []float32) int {
	if len(v) == 0 {
		panic("vec: Argmin called on empty slice")
	}

	start := 0
	for ; start < len(v); start++ {
		if !math.IsNaN(float64(v[start])) {
			break
		}
	}
	if start >= len(v) {
		return 0
	}

	minIdx := start
	minVal := v[start]
	for i := start + 1; i < len(v); i++ {
		if !math.IsNaN(float64(v[i])) && v[i] < minVal {
			minVal = v[i]
			minIdx = i
		}
	}
	return minIdx
}

// BaseArgminFloat64 returns the index of the minimum value in a float64 slice.
func BaseArgminFloat64(v []float64) int {
	if len(v) == 0 {
		panic("vec: Argmin called on empty slice")
	}

	start := 0
	for ; start < len(v); start++ {
		if !math.IsNaN(v[start]) {
			break
		}
	}
	if start >= len(v) {
		return 0
	}

	minIdx := start
	minVal := v[start]
	for i := start + 1; i < len(v); i++ {
		if !math.IsNaN(v[i]) && v[i] < minVal {
			minVal = v[i]
			minIdx = i
		}
	}
	return minIdx
}

// BaseArgminFloat16 returns the index of the minimum value in a Float16 slice.
func BaseArgminFloat16(v []hwy.Float16) int {
	if len(v) == 0 {
		panic("vec: Argmin called on empty slice")
	}

	start := 0
	for ; start < len(v); start++ {
		f32 := v[start].Float32()
		if !math.IsNaN(float64(f32)) {
			break
		}
	}
	if start >= len(v) {
		return 0
	}

	minIdx := start
	minVal := v[start].Float32()
	for i := start + 1; i < len(v); i++ {
		val := v[i].Float32()
		if !math.IsNaN(float64(val)) && val < minVal {
			minVal = val
			minIdx = i
		}
	}
	return minIdx
}

// BaseArgminBFloat16 returns the index of the minimum value in a BFloat16 slice.
func BaseArgminBFloat16(v []hwy.BFloat16) int {
	if len(v) == 0 {
		panic("vec: Argmin called on empty slice")
	}

	start := 0
	for ; start < len(v); start++ {
		f32 := v[start].Float32()
		if !math.IsNaN(float64(f32)) {
			break
		}
	}
	if start >= len(v) {
		return 0
	}

	minIdx := start
	minVal := v[start].Float32()
	for i := start + 1; i < len(v); i++ {
		val := v[i].Float32()
		if !math.IsNaN(float64(val)) && val < minVal {
			minVal = val
			minIdx = i
		}
	}
	return minIdx
}
