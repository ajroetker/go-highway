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

//go:generate go run ../../../cmd/hwygen -input prefix_sum_base.go -output . -targets avx2,avx512,neon,fallback -dispatch prefix_sum

// BasePrefixSum computes the inclusive prefix sum in place.
// Result[i] = data[0] + data[1] + ... + data[i]
//
// Example:
//
//	data := []int64{1, 2, 3, 4, 5, 6, 7, 8}
//	BasePrefixSum(data)
//	// data = [1, 3, 6, 10, 15, 21, 28, 36]
//
// If you need to preserve the original, copy first:
//
//	result := slices.Clone(src)
//	BasePrefixSum(result)
func BasePrefixSum[T hwy.Integers | hwy.FloatsNative](data []T) {
	n := len(data)
	if n == 0 {
		return
	}

	lanes := hwy.MaxLanes[T]()
	carry := T(0)
	i := 0

	for ; i+lanes <= n; i += lanes {
		v := hwy.Load(data[i:])
		prefixed := BasePrefixSumVec(v)
		prefixed = hwy.Add(prefixed, hwy.Set[T](carry))
		hwy.Store(prefixed, data[i:])
		carry = hwy.GetLane(prefixed, lanes-1)
	}

	for ; i < n; i++ {
		carry += data[i]
		data[i] = carry
	}
}

// BaseDeltaDecode decodes delta-encoded values in place.
// Each value represents a delta from the previous value.
// Result[i] = base + data[0] + data[1] + ... + data[i]
//
// This is equivalent to PrefixSum with a base offset, commonly used
// for decoding posting lists in full-text search indexes.
//
// Example:
//
//	// Encoded deltas: [3, 2, 5, 1] with base=10
//	// Represents document IDs: 13, 15, 20, 21
//	data := []uint64{3, 2, 5, 1}
//	BaseDeltaDecode(data, 10)
//	// data = [13, 15, 20, 21]
//
// If you need to preserve the original, copy first:
//
//	result := slices.Clone(src)
//	BaseDeltaDecode(result, base)
func BaseDeltaDecode[T hwy.Integers](data []T, base T) {
	n := len(data)
	if n == 0 {
		return
	}

	lanes := hwy.MaxLanes[T]()
	carry := base
	i := 0

	for ; i+lanes <= n; i += lanes {
		v := hwy.Load(data[i:])
		prefixed := BasePrefixSumVec(v)
		prefixed = hwy.Add(prefixed, hwy.Set[T](carry))
		hwy.Store(prefixed, data[i:])
		carry = hwy.GetLane(prefixed, lanes-1)
	}

	for ; i < n; i++ {
		carry += data[i]
		data[i] = carry
	}
}

// BasePrefixSumVec computes the inclusive prefix sum within a single vector
// using the Hillis-Steele algorithm.
//
// For a vector [a, b, c, d]:
//   - Step 1: shift by 1, add -> [a, a+b, b+c, c+d]
//   - Step 2: shift by 2, add -> [a, a+b, a+b+c, a+b+c+d]
//
// The algorithm generalizes to any power-of-2 vector width.
// Steps are explicit (not a loop) so hwygen generates unrolled code.
func BasePrefixSumVec[T hwy.Integers | hwy.FloatsNative](v hwy.Vec[T]) hwy.Vec[T] {
	n := v.NumLanes()

	// Step 1: shift by 1 and add (always needed for n >= 2)
	if n >= 2 {
		v = hwy.Add(v, hwy.SlideUpLanes(v, 1))
	}

	// Step 2: shift by 2 and add (needed for n >= 4)
	if n >= 4 {
		v = hwy.Add(v, hwy.SlideUpLanes(v, 2))
	}

	// Step 3: shift by 4 and add (needed for n >= 8)
	if n >= 8 {
		v = hwy.Add(v, hwy.SlideUpLanes(v, 4))
	}

	// Step 4: shift by 8 and add (needed for n >= 16, e.g., AVX-512 Float32x16)
	if n >= 16 {
		v = hwy.Add(v, hwy.SlideUpLanes(v, 8))
	}

	return v
}
