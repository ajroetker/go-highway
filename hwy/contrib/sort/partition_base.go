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

//go:generate go run ../../../cmd/hwygen -input partition_base.go -output . -targets avx2,avx512,neon,fallback -dispatch partition

// BasePartition3Way performs 3-way partitioning around a pivot.
// Returns (lt, gt) indices where:
//   - data[0:lt] < pivot
//   - data[lt:gt] == pivot
//   - data[gt:n] > pivot
func BasePartition3Way[T hwy.Lanes](data []T, pivot T) (int, int) {
	n := len(data)
	if n == 0 {
		return 0, 0
	}

	lanes := hwy.MaxLanes[T]()

	// For small arrays, use scalar directly
	if n < lanes*4 {
		return scalarPartition3Way(data, pivot)
	}

	pivotVec := hwy.Set(pivot)

	lt := 0
	gt := n
	i := 0

	// Main loop with SIMD fast path
	for i+lanes <= gt {
		// Check overlap condition for gt swaps
		if gt-lanes < i+lanes {
			break // Too close, finish with scalar
		}

		v := hwy.LoadFull(data[i:])
		maskLess := hwy.LessThan(v, pivotVec)
		maskGreater := hwy.GreaterThan(v, pivotVec)

		// All elements < pivot (use AllTrue instead of CountTrue for speed)
		if hwy.AllTrue(maskLess) {
			if lt == i {
				lt += lanes
				i += lanes
				continue
			}
			if lt+lanes <= i {
				vLt := hwy.LoadFull(data[lt:])
				hwy.StoreFull(v, data[lt:])
				hwy.StoreFull(vLt, data[i:])
				lt += lanes
				i += lanes
				continue
			}
			break // Overlap with lt region
		}

		// All elements > pivot
		if hwy.AllTrue(maskGreater) {
			gt -= lanes
			vGt := hwy.LoadFull(data[gt:])
			hwy.StoreFull(v, data[gt:])
			hwy.StoreFull(vGt, data[i:])
			continue
		}

		// All elements == pivot (neither < nor >)
		if hwy.AllFalse(maskLess) && hwy.AllFalse(maskGreater) {
			i += lanes
			continue
		}

		// Mixed vector: process this vector's elements via scalar, then continue SIMD
		end := min(i+lanes, gt)
		for i < end {
			if data[i] < pivot {
				data[lt], data[i] = data[i], data[lt]
				lt++
				i++
			} else if data[i] > pivot {
				gt--
				data[i], data[gt] = data[gt], data[i]
				if gt < end {
					end = gt
				}
			} else {
				i++
			}
		}
	}

	// Finish remaining elements with scalar
	for i < gt {
		if data[i] < pivot {
			data[lt], data[i] = data[i], data[lt]
			lt++
			i++
		} else if data[i] > pivot {
			gt--
			data[i], data[gt] = data[gt], data[i]
		} else {
			i++
		}
	}

	return lt, gt
}

// BasePartition performs 2-way partitioning around a pivot.
// Returns index where data[0:idx] <= pivot and data[idx:n] > pivot.
func BasePartition[T hwy.Lanes](data []T, pivot T) int {
	n := len(data)
	if n == 0 {
		return 0
	}

	lanes := hwy.MaxLanes[T]()

	if n < lanes*4 {
		return scalarPartition2Way(data, pivot)
	}

	pivotVec := hwy.Set(pivot)

	left := 0
	right := n

	// Main loop with SIMD fast path
	for left+lanes <= right {
		if right-lanes < left+lanes {
			break
		}

		v := hwy.LoadFull(data[left:])
		mask := hwy.LessEqual(v, pivotVec)

		// All elements <= pivot
		if hwy.AllTrue(mask) {
			left += lanes
			continue
		}

		// All elements > pivot
		if hwy.AllFalse(mask) {
			right -= lanes
			vRight := hwy.LoadFull(data[right:])
			hwy.StoreFull(v, data[right:])
			hwy.StoreFull(vRight, data[left:])
			continue
		}

		// Mixed: process this vector via scalar
		end := min(left+lanes, right)
		for left < end {
			if data[left] <= pivot {
				left++
			} else {
				right--
				data[left], data[right] = data[right], data[left]
				if right < end {
					end = right
				}
			}
		}
	}

	// Finish remaining
	for left < right {
		if data[left] <= pivot {
			left++
		} else {
			right--
			data[left], data[right] = data[right], data[left]
		}
	}

	return left
}
