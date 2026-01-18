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

		v := hwy.Load(data[i:])
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
				vLt := hwy.Load(data[lt:])
				hwy.Store(v, data[lt:])
				hwy.Store(vLt, data[i:])
				lt += lanes
				i += lanes
				continue
			}
			break // Overlap with lt region
		}

		// All elements > pivot
		if hwy.AllTrue(maskGreater) {
			gt -= lanes
			vGt := hwy.Load(data[gt:])
			hwy.Store(v, data[gt:])
			hwy.Store(vGt, data[i:])
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

		v := hwy.Load(data[left:])
		mask := hwy.LessEqual(v, pivotVec)

		// All elements <= pivot
		if hwy.AllTrue(mask) {
			left += lanes
			continue
		}

		// All elements > pivot
		if hwy.AllFalse(mask) {
			right -= lanes
			vRight := hwy.Load(data[right:])
			hwy.Store(v, data[right:])
			hwy.Store(vRight, data[left:])
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
