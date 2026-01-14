package sort

import "github.com/ajroetker/go-highway/hwy"

// Thresholds for different sorting strategies.
const (
	// sortNetworkThreshold: use sorting network for arrays this size or smaller.
	sortNetworkThreshold = 32

	// sortInsertionThreshold: use insertion sort for arrays this size or smaller.
	sortInsertionThreshold = 64
)

// Sort sorts data in-place using the best algorithm for the type:
//   - Signed integers (int32, int64): SIMD radix sort for large arrays, stdlib for small
//   - Floats (float32, float64): VQSort (vectorized quicksort)
//
// For explicit algorithm selection, use VQSort or RadixSort directly.
func Sort[T hwy.Lanes](data []T) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Dispatch to radix sort for all types (O(n) vs O(n log n))
	var zero T
	switch any(zero).(type) {
	case float32:
		RadixSortFloat32(any(data).([]float32))
	case float64:
		RadixSortFloat64(any(data).([]float64))
	case int32:
		RadixSort(any(data).([]int32))
	case int64:
		RadixSort(any(data).([]int64))
	}
}

// VQSort sorts data in-place using vectorized quicksort.
// This is an introsort variant that combines:
//   - Sorting networks for very small arrays
//   - Vectorized quicksort partitioning for larger arrays
//   - Heapsort fallback for worst-case guarantee
//
// Supported types: float32, float64, int32, int64
func VQSort[T hwy.Lanes](data []T) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Calculate max recursion depth: 2 * floor(log2(n))
	maxDepth := 0
	for tmp := n; tmp > 0; tmp >>= 1 {
		maxDepth++
	}
	maxDepth *= 2

	sortImpl(data, maxDepth)
}

// sortImpl is the recursive implementation of VQSort.
func sortImpl[T hwy.Lanes](data []T, depthLimit int) {
	n := len(data)

	if n <= 1 {
		return
	}

	// Use sorting network for very small arrays
	if n <= sortNetworkThreshold {
		SortSmall(data)
		return
	}

	// Use insertion sort for small arrays
	if n <= sortInsertionThreshold {
		sortInsertion(data)
		return
	}

	// Fallback to heapsort if recursion too deep
	if depthLimit == 0 {
		sortHeap(data)
		return
	}

	// Select pivot using sampled median
	pivot := PivotSampled(data)

	// Partition using vectorized 3-way partition
	lt, gt := CompressPartition3Way(data, pivot)

	// Recurse on partitions
	if lt > 0 {
		sortImpl(data[:lt], depthLimit-1)
	}
	if gt < n {
		sortImpl(data[gt:], depthLimit-1)
	}
}

// sortInsertion is insertion sort for small arrays.
func sortInsertion[T hwy.Lanes](data []T) {
	for i := 1; i < len(data); i++ {
		key := data[i]
		j := i - 1
		for j >= 0 && data[j] > key {
			data[j+1] = data[j]
			j--
		}
		data[j+1] = key
	}
}

// sortHeap is heapsort for O(n log n) worst-case guarantee.
func sortHeap[T hwy.Lanes](data []T) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Build max-heap
	for i := n/2 - 1; i >= 0; i-- {
		siftDown(data, i, n)
	}

	// Extract elements
	for i := n - 1; i > 0; i-- {
		data[0], data[i] = data[i], data[0]
		siftDown(data, 0, i)
	}
}

func siftDown[T hwy.Lanes](data []T, i, n int) {
	for {
		largest := i
		left := 2*i + 1
		right := 2*i + 2

		if left < n && data[left] > data[largest] {
			largest = left
		}
		if right < n && data[right] > data[largest] {
			largest = right
		}

		if largest == i {
			break
		}

		data[i], data[largest] = data[largest], data[i]
		i = largest
	}
}

// NthElement rearranges data such that the element at index k
// is the element that would be at that position if data were sorted.
// Elements before k are <= data[k], elements after are >= data[k].
func NthElement[T hwy.Lanes](data []T, k int) {
	n := len(data)
	if k < 0 || k >= n {
		return
	}

	// Calculate max depth
	maxDepth := 0
	for tmp := n; tmp > 0; tmp >>= 1 {
		maxDepth++
	}
	maxDepth *= 2

	nthElementImpl(data, k, maxDepth)
}

func nthElementImpl[T hwy.Lanes](data []T, k, depthLimit int) {
	n := len(data)
	if n <= 1 {
		return
	}

	if depthLimit == 0 || n <= sortInsertionThreshold {
		VQSort(data)
		return
	}

	pivot := PivotSampled(data)
	lt, gt := CompressPartition3Way(data, pivot)

	if k < lt {
		nthElementImpl(data[:lt], k, depthLimit-1)
	} else if k >= gt {
		nthElementImpl(data[gt:], k-gt, depthLimit-1)
	}
	// If lt <= k < gt, k is in the equal partition - done
}
