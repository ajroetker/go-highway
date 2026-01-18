package sort

import (
	"slices"
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// Helper functions for sorting that are shared across all implementations.
// These are not processed by hwygen - they remain generic.

// InsertionSortSmall is a simple insertion sort for small arrays.
func InsertionSortSmall[T hwy.Lanes](data []T) {
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

// MaxValue returns the maximum representable value for the type.
func MaxValue[T hwy.Lanes]() T {
	var maxVal T
	switch any(maxVal).(type) {
	case float32:
		maxVal = T(any(float32(3.4028235e+38)).(float32))
	case float64:
		maxVal = T(any(float64(1.7976931348623157e+308)).(float64))
	case int32:
		maxVal = T(any(int32(2147483647)).(int32))
	case int64:
		maxVal = T(any(int64(9223372036854775807)).(int64))
	}
	return maxVal
}

// BitonicMerge performs in-place bitonic merge on a slice.
func BitonicMerge[T hwy.Lanes](data []T) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Bitonic merge network
	for k := n / 2; k > 0; k /= 2 {
		for i := range n {
			j := i ^ k
			if j > i && data[i] > data[j] {
				data[i], data[j] = data[j], data[i]
			}
		}
	}
}

// SortSingleVector sorts elements that fit in one vector.
func SortSingleVector[T hwy.Lanes](data []T) {
	n := len(data)
	if n <= 1 {
		return
	}

	lanes := hwy.MaxLanes[T]()

	// Pad to full vector
	buf := make([]T, lanes)
	copy(buf, data)

	// Fill remaining with max value to push to end
	maxVal := MaxValue[T]()
	for i := n; i < lanes; i++ {
		buf[i] = maxVal
	}

	// Sort using insertion sort (simple for base impl)
	InsertionSortSmall(buf)

	// Copy back valid elements
	copy(data, buf[:n])
}

// PivotMedianOf3 selects pivot as median of first, middle, and last elements.
func PivotMedianOf3[T hwy.Lanes](data []T) T {
	n := len(data)
	if n <= 2 {
		return data[0]
	}

	a := data[0]
	b := data[n/2]
	c := data[n-1]

	if a > b {
		a, b = b, a
	}
	if b > c {
		b = c
		if a > b {
			b = a
		}
	}
	return b
}

// PivotSampled selects pivot by sampling elements at regular intervals.
// For larger arrays, this gives a better pivot estimate than median-of-3.
func PivotSampled[T hwy.Lanes](data []T) T {
	n := len(data)
	if n <= 8 {
		return PivotMedianOf3(data)
	}

	samples := []T{
		data[0],
		data[n/4],
		data[n/2],
		data[3*n/4],
		data[n-1],
	}

	InsertionSortSmall(samples)
	return samples[2]
}

// scalarPartition3Way performs scalar 3-way partitioning (Dutch National Flag).
// Used as fallback for small arrays.
func scalarPartition3Way[T hwy.Lanes](data []T, pivot T) (int, int) {
	lt := 0
	gt := len(data)
	i := 0

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

// scalarPartition2Way performs scalar 2-way partitioning.
// Used as fallback for small arrays.
func scalarPartition2Way[T hwy.Lanes](data []T, pivot T) int {
	left := 0
	right := len(data)

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

// SortTwoVectors sorts elements that fit in two vectors using bitonic merge.
func SortTwoVectors[T hwy.Lanes](data []T) {
	n := len(data)
	if n <= 1 {
		return
	}

	lanes := hwy.MaxLanes[T]()

	// Pad both vectors
	buf1 := make([]T, lanes)
	buf2 := make([]T, lanes)

	copy(buf1, data)
	if n > lanes {
		copy(buf2, data[lanes:])
	}

	// Fill with max value
	maxVal := MaxValue[T]()

	// Pad buf1 if needed
	for i := n; i < lanes; i++ {
		buf1[i] = maxVal
	}

	// Pad buf2
	remaining := max(n-lanes, 0)
	for i := remaining; i < lanes; i++ {
		buf2[i] = maxVal
	}

	// Sort each half
	InsertionSortSmall(buf1)
	InsertionSortSmall(buf2)

	// Create bitonic sequence: first half ascending, second half descending
	// Bitonic merge requires input to be bitonic (increasing then decreasing)
	merged := make([]T, lanes*2)
	copy(merged[:lanes], buf1)
	// Reverse buf2 into the second half to create bitonic sequence
	for i := range lanes {
		merged[lanes+i] = buf2[lanes-1-i]
	}
	BitonicMerge(merged)

	// Copy back
	copy(data, merged[:n])
}

// radixSortThreshold is the minimum size where radix sort beats stdlib.
// Below this, we use slices.Sort which has zero allocations.
// 16K chosen as crossover point for both int32 (8-bit radix) and int64 (16-bit radix).
const radixSortThreshold = 16000

// RadixSort sorts signed integer data using LSD radix sort.
// This is an O(n) stable sort that's faster than comparison sorts for large arrays.
// For arrays smaller than 16K elements, uses stdlib which is faster due to zero allocations.
func RadixSort[T hwy.SignedInts](data []T) {
	n := len(data)
	if n <= 1 {
		return
	}

	// For small arrays, stdlib is faster (no allocation overhead)
	if n < radixSortThreshold {
		slices.Sort(data)
		return
	}

	// Allocate temp buffer
	temp := make([]T, n)

	// Determine number of passes based on type size
	var zero T
	switch any(zero).(type) {
	case int32:
		// LSD radix sort: 4 passes of 8 bits each
		RadixPass(data, temp, 0)
		RadixPass(temp, data, 8)
		RadixPass(data, temp, 16)
		RadixPassSigned(temp, data, 24)
	case int64:
		// LSD radix sort: 4 passes of 16 bits each (faster than 8 passes of 8 bits)
		RadixPass16(data, temp, 0)
		RadixPass16(temp, data, 16)
		RadixPass16(data, temp, 32)
		RadixPass16Signed(temp, data, 48)
	}
}

// RadixSortFloat32 sorts float32 data using radix sort.
// Floats are converted to sortable uint32, radix sorted, then converted back.
func RadixSortFloat32(data []float32) {
	n := len(data)
	if n <= 1 {
		return
	}

	if n < radixSortThreshold {
		slices.Sort(data)
		return
	}

	// Reinterpret as uint32 for bit manipulation
	udata := unsafe.Slice((*uint32)(unsafe.Pointer(&data[0])), n)

	// Transform floats to sortable uint32
	// Positive floats: flip sign bit (0x80000000)
	// Negative floats: flip all bits (0xFFFFFFFF)
	for i := range n {
		if udata[i]&0x80000000 != 0 {
			udata[i] ^= 0xFFFFFFFF // negative: flip all bits
		} else {
			udata[i] ^= 0x80000000 // positive: flip sign bit
		}
	}

	// Radix sort (unsigned, so no signed handling in final pass)
	temp := make([]uint32, n)
	radixPassU32(udata, temp, 0)
	radixPassU32(temp, udata, 8)
	radixPassU32(udata, temp, 16)
	radixPassU32(temp, udata, 24)

	// Transform back to floats
	for i := range n {
		if udata[i]&0x80000000 != 0 {
			udata[i] ^= 0x80000000 // was positive: flip sign bit back
		} else {
			udata[i] ^= 0xFFFFFFFF // was negative: flip all bits back
		}
	}
}

// RadixSortFloat64 sorts float64 data using radix sort.
func RadixSortFloat64(data []float64) {
	n := len(data)
	if n <= 1 {
		return
	}

	if n < radixSortThreshold {
		slices.Sort(data)
		return
	}

	// Reinterpret as uint64 for bit manipulation
	udata := unsafe.Slice((*uint64)(unsafe.Pointer(&data[0])), n)

	// Transform floats to sortable uint64
	for i := range n {
		if udata[i]&0x8000000000000000 != 0 {
			udata[i] ^= 0xFFFFFFFFFFFFFFFF // negative: flip all bits
		} else {
			udata[i] ^= 0x8000000000000000 // positive: flip sign bit
		}
	}

	// Radix sort using 16-bit passes (4 passes for uint64)
	temp := make([]uint64, n)
	radixPass16U64(udata, temp, 0)
	radixPass16U64(temp, udata, 16)
	radixPass16U64(udata, temp, 32)
	radixPass16U64(temp, udata, 48)

	// Transform back to floats
	for i := range n {
		if udata[i]&0x8000000000000000 != 0 {
			udata[i] ^= 0x8000000000000000 // was positive: flip sign bit back
		} else {
			udata[i] ^= 0xFFFFFFFFFFFFFFFF // was negative: flip all bits back
		}
	}
}

// radixPassU32 performs one pass of unsigned 8-bit radix sort for uint32.
func radixPassU32(src, dst []uint32, shift int) {
	n := len(src)
	var count [256]int

	for i := range n {
		digit := (src[i] >> shift) & 0xFF
		count[digit]++
	}

	offset := 0
	for b := range 256 {
		c := count[b]
		count[b] = offset
		offset += c
	}

	for i := range n {
		digit := (src[i] >> shift) & 0xFF
		dst[count[digit]] = src[i]
		count[digit]++
	}
}

// radixPass16U64 performs one pass of unsigned 16-bit radix sort for uint64.
func radixPass16U64(src, dst []uint64, shift int) {
	n := len(src)
	var count [65536]int

	for i := range n {
		digit := (src[i] >> shift) & 0xFFFF
		count[digit]++
	}

	offset := 0
	for b := range 65536 {
		c := count[b]
		count[b] = offset
		offset += c
	}

	for i := range n {
		digit := (src[i] >> shift) & 0xFFFF
		dst[count[digit]] = src[i]
		count[digit]++
	}
}
