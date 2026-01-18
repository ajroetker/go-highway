package hwy

// This file provides pure Go (scalar) implementations of gather and scatter operations.
// When SIMD implementations are available (gather_avx2.go, gather_avx512.go),
// they can be used for higher performance on supported hardware.

// GatherIndex loads elements from non-contiguous memory locations specified by indices.
// For each lane i in the index vector, it loads src[indices[i]].
// If an index is out of bounds (negative or >= len(src)), the result for that lane is zero.
func GatherIndex[T Lanes, I ~int32 | ~int64](src []T, indices Vec[I]) Vec[T] {
	n := len(indices.data)
	result := make([]T, n)
	for i := range n {
		idx := int(indices.data[i])
		if idx >= 0 && idx < len(src) {
			result[i] = src[idx]
		}
		// else: leave as zero value
	}
	return Vec[T]{data: result}
}

// GatherIndexMasked loads elements from non-contiguous memory locations specified by indices,
// but only for lanes where the mask is true.
// If an index is out of bounds or the mask is false, the result for that lane is zero.
func GatherIndexMasked[T Lanes, I ~int32 | ~int64](src []T, indices Vec[I], mask Mask[T]) Vec[T] {
	n := min(len(mask.bits), len(indices.data))
	result := make([]T, len(indices.data))
	for i := range n {
		if mask.bits[i] {
			idx := int(indices.data[i])
			if idx >= 0 && idx < len(src) {
				result[i] = src[idx]
			}
		}
		// else: leave as zero value
	}
	return Vec[T]{data: result}
}

// ScatterIndex stores elements to non-contiguous memory locations specified by indices.
// For each lane i in the vectors, it stores v[i] to dst[indices[i]].
// If an index is out of bounds (negative or >= len(dst)), that store is skipped.
func ScatterIndex[T Lanes, I ~int32 | ~int64](v Vec[T], dst []T, indices Vec[I]) {
	n := min(len(indices.data), len(v.data))
	for i := range n {
		idx := int(indices.data[i])
		if idx >= 0 && idx < len(dst) {
			dst[idx] = v.data[i]
		}
	}
}

// ScatterIndexMasked stores elements to non-contiguous memory locations specified by indices,
// but only for lanes where the mask is true.
// If an index is out of bounds or the mask is false, that store is skipped.
func ScatterIndexMasked[T Lanes, I ~int32 | ~int64](v Vec[T], dst []T, indices Vec[I], mask Mask[T]) {
	n := min(len(mask.bits), min(len(indices.data), len(v.data)))
	for i := range n {
		if mask.bits[i] {
			idx := int(indices.data[i])
			if idx >= 0 && idx < len(dst) {
				dst[idx] = v.data[i]
			}
		}
	}
}

// GatherIndexOffset loads elements using base + index*scale addressing.
// This is useful for accessing elements with a fixed stride.
// For each lane i, it loads src[base + indices[i]*scale].
func GatherIndexOffset[T Lanes, I ~int32 | ~int64](src []T, base int, indices Vec[I], scale int) Vec[T] {
	n := len(indices.data)
	result := make([]T, n)
	for i := range n {
		idx := base + int(indices.data[i])*scale
		if idx >= 0 && idx < len(src) {
			result[i] = src[idx]
		}
		// else: leave as zero value
	}
	return Vec[T]{data: result}
}

// IndicesFromFunc creates an index vector by calling a function for each lane.
// This is useful for creating custom gather patterns.
func IndicesFromFunc[I ~int32 | ~int64](numLanes int, f func(lane int) I) Vec[I] {
	result := make([]I, numLanes)
	for i := range numLanes {
		result[i] = f(i)
	}
	return Vec[I]{data: result}
}

// IndicesIota creates an index vector with values [0, 1, 2, 3, ...].
func IndicesIota[I ~int32 | ~int64](numLanes int) Vec[I] {
	result := make([]I, numLanes)
	for i := range numLanes {
		result[i] = I(i)
	}
	return Vec[I]{data: result}
}

// IndicesStride creates an index vector with values [start, start+stride, start+2*stride, ...].
func IndicesStride[I ~int32 | ~int64](numLanes int, start, stride I) Vec[I] {
	result := make([]I, numLanes)
	for i := range numLanes {
		result[i] = start + I(i)*stride
	}
	return Vec[I]{data: result}
}
