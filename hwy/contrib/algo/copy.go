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

// CopyIf conditionally copies elements from src to dst based on a predicate.
// Elements where pred(element) is true are packed together in dst (stream compaction).
// Returns the number of elements copied (limited by dst capacity).
//
// Example: Copy only positive values
//
//	copied := CopyIf(src, dst, func(v hwy.Vec[float32]) hwy.Mask[float32] {
//	    return hwy.GreaterThan(v, hwy.Zero[float32]())
//	})
//
// Note: This implementation uses scalar predicate evaluation for compatibility.
// For maximum SIMD performance with specific predicates, use CopyIfP with
// built-in predicate types.
func CopyIf[T hwy.Lanes](src, dst []T, pred func(hwy.Vec[T]) hwy.Mask[T]) int {
	return CopyIfP(src, dst, FuncPredicate[T]{Fn: pred})
}

// CopyIfP conditionally copies elements from src to dst based on a predicate.
// Uses the Predicate interface for better performance with built-in predicates.
// Returns the number of elements copied (limited by dst capacity).
func CopyIfP[T hwy.Lanes, P Predicate[T]](src, dst []T, pred P) int {
	n := len(src)
	dstLen := len(dst)
	if n == 0 || dstLen == 0 {
		return 0
	}

	dstIdx := 0
	for i := 0; i < n && dstIdx < dstLen; i++ {
		if pred.Test(src[i]) {
			dst[dstIdx] = src[i]
			dstIdx++
		}
	}
	return dstIdx
}
