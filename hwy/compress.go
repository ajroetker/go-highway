package hwy

// This file provides compress and expand operations for vectors.
// Compress packs elements where the mask is true to the front.
// Expand unpacks elements into positions where the mask is true.
// These are essential for SIMD stream compaction and sparse operations.

// Compress packs elements where mask is true to the front.
// Returns compressed vector and count of valid elements.
// For example: v=[1,2,3,4], mask=[T,F,T,F] -> result=[1,3,0,0], count=2
func Compress[T Lanes](v Vec[T], mask Mask[T]) (Vec[T], int) {
	n := min(len(mask.bits), len(v.data))

	result := make([]T, len(v.data))
	count := 0
	for i := range n {
		if mask.bits[i] {
			result[count] = v.data[i]
			count++
		}
	}
	return Vec[T]{data: result}, count
}

// Expand unpacks elements into positions where mask is true.
// Elements from v fill true positions, false positions get zero.
// For example: v=[1,2,0,0], mask=[T,F,T,F] -> result=[1,0,2,0]
func Expand[T Lanes](v Vec[T], mask Mask[T]) Vec[T] {
	n := len(mask.bits)
	result := make([]T, n)

	srcIdx := 0
	for i := range n {
		if mask.bits[i] {
			if srcIdx < len(v.data) {
				result[i] = v.data[srcIdx]
				srcIdx++
			}
		}
		// else: leave as zero value
	}
	return Vec[T]{data: result}
}

// CompressStore compresses and stores directly to slice.
// Returns number of elements stored.
func CompressStore[T Lanes](v Vec[T], mask Mask[T], dst []T) int {
	n := min(len(mask.bits), len(v.data))

	count := 0
	for i := range n {
		if mask.bits[i] {
			if count < len(dst) {
				dst[count] = v.data[i]
			}
			count++
		}
	}
	return count
}

// CountTrue counts true lanes in mask.
// This is a function wrapper around Mask.CountTrue() for consistency.
func CountTrue[T Lanes](mask Mask[T]) int {
	return mask.CountTrue()
}

// AllTrue returns true if all lanes are true.
// This is a function wrapper around Mask.AllTrue() for consistency.
func AllTrue[T Lanes](mask Mask[T]) bool {
	return mask.AllTrue()
}

// AllFalse returns true if all lanes are false.
func AllFalse[T Lanes](mask Mask[T]) bool {
	for _, bit := range mask.bits {
		if bit {
			return false
		}
	}
	return true
}

// FindFirstTrue returns index of first true lane, or -1 if none.
func FindFirstTrue[T Lanes](mask Mask[T]) int {
	for i, bit := range mask.bits {
		if bit {
			return i
		}
	}
	return -1
}

// FindLastTrue returns index of last true lane, or -1 if none.
func FindLastTrue[T Lanes](mask Mask[T]) int {
	for i := len(mask.bits) - 1; i >= 0; i-- {
		if mask.bits[i] {
			return i
		}
	}
	return -1
}

// FirstN creates a mask with the first n lanes set to true.
// This is similar to TailMask but more explicit in naming.
func FirstN[T Lanes](n int) Mask[T] {
	maxLanes := MaxLanes[T]()
	if n < 0 {
		n = 0
	}
	if n > maxLanes {
		n = maxLanes
	}

	bits := make([]bool, maxLanes)
	for i := 0; i < n; i++ {
		bits[i] = true
	}
	return Mask[T]{bits: bits}
}

// LastN creates a mask with the last n lanes set to true.
func LastN[T Lanes](n int) Mask[T] {
	maxLanes := MaxLanes[T]()
	if n < 0 {
		n = 0
	}
	if n > maxLanes {
		n = maxLanes
	}

	bits := make([]bool, maxLanes)
	start := maxLanes - n
	for i := start; i < maxLanes; i++ {
		bits[i] = true
	}
	return Mask[T]{bits: bits}
}

// MaskFromBits creates a mask from a bitmask integer.
// Bit i of bits corresponds to lane i.
func MaskFromBits[T Lanes](bits uint64) Mask[T] {
	maxLanes := MaxLanes[T]()
	result := make([]bool, maxLanes)
	for i := 0; i < maxLanes && i < 64; i++ {
		result[i] = (bits & (1 << i)) != 0
	}
	return Mask[T]{bits: result}
}

// BitsFromMask converts mask to bitmask integer.
// Lane i corresponds to bit i of the result.
func BitsFromMask[T Lanes](mask Mask[T]) uint64 {
	var result uint64
	for i, bit := range mask.bits {
		if bit && i < 64 {
			result |= 1 << i
		}
	}
	return result
}

// CompressBlendedStore compresses elements and blends with existing destination.
// Elements where mask is true are compressed and stored; other elements unchanged.
// Returns number of elements stored.
func CompressBlendedStore[T Lanes](v Vec[T], mask Mask[T], dst []T) int {
	compressed, count := Compress(v, mask)
	for i := 0; i < count && i < len(dst); i++ {
		dst[i] = compressed.data[i]
	}
	return count
}

// MaskAnd performs bitwise AND on two masks.
func MaskAnd[T Lanes](a, b Mask[T]) Mask[T] {
	n := min(len(b.bits), len(a.bits))
	result := make([]bool, n)
	for i := range n {
		result[i] = a.bits[i] && b.bits[i]
	}
	return Mask[T]{bits: result}
}

// MaskOr performs bitwise OR on two masks.
func MaskOr[T Lanes](a, b Mask[T]) Mask[T] {
	n := min(len(b.bits), len(a.bits))
	result := make([]bool, n)
	for i := range n {
		result[i] = a.bits[i] || b.bits[i]
	}
	return Mask[T]{bits: result}
}

// MaskXor performs bitwise XOR on two masks.
func MaskXor[T Lanes](a, b Mask[T]) Mask[T] {
	n := min(len(b.bits), len(a.bits))
	result := make([]bool, n)
	for i := range n {
		result[i] = a.bits[i] != b.bits[i]
	}
	return Mask[T]{bits: result}
}

// MaskNot inverts all bits in a mask.
func MaskNot[T Lanes](mask Mask[T]) Mask[T] {
	result := make([]bool, len(mask.bits))
	for i, bit := range mask.bits {
		result[i] = !bit
	}
	return Mask[T]{bits: result}
}

// MaskAndNot performs (~a) & b on masks.
func MaskAndNot[T Lanes](a, b Mask[T]) Mask[T] {
	n := min(len(b.bits), len(a.bits))
	result := make([]bool, n)
	for i := range n {
		result[i] = !a.bits[i] && b.bits[i]
	}
	return Mask[T]{bits: result}
}
