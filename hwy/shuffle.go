package hwy

// This file provides shuffle and permutation operations for vectors.
// These are pure Go (scalar) implementations that work with any type.
// When SIMD implementations are available, they will be used via dispatch.

// Reverse reverses the order of lanes in the vector.
func Reverse[T Lanes](v Vec[T]) Vec[T] {
	n := len(v.data)
	result := make([]T, n)
	for i := range n {
		result[i] = v.data[n-1-i]
	}
	return Vec[T]{data: result}
}

// Broadcast broadcasts a single lane to all lanes in the vector.
func Broadcast[T Lanes](v Vec[T], lane int) Vec[T] {
	n := len(v.data)
	if lane < 0 || lane >= n {
		// Return zero vector if lane is out of bounds
		return Zero[T]()
	}
	result := make([]T, n)
	value := v.data[lane]
	for i := range result {
		result[i] = value
	}
	return Vec[T]{data: result}
}

// Reverse2 reverses pairs of lanes.
// [0,1,2,3,4,5,6,7] -> [1,0,3,2,5,4,7,6]
func Reverse2[T Lanes](v Vec[T]) Vec[T] {
	n := len(v.data)
	result := make([]T, n)
	for i := 0; i < n-1; i += 2 {
		result[i] = v.data[i+1]
		result[i+1] = v.data[i]
	}
	// Handle odd lane count (copy last element)
	if n%2 == 1 {
		result[n-1] = v.data[n-1]
	}
	return Vec[T]{data: result}
}

// Reverse4 reverses groups of 4 lanes.
// [0,1,2,3,4,5,6,7] -> [3,2,1,0,7,6,5,4]
func Reverse4[T Lanes](v Vec[T]) Vec[T] {
	n := len(v.data)
	result := make([]T, n)
	for i := 0; i < n; i += 4 {
		remaining := n - i
		if remaining >= 4 {
			result[i] = v.data[i+3]
			result[i+1] = v.data[i+2]
			result[i+2] = v.data[i+1]
			result[i+3] = v.data[i]
		} else {
			// Handle partial group at end
			for j := range remaining {
				result[i+j] = v.data[i+remaining-1-j]
			}
		}
	}
	return Vec[T]{data: result}
}

// Reverse8 reverses groups of 8 lanes.
// [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] -> [7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8]
func Reverse8[T Lanes](v Vec[T]) Vec[T] {
	n := len(v.data)
	result := make([]T, n)
	for i := 0; i < n; i += 8 {
		remaining := n - i
		if remaining >= 8 {
			for j := range 8 {
				result[i+j] = v.data[i+7-j]
			}
		} else {
			// Handle partial group at end
			for j := range remaining {
				result[i+j] = v.data[i+remaining-1-j]
			}
		}
	}
	return Vec[T]{data: result}
}

// GetLane extracts a single lane value from the vector.
// Returns zero value if index is out of bounds.
func GetLane[T Lanes](v Vec[T], idx int) T {
	if idx < 0 || idx >= len(v.data) {
		var zero T
		return zero
	}
	return v.data[idx]
}

// InsertLane returns a new vector with the value inserted at the given lane.
// Returns original vector if index is out of bounds.
func InsertLane[T Lanes](v Vec[T], idx int, val T) Vec[T] {
	n := len(v.data)
	if idx < 0 || idx >= n {
		return v
	}
	result := make([]T, n)
	copy(result, v.data)
	result[idx] = val
	return Vec[T]{data: result}
}

// InterleaveLower interleaves the lower halves of two vectors.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a0,b0,a1,b1]
func InterleaveLower[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	half := n / 2
	result := make([]T, n)
	for i := range half {
		result[2*i] = a.data[i]
		result[2*i+1] = b.data[i]
	}
	return Vec[T]{data: result}
}

// InterleaveUpper interleaves the upper halves of two vectors.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a2,b2,a3,b3]
func InterleaveUpper[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	half := n / 2
	result := make([]T, n)
	for i := range half {
		result[2*i] = a.data[half+i]
		result[2*i+1] = b.data[half+i]
	}
	return Vec[T]{data: result}
}

// ConcatLowerLower concatenates the lower halves of two vectors.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a0,a1,b0,b1]
func ConcatLowerLower[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	half := n / 2
	result := make([]T, n)
	copy(result[:half], a.data[:half])
	copy(result[half:], b.data[:half])
	return Vec[T]{data: result}
}

// ConcatUpperUpper concatenates the upper halves of two vectors.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a2,a3,b2,b3]
func ConcatUpperUpper[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	half := n / 2
	result := make([]T, n)
	copy(result[:half], a.data[half:])
	copy(result[half:], b.data[half:])
	return Vec[T]{data: result}
}

// ConcatLowerUpper concatenates lower half of a with upper half of b.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a0,a1,b2,b3]
func ConcatLowerUpper[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	half := n / 2
	result := make([]T, n)
	copy(result[:half], a.data[:half])
	copy(result[half:], b.data[half:])
	return Vec[T]{data: result}
}

// ConcatUpperLower concatenates upper half of a with lower half of b.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a2,a3,b0,b1]
func ConcatUpperLower[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	half := n / 2
	result := make([]T, n)
	copy(result[:half], a.data[half:])
	copy(result[half:], b.data[:half])
	return Vec[T]{data: result}
}

// OddEven combines odd lanes from a with even lanes from b.
// Returns a vector where even indices have b's values and odd indices have a's values.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [b0,a1,b2,a3]
func OddEven[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		if i%2 == 0 {
			result[i] = b.data[i]
		} else {
			result[i] = a.data[i]
		}
	}
	return Vec[T]{data: result}
}

// DupEven duplicates even lanes.
// [a0,a1,a2,a3] -> [a0,a0,a2,a2]
func DupEven[T Lanes](v Vec[T]) Vec[T] {
	n := len(v.data)
	result := make([]T, n)
	for i := 0; i < n; i += 2 {
		result[i] = v.data[i]
		if i+1 < n {
			result[i+1] = v.data[i]
		}
	}
	return Vec[T]{data: result}
}

// DupOdd duplicates odd lanes.
// [a0,a1,a2,a3] -> [a1,a1,a3,a3]
func DupOdd[T Lanes](v Vec[T]) Vec[T] {
	n := len(v.data)
	result := make([]T, n)
	for i := 0; i < n; i += 2 {
		if i+1 < n {
			result[i] = v.data[i+1]
			result[i+1] = v.data[i+1]
		} else {
			result[i] = v.data[i]
		}
	}
	return Vec[T]{data: result}
}

// SwapAdjacentBlocks swaps adjacent 128-bit blocks.
// For AVX2 (256-bit), this swaps the two 128-bit halves.
// For AVX-512 (512-bit), this swaps pairs of 128-bit blocks.
func SwapAdjacentBlocks[T Lanes](v Vec[T]) Vec[T] {
	n := len(v.data)
	result := make([]T, n)

	// Calculate 128-bit block size in lanes
	var dummy T
	elemSize := sizeOf(dummy)
	if elemSize == 0 {
		return v
	}
	blockLanes := 16 / elemSize // 128 bits / element size

	// Swap adjacent blocks
	for i := 0; i < n; i += 2 * blockLanes {
		if i+2*blockLanes <= n {
			// Swap blocks
			copy(result[i:i+blockLanes], v.data[i+blockLanes:i+2*blockLanes])
			copy(result[i+blockLanes:i+2*blockLanes], v.data[i:i+blockLanes])
		} else {
			// Copy remaining lanes
			copy(result[i:], v.data[i:])
		}
	}
	return Vec[T]{data: result}
}

// TableLookupBytes performs a byte-level table lookup.
// Each lane in idx specifies which byte from tbl to select.
// This is a scalar fallback; SIMD versions use PSHUFB/TBL instructions.
func TableLookupBytes[T Lanes](tbl, idx Vec[T]) Vec[T] {
	// Convert to bytes, lookup, convert back
	n := min(len(idx.data), len(tbl.data))
	result := make([]T, n)

	// For scalar fallback, treat each lane as a lookup index
	for i := range n {
		idxVal := int(idx.data[i])
		if idxVal >= 0 && idxVal < n {
			result[i] = tbl.data[idxVal]
		}
		// else: leave as zero
	}
	return Vec[T]{data: result}
}

// TableLookupLanes performs a lane-level table lookup.
// Each lane in idx specifies which lane from tbl to select.
// Unlike TableLookupBytes which works at byte granularity,
// this operates on full lanes (elements).
func TableLookupLanes[T Lanes](tbl Vec[T], idx Vec[int32]) Vec[T] {
	n := min(len(idx.data), len(tbl.data))
	result := make([]T, n)

	for i := range n {
		idxVal := int(idx.data[i])
		if idxVal >= 0 && idxVal < len(tbl.data) {
			result[i] = tbl.data[idxVal]
		}
		// else: leave as zero
	}
	return Vec[T]{data: result}
}

// TableLookupLanesOr returns fallback[i] when idx[i] is out of bounds.
func TableLookupLanesOr[T Lanes](tbl Vec[T], idx Vec[int32], fallback Vec[T]) Vec[T] {
	n := min(len(fallback.data), min(len(idx.data), len(tbl.data)))
	result := make([]T, n)

	for i := range n {
		idxVal := int(idx.data[i])
		if idxVal >= 0 && idxVal < len(tbl.data) {
			result[i] = tbl.data[idxVal]
		} else {
			result[i] = fallback.data[i]
		}
	}
	return Vec[T]{data: result}
}

// ZipLower interleaves the lower halves of two vectors into one.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a0,b0,a1,b1]
// This is the same as InterleaveLower but named for consistency with C++ Highway.
func ZipLower[T Lanes](a, b Vec[T]) Vec[T] {
	return InterleaveLower(a, b)
}

// ZipUpper interleaves the upper halves of two vectors into one.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a2,b2,a3,b3]
// This is the same as InterleaveUpper but named for consistency with C++ Highway.
func ZipUpper[T Lanes](a, b Vec[T]) Vec[T] {
	return InterleaveUpper(a, b)
}

// Shuffle0123 shuffles 4 lanes according to the given indices.
// Each index specifies which lane from the source to place in that position.
// For example: Shuffle0123(v, 3, 2, 1, 0) reverses a 4-lane vector.
func Shuffle0123[T Lanes](v Vec[T], i0, i1, i2, i3 int) Vec[T] {
	n := len(v.data)
	if n < 4 {
		return v
	}
	result := make([]T, n)

	// Process in groups of 4
	for base := 0; base < n; base += 4 {
		if base+4 <= n {
			result[base+0] = v.data[base+i0]
			result[base+1] = v.data[base+i1]
			result[base+2] = v.data[base+i2]
			result[base+3] = v.data[base+i3]
		} else {
			// Copy remaining
			copy(result[base:], v.data[base:])
		}
	}
	return Vec[T]{data: result}
}

// Per4LaneBlockShuffle performs a complex shuffle within each 4-lane block.
// pattern bits encode the shuffle: each 2-bit field selects a lane (0-3).
// Bits [1:0] select for lane 0, bits [3:2] for lane 1, etc.
func Per4LaneBlockShuffle[T Lanes](v Vec[T], pattern uint8) Vec[T] {
	n := len(v.data)
	result := make([]T, n)

	// Decode pattern: 2 bits per lane
	idx0 := int(pattern & 0x03)
	idx1 := int((pattern >> 2) & 0x03)
	idx2 := int((pattern >> 4) & 0x03)
	idx3 := int((pattern >> 6) & 0x03)

	// Process in groups of 4
	for base := 0; base < n; base += 4 {
		if base+4 <= n {
			result[base+0] = v.data[base+idx0]
			result[base+1] = v.data[base+idx1]
			result[base+2] = v.data[base+idx2]
			result[base+3] = v.data[base+idx3]
		} else {
			// Copy remaining
			copy(result[base:], v.data[base:])
		}
	}
	return Vec[T]{data: result}
}

// sizeOf returns the size in bytes of a type.
func sizeOf[T Lanes](v T) int {
	switch any(v).(type) {
	case float32, int32, uint32:
		return 4
	case float64, int64, uint64:
		return 8
	case int16, uint16:
		return 2
	case int8, uint8:
		return 1
	default:
		return 0
	}
}

// SlideUpLanes shifts all lanes up (toward higher indices) by the given offset.
// Lower lanes are filled with zeros, upper lanes that slide out are discarded.
// [1,2,3,4,5,6,7,8] with offset=2 -> [0,0,1,2,3,4,5,6]
func SlideUpLanes[T Lanes](v Vec[T], offset int) Vec[T] {
	n := len(v.data)
	result := make([]T, n)

	// Handle edge cases
	if offset <= 0 {
		copy(result, v.data)
		return Vec[T]{data: result}
	}
	if offset >= n {
		// All zeros (result is already zero-initialized)
		return Vec[T]{data: result}
	}

	// Copy elements shifted up
	// result[offset:n] = v.data[0:n-offset]
	copy(result[offset:], v.data[:n-offset])
	// result[0:offset] remains zero

	return Vec[T]{data: result}
}

// SlideDownLanes shifts all lanes down (toward lower indices) by the given offset.
// Upper lanes are filled with zeros, lower lanes that slide out are discarded.
// [1,2,3,4,5,6,7,8] with offset=2 -> [3,4,5,6,7,8,0,0]
func SlideDownLanes[T Lanes](v Vec[T], offset int) Vec[T] {
	n := len(v.data)
	result := make([]T, n)

	// Handle edge cases
	if offset <= 0 {
		copy(result, v.data)
		return Vec[T]{data: result}
	}
	if offset >= n {
		// All zeros (result is already zero-initialized)
		return Vec[T]{data: result}
	}

	// Copy elements shifted down
	// result[0:n-offset] = v.data[offset:n]
	copy(result[:n-offset], v.data[offset:])
	// result[n-offset:n] remains zero

	return Vec[T]{data: result}
}

// Slide1Up shifts all lanes up by 1, filling the first lane with zero.
// [1,2,3,4,5,6,7,8] -> [0,1,2,3,4,5,6,7]
func Slide1Up[T Lanes](v Vec[T]) Vec[T] {
	return SlideUpLanes(v, 1)
}

// Slide1Down shifts all lanes down by 1, filling the last lane with zero.
// [1,2,3,4,5,6,7,8] -> [2,3,4,5,6,7,8,0]
func Slide1Down[T Lanes](v Vec[T]) Vec[T] {
	return SlideDownLanes(v, 1)
}
