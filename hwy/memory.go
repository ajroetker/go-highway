package hwy

import "unsafe"

// This file provides additional memory operations for Highway.
// These are pure Go (scalar) implementations that work with any type.

// BlendedStore stores elements from v to dst only where mask is true.
// Unlike some SIMD implementations of masked stores, this explicitly
// preserves existing values in dst where mask is false.
//
// This is useful when you want conditional updates without affecting
// the non-selected lanes in the destination.
func BlendedStore[T Lanes](v Vec[T], mask Mask[T], dst []T) {
	n := min(len(dst), min(len(mask.bits), len(v.data)))
	for i := range n {
		if mask.bits[i] {
			dst[i] = v.data[i]
		}
		// else: dst[i] unchanged (the "blend" part)
	}
}

// Undefined returns a vector with undefined (implementation-defined) values.
// In Go, this returns a zero-initialized vector for safety, but callers
// should not rely on any specific value.
//
// Use this when initial values don't matter, such as:
// - Output of operations that will overwrite all lanes
// - First iteration of a reduction where initial value is unused
// - Temporary storage that will be fully written before reading
func Undefined[T Lanes]() Vec[T] {
	n := MaxLanes[T]()
	return Vec[T]{data: make([]T, n)}
}

// LoadDup128 loads a 128-bit (16 byte) block from src and duplicates it
// to fill the entire vector.
//
// For different vector widths:
//   - 128-bit (NEON, SSE): loads directly, no duplication needed
//   - 256-bit (AVX2): duplicates once: [A,B,C,D] -> [A,B,C,D,A,B,C,D]
//   - 512-bit (AVX-512): duplicates 4x to fill all 64 bytes
//
// This is useful for broadcasting a 128-bit constant or value across
// wider vector registers.
func LoadDup128[T Lanes](src []T) Vec[T] {
	var dummy T
	elemSize := int(unsafe.Sizeof(dummy))
	if elemSize == 0 {
		return Vec[T]{}
	}

	blockLanes := 16 / elemSize // lanes per 128-bit block
	totalLanes := MaxLanes[T]()

	result := make([]T, totalLanes)

	// Determine how many source lanes we can load (up to one 128-bit block)
	srcLanes := min(len(src), blockLanes)

	// Duplicate the 128-bit block across the entire vector
	for block := 0; block < totalLanes; block += blockLanes {
		for i := 0; i < blockLanes && block+i < totalLanes; i++ {
			if i < srcLanes {
				result[block+i] = src[i]
			}
			// else: leave as zero
		}
	}

	return Vec[T]{data: result}
}

// LoadInterleaved2 loads interleaved pairs and deinterleaves into two vectors.
// This converts Array-of-Structures (AoS) format to Structure-of-Arrays (SoA).
//
// Input memory layout (interleaved pairs):
//
//	[a0, b0, a1, b1, a2, b2, a3, b3, ...]
//
// Output vectors:
//
//	vec_a = [a0, a1, a2, a3, ...]
//	vec_b = [b0, b1, b2, b3, ...]
//
// This is useful for processing 2D coordinates, complex numbers,
// or any paired data stored in interleaved format.
func LoadInterleaved2[T Lanes](src []T) (Vec[T], Vec[T]) {
	n := MaxLanes[T]()
	a := make([]T, n)
	b := make([]T, n)

	srcIdx := 0
	for i := 0; i < n && srcIdx+1 < len(src); i++ {
		a[i] = src[srcIdx]
		b[i] = src[srcIdx+1]
		srcIdx += 2
	}

	return Vec[T]{data: a}, Vec[T]{data: b}
}

// LoadInterleaved3 loads interleaved triples and deinterleaves into three vectors.
// This converts Array-of-Structures (AoS) format to Structure-of-Arrays (SoA).
//
// Input memory layout (interleaved triples):
//
//	[a0, b0, c0, a1, b1, c1, a2, b2, c2, ...]
//
// Output vectors:
//
//	vec_a = [a0, a1, a2, ...]
//	vec_b = [b0, b1, b2, ...]
//	vec_c = [c0, c1, c2, ...]
//
// This is useful for processing RGB colors, 3D coordinates (XYZ),
// or any triple data stored in interleaved format.
func LoadInterleaved3[T Lanes](src []T) (Vec[T], Vec[T], Vec[T]) {
	n := MaxLanes[T]()
	a := make([]T, n)
	b := make([]T, n)
	c := make([]T, n)

	srcIdx := 0
	for i := 0; i < n && srcIdx+2 < len(src); i++ {
		a[i] = src[srcIdx]
		b[i] = src[srcIdx+1]
		c[i] = src[srcIdx+2]
		srcIdx += 3
	}

	return Vec[T]{data: a}, Vec[T]{data: b}, Vec[T]{data: c}
}

// LoadInterleaved4 loads interleaved quads and deinterleaves into four vectors.
// This converts Array-of-Structures (AoS) format to Structure-of-Arrays (SoA).
//
// Input memory layout (interleaved quads):
//
//	[a0, b0, c0, d0, a1, b1, c1, d1, ...]
//
// Output vectors:
//
//	vec_a = [a0, a1, ...]
//	vec_b = [b0, b1, ...]
//	vec_c = [c0, c1, ...]
//	vec_d = [d0, d1, ...]
//
// This is useful for processing RGBA colors, quaternions,
// or any quad data stored in interleaved format.
func LoadInterleaved4[T Lanes](src []T) (Vec[T], Vec[T], Vec[T], Vec[T]) {
	n := MaxLanes[T]()
	a := make([]T, n)
	b := make([]T, n)
	c := make([]T, n)
	d := make([]T, n)

	srcIdx := 0
	for i := 0; i < n && srcIdx+3 < len(src); i++ {
		a[i] = src[srcIdx]
		b[i] = src[srcIdx+1]
		c[i] = src[srcIdx+2]
		d[i] = src[srcIdx+3]
		srcIdx += 4
	}

	return Vec[T]{data: a}, Vec[T]{data: b}, Vec[T]{data: c}, Vec[T]{data: d}
}

// StoreInterleaved2 stores two vectors interleaved to dst.
// This converts Structure-of-Arrays (SoA) format to Array-of-Structures (AoS).
//
// Input vectors:
//
//	vec_a = [a0, a1, a2, a3, ...]
//	vec_b = [b0, b1, b2, b3, ...]
//
// Output memory layout (interleaved pairs):
//
//	[a0, b0, a1, b1, a2, b2, a3, b3, ...]
//
// This is the inverse of LoadInterleaved2.
func StoreInterleaved2[T Lanes](a, b Vec[T], dst []T) {
	n := min(len(b.data), len(a.data))

	dstIdx := 0
	for i := 0; i < n && dstIdx+1 < len(dst); i++ {
		dst[dstIdx] = a.data[i]
		dst[dstIdx+1] = b.data[i]
		dstIdx += 2
	}
}

// StoreInterleaved3 stores three vectors interleaved to dst.
// This converts Structure-of-Arrays (SoA) format to Array-of-Structures (AoS).
//
// Input vectors:
//
//	vec_a = [a0, a1, a2, ...]
//	vec_b = [b0, b1, b2, ...]
//	vec_c = [c0, c1, c2, ...]
//
// Output memory layout (interleaved triples):
//
//	[a0, b0, c0, a1, b1, c1, a2, b2, c2, ...]
//
// This is the inverse of LoadInterleaved3.
func StoreInterleaved3[T Lanes](a, b, c Vec[T], dst []T) {
	n := min(len(c.data), min(len(b.data), len(a.data)))

	dstIdx := 0
	for i := 0; i < n && dstIdx+2 < len(dst); i++ {
		dst[dstIdx] = a.data[i]
		dst[dstIdx+1] = b.data[i]
		dst[dstIdx+2] = c.data[i]
		dstIdx += 3
	}
}

// StoreInterleaved4 stores four vectors interleaved to dst.
// This converts Structure-of-Arrays (SoA) format to Array-of-Structures (AoS).
//
// Input vectors:
//
//	vec_a = [a0, a1, ...]
//	vec_b = [b0, b1, ...]
//	vec_c = [c0, c1, ...]
//	vec_d = [d0, d1, ...]
//
// Output memory layout (interleaved quads):
//
//	[a0, b0, c0, d0, a1, b1, c1, d1, ...]
//
// This is the inverse of LoadInterleaved4.
func StoreInterleaved4[T Lanes](a, b, c, d Vec[T], dst []T) {
	n := min(len(d.data), min(len(c.data), min(len(b.data), len(a.data))))

	dstIdx := 0
	for i := 0; i < n && dstIdx+3 < len(dst); i++ {
		dst[dstIdx] = a.data[i]
		dst[dstIdx+1] = b.data[i]
		dst[dstIdx+2] = c.data[i]
		dst[dstIdx+3] = d.data[i]
		dstIdx += 4
	}
}
