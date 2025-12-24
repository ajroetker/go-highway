// Package hwy provides portable SIMD operations with runtime CPU dispatch.
//
// It follows the Highway C++ library's design philosophy: write once,
// run optimally everywhere. Operations automatically use the best available
// SIMD instructions (AVX2, AVX-512, NEON, SVE) or fall back to scalar code.
//
// Basic usage:
//
//	import "github.com/ajroetker/go-highway/hwy"
//
//	// Load data into vectors
//	a := hwy.Load(data1)
//	b := hwy.Load(data2)
//
//	// Perform SIMD operations
//	result := hwy.Add(a, b)
//
//	// Store results
//	hwy.Store(result, output)
package hwy

// Floats is a constraint for floating-point types.
type Floats interface {
	~float32 | ~float64
}

// SignedInts is a constraint for signed integer types.
type SignedInts interface {
	~int8 | ~int16 | ~int32 | ~int64
}

// UnsignedInts is a constraint for unsigned integer types.
type UnsignedInts interface {
	~uint8 | ~uint16 | ~uint32 | ~uint64
}

// Integers is a constraint for all integer types.
type Integers interface {
	SignedInts | UnsignedInts
}

// Lanes is a constraint for all types that can be stored in SIMD lanes.
type Lanes interface {
	Floats | Integers
}

// Vec is a portable vector handle that wraps SIMD operations.
// In base (scalar) mode, it wraps a slice. In SIMD modes, it may wrap
// architecture-specific vector types.
//
// Vec instances should not be created directly; use Load, Set, or Zero instead.
type Vec[T Lanes] struct {
	// data holds the vector elements in base mode.
	// In SIMD modes, this may be empty and the actual data stored
	// in architecture-specific fields.
	data []T
}

// NumLanes returns the number of lanes (elements) in this vector.
func (v Vec[T]) NumLanes() int {
	return len(v.data)
}

// Data returns the underlying slice representation of the vector.
// This is primarily for testing and should not be used in performance-critical code.
func (v Vec[T]) Data() []T {
	return v.data
}

// Mask represents the result of a comparison operation.
// It can be used with IfThenElse, MaskLoad, and MaskStore to perform
// conditional operations.
//
// Mask instances should not be created directly; use comparison operations
// like Equal, LessThan, or GreaterThan instead.
type Mask[T Lanes] struct {
	// bits stores which lanes are active (true).
	// bit i is set if lane i is active.
	bits []bool
}

// NumLanes returns the number of lanes in this mask.
func (m Mask[T]) NumLanes() int {
	return len(m.bits)
}

// AllTrue returns true if all lanes in the mask are active.
func (m Mask[T]) AllTrue() bool {
	for _, bit := range m.bits {
		if !bit {
			return false
		}
	}
	return true
}

// AnyTrue returns true if at least one lane in the mask is active.
func (m Mask[T]) AnyTrue() bool {
	for _, bit := range m.bits {
		if bit {
			return true
		}
	}
	return false
}

// CountTrue returns the number of active lanes in the mask.
func (m Mask[T]) CountTrue() int {
	count := 0
	for _, bit := range m.bits {
		if bit {
			count++
		}
	}
	return count
}

// GetBit returns whether lane i is active.
func (m Mask[T]) GetBit(i int) bool {
	if i < 0 || i >= len(m.bits) {
		return false
	}
	return m.bits[i]
}
