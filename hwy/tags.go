package hwy

import "unsafe"

// Tag represents a vector size tag that determines how many lanes
// are used in SIMD operations.
type Tag interface {
	// Width returns the width in bytes (16 for 128-bit, 32 for 256-bit, etc.)
	Width() int

	// Name returns a human-readable name for this tag ("sse2", "avx2", etc.)
	Name() string
}

// ScalableTag adapts to the widest SIMD available at runtime.
// This is the recommended tag for most use cases as it provides
// optimal performance across different CPU architectures.
//
// Usage:
//
//	tag := hwy.ScalableTag[float32]{}
//	maxLanes := tag.MaxLanes()
type ScalableTag[T Lanes] struct{}

// Width returns the current runtime SIMD width in bytes.
func (ScalableTag[T]) Width() int {
	return currentWidth
}

// Name returns the current runtime SIMD target name.
func (ScalableTag[T]) Name() string {
	return currentName
}

// MaxLanes returns the maximum number of lanes for type T
// with the current SIMD width.
func (t ScalableTag[T]) MaxLanes() int {
	return MaxLanes[T]()
}

// FixedTag128 forces 128-bit SIMD operations (SSE, NEON).
// Use this when you need consistent behavior across platforms
// or when you know 128-bit is optimal for your use case.
type FixedTag128[T Lanes] struct{}

// Width returns 16 bytes (128 bits).
func (FixedTag128[T]) Width() int {
	return 16
}

// Name returns "128bit".
func (FixedTag128[T]) Name() string {
	return "128bit"
}

// MaxLanes returns the number of T values that fit in 128 bits.
func (t FixedTag128[T]) MaxLanes() int {
	var dummy T
	return 16 / int(unsafe.Sizeof(dummy))
}

// FixedTag256 forces 256-bit SIMD operations (AVX2).
// Use this when you need AVX2-specific behavior or when 256-bit
// is optimal for your algorithm.
type FixedTag256[T Lanes] struct{}

// Width returns 32 bytes (256 bits).
func (FixedTag256[T]) Width() int {
	return 32
}

// Name returns "256bit".
func (FixedTag256[T]) Name() string {
	return "256bit"
}

// MaxLanes returns the number of T values that fit in 256 bits.
func (t FixedTag256[T]) MaxLanes() int {
	var dummy T
	return 32 / int(unsafe.Sizeof(dummy))
}

// FixedTag512 forces 512-bit SIMD operations (AVX-512, SVE).
// Use this when you need AVX-512-specific behavior or when 512-bit
// is optimal for your algorithm.
type FixedTag512[T Lanes] struct{}

// Width returns 64 bytes (512 bits).
func (FixedTag512[T]) Width() int {
	return 64
}

// Name returns "512bit".
func (FixedTag512[T]) Name() string {
	return "512bit"
}

// MaxLanes returns the number of T values that fit in 512 bits.
func (t FixedTag512[T]) MaxLanes() int {
	var dummy T
	return 64 / int(unsafe.Sizeof(dummy))
}
