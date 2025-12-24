package hwy

import "math"

// This file provides pure Go (scalar) implementations of all Highway operations.
// When SIMD implementations are available (ops_simd_*.go), they will replace these
// implementations via build tags. The scalar implementations serve as the fallback
// and are also used when HWY_NO_SIMD is set.

// Load creates a vector by loading data from a slice.
func Load[T Lanes](src []T) Vec[T] {
	n := MaxLanes[T]()
	if len(src) < n {
		n = len(src)
	}
	data := make([]T, n)
	copy(data, src[:n])
	return Vec[T]{data: data}
}

// Store writes a vector's data to a slice.
func Store[T Lanes](v Vec[T], dst []T) {
	n := len(v.data)
	if len(dst) < n {
		n = len(dst)
	}
	copy(dst[:n], v.data[:n])
}

// Set creates a vector with all lanes set to the same value.
func Set[T Lanes](value T) Vec[T] {
	n := MaxLanes[T]()
	data := make([]T, n)
	for i := range data {
		data[i] = value
	}
	return Vec[T]{data: data}
}

// Zero creates a vector with all lanes set to zero.
func Zero[T Lanes]() Vec[T] {
	n := MaxLanes[T]()
	data := make([]T, n)
	return Vec[T]{data: data}
}

// Add performs element-wise addition.
func Add[T Lanes](a, b Vec[T]) Vec[T] {
	n := len(a.data)
	if len(b.data) < n {
		n = len(b.data)
	}
	result := make([]T, n)
	for i := 0; i < n; i++ {
		result[i] = a.data[i] + b.data[i]
	}
	return Vec[T]{data: result}
}

// Sub performs element-wise subtraction.
func Sub[T Lanes](a, b Vec[T]) Vec[T] {
	n := len(a.data)
	if len(b.data) < n {
		n = len(b.data)
	}
	result := make([]T, n)
	for i := 0; i < n; i++ {
		result[i] = a.data[i] - b.data[i]
	}
	return Vec[T]{data: result}
}

// Mul performs element-wise multiplication.
func Mul[T Lanes](a, b Vec[T]) Vec[T] {
	n := len(a.data)
	if len(b.data) < n {
		n = len(b.data)
	}
	result := make([]T, n)
	for i := 0; i < n; i++ {
		result[i] = a.data[i] * b.data[i]
	}
	return Vec[T]{data: result}
}

// Div performs element-wise division.
func Div[T Floats](a, b Vec[T]) Vec[T] {
	n := len(a.data)
	if len(b.data) < n {
		n = len(b.data)
	}
	result := make([]T, n)
	for i := 0; i < n; i++ {
		result[i] = a.data[i] / b.data[i]
	}
	return Vec[T]{data: result}
}

// Neg negates all lanes.
func Neg[T Lanes](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = -v.data[i]
	}
	return Vec[T]{data: result}
}

// Abs computes absolute value.
func Abs[T Lanes](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		val := v.data[i]
		if val < 0 {
			result[i] = -val
		} else {
			result[i] = val
		}
	}
	return Vec[T]{data: result}
}

// Min returns element-wise minimum.
func Min[T Lanes](a, b Vec[T]) Vec[T] {
	n := len(a.data)
	if len(b.data) < n {
		n = len(b.data)
	}
	result := make([]T, n)
	for i := 0; i < n; i++ {
		if a.data[i] < b.data[i] {
			result[i] = a.data[i]
		} else {
			result[i] = b.data[i]
		}
	}
	return Vec[T]{data: result}
}

// Max returns element-wise maximum.
func Max[T Lanes](a, b Vec[T]) Vec[T] {
	n := len(a.data)
	if len(b.data) < n {
		n = len(b.data)
	}
	result := make([]T, n)
	for i := 0; i < n; i++ {
		if a.data[i] > b.data[i] {
			result[i] = a.data[i]
		} else {
			result[i] = b.data[i]
		}
	}
	return Vec[T]{data: result}
}

// Sqrt computes square root.
func Sqrt[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		// Use type assertion to handle float32 vs float64
		switch any(v.data[i]).(type) {
		case float32:
			result[i] = T(math.Sqrt(float64(v.data[i])))
		case float64:
			result[i] = T(math.Sqrt(float64(v.data[i])))
		}
	}
	return Vec[T]{data: result}
}

// FMA performs fused multiply-add.
func FMA[T Floats](a, b, c Vec[T]) Vec[T] {
	n := len(a.data)
	if len(b.data) < n {
		n = len(b.data)
	}
	if len(c.data) < n {
		n = len(c.data)
	}
	result := make([]T, n)
	for i := 0; i < n; i++ {
		// Use type assertion to handle float32 vs float64
		switch any(a.data[i]).(type) {
		case float32:
			result[i] = T(math.FMA(float64(a.data[i]), float64(b.data[i]), float64(c.data[i])))
		case float64:
			result[i] = T(math.FMA(float64(a.data[i]), float64(b.data[i]), float64(c.data[i])))
		}
	}
	return Vec[T]{data: result}
}

// ReduceSum sums all lanes.
func ReduceSum[T Lanes](v Vec[T]) T {
	var sum T
	for i := 0; i < len(v.data); i++ {
		sum += v.data[i]
	}
	return sum
}

// Equal performs element-wise equality comparison.
func Equal[T Lanes](a, b Vec[T]) Mask[T] {
	n := len(a.data)
	if len(b.data) < n {
		n = len(b.data)
	}
	bits := make([]bool, n)
	for i := 0; i < n; i++ {
		bits[i] = a.data[i] == b.data[i]
	}
	return Mask[T]{bits: bits}
}

// LessThan performs element-wise less-than comparison.
func LessThan[T Lanes](a, b Vec[T]) Mask[T] {
	n := len(a.data)
	if len(b.data) < n {
		n = len(b.data)
	}
	bits := make([]bool, n)
	for i := 0; i < n; i++ {
		bits[i] = a.data[i] < b.data[i]
	}
	return Mask[T]{bits: bits}
}

// GreaterThan performs element-wise greater-than comparison.
func GreaterThan[T Lanes](a, b Vec[T]) Mask[T] {
	n := len(a.data)
	if len(b.data) < n {
		n = len(b.data)
	}
	bits := make([]bool, n)
	for i := 0; i < n; i++ {
		bits[i] = a.data[i] > b.data[i]
	}
	return Mask[T]{bits: bits}
}

// IfThenElse performs conditional selection.
func IfThenElse[T Lanes](mask Mask[T], a, b Vec[T]) Vec[T] {
	n := len(mask.bits)
	if len(a.data) < n {
		n = len(a.data)
	}
	if len(b.data) < n {
		n = len(b.data)
	}
	result := make([]T, n)
	for i := 0; i < n; i++ {
		if mask.bits[i] {
			result[i] = a.data[i]
		} else {
			result[i] = b.data[i]
		}
	}
	return Vec[T]{data: result}
}

// MaskLoad loads data from a slice only for lanes where the mask is true.
func MaskLoad[T Lanes](mask Mask[T], src []T) Vec[T] {
	n := len(mask.bits)
	if len(src) < n {
		n = len(src)
	}
	result := make([]T, len(mask.bits))
	for i := 0; i < n; i++ {
		if mask.bits[i] {
			result[i] = src[i]
		}
		// else: leave as zero value
	}
	return Vec[T]{data: result}
}

// MaskStore stores vector data to a slice only for lanes where the mask is true.
func MaskStore[T Lanes](mask Mask[T], v Vec[T], dst []T) {
	n := len(mask.bits)
	if len(v.data) < n {
		n = len(v.data)
	}
	if len(dst) < n {
		n = len(dst)
	}
	for i := 0; i < n; i++ {
		if mask.bits[i] {
			dst[i] = v.data[i]
		}
	}
}
