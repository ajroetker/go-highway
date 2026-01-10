//go:build !noasm && arm64

package asm

import "unsafe"

// Float32 operations - exported wrappers

// AddF32 performs element-wise addition: result[i] = a[i] + b[i]
func AddF32(a, b, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	add_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SubF32 performs element-wise subtraction: result[i] = a[i] - b[i]
func SubF32(a, b, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	sub_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MulF32 performs element-wise multiplication: result[i] = a[i] * b[i]
func MulF32(a, b, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	mul_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// DivF32 performs element-wise division: result[i] = a[i] / b[i]
func DivF32(a, b, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	div_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// FmaF32 performs fused multiply-add: result[i] = a[i] * b[i] + c[i]
func FmaF32(a, b, c, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	fma_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MinF32 performs element-wise minimum: result[i] = min(a[i], b[i])
func MinF32(a, b, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	min_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MaxF32 performs element-wise maximum: result[i] = max(a[i], b[i])
func MaxF32(a, b, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	max_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ReduceSumF32 returns the sum of all elements
func ReduceSumF32(input []float32) float32 {
	if len(input) == 0 {
		return 0
	}
	n := int64(len(input))
	var result float32
	reduce_sum_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result), unsafe.Pointer(&n))
	return result
}

// ReduceMinF32 returns the minimum element
func ReduceMinF32(input []float32) float32 {
	if len(input) == 0 {
		return 0
	}
	n := int64(len(input))
	var result float32
	reduce_min_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result), unsafe.Pointer(&n))
	return result
}

// ReduceMaxF32 returns the maximum element
func ReduceMaxF32(input []float32) float32 {
	if len(input) == 0 {
		return 0
	}
	n := int64(len(input))
	var result float32
	reduce_max_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result), unsafe.Pointer(&n))
	return result
}

// SqrtF32 performs element-wise square root: result[i] = sqrt(a[i])
func SqrtF32(a, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	sqrt_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// AbsF32 performs element-wise absolute value: result[i] = abs(a[i])
func AbsF32(a, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	abs_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// NegF32 performs element-wise negation: result[i] = -a[i]
func NegF32(a, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	neg_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Float64 operations - exported wrappers

// AddF64 performs element-wise addition: result[i] = a[i] + b[i]
func AddF64(a, b, result []float64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	add_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MulF64 performs element-wise multiplication: result[i] = a[i] * b[i]
func MulF64(a, b, result []float64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	mul_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// FmaF64 performs fused multiply-add: result[i] = a[i] * b[i] + c[i]
func FmaF64(a, b, c, result []float64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	fma_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ReduceSumF64 returns the sum of all elements
func ReduceSumF64(input []float64) float64 {
	if len(input) == 0 {
		return 0
	}
	n := int64(len(input))
	var result float64
	reduce_sum_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result), unsafe.Pointer(&n))
	return result
}
