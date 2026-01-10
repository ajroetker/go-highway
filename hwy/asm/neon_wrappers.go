//go:build !noasm && arm64

package asm

//go:generate go tool goat ../c/ops_neon_arm64.c -O3 -e="--target=arm64" -e="-march=armv8-a+simd+fp"

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

// Type conversions (Phase 5)

// PromoteF32ToF64 converts float32 to float64: result[i] = float64(input[i])
func PromoteF32ToF64(input []float32, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	promote_f32_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// DemoteF64ToF32 converts float64 to float32: result[i] = float32(input[i])
func DemoteF64ToF32(input []float64, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	demote_f64_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ConvertF32ToI32 converts float32 to int32 (truncates toward zero)
func ConvertF32ToI32(input []float32, result []int32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	convert_f32_i32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ConvertI32ToF32 converts int32 to float32
func ConvertI32ToF32(input []int32, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	convert_i32_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// RoundF32 rounds to nearest (ties to even): result[i] = round(input[i])
func RoundF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	round_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// TruncF32 truncates toward zero: result[i] = trunc(input[i])
func TruncF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	trunc_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// CeilF32 rounds up: result[i] = ceil(input[i])
func CeilF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	ceil_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// FloorF32 rounds down: result[i] = floor(input[i])
func FloorF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	floor_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Memory operations (Phase 4)

// GatherF32 gathers values: result[i] = base[indices[i]]
func GatherF32(base []float32, indices []int32, result []float32) {
	if len(indices) == 0 {
		return
	}
	n := int64(len(indices))
	gather_f32_neon(unsafe.Pointer(&base[0]), unsafe.Pointer(&indices[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// GatherF64 gathers values: result[i] = base[indices[i]]
func GatherF64(base []float64, indices []int32, result []float64) {
	if len(indices) == 0 {
		return
	}
	n := int64(len(indices))
	gather_f64_neon(unsafe.Pointer(&base[0]), unsafe.Pointer(&indices[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// GatherI32 gathers values: result[i] = base[indices[i]]
func GatherI32(base []int32, indices []int32, result []int32) {
	if len(indices) == 0 {
		return
	}
	n := int64(len(indices))
	gather_i32_neon(unsafe.Pointer(&base[0]), unsafe.Pointer(&indices[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ScatterF32 scatters values: base[indices[i]] = values[i]
func ScatterF32(values []float32, indices []int32, base []float32) {
	if len(indices) == 0 {
		return
	}
	n := int64(len(indices))
	scatter_f32_neon(unsafe.Pointer(&values[0]), unsafe.Pointer(&indices[0]), unsafe.Pointer(&base[0]), unsafe.Pointer(&n))
}

// ScatterF64 scatters values: base[indices[i]] = values[i]
func ScatterF64(values []float64, indices []int32, base []float64) {
	if len(indices) == 0 {
		return
	}
	n := int64(len(indices))
	scatter_f64_neon(unsafe.Pointer(&values[0]), unsafe.Pointer(&indices[0]), unsafe.Pointer(&base[0]), unsafe.Pointer(&n))
}

// ScatterI32 scatters values: base[indices[i]] = values[i]
func ScatterI32(values []int32, indices []int32, base []int32) {
	if len(indices) == 0 {
		return
	}
	n := int64(len(indices))
	scatter_i32_neon(unsafe.Pointer(&values[0]), unsafe.Pointer(&indices[0]), unsafe.Pointer(&base[0]), unsafe.Pointer(&n))
}

// MaskedLoadF32 loads with mask: result[i] = mask[i] ? input[i] : 0
func MaskedLoadF32(input []float32, mask []int32, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	masked_load_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&mask[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MaskedStoreF32 stores with mask: if mask[i] then output[i] = input[i]
func MaskedStoreF32(input []float32, mask []int32, output []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	masked_store_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&mask[0]), unsafe.Pointer(&output[0]), unsafe.Pointer(&n))
}
