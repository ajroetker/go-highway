//go:build !noasm && arm64

package asm

// -fno-builtin-memset prevents clang from optimizing NEON store loops into memset calls,
// which don't exist in Go assembly context and cause runtime failures.
//go:generate go tool goat ../c/ops_neon_arm64.c -O3 -e="--target=arm64" -e="-march=armv8-a+simd+fp" -e="-fno-builtin-memset"

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

// SubF64 performs element-wise subtraction: result[i] = a[i] - b[i]
func SubF64(a, b, result []float64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	sub_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// DivF64 performs element-wise division: result[i] = a[i] / b[i]
func DivF64(a, b, result []float64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	div_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MinF64 computes element-wise minimum: result[i] = min(a[i], b[i])
func MinF64(a, b, result []float64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	min_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MaxF64 computes element-wise maximum: result[i] = max(a[i], b[i])
func MaxF64(a, b, result []float64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	max_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SqrtF64 computes square root: result[i] = sqrt(a[i])
func SqrtF64(a, result []float64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	sqrt_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// AbsF64 computes absolute value: result[i] = |a[i]|
func AbsF64(a, result []float64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	abs_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// NegF64 computes negation: result[i] = -a[i]
func NegF64(a, result []float64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	neg_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ReduceMinF64 finds the minimum value across all elements
func ReduceMinF64(input []float64) float64 {
	if len(input) == 0 {
		return 0
	}
	n := int64(len(input))
	var result float64
	reduce_min_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result), unsafe.Pointer(&n))
	return result
}

// ReduceMaxF64 finds the maximum value across all elements
func ReduceMaxF64(input []float64) float64 {
	if len(input) == 0 {
		return 0
	}
	n := int64(len(input))
	var result float64
	reduce_max_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result), unsafe.Pointer(&n))
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

// GatherI64 gathers values: result[i] = base[indices[i]]
func GatherI64(base []int64, indices []int32, result []int64) {
	if len(indices) == 0 {
		return
	}
	n := int64(len(indices))
	gather_i64_neon(unsafe.Pointer(&base[0]), unsafe.Pointer(&indices[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
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

// ScatterI64 scatters values: base[indices[i]] = values[i]
func ScatterI64(values []int64, indices []int32, base []int64) {
	if len(indices) == 0 {
		return
	}
	n := int64(len(indices))
	scatter_i64_neon(unsafe.Pointer(&values[0]), unsafe.Pointer(&indices[0]), unsafe.Pointer(&base[0]), unsafe.Pointer(&n))
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

// MaskedLoadF64 loads with mask: result[i] = mask[i] ? input[i] : 0
func MaskedLoadF64(input []float64, mask []int64, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	masked_load_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&mask[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MaskedStoreF64 stores with mask: if mask[i] then output[i] = input[i]
func MaskedStoreF64(input []float64, mask []int64, output []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	masked_store_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&mask[0]), unsafe.Pointer(&output[0]), unsafe.Pointer(&n))
}

// MaskedLoadI32 loads with mask: result[i] = mask[i] ? input[i] : 0
func MaskedLoadI32(input []int32, mask []int32, result []int32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	masked_load_i32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&mask[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MaskedStoreI32 stores with mask: if mask[i] then output[i] = input[i]
func MaskedStoreI32(input []int32, mask []int32, output []int32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	masked_store_i32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&mask[0]), unsafe.Pointer(&output[0]), unsafe.Pointer(&n))
}

// MaskedLoadI64 loads with mask: result[i] = mask[i] ? input[i] : 0
func MaskedLoadI64(input []int64, mask []int64, result []int64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	masked_load_i64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&mask[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MaskedStoreI64 stores with mask: if mask[i] then output[i] = input[i]
func MaskedStoreI64(input []int64, mask []int64, output []int64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	masked_store_i64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&mask[0]), unsafe.Pointer(&output[0]), unsafe.Pointer(&n))
}

// Shuffle/Permutation operations (Phase 6)

// ReverseF32 reverses the order of elements: result[i] = input[n-1-i]
func ReverseF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	reverse_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ReverseF64 reverses the order of elements: result[i] = input[n-1-i]
func ReverseF64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	reverse_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Reverse2F32 swaps adjacent pairs: [0,1,2,3] -> [1,0,3,2]
func Reverse2F32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	reverse2_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Reverse4F32 reverses groups of 4: [0,1,2,3,4,5,6,7] -> [3,2,1,0,7,6,5,4]
func Reverse4F32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	reverse4_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// BroadcastF32 fills result with input[lane]
func BroadcastF32(input []float32, lane int, result []float32) {
	if len(input) == 0 || len(result) == 0 {
		return
	}
	l := int64(lane)
	n := int64(len(result))
	broadcast_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&l), unsafe.Pointer(&n))
}

// GetLaneF32 extracts a single lane value
func GetLaneF32(input []float32, lane int) float32 {
	if len(input) == 0 || lane < 0 || lane >= len(input) {
		return 0
	}
	var result float32
	l := int64(lane)
	getlane_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result), unsafe.Pointer(&l))
	return result
}

// InsertLaneF32 inserts value at specified lane
func InsertLaneF32(input []float32, lane int, value float32, result []float32) {
	if len(input) == 0 {
		return
	}
	l := int64(lane)
	n := int64(len(input))
	insertlane_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&value), unsafe.Pointer(&l), unsafe.Pointer(&n))
}

// InterleaveLowerF32 interleaves lower halves: [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a0,b0,a1,b1]
func InterleaveLowerF32(a, b, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	interleave_lo_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// InterleaveUpperF32 interleaves upper halves: [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a2,b2,a3,b3]
func InterleaveUpperF32(a, b, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	interleave_hi_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// TableLookupBytesU8 performs byte-level table lookup: result[i] = tbl[idx[i]]
func TableLookupBytesU8(tbl, idx, result []uint8) {
	if len(tbl) < 16 || len(idx) == 0 {
		return
	}
	n := int64(len(idx))
	tbl_u8_neon(unsafe.Pointer(&tbl[0]), unsafe.Pointer(&idx[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Comparison operations (Phase 7)

// EqF32 compares for equality: result[i] = (a[i] == b[i]) ? -1 : 0
func EqF32(a, b []float32, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	eq_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// EqI32 compares for equality: result[i] = (a[i] == b[i]) ? -1 : 0
func EqI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	eq_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// NeF32 compares for inequality: result[i] = (a[i] != b[i]) ? -1 : 0
func NeF32(a, b []float32, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	ne_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// NeI32 compares for inequality: result[i] = (a[i] != b[i]) ? -1 : 0
func NeI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	ne_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// LtF32 compares for less than: result[i] = (a[i] < b[i]) ? -1 : 0
func LtF32(a, b []float32, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	lt_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// LtI32 compares for less than: result[i] = (a[i] < b[i]) ? -1 : 0
func LtI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	lt_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// LeF32 compares for less or equal: result[i] = (a[i] <= b[i]) ? -1 : 0
func LeF32(a, b []float32, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	le_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// LeI32 compares for less or equal: result[i] = (a[i] <= b[i]) ? -1 : 0
func LeI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	le_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// GtF32 compares for greater than: result[i] = (a[i] > b[i]) ? -1 : 0
func GtF32(a, b []float32, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	gt_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// GtI32 compares for greater than: result[i] = (a[i] > b[i]) ? -1 : 0
func GtI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	gt_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// GeF32 compares for greater or equal: result[i] = (a[i] >= b[i]) ? -1 : 0
func GeF32(a, b []float32, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	ge_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// GeI32 compares for greater or equal: result[i] = (a[i] >= b[i]) ? -1 : 0
func GeI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	ge_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Float64 comparison operations

// EqF64 compares for equality: result[i] = (a[i] == b[i]) ? -1 : 0
func EqF64(a, b []float64, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	eq_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// GtF64 compares for greater than: result[i] = (a[i] > b[i]) ? -1 : 0
func GtF64(a, b []float64, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	gt_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// GeF64 compares for greater or equal: result[i] = (a[i] >= b[i]) ? -1 : 0
func GeF64(a, b []float64, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	ge_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// LtF64 compares for less than: result[i] = (a[i] < b[i]) ? -1 : 0
func LtF64(a, b []float64, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	lt_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// LeF64 compares for less or equal: result[i] = (a[i] <= b[i]) ? -1 : 0
func LeF64(a, b []float64, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	le_f64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Power of 2 operations (for exp/log implementations)

// Pow2F32 computes 2^k for each int32 k, result as float32
func Pow2F32(k []int32, result []float32) {
	if len(k) == 0 {
		return
	}
	n := int64(len(k))
	pow2_f32_neon(unsafe.Pointer(&k[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Pow2F64 computes 2^k for each int32 k, result as float64
func Pow2F64(k []int32, result []float64) {
	if len(k) == 0 {
		return
	}
	n := int64(len(k))
	pow2_f64_neon(unsafe.Pointer(&k[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Bitwise operations (Phase 8)

// AndI32 performs bitwise AND: result[i] = a[i] & b[i]
func AndI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	and_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// OrI32 performs bitwise OR: result[i] = a[i] | b[i]
func OrI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	or_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// XorI32 performs bitwise XOR: result[i] = a[i] ^ b[i]
func XorI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	xor_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// AndNotI32 performs bitwise AND-NOT: result[i] = a[i] & ~b[i]
func AndNotI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	andnot_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// NotI32 performs bitwise NOT: result[i] = ~a[i]
func NotI32(a, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	not_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ShiftLeftI32 performs left shift: result[i] = a[i] << shift
func ShiftLeftI32(a []int32, shift int, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	s := int64(shift)
	shl_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&s), unsafe.Pointer(&n))
}

// ShiftRightI32 performs arithmetic right shift: result[i] = a[i] >> shift
func ShiftRightI32(a []int32, shift int, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	s := int64(shift)
	shr_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&s), unsafe.Pointer(&n))
}

// Mask operations (Phase 9)

// IfThenElseF32 selects elements: result[i] = mask[i] ? yes[i] : no[i]
func IfThenElseF32(mask []int32, yes, no, result []float32) {
	if len(mask) == 0 {
		return
	}
	n := int64(len(mask))
	ifthenelse_f32_neon(unsafe.Pointer(&mask[0]), unsafe.Pointer(&yes[0]), unsafe.Pointer(&no[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// IfThenElseI32 selects elements: result[i] = mask[i] ? yes[i] : no[i]
func IfThenElseI32(mask, yes, no, result []int32) {
	if len(mask) == 0 {
		return
	}
	n := int64(len(mask))
	ifthenelse_i32_neon(unsafe.Pointer(&mask[0]), unsafe.Pointer(&yes[0]), unsafe.Pointer(&no[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// CountTrueI32 counts non-zero elements in mask
func CountTrueI32(mask []int32) int64 {
	if len(mask) == 0 {
		return 0
	}
	n := int64(len(mask))
	var result int64
	count_true_i32_neon(unsafe.Pointer(&mask[0]), unsafe.Pointer(&result), unsafe.Pointer(&n))
	return result
}

// AllTrueI32 returns true if all mask elements are non-zero
func AllTrueI32(mask []int32) bool {
	if len(mask) == 0 {
		return true
	}
	n := int64(len(mask))
	var result int64
	all_true_i32_neon(unsafe.Pointer(&mask[0]), unsafe.Pointer(&result), unsafe.Pointer(&n))
	return result != 0
}

// AllFalseI32 returns true if all mask elements are zero
func AllFalseI32(mask []int32) bool {
	if len(mask) == 0 {
		return true
	}
	n := int64(len(mask))
	var result int64
	all_false_i32_neon(unsafe.Pointer(&mask[0]), unsafe.Pointer(&result), unsafe.Pointer(&n))
	return result != 0
}

// FirstNI32 generates a mask with first count elements set to -1, rest 0
func FirstNI32(count int, result []int32) {
	if len(result) == 0 {
		return
	}
	n := int64(len(result))
	c := int64(count)
	firstn_i32_neon(unsafe.Pointer(&result[0]), unsafe.Pointer(&c), unsafe.Pointer(&n))
}

// CompressF32 packs elements where mask is non-zero
// Returns the number of elements written to result
func CompressF32(input []float32, mask []int32, result []float32) int64 {
	if len(input) == 0 {
		return 0
	}
	n := int64(len(input))
	var count int64
	compress_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&mask[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&count), unsafe.Pointer(&n))
	return count
}

// ExpandF32 unpacks elements to positions where mask is non-zero
func ExpandF32(input []float32, mask []int32, result []float32) {
	if len(result) == 0 {
		return
	}
	n := int64(len(result))
	var count int64
	expand_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&mask[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&count), unsafe.Pointer(&n))
}

// Transcendental math operations (Phase 10)

// ExpF32 computes exponential: result[i] = exp(input[i])
func ExpF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	exp_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// LogF32 computes natural logarithm: result[i] = log(input[i])
func LogF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	log_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SinF32 computes sine: result[i] = sin(input[i])
func SinF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	sin_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// CosF32 computes cosine: result[i] = cos(input[i])
func CosF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	cos_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// TanhF32 computes hyperbolic tangent: result[i] = tanh(input[i])
func TanhF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	tanh_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SigmoidF32 computes sigmoid: result[i] = 1 / (1 + exp(-input[i]))
func SigmoidF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	sigmoid_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ============================================================================
// Int32 Arithmetic Operations
// ============================================================================

// AddI32 performs element-wise addition: result[i] = a[i] + b[i]
func AddI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	add_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SubI32 performs element-wise subtraction: result[i] = a[i] - b[i]
func SubI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	sub_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MulI32 performs element-wise multiplication: result[i] = a[i] * b[i]
func MulI32(a, b, result []int32) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	mul_i32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ============================================================================
// Int64 Arithmetic Operations
// ============================================================================

// AddI64 performs element-wise addition: result[i] = a[i] + b[i]
func AddI64(a, b, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	add_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SubI64 performs element-wise subtraction: result[i] = a[i] - b[i]
func SubI64(a, b, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	sub_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ============================================================================
// Int64 Bitwise Operations
// ============================================================================

// AndI64 performs bitwise AND: result[i] = a[i] & b[i]
func AndI64(a, b, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	and_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// OrI64 performs bitwise OR: result[i] = a[i] | b[i]
func OrI64(a, b, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	or_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// XorI64 performs bitwise XOR: result[i] = a[i] ^ b[i]
func XorI64(a, b, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	xor_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ============================================================================
// Int64 Shift Operations
// ============================================================================

// ShiftLeftI64 performs left shift: result[i] = a[i] << shift
func ShiftLeftI64(a []int64, shift int, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	s := int64(shift)
	shl_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&s), unsafe.Pointer(&n))
}

// ShiftRightI64 performs arithmetic right shift: result[i] = a[i] >> shift
func ShiftRightI64(a []int64, shift int, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	s := int64(shift)
	shr_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&s), unsafe.Pointer(&n))
}

// ============================================================================
// Int64 Comparison Operations
// ============================================================================

// EqI64 compares for equality: result[i] = (a[i] == b[i]) ? -1 : 0
func EqI64(a, b, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	eq_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// GtI64 compares for greater than: result[i] = (a[i] > b[i]) ? -1 : 0
func GtI64(a, b, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	gt_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// GeI64 compares for greater or equal: result[i] = (a[i] >= b[i]) ? -1 : 0
func GeI64(a, b, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	ge_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// LtI64 compares for less than: result[i] = (a[i] < b[i]) ? -1 : 0
func LtI64(a, b, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	lt_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// LeI64 compares for less or equal: result[i] = (a[i] <= b[i]) ? -1 : 0
func LeI64(a, b, result []int64) {
	if len(a) == 0 {
		return
	}
	n := int64(len(a))
	le_i64_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ============================================================================
// Int64 If-Then-Else
// ============================================================================

// IfThenElseI64 selects elements: result[i] = mask[i] ? yes[i] : no[i]
func IfThenElseI64(mask, yes, no, result []int64) {
	if len(mask) == 0 {
		return
	}
	n := int64(len(mask))
	ifthenelse_i64_neon(unsafe.Pointer(&mask[0]), unsafe.Pointer(&yes[0]), unsafe.Pointer(&no[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

