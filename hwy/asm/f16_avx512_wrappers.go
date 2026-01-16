//go:build amd64 && goexperiment.simd

// Native Float16 SIMD operations for x86-64 with AVX-512 FP16 extension.
// Requires: Intel Sapphire Rapids (2023+), AMD Zen5+
//
// AVX-512 FP16 provides NATIVE float16 arithmetic with 32 lanes per 512-bit register!
// This eliminates the promote-compute-demote overhead of F16C (conversion-only).
//
// Key instructions:
// - VADDPH, VSUBPH, VMULPH, VDIVPH: Native FP16 arithmetic
// - VFMADD132PH: Fused multiply-add
// - VSQRTPH: Square root
// - VMINPH, VMAXPH: Min/max operations
// - VCVTPH2PSX, VCVTPS2PHX: Conversions (AVX-512 widths)

package asm

import "unsafe"

// -mavx512fp16 enables native AVX-512 FP16 arithmetic instructions
//go:generate go tool goat ../c/ops_f16_avx512_amd64.c -O3 --target amd64 -e="-mavx512fp16" -e="-mavx512f" -e="-mavx512vl"

// ============================================================================
// Native Float16 Arithmetic Operations
// ============================================================================

// AddF16AVX512 performs element-wise addition: result[i] = a[i] + b[i]
// Uses native VADDPH instruction with 32 float16 lanes per operation.
func AddF16AVX512(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	add_f16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// SubF16AVX512 performs element-wise subtraction: result[i] = a[i] - b[i]
// Uses native VSUBPH instruction with 32 float16 lanes per operation.
func SubF16AVX512(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	sub_f16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// MulF16AVX512 performs element-wise multiplication: result[i] = a[i] * b[i]
// Uses native VMULPH instruction with 32 float16 lanes per operation.
func MulF16AVX512(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	mul_f16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// DivF16AVX512 performs element-wise division: result[i] = a[i] / b[i]
// Uses native VDIVPH instruction with 32 float16 lanes per operation.
func DivF16AVX512(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	div_f16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// FmaF16AVX512 performs fused multiply-add: result[i] = a[i] * b[i] + c[i]
// Uses native VFMADD132PH instruction with 32 float16 lanes per operation.
// This is a single-rounding operation, more accurate than separate mul+add.
func FmaF16AVX512(a, b, c, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), min(len(c), len(result)))))
	fma_f16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// SqrtF16AVX512 performs element-wise square root: result[i] = sqrt(a[i])
// Uses native VSQRTPH instruction with 32 float16 lanes per operation.
func SqrtF16AVX512(a, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	sqrt_f16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// MinF16AVX512 performs element-wise minimum: result[i] = min(a[i], b[i])
// Uses native VMINPH instruction with 32 float16 lanes per operation.
func MinF16AVX512(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	min_f16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// MaxF16AVX512 performs element-wise maximum: result[i] = max(a[i], b[i])
// Uses native VMAXPH instruction with 32 float16 lanes per operation.
func MaxF16AVX512(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	max_f16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// ============================================================================
// Float16 <-> Float32 Conversions
// ============================================================================

// PromoteF16ToF32AVX512 converts float16 to float32 using AVX-512 FP16 instructions.
// Uses VCVTPH2PSX for efficient conversion with AVX-512 widths.
// Input: a - slice of uint16 (float16 bit patterns)
// Output: result - slice of float32
func PromoteF16ToF32AVX512(a []uint16, result []float32) {
	if len(a) == 0 || len(result) == 0 {
		return
	}
	n := int64(len(a))
	if int64(len(result)) < n {
		n = int64(len(result))
	}
	promote_f16_to_f32_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// DemoteF32ToF16AVX512 converts float32 to float16 using AVX-512 FP16 instructions.
// Uses VCVTPS2PHX for efficient conversion with AVX-512 widths.
// Input: a - slice of float32
// Output: result - slice of uint16 (float16 bit patterns)
func DemoteF32ToF16AVX512(a []float32, result []uint16) {
	if len(a) == 0 || len(result) == 0 {
		return
	}
	n := int64(len(a))
	if int64(len(result)) < n {
		n = int64(len(result))
	}
	demote_f32_to_f16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// Assembly function declarations are in ops_f16_avx512_amd64.go (generated by GoAT)
