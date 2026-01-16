//go:build amd64 && goexperiment.simd

// BFloat16 SIMD operations for x86-64 with AVX-512 BF16 extension.
// Requires: Intel Cooper Lake (2020+), AMD Zen4+
//
// AVX-512 BF16 provides:
// - VCVTNEPS2BF16: Convert F32 to BF16 (round to nearest even)
// - VCVTNE2PS2BF16: Convert two F32 vectors to one BF16 vector
// - VDPBF16PS: BF16 dot product with F32 accumulator (KEY for ML!)
//
// NOTE: AVX-512 BF16 does NOT provide native BF16 arithmetic.
// Arithmetic operations use the promote -> compute (F32) -> demote pattern.
//
// BFloat16 is primarily used for ML training where the F32-compatible
// exponent range provides numerical stability during gradient updates.

package asm

import "unsafe"

// -mavx512bf16 enables AVX-512 BF16 instructions
//go:generate go tool goat ../c/ops_bf16_avx512_amd64.c -O3 --target amd64 -e="-mavx512bf16" -e="-mavx512f" -e="-mavx512vl" -e="-mavx512dq"

// ============================================================================
// BFloat16 <-> Float32 Conversions
// ============================================================================

// DemoteF32ToBF16AVX512 converts float32 to bfloat16 using VCVTNEPS2BF16.
// Uses round-to-nearest-even mode for optimal numerical behavior.
// Input: a - slice of float32
// Output: result - slice of uint16 (bfloat16 bit patterns)
func DemoteF32ToBF16AVX512(a []float32, result []uint16) {
	if len(a) == 0 || len(result) == 0 {
		return
	}
	n := int64(len(a))
	if int64(len(result)) < n {
		n = int64(len(result))
	}
	demote_f32_to_bf16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// DemoteTwoF32ToBF16AVX512 converts two float32 vectors to one bfloat16 vector.
// Uses VCVTNE2PS2BF16 to pack two F32 vectors into one BF16 vector.
// Input: lo, hi - slices of float32 (same length)
// Output: result - slice of uint16 with length 2*len(lo)
// Result layout: result[0:n] = demote(lo), result[n:2n] = demote(hi)
func DemoteTwoF32ToBF16AVX512(lo, hi []float32, result []uint16) {
	if len(lo) == 0 || len(hi) == 0 || len(result) == 0 {
		return
	}
	n := int64(len(lo))
	if int64(len(hi)) < n {
		n = int64(len(hi))
	}
	if int64(len(result)) < n*2 {
		n = int64(len(result)) / 2
	}
	demote_two_f32_to_bf16_avx512(
		unsafe.Pointer(&lo[0]),
		unsafe.Pointer(&hi[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// PromoteBF16ToF32AVX512 converts bfloat16 to float32.
// BFloat16 promotion is trivial: left-shift by 16 bits.
// No special AVX-512 BF16 instruction exists for this.
// Input: a - slice of uint16 (bfloat16 bit patterns)
// Output: result - slice of float32
func PromoteBF16ToF32AVX512(a []uint16, result []float32) {
	if len(a) == 0 || len(result) == 0 {
		return
	}
	n := int64(len(a))
	if int64(len(result)) < n {
		n = int64(len(result))
	}
	promote_bf16_to_f32_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// ============================================================================
// BFloat16 Dot Product with Float32 Accumulator (VDPBF16PS)
// This is the KEY operation for ML inference and training!
// ============================================================================

// DotBF16AVX512 computes dot product of BF16 vectors with F32 accumulation.
// Uses VDPBF16PS instruction for maximum throughput.
//
// The operation processes BF16 pairs:
//   acc[i] += a[2j]*b[2j] + a[2j+1]*b[2j+1] for j in range
//
// Input: a, b - slices of uint16 (bfloat16 bit patterns), must be even length
// Input/Output: acc - slice of float32 accumulators (at least 16 elements for full utilization)
//
// This is the core operation for BF16 matrix multiplication and convolutions.
// The F32 accumulator prevents precision loss during accumulation.
func DotBF16AVX512(a, b []uint16, acc []float32) {
	if len(a) == 0 || len(b) == 0 || len(acc) == 0 {
		return
	}
	n := int64(len(a))
	if int64(len(b)) < n {
		n = int64(len(b))
	}
	dot_bf16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&acc[0]),
		unsafe.Pointer(&n),
	)
}

// ============================================================================
// BFloat16 Arithmetic Operations (via promote-compute-demote)
// ============================================================================

// AddBF16AVX512 performs element-wise addition: result[i] = a[i] + b[i]
// Uses promote -> F32 add -> demote pattern since AVX-512 BF16 lacks native arithmetic.
func AddBF16AVX512(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	add_bf16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// MulBF16AVX512 performs element-wise multiplication: result[i] = a[i] * b[i]
// Uses promote -> F32 mul -> demote pattern since AVX-512 BF16 lacks native arithmetic.
func MulBF16AVX512(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	mul_bf16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// FmaBF16AVX512 performs fused multiply-add: result[i] = a[i] * b[i] + c[i]
// Uses promote -> F32 FMA -> demote pattern.
// The F32 FMA is a single-rounding operation for better accuracy.
func FmaBF16AVX512(a, b, c, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), min(len(c), len(result)))))
	fma_bf16_avx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&n),
	)
}

// Assembly function declarations are in ops_bf16_avx512_amd64.go (generated by GoAT)
