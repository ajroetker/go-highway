//go:build !noasm && darwin && arm64

// SME Matrix Multiplication for ARM64 with SME extension
// Uses FMOPA outer product accumulate with ZA tiles for efficient matrix multiply.
package asm

import (
	"runtime"
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// smeGuard locks the current goroutine to its OS thread and attempts to
// minimize the chance of async preemption during SME streaming mode.
// SME streaming mode makes ASIMD (NEON) instructions illegal, but Go's
// runtime uses ASIMD for signal handlers and other operations.
//
//go:nosplit
func smeGuard() func() {
	runtime.LockOSThread()
	return runtime.UnlockOSThread
}

// -march=armv9-a+sme+sme-f64f64+sme-f16f16+bf16 enables SME with f32/f64/f16/bf16 support
//go:generate go tool goat ../c/matmul_sme_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv9-a+sme+sme-f64f64+sme-f16f16+bf16"

// ============================================================================
// SME FMOPA Matrix Multiplication
// ============================================================================

// MatMulFMOPAF32 performs matrix multiplication using SME FMOPA: C = AT^T * B
// AT is the transposed A matrix (K x M, row-major) for contiguous column access.
// B is K x N (row-major), C is M x N (row-major).
//
// Requires M, N to be multiples of 16 (SVL = 512 bits = 16 x float32).
// Uses 16x16 tile processing with outer product accumulate.
//
// Parameters:
//   - at: K x M matrix (A transposed, row-major)
//   - b: K x N matrix (row-major)
//   - c: M x N matrix (row-major, output)
//   - m, n, k: matrix dimensions
func MatMulFMOPAF32(at, b, c []float32, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(at) < k*m || len(b) < k*n || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	matmul_fmopa_at_f32(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

// MatMulFMOPAF64 performs matrix multiplication using SME FMOPA: C = AT^T * B
// AT is the transposed A matrix (K x M, row-major) for contiguous column access.
// B is K x N (row-major), C is M x N (row-major).
//
// Requires M, N to be multiples of 8 (SVL = 512 bits = 8 x float64).
// Uses 8x8 tile processing with outer product accumulate.
//
// Parameters:
//   - at: K x M matrix (A transposed, row-major)
//   - b: K x N matrix (row-major)
//   - c: M x N matrix (row-major, output)
//   - m, n, k: matrix dimensions
func MatMulFMOPAF64(at, b, c []float64, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(at) < k*m || len(b) < k*n || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	matmul_fmopa_at_f64(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

// MatMulFMOPAF16 performs matrix multiplication using SME FMOPA: C = AT^T * B
// AT is the transposed A matrix (K x M, row-major) for contiguous column access.
// B is K x N (row-major), C is M x N (row-major).
//
// Uses widening approach: f16 -> f32 -> FMOPA -> f32 -> f16
// Requires M, N to be multiples of 16 (same as f32 since accumulator is f32).
// Uses 16x16 tile processing with outer product accumulate.
//
// Parameters:
//   - at: K x M matrix (A transposed, row-major)
//   - b: K x N matrix (row-major)
//   - c: M x N matrix (row-major, output)
//   - m, n, k: matrix dimensions
func MatMulFMOPAF16(at, b, c []hwy.Float16, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(at) < k*m || len(b) < k*n || len(c) < m*n {
		return
	}
	// Lock OS thread to prevent goroutine migration during SME streaming mode
	defer smeGuard()()

	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	// Scratch buffer for f32->f16 conversion (avoids SVE-dependent stack allocation in SME code)
	var scratch [16]float32
	matmul_fmopa_at_f16(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)
}

// MatMulBFMOPABF16 performs matrix multiplication using SME BFMOPA: C = AT^T * B
// Uses widening BFMOPA: bf16 inputs accumulate to f32, then convert back.
// AT is the transposed A matrix (K x M, row-major) for contiguous column access.
// B is K x N (row-major), C is M x N (row-major).
//
// Requires M, N to be multiples of 16 (same as f32 since accumulator is f32).
// Uses 16x16 tile processing with widening outer product accumulate.
//
// Parameters:
//   - at: K x M matrix (A transposed, row-major)
//   - b: K x N matrix (row-major)
//   - c: M x N matrix (row-major, output)
//   - m, n, k: matrix dimensions
func MatMulBFMOPABF16(at, b, c []hwy.BFloat16, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(at) < k*m || len(b) < k*n || len(c) < m*n {
		return
	}
	// Lock OS thread to prevent goroutine migration during SME streaming mode
	defer smeGuard()()

	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	// Scratch buffer for f32->bf16 conversion (avoids SVE-dependent stack allocation in SME code)
	var scratch [16]float32
	matmul_bfmopa_at_bf16(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)
}

// Assembly function declarations are in matmul_sme_arm64.go (generated by GoAT)

// MatmulFmopaAtF16 is a low-level wrapper for the raw assembly function.
// Used by the main matmul package to avoid duplicate assembly.
func MatmulFmopaAtF16(at, b, c, pm, pn, pk, scratch unsafe.Pointer) {
	matmul_fmopa_at_f16(at, b, c, pm, pn, pk, scratch)
}

// MatmulBfmopaAtBF16 is a low-level wrapper for the raw assembly function.
// Used by the main matmul package to avoid duplicate assembly.
func MatmulBfmopaAtBF16(at, b, c, pm, pn, pk, scratch unsafe.Pointer) {
	matmul_bfmopa_at_bf16(at, b, c, pm, pn, pk, scratch)
}
