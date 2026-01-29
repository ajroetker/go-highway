// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !noasm && arm64

// MatMulKLast NEON implementations for ARM64
// Uses tiled dot-product algorithm optimized for K-last layout.
package asm

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// Generate NEON assembly from C source
//go:generate go tool goat ../c/matmul_klast_neon_arm64.c -O3 --target arm64 -e="-march=armv8.2-a+fp16+bf16"

// ============================================================================
// MatMulKLast NEON - K-Last Layout (PyTorch weights)
// ============================================================================
// Computes C = A * B^T where:
//   - A is M x K (row-major, K last)
//   - B is N x K (row-major, K last)
//   - C is M x N (row-major)
//
// This is the natural layout for PyTorch weights and avoids transpose overhead.

// MatMulKLastNEONF32 performs KLast matrix multiplication using NEON: C = A * B^T
// Uses tiled 4×4 dot-product algorithm with horizontal sums.
//
// Parameters:
//   - a: M x K matrix (row-major)
//   - b: N x K matrix (row-major)
//   - c: M x N matrix (row-major, output)
//   - m, n, k: matrix dimensions
func MatMulKLastNEONF32(a, b, c []float32, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(a) < m*k || len(b) < n*k || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	matmul_klast_neon_f32(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

// MatMulKLastNEONF32Aligned performs KLast matrix multiplication for aligned dimensions.
// Fast path when M and N are multiples of 4 (no boundary checks).
//
// Parameters:
//   - a: M x K matrix (row-major)
//   - b: N x K matrix (row-major)
//   - c: M x N matrix (row-major, output)
//   - m, n, k: matrix dimensions (M, N must be multiples of 4)
func MatMulKLastNEONF32Aligned(a, b, c []float32, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(a) < m*k || len(b) < n*k || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	matmul_klast_neon_f32_aligned(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

// MatMulKLastNEONF64 performs KLast matrix multiplication using NEON: C = A * B^T
// Uses tiled 2×2 dot-product algorithm with horizontal sums.
//
// Parameters:
//   - a: M x K matrix (row-major)
//   - b: N x K matrix (row-major)
//   - c: M x N matrix (row-major, output)
//   - m, n, k: matrix dimensions
func MatMulKLastNEONF64(a, b, c []float64, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(a) < m*k || len(b) < n*k || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	matmul_klast_neon_f64(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

// MatMulKLastNEONF16 performs KLast matrix multiplication using NEON: C = A * B^T
// Uses f16 loads with f32 accumulation for precision.
//
// Parameters:
//   - a: M x K matrix (row-major)
//   - b: N x K matrix (row-major)
//   - c: M x N matrix (row-major, output)
//   - m, n, k: matrix dimensions
func MatMulKLastNEONF16(a, b, c []hwy.Float16, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(a) < m*k || len(b) < n*k || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	matmul_klast_neon_f16(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

// MatMulKLastNEONBF16 performs KLast matrix multiplication using NEON: C = A * B^T
// Uses BFDOT for bf16 computation with f32 accumulation.
//
// Parameters:
//   - a: M x K matrix (row-major)
//   - b: N x K matrix (row-major)
//   - c: M x N matrix (row-major, output)
//   - m, n, k: matrix dimensions
func MatMulKLastNEONBF16(a, b, c []hwy.BFloat16, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(a) < m*k || len(b) < n*k || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	matmul_klast_neon_bf16(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

// Assembly function declarations (generated by GoAT from matmul_klast_neon_arm64.c)
