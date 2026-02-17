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
