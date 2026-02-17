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

// Blocked/Cache-Tiled NEON Matrix Multiplication wrappers for ARM64
package asm

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// BF16: Requires ARMv8.6-A with BF16 extension
//go:generate go tool goat ../c/matmul_blocked_bf16_arm64.c -O3 --target arm64 -e="-march=armv8.6-a+bf16"

// BlockedMatMulNEONBF16 performs cache-tiled matrix multiplication using NEON: C = A * B
// A is M x K (row-major), B is K x N (row-major), C is M x N (row-major).
//
// Uses f32 accumulation with BFDOT for bf16 computation.
// Requires ARMv8.6-A with BF16 extension.
//
// Parameters:
//   - a: M x K matrix (row-major)
//   - b: K x N matrix (row-major)
//   - c: M x N matrix (row-major, output)
//   - m, n, k: matrix dimensions
func BlockedMatMulNEONBF16(a, b, c []hwy.BFloat16, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(a) < m*k || len(b) < k*n || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	blocked_matmul_neon_bf16(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}
