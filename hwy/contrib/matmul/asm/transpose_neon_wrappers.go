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

// NEON Transpose for ARM64
// Uses NEON TRN1/TRN2 for efficient tiled transpose.
package asm

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go tool goat ../c/transpose_neon_arm64.c -O3 --target arm64 -e="-march=armv8.2-a+fp16"

// TransposeNEONF32 transposes M×K float32 matrix to K×M using NEON.
func TransposeNEONF32(src []float32, m, k int, dst []float32) {
	if m == 0 || k == 0 {
		return
	}
	if len(src) < m*k || len(dst) < k*m {
		return
	}
	mVal, kVal := int64(m), int64(k)
	transpose_neon_f32(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
	)
}

// TransposeNEONF64 transposes M×K float64 matrix to K×M using NEON.
func TransposeNEONF64(src []float64, m, k int, dst []float64) {
	if m == 0 || k == 0 {
		return
	}
	if len(src) < m*k || len(dst) < k*m {
		return
	}
	mVal, kVal := int64(m), int64(k)
	transpose_neon_f64(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
	)
}

// TransposeNEONF16 transposes M×K float16 matrix to K×M using NEON.
func TransposeNEONF16(src []hwy.Float16, m, k int, dst []hwy.Float16) {
	if m == 0 || k == 0 {
		return
	}
	if len(src) < m*k || len(dst) < k*m {
		return
	}
	mVal, kVal := int64(m), int64(k)
	transpose_neon_f16(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
	)
}

// TransposeNEONBF16 transposes M×K bfloat16 matrix to K×M using NEON.
func TransposeNEONBF16(src []hwy.BFloat16, m, k int, dst []hwy.BFloat16) {
	if m == 0 || k == 0 {
		return
	}
	if len(src) < m*k || len(dst) < k*m {
		return
	}
	mVal, kVal := int64(m), int64(k)
	transpose_neon_bf16(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
	)
}
