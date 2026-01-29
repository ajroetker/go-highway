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

// SME Transpose for ARM64 with SME extension
// Uses ZA tile for efficient matrix transpose.
package asm

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go tool goat ../c/transpose_sme_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv9-a+sme+sme-f64f64+sme-f16f16"

// TransposeSMEF32 transposes M×K float32 matrix to K×M using SME.
// Uses 16x16 tiles with ZA accumulator.
func TransposeSMEF32(src []float32, m, k int, dst []float32) {
	if m == 0 || k == 0 {
		return
	}
	if len(src) < m*k || len(dst) < k*m {
		return
	}
	// Lock OS thread and block SIGURG to prevent ZA register corruption
	defer SMEGuard()()

	mVal, kVal := int64(m), int64(k)
	transpose_sme_f32(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
	)
}

// TransposeSMEF64 transposes M×K float64 matrix to K×M using SME.
// Uses 8x8 tiles with ZA accumulator.
func TransposeSMEF64(src []float64, m, k int, dst []float64) {
	if m == 0 || k == 0 {
		return
	}
	if len(src) < m*k || len(dst) < k*m {
		return
	}
	defer SMEGuard()()

	mVal, kVal := int64(m), int64(k)
	transpose_sme_f64(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
	)
}

// TransposeSMEF16 transposes M×K float16 matrix to K×M using SME.
// Uses 32x32 tiles with ZA accumulator.
func TransposeSMEF16(src []hwy.Float16, m, k int, dst []hwy.Float16) {
	if m == 0 || k == 0 {
		return
	}
	if len(src) < m*k || len(dst) < k*m {
		return
	}
	defer SMEGuard()()

	mVal, kVal := int64(m), int64(k)
	transpose_sme_f16(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
	)
}

// TransposeSMEBF16 transposes M×K bfloat16 matrix to K×M using SME.
// Uses 32x32 tiles with ZA accumulator.
func TransposeSMEBF16(src []hwy.BFloat16, m, k int, dst []hwy.BFloat16) {
	if m == 0 || k == 0 {
		return
	}
	if len(src) < m*k || len(dst) < k*m {
		return
	}
	defer SMEGuard()()

	mVal, kVal := int64(m), int64(k)
	transpose_sme_bf16(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
	)
}
