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

// NEON Strided Transpose for ARM64
// Uses NEON TRN1/TRN2 for efficient tiled transpose with strided output.
// Enables parallel transpose by processing row strips independently.
package asm

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go tool goat ../c/transpose_strided_neon_arm64.c -O3 --target arm64 -e="-march=armv8.2-a+fp16"

// TransposeStridedNEONF32 transposes rows [rowStart, rowEnd) with dstM stride.
// This enables parallel transpose by processing row strips independently.
func TransposeStridedNEONF32(src []float32, rowStart, rowEnd, k, dstM int, dst []float32) {
	if rowStart >= rowEnd || k == 0 {
		return
	}
	rowStartVal, rowEndVal := int64(rowStart), int64(rowEnd)
	kVal, dstMVal := int64(k), int64(dstM)
	transpose_strided_neon_f32(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&rowStartVal),
		unsafe.Pointer(&rowEndVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&dstMVal),
	)
}

// TransposeStridedNEONF64 transposes rows [rowStart, rowEnd) with dstM stride.
func TransposeStridedNEONF64(src []float64, rowStart, rowEnd, k, dstM int, dst []float64) {
	if rowStart >= rowEnd || k == 0 {
		return
	}
	rowStartVal, rowEndVal := int64(rowStart), int64(rowEnd)
	kVal, dstMVal := int64(k), int64(dstM)
	transpose_strided_neon_f64(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&rowStartVal),
		unsafe.Pointer(&rowEndVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&dstMVal),
	)
}

// TransposeStridedNEONF16 transposes rows [rowStart, rowEnd) with dstM stride.
func TransposeStridedNEONF16(src []hwy.Float16, rowStart, rowEnd, k, dstM int, dst []hwy.Float16) {
	if rowStart >= rowEnd || k == 0 {
		return
	}
	rowStartVal, rowEndVal := int64(rowStart), int64(rowEnd)
	kVal, dstMVal := int64(k), int64(dstM)
	transpose_strided_neon_f16(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&rowStartVal),
		unsafe.Pointer(&rowEndVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&dstMVal),
	)
}

// TransposeStridedNEONBF16 transposes rows [rowStart, rowEnd) with dstM stride.
func TransposeStridedNEONBF16(src []hwy.BFloat16, rowStart, rowEnd, k, dstM int, dst []hwy.BFloat16) {
	if rowStart >= rowEnd || k == 0 {
		return
	}
	rowStartVal, rowEndVal := int64(rowStart), int64(rowEnd)
	kVal, dstMVal := int64(k), int64(dstM)
	transpose_strided_neon_bf16(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&rowStartVal),
		unsafe.Pointer(&rowEndVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&dstMVal),
	)
}
