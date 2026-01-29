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

// NOTE: This file is named "z_transpose_arm64.go" (starting with 'z')
// to ensure its init() runs AFTER the generated dispatch files.
// Go executes init() functions in lexicographic filename order within a package.

package matmul

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// Minimum size for NEON transpose (function call overhead dominates below this)
// Benchmarks show scalar is faster for very small matrices
const minSizeForNEONTranspose = 32

// Minimum size for SME (streaming mode has fixed overhead)
// Benchmarks on M4 show:
// - Float32/Float64: SME wins at 256x256 and above
// - Float16/BFloat16: SME wins at 512x512 and above (higher overhead)
const (
	minSizeForSMETransposeF32 = 256
	minSizeForSMETransposeF64 = 256
	minSizeForSMETransposeF16 = 512
)

// transposeScalar is a simple scalar transpose for small matrices
func transposeScalar[T any](src []T, m, k int, dst []T) {
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			dst[j*m+i] = src[i*k+j]
		}
	}
}

// transposeStridedScalar is a simple scalar strided transpose for small matrices
func transposeStridedScalar[T any](src []T, rowStart, rowEnd, k, dstM int, dst []T) {
	for i := rowStart; i < rowEnd; i++ {
		for j := 0; j < k; j++ {
			dst[j*dstM+i] = src[i*k+j]
		}
	}
}

func init() {
	// Override with NEON assembly implementations for large matrices
	// For small matrices, use pure scalar (hwygen SIMD has lane mismatch issues)
	Transpose2DFloat32 = func(src []float32, m, k int, dst []float32) {
		if m >= minSizeForNEONTranspose && k >= minSizeForNEONTranspose {
			asm.TransposeNEONF32(src, m, k, dst)
		} else {
			transposeScalar(src, m, k, dst)
		}
	}

	Transpose2DFloat64 = func(src []float64, m, k int, dst []float64) {
		if m >= minSizeForNEONTranspose && k >= minSizeForNEONTranspose {
			asm.TransposeNEONF64(src, m, k, dst)
		} else {
			transposeScalar(src, m, k, dst)
		}
	}

	Transpose2DFloat16 = func(src []hwy.Float16, m, k int, dst []hwy.Float16) {
		if m >= minSizeForNEONTranspose && k >= minSizeForNEONTranspose {
			asm.TransposeNEONF16(src, m, k, dst)
		} else {
			transposeScalar(src, m, k, dst)
		}
	}

	Transpose2DBFloat16 = func(src []hwy.BFloat16, m, k int, dst []hwy.BFloat16) {
		if m >= minSizeForNEONTranspose && k >= minSizeForNEONTranspose {
			asm.TransposeNEONBF16(src, m, k, dst)
		} else {
			transposeScalar(src, m, k, dst)
		}
	}

	// Strided transpose overrides for parallel transpose
	Transpose2DStridedFloat32 = func(src []float32, rowStart, rowEnd, k, dstM int, dst []float32) {
		numRows := rowEnd - rowStart
		if numRows >= minSizeForNEONTranspose && k >= minSizeForNEONTranspose {
			asm.TransposeStridedNEONF32(src, rowStart, rowEnd, k, dstM, dst)
		} else {
			transposeStridedScalar(src, rowStart, rowEnd, k, dstM, dst)
		}
	}

	Transpose2DStridedFloat64 = func(src []float64, rowStart, rowEnd, k, dstM int, dst []float64) {
		numRows := rowEnd - rowStart
		if numRows >= minSizeForNEONTranspose && k >= minSizeForNEONTranspose {
			asm.TransposeStridedNEONF64(src, rowStart, rowEnd, k, dstM, dst)
		} else {
			transposeStridedScalar(src, rowStart, rowEnd, k, dstM, dst)
		}
	}

	Transpose2DStridedFloat16 = func(src []hwy.Float16, rowStart, rowEnd, k, dstM int, dst []hwy.Float16) {
		numRows := rowEnd - rowStart
		if numRows >= minSizeForNEONTranspose && k >= minSizeForNEONTranspose {
			asm.TransposeStridedNEONF16(src, rowStart, rowEnd, k, dstM, dst)
		} else {
			transposeStridedScalar(src, rowStart, rowEnd, k, dstM, dst)
		}
	}

	Transpose2DStridedBFloat16 = func(src []hwy.BFloat16, rowStart, rowEnd, k, dstM int, dst []hwy.BFloat16) {
		numRows := rowEnd - rowStart
		if numRows >= minSizeForNEONTranspose && k >= minSizeForNEONTranspose {
			asm.TransposeStridedNEONBF16(src, rowStart, rowEnd, k, dstM, dst)
		} else {
			transposeStridedScalar(src, rowStart, rowEnd, k, dstM, dst)
		}
	}

	// Override with SME for large matrices when SME is available
	if hwy.HasSME() {
		neonF32 := Transpose2DFloat32
		Transpose2DFloat32 = func(src []float32, m, k int, dst []float32) {
			if m >= minSizeForSMETransposeF32 && k >= minSizeForSMETransposeF32 {
				asm.TransposeSMEF32(src, m, k, dst)
			} else {
				neonF32(src, m, k, dst)
			}
		}

		neonF64 := Transpose2DFloat64
		Transpose2DFloat64 = func(src []float64, m, k int, dst []float64) {
			if m >= minSizeForSMETransposeF64 && k >= minSizeForSMETransposeF64 {
				asm.TransposeSMEF64(src, m, k, dst)
			} else {
				neonF64(src, m, k, dst)
			}
		}

		neonF16 := Transpose2DFloat16
		Transpose2DFloat16 = func(src []hwy.Float16, m, k int, dst []hwy.Float16) {
			if m >= minSizeForSMETransposeF16 && k >= minSizeForSMETransposeF16 {
				asm.TransposeSMEF16(src, m, k, dst)
			} else {
				neonF16(src, m, k, dst)
			}
		}

		neonBF16 := Transpose2DBFloat16
		Transpose2DBFloat16 = func(src []hwy.BFloat16, m, k int, dst []hwy.BFloat16) {
			if m >= minSizeForSMETransposeF16 && k >= minSizeForSMETransposeF16 {
				asm.TransposeSMEBF16(src, m, k, dst)
			} else {
				neonBF16(src, m, k, dst)
			}
		}
	}
}
