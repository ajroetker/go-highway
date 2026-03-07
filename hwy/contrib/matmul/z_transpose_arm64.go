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

// Minimum dimensions for NEON transpose.
// NEON uses 4×4 blocks (f32), 2×2 blocks (f64), or 8×8 blocks (f16/bf16)
// with scalar edge handling. Benchmarks on M4 Max show NEON is 5-12x faster
// than scalar even for small-M prefill shapes (e.g., 16×640, 4×1024).
// We require at least one full block per dimension.
const (
	minDimForNEONF32 = 4 // 4×4 TRN block
	minDimForNEONF64 = 2 // 2×2 TRN block
	minDimForNEONF16 = 8 // 8×8 TRN block
)

// Minimum total elements for SME transpose (streaming mode has fixed overhead).
// Benchmarks on M4 Max show SME only wins over NEON for large matrices:
// - f32: SME wins at 16×2048 (~32K elements) but loses at 16×1024 (~16K)
// - f64: SME is competitive at 16×640 (~10K) but NEON still edges it
// - For square matrices, SME wins at 256×256 (f32) and 512×512 (f16/bf16)
// We also require at least one full tile per dimension (16 for f32, 8 for f64, 32 for f16).
const (
	minDimForSMEF32       = 16    // 16×16 ZA tile
	minDimForSMEF64       = 8     // 8×8 ZA tile
	minDimForSMEF16       = 32    // 32×32 ZA tile
	minElemsForSMEF32     = 32768 // ~128 KB, 16×2048 or 256×128
	minElemsForSMEF64     = 32768 // ~256 KB
	minElemsForSMEF16     = 65536 // ~128 KB (2 bytes each)
)

// transposeScalar is a simple scalar transpose for small matrices
func transposeScalar[T any](src []T, m, k int, dst []T) {
	for i := range m {
		for j := range k {
			dst[j*m+i] = src[i*k+j]
		}
	}
}

// transposeStridedScalar is a simple scalar strided transpose for small matrices
func transposeStridedScalar[T any](src []T, rowStart, rowEnd, k, dstM int, dst []T) {
	for i := rowStart; i < rowEnd; i++ {
		for j := range k {
			dst[j*dstM+i] = src[i*k+j]
		}
	}
}

func init() {
	// Override with NEON assembly implementations.
	// NEON requires at least one full block per dimension (4×4 for f32, etc.)
	// but works well even for tall-skinny prefill shapes like 16×640.
	Transpose2DFloat32 = func(src []float32, m, k int, dst []float32) {
		if m >= minDimForNEONF32 && k >= minDimForNEONF32 {
			transpose2DAsmF32(src, m, k, dst)
		} else {
			transposeScalar(src, m, k, dst)
		}
	}

	Transpose2DFloat64 = func(src []float64, m, k int, dst []float64) {
		if m >= minDimForNEONF64 && k >= minDimForNEONF64 {
			transpose2DAsmF64(src, m, k, dst)
		} else {
			transposeScalar(src, m, k, dst)
		}
	}

	Transpose2DFloat16 = func(src []hwy.Float16, m, k int, dst []hwy.Float16) {
		if m >= minDimForNEONF16 && k >= minDimForNEONF16 {
			transpose2DAsmF16(src, m, k, dst)
		} else {
			transposeScalar(src, m, k, dst)
		}
	}

	Transpose2DBFloat16 = func(src []hwy.BFloat16, m, k int, dst []hwy.BFloat16) {
		if m >= minDimForNEONF16 && k >= minDimForNEONF16 {
			asm.TransposeNEONBF16(src, m, k, dst)
		} else {
			transposeScalar(src, m, k, dst)
		}
	}

	// Strided transpose overrides for parallel transpose
	Transpose2DStridedFloat32 = func(src []float32, rowStart, rowEnd, k, dstM int, dst []float32) {
		numRows := rowEnd - rowStart
		if numRows >= minDimForNEONF32 && k >= minDimForNEONF32 {
			transpose2DStridedAsmF32(src, rowStart, rowEnd, k, dstM, dst)
		} else {
			transposeStridedScalar(src, rowStart, rowEnd, k, dstM, dst)
		}
	}

	Transpose2DStridedFloat64 = func(src []float64, rowStart, rowEnd, k, dstM int, dst []float64) {
		numRows := rowEnd - rowStart
		if numRows >= minDimForNEONF64 && k >= minDimForNEONF64 {
			transpose2DStridedAsmF64(src, rowStart, rowEnd, k, dstM, dst)
		} else {
			transposeStridedScalar(src, rowStart, rowEnd, k, dstM, dst)
		}
	}

	Transpose2DStridedFloat16 = func(src []hwy.Float16, rowStart, rowEnd, k, dstM int, dst []hwy.Float16) {
		numRows := rowEnd - rowStart
		if numRows >= minDimForNEONF16 && k >= minDimForNEONF16 {
			transpose2DStridedAsmF16(src, rowStart, rowEnd, k, dstM, dst)
		} else {
			transposeStridedScalar(src, rowStart, rowEnd, k, dstM, dst)
		}
	}

	Transpose2DStridedBFloat16 = func(src []hwy.BFloat16, rowStart, rowEnd, k, dstM int, dst []hwy.BFloat16) {
		numRows := rowEnd - rowStart
		if numRows >= minDimForNEONF16 && k >= minDimForNEONF16 {
			asm.TransposeStridedNEONBF16(src, rowStart, rowEnd, k, dstM, dst)
		} else {
			transposeStridedScalar(src, rowStart, rowEnd, k, dstM, dst)
		}
	}

	// Override with SME for large matrices when SME is available.
	// SME requires at least one full tile per dimension AND enough total
	// elements to amortize streaming mode entry/exit overhead.
	if hwy.HasSME() {
		neonF32 := Transpose2DFloat32
		Transpose2DFloat32 = func(src []float32, m, k int, dst []float32) {
			if m >= minDimForSMEF32 && k >= minDimForSMEF32 && m*k >= minElemsForSMEF32 {
				asm.TransposeSMEF32(src, m, k, dst)
			} else {
				neonF32(src, m, k, dst)
			}
		}

		neonF64 := Transpose2DFloat64
		Transpose2DFloat64 = func(src []float64, m, k int, dst []float64) {
			if m >= minDimForSMEF64 && k >= minDimForSMEF64 && m*k >= minElemsForSMEF64 {
				asm.TransposeSMEF64(src, m, k, dst)
			} else {
				neonF64(src, m, k, dst)
			}
		}

		neonF16 := Transpose2DFloat16
		Transpose2DFloat16 = func(src []hwy.Float16, m, k int, dst []hwy.Float16) {
			if m >= minDimForSMEF16 && k >= minDimForSMEF16 && m*k >= minElemsForSMEF16 {
				asm.TransposeSMEF16(src, m, k, dst)
			} else {
				neonF16(src, m, k, dst)
			}
		}

		neonBF16 := Transpose2DBFloat16
		Transpose2DBFloat16 = func(src []hwy.BFloat16, m, k int, dst []hwy.BFloat16) {
			if m >= minDimForSMEF16 && k >= minDimForSMEF16 && m*k >= minElemsForSMEF16 {
				asm.TransposeSMEBF16(src, m, k, dst)
			} else {
				neonBF16(src, m, k, dst)
			}
		}
	}
}
