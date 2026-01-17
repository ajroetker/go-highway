//go:build !noasm && darwin && arm64

// SME Matrix-Vector Multiplication for ARM64 with SME extension
// Uses FMOPA outer product accumulate with ZA tiles for efficient matvec.
package asm

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// -march=armv9-a+sme+sme-f64f64+sme-f16f16+bf16 enables SME with f32/f64/f16/bf16 support
//go:generate go tool goat ../c/matvec_sme_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv9-a+sme+sme-f64f64+sme-f16f16+bf16"

// ============================================================================
// SME FMOPA Matrix-Vector Multiplication
// ============================================================================

// MatVecSMEF32 performs matrix-vector multiplication using SME FMOPA: result = MT^T * v = M * v
// MT is the transposed M matrix (cols x rows, row-major) for contiguous column access.
// v is the input vector (cols elements), result is the output vector (rows elements).
//
// Requires rows to be a multiple of 16 (SVL = 512 bits = 16 x float32).
//
// Parameters:
//   - mt: cols x rows matrix (M transposed, row-major)
//   - v: input vector (cols elements)
//   - result: output vector (rows elements)
//   - rows, cols: matrix dimensions
func MatVecSMEF32(mt []float32, v []float32, result []float32, rows, cols int) {
	if rows == 0 || cols == 0 {
		return
	}
	if len(mt) < cols*rows || len(v) < cols || len(result) < rows {
		return
	}
	rowsVal := int64(rows)
	colsVal := int64(cols)
	matvec_sme_f32(
		unsafe.Pointer(&mt[0]),
		unsafe.Pointer(&v[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&rowsVal),
		unsafe.Pointer(&colsVal),
	)
}

// MatVecSMEF64 performs matrix-vector multiplication using SME FMOPA: result = MT^T * v = M * v
// MT is the transposed M matrix (cols x rows, row-major) for contiguous column access.
// v is the input vector (cols elements), result is the output vector (rows elements).
//
// Requires rows to be a multiple of 8 (SVL = 512 bits = 8 x float64).
//
// Parameters:
//   - mt: cols x rows matrix (M transposed, row-major)
//   - v: input vector (cols elements)
//   - result: output vector (rows elements)
//   - rows, cols: matrix dimensions
func MatVecSMEF64(mt []float64, v []float64, result []float64, rows, cols int) {
	if rows == 0 || cols == 0 {
		return
	}
	if len(mt) < cols*rows || len(v) < cols || len(result) < rows {
		return
	}
	rowsVal := int64(rows)
	colsVal := int64(cols)
	matvec_sme_f64(
		unsafe.Pointer(&mt[0]),
		unsafe.Pointer(&v[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&rowsVal),
		unsafe.Pointer(&colsVal),
	)
}

// MatVecSMEF16 performs matrix-vector multiplication using SME FMOPA: result = MT^T * v = M * v
// MT is the transposed M matrix (cols x rows, row-major) for contiguous column access.
// v is the input vector (cols elements), result is the output vector (rows elements).
//
// Uses widening approach: f16 -> f32 -> FMOPA -> f32 -> f16
// Requires rows to be a multiple of 16 (same as f32 since accumulator is f32).
//
// Parameters:
//   - mt: cols x rows matrix (M transposed, row-major)
//   - v: input vector (cols elements)
//   - result: output vector (rows elements)
//   - rows, cols: matrix dimensions
func MatVecSMEF16(mt []hwy.Float16, v []hwy.Float16, result []hwy.Float16, rows, cols int) {
	if rows == 0 || cols == 0 {
		return
	}
	if len(mt) < cols*rows || len(v) < cols || len(result) < rows {
		return
	}
	rowsVal := int64(rows)
	colsVal := int64(cols)
	// Scratch buffer for f32->f16 conversion (avoids SVE-dependent stack allocation in SME code)
	var scratch [16]float32
	matvec_sme_f16(
		unsafe.Pointer(&mt[0]),
		unsafe.Pointer(&v[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&rowsVal),
		unsafe.Pointer(&colsVal),
		unsafe.Pointer(&scratch[0]),
	)
}

// MatVecSMEBF16 performs matrix-vector multiplication using SME BFMOPA: result = MT^T * v = M * v
// Uses widening BFMOPA: bf16 inputs accumulate to f32, then convert back.
// MT is the transposed M matrix (cols x rows, row-major) for contiguous column access.
// v is the input vector (cols elements), result is the output vector (rows elements).
//
// Requires rows to be a multiple of 16 (same as f32 since accumulator is f32).
//
// Parameters:
//   - mt: cols x rows matrix (M transposed, row-major)
//   - v: input vector (cols elements)
//   - result: output vector (rows elements)
//   - rows, cols: matrix dimensions
func MatVecSMEBF16(mt []hwy.BFloat16, v []hwy.BFloat16, result []hwy.BFloat16, rows, cols int) {
	if rows == 0 || cols == 0 {
		return
	}
	if len(mt) < cols*rows || len(v) < cols || len(result) < rows {
		return
	}
	rowsVal := int64(rows)
	colsVal := int64(cols)
	// Scratch buffer for f32->bf16 conversion (avoids SVE-dependent stack allocation in SME code)
	var scratch [16]float32
	matvec_sme_bf16(
		unsafe.Pointer(&mt[0]),
		unsafe.Pointer(&v[0]),
		unsafe.Pointer(&result[0]),
		unsafe.Pointer(&rowsVal),
		unsafe.Pointer(&colsVal),
		unsafe.Pointer(&scratch[0]),
	)
}

// Assembly function declarations are in matvec_sme_arm64.go (generated by GoAT)
