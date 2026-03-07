//go:build !noasm && arm64

// Multi-Tile SME FMOPA Matrix Multiplication wrappers for ARM64
package asm

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// Generate assembly from C using goat
//go:generate go tool goat ../c/multitile_fmopa_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv9-a+sme+sme-f64f64+sme-f16f16+bf16"

// MultiTileMatMulFMOPAF32 performs multi-tile matrix multiplication using SME FMOPA: C = AT^T * B
// Uses all 4 ZA tiles (ZA0-ZA3) in a 2x2 arrangement for 32x32 output blocks,
// with single-tile fallback for 16-row/16-col remainders.
//
// AT is the transposed A matrix (K x M, row-major).
// B is K x N (row-major), C is M x N (row-major).
// Requires M, N to be multiples of 16.
func MultiTileMatMulFMOPAF32(at, b, c []float32, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(at) < k*m || len(b) < k*n || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	multitile_fmopa_at_f32(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

// MultiTileMatMulFMOPAF32Strided performs multi-tile FMOPA matmul writing to C
// with leading dimension ldc at column offset coff. B has leading dimension n.
// C[i, coff+j] = sum_p AT^T[i,p] * B[p,j] for i in [0,m), j in [0,n).
//
// This enables incremental B transpose: transpose a strip of B, call this
// function to write directly into the correct columns of the full output.
func MultiTileMatMulFMOPAF32Strided(at []float32, b []float32, c []float32, m, n, k, ldc, coff int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	ldcVal := int64(ldc)
	coffVal := int64(coff)
	multitile_fmopa_at_f32_strided(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&ldcVal),
		unsafe.Pointer(&coffVal),
	)
}

// MultiTileMatMulFMOPAF32NTile performs multi-tile FMOPA matmul on an N-tile.
// B is accessed with row stride ldb (full N of original matrix), but only n
// columns are processed (tile width). C is written with row stride ldc at
// column offset coff. Caller passes b + j0 as B pointer.
//
// This enables parallel SME across N tiles: pad+transpose A once (shared),
// then each worker calls this for its column range.
func MultiTileMatMulFMOPAF32NTile(at []float32, b []float32, c []float32, m, n, k, ldb, ldc, coff int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	ldbVal := int64(ldb)
	ldcVal := int64(ldc)
	coffVal := int64(coff)
	multitile_fmopa_at_f32_ntile(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&ldbVal),
		unsafe.Pointer(&ldcVal),
		unsafe.Pointer(&coffVal),
	)
}

// MultiTileMatMulFMOPAF64NTile performs multi-tile FMOPA matmul (float64) on an N-tile.
// Same as F32NTile but with 8×8 tiles. B accessed with row stride ldb,
// n columns processed, C written with row stride ldc at column offset coff.
func MultiTileMatMulFMOPAF64NTile(at []float64, b []float64, c []float64, m, n, k, ldb, ldc, coff int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	ldbVal := int64(ldb)
	ldcVal := int64(ldc)
	coffVal := int64(coff)
	multitile_fmopa_at_f64_ntile(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&ldbVal),
		unsafe.Pointer(&ldcVal),
		unsafe.Pointer(&coffVal),
	)
}

// MultiTileMatMulFMOPAF64Strided performs multi-tile FMOPA matmul (float64)
// writing to C with leading dimension ldc at column offset coff.
func MultiTileMatMulFMOPAF64Strided(at []float64, b []float64, c []float64, m, n, k, ldc, coff int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	ldcVal := int64(ldc)
	coffVal := int64(coff)
	multitile_fmopa_at_f64_strided(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&ldcVal),
		unsafe.Pointer(&coffVal),
	)
}

// MultiTileMatMulFMOPAF64 performs multi-tile matrix multiplication using SME FMOPA: C = AT^T * B
// Uses all 4 ZA tiles in a 2x2 arrangement for 16x16 output blocks (8x8 per tile),
// with single-tile fallback for 8-row/8-col remainders.
//
// Requires M, N to be multiples of 8.
func MultiTileMatMulFMOPAF64(at, b, c []float64, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(at) < k*m || len(b) < k*n || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	multitile_fmopa_at_f64(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

// MultiTileMatMulFMOPAF16 performs multi-tile matrix multiplication using SME FMOPA: C = AT^T * B
// Uses widening approach: f16 -> f32 FMOPA -> f16, with all 4 ZA tiles for 32x32 output blocks.
// Requires M, N to be multiples of 16.
func MultiTileMatMulFMOPAF16(at, b, c []hwy.Float16, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(at) < k*m || len(b) < k*n || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	var scratch [16]float32
	multitile_fmopa_at_f16(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)
}

// MultiTileMatMulFMOPAF16Strided performs multi-tile FMOPA matmul (float16)
// writing to C with leading dimension ldc at column offset coff.
func MultiTileMatMulFMOPAF16Strided(at []hwy.Float16, b []hwy.Float16, c []hwy.Float16, m, n, k, ldc, coff int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	ldcVal := int64(ldc)
	coffVal := int64(coff)
	var scratch [16]float32
	multitile_fmopa_at_f16_strided(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&ldcVal),
		unsafe.Pointer(&coffVal),
		unsafe.Pointer(&scratch[0]),
	)
}

// MultiTileMatMulFMOPAF16NTile performs multi-tile FMOPA matmul (float16) on an N-tile.
// B is accessed with row stride ldb (full N of original matrix), but only n
// columns are processed (tile width). C is written with row stride ldc at
// column offset coff. Caller passes b + j0 as B pointer.
func MultiTileMatMulFMOPAF16NTile(at []hwy.Float16, b []hwy.Float16, c []hwy.Float16, m, n, k, ldb, ldc, coff int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	ldbVal := int64(ldb)
	ldcVal := int64(ldc)
	coffVal := int64(coff)
	var scratch [16]float32
	multitile_fmopa_at_f16_ntile(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&ldbVal),
		unsafe.Pointer(&ldcVal),
		unsafe.Pointer(&coffVal),
		unsafe.Pointer(&scratch[0]),
	)
}

// MultiTileMatMulFMOPABF16 performs multi-tile matrix multiplication using SME FMOPA: C = AT^T * B
// Uses widening approach: bf16 -> f32 FMOPA -> bf16, with all 4 ZA tiles for 32x32 output blocks.
// Requires M, N to be multiples of 16.
func MultiTileMatMulFMOPABF16(at, b, c []hwy.BFloat16, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(at) < k*m || len(b) < k*n || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	var scratch [16]float32
	multitile_bfmopa_at_bf16(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)
}

// MultiTileMatMulFMOPABF16Strided performs multi-tile FMOPA matmul (bfloat16)
// writing to C with leading dimension ldc at column offset coff.
func MultiTileMatMulFMOPABF16Strided(at []hwy.BFloat16, b []hwy.BFloat16, c []hwy.BFloat16, m, n, k, ldc, coff int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	ldcVal := int64(ldc)
	coffVal := int64(coff)
	var scratch [16]float32
	multitile_bfmopa_at_bf16_strided(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&ldcVal),
		unsafe.Pointer(&coffVal),
		unsafe.Pointer(&scratch[0]),
	)
}

// MultiTileMatMulFMOPABF16NTile performs multi-tile FMOPA matmul (bfloat16) on an N-tile.
// B is accessed with row stride ldb (full N of original matrix), but only n
// columns are processed (tile width). C is written with row stride ldc at
// column offset coff. Caller passes b + j0 as B pointer.
func MultiTileMatMulFMOPABF16NTile(at []hwy.BFloat16, b []hwy.BFloat16, c []hwy.BFloat16, m, n, k, ldb, ldc, coff int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	ldbVal := int64(ldb)
	ldcVal := int64(ldc)
	coffVal := int64(coff)
	var scratch [16]float32
	multitile_bfmopa_at_bf16_ntile(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&ldbVal),
		unsafe.Pointer(&ldcVal),
		unsafe.Pointer(&coffVal),
		unsafe.Pointer(&scratch[0]),
	)
}
