// Copyright 2024 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64

package matmul

import "unsafe"

// =============================================================================
// Float32 NEON
// =============================================================================

// block_muladd_neon_f32 computes C += A^T * B using NEON SIMD.
// A must be pre-transposed (rows are original A columns).
// Uses register blocking for efficiency.
//
//go:noescape
func block_muladd_neon_f32(aT, b, c unsafe.Pointer, blockDim int64)

// BlockMulAddNEON computes C += A^T * B for square blocks using hand-optimized NEON.
// aT must be pre-transposed (rows are original A columns).
// b is normal row-major (rows are B rows).
// This computes C += A * B where A is the original (non-transposed) matrix.
func BlockMulAddNEON(aT, b, c []float32, blockDim int) {
	if len(aT) < blockDim*blockDim {
		panic("BlockMulAddNEON: aT slice too short")
	}
	if len(b) < blockDim*blockDim {
		panic("BlockMulAddNEON: B slice too short")
	}
	if len(c) < blockDim*blockDim {
		panic("BlockMulAddNEON: C slice too short")
	}
	block_muladd_neon_f32(unsafe.Pointer(&aT[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), int64(blockDim))
}

// =============================================================================
// Float64 NEON
// =============================================================================

// block_muladd_neon_f64 computes C += A^T * B using NEON SIMD for float64.
// A must be pre-transposed (rows are original A columns).
// Uses register blocking for efficiency.
//
//go:noescape
func block_muladd_neon_f64(aT, b, c unsafe.Pointer, blockDim int64)

// BlockMulAddNEONFloat64 computes C += A^T * B for square blocks using hand-optimized NEON.
// aT must be pre-transposed (rows are original A columns).
// b is normal row-major (rows are B rows).
func BlockMulAddNEONFloat64(aT, b, c []float64, blockDim int) {
	if len(aT) < blockDim*blockDim {
		panic("BlockMulAddNEONFloat64: aT slice too short")
	}
	if len(b) < blockDim*blockDim {
		panic("BlockMulAddNEONFloat64: B slice too short")
	}
	if len(c) < blockDim*blockDim {
		panic("BlockMulAddNEONFloat64: C slice too short")
	}
	block_muladd_neon_f64(unsafe.Pointer(&aT[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), int64(blockDim))
}

// =============================================================================
// Float32 SME FMOPA (true outer product)
// =============================================================================

// block_muladd_fmopa_f32 computes C += A^T * B using SME FMOPA outer products.
// A must be pre-transposed (rows are original A columns).
// Uses 16x16 outer product tiles for maximum performance on Apple Silicon.
//
//go:noescape
func block_muladd_fmopa_f32(aT, b, c unsafe.Pointer, blockDim int64)

// BlockMulAddFMOPA computes C += A^T * B for square blocks using SME FMOPA.
// aT must be pre-transposed (rows are original A columns).
// b is normal row-major (rows are B rows).
// Uses true outer product instructions for 16x16 tiles.
func BlockMulAddFMOPA(aT, b, c []float32, blockDim int) {
	if len(aT) < blockDim*blockDim {
		panic("BlockMulAddFMOPA: aT slice too short")
	}
	if len(b) < blockDim*blockDim {
		panic("BlockMulAddFMOPA: B slice too short")
	}
	if len(c) < blockDim*blockDim {
		panic("BlockMulAddFMOPA: C slice too short")
	}
	block_muladd_fmopa_f32(unsafe.Pointer(&aT[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), int64(blockDim))
}

// =============================================================================
// Float64 SME FMOPA
// =============================================================================

// block_muladd_fmopa_f64 computes C += A^T * B using SME FMOPA for float64.
// A must be pre-transposed (rows are original A columns).
// Uses 8x8 outer product tiles (SVL=512 bits = 8 doubles).
//
//go:noescape
func block_muladd_fmopa_f64(aT, b, c unsafe.Pointer, blockDim int64)

// BlockMulAddFMOPAFloat64 computes C += A^T * B for square blocks using SME FMOPA.
// aT must be pre-transposed (rows are original A columns).
// b is normal row-major (rows are B rows).
func BlockMulAddFMOPAFloat64(aT, b, c []float64, blockDim int) {
	if len(aT) < blockDim*blockDim {
		panic("BlockMulAddFMOPAFloat64: aT slice too short")
	}
	if len(b) < blockDim*blockDim {
		panic("BlockMulAddFMOPAFloat64: B slice too short")
	}
	if len(c) < blockDim*blockDim {
		panic("BlockMulAddFMOPAFloat64: C slice too short")
	}
	block_muladd_fmopa_f64(unsafe.Pointer(&aT[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), int64(blockDim))
}
