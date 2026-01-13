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
