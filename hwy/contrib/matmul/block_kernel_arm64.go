// Copyright 2024 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64

package matmul

import "github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"

// =============================================================================
// Float32 NEON
// =============================================================================

// BlockMulAddNEON computes C += A^T * B for square blocks using NEON.
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
	asm.BlockMulAddNEONF32(aT, b, c, blockDim)
}

// =============================================================================
// Float64 NEON
// =============================================================================

// BlockMulAddNEONFloat64 computes C += A^T * B for square blocks using NEON.
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
	asm.BlockMulAddNEONF64(aT, b, c, blockDim)
}
