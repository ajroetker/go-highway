// Copyright 2024 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && arm64

package matmul

import "github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"

// =============================================================================
// Float32 SME FMOPA (true outer product)
// =============================================================================

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
	asm.BlockMulAddFMOPAF32(aT, b, c, blockDim)
}

// =============================================================================
// Float64 SME FMOPA
// =============================================================================

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
	asm.BlockMulAddFMOPAF64(aT, b, c, blockDim)
}
