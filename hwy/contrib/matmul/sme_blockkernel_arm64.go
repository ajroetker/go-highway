//go:build !noasm && darwin && arm64

// NOTE: This file is named "sme_blockkernel_arm64.go" (starting with 's')
// to ensure its init() runs AFTER "dispatch_blockkernel_arm64.gen.go" (starting with 'd').
// Go executes init() functions in lexicographic filename order within a package.
// The generated dispatch sets BlockMulAddFloat32/64 to NEON; this file's init()
// must run afterward to override with the SME FMOPA implementation when available.

package matmul

import "github.com/ajroetker/go-highway/hwy"

// blockMulAddFMOPAWrapper wraps the FMOPA implementation with dimension checks.
// Falls back to NEON for non-aligned dimensions or small blocks.
func blockMulAddFMOPAWrapper(aT, b, c []float32, blockDim int) {
	// FMOPA requires blockDim to be multiple of 16 (tile size for f32)
	if blockDim%16 != 0 || blockDim < 16 {
		BlockMulAddNEON(aT, b, c, blockDim)
		return
	}
	BlockMulAddFMOPA(aT, b, c, blockDim)
}

// blockMulAddFMOPAWrapper64 wraps the FMOPA implementation for float64.
// Falls back to NEON for non-aligned dimensions or small blocks.
func blockMulAddFMOPAWrapper64(aT, b, c []float64, blockDim int) {
	// FMOPA f64 requires blockDim to be multiple of 8 (tile size for f64)
	if blockDim%8 != 0 || blockDim < 8 {
		BlockMulAddNEONFloat64(aT, b, c, blockDim)
		return
	}
	BlockMulAddFMOPAFloat64(aT, b, c, blockDim)
}

func init() {
	if hwy.HasSME() {
		// Override dispatch to use FMOPA for aligned dimensions
		BlockMulAddFloat32 = blockMulAddFMOPAWrapper
		BlockMulAddFloat64 = blockMulAddFMOPAWrapper64
	}
}
