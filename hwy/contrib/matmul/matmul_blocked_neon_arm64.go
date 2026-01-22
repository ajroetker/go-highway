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

package matmul

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// Minimum dimensions to use NEON blocked matmul
const minDimForBlockedNEON = 16

// blockedMatMulNEONF16 uses ARM NEON for cache-tiled float16 matrix multiplication.
// Falls back to generated code for small matrices.
func blockedMatMulNEONF16(a, b, c []hwy.Float16, m, n, k int) {
	// For small matrices, use generated fallback
	if m < minDimForBlockedNEON || n < minDimForBlockedNEON || k < minDimForBlockedNEON {
		BaseBlockedMatMul_neon_Float16(a, b, c, m, n, k)
		return
	}
	asm.BlockedMatMulNEONF16(a, b, c, m, n, k)
}

// blockedMatMulNEONBF16 uses ARM NEON for cache-tiled bfloat16 matrix multiplication.
// Falls back to generated code for small matrices.
func blockedMatMulNEONBF16(a, b, c []hwy.BFloat16, m, n, k int) {
	// For small matrices, use generated fallback
	if m < minDimForBlockedNEON || n < minDimForBlockedNEON || k < minDimForBlockedNEON {
		BaseBlockedMatMul_neon_BFloat16(a, b, c, m, n, k)
		return
	}
	asm.BlockedMatMulNEONBF16(a, b, c, m, n, k)
}

func init() {
	// Override blocked matmul dispatch for F16 and BF16 on ARM64
	// This uses optimized NEON assembly instead of generated fallback
	// FP16/BF16 NEON instructions require ARMv8.2+ extensions
	if hwy.HasARMFP16() {
		BlockedMatMulFloat16 = blockedMatMulNEONF16
	}
	if hwy.HasARMBF16() {
		BlockedMatMulBFloat16 = blockedMatMulNEONBF16
	}
}
