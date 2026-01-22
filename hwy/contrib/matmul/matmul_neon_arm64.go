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
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// Minimum dimensions to use NEON vectorization
const minDimForNEON = 16

//go:noescape
func matmul_neon_f32(a, b, c unsafe.Pointer, m, n, k int64)


// matmulNEON uses ARM NEON FMLA instructions for matrix multiplication.
// Falls back to scalar for small matrices.
func matmulNEON(a, b, c []float32, m, n, k int) {
	// For small matrices, use scalar
	if m < minDimForNEON || n < minDimForNEON || k < minDimForNEON {
		matmulScalar(a, b, c, m, n, k)
		return
	}

	matmul_neon_f32(
		unsafe.Pointer(unsafe.SliceData(a)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		int64(m),
		int64(n),
		int64(k),
	)
}

// matmulNEONF16 uses ARM NEON for float16 matrix multiplication.
// Uses hand-written assembly with FMLA f16 instructions.
func matmulNEONF16(a, b, c []hwy.Float16, m, n, k int) {
	// For small matrices, use generated fallback
	if m < minDimForNEON || n < minDimForNEON || k < minDimForNEON {
		BaseMatMul_neon_Float16(a, b, c, m, n, k)
		return
	}
	asm.MatMulNEONF16(a, b, c, m, n, k)
}

// matmulNEONBF16 uses ARM NEON for bfloat16 matrix multiplication.
// Uses hand-written assembly with BFDOT bf16 instructions.
func matmulNEONBF16(a, b, c []hwy.BFloat16, m, n, k int) {
	// For small matrices, use generated fallback
	if m < minDimForNEON || n < minDimForNEON || k < minDimForNEON {
		BaseMatMul_neon_BFloat16(a, b, c, m, n, k)
		return
	}
	asm.MatMulNEONBF16(a, b, c, m, n, k)
}

func init() {
	// Use hand-written NEON implementation on arm64
	// This overrides the generated dispatch for better performance
	// Will be overridden by SME if available
	MatMulFloat32 = matmulNEON

	// FP16/BF16 NEON instructions require ARMv8.2+ extensions
	// Only use them if the CPU supports the extensions
	if hwy.HasARMFP16() {
		MatMulFloat16 = matmulNEONF16
	}
	if hwy.HasARMBF16() {
		MatMulBFloat16 = matmulNEONBF16
	}
}
