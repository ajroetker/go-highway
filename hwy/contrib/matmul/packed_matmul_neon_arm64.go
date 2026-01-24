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

// packedMicroKernelNEONF32 wraps the GOAT-generated NEON micro-kernel.
// It adapts the signature to match the dispatched interface.
//
// Parameters:
//   - packedA: Pre-sliced packed A buffer for this micro-panel
//   - packedB: Pre-sliced packed B buffer for this micro-panel
//   - c: Output C matrix (row-major), full matrix
//   - n: Leading dimension of C (total columns)
//   - ir: Starting row index in C (absolute, not panel index)
//   - jr: Starting column index in C (absolute, not panel index)
//   - kc: K-blocking size (actual panelK, not cache param Kc)
//   - mr: Micro-tile row dimension
//   - nr: Micro-tile column dimension
func packedMicroKernelNEONF32(packedA []float32, packedB []float32, c []float32, n, ir, jr, kc, mr, nr int) {
	// packedA and packedB are already sliced to the correct offset by gebp
	// ir and jr are absolute row/column indices in C
	// C offset: row ir, column jr
	cOffset := ir*n + jr

	asm.PackedMicroKernelNEONF32(
		packedA,
		packedB,
		c[cOffset:],
		kc, n, mr, nr,
	)
}

// packedMicroKernelPartialNEONF32 handles edge micro-tiles with partial rows/columns.
func packedMicroKernelPartialNEONF32(packedA []float32, packedB []float32, c []float32, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	// packedA and packedB are already sliced to the correct offset by gebp
	// ir and jr are absolute row/column indices in C
	cOffset := ir*n + jr

	// The NEON kernel handles partial tiles internally via activeRows/activeCols parameters
	asm.PackedMicroKernelNEONF32(
		packedA,
		packedB,
		c[cOffset:],
		kc, n, activeRows, activeCols,
	)
}

// packedMicroKernelNEONF64 wraps the GOAT-generated NEON micro-kernel for float64.
func packedMicroKernelNEONF64(packedA []float64, packedB []float64, c []float64, n, ir, jr, kc, mr, nr int) {
	// packedA and packedB are already sliced to the correct offset by gebp
	// ir and jr are absolute row/column indices in C
	cOffset := ir*n + jr

	asm.PackedMicroKernelNEONF64(
		packedA,
		packedB,
		c[cOffset:],
		kc, n, mr, nr,
	)
}

func packedMicroKernelPartialNEONF64(packedA []float64, packedB []float64, c []float64, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	// packedA and packedB are already sliced to the correct offset by gebp
	// ir and jr are absolute row/column indices in C
	cOffset := ir*n + jr

	asm.PackedMicroKernelNEONF64(
		packedA,
		packedB,
		c[cOffset:],
		kc, n, activeRows, activeCols,
	)
}

// packedMicroKernelNEONF16 wraps the GOAT-generated NEON FP16 micro-kernel.
func packedMicroKernelNEONF16(packedA []hwy.Float16, packedB []hwy.Float16, c []hwy.Float16, n, ir, jr, kc, mr, nr int) {
	// packedA and packedB are already sliced to the correct offset by gebp
	// ir and jr are absolute row/column indices in C
	cOffset := ir*n + jr

	asm.PackedMicroKernelNEONF16(
		packedA,
		packedB,
		c[cOffset:],
		kc, n, mr, nr,
	)
}

func packedMicroKernelPartialNEONF16(packedA []hwy.Float16, packedB []hwy.Float16, c []hwy.Float16, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	// packedA and packedB are already sliced to the correct offset by gebp
	// ir and jr are absolute row/column indices in C
	cOffset := ir*n + jr

	asm.PackedMicroKernelNEONF16(
		packedA,
		packedB,
		c[cOffset:],
		kc, n, activeRows, activeCols,
	)
}

// packedMicroKernelNEONBF16 wraps the GOAT-generated NEON BF16 micro-kernel.
func packedMicroKernelNEONBF16(packedA []hwy.BFloat16, packedB []hwy.BFloat16, c []hwy.BFloat16, n, ir, jr, kc, mr, nr int) {
	// packedA and packedB are already sliced to the correct offset by gebp
	// ir and jr are absolute row/column indices in C
	cOffset := ir*n + jr

	asm.PackedMicroKernelNEONBF16(
		packedA,
		packedB,
		c[cOffset:],
		kc, n, mr, nr,
	)
}

func packedMicroKernelPartialNEONBF16(packedA []hwy.BFloat16, packedB []hwy.BFloat16, c []hwy.BFloat16, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	// packedA and packedB are already sliced to the correct offset by gebp
	// ir and jr are absolute row/column indices in C
	cOffset := ir*n + jr

	asm.PackedMicroKernelNEONBF16(
		packedA,
		packedB,
		c[cOffset:],
		kc, n, activeRows, activeCols,
	)
}

func init() {
	// On ARM64 without SME, use NEON assembly micro-kernels for packed GEBP
	// This overrides the pure Go hwy implementation with optimized NEON
	//
	// Skip NEON assembly if HWY_NO_SIMD is set - use pure Go fallback instead.
	if hwy.NoSimdEnv() {
		return
	}

	// Only enable if hwy detected NEON (lanes >= 4 for float32).
	// This avoids using NEON assembly on emulators or fallback environments
	// where NEON might not be properly supported.
	lanesF32 := hwy.Zero[float32]().NumLanes()
	hasNEON := lanesF32 >= 4

	if hasNEON && !hwy.HasSME() {
		// Float32
		PackedMicroKernelFloat32 = packedMicroKernelNEONF32
		PackedMicroKernelPartialFloat32 = packedMicroKernelPartialNEONF32

		// Float64
		PackedMicroKernelFloat64 = packedMicroKernelNEONF64
		PackedMicroKernelPartialFloat64 = packedMicroKernelPartialNEONF64
	}

	// F16: Requires ARMv8.2-A FP16 extension
	if hasNEON && hwy.HasARMFP16() && !hwy.HasSME() {
		PackedMicroKernelFloat16 = packedMicroKernelNEONF16
		PackedMicroKernelPartialFloat16 = packedMicroKernelPartialNEONF16
	}

	// BF16: Requires ARMv8.6-A BF16 extension
	if hasNEON && hwy.HasARMBF16() && !hwy.HasSME() {
		PackedMicroKernelBFloat16 = packedMicroKernelNEONBF16
		PackedMicroKernelPartialBFloat16 = packedMicroKernelPartialNEONBF16
	}
}
