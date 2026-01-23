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
//   - packedA: Packed A buffer, total size >= ir*kc + kc*mr
//   - packedB: Packed B buffer, total size >= jr*kc + kc*nr
//   - c: Output C matrix (row-major), size >= m*n
//   - n: Leading dimension of C (total columns)
//   - ir: Starting row micro-panel index in A
//   - jr: Starting column micro-panel index in B
//   - kc: K-blocking size
//   - mr: Micro-tile row dimension
//   - nr: Micro-tile column dimension
func packedMicroKernelNEONF32(packedA []float32, packedB []float32, c []float32, n, ir, jr, kc, mr, nr int) {
	// packedA offset: ir-th panel, each panel has kc*mr elements
	aOffset := ir * kc * mr
	// packedB offset: jr-th panel, each panel has kc*nr elements
	bOffset := jr * kc * nr
	// C offset: row ir*mr, column jr*nr
	cOffset := ir*mr*n + jr*nr

	asm.PackedMicroKernelNEONF32(
		packedA[aOffset:],
		packedB[bOffset:],
		c[cOffset:],
		kc, n, mr, nr,
	)
}

// packedMicroKernelPartialNEONF32 handles edge micro-tiles with partial rows/columns.
func packedMicroKernelPartialNEONF32(packedA []float32, packedB []float32, c []float32, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	aOffset := ir * kc * mr
	bOffset := jr * kc * nr
	cOffset := ir*mr*n + jr*nr

	// The NEON kernel handles partial tiles internally via mr/nr parameters
	asm.PackedMicroKernelNEONF32(
		packedA[aOffset:],
		packedB[bOffset:],
		c[cOffset:],
		kc, n, activeRows, activeCols,
	)
}

// packedMicroKernelNEONF64 wraps the GOAT-generated NEON micro-kernel for float64.
func packedMicroKernelNEONF64(packedA []float64, packedB []float64, c []float64, n, ir, jr, kc, mr, nr int) {
	aOffset := ir * kc * mr
	bOffset := jr * kc * nr
	cOffset := ir*mr*n + jr*nr

	asm.PackedMicroKernelNEONF64(
		packedA[aOffset:],
		packedB[bOffset:],
		c[cOffset:],
		kc, n, mr, nr,
	)
}

func packedMicroKernelPartialNEONF64(packedA []float64, packedB []float64, c []float64, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	aOffset := ir * kc * mr
	bOffset := jr * kc * nr
	cOffset := ir*mr*n + jr*nr

	asm.PackedMicroKernelNEONF64(
		packedA[aOffset:],
		packedB[bOffset:],
		c[cOffset:],
		kc, n, activeRows, activeCols,
	)
}

// packedMicroKernelNEONF16 wraps the GOAT-generated NEON FP16 micro-kernel.
func packedMicroKernelNEONF16(packedA []hwy.Float16, packedB []hwy.Float16, c []hwy.Float16, n, ir, jr, kc, mr, nr int) {
	aOffset := ir * kc * mr
	bOffset := jr * kc * nr
	cOffset := ir*mr*n + jr*nr

	asm.PackedMicroKernelNEONF16(
		packedA[aOffset:],
		packedB[bOffset:],
		c[cOffset:],
		kc, n, mr, nr,
	)
}

func packedMicroKernelPartialNEONF16(packedA []hwy.Float16, packedB []hwy.Float16, c []hwy.Float16, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	aOffset := ir * kc * mr
	bOffset := jr * kc * nr
	cOffset := ir*mr*n + jr*nr

	asm.PackedMicroKernelNEONF16(
		packedA[aOffset:],
		packedB[bOffset:],
		c[cOffset:],
		kc, n, activeRows, activeCols,
	)
}

// packedMicroKernelNEONBF16 wraps the GOAT-generated NEON BF16 micro-kernel.
func packedMicroKernelNEONBF16(packedA []hwy.BFloat16, packedB []hwy.BFloat16, c []hwy.BFloat16, n, ir, jr, kc, mr, nr int) {
	aOffset := ir * kc * mr
	bOffset := jr * kc * nr
	cOffset := ir*mr*n + jr*nr

	asm.PackedMicroKernelNEONBF16(
		packedA[aOffset:],
		packedB[bOffset:],
		c[cOffset:],
		kc, n, mr, nr,
	)
}

func packedMicroKernelPartialNEONBF16(packedA []hwy.BFloat16, packedB []hwy.BFloat16, c []hwy.BFloat16, n, ir, jr, kc, mr, nr, activeRows, activeCols int) {
	aOffset := ir * kc * mr
	bOffset := jr * kc * nr
	cOffset := ir*mr*n + jr*nr

	asm.PackedMicroKernelNEONBF16(
		packedA[aOffset:],
		packedB[bOffset:],
		c[cOffset:],
		kc, n, activeRows, activeCols,
	)
}

func init() {
	// On ARM64 without SME, use NEON assembly micro-kernels for packed GEBP
	// This overrides the pure Go hwy implementation with optimized NEON
	if !hwy.HasSME() {
		// Float32
		PackedMicroKernelFloat32 = packedMicroKernelNEONF32
		PackedMicroKernelPartialFloat32 = packedMicroKernelPartialNEONF32

		// Float64
		PackedMicroKernelFloat64 = packedMicroKernelNEONF64
		PackedMicroKernelPartialFloat64 = packedMicroKernelPartialNEONF64
	}

	// F16: Requires ARMv8.2-A FP16 extension
	if hwy.HasARMFP16() && !hwy.HasSME() {
		PackedMicroKernelFloat16 = packedMicroKernelNEONF16
		PackedMicroKernelPartialFloat16 = packedMicroKernelPartialNEONF16
	}

	// BF16: Requires ARMv8.6-A BF16 extension
	if hwy.HasARMBF16() && !hwy.HasSME() {
		PackedMicroKernelBFloat16 = packedMicroKernelNEONBF16
		PackedMicroKernelPartialBFloat16 = packedMicroKernelPartialNEONBF16
	}
}
