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

// NEON Fused NF4/Int4 + Activation Matrix Multiplication for ARM64
// Uses NEON SIMD instructions for fused dequantization + matmul + activation.
package asm

import "unsafe"

// NEON FMA for ARM64 with SiLU activation
// Note: Use --target-os linux on macOS to work around objdump format issues
//go:generate go tool goat ../c/matmul_fused_nf4_silu_neon_arm64.c -O3 --target arm64 --target-os linux
//go:generate go tool goat ../c/matmul_fused_nf4_gelu_neon_arm64.c -O3 --target arm64 --target-os linux

// ============================================================================
// NEON Fused NF4 + SiLU Matrix Multiplication
// ============================================================================

// FusedNF4SiLUMatMulNEON performs fused NF4 dequant + matmul + SiLU using NEON.
// output[m,n] = SiLU(sum_k(input[m,k] * dequant(packed[k,n])))
//
// Parameters:
//   - input: [M, K] float32 input matrix (row-major)
//   - packed: [K, N/2] uint8 packed NF4 weights (2 values per byte, low nibble first)
//   - scales: [K, numGroups] float32 per-group scales
//   - output: [M, N] float32 output matrix (row-major, pre-allocated)
//   - m, k, n: matrix dimensions
//   - groupSize: number of columns per scale group
//
// N must be a multiple of 4 (NEON f32 vector width).
func FusedNF4SiLUMatMulNEON(input []float32, packed []uint8, scales []float32, output []float32, m, k, n, groupSize int) {
	if m == 0 || k == 0 || n == 0 || n%4 != 0 {
		return
	}
	packedSize := (k * n + 1) / 2
	numGroups := (n + groupSize - 1) / groupSize
	if len(input) < m*k || len(packed) < packedSize || len(scales) < k*numGroups || len(output) < m*n {
		return
	}
	mVal := int64(m)
	kVal := int64(k)
	nVal := int64(n)
	groupSizeVal := int64(groupSize)
	numGroupsVal := int64(numGroups)
	fused_nf4_silu_matmul_neon(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&packed[0]),
		unsafe.Pointer(&scales[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&groupSizeVal),
		unsafe.Pointer(&numGroupsVal),
	)
}

// FusedInt4SiLUMatMulNEON performs fused Int4 dequant + matmul + SiLU using NEON.
// output[m,n] = SiLU(sum_k(input[m,k] * dequant(packed[k,n])))
//
// Int4 uses symmetric quantization: values in [0,15] map to [-8,7].
//
// N must be a multiple of 4 (NEON f32 vector width).
func FusedInt4SiLUMatMulNEON(input []float32, packed []uint8, scales []float32, output []float32, m, k, n, groupSize int) {
	if m == 0 || k == 0 || n == 0 || n%4 != 0 {
		return
	}
	packedSize := (k * n + 1) / 2
	numGroups := (n + groupSize - 1) / groupSize
	if len(input) < m*k || len(packed) < packedSize || len(scales) < k*numGroups || len(output) < m*n {
		return
	}
	mVal := int64(m)
	kVal := int64(k)
	nVal := int64(n)
	groupSizeVal := int64(groupSize)
	numGroupsVal := int64(numGroups)
	fused_int4_silu_matmul_neon(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&packed[0]),
		unsafe.Pointer(&scales[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&groupSizeVal),
		unsafe.Pointer(&numGroupsVal),
	)
}

// ============================================================================
// NEON Fused NF4 + GELU Matrix Multiplication
// ============================================================================

// FusedNF4GELUMatMulNEON performs fused NF4 dequant + matmul + GELU using NEON.
// output[m,n] = GELU(sum_k(input[m,k] * dequant(packed[k,n])))
//
// N must be a multiple of 4 (NEON f32 vector width).
func FusedNF4GELUMatMulNEON(input []float32, packed []uint8, scales []float32, output []float32, m, k, n, groupSize int) {
	if m == 0 || k == 0 || n == 0 || n%4 != 0 {
		return
	}
	packedSize := (k * n + 1) / 2
	numGroups := (n + groupSize - 1) / groupSize
	if len(input) < m*k || len(packed) < packedSize || len(scales) < k*numGroups || len(output) < m*n {
		return
	}
	mVal := int64(m)
	kVal := int64(k)
	nVal := int64(n)
	groupSizeVal := int64(groupSize)
	numGroupsVal := int64(numGroups)
	fused_nf4_gelu_matmul_neon(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&packed[0]),
		unsafe.Pointer(&scales[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&groupSizeVal),
		unsafe.Pointer(&numGroupsVal),
	)
}

// FusedInt4GELUMatMulNEON performs fused Int4 dequant + matmul + GELU using NEON.
// output[m,n] = GELU(sum_k(input[m,k] * dequant(packed[k,n])))
//
// N must be a multiple of 4 (NEON f32 vector width).
func FusedInt4GELUMatMulNEON(input []float32, packed []uint8, scales []float32, output []float32, m, k, n, groupSize int) {
	if m == 0 || k == 0 || n == 0 || n%4 != 0 {
		return
	}
	packedSize := (k * n + 1) / 2
	numGroups := (n + groupSize - 1) / groupSize
	if len(input) < m*k || len(packed) < packedSize || len(scales) < k*numGroups || len(output) < m*n {
		return
	}
	mVal := int64(m)
	kVal := int64(k)
	nVal := int64(n)
	groupSizeVal := int64(groupSize)
	numGroupsVal := int64(numGroups)
	fused_int4_gelu_matmul_neon(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&packed[0]),
		unsafe.Pointer(&scales[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&groupSizeVal),
		unsafe.Pointer(&numGroupsVal),
	)
}

// Assembly function declarations are generated by GoAT
