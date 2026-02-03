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

package matmul

// =============================================================================
// Dispatch Variables for Fused NF4/Int4 MatMul + Activation
// =============================================================================
// These are the dispatch variables that platform-specific init() functions
// can override with optimized implementations (e.g., SME on ARM64).

// FusedNF4MatMulSiLU performs fused NF4 dequantization + matmul + SiLU activation.
// Dispatches to the best available implementation for the current platform.
var FusedNF4MatMulSiLU func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// FusedNF4MatMulGELU performs fused NF4 dequantization + matmul + GELU activation.
// Dispatches to the best available implementation for the current platform.
var FusedNF4MatMulGELU func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// FusedNF4MatMulGELUApprox performs fused NF4 dequantization + matmul + approximate GELU.
var FusedNF4MatMulGELUApprox func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// FusedNF4MatMulReLU performs fused NF4 dequantization + matmul + ReLU activation.
var FusedNF4MatMulReLU func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// FusedInt4MatMulSiLU performs fused Int4 dequantization + matmul + SiLU activation.
var FusedInt4MatMulSiLU func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// FusedInt4MatMulGELU performs fused Int4 dequantization + matmul + GELU activation.
var FusedInt4MatMulGELU func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// FusedInt4MatMulGELUApprox performs fused Int4 dequantization + matmul + approximate GELU.
var FusedInt4MatMulGELUApprox func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// FusedInt4MatMulReLU performs fused Int4 dequantization + matmul + ReLU activation.
var FusedInt4MatMulReLU func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// =============================================================================
// Parallel Dispatch Variables
// =============================================================================

// ParallelFusedNF4MatMulSiLU performs parallel fused NF4 + SiLU for large matrices.
var ParallelFusedNF4MatMulSiLU func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// ParallelFusedNF4MatMulGELU performs parallel fused NF4 + GELU for large matrices.
var ParallelFusedNF4MatMulGELU func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// ParallelFusedInt4MatMulSiLU performs parallel fused Int4 + SiLU for large matrices.
var ParallelFusedInt4MatMulSiLU func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// ParallelFusedInt4MatMulGELU performs parallel fused Int4 + GELU for large matrices.
var ParallelFusedInt4MatMulGELU func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// =============================================================================
// Default Initialization
// =============================================================================

func init() {
	// Set defaults to base implementations.
	// Platform-specific init() functions (like matmul_fused_nf4_act_sme.go)
	// will override these with optimized versions when available.

	// NF4 + activation
	FusedNF4MatMulSiLU = BaseFusedNF4MatMulSiLU
	FusedNF4MatMulGELU = BaseFusedNF4MatMulGELU
	FusedNF4MatMulGELUApprox = BaseFusedNF4MatMulGELUApprox
	FusedNF4MatMulReLU = BaseFusedNF4MatMulReLU

	// Int4 + activation
	FusedInt4MatMulSiLU = BaseFusedInt4MatMulSiLU
	FusedInt4MatMulGELU = BaseFusedInt4MatMulGELU
	FusedInt4MatMulGELUApprox = BaseFusedInt4MatMulGELUApprox
	FusedInt4MatMulReLU = BaseFusedInt4MatMulReLU

	// Parallel variants default to serial implementations
	ParallelFusedNF4MatMulSiLU = func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
		FusedNF4MatMulSiLU(input, packed, scales, output, M, K, N, groupSize)
	}
	ParallelFusedNF4MatMulGELU = func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
		FusedNF4MatMulGELU(input, packed, scales, output, M, K, N, groupSize)
	}
	ParallelFusedInt4MatMulSiLU = func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
		FusedInt4MatMulSiLU(input, packed, scales, output, M, K, N, groupSize)
	}
	ParallelFusedInt4MatMulGELU = func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
		FusedInt4MatMulGELU(input, packed, scales, output, M, K, N, groupSize)
	}
}
