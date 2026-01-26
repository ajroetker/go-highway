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

// ParallelFusedNF4MatMul performs fused NF4 dequantization + matrix multiplication
// with parallel execution for large matrices.
// Dispatches to the best available implementation for the current platform.
// On platforms with SME, this uses tiled parallel execution.
// On other platforms, this falls back to the serial implementation.
var ParallelFusedNF4MatMul func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

// ParallelFusedInt4MatMul performs fused Int4 dequantization + matrix multiplication
// with parallel execution for large matrices.
// Dispatches to the best available implementation for the current platform.
// On platforms with SME, this uses tiled parallel execution.
// On other platforms, this falls back to the serial implementation.
var ParallelFusedInt4MatMul func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int)

func init() {
	// Default parallel implementations just call the serial versions.
	// SME-enabled platforms override these in matmul_fused_nf4_sme.go init().
	ParallelFusedNF4MatMul = func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
		FusedNF4MatMul(input, packed, scales, output, M, K, N, groupSize)
	}
	ParallelFusedInt4MatMul = func(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
		FusedInt4MatMul(input, packed, scales, output, M, K, N, groupSize)
	}
}
