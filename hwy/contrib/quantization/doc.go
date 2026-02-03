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

// Package quantization documents the quantization formats supported by go-highway.
//
// Supported formats:
//   - NF4 (4-bit NormalFloat): Used in QLoRA for efficient LLM fine-tuning
//   - Int4 (4-bit signed integer): Symmetric quantization with range [-8, 7]
//   - Int8 (8-bit signed integer): Standard quantization with range [-128, 127]
//
// All formats use per-group scaling for improved accuracy. The groupSize
// parameter controls how many weights share a single scale factor.
//
// # Fused Dequantization + MatMul
//
// For optimal performance, use the fused kernels in the matmul package which
// dequantize on-the-fly without materializing the full weight matrix:
//
//	import "github.com/ajroetker/go-highway/hwy/contrib/matmul"
//
//	// Fused NF4 dequant + matmul
//	matmul.FusedNF4MatMul(input, packedWeights, scales, output, M, K, N, groupSize)
//
//	// Fused Int4 dequant + matmul
//	matmul.FusedInt4MatMul(input, packedWeights, scales, output, M, K, N, groupSize)
//
//	// Fused Int8 dequant + matmul
//	matmul.FusedInt8MatMul(input, weights, scales, output, M, K, N, groupSize)
//
// # Fused Dequantization + MatMul + Activation
//
// For MLP layers that immediately apply an activation function after matmul,
// use the fused activation variants to avoid an extra memory round-trip:
//
//	// NF4 with activation functions
//	matmul.BaseFusedNF4MatMulSiLU(input, packed, scales, output, M, K, N, groupSize)
//	matmul.BaseFusedNF4MatMulGELU(input, packed, scales, output, M, K, N, groupSize)
//	matmul.BaseFusedNF4MatMulGELUApprox(input, packed, scales, output, M, K, N, groupSize)
//	matmul.BaseFusedNF4MatMulReLU(input, packed, scales, output, M, K, N, groupSize)
//	matmul.BaseFusedNF4MatMulSwiGLU(input, gatePacked, gateScales, upPacked, upScales, output, M, K, N, groupSize)
//
//	// Int4 with activation functions
//	matmul.BaseFusedInt4MatMulSiLU(input, packed, scales, output, M, K, N, groupSize)
//	matmul.BaseFusedInt4MatMulGELU(input, packed, scales, output, M, K, N, groupSize)
//	matmul.BaseFusedInt4MatMulGELUApprox(input, packed, scales, output, M, K, N, groupSize)
//	matmul.BaseFusedInt4MatMulReLU(input, packed, scales, output, M, K, N, groupSize)
//	matmul.BaseFusedInt4MatMulSwiGLU(input, gatePacked, gateScales, upPacked, upScales, output, M, K, N, groupSize)
//
// SwiGLU is commonly used in LLaMA, Mistral, and other modern architectures.
// It computes SiLU(gate_projection) * up_projection in a single fused pass.
//
// These fused operations avoid an O(K*N) memory allocation for the dequantized
// weights, which is critical for large language models.
package quantization
