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

// Package nn provides SIMD-accelerated neural network layer operations.
// This package corresponds to common operations in deep learning layers.
//
// # Supported Operations
//
// Normalization operations:
//   - Softmax - Softmax normalization over a slice
//   - LogSoftmax - Log of softmax (more numerically stable for NLL loss)
//   - LayerNorm - Layer normalization with optional affine transform
//
// Dense (fully-connected) layer operations:
//   - Dense - SIMD dot-product based dense layer (hwygen dispatch)
//   - DenseAuto - Composition-based dense using best available matmul
//   - DenseActivationAuto - Dense + fused activation (GELU, ReLU, SiLU, Tanh)
//
// Fused projection operations:
//   - QKVLinear - Fused QKV projection: x @ wQKV^T -> q, k, v with bias
//   - QKVLinearAuto - Composition-based QKV using MatMulKLastAuto + scatter + bias
//
// Attention operations:
//   - SDPA - Scaled Dot-Product Attention: softmax(Q@K^T * scale + mask) @ V
//   - SDPACausal - Causal variant with lower-triangular mask
//   - SDPAAuto / SDPACausalAuto - Auto-dispatched with internal scratch buffer
//   - MultiHeadSDPAAuto - Multi-head attention with GQA (grouped-query) support
//
// Future operations (planned):
//   - BatchNorm - Batch normalization
//   - RMSNorm - Root mean square normalization
//
// # Example Usage
//
//	import "github.com/ajroetker/go-highway/hwy/contrib/nn"
//
//	func ComputeSoftmax(logits []float32) []float32 {
//	    probs := make([]float32, len(logits))
//	    nn.Softmax(logits, probs)
//	    return probs
//	}
//
//	func TransformerFFN(x, w1, b1, w2, b2 []float32, batch, dim, ffnDim int) []float32 {
//	    hidden := make([]float32, batch*ffnDim)
//	    nn.DenseActivationAuto(x, w1, b1, hidden, batch, dim, ffnDim, nn.ActivationGelu)
//	    output := make([]float32, batch*dim)
//	    nn.DenseAuto(hidden, w2, b2, output, batch, ffnDim, dim)
//	    return output
//	}
//
//	func SelfAttention(q, k, v []float32, seqLen, headDim int) []float32 {
//	    scale := float32(1.0 / math.Sqrt(float64(headDim)))
//	    output := make([]float32, seqLen*headDim)
//	    nn.SDPACausalAuto(q, k, v, output, seqLen, seqLen, headDim, scale)
//	    return output
//	}
//
// # Build Requirements
//
// The SIMD implementations require:
//   - GOEXPERIMENT=simd build flag
//   - AMD64 architecture with AVX2/AVX-512, or ARM64 with NEON
//
// On non-SIMD builds, the functions fall back to scalar implementations.
package nn
