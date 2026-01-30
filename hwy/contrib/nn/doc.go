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
// Linear (fully-connected) layer operations:
//   - Linear - SIMD dot-product based linear layer (hwygen dispatch)
//   - LinearAuto - Composition-based linear using best available matmul
//   - LinearActivationAuto - Linear + fused activation (GELU, ReLU, SiLU, Tanh)
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
//	    nn.LinearActivationAuto(x, w1, b1, hidden, batch, dim, ffnDim, nn.ActivationGelu)
//	    output := make([]float32, batch*dim)
//	    nn.LinearAuto(hidden, w2, b2, output, batch, ffnDim, dim)
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
