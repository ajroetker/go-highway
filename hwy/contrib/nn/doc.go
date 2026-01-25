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
//
// Future operations (planned):
//   - LayerNorm - Layer normalization
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
// # Build Requirements
//
// The SIMD implementations require:
//   - GOEXPERIMENT=simd build flag
//   - AMD64 architecture with AVX2/AVX-512, or ARM64 with NEON
//
// On non-SIMD builds, the functions fall back to scalar implementations.
package nn
