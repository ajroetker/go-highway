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

// Package activation provides SIMD-accelerated neural network activation functions.
// This package corresponds to common activation functions used in deep learning.
//
// # Supported Activations
//
// Element-wise activation functions:
//   - GELU - Gaussian Error Linear Unit: x * 0.5 * (1 + erf(x / sqrt(2)))
//   - GELUApprox - Fast GELU approximation: x * sigmoid(1.702 * x)
//   - ReLU - Rectified Linear Unit: max(0, x)
//   - SiLU/Swish - Sigmoid Linear Unit: x * sigmoid(x)
//
// # Example Usage
//
//	import "github.com/ajroetker/go-highway/hwy/contrib/activation"
//
//	func ApplyGELU(input []float32) []float32 {
//	    output := make([]float32, len(input))
//	    activation.GELU(input, output)
//	    return output
//	}
//
//	func ApplyReLU(input []float32) []float32 {
//	    output := make([]float32, len(input))
//	    activation.ReLU(input, output)
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
package activation
