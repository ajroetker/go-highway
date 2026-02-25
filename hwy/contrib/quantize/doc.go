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

// Package quantize provides SIMD-accelerated uint8/float32 quantization and dequantization.
//
// The uint8 → float32 promotion chain is architecture-specific: AVX2 uses
// _mm256_cvtepu8_epi32 + _mm256_cvtepi32_ps, NEON uses vmovl widening + vcvtq.
// This package uses hwygen to generate optimized implementations for each target.
//
// # Core Functions
//
//   - DequantizeUint8(input []uint8, output []float32, min, scale float32)
//   - QuantizeFloat32(input []float32, output []uint8, min, scale float32)
//
// # Dequantization
//
// Converts quantized uint8 values back to float32:
//
//	output[i] = min + float32(input[i]) * scale
//
// # Quantization
//
// Converts float32 values to quantized uint8:
//
//	output[i] = uint8(round(clamp((input[i] - min) / scale, 0, 255)))
//
// # Example Usage
//
//	import "github.com/ajroetker/go-highway/hwy/contrib/quantize"
//
//	// Dequantize uint8 embeddings to float32
//	raw := []uint8{0, 128, 255, 64}
//	floats := make([]float32, len(raw))
//	quantize.DequantizeUint8(raw, floats, -1.0, 2.0/255.0)
//
//	// Quantize float32 back to uint8
//	packed := make([]uint8, len(floats))
//	quantize.QuantizeFloat32(floats, packed, -1.0, 2.0/255.0)
//
// # Build Requirements
//
// The SIMD implementations require:
//   - GOEXPERIMENT=simd build flag
//   - AMD64 architecture with AVX2 or AVX-512 support, or ARM64 with NEON
package quantize
