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

// Package rabitq provides SIMD-accelerated primitives for RaBitQ vector quantization.
//
// RaBitQ is a 1-bit quantization algorithm for high-dimensional vectors that provides:
//   - Compact representation (1 bit per dimension)
//   - Theoretical error bounds for approximate nearest neighbor search
//   - Fast distance estimation via weighted popcount operations
//
// This package implements the low-level SIMD operations described in:
// "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
// for Approximate Nearest Neighbor Search" by Jianyang Gao & Cheng Long.
// URL: https://arxiv.org/pdf/2405.12497
//
// # Core Operations
//
// BitProduct computes the weighted bit product used in distance estimation:
//
//	result = 1*popcount(code & q1) + 2*popcount(code & q2) +
//	         4*popcount(code & q3) + 8*popcount(code & q4)
//
// This operation is the hot path in RaBitQ distance computation, as it must be
// performed for every candidate vector during search.
//
// QuantizeVectors performs bulk quantization of unit vectors into 1-bit codes:
//   - Extracts sign bits (1 for positive, 0 for negative)
//   - Packs bits into uint64 codes
//   - Computes dot products between unit vectors and their quantized form
//   - Counts the number of 1-bits in each code
//
// # Usage
//
// These primitives are designed to be used by higher-level quantizer implementations.
// A typical workflow:
//
//  1. Normalize vectors relative to a centroid
//  2. Call QuantizeVectors to create 1-bit codes and metadata
//  3. During search, call BitProduct to estimate distances
//
// Example:
//
//	// Quantize a batch of unit vectors
//	codes := make([]uint64, count*width)
//	dotProducts := make([]float32, count)
//	codeCounts := make([]uint32, count)
//	rabitq.QuantizeVectors(unitVectors, codes, dotProducts, codeCounts,
//	    sqrtDimsInv, count, dims, width)
//
//	// During search, compute bit product for distance estimation
//	bitProduct := rabitq.BitProduct(dataCode, queryQ1, queryQ2, queryQ3, queryQ4)
//
// # SIMD Acceleration
//
// Operations automatically use the best available SIMD instructions:
//   - AVX-512 VPOPCNT on x86_64 with AVX-512 BITALG
//   - AVX2 with byte-wise popcount on x86_64
//   - NEON with vcnt on ARM64
//   - SME on Apple M4+ processors (with adaptive dispatch)
//   - Pure Go fallback on other platforms
//
// # Build Requirements
//
// For SIMD acceleration on AMD64:
//
//	GOEXPERIMENT=simd go build
//
// ARM64 NEON is available without special build flags.
package rabitq
