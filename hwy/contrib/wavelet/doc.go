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

// Package wavelet provides SIMD-accelerated wavelet transforms for image processing.
//
// This package implements the CDF 5/3 (reversible) biorthogonal wavelet used in
// JPEG 2000. All transforms use the lifting scheme for efficient computation.
//
// # Wavelet Types
//
// CDF 5/3 (Le Gall 5/3):
//   - Reversible (lossless) transform for integer data
//   - Two lifting steps: predict and update
//   - Used in JPEG 2000 Part-1 lossless mode
//
// # Phase Parameter
//
// The phase parameter controls how samples are partitioned into even/odd:
//
//   - phase=0: First sample is even (standard for most cases)
//   - phase=1: First sample is odd (used for odd-positioned tiles)
//
// In JPEG 2000, the phase depends on the tile/subband position relative to
// the image origin. Incorrect phase causes boundary artifacts.
//
// # 1D Transform Functions
//
// All functions require pre-allocated low and high buffers (each with
// capacity >= ceil(n/2)) to avoid per-call allocations:
//
//	Synthesize53(data, phase, low, high)       // inverse 5/3 transform
//	Analyze53(data, phase, low, high)          // forward 5/3 transform
//	Synthesize53Cols(colBuf, height, phase, lowBuf, highBuf) // column-batched inverse
//
// Data layout:
//   - Analysis (forward): interleaved samples → [low-pass | high-pass]
//   - Synthesis (inverse): [low-pass | high-pass] → interleaved samples
//
// # Usage Example
//
//	// 1D inverse transform
//	data := []int32{10, 20, 5, -3, 15, 8}  // [low | high] format
//	maxHalf := (len(data) + 1) / 2
//	low := make([]int32, maxHalf)
//	high := make([]int32, maxHalf)
//	wavelet.Synthesize53(data, 0, low, high)
//	// data now contains reconstructed samples
//
// # Coefficient Normalization
//
// The 9/7 lifting primitives (LiftStep97, ScaleSlice) use standard coefficients.
// When integrating with JPEG 2000 codecs, apply the appropriate scaling
// factor to high-pass subbands after inverse transform.
package wavelet
