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
// This package implements the CDF 5/3 (reversible) and CDF 9/7 (irreversible)
// biorthogonal wavelets used in JPEG 2000. All transforms use the lifting scheme
// for efficient computation.
//
// # Wavelet Types
//
// CDF 5/3 (Le Gall 5/3):
//   - Reversible (lossless) transform for integer data
//   - Two lifting steps: predict and update
//   - Used in JPEG 2000 Part-1 lossless mode
//
// CDF 9/7 (Cohen-Daubechies-Feauveau 9/7):
//   - Irreversible transform for floating-point data
//   - Four lifting steps with normalization
//   - Used in JPEG 2000 Part-1 lossy mode
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
// Low-level transforms operate on slices:
//
//	Synthesize53(data, phase)    // inverse 5/3 transform
//	Analyze53(data, phase)       // forward 5/3 transform
//	Synthesize97(data, phase)    // inverse 9/7 transform
//	Analyze97(data, phase)       // forward 9/7 transform
//
// Data layout:
//   - Analysis (forward): interleaved samples → [low-pass | high-pass]
//   - Synthesis (inverse): [low-pass | high-pass] → interleaved samples
//
// # 2D Transform Functions
//
// Higher-level transforms operate on Image[T] types:
//
//	Synthesize2D_53(img, levels, phaseFn)
//	Analyze2D_53(img, levels, phaseFn)
//	Synthesize2D_97(img, levels, phaseFn)
//	Analyze2D_97(img, levels, phaseFn)
//
// The phaseFn callback returns (phaseH, phaseV) for each decomposition level,
// allowing proper handling of tile boundaries.
//
// # Usage Example
//
//	// 1D inverse transform
//	data := []int32{10, 20, 5, -3, 15, 8}  // [low | high] format
//	wavelet.Synthesize53(data, 0)
//	// data now contains reconstructed samples
//
//	// 2D multi-level decomposition
//	img := image.NewImage[float32](512, 512)
//	phaseFn := func(level int) (int, int) { return 0, 0 }
//	wavelet.Analyze2D_97(img, 3, phaseFn)  // 3-level decomposition
//
// # Coefficient Normalization
//
// The 9/7 transform uses standard K normalization (not JPEG 2000's 2/K).
// When integrating with JPEG 2000 codecs, apply the appropriate scaling
// factor to high-pass subbands after inverse transform.
package wavelet
