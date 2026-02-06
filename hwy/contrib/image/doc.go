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

// Package image provides SIMD-friendly 2D image types and operations.
//
// The core types are Image[T] for single-channel images and Image3[T] for
// three-plane images (e.g., RGB or YUV). Rows are aligned to SIMD vector width
// for efficient vectorized processing.
//
// # Point Operations
//
// Point operations transform each pixel independently:
//
//	BrightnessContrast(img, out, scale, offset) // out = img * scale + offset
//	ClampImage(img, out, minVal, maxVal)        // clamp to range
//	Threshold(img, out, thresh, below, above)   // binary threshold
//
// # Usage Example
//
//	// Create a 1080p image
//	img := image.NewImage[float32](1920, 1080)
//
//	// Apply brightness/contrast adjustment
//	out := image.NewImage[float32](1920, 1080)
//	image.BrightnessContrast(img, out, 1.5, 0.1)
//
// # Color Transforms
//
// JPEG 2000 color space transformations for RGB ↔ YCbCr conversion:
//
// Reversible Color Transform (RCT) - lossless integer arithmetic:
//
//	ForwardRCT(r, g, b, outY, outCb, outCr) // RGB → YCbCr
//	InverseRCT(y, cb, cr, outR, outG, outB) // YCbCr → RGB
//
// Irreversible Color Transform (ICT) - floating-point with ITU-T T.800 coefficients:
//
//	ForwardICT(r, g, b, outY, outCb, outCr) // RGB → YCbCr
//	InverseICT(y, cb, cr, outR, outG, outB) // YCbCr → RGB
//
// # Edge Handling
//
// Coordinate helper functions for handling out-of-bounds pixel access:
//
//	Mirror(index, size) - reflect at boundaries
//	Clamp(index, size)  - repeat edge pixels
//	Wrap(index, size)   - tile/wrap around
package image
