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

package nn

import "math"

// QuantizeAffine performs per-tensor affine quantization from float32 to uint8.
//
// The mapping is:
//
//	float_val ≈ scale * (uint8_val - zp)
//
// Returns:
//   - scale: the quantization scale factor
//   - zp: the uint8 zero point
//
// The output slice must be pre-allocated with at least size elements.
func QuantizeAffine(input []float32, output []uint8, size int) (scale float32, zp uint8) {
	if size == 0 {
		return 0, 0
	}

	// Find min/max of input.
	minVal := input[0]
	maxVal := input[0]
	for i := 1; i < size; i++ {
		if input[i] < minVal {
			minVal = input[i]
		}
		if input[i] > maxVal {
			maxVal = input[i]
		}
	}

	// Handle constant input.
	if minVal == maxVal {
		zp = 0
		if minVal == 0 {
			scale = 1.0 // arbitrary, all outputs are 0
		} else {
			scale = minVal / -float32(zp) // won't be used meaningfully
			scale = 1.0
		}
		for i := range size {
			output[i] = 0
		}
		return scale, zp
	}

	// Compute scale and zero point.
	scale = (maxVal - minVal) / 255.0
	invScale := 1.0 / scale
	zpFloat := -minVal * invScale
	zpClamped := math.Round(float64(zpFloat))
	if zpClamped < 0 {
		zpClamped = 0
	} else if zpClamped > 255 {
		zpClamped = 255
	}
	zp = uint8(zpClamped)

	// Quantize.
	for i := range size {
		v := math.Round(float64(input[i]*invScale + float32(zp)))
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		output[i] = uint8(v)
	}

	return scale, zp
}

// DequantizeInt32ToFloat32 converts int32 accumulator values to float32
// using a combined scale factor.
//
//	output[i] = combinedScale * float32(input[i])
//
// The combinedScale is typically the product of the quantization scales
// of the two matrices that were multiplied (e.g., qScale * kScale * attentionScale).
func DequantizeInt32ToFloat32(input []int32, output []float32, size int, combinedScale float32) {
	for i := range size {
		output[i] = combinedScale * float32(input[i])
	}
}
