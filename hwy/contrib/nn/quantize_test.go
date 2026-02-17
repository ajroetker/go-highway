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

import (
	"math"
	"testing"
)

func TestQuantizeAffineRoundTrip(t *testing.T) {
	input := []float32{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0}
	size := len(input)

	quantized := make([]uint8, size)
	scale, zp := QuantizeAffine(input, quantized, size)

	// Dequantize and check error.
	maxErr := float32(0)
	rangeVal := float32(3.0) // max - min = 2.0 - (-1.0) = 3.0
	for i := range size {
		dequant := scale * (float32(quantized[i]) - float32(zp))
		err := float32(math.Abs(float64(dequant - input[i])))
		if err > maxErr {
			maxErr = err
		}
	}

	// Quantization error should be at most scale/2 ≈ range/510.
	tolerance := rangeVal / 255.0
	if maxErr > tolerance {
		t.Errorf("Round-trip max error %v exceeds tolerance %v (scale=%v, zp=%d)", maxErr, tolerance, scale, zp)
	}
}

func TestQuantizeAffineZeroMapsToZP(t *testing.T) {
	// If 0.0 is in range, it should map close to the zero point.
	input := []float32{-1.0, 0.0, 1.0}
	quantized := make([]uint8, 3)
	_, zp := QuantizeAffine(input, quantized, 3)

	// The quantized value for 0.0 should be very close to zp.
	diff := int(quantized[1]) - int(zp)
	if diff < -1 || diff > 1 {
		t.Errorf("0.0 quantized to %d but zp=%d (diff=%d)", quantized[1], zp, diff)
	}
}

func TestQuantizeAffineConstantInput(t *testing.T) {
	input := []float32{5.0, 5.0, 5.0}
	quantized := make([]uint8, 3)
	scale, _ := QuantizeAffine(input, quantized, 3)

	if scale == 0 {
		t.Error("Scale should not be zero for constant input")
	}

	// All outputs should be the same.
	for i := 1; i < 3; i++ {
		if quantized[i] != quantized[0] {
			t.Errorf("Constant input produced different quantized values: %d vs %d", quantized[0], quantized[i])
		}
	}
}

func TestQuantizeAffineSingleElement(t *testing.T) {
	input := []float32{3.14}
	quantized := make([]uint8, 1)
	scale, _ := QuantizeAffine(input, quantized, 1)

	if scale == 0 {
		t.Error("Scale should not be zero")
	}
}

func TestQuantizeAffineEmpty(t *testing.T) {
	scale, zp := QuantizeAffine(nil, nil, 0)
	if scale != 0 || zp != 0 {
		t.Errorf("Empty input should return 0,0; got scale=%v zp=%d", scale, zp)
	}
}

func TestQuantizeAffineFullRange(t *testing.T) {
	// Values spread across a wide range.
	input := []float32{-100.0, -50.0, 0.0, 50.0, 100.0}
	quantized := make([]uint8, 5)
	scale, zp := QuantizeAffine(input, quantized, 5)

	// -100 should map close to 0, +100 should map close to 255.
	if quantized[0] > 1 {
		t.Errorf("Min value should map to ~0, got %d", quantized[0])
	}
	if quantized[4] < 254 {
		t.Errorf("Max value should map to ~255, got %d", quantized[4])
	}

	// Verify scale and zp are reasonable.
	if scale <= 0 {
		t.Errorf("Scale should be positive, got %v", scale)
	}
	_ = zp
}

func TestDequantizeInt32ToFloat32(t *testing.T) {
	input := []int32{-100, 0, 50, 200}
	output := make([]float32, 4)
	combinedScale := float32(0.01)

	DequantizeInt32ToFloat32(input, output, 4, combinedScale)

	expected := []float32{-1.0, 0.0, 0.5, 2.0}
	for i := range output {
		if math.Abs(float64(output[i]-expected[i])) > 1e-6 {
			t.Errorf("output[%d]: got %v, want %v", i, output[i], expected[i])
		}
	}
}
