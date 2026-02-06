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
	"fmt"
	stdmath "math"
	"testing"
)

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "simple",
			input: []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:  "negative",
			input: []float32{-1.0, -2.0, -3.0, -4.0},
		},
		{
			name:  "mixed",
			input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
		},
		{
			name:  "large values",
			input: []float32{100.0, 101.0, 102.0, 103.0},
		},
		{
			name:  "simd width",
			input: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:  "larger than simd",
			input: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			Softmax(tt.input, output)

			// Verify properties of softmax:
			// 1. All values between 0 and 1
			// 2. Sum equals 1
			var sum float32
			for i, v := range output {
				if v < 0 || v > 1 {
					t.Errorf("output[%d] = %v, want value in [0, 1]", i, v)
				}
				sum += v
			}

			if stdmath.Abs(float64(sum-1.0)) > 1e-5 {
				t.Errorf("sum of softmax = %v, want 1.0", sum)
			}

			// Verify relative ordering is preserved (larger input -> larger output)
			for i := 0; i < len(tt.input)-1; i++ {
				for j := i + 1; j < len(tt.input); j++ {
					if tt.input[i] > tt.input[j] && output[i] <= output[j] {
						t.Errorf("ordering not preserved: input[%d]=%v > input[%d]=%v but output[%d]=%v <= output[%d]=%v",
							i, tt.input[i], j, tt.input[j], i, output[i], j, output[j])
					}
				}
			}
		})
	}
}

func TestSoftmax64(t *testing.T) {
	input := []float64{1.0, 2.0, 3.0, 4.0}
	output := make([]float64, len(input))

	Softmax(input, output)

	var sum float64
	for _, v := range output {
		sum += v
	}

	if stdmath.Abs(sum-1.0) > 1e-10 {
		t.Errorf("sum of softmax = %v, want 1.0", sum)
	}
}

func TestLogSoftmax(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "simple",
			input: []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:  "negative",
			input: []float32{-1.0, -2.0, -3.0, -4.0},
		},
		{
			name:  "mixed",
			input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			LogSoftmax(tt.input, output)

			// Verify properties of log-softmax:
			// 1. All values <= 0 (log of probability)
			// 2. exp(log_softmax).sum() = 1
			for i, v := range output {
				if v > 0 {
					t.Errorf("output[%d] = %v, want value <= 0", i, v)
				}
			}

			// Check exp(log_softmax) sums to 1
			var sum float32
			for _, v := range output {
				sum += float32(stdmath.Exp(float64(v)))
			}
			if stdmath.Abs(float64(sum-1.0)) > 1e-5 {
				t.Errorf("sum of exp(log_softmax) = %v, want 1.0", sum)
			}
		})
	}
}

func TestSoftmaxWithTemperature(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0}

	// Test temperature = 1 (should be same as regular softmax)
	t.Run("temperature=1", func(t *testing.T) {
		output := make([]float32, len(input))
		expected := make([]float32, len(input))

		SoftmaxWithTemperature(input, output, 1.0)
		Softmax(input, expected)

		for i := range output {
			if stdmath.Abs(float64(output[i]-expected[i])) > 1e-6 {
				t.Errorf("output[%d] = %v, want %v", i, output[i], expected[i])
			}
		}
	})

	// Test low temperature (should be sharper)
	t.Run("temperature=0.5", func(t *testing.T) {
		output := make([]float32, len(input))
		SoftmaxWithTemperature(input, output, 0.5)

		// Lower temperature should make the max probability higher
		var sum float32
		var maxProb float32
		for _, v := range output {
			sum += v
			if v > maxProb {
				maxProb = v
			}
		}

		if stdmath.Abs(float64(sum-1.0)) > 1e-5 {
			t.Errorf("sum = %v, want 1.0", sum)
		}

		// Max prob should be higher than with T=1
		expected := make([]float32, len(input))
		Softmax(input, expected)
		var expectedMax float32
		for _, v := range expected {
			if v > expectedMax {
				expectedMax = v
			}
		}

		if maxProb <= expectedMax {
			t.Errorf("maxProb with T=0.5 (%v) should be > maxProb with T=1 (%v)", maxProb, expectedMax)
		}
	})

	// Test high temperature (should be softer)
	t.Run("temperature=2.0", func(t *testing.T) {
		output := make([]float32, len(input))
		SoftmaxWithTemperature(input, output, 2.0)

		var sum float32
		for _, v := range output {
			sum += v
		}

		if stdmath.Abs(float64(sum-1.0)) > 1e-5 {
			t.Errorf("sum = %v, want 1.0", sum)
		}
	})
}

func TestSoftmaxInPlace(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0}
	expected := make([]float32, len(input))
	copy(expected, input)
	Softmax(expected, expected)

	data := []float32{1.0, 2.0, 3.0, 4.0}
	SoftmaxInPlace(data)

	for i := range data {
		if stdmath.Abs(float64(data[i]-expected[i])) > 1e-6 {
			t.Errorf("data[%d] = %v, want %v", i, data[i], expected[i])
		}
	}
}

func TestSoftmaxScalarMatch(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	simdOutput := make([]float32, len(input))
	scalarOutput := make([]float32, len(input))

	Softmax(input, simdOutput)
	SoftmaxScalar(input, scalarOutput)

	for i := range simdOutput {
		if stdmath.Abs(float64(simdOutput[i]-scalarOutput[i])) > 1e-5 {
			t.Errorf("SIMD[%d] = %v, scalar[%d] = %v, mismatch", i, simdOutput[i], i, scalarOutput[i])
		}
	}
}

func BenchmarkSoftmax(b *testing.B) {
	sizes := []int{8, 64, 256, 1024}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i) * 0.1
		}

		b.Run(fmt.Sprintf("SIMD/%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Softmax(input, output)
			}
		})

		b.Run(fmt.Sprintf("Scalar/%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SoftmaxScalar(input, output)
			}
		})
	}
}

func BenchmarkLogSoftmax(b *testing.B) {
	sizes := []int{8, 64, 256, 1024}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i) * 0.1
		}

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				LogSoftmax(input, output)
			}
		})
	}
}

