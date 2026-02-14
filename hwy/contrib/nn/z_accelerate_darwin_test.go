// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build cgo && darwin

package nn

import (
	"fmt"
	stdmath "math"
	"testing"
)

func TestAccelerateSoftmax(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{name: "simple", input: []float32{1.0, 2.0, 3.0, 4.0}},
		{name: "negative", input: []float32{-1.0, -2.0, -3.0, -4.0}},
		{name: "mixed", input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0}},
		{name: "large values", input: []float32{100.0, 101.0, 102.0, 103.0}},
		{name: "simd width", input: []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{name: "larger than simd", input: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			Softmax(tt.input, output)

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

			// Verify ordering preserved
			for i := 0; i < len(tt.input)-1; i++ {
				for j := i + 1; j < len(tt.input); j++ {
					if tt.input[i] > tt.input[j] && output[i] <= output[j] {
						t.Errorf("ordering not preserved at [%d]=%v > [%d]=%v", i, tt.input[i], j, tt.input[j])
					}
				}
			}
		})
	}
}

func TestAccelerateSoftmax64(t *testing.T) {
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

func TestAccelerateSoftmaxInPlace(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0}
	expected := make([]float32, len(input))
	Softmax(input, expected)

	data := []float32{1.0, 2.0, 3.0, 4.0}
	SoftmaxInPlace(data)

	for i := range data {
		if stdmath.Abs(float64(data[i]-expected[i])) > 1e-6 {
			t.Errorf("data[%d] = %v, want %v", i, data[i], expected[i])
		}
	}
}

func TestAccelerateLogSoftmax(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{name: "simple", input: []float32{1.0, 2.0, 3.0, 4.0}},
		{name: "negative", input: []float32{-1.0, -2.0, -3.0, -4.0}},
		{name: "mixed", input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			LogSoftmax(tt.input, output)

			for i, v := range output {
				if v > 0 {
					t.Errorf("output[%d] = %v, want value <= 0", i, v)
				}
			}

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

func TestAccelerateSoftmaxWithTemperature(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0}

	t.Run("temperature=1", func(t *testing.T) {
		output := make([]float32, len(input))
		expected := make([]float32, len(input))
		SoftmaxWithTemperature(input, output, 1.0)
		Softmax(input, expected)
		for i := range output {
			if stdmath.Abs(float64(output[i]-expected[i])) > 1e-5 {
				t.Errorf("output[%d] = %v, want %v", i, output[i], expected[i])
			}
		}
	})

	t.Run("temperature=0.5", func(t *testing.T) {
		output := make([]float32, len(input))
		SoftmaxWithTemperature(input, output, 0.5)
		var sum float32
		for _, v := range output {
			sum += v
		}
		if stdmath.Abs(float64(sum-1.0)) > 1e-5 {
			t.Errorf("sum = %v, want 1.0", sum)
		}
	})
}

// Benchmarks

func BenchmarkAccelerateSoftmax(b *testing.B) {
	for _, size := range []int{8, 64, 256, 1024, 4096} {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i) * 0.1
		}
		b.Run(fmt.Sprintf("f32/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Softmax(input, output)
			}
		})
	}
}

func BenchmarkAccelerateLogSoftmax(b *testing.B) {
	for _, size := range []int{8, 64, 256, 1024, 4096} {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i) * 0.1
		}
		b.Run(fmt.Sprintf("f32/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				LogSoftmax(input, output)
			}
		})
	}
}
