// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build cgo && darwin

package activation

import (
	"fmt"
	stdmath "math"
	"testing"
)

func TestAccelerateGELU(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{name: "simple positive", input: []float32{0.0, 0.5, 1.0, 2.0}},
		{name: "simple negative", input: []float32{-2.0, -1.0, -0.5, 0.0}},
		{name: "mixed", input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0}},
		{name: "simd width", input: []float32{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5}},
		{name: "larger", input: []float32{-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			GELU(tt.input, output)

			for i, x := range tt.input {
				expected := float32(float64(x) * 0.5 * (1.0 + stdmath.Erf(float64(x)*0.7071067811865476)))
				if stdmath.Abs(float64(output[i]-expected)) > 1e-5 {
					t.Errorf("GELU(%v) = %v, want %v", x, output[i], expected)
				}
			}
		})
	}
}

func TestAccelerateGELU64(t *testing.T) {
	input := []float64{-2.0, -1.0, 0.0, 1.0, 2.0}
	output := make([]float64, len(input))
	GELU(input, output)

	for i, x := range input {
		expected := x * 0.5 * (1.0 + stdmath.Erf(x*0.7071067811865476))
		if stdmath.Abs(output[i]-expected) > 1e-6 {
			t.Errorf("GELU(%v) = %v, want %v", x, output[i], expected)
		}
	}
}

func TestAccelerateGELUApprox(t *testing.T) {
	input := []float32{-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5}
	output := make([]float32, len(input))
	GELUApprox(input, output)

	for i, x := range input {
		sigmoid := 1.0 / (1.0 + stdmath.Exp(-1.702*float64(x)))
		expected := float32(float64(x) * sigmoid)
		if stdmath.Abs(float64(output[i]-expected)) > 1e-5 {
			t.Errorf("GELUApprox(%v) = %v, want %v", x, output[i], expected)
		}
	}
}

func TestAccelerateTanh(t *testing.T) {
	input := []float32{-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, -3, -0.1, 0.1, 3, -4, 4, 1.5}
	output := make([]float32, len(input))
	Tanh(input, output)

	for i, x := range input {
		expected := float32(stdmath.Tanh(float64(x)))
		if stdmath.Abs(float64(output[i]-expected)) > 1e-5 {
			t.Errorf("Tanh(%v) = %v, want %v", x, output[i], expected)
		}
	}
}

func TestAccelerateSiLU(t *testing.T) {
	input := []float32{-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0}
	output := make([]float32, len(input))
	SiLU(input, output)

	for i, x := range input {
		sigmoid := 1.0 / (1.0 + stdmath.Exp(-float64(x)))
		expected := float32(float64(x) * sigmoid)
		if stdmath.Abs(float64(output[i]-expected)) > 1e-5 {
			t.Errorf("SiLU(%v) = %v, want %v", x, output[i], expected)
		}
	}
}

func TestAccelerateTailHandling(t *testing.T) {
	sizes := []int{1, 3, 5, 7, 9, 15, 17}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("gelu_size_%d", size), func(t *testing.T) {
			input := make([]float32, size)
			output := make([]float32, size)
			for i := range input {
				input[i] = float32(i-size/2) * 0.5
			}
			GELU(input, output)
			for i, x := range input {
				expected := float32(float64(x) * 0.5 * (1.0 + stdmath.Erf(float64(x)*0.7071067811865476)))
				if stdmath.Abs(float64(output[i]-expected)) > 1e-5 {
					t.Errorf("GELU[%d] size=%d: got %v, want %v", i, size, output[i], expected)
				}
			}
		})
	}
}

// Benchmarks

func BenchmarkAccelerateGELU(b *testing.B) {
	for _, size := range []int{8, 64, 256, 1024, 4096} {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i-size/2) * 0.1
		}
		b.Run(fmt.Sprintf("f32/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				GELU(input, output)
			}
		})
	}
}

func BenchmarkAccelerateGELUApprox(b *testing.B) {
	for _, size := range []int{8, 64, 256, 1024, 4096} {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i-size/2) * 0.1
		}
		b.Run(fmt.Sprintf("f32/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				GELUApprox(input, output)
			}
		})
	}
}

func BenchmarkAccelerateTanh(b *testing.B) {
	for _, size := range []int{8, 64, 256, 1024, 4096} {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i-size/2) * 0.01
		}
		b.Run(fmt.Sprintf("f32/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Tanh(input, output)
			}
		})
	}
}

func BenchmarkAccelerateSiLU(b *testing.B) {
	for _, size := range []int{8, 64, 256, 1024, 4096} {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i-size/2) * 0.1
		}
		b.Run(fmt.Sprintf("f32/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				SiLU(input, output)
			}
		})
	}
}
