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

package activation

import (
	"fmt"
	stdmath "math"
	"testing"
)

func TestGELU(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "simple positive",
			input: []float32{0.0, 0.5, 1.0, 2.0},
		},
		{
			name:  "simple negative",
			input: []float32{-2.0, -1.0, -0.5, 0.0},
		},
		{
			name:  "mixed",
			input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
		},
		{
			name:  "simd width",
			input: []float32{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5},
		},
		{
			name:  "larger than simd",
			input: []float32{-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			GELU(tt.input, output)

			// Verify against scalar reference
			for i, x := range tt.input {
				expected := float32(float64(x) * 0.5 * (1.0 + stdmath.Erf(float64(x)*0.7071067811865476)))
				if stdmath.Abs(float64(output[i]-expected)) > 1e-5 {
					t.Errorf("GELU(%v) = %v, want %v", x, output[i], expected)
				}
			}

			// Property: GELU(0) = 0
			for i, x := range tt.input {
				if x == 0 && output[i] != 0 {
					t.Errorf("GELU(0) = %v, want 0", output[i])
				}
			}

			// Property: GELU(x) < x for x < ~0.5, GELU(x) > x for large positive x
			// (This is an approximate property due to GELU's shape)
		})
	}
}

func TestGELUApprox(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "simple",
			input: []float32{0.0, 0.5, 1.0, 2.0},
		},
		{
			name:  "mixed",
			input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
		},
		{
			name:  "simd width",
			input: []float32{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			GELUApprox(tt.input, output)

			// Verify against scalar reference approximation
			for i, x := range tt.input {
				sigmoid := 1.0 / (1.0 + stdmath.Exp(-1.702*float64(x)))
				expected := float32(float64(x) * sigmoid)
				if stdmath.Abs(float64(output[i]-expected)) > 1e-5 {
					t.Errorf("GELUApprox(%v) = %v, want %v", x, output[i], expected)
				}
			}
		})
	}
}

func TestReLU(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "all positive",
			input: []float32{0.5, 1.0, 2.0, 3.0},
		},
		{
			name:  "all negative",
			input: []float32{-3.0, -2.0, -1.0, -0.5},
		},
		{
			name:  "mixed",
			input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
		},
		{
			name:  "simd width",
			input: []float32{-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0},
		},
		{
			name:  "larger than simd",
			input: []float32{-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			ReLU(tt.input, output)

			for i, x := range tt.input {
				var expected float32
				if x > 0 {
					expected = x
				} else {
					expected = 0
				}
				if output[i] != expected {
					t.Errorf("ReLU(%v) = %v, want %v", x, output[i], expected)
				}
			}
		})
	}
}

func TestSiLU(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "simple positive",
			input: []float32{0.0, 0.5, 1.0, 2.0},
		},
		{
			name:  "simple negative",
			input: []float32{-2.0, -1.0, -0.5, 0.0},
		},
		{
			name:  "mixed",
			input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
		},
		{
			name:  "simd width",
			input: []float32{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			SiLU(tt.input, output)

			// Verify against scalar reference
			for i, x := range tt.input {
				sigmoid := 1.0 / (1.0 + stdmath.Exp(-float64(x)))
				expected := float32(float64(x) * sigmoid)
				if stdmath.Abs(float64(output[i]-expected)) > 1e-5 {
					t.Errorf("SiLU(%v) = %v, want %v", x, output[i], expected)
				}
			}

			// Property: SiLU(0) = 0
			for i, x := range tt.input {
				if x == 0 && output[i] != 0 {
					t.Errorf("SiLU(0) = %v, want 0", output[i])
				}
			}
		})
	}
}

func TestLeakyReLU(t *testing.T) {
	var alpha float32 = 0.01
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "all positive",
			input: []float32{0.5, 1.0, 2.0, 3.0},
		},
		{
			name:  "all negative",
			input: []float32{-3.0, -2.0, -1.0, -0.5},
		},
		{
			name:  "mixed",
			input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			LeakyReLU(tt.input, output, alpha)

			for i, x := range tt.input {
				var expected float32
				if x > 0 {
					expected = x
				} else {
					expected = alpha * x
				}
				if stdmath.Abs(float64(output[i]-expected)) > 1e-6 {
					t.Errorf("LeakyReLU(%v) = %v, want %v", x, output[i], expected)
				}
			}
		})
	}
}

func TestELU(t *testing.T) {
	var alpha float32 = 1.0
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "all positive",
			input: []float32{0.5, 1.0, 2.0, 3.0},
		},
		{
			name:  "all negative",
			input: []float32{-3.0, -2.0, -1.0, -0.5},
		},
		{
			name:  "mixed",
			input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			ELU(tt.input, output, alpha)

			for i, x := range tt.input {
				var expected float32
				if x > 0 {
					expected = x
				} else {
					expected = float32(float64(alpha) * (stdmath.Exp(float64(x)) - 1.0))
				}
				if stdmath.Abs(float64(output[i]-expected)) > 1e-5 {
					t.Errorf("ELU(%v) = %v, want %v", x, output[i], expected)
				}
			}
		})
	}
}

func TestGELU64(t *testing.T) {
	input := []float64{-2.0, -1.0, 0.0, 1.0, 2.0}
	output := make([]float64, len(input))

	GELU(input, output)

	// Verify against scalar reference
	for i, x := range input {
		expected := x * 0.5 * (1.0 + stdmath.Erf(x*0.7071067811865476))
		if stdmath.Abs(output[i]-expected) > 1e-6 {
			t.Errorf("GELU(%v) = %v, want %v, diff=%v", x, output[i], expected, stdmath.Abs(output[i]-expected))
		}
	}
}

func BenchmarkGELU(b *testing.B) {
	sizes := []int{8, 64, 256, 1024}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i-size/2) * 0.1
		}

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				GELU(input, output)
			}
		})
	}
}

func BenchmarkGELUApprox(b *testing.B) {
	sizes := []int{8, 64, 256, 1024}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i-size/2) * 0.1
		}

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				GELUApprox(input, output)
			}
		})
	}
}

func BenchmarkReLU(b *testing.B) {
	sizes := []int{8, 64, 256, 1024}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i-size/2) * 0.1
		}

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ReLU(input, output)
			}
		})
	}
}

func BenchmarkSiLU(b *testing.B) {
	sizes := []int{8, 64, 256, 1024}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i-size/2) * 0.1
		}

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SiLU(input, output)
			}
		})
	}
}
