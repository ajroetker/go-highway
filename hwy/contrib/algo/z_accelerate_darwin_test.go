// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build cgo && darwin

package algo

import (
	"fmt"
	"math"
	"testing"
)

// TestAccelerateExpTransform verifies the Accelerate vForce exp matches stdlib.
func TestAccelerateExpTransform(t *testing.T) {
	input := []float32{-10, -5, -1, 0, 0.5, 1, 2, 5, 10, -2, 0.1, 0.9, 3, 4, 6, 7}
	output := make([]float32, len(input))

	ExpTransform(input, output)

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		if !relClose32(output[i], expected, 1e-5) {
			t.Errorf("ExpTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestAccelerateExpTransformFloat64(t *testing.T) {
	input := []float64{-10, -5, -1, 0, 0.5, 1, 2, 5, 10, -2, 0.1, 0.9, 3, 4, 6, 7}
	output := make([]float64, len(input))

	ExpTransform(input, output)

	for i := range input {
		expected := math.Exp(input[i])
		if math.Abs(output[i]-expected)/math.Max(math.Abs(expected), 1e-10) > 1e-10 {
			t.Errorf("ExpTransform64[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestAccelerateLogTransform(t *testing.T) {
	input := []float32{0.01, 0.1, 0.5, 1.0, 2.0, 2.718, 5.0, 10.0, 100.0, 0.25, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0}
	output := make([]float32, len(input))

	LogTransform(input, output)

	for i := range input {
		expected := float32(math.Log(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("LogTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestAccelerateSinTransform(t *testing.T) {
	input := []float32{0, 0.5, 1.0, 1.57, 2.0, 3.14, 4.0, 5.0, -0.5, -1.0, -1.57, -2.0, -3.14, -4.0, 6.0, 6.28}
	output := make([]float32, len(input))

	SinTransform(input, output)

	for i := range input {
		expected := float32(math.Sin(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("SinTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestAccelerateCosTransform(t *testing.T) {
	input := []float32{0, 0.5, 1.0, 1.57, 2.0, 3.14, 4.0, 5.0, -0.5, -1.0, -1.57, -2.0, -3.14, -4.0, 6.0, 6.28}
	output := make([]float32, len(input))

	CosTransform(input, output)

	for i := range input {
		expected := float32(math.Cos(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("CosTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestAccelerateTanhTransform(t *testing.T) {
	input := []float32{-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, -3, -0.1, 0.1, 3, -4, 4, 1.5}
	output := make([]float32, len(input))

	TanhTransform(input, output)

	for i := range input {
		expected := float32(math.Tanh(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("TanhTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestAccelerateSigmoidTransform(t *testing.T) {
	input := []float32{-10, -5, -2, -1, 0, 1, 2, 5, 10, -3, -0.5, 0.5, 3, -4, 4, 6}
	output := make([]float32, len(input))

	SigmoidTransform(input, output)

	for i := range input {
		expected := float32(1.0 / (1.0 + math.Exp(-float64(input[i]))))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("SigmoidTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestAccelerateErfTransform(t *testing.T) {
	input := []float32{-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, -1.5, -0.25, 0.25, 1.5, -2.5, 2.5, 0.75}
	output := make([]float32, len(input))

	ErfTransform(input, output)

	for i := range input {
		expected := float32(math.Erf(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("ErfTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

// TestAccelerateTailHandling verifies non-aligned sizes work.
func TestAccelerateTailHandling(t *testing.T) {
	sizes := []int{1, 3, 5, 7, 9, 15, 17}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("exp_size_%d", size), func(t *testing.T) {
			input := make([]float32, size)
			output := make([]float32, size)
			for i := range input {
				input[i] = float32(i) * 0.5
			}
			ExpTransform(input, output)
			for i := range input {
				expected := float32(math.Exp(float64(input[i])))
				if !relClose32(output[i], expected, 1e-5) {
					t.Errorf("ExpTransform[%d] size=%d: got %v, want %v", i, size, output[i], expected)
				}
			}
		})
	}
}

// Benchmarks

const accelerateBenchSize = 4096

func BenchmarkAccelerateExpTransform(b *testing.B) {
	for _, size := range []int{64, 256, 1024, 4096} {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i) * 0.01
		}
		b.Run(fmt.Sprintf("f32/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				ExpTransform(input, output)
			}
		})
	}
}

func BenchmarkAccelerateLogTransform(b *testing.B) {
	input := make([]float32, accelerateBenchSize)
	output := make([]float32, accelerateBenchSize)
	for i := range input {
		input[i] = float32(i+1) * 0.01
	}
	b.ReportAllocs()
	for b.Loop() {
		LogTransform(input, output)
	}
}

func BenchmarkAccelerateTanhTransform(b *testing.B) {
	input := make([]float32, accelerateBenchSize)
	output := make([]float32, accelerateBenchSize)
	for i := range input {
		input[i] = float32(i-accelerateBenchSize/2) * 0.01
	}
	b.ReportAllocs()
	for b.Loop() {
		TanhTransform(input, output)
	}
}

func BenchmarkAccelerateSigmoidTransform(b *testing.B) {
	input := make([]float32, accelerateBenchSize)
	output := make([]float32, accelerateBenchSize)
	for i := range input {
		input[i] = float32(i-accelerateBenchSize/2) * 0.01
	}
	b.ReportAllocs()
	for b.Loop() {
		SigmoidTransform(input, output)
	}
}

func BenchmarkAccelerateErfTransform(b *testing.B) {
	input := make([]float32, accelerateBenchSize)
	output := make([]float32, accelerateBenchSize)
	for i := range input {
		input[i] = float32(i-accelerateBenchSize/2) * 0.01
	}
	b.ReportAllocs()
	for b.Loop() {
		ErfTransform(input, output)
	}
}
