//go:build (amd64 && goexperiment.simd) || arm64

package algo

import (
	"fmt"
	"math"
	"testing"
)

const benchSize = 1024

// Test correctness using input ranges that match the existing accuracy tests.
// The SIMD implementations have been validated for these ranges.

func TestExpTransform(t *testing.T) {
	// Use values from TestExp32_Accuracy that are known to work
	input := []float32{-10, -1, 0, 0.5, 1, 2, 5, 10, -5, -2, 0.1, 0.9, 3, 4, 6, 7}
	output := make([]float32, len(input))

	ExpTransform(input, output)

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("ExpTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestLogTransform(t *testing.T) {
	// Use positive values from reasonable range
	input := []float32{0.1, 0.5, 1.0, 2.0, 2.718, 5.0, 10.0, 100.0, 0.01, 0.25, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0}
	output := make([]float32, len(input))

	LogTransform(input, output)

	for i := range input {
		expected := float32(math.Log(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("LogTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestSinTransform(t *testing.T) {
	// Use values in [-2π, 2π] range
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

func TestCosTransform(t *testing.T) {
	// Use values in [-2π, 2π] range
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

func TestTanhTransform(t *testing.T) {
	// Use values in [-5, 5] range (tanh saturates beyond this)
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

func TestSigmoidTransform(t *testing.T) {
	// Use values in [-10, 10] range
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

func TestErfTransform(t *testing.T) {
	// Use values in [-3, 3] range (erf saturates beyond this)
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

// TestExpTransform_TailHandling tests that non-vector-aligned sizes work correctly
func TestExpTransform_TailHandling(t *testing.T) {
	// Test sizes that don't align with vector width (8 for AVX2)
	sizes := []int{1, 3, 5, 7, 9, 15, 17}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			input := make([]float32, size)
			output := make([]float32, size)

			for i := range input {
				input[i] = float32(i) * 0.5
			}

			ExpTransform(input, output)

			for i := range input {
				expected := float32(math.Exp(float64(input[i])))
				if !closeEnough32(output[i], expected, 1e-3) {
					t.Errorf("ExpTransform[%d] size=%d input=%v: got %v, want %v", i, size, input[i], output[i], expected)
				}
			}
		})
	}
}

func closeEnough32(a, b, tol float32) bool {
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return true
	}
	if math.IsInf(float64(a), 0) && math.IsInf(float64(b), 0) {
		return (a > 0) == (b > 0)
	}
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= tol
}

// Benchmarks - Transform API (zero allocation)

func BenchmarkExpTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		ExpTransform(input, output)
	}
}

func BenchmarkLogTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i+1) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		LogTransform(input, output)
	}
}

func BenchmarkSinTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		SinTransform(input, output)
	}
}

func BenchmarkCosTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		CosTransform(input, output)
	}
}

func BenchmarkTanhTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		TanhTransform(input, output)
	}
}

func BenchmarkSigmoidTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		SigmoidTransform(input, output)
	}
}

func BenchmarkErfTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		ErfTransform(input, output)
	}
}

// Benchmarks - Stdlib comparison

func BenchmarkExpTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		for j := range input {
			output[j] = float32(math.Exp(float64(input[j])))
		}
	}
}

func BenchmarkLogTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i+1) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		for j := range input {
			output[j] = float32(math.Log(float64(input[j])))
		}
	}
}

func BenchmarkSinTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		for j := range input {
			output[j] = float32(math.Sin(float64(input[j])))
		}
	}
}

func BenchmarkCosTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		for j := range input {
			output[j] = float32(math.Cos(float64(input[j])))
		}
	}
}

func BenchmarkTanhTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		for j := range input {
			output[j] = float32(math.Tanh(float64(input[j])))
		}
	}
}

func BenchmarkSigmoidTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		for j := range input {
			output[j] = float32(1.0 / (1.0 + math.Exp(-float64(input[j]))))
		}
	}
}

func BenchmarkErfTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		for j := range input {
			output[j] = float32(math.Erf(float64(input[j])))
		}
	}
}
