//go:build arm64 && !noasm

package asm

import (
	"math"
	"testing"
)

func TestAddF32(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	result := make([]float32, len(a))

	AddF32(a, b, result)

	for i := range a {
		expected := a[i] + b[i]
		if result[i] != expected {
			t.Errorf("AddF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestSubF32(t *testing.T) {
	a := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	b := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	result := make([]float32, len(a))

	SubF32(a, b, result)

	for i := range a {
		expected := a[i] - b[i]
		if result[i] != expected {
			t.Errorf("SubF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestMulF32(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float32{2, 2, 2, 2, 2, 2, 2, 2}
	result := make([]float32, len(a))

	MulF32(a, b, result)

	for i := range a {
		expected := a[i] * b[i]
		if result[i] != expected {
			t.Errorf("MulF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestDivF32(t *testing.T) {
	a := []float32{2, 4, 6, 8, 10, 12, 14, 16}
	b := []float32{2, 2, 2, 2, 2, 2, 2, 2}
	result := make([]float32, len(a))

	DivF32(a, b, result)

	for i := range a {
		expected := a[i] / b[i]
		if result[i] != expected {
			t.Errorf("DivF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestFmaF32(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float32{2, 2, 2, 2, 2, 2, 2, 2}
	c := []float32{10, 10, 10, 10, 10, 10, 10, 10}
	result := make([]float32, len(a))

	FmaF32(a, b, c, result)

	for i := range a {
		expected := a[i]*b[i] + c[i]
		if result[i] != expected {
			t.Errorf("FmaF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestMinF32(t *testing.T) {
	a := []float32{1, 5, 3, 7, 2, 8, 4, 6}
	b := []float32{2, 3, 4, 5, 6, 4, 5, 3}
	result := make([]float32, len(a))

	MinF32(a, b, result)

	for i := range a {
		expected := min(a[i], b[i])
		if result[i] != expected {
			t.Errorf("MinF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestMaxF32(t *testing.T) {
	a := []float32{1, 5, 3, 7, 2, 8, 4, 6}
	b := []float32{2, 3, 4, 5, 6, 4, 5, 3}
	result := make([]float32, len(a))

	MaxF32(a, b, result)

	for i := range a {
		expected := max(a[i], b[i])
		if result[i] != expected {
			t.Errorf("MaxF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestReduceSumF32(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	expected := float32(0)
	for _, v := range input {
		expected += v
	}

	result := ReduceSumF32(input)
	if result != expected {
		t.Errorf("ReduceSumF32: got %v, want %v", result, expected)
	}
}

func TestReduceMinF32(t *testing.T) {
	input := []float32{5, 2, 8, 1, 9, 3, 7, 4, 10, 6, 15, 11, 13, 12, 14, 0}
	expected := float32(0)

	result := ReduceMinF32(input)
	if result != expected {
		t.Errorf("ReduceMinF32: got %v, want %v", result, expected)
	}
}

func TestReduceMaxF32(t *testing.T) {
	input := []float32{5, 2, 8, 1, 9, 3, 7, 4, 10, 6, 15, 11, 13, 12, 14, 0}
	expected := float32(15)

	result := ReduceMaxF32(input)
	if result != expected {
		t.Errorf("ReduceMaxF32: got %v, want %v", result, expected)
	}
}

func TestSqrtF32(t *testing.T) {
	input := []float32{1, 4, 9, 16, 25, 36, 49, 64}
	result := make([]float32, len(input))

	SqrtF32(input, result)

	for i := range input {
		expected := float32(math.Sqrt(float64(input[i])))
		if math.Abs(float64(result[i]-expected)) > 1e-6 {
			t.Errorf("SqrtF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestAbsF32(t *testing.T) {
	input := []float32{1, -2, 3, -4, 5, -6, 7, -8}
	result := make([]float32, len(input))

	AbsF32(input, result)

	for i := range input {
		expected := float32(math.Abs(float64(input[i])))
		if result[i] != expected {
			t.Errorf("AbsF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestNegF32(t *testing.T) {
	input := []float32{1, -2, 3, -4, 5, -6, 7, -8}
	result := make([]float32, len(input))

	NegF32(input, result)

	for i := range input {
		expected := -input[i]
		if result[i] != expected {
			t.Errorf("NegF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

// Test with non-aligned sizes to verify scalar fallback
func TestNonAlignedSizes(t *testing.T) {
	// Test with 7 elements (not a multiple of 4 or 16)
	a := []float32{1, 2, 3, 4, 5, 6, 7}
	b := []float32{1, 1, 1, 1, 1, 1, 1}
	result := make([]float32, len(a))

	AddF32(a, b, result)

	for i := range a {
		expected := a[i] + b[i]
		if result[i] != expected {
			t.Errorf("AddF32 (non-aligned)[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

// Benchmarks

func BenchmarkAddF32_NEON(b *testing.B) {
	n := 1024
	a := make([]float32, n)
	bb := make([]float32, n)
	result := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		bb[i] = float32(i)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		AddF32(a, bb, result)
	}
}

func BenchmarkMulF32_NEON(b *testing.B) {
	n := 1024
	a := make([]float32, n)
	bb := make([]float32, n)
	result := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		bb[i] = float32(i)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		MulF32(a, bb, result)
	}
}

func BenchmarkReduceSumF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(i)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = ReduceSumF32(input)
	}
}

func BenchmarkSqrtF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i + 1)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		SqrtF32(input, result)
	}
}

// Compare with scalar
func BenchmarkAddF32_Scalar(b *testing.B) {
	n := 1024
	a := make([]float32, n)
	bb := make([]float32, n)
	result := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		bb[i] = float32(i)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range a {
			result[j] = a[j] + bb[j]
		}
	}
}

func BenchmarkReduceSumF32_Scalar(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(i)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var sum float32
		for _, v := range input {
			sum += v
		}
		_ = sum
	}
}
