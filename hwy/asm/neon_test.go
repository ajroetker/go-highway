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

// Phase 5: Type Conversions Tests

func TestPromoteF32ToF64(t *testing.T) {
	input := []float32{1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5}
	result := make([]float64, len(input))

	PromoteF32ToF64(input, result)

	for i := range input {
		expected := float64(input[i])
		if result[i] != expected {
			t.Errorf("PromoteF32ToF64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestDemoteF64ToF32(t *testing.T) {
	input := []float64{1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5}
	result := make([]float32, len(input))

	DemoteF64ToF32(input, result)

	for i := range input {
		expected := float32(input[i])
		if result[i] != expected {
			t.Errorf("DemoteF64ToF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestConvertF32ToI32(t *testing.T) {
	input := []float32{1.9, -2.9, 3.1, -4.1, 5.5, -6.5, 7.0, -8.0}
	result := make([]int32, len(input))

	ConvertF32ToI32(input, result)

	for i := range input {
		expected := int32(input[i]) // truncation toward zero
		if result[i] != expected {
			t.Errorf("ConvertF32ToI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestConvertI32ToF32(t *testing.T) {
	input := []int32{1, -2, 3, -4, 5, -6, 7, -8}
	result := make([]float32, len(input))

	ConvertI32ToF32(input, result)

	for i := range input {
		expected := float32(input[i])
		if result[i] != expected {
			t.Errorf("ConvertI32ToF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestRoundF32(t *testing.T) {
	input := []float32{1.4, 1.5, 1.6, 2.5, -1.4, -1.5, -1.6, -2.5}
	result := make([]float32, len(input))

	RoundF32(input, result)

	// Note: NEON uses round-to-nearest-even for ties
	expected := []float32{1, 2, 2, 2, -1, -2, -2, -2}
	for i := range input {
		if result[i] != expected[i] {
			t.Errorf("RoundF32[%d]: got %v, want %v", i, result[i], expected[i])
		}
	}
}

func TestTruncF32(t *testing.T) {
	input := []float32{1.9, -1.9, 2.1, -2.1, 3.5, -3.5, 4.0, -4.0}
	result := make([]float32, len(input))

	TruncF32(input, result)

	for i := range input {
		expected := float32(math.Trunc(float64(input[i])))
		if result[i] != expected {
			t.Errorf("TruncF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestCeilF32(t *testing.T) {
	input := []float32{1.1, -1.1, 2.9, -2.9, 3.0, -3.0, 4.5, -4.5}
	result := make([]float32, len(input))

	CeilF32(input, result)

	for i := range input {
		expected := float32(math.Ceil(float64(input[i])))
		if result[i] != expected {
			t.Errorf("CeilF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestFloorF32(t *testing.T) {
	input := []float32{1.1, -1.1, 2.9, -2.9, 3.0, -3.0, 4.5, -4.5}
	result := make([]float32, len(input))

	FloorF32(input, result)

	for i := range input {
		expected := float32(math.Floor(float64(input[i])))
		if result[i] != expected {
			t.Errorf("FloorF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

// Phase 4: Memory Operations Tests

func TestGatherF32(t *testing.T) {
	base := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	indices := []int32{7, 0, 3, 5, 2, 1, 6, 4}
	result := make([]float32, len(indices))

	GatherF32(base, indices, result)

	for i := range indices {
		expected := base[indices[i]]
		if result[i] != expected {
			t.Errorf("GatherF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestGatherF64(t *testing.T) {
	base := []float64{10, 20, 30, 40, 50, 60, 70, 80}
	indices := []int32{7, 0, 3, 5, 2, 1, 6, 4}
	result := make([]float64, len(indices))

	GatherF64(base, indices, result)

	for i := range indices {
		expected := base[indices[i]]
		if result[i] != expected {
			t.Errorf("GatherF64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestGatherI32(t *testing.T) {
	base := []int32{10, 20, 30, 40, 50, 60, 70, 80}
	indices := []int32{7, 0, 3, 5, 2, 1, 6, 4}
	result := make([]int32, len(indices))

	GatherI32(base, indices, result)

	for i := range indices {
		expected := base[indices[i]]
		if result[i] != expected {
			t.Errorf("GatherI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestScatterF32(t *testing.T) {
	values := []float32{100, 200, 300, 400}
	indices := []int32{3, 1, 7, 5}
	base := make([]float32, 8)

	ScatterF32(values, indices, base)

	for i := range indices {
		if base[indices[i]] != values[i] {
			t.Errorf("ScatterF32: base[%d] = %v, want %v", indices[i], base[indices[i]], values[i])
		}
	}
}

func TestScatterF64(t *testing.T) {
	values := []float64{100, 200, 300, 400}
	indices := []int32{3, 1, 7, 5}
	base := make([]float64, 8)

	ScatterF64(values, indices, base)

	for i := range indices {
		if base[indices[i]] != values[i] {
			t.Errorf("ScatterF64: base[%d] = %v, want %v", indices[i], base[indices[i]], values[i])
		}
	}
}

func TestScatterI32(t *testing.T) {
	values := []int32{100, 200, 300, 400}
	indices := []int32{3, 1, 7, 5}
	base := make([]int32, 8)

	ScatterI32(values, indices, base)

	for i := range indices {
		if base[indices[i]] != values[i] {
			t.Errorf("ScatterI32: base[%d] = %v, want %v", indices[i], base[indices[i]], values[i])
		}
	}
}

func TestMaskedLoadF32(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	mask := []int32{1, 0, 1, 0, 1, 0, 1, 0}
	result := make([]float32, len(input))

	MaskedLoadF32(input, mask, result)

	for i := range input {
		var expected float32
		if mask[i] != 0 {
			expected = input[i]
		} else {
			expected = 0
		}
		if result[i] != expected {
			t.Errorf("MaskedLoadF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestMaskedStoreF32(t *testing.T) {
	input := []float32{100, 200, 300, 400, 500, 600, 700, 800}
	mask := []int32{1, 0, 1, 0, 1, 0, 1, 0}
	output := []float32{1, 2, 3, 4, 5, 6, 7, 8}

	MaskedStoreF32(input, mask, output)

	for i := range input {
		var expected float32
		if mask[i] != 0 {
			expected = input[i]
		} else {
			expected = float32(i + 1) // original value
		}
		if output[i] != expected {
			t.Errorf("MaskedStoreF32[%d]: got %v, want %v", i, output[i], expected)
		}
	}
}

// Test non-aligned sizes for new operations
func TestTypeConversionsNonAligned(t *testing.T) {
	// Test with 7 elements (not multiple of 4)
	f32 := []float32{1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5}
	f64 := make([]float64, len(f32))

	PromoteF32ToF64(f32, f64)

	for i := range f32 {
		expected := float64(f32[i])
		if f64[i] != expected {
			t.Errorf("PromoteF32ToF64 (non-aligned)[%d]: got %v, want %v", i, f64[i], expected)
		}
	}
}

func TestGatherNonAligned(t *testing.T) {
	base := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	indices := []int32{7, 0, 3} // 3 elements
	result := make([]float32, len(indices))

	GatherF32(base, indices, result)

	for i := range indices {
		expected := base[indices[i]]
		if result[i] != expected {
			t.Errorf("GatherF32 (non-aligned)[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

// Benchmarks for new operations

func BenchmarkGatherF32_NEON(b *testing.B) {
	n := 1024
	base := make([]float32, n)
	indices := make([]int32, n)
	result := make([]float32, n)
	for i := range base {
		base[i] = float32(i)
		indices[i] = int32((i * 7) % n) // pseudo-random indices
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		GatherF32(base, indices, result)
	}
}

func BenchmarkPromoteF32ToF64_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float32(i)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		PromoteF32ToF64(input, result)
	}
}

func BenchmarkRoundF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i) + 0.5
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		RoundF32(input, result)
	}
}
