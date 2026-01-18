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

	b.ReportAllocs()
	for b.Loop() {
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

	b.ReportAllocs()
	for b.Loop() {
		MulF32(a, bb, result)
	}
}

func BenchmarkReduceSumF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(i)
	}

	b.ReportAllocs()
	for b.Loop() {
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

	b.ReportAllocs()
	for b.Loop() {
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

	b.ReportAllocs()
	for b.Loop() {
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

	b.ReportAllocs()
	for b.Loop() {
		var sum float32
		for _, v := range input {
			sum += v
		}
		_ = sum
	}
}

func BenchmarkSqrtF32_Scalar(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i + 1)
	}

	b.ReportAllocs()
	for b.Loop() {
		for j, v := range input {
			result[j] = float32(math.Sqrt(float64(v)))
		}
	}
}

func BenchmarkExpF32_Scalar(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		for j, v := range input {
			result[j] = float32(math.Exp(float64(v)))
		}
	}
}

func BenchmarkSinF32_Scalar(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		for j, v := range input {
			result[j] = float32(math.Sin(float64(v)))
		}
	}
}

func BenchmarkAtanF32_Scalar(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i-512) * 0.01 // range [-5.12, 5.11]
	}

	b.ReportAllocs()
	for b.Loop() {
		for j, v := range input {
			result[j] = float32(math.Atan(float64(v)))
		}
	}
}

func BenchmarkLog2F32_Scalar(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i+1) * 0.1 // range [0.1, 102.4]
	}

	b.ReportAllocs()
	for b.Loop() {
		for j, v := range input {
			result[j] = float32(math.Log2(float64(v)))
		}
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

func TestGatherI64(t *testing.T) {
	base := []int64{10, 20, 30, 40, 50, 60, 70, 80}
	indices := []int32{7, 0, 3, 5, 2, 1, 6, 4}
	result := make([]int64, len(indices))

	GatherI64(base, indices, result)

	for i := range indices {
		expected := base[indices[i]]
		if result[i] != expected {
			t.Errorf("GatherI64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestScatterI64(t *testing.T) {
	values := []int64{100, 200, 300, 400}
	indices := []int32{3, 1, 7, 5}
	base := make([]int64, 8)

	ScatterI64(values, indices, base)

	for i := range indices {
		if base[indices[i]] != values[i] {
			t.Errorf("ScatterI64: base[%d] = %v, want %v", indices[i], base[indices[i]], values[i])
		}
	}
}

func TestMaskedLoadF64(t *testing.T) {
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	mask := []int64{1, 0, 1, 0, 1, 0, 1, 0}
	result := make([]float64, len(input))

	MaskedLoadF64(input, mask, result)

	for i := range input {
		var expected float64
		if mask[i] != 0 {
			expected = input[i]
		} else {
			expected = 0
		}
		if result[i] != expected {
			t.Errorf("MaskedLoadF64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestMaskedStoreF64(t *testing.T) {
	input := []float64{100, 200, 300, 400, 500, 600, 700, 800}
	mask := []int64{1, 0, 1, 0, 1, 0, 1, 0}
	output := []float64{1, 2, 3, 4, 5, 6, 7, 8}

	MaskedStoreF64(input, mask, output)

	for i := range input {
		var expected float64
		if mask[i] != 0 {
			expected = input[i]
		} else {
			expected = float64(i + 1) // original value
		}
		if output[i] != expected {
			t.Errorf("MaskedStoreF64[%d]: got %v, want %v", i, output[i], expected)
		}
	}
}

func TestMaskedLoadI32(t *testing.T) {
	input := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	mask := []int32{1, 0, 1, 0, 1, 0, 1, 0}
	result := make([]int32, len(input))

	MaskedLoadI32(input, mask, result)

	for i := range input {
		var expected int32
		if mask[i] != 0 {
			expected = input[i]
		} else {
			expected = 0
		}
		if result[i] != expected {
			t.Errorf("MaskedLoadI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestMaskedStoreI32(t *testing.T) {
	input := []int32{100, 200, 300, 400, 500, 600, 700, 800}
	mask := []int32{1, 0, 1, 0, 1, 0, 1, 0}
	output := []int32{1, 2, 3, 4, 5, 6, 7, 8}

	MaskedStoreI32(input, mask, output)

	for i := range input {
		var expected int32
		if mask[i] != 0 {
			expected = input[i]
		} else {
			expected = int32(i + 1) // original value
		}
		if output[i] != expected {
			t.Errorf("MaskedStoreI32[%d]: got %v, want %v", i, output[i], expected)
		}
	}
}

func TestMaskedLoadI64(t *testing.T) {
	input := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	mask := []int64{1, 0, 1, 0, 1, 0, 1, 0}
	result := make([]int64, len(input))

	MaskedLoadI64(input, mask, result)

	for i := range input {
		var expected int64
		if mask[i] != 0 {
			expected = input[i]
		} else {
			expected = 0
		}
		if result[i] != expected {
			t.Errorf("MaskedLoadI64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestMaskedStoreI64(t *testing.T) {
	input := []int64{100, 200, 300, 400, 500, 600, 700, 800}
	mask := []int64{1, 0, 1, 0, 1, 0, 1, 0}
	output := []int64{1, 2, 3, 4, 5, 6, 7, 8}

	MaskedStoreI64(input, mask, output)

	for i := range input {
		var expected int64
		if mask[i] != 0 {
			expected = input[i]
		} else {
			expected = int64(i + 1) // original value
		}
		if output[i] != expected {
			t.Errorf("MaskedStoreI64[%d]: got %v, want %v", i, output[i], expected)
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

	b.ReportAllocs()
	for b.Loop() {
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

	b.ReportAllocs()
	for b.Loop() {
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

	b.ReportAllocs()
	for b.Loop() {
		RoundF32(input, result)
	}
}

// Phase 6: Shuffle/Permutation Tests

func TestReverseF32(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	result := make([]float32, len(input))

	ReverseF32(input, result)

	for i := range input {
		expected := input[len(input)-1-i]
		if result[i] != expected {
			t.Errorf("ReverseF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestReverseF64(t *testing.T) {
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	result := make([]float64, len(input))

	ReverseF64(input, result)

	for i := range input {
		expected := input[len(input)-1-i]
		if result[i] != expected {
			t.Errorf("ReverseF64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestReverse2F32(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	result := make([]float32, len(input))

	Reverse2F32(input, result)

	// Expected: [2,1,4,3,6,5,8,7]
	expected := []float32{2, 1, 4, 3, 6, 5, 8, 7}
	for i := range input {
		if result[i] != expected[i] {
			t.Errorf("Reverse2F32[%d]: got %v, want %v", i, result[i], expected[i])
		}
	}
}

func TestReverse4F32(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	result := make([]float32, len(input))

	Reverse4F32(input, result)

	// Expected: [4,3,2,1,8,7,6,5]
	expected := []float32{4, 3, 2, 1, 8, 7, 6, 5}
	for i := range input {
		if result[i] != expected[i] {
			t.Errorf("Reverse4F32[%d]: got %v, want %v", i, result[i], expected[i])
		}
	}
}

func TestBroadcastF32(t *testing.T) {
	input := []float32{10, 20, 30, 40}
	result := make([]float32, 8)

	BroadcastF32(input, 2, result) // Broadcast input[2] = 30

	for i := range result {
		if result[i] != 30 {
			t.Errorf("BroadcastF32[%d]: got %v, want 30", i, result[i])
		}
	}
}

func TestGetLaneF32(t *testing.T) {
	input := []float32{10, 20, 30, 40, 50, 60, 70, 80}

	for i, expected := range input {
		got := GetLaneF32(input, i)
		if got != expected {
			t.Errorf("GetLaneF32(%d): got %v, want %v", i, got, expected)
		}
	}
}

func TestInsertLaneF32(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	result := make([]float32, len(input))

	InsertLaneF32(input, 3, 999, result)

	for i := range input {
		var expected float32
		if i == 3 {
			expected = 999
		} else {
			expected = input[i]
		}
		if result[i] != expected {
			t.Errorf("InsertLaneF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestInterleaveLowerF32(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{10, 20, 30, 40}
	result := make([]float32, len(a))

	InterleaveLowerF32(a, b, result)

	// Expected: [a0,b0,a1,b1] = [1,10,2,20]
	expected := []float32{1, 10, 2, 20}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("InterleaveLowerF32[%d]: got %v, want %v", i, result[i], expected[i])
		}
	}
}

func TestInterleaveUpperF32(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{10, 20, 30, 40}
	result := make([]float32, len(a))

	InterleaveUpperF32(a, b, result)

	// Expected: [a2,b2,a3,b3] = [3,30,4,40]
	expected := []float32{3, 30, 4, 40}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("InterleaveUpperF32[%d]: got %v, want %v", i, result[i], expected[i])
		}
	}
}

func TestTableLookupBytesU8(t *testing.T) {
	// Table: 0-15
	tbl := make([]uint8, 16)
	for i := range tbl {
		tbl[i] = uint8(i * 10) // 0, 10, 20, ..., 150
	}

	idx := []uint8{0, 1, 2, 3, 15, 14, 13, 12, 0, 0, 0, 0, 5, 5, 5, 5}
	result := make([]uint8, len(idx))

	TableLookupBytesU8(tbl, idx, result)

	for i := range idx {
		var expected uint8
		if idx[i] < 16 {
			expected = tbl[idx[i]]
		} else {
			expected = 0
		}
		if result[i] != expected {
			t.Errorf("TableLookupBytesU8[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

// Phase 7: Comparison Tests

func TestEqF32(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float32{1, 0, 3, 0, 5, 0, 7, 0}
	result := make([]int32, len(a))

	EqF32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] == b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("EqF32[%d]: got %v, want %v (a=%v, b=%v)", i, result[i], expected, a[i], b[i])
		}
	}
}

func TestEqI32(t *testing.T) {
	a := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []int32{1, 0, 3, 0, 5, 0, 7, 0}
	result := make([]int32, len(a))

	EqI32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] == b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("EqI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestNeF32(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float32{1, 0, 3, 0, 5, 0, 7, 0}
	result := make([]int32, len(a))

	NeF32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] != b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("NeF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestNeI32(t *testing.T) {
	a := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []int32{1, 0, 3, 0, 5, 0, 7, 0}
	result := make([]int32, len(a))

	NeI32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] != b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("NeI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestLtF32(t *testing.T) {
	a := []float32{1, 5, 3, 7, 2, 8, 4, 6}
	b := []float32{2, 3, 4, 5, 6, 4, 5, 3}
	result := make([]int32, len(a))

	LtF32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] < b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("LtF32[%d]: got %v, want %v (a=%v < b=%v)", i, result[i], expected, a[i], b[i])
		}
	}
}

func TestLtI32(t *testing.T) {
	a := []int32{1, 5, 3, 7, 2, 8, 4, 6}
	b := []int32{2, 3, 4, 5, 6, 4, 5, 3}
	result := make([]int32, len(a))

	LtI32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] < b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("LtI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestLeF32(t *testing.T) {
	a := []float32{1, 5, 3, 7, 2, 8, 4, 6}
	b := []float32{1, 3, 3, 5, 2, 4, 4, 3}
	result := make([]int32, len(a))

	LeF32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] <= b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("LeF32[%d]: got %v, want %v (a=%v <= b=%v)", i, result[i], expected, a[i], b[i])
		}
	}
}

func TestLeI32(t *testing.T) {
	a := []int32{1, 5, 3, 7, 2, 8, 4, 6}
	b := []int32{1, 3, 3, 5, 2, 4, 4, 3}
	result := make([]int32, len(a))

	LeI32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] <= b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("LeI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestGtF32(t *testing.T) {
	a := []float32{1, 5, 3, 7, 2, 8, 4, 6}
	b := []float32{2, 3, 4, 5, 6, 4, 5, 3}
	result := make([]int32, len(a))

	GtF32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] > b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("GtF32[%d]: got %v, want %v (a=%v > b=%v)", i, result[i], expected, a[i], b[i])
		}
	}
}

func TestGtI32(t *testing.T) {
	a := []int32{1, 5, 3, 7, 2, 8, 4, 6}
	b := []int32{2, 3, 4, 5, 6, 4, 5, 3}
	result := make([]int32, len(a))

	GtI32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] > b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("GtI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestGeF32(t *testing.T) {
	a := []float32{1, 5, 3, 7, 2, 8, 4, 6}
	b := []float32{1, 3, 3, 5, 2, 4, 4, 3}
	result := make([]int32, len(a))

	GeF32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] >= b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("GeF32[%d]: got %v, want %v (a=%v >= b=%v)", i, result[i], expected, a[i], b[i])
		}
	}
}

func TestGeI32(t *testing.T) {
	a := []int32{1, 5, 3, 7, 2, 8, 4, 6}
	b := []int32{1, 3, 3, 5, 2, 4, 4, 3}
	result := make([]int32, len(a))

	GeI32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] >= b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("GeI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

// Float64 comparison tests

func TestEqF64(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float64{1, 0, 3, 0, 5, 0, 7, 0}
	result := make([]int64, len(a))

	EqF64(a, b, result)

	for i := range a {
		expected := int64(0)
		if a[i] == b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("EqF64[%d]: got %v, want %v (a=%v, b=%v)", i, result[i], expected, a[i], b[i])
		}
	}
}

func TestGtF64(t *testing.T) {
	a := []float64{1, 5, 3, 7, 2, 8, 4, 6}
	b := []float64{1, 3, 3, 5, 2, 4, 4, 3}
	result := make([]int64, len(a))

	GtF64(a, b, result)

	for i := range a {
		expected := int64(0)
		if a[i] > b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("GtF64[%d]: got %v, want %v (a=%v, b=%v)", i, result[i], expected, a[i], b[i])
		}
	}
}

func TestGeF64(t *testing.T) {
	a := []float64{1, 5, 3, 7, 2, 8, 4, 6}
	b := []float64{1, 3, 3, 5, 2, 4, 4, 3}
	result := make([]int64, len(a))

	GeF64(a, b, result)

	for i := range a {
		expected := int64(0)
		if a[i] >= b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("GeF64[%d]: got %v, want %v (a=%v, b=%v)", i, result[i], expected, a[i], b[i])
		}
	}
}

func TestLtF64(t *testing.T) {
	a := []float64{1, 5, 3, 7, 2, 8, 4, 6}
	b := []float64{1, 3, 3, 5, 2, 4, 4, 3}
	result := make([]int64, len(a))

	LtF64(a, b, result)

	for i := range a {
		expected := int64(0)
		if a[i] < b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("LtF64[%d]: got %v, want %v (a=%v, b=%v)", i, result[i], expected, a[i], b[i])
		}
	}
}

func TestLeF64(t *testing.T) {
	a := []float64{1, 5, 3, 7, 2, 8, 4, 6}
	b := []float64{1, 3, 3, 5, 2, 4, 4, 3}
	result := make([]int64, len(a))

	LeF64(a, b, result)

	for i := range a {
		expected := int64(0)
		if a[i] <= b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("LeF64[%d]: got %v, want %v (a=%v, b=%v)", i, result[i], expected, a[i], b[i])
		}
	}
}

// Non-aligned size tests for new operations
func TestShuffleNonAligned(t *testing.T) {
	// Test with 7 elements (not multiple of 4)
	input := []float32{1, 2, 3, 4, 5, 6, 7}
	result := make([]float32, len(input))

	ReverseF32(input, result)

	for i := range input {
		expected := input[len(input)-1-i]
		if result[i] != expected {
			t.Errorf("ReverseF32 (non-aligned)[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestComparisonNonAligned(t *testing.T) {
	// Test with 7 elements
	a := []float32{1, 2, 3, 4, 5, 6, 7}
	b := []float32{1, 0, 3, 0, 5, 0, 7}
	result := make([]int32, len(a))

	EqF32(a, b, result)

	for i := range a {
		expected := int32(0)
		if a[i] == b[i] {
			expected = -1
		}
		if result[i] != expected {
			t.Errorf("EqF32 (non-aligned)[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

// Benchmarks for new operations

func BenchmarkReverseF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i)
	}

	b.ReportAllocs()
	for b.Loop() {
		ReverseF32(input, result)
	}
}

func BenchmarkEqF32_NEON(b *testing.B) {
	n := 1024
	a := make([]float32, n)
	bb := make([]float32, n)
	result := make([]int32, n)
	for i := range a {
		a[i] = float32(i)
		bb[i] = float32(i % 10)
	}

	b.ReportAllocs()
	for b.Loop() {
		EqF32(a, bb, result)
	}
}

func BenchmarkLtF32_NEON(b *testing.B) {
	n := 1024
	a := make([]float32, n)
	bb := make([]float32, n)
	result := make([]int32, n)
	for i := range a {
		a[i] = float32(i)
		bb[i] = float32(n - i)
	}

	b.ReportAllocs()
	for b.Loop() {
		LtF32(a, bb, result)
	}
}

// Phase 8: Bitwise Operations Tests

func TestAndI32(t *testing.T) {
	a := []int32{0xFF, 0xF0, 0x0F, 0x00, 0xFF00, 0x00FF, 0xFFFF, 0x1234}
	b := []int32{0x0F, 0x0F, 0x0F, 0x0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x00FF}
	result := make([]int32, len(a))

	AndI32(a, b, result)

	for i := range a {
		expected := a[i] & b[i]
		if result[i] != expected {
			t.Errorf("AndI32[%d]: got 0x%X, want 0x%X", i, result[i], expected)
		}
	}
}

func TestOrI32(t *testing.T) {
	a := []int32{0xFF, 0xF0, 0x0F, 0x00, 0xFF00, 0x00FF, 0xFFFF, 0x1234}
	b := []int32{0x0F, 0x0F, 0x0F, 0x0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x00FF}
	result := make([]int32, len(a))

	OrI32(a, b, result)

	for i := range a {
		expected := a[i] | b[i]
		if result[i] != expected {
			t.Errorf("OrI32[%d]: got 0x%X, want 0x%X", i, result[i], expected)
		}
	}
}

func TestXorI32(t *testing.T) {
	a := []int32{0xFF, 0xF0, 0x0F, 0x00, 0xFF00, 0x00FF, 0xFFFF, 0x1234}
	b := []int32{0x0F, 0x0F, 0x0F, 0x0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x00FF}
	result := make([]int32, len(a))

	XorI32(a, b, result)

	for i := range a {
		expected := a[i] ^ b[i]
		if result[i] != expected {
			t.Errorf("XorI32[%d]: got 0x%X, want 0x%X", i, result[i], expected)
		}
	}
}

func TestAndNotI32(t *testing.T) {
	a := []int32{0xFF, 0xF0, 0x0F, 0x00, 0xFF00, 0x00FF, 0xFFFF, 0x1234}
	b := []int32{0x0F, 0x0F, 0x0F, 0x0F, 0x0F0F, 0x0F0F, 0x0F0F, 0x00FF}
	result := make([]int32, len(a))

	AndNotI32(a, b, result)

	for i := range a {
		expected := a[i] & ^b[i]
		if result[i] != expected {
			t.Errorf("AndNotI32[%d]: got 0x%X, want 0x%X", i, result[i], expected)
		}
	}
}

func TestNotI32(t *testing.T) {
	a := []int32{0x00000000, -1, 0x0F0F0F0F, -0x0F0F0F10, 0x12345678, -0x789ABCDF, 1, -2}
	result := make([]int32, len(a))

	NotI32(a, result)

	for i := range a {
		expected := ^a[i]
		if result[i] != expected {
			t.Errorf("NotI32[%d]: got 0x%X, want 0x%X", i, result[i], expected)
		}
	}
}

func TestShiftLeftI32(t *testing.T) {
	a := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	result := make([]int32, len(a))

	ShiftLeftI32(a, 2, result)

	for i := range a {
		expected := a[i] << 2
		if result[i] != expected {
			t.Errorf("ShiftLeftI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestShiftRightI32(t *testing.T) {
	a := []int32{16, 32, 64, 128, -16, -32, -64, -128}
	result := make([]int32, len(a))

	ShiftRightI32(a, 2, result)

	for i := range a {
		expected := a[i] >> 2
		if result[i] != expected {
			t.Errorf("ShiftRightI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

// Phase 9: Mask Operations Tests

func TestIfThenElseF32(t *testing.T) {
	mask := []int32{-1, 0, -1, 0, -1, 0, -1, 0}
	yes := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	no := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	result := make([]float32, len(mask))

	IfThenElseF32(mask, yes, no, result)

	for i := range mask {
		var expected float32
		if mask[i] != 0 {
			expected = yes[i]
		} else {
			expected = no[i]
		}
		if result[i] != expected {
			t.Errorf("IfThenElseF32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestIfThenElseI32(t *testing.T) {
	mask := []int32{-1, 0, -1, 0, -1, 0, -1, 0}
	yes := []int32{10, 20, 30, 40, 50, 60, 70, 80}
	no := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	result := make([]int32, len(mask))

	IfThenElseI32(mask, yes, no, result)

	for i := range mask {
		var expected int32
		if mask[i] != 0 {
			expected = yes[i]
		} else {
			expected = no[i]
		}
		if result[i] != expected {
			t.Errorf("IfThenElseI32[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestCountTrueI32(t *testing.T) {
	tests := []struct {
		mask     []int32
		expected int64
	}{
		{[]int32{-1, -1, -1, -1, -1, -1, -1, -1}, 8},
		{[]int32{0, 0, 0, 0, 0, 0, 0, 0}, 0},
		{[]int32{-1, 0, -1, 0, -1, 0, -1, 0}, 4},
		{[]int32{-1, 0, 0, 0, 0, 0, 0, 0}, 1},
	}

	for _, tt := range tests {
		result := CountTrueI32(tt.mask)
		if result != tt.expected {
			t.Errorf("CountTrueI32(%v): got %v, want %v", tt.mask, result, tt.expected)
		}
	}
}

func TestAllTrueI32(t *testing.T) {
	tests := []struct {
		mask     []int32
		expected bool
	}{
		{[]int32{-1, -1, -1, -1, -1, -1, -1, -1}, true},
		{[]int32{-1, -1, -1, -1, -1, -1, -1, 0}, false},
		{[]int32{0, 0, 0, 0, 0, 0, 0, 0}, false},
		{[]int32{1, 2, 3, 4, 5, 6, 7, 8}, true}, // any non-zero is true
	}

	for _, tt := range tests {
		result := AllTrueI32(tt.mask)
		if result != tt.expected {
			t.Errorf("AllTrueI32(%v): got %v, want %v", tt.mask, result, tt.expected)
		}
	}
}

func TestAllFalseI32(t *testing.T) {
	tests := []struct {
		mask     []int32
		expected bool
	}{
		{[]int32{0, 0, 0, 0, 0, 0, 0, 0}, true},
		{[]int32{0, 0, 0, 0, 0, 0, 0, -1}, false},
		{[]int32{-1, -1, -1, -1, -1, -1, -1, -1}, false},
		{[]int32{0, 0, 0, 0, 0, 0, 0, 1}, false},
	}

	for _, tt := range tests {
		result := AllFalseI32(tt.mask)
		if result != tt.expected {
			t.Errorf("AllFalseI32(%v): got %v, want %v", tt.mask, result, tt.expected)
		}
	}
}

func TestFirstNI32(t *testing.T) {
	tests := []struct {
		count    int
		len      int
		expected []int32
	}{
		{0, 8, []int32{0, 0, 0, 0, 0, 0, 0, 0}},
		{3, 8, []int32{-1, -1, -1, 0, 0, 0, 0, 0}},
		{8, 8, []int32{-1, -1, -1, -1, -1, -1, -1, -1}},
		{4, 4, []int32{-1, -1, -1, -1}},
	}

	for _, tt := range tests {
		result := make([]int32, tt.len)
		FirstNI32(tt.count, result)
		for i := range result {
			if result[i] != tt.expected[i] {
				t.Errorf("FirstNI32(%d)[%d]: got %v, want %v", tt.count, i, result[i], tt.expected[i])
			}
		}
	}
}

func TestCompressF32(t *testing.T) {
	input := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	mask := []int32{-1, 0, -1, 0, -1, 0, -1, 0}
	result := make([]float32, len(input))

	count := CompressF32(input, mask, result)

	expected := []float32{10, 30, 50, 70}
	if count != 4 {
		t.Errorf("CompressF32: count = %v, want 4", count)
	}
	for i := range count {
		if result[i] != expected[i] {
			t.Errorf("CompressF32[%d]: got %v, want %v", i, result[i], expected[i])
		}
	}
}

func TestExpandF32(t *testing.T) {
	input := []float32{100, 200, 300, 400, 0, 0, 0, 0}
	mask := []int32{-1, 0, -1, 0, -1, 0, -1, 0}
	result := make([]float32, len(mask))

	ExpandF32(input, mask, result)

	expected := []float32{100, 0, 200, 0, 300, 0, 400, 0}
	for i := range result {
		if result[i] != expected[i] {
			t.Errorf("ExpandF32[%d]: got %v, want %v", i, result[i], expected[i])
		}
	}
}

// Phase 10: Transcendental Math Tests

func TestExpF32(t *testing.T) {
	input := []float32{0, 1, -1, 2, -2, 0.5, -0.5, 3}
	result := make([]float32, len(input))

	ExpF32(input, result)

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		relErr := math.Abs(float64(result[i]-expected)) / math.Abs(float64(expected))
		if relErr > 1e-5 {
			t.Errorf("ExpF32[%d]: got %v, want %v (relative error: %v)", i, result[i], expected, relErr)
		}
	}
}

func TestLogF32(t *testing.T) {
	input := []float32{1, 2, 3, 4, 0.5, 10, 100, 0.1}
	result := make([]float32, len(input))

	LogF32(input, result)

	// Polynomial approximations have lower precision than math.Log
	// Expect ~0.2% relative error for SIMD implementations
	for i := range input {
		expected := float32(math.Log(float64(input[i])))
		relErr := math.Abs(float64(result[i]-expected)) / math.Abs(float64(expected)+1e-10)
		if relErr > 2e-3 {
			t.Errorf("LogF32[%d]: got %v, want %v (relative error: %v)", i, result[i], expected, relErr)
		}
	}
}

func TestSinF32(t *testing.T) {
	input := []float32{0, 0.5, 1, 1.5, 2, 3, -1, -2}
	result := make([]float32, len(input))

	SinF32(input, result)

	// 7th-order polynomial approximation achieves ~1e-4 accuracy
	for i := range input {
		expected := float32(math.Sin(float64(input[i])))
		absErr := math.Abs(float64(result[i] - expected))
		if absErr > 1e-3 {
			t.Errorf("SinF32[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestCosF32(t *testing.T) {
	input := []float32{0, 0.5, 1, 1.5, 2, 3, -1, -2}
	result := make([]float32, len(input))

	CosF32(input, result)

	// 6th-order polynomial approximation achieves ~1e-3 accuracy
	for i := range input {
		expected := float32(math.Cos(float64(input[i])))
		absErr := math.Abs(float64(result[i] - expected))
		if absErr > 1e-3 {
			t.Errorf("CosF32[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestTanhF32(t *testing.T) {
	input := []float32{0, 0.5, 1, 2, -0.5, -1, -2, 3}
	result := make([]float32, len(input))

	TanhF32(input, result)

	// Rational approximation achieves ~2e-3 accuracy
	for i := range input {
		expected := float32(math.Tanh(float64(input[i])))
		absErr := math.Abs(float64(result[i] - expected))
		if absErr > 3e-3 {
			t.Errorf("TanhF32[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestSigmoidF32(t *testing.T) {
	input := []float32{0, 1, -1, 2, -2, 5, -5, 10}
	result := make([]float32, len(input))

	SigmoidF32(input, result)

	// Sigmoid via exp approximation achieves ~1e-4 accuracy
	for i := range input {
		expected := float32(1.0 / (1.0 + math.Exp(-float64(input[i]))))
		absErr := math.Abs(float64(result[i] - expected))
		if absErr > 2e-4 {
			t.Errorf("SigmoidF32[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

// Non-aligned size tests for Phase 8-10
func TestBitwiseNonAligned(t *testing.T) {
	a := []int32{0xFF, 0xF0, 0x0F, 0x00, 0xFF00, 0x00FF, 0xFFFF}
	b := []int32{0x0F, 0x0F, 0x0F, 0x0F, 0x0F0F, 0x0F0F, 0x0F0F}
	result := make([]int32, len(a))

	AndI32(a, b, result)

	for i := range a {
		expected := a[i] & b[i]
		if result[i] != expected {
			t.Errorf("AndI32 (non-aligned)[%d]: got 0x%X, want 0x%X", i, result[i], expected)
		}
	}
}

func TestMaskOpsNonAligned(t *testing.T) {
	mask := []int32{-1, 0, -1, 0, -1, 0, -1}
	yes := []float32{10, 20, 30, 40, 50, 60, 70}
	no := []float32{1, 2, 3, 4, 5, 6, 7}
	result := make([]float32, len(mask))

	IfThenElseF32(mask, yes, no, result)

	for i := range mask {
		var expected float32
		if mask[i] != 0 {
			expected = yes[i]
		} else {
			expected = no[i]
		}
		if result[i] != expected {
			t.Errorf("IfThenElseF32 (non-aligned)[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestTranscendentalNonAligned(t *testing.T) {
	input := []float32{0, 1, -1, 2, -2, 0.5, -0.5}
	result := make([]float32, len(input))

	ExpF32(input, result)

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		relErr := math.Abs(float64(result[i]-expected)) / math.Abs(float64(expected))
		if relErr > 1e-5 {
			t.Errorf("ExpF32 (non-aligned)[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

// Benchmarks for Phase 8-10 operations

func BenchmarkAndI32_NEON(b *testing.B) {
	n := 1024
	a := make([]int32, n)
	bb := make([]int32, n)
	result := make([]int32, n)
	for i := range a {
		a[i] = int32(i * 0x10101)
		bb[i] = int32(i * 0x01010)
	}

	b.ReportAllocs()
	for b.Loop() {
		AndI32(a, bb, result)
	}
}

func BenchmarkIfThenElseF32_NEON(b *testing.B) {
	n := 1024
	mask := make([]int32, n)
	yes := make([]float32, n)
	no := make([]float32, n)
	result := make([]float32, n)
	for i := range mask {
		if i%2 == 0 {
			mask[i] = -1
		}
		yes[i] = float32(i * 10)
		no[i] = float32(i)
	}

	b.ReportAllocs()
	for b.Loop() {
		IfThenElseF32(mask, yes, no, result)
	}
}

func BenchmarkExpF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i%10) - 5 // range [-5, 4]
	}

	b.ReportAllocs()
	for b.Loop() {
		ExpF32(input, result)
	}
}

func BenchmarkSinF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i) * 0.01 // range [0, ~10]
	}

	b.ReportAllocs()
	for b.Loop() {
		SinF32(input, result)
	}
}

func BenchmarkTanhF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i%20) - 10 // range [-10, 9]
	}

	b.ReportAllocs()
	for b.Loop() {
		TanhF32(input, result)
	}
}

func BenchmarkSigmoidF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i%20) - 10 // range [-10, 9]
	}

	b.ReportAllocs()
	for b.Loop() {
		SigmoidF32(input, result)
	}
}

// Additional F32 Transcendental Tests

func TestTanF32(t *testing.T) {
	// Avoid values near pi/2 where tan has asymptotes
	input := []float32{0, 0.1, 0.5, 1.0, -0.1, -0.5, -1.0, 0.3}
	result := make([]float32, len(input))

	TanF32(input, result)

	for i := range input {
		expected := float32(math.Tan(float64(input[i])))
		absErr := math.Abs(float64(result[i] - expected))
		relErr := absErr / (math.Abs(float64(expected)) + 1e-10)
		if absErr > 1e-3 && relErr > 1e-3 {
			t.Errorf("TanF32[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestAtanF32(t *testing.T) {
	input := []float32{0, 0.5, 1, 2, -0.5, -1, -2, 10}
	result := make([]float32, len(input))

	AtanF32(input, result)

	for i := range input {
		expected := float32(math.Atan(float64(input[i])))
		absErr := math.Abs(float64(result[i] - expected))
		if absErr > 1e-3 {
			t.Errorf("AtanF32[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestAtan2F32(t *testing.T) {
	y := []float32{1, 1, -1, -1, 0, 1, 0, 2}
	x := []float32{1, -1, 1, -1, 1, 0, -1, 2}
	result := make([]float32, len(y))

	Atan2F32(y, x, result)

	for i := range y {
		expected := float32(math.Atan2(float64(y[i]), float64(x[i])))
		absErr := math.Abs(float64(result[i] - expected))
		if absErr > 1e-3 {
			t.Errorf("Atan2F32[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestPowF32(t *testing.T) {
	base := []float32{2, 2, 2, 3, 10, 2.5, 4, 8}
	exp := []float32{0, 1, 2, 2, 2, 2, 0.5, 0.333333}
	result := make([]float32, len(base))

	PowF32(base, exp, result)

	for i := range base {
		expected := float32(math.Pow(float64(base[i]), float64(exp[i])))
		relErr := math.Abs(float64(result[i]-expected)) / (math.Abs(float64(expected)) + 1e-10)
		if relErr > 1e-2 {
			t.Errorf("PowF32[%d]: got %v, want %v (relative error: %v)", i, result[i], expected, relErr)
		}
	}
}

func TestErfF32(t *testing.T) {
	input := []float32{0, 0.5, 1, 1.5, 2, -0.5, -1, -2}
	result := make([]float32, len(input))

	ErfF32(input, result)

	for i := range input {
		expected := float32(math.Erf(float64(input[i])))
		absErr := math.Abs(float64(result[i] - expected))
		if absErr > 1e-3 {
			t.Errorf("ErfF32[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestExp2F32(t *testing.T) {
	input := []float32{0, 1, 2, 3, -1, -2, 0.5, 4}
	result := make([]float32, len(input))

	Exp2F32(input, result)

	for i := range input {
		expected := float32(math.Pow(2, float64(input[i])))
		relErr := math.Abs(float64(result[i]-expected)) / (math.Abs(float64(expected)) + 1e-10)
		if relErr > 1e-4 {
			t.Errorf("Exp2F32[%d]: got %v, want %v (relative error: %v)", i, result[i], expected, relErr)
		}
	}
}

func TestLog2F32(t *testing.T) {
	input := []float32{1, 2, 4, 8, 16, 0.5, 0.25, 3}
	result := make([]float32, len(input))

	Log2F32(input, result)

	for i := range input {
		expected := float32(math.Log2(float64(input[i])))
		absErr := math.Abs(float64(result[i] - expected))
		if absErr > 1e-3 { // SIMD polynomial approximation with sqrt(2) range reduction
			t.Errorf("Log2F32[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

// F64 Transcendental Tests
// Note: F64 functions use SIMD-only processing (multiples of 2 elements)

func TestExp2F64(t *testing.T) {
	input := []float64{0, 1, 2, 3, -1, -2, 0.5, 4}
	result := make([]float64, len(input))

	Exp2F64(input, result)

	for i := range input {
		expected := math.Pow(2, input[i])
		relErr := math.Abs(result[i]-expected) / (math.Abs(expected) + 1e-15)
		if relErr > 1e-10 {
			t.Errorf("Exp2F64[%d]: got %v, want %v (relative error: %v)", i, result[i], expected, relErr)
		}
	}
}

func TestLog2F64(t *testing.T) {
	input := []float64{1, 2, 4, 8, 16, 0.5, 0.25, 3}
	result := make([]float64, len(input))

	Log2F64(input, result)

	for i := range input {
		expected := math.Log2(input[i])
		absErr := math.Abs(result[i] - expected)
		if absErr > 1e-3 { // Relaxed for SIMD polynomial approximation
			t.Errorf("Log2F64[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestExpF64(t *testing.T) {
	input := []float64{0, 1, -1, 2, -2, 0.5, -0.5, 3}
	result := make([]float64, len(input))

	ExpF64(input, result)

	for i := range input {
		expected := math.Exp(input[i])
		relErr := math.Abs(result[i]-expected) / (math.Abs(expected) + 1e-15)
		if relErr > 1e-10 {
			t.Errorf("ExpF64[%d]: got %v, want %v (relative error: %v)", i, result[i], expected, relErr)
		}
	}
}

func TestLogF64(t *testing.T) {
	input := []float64{1, 2, 3, 4, 0.5, 10, 100, 0.1}
	result := make([]float64, len(input))

	LogF64(input, result)

	for i := range input {
		expected := math.Log(input[i])
		absErr := math.Abs(result[i] - expected)
		if absErr > 1e-3 { // Relaxed for SIMD polynomial approximation
			t.Errorf("LogF64[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestSinF64(t *testing.T) {
	input := []float64{0, 0.5, 1, 1.5, 2, 3, -1, -2}
	result := make([]float64, len(input))

	SinF64(input, result)

	for i := range input {
		expected := math.Sin(input[i])
		absErr := math.Abs(result[i] - expected)
		if absErr > 1e-6 { // Relaxed for SIMD polynomial approximation
			t.Errorf("SinF64[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestCosF64(t *testing.T) {
	input := []float64{0, 0.5, 1, 1.5, 2, 3, -1, -2}
	result := make([]float64, len(input))

	CosF64(input, result)

	for i := range input {
		expected := math.Cos(input[i])
		absErr := math.Abs(result[i] - expected)
		if absErr > 1e-6 { // Relaxed for SIMD polynomial approximation
			t.Errorf("CosF64[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestTanhF64(t *testing.T) {
	input := []float64{0, 0.5, 1, 2, -0.5, -1, -2, 3}
	result := make([]float64, len(input))

	TanhF64(input, result)

	for i := range input {
		expected := math.Tanh(input[i])
		absErr := math.Abs(result[i] - expected)
		if absErr > 1e-10 {
			t.Errorf("TanhF64[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestSigmoidF64(t *testing.T) {
	input := []float64{0, 1, -1, 2, -2, 5, -5, 10}
	result := make([]float64, len(input))

	SigmoidF64(input, result)

	for i := range input {
		expected := 1.0 / (1.0 + math.Exp(-input[i]))
		absErr := math.Abs(result[i] - expected)
		if absErr > 1e-10 {
			t.Errorf("SigmoidF64[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

// Benchmarks for additional F32 transcendentals

func BenchmarkTanF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i) * 0.001 // avoid asymptotes
	}

	b.ReportAllocs()
	for b.Loop() {
		TanF32(input, result)
	}
}

func BenchmarkAtanF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i%20) - 10
	}

	b.ReportAllocs()
	for b.Loop() {
		AtanF32(input, result)
	}
}

func BenchmarkPowF32_NEON(b *testing.B) {
	n := 1024
	base := make([]float32, n)
	exp := make([]float32, n)
	result := make([]float32, n)
	for i := range base {
		base[i] = float32(i%10) + 1
		exp[i] = float32(i%5) * 0.5
	}

	b.ReportAllocs()
	for b.Loop() {
		PowF32(base, exp, result)
	}
}

func BenchmarkErfF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i%10) - 5
	}

	b.ReportAllocs()
	for b.Loop() {
		ErfF32(input, result)
	}
}

func BenchmarkExp2F32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i%16) - 8
	}

	b.ReportAllocs()
	for b.Loop() {
		Exp2F32(input, result)
	}
}

func BenchmarkLog2F32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i+1) * 0.1
	}

	b.ReportAllocs()
	for b.Loop() {
		Log2F32(input, result)
	}
}

// Benchmarks for F64 transcendentals

func BenchmarkExp2F64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i%16) - 8
	}

	b.ReportAllocs()
	for b.Loop() {
		Exp2F64(input, result)
	}
}

func BenchmarkLog2F64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i+1) * 0.1
	}

	b.ReportAllocs()
	for b.Loop() {
		Log2F64(input, result)
	}
}

func BenchmarkExpF64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i%10) - 5
	}

	b.ReportAllocs()
	for b.Loop() {
		ExpF64(input, result)
	}
}

func BenchmarkLogF64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i+1) * 0.1
	}

	b.ReportAllocs()
	for b.Loop() {
		LogF64(input, result)
	}
}

func BenchmarkSinF64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		SinF64(input, result)
	}
}

func BenchmarkCosF64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		CosF64(input, result)
	}
}

func BenchmarkTanhF64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i%20) - 10
	}

	b.ReportAllocs()
	for b.Loop() {
		TanhF64(input, result)
	}
}

func BenchmarkSigmoidF64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i%20) - 10
	}

	b.ReportAllocs()
	for b.Loop() {
		SigmoidF64(input, result)
	}
}

// Log10, Exp10, SinCos Tests

func TestLog10F32(t *testing.T) {
	input := []float32{1, 10, 100, 1000, 0.1, 0.01, 2, 5}
	result := make([]float32, len(input))

	Log10F32(input, result)

	for i := range input {
		expected := float32(math.Log10(float64(input[i])))
		absErr := math.Abs(float64(result[i] - expected))
		if absErr > 1e-3 {
			t.Errorf("Log10F32[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestExp10F32(t *testing.T) {
	input := []float32{0, 1, 2, -1, -2, 0.5, 1.5, 3}
	result := make([]float32, len(input))

	Exp10F32(input, result)

	for i := range input {
		expected := float32(math.Pow(10, float64(input[i])))
		relErr := math.Abs(float64(result[i]-expected)) / (math.Abs(float64(expected)) + 1e-10)
		if relErr > 1e-3 {
			t.Errorf("Exp10F32[%d]: got %v, want %v (relative error: %v)", i, result[i], expected, relErr)
		}
	}
}

func TestSinCosF32(t *testing.T) {
	input := []float32{0, 0.5, 1.0, 1.5, 2.0, 3.14159, -1.0, -2.0}
	sinResult := make([]float32, len(input))
	cosResult := make([]float32, len(input))

	SinCosF32(input, sinResult, cosResult)

	for i := range input {
		expectedSin := float32(math.Sin(float64(input[i])))
		expectedCos := float32(math.Cos(float64(input[i])))

		sinErr := math.Abs(float64(sinResult[i] - expectedSin))
		cosErr := math.Abs(float64(cosResult[i] - expectedCos))

		if sinErr > 1e-3 {
			t.Errorf("SinCosF32[%d] sin: got %v, want %v (error: %v)", i, sinResult[i], expectedSin, sinErr)
		}
		if cosErr > 1e-3 {
			t.Errorf("SinCosF32[%d] cos: got %v, want %v (error: %v)", i, cosResult[i], expectedCos, cosErr)
		}
	}
}

// Log10, Exp10, SinCos Benchmarks

func BenchmarkLog10F32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i+1) * 0.1
	}

	b.ReportAllocs()
	for b.Loop() {
		Log10F32(input, result)
	}
}

func BenchmarkLog10F32_Scalar(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i+1) * 0.1
	}

	b.ReportAllocs()
	for b.Loop() {
		for j, v := range input {
			result[j] = float32(math.Log10(float64(v)))
		}
	}
}

func BenchmarkExp10F32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i%6) - 2 // range [-2, 3]
	}

	b.ReportAllocs()
	for b.Loop() {
		Exp10F32(input, result)
	}
}

func BenchmarkExp10F32_Scalar(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	result := make([]float32, n)
	for i := range input {
		input[i] = float32(i%6) - 2
	}

	b.ReportAllocs()
	for b.Loop() {
		for j, v := range input {
			result[j] = float32(math.Pow(10, float64(v)))
		}
	}
}

func BenchmarkSinCosF32_NEON(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	sinResult := make([]float32, n)
	cosResult := make([]float32, n)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		SinCosF32(input, sinResult, cosResult)
	}
}

func BenchmarkSinCosF32_Scalar(b *testing.B) {
	n := 1024
	input := make([]float32, n)
	sinResult := make([]float32, n)
	cosResult := make([]float32, n)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ReportAllocs()
	for b.Loop() {
		for j, v := range input {
			sinResult[j] = float32(math.Sin(float64(v)))
			cosResult[j] = float32(math.Cos(float64(v)))
		}
	}
}

// F64 Additional Transcendental Tests

func TestTanF64(t *testing.T) {
	input := []float64{0, 0.5, 1.0, -0.5, -1.0, 0.25, 0.75, 1.25}
	result := make([]float64, len(input))

	TanF64(input, result)

	for i := range input {
		expected := math.Tan(input[i])
		relErr := math.Abs(result[i]-expected) / (math.Abs(expected) + 1e-15)
		if relErr > 1e-6 {
			t.Errorf("TanF64[%d]: got %v, want %v (relative error: %v)", i, result[i], expected, relErr)
		}
	}
}

func TestAtanF64(t *testing.T) {
	input := []float64{0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0, 10.0}
	result := make([]float64, len(input))

	AtanF64(input, result)

	for i := range input {
		expected := math.Atan(input[i])
		absErr := math.Abs(result[i] - expected)
		if absErr > 1e-6 {
			t.Errorf("AtanF64[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestAtan2F64(t *testing.T) {
	y := []float64{1, 1, -1, -1, 0, 1, 0, -1}
	x := []float64{1, -1, 1, -1, 1, 0, -1, 0}
	result := make([]float64, len(y))

	Atan2F64(y, x, result)

	for i := range y {
		expected := math.Atan2(y[i], x[i])
		absErr := math.Abs(result[i] - expected)
		if absErr > 1e-6 {
			t.Errorf("Atan2F64[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestPowF64(t *testing.T) {
	base := []float64{2, 2, 10, 10, 3, 4, 2.5, 1.5}
	exp := []float64{0, 1, 2, 0.5, 3, 0.5, 2, 3}
	result := make([]float64, len(base))

	PowF64(base, exp, result)

	for i := range base {
		expected := math.Pow(base[i], exp[i])
		relErr := math.Abs(result[i]-expected) / (math.Abs(expected) + 1e-15)
		if relErr > 1e-3 {
			t.Errorf("PowF64[%d]: got %v, want %v (relative error: %v)", i, result[i], expected, relErr)
		}
	}
}

func TestErfF64(t *testing.T) {
	input := []float64{0, 0.5, 1.0, 2.0, -0.5, -1.0, 0.25, 1.5}
	result := make([]float64, len(input))

	ErfF64(input, result)

	for i := range input {
		expected := math.Erf(input[i])
		absErr := math.Abs(result[i] - expected)
		if absErr > 1e-3 {
			t.Errorf("ErfF64[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestLog10F64(t *testing.T) {
	input := []float64{1, 10, 100, 1000, 0.1, 0.01, 2, 5}
	result := make([]float64, len(input))

	Log10F64(input, result)

	for i := range input {
		expected := math.Log10(input[i])
		absErr := math.Abs(result[i] - expected)
		if absErr > 1e-5 { // sqrt(2) range reduction achieves ~1e-5 accuracy
			t.Errorf("Log10F64[%d]: got %v, want %v (error: %v)", i, result[i], expected, absErr)
		}
	}
}

func TestExp10F64(t *testing.T) {
	input := []float64{0, 1, 2, -1, -2, 0.5, 1.5, 3}
	result := make([]float64, len(input))

	Exp10F64(input, result)

	for i := range input {
		expected := math.Pow(10, input[i])
		relErr := math.Abs(result[i]-expected) / (math.Abs(expected) + 1e-15)
		if relErr > 1e-6 {
			t.Errorf("Exp10F64[%d]: got %v, want %v (relative error: %v)", i, result[i], expected, relErr)
		}
	}
}

func TestSinCosF64(t *testing.T) {
	input := []float64{0, 0.5, 1.0, 1.5, 2.0, 3.14159265, -1.0, -2.0}
	sinResult := make([]float64, len(input))
	cosResult := make([]float64, len(input))

	SinCosF64(input, sinResult, cosResult)

	for i := range input {
		expectedSin := math.Sin(input[i])
		expectedCos := math.Cos(input[i])

		sinErr := math.Abs(sinResult[i] - expectedSin)
		cosErr := math.Abs(cosResult[i] - expectedCos)

		if sinErr > 1e-6 {
			t.Errorf("SinCosF64[%d] sin: got %v, want %v (error: %v)", i, sinResult[i], expectedSin, sinErr)
		}
		if cosErr > 1e-6 {
			t.Errorf("SinCosF64[%d] cos: got %v, want %v (error: %v)", i, cosResult[i], expectedCos, cosErr)
		}
	}
}

// F64 Additional Benchmarks

func BenchmarkTanF64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i) * 0.01
	}

	for b.Loop() {
		TanF64(input, result)
	}
}

func BenchmarkAtanF64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i-512) * 0.01
	}

	for b.Loop() {
		AtanF64(input, result)
	}
}

func BenchmarkPowF64_NEON(b *testing.B) {
	n := 1024
	base := make([]float64, n)
	exp := make([]float64, n)
	result := make([]float64, n)
	for i := range base {
		base[i] = float64(i%10) + 1
		exp[i] = float64(i%5) + 0.5
	}

	for b.Loop() {
		PowF64(base, exp, result)
	}
}

func BenchmarkLog10F64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i+1) * 0.1
	}

	for b.Loop() {
		Log10F64(input, result)
	}
}

func BenchmarkExp10F64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	result := make([]float64, n)
	for i := range input {
		input[i] = float64(i%6) - 2
	}

	for b.Loop() {
		Exp10F64(input, result)
	}
}

func BenchmarkSinCosF64_NEON(b *testing.B) {
	n := 1024
	input := make([]float64, n)
	sinResult := make([]float64, n)
	cosResult := make([]float64, n)
	for i := range input {
		input[i] = float64(i) * 0.01
	}

	for b.Loop() {
		SinCosF64(input, sinResult, cosResult)
	}
}

// F64 Core Arithmetic Tests (Phase 2 & 3)

// Test non-aligned sizes (scalar remainder)
func TestF64NonAlignedSizes(t *testing.T) {
	// Test SqrtF64 with 9 elements (not multiple of 2)
	input := []float64{1, 4, 9, 16, 25, 36, 49, 64, 81}
	result := make([]float64, 9)
	SqrtF64(input, result)
	for i := range result {
		expected := math.Sqrt(input[i])
		if result[i] != expected {
			t.Errorf("SqrtF64 (9 elements)[%d]: got %v, want %v", i, result[i], expected)
		}
	}

	// Test ReduceMinF64 with 11 elements
	input2 := []float64{5, 2, 8, 1, 7, 3, 9, 4, 6, 0.5, 10}
	result2 := ReduceMinF64(input2)
	if result2 != 0.5 {
		t.Errorf("ReduceMinF64 (11 elements): got %v, want 0.5", result2)
	}

	// Test ReduceMaxF64 with 11 elements
	result3 := ReduceMaxF64(input2)
	if result3 != 10 {
		t.Errorf("ReduceMaxF64 (11 elements): got %v, want 10", result3)
	}

	// Test SubF64 with 5 elements
	a := []float64{10, 20, 30, 40, 50}
	b := []float64{1, 2, 3, 4, 5}
	result4 := make([]float64, 5)
	SubF64(a, b, result4)
	for i := range result4 {
		expected := a[i] - b[i]
		if result4[i] != expected {
			t.Errorf("SubF64 (5 elements)[%d]: got %v, want %v", i, result4[i], expected)
		}
	}
}

func TestSubF64(t *testing.T) {
	a := []float64{5.0, 10.0, 3.0, 8.0, 1.0, 7.0, 4.0, 9.0}
	b := []float64{2.0, 3.0, 1.0, 4.0, 0.5, 2.5, 1.5, 3.5}
	result := make([]float64, len(a))
	SubF64(a, b, result)
	for i := range result {
		expected := a[i] - b[i]
		if result[i] != expected {
			t.Errorf("SubF64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestDivF64(t *testing.T) {
	a := []float64{10.0, 20.0, 9.0, 16.0, 5.0, 15.0, 8.0, 12.0}
	b := []float64{2.0, 4.0, 3.0, 4.0, 2.5, 3.0, 2.0, 4.0}
	result := make([]float64, len(a))
	DivF64(a, b, result)
	for i := range result {
		expected := a[i] / b[i]
		if result[i] != expected {
			t.Errorf("DivF64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestMinF64(t *testing.T) {
	a := []float64{5.0, 2.0, 8.0, 1.0, 7.0, 3.0, 9.0, 4.0}
	b := []float64{3.0, 4.0, 6.0, 2.0, 5.0, 8.0, 7.0, 6.0}
	result := make([]float64, len(a))
	MinF64(a, b, result)
	for i := range result {
		expected := a[i]
		if b[i] < expected {
			expected = b[i]
		}
		if result[i] != expected {
			t.Errorf("MinF64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestMaxF64(t *testing.T) {
	a := []float64{5.0, 2.0, 8.0, 1.0, 7.0, 3.0, 9.0, 4.0}
	b := []float64{3.0, 4.0, 6.0, 2.0, 5.0, 8.0, 7.0, 6.0}
	result := make([]float64, len(a))
	MaxF64(a, b, result)
	for i := range result {
		expected := a[i]
		if b[i] > expected {
			expected = b[i]
		}
		if result[i] != expected {
			t.Errorf("MaxF64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestSqrtF64(t *testing.T) {
	input := []float64{1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0}
	result := make([]float64, len(input))
	SqrtF64(input, result)
	for i := range result {
		expected := math.Sqrt(input[i])
		if result[i] != expected {
			t.Errorf("SqrtF64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestAbsF64(t *testing.T) {
	input := []float64{-5.0, 3.0, -8.0, 0.0, -1.5, 2.5, -7.0, 4.0}
	result := make([]float64, len(input))
	AbsF64(input, result)
	for i := range result {
		expected := math.Abs(input[i])
		if result[i] != expected {
			t.Errorf("AbsF64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestNegF64(t *testing.T) {
	input := []float64{5.0, -3.0, 8.0, 0.0, 1.5, -2.5, 7.0, -4.0}
	result := make([]float64, len(input))
	NegF64(input, result)
	for i := range result {
		expected := -input[i]
		if result[i] != expected {
			t.Errorf("NegF64[%d]: got %v, want %v", i, result[i], expected)
		}
	}
}

func TestReduceMinF64(t *testing.T) {
	input := []float64{5.0, 2.0, 8.0, 1.0, 7.0, 3.0, 9.0, 4.0}
	result := ReduceMinF64(input)
	expected := 1.0
	if result != expected {
		t.Errorf("ReduceMinF64: got %v, want %v", result, expected)
	}
}

func TestReduceMaxF64(t *testing.T) {
	input := []float64{5.0, 2.0, 8.0, 1.0, 7.0, 3.0, 9.0, 4.0}
	result := ReduceMaxF64(input)
	expected := 9.0
	if result != expected {
		t.Errorf("ReduceMaxF64: got %v, want %v", result, expected)
	}
}

// F64 Core Arithmetic Benchmarks

func BenchmarkSubF64_NEON(b *testing.B) {
	a := make([]float64, 1024)
	bv := make([]float64, 1024)
	result := make([]float64, 1024)
	for i := range a {
		a[i] = float64(i)
		bv[i] = float64(i) * 0.5
	}

	for b.Loop() {
		SubF64(a, bv, result)
	}
}

func BenchmarkDivF64_NEON(b *testing.B) {
	a := make([]float64, 1024)
	bv := make([]float64, 1024)
	result := make([]float64, 1024)
	for i := range a {
		a[i] = float64(i + 1)
		bv[i] = float64(i+1) * 0.5
	}

	for b.Loop() {
		DivF64(a, bv, result)
	}
}

func BenchmarkMinF64_NEON(b *testing.B) {
	a := make([]float64, 1024)
	bv := make([]float64, 1024)
	result := make([]float64, 1024)
	for i := range a {
		a[i] = float64(i)
		bv[i] = float64(1024 - i)
	}

	for b.Loop() {
		MinF64(a, bv, result)
	}
}

func BenchmarkMaxF64_NEON(b *testing.B) {
	a := make([]float64, 1024)
	bv := make([]float64, 1024)
	result := make([]float64, 1024)
	for i := range a {
		a[i] = float64(i)
		bv[i] = float64(1024 - i)
	}

	for b.Loop() {
		MaxF64(a, bv, result)
	}
}

func BenchmarkSqrtF64_NEON(b *testing.B) {
	input := make([]float64, 1024)
	result := make([]float64, 1024)
	for i := range input {
		input[i] = float64(i + 1)
	}

	for b.Loop() {
		SqrtF64(input, result)
	}
}

func BenchmarkAbsF64_NEON(b *testing.B) {
	input := make([]float64, 1024)
	result := make([]float64, 1024)
	for i := range input {
		input[i] = float64(i) - 512
	}

	for b.Loop() {
		AbsF64(input, result)
	}
}

func BenchmarkNegF64_NEON(b *testing.B) {
	input := make([]float64, 1024)
	result := make([]float64, 1024)
	for i := range input {
		input[i] = float64(i)
	}

	for b.Loop() {
		NegF64(input, result)
	}
}

func BenchmarkReduceMinF64_NEON(b *testing.B) {
	input := make([]float64, 1024)
	for i := range input {
		input[i] = float64(i)
	}

	for b.Loop() {
		ReduceMinF64(input)
	}
}

func BenchmarkReduceMaxF64_NEON(b *testing.B) {
	input := make([]float64, 1024)
	for i := range input {
		input[i] = float64(i)
	}

	for b.Loop() {
		ReduceMaxF64(input)
	}
}

// ============================================================================
// Unsigned Integer Vector Tests
// ============================================================================

func TestUint8x16_Saturating(t *testing.T) {
	// Test saturating add at boundary
	a := BroadcastUint8x16(250)
	b := BroadcastUint8x16(10)

	// Saturating add should clamp at 255
	sat := a.AddSaturated(b)
	for i := range 16 {
		if sat.Get(i) != 255 {
			t.Errorf("AddSaturated[%d]: got %d, want 255", i, sat.Get(i))
		}
	}

	// Saturating sub should clamp at 0
	c := BroadcastUint8x16(5)
	d := BroadcastUint8x16(10)
	satSub := c.SubSaturated(d)
	for i := range 16 {
		if satSub.Get(i) != 0 {
			t.Errorf("SubSaturated[%d]: got %d, want 0", i, satSub.Get(i))
		}
	}
}

func TestUint8x16_MinMax(t *testing.T) {
	a := LoadUint8x16([]uint8{1, 100, 200, 50, 10, 255, 0, 128, 1, 100, 200, 50, 10, 255, 0, 128})
	b := LoadUint8x16([]uint8{2, 50, 100, 100, 20, 100, 1, 64, 2, 50, 100, 100, 20, 100, 1, 64})

	minResult := a.Min(b)
	maxResult := a.Max(b)

	expectedMin := []uint8{1, 50, 100, 50, 10, 100, 0, 64, 1, 50, 100, 50, 10, 100, 0, 64}
	expectedMax := []uint8{2, 100, 200, 100, 20, 255, 1, 128, 2, 100, 200, 100, 20, 255, 1, 128}

	for i := range 16 {
		if minResult.Get(i) != expectedMin[i] {
			t.Errorf("Min[%d]: got %d, want %d", i, minResult.Get(i), expectedMin[i])
		}
		if maxResult.Get(i) != expectedMax[i] {
			t.Errorf("Max[%d]: got %d, want %d", i, maxResult.Get(i), expectedMax[i])
		}
	}
}

func TestUint8x16_Comparisons(t *testing.T) {
	a := LoadUint8x16([]uint8{10, 200, 100, 0, 10, 200, 100, 0, 10, 200, 100, 0, 10, 200, 100, 0})
	b := LoadUint8x16([]uint8{20, 100, 100, 1, 20, 100, 100, 1, 20, 100, 100, 1, 20, 100, 100, 1})

	lt := a.LessThan(b)
	gt := a.GreaterThan(b)
	eq := a.Equal(b)

	// LessThan: 10<20=true, 200<100=false, 100<100=false, 0<1=true
	if lt.Get(0) != 0xFF || lt.Get(1) != 0 || lt.Get(2) != 0 || lt.Get(3) != 0xFF {
		t.Errorf("LessThan: unexpected results at first 4 lanes")
	}

	// GreaterThan: 10>20=false, 200>100=true, 100>100=false, 0>1=false
	if gt.Get(0) != 0 || gt.Get(1) != 0xFF || gt.Get(2) != 0 || gt.Get(3) != 0 {
		t.Errorf("GreaterThan: unexpected results at first 4 lanes")
	}

	// Equal: 10==20=false, 200==100=false, 100==100=true, 0==1=false
	if eq.Get(0) != 0 || eq.Get(1) != 0 || eq.Get(2) != 0xFF || eq.Get(3) != 0 {
		t.Errorf("Equal: unexpected results at first 4 lanes")
	}
}

func TestUint32x4_Saturating(t *testing.T) {
	// Test saturating add at boundary
	a := BroadcastUint32x4(0xFFFFFFFF - 5)
	b := BroadcastUint32x4(10)

	sat := a.AddSaturated(b)
	for i := range 4 {
		if sat.Get(i) != 0xFFFFFFFF {
			t.Errorf("AddSaturated[%d]: got %d, want %d", i, sat.Get(i), uint32(0xFFFFFFFF))
		}
	}

	// Saturating sub should clamp at 0
	c := BroadcastUint32x4(5)
	d := BroadcastUint32x4(10)
	satSub := c.SubSaturated(d)
	for i := range 4 {
		if satSub.Get(i) != 0 {
			t.Errorf("SubSaturated[%d]: got %d, want 0", i, satSub.Get(i))
		}
	}
}

func TestUint32x4_MinMax(t *testing.T) {
	// Test with values that differ in signed vs unsigned interpretation
	// 0x80000000 is MAX_INT32+1 in unsigned, but MIN_INT32 in signed
	a := LoadUint32x4([]uint32{100, 0x80000000, 0xFFFFFFFF, 0})
	b := LoadUint32x4([]uint32{200, 0x7FFFFFFF, 0xFFFFFFFE, 1})

	minResult := a.Min(b)
	maxResult := a.Max(b)

	// Unsigned min: 100, 0x7FFFFFFF, 0xFFFFFFFE, 0
	expectedMin := []uint32{100, 0x7FFFFFFF, 0xFFFFFFFE, 0}
	// Unsigned max: 200, 0x80000000, 0xFFFFFFFF, 1
	expectedMax := []uint32{200, 0x80000000, 0xFFFFFFFF, 1}

	for i := range 4 {
		if minResult.Get(i) != expectedMin[i] {
			t.Errorf("Min[%d]: got %d, want %d", i, minResult.Get(i), expectedMin[i])
		}
		if maxResult.Get(i) != expectedMax[i] {
			t.Errorf("Max[%d]: got %d, want %d", i, maxResult.Get(i), expectedMax[i])
		}
	}
}

func TestUint32x4_Comparisons(t *testing.T) {
	// Test unsigned comparisons with values that differ in signed vs unsigned
	a := LoadUint32x4([]uint32{10, 0x80000000, 100, 0})
	b := LoadUint32x4([]uint32{20, 0x7FFFFFFF, 100, 1})

	lt := a.LessThan(b)
	gt := a.GreaterThan(b)
	eq := a.Equal(b)

	// Unsigned LessThan: 10<20=true, 0x80000000<0x7FFFFFFF=false (unsigned!), 100<100=false, 0<1=true
	if !lt.GetBit(0) || lt.GetBit(1) || lt.GetBit(2) || !lt.GetBit(3) {
		t.Errorf("LessThan: unexpected results: [%v, %v, %v, %v]",
			lt.GetBit(0), lt.GetBit(1), lt.GetBit(2), lt.GetBit(3))
	}

	// Unsigned GreaterThan: 10>20=false, 0x80000000>0x7FFFFFFF=true (unsigned!), 100>100=false, 0>1=false
	if gt.GetBit(0) || !gt.GetBit(1) || gt.GetBit(2) || gt.GetBit(3) {
		t.Errorf("GreaterThan: unexpected results: [%v, %v, %v, %v]",
			gt.GetBit(0), gt.GetBit(1), gt.GetBit(2), gt.GetBit(3))
	}

	// Equal: 10==20=false, 0x80000000==0x7FFFFFFF=false, 100==100=true, 0==1=false
	if eq.GetBit(0) || eq.GetBit(1) || !eq.GetBit(2) || eq.GetBit(3) {
		t.Errorf("Equal: unexpected results: [%v, %v, %v, %v]",
			eq.GetBit(0), eq.GetBit(1), eq.GetBit(2), eq.GetBit(3))
	}
}

func TestUint64x2_Saturating(t *testing.T) {
	// Test saturating add at boundary
	a := BroadcastUint64x2(0xFFFFFFFFFFFFFFFF - 5)
	b := BroadcastUint64x2(10)

	sat := a.AddSaturated(b)
	for i := range 2 {
		if sat.Get(i) != 0xFFFFFFFFFFFFFFFF {
			t.Errorf("AddSaturated[%d]: got %d, want max uint64", i, sat.Get(i))
		}
	}

	// Saturating sub should clamp at 0
	c := BroadcastUint64x2(5)
	d := BroadcastUint64x2(10)
	satSub := c.SubSaturated(d)
	for i := range 2 {
		if satSub.Get(i) != 0 {
			t.Errorf("SubSaturated[%d]: got %d, want 0", i, satSub.Get(i))
		}
	}
}

func TestUint64x2_MinMax(t *testing.T) {
	// Test with values that differ in signed vs unsigned interpretation
	a := LoadUint64x2([]uint64{0x8000000000000000, 100})
	b := LoadUint64x2([]uint64{0x7FFFFFFFFFFFFFFF, 200})

	minResult := a.Min(b)
	maxResult := a.Max(b)

	// Unsigned min: 0x7FFFFFFFFFFFFFFF, 100
	if minResult.Get(0) != 0x7FFFFFFFFFFFFFFF {
		t.Errorf("Min[0]: got %x, want %x", minResult.Get(0), uint64(0x7FFFFFFFFFFFFFFF))
	}
	if minResult.Get(1) != 100 {
		t.Errorf("Min[1]: got %d, want 100", minResult.Get(1))
	}

	// Unsigned max: 0x8000000000000000, 200
	if maxResult.Get(0) != 0x8000000000000000 {
		t.Errorf("Max[0]: got %x, want %x", maxResult.Get(0), uint64(0x8000000000000000))
	}
	if maxResult.Get(1) != 200 {
		t.Errorf("Max[1]: got %d, want 200", maxResult.Get(1))
	}
}

func TestUint64x2_Comparisons(t *testing.T) {
	// Test unsigned comparisons
	a := LoadUint64x2([]uint64{0x8000000000000000, 100})
	b := LoadUint64x2([]uint64{0x7FFFFFFFFFFFFFFF, 100})

	lt := a.LessThan(b)
	gt := a.GreaterThan(b)
	eq := a.Equal(b)

	// Unsigned: 0x8000000000000000 > 0x7FFFFFFFFFFFFFFF
	if lt.GetBit(0) {
		t.Error("LessThan[0]: expected false (0x8000... > 0x7FFF... in unsigned)")
	}
	if !gt.GetBit(0) {
		t.Error("GreaterThan[0]: expected true (0x8000... > 0x7FFF... in unsigned)")
	}
	if eq.GetBit(0) {
		t.Error("Equal[0]: expected false")
	}

	// 100 == 100
	if lt.GetBit(1) {
		t.Error("LessThan[1]: expected false (100 == 100)")
	}
	if gt.GetBit(1) {
		t.Error("GreaterThan[1]: expected false (100 == 100)")
	}
	if !eq.GetBit(1) {
		t.Error("Equal[1]: expected true (100 == 100)")
	}
}

func TestUint16x8_Saturating(t *testing.T) {
	// Test saturating add at boundary
	a := BroadcastUint16x8(65530)
	b := BroadcastUint16x8(10)

	sat := a.AddSaturated(b)
	for i := range 8 {
		if sat.Get(i) != 65535 {
			t.Errorf("AddSaturated[%d]: got %d, want 65535", i, sat.Get(i))
		}
	}

	// Saturating sub should clamp at 0
	c := BroadcastUint16x8(5)
	d := BroadcastUint16x8(10)
	satSub := c.SubSaturated(d)
	for i := range 8 {
		if satSub.Get(i) != 0 {
			t.Errorf("SubSaturated[%d]: got %d, want 0", i, satSub.Get(i))
		}
	}
}

func TestUint32x4_Arithmetic(t *testing.T) {
	a := LoadUint32x4([]uint32{10, 20, 30, 40})
	b := LoadUint32x4([]uint32{5, 10, 15, 20})

	// Add
	add := a.Add(b)
	for i, expected := range []uint32{15, 30, 45, 60} {
		if add.Get(i) != expected {
			t.Errorf("Add[%d]: got %d, want %d", i, add.Get(i), expected)
		}
	}

	// Sub
	sub := a.Sub(b)
	for i, expected := range []uint32{5, 10, 15, 20} {
		if sub.Get(i) != expected {
			t.Errorf("Sub[%d]: got %d, want %d", i, sub.Get(i), expected)
		}
	}

	// Mul
	mul := a.Mul(b)
	for i, expected := range []uint32{50, 200, 450, 800} {
		if mul.Get(i) != expected {
			t.Errorf("Mul[%d]: got %d, want %d", i, mul.Get(i), expected)
		}
	}
}

func TestUint32x4_ReduceSum(t *testing.T) {
	a := LoadUint32x4([]uint32{100, 200, 300, 400})
	sum := a.ReduceSum()
	if sum != 1000 {
		t.Errorf("ReduceSum: got %d, want 1000", sum)
	}
}

func TestUint32x4_Shifts(t *testing.T) {
	a := LoadUint32x4([]uint32{0x80000000, 0xFF, 0x12345678, 1})

	// Shift left by 1
	left := a.ShiftAllLeft(1)
	expectedLeft := []uint32{0, 0x1FE, 0x2468ACF0, 2}
	for i, expected := range expectedLeft {
		if left.Get(i) != expected {
			t.Errorf("ShiftAllLeft[%d]: got %x, want %x", i, left.Get(i), expected)
		}
	}

	// Shift right by 4 (logical for unsigned)
	right := a.ShiftAllRight(4)
	expectedRight := []uint32{0x08000000, 0x0F, 0x01234567, 0}
	for i, expected := range expectedRight {
		if right.Get(i) != expected {
			t.Errorf("ShiftAllRight[%d]: got %x, want %x", i, right.Get(i), expected)
		}
	}
}
