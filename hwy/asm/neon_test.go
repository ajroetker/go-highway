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

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
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

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
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

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
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
	for i := int64(0); i < count; i++ {
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

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
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

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
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

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
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

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
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

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
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

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		SigmoidF32(input, result)
	}
}
