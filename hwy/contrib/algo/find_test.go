//go:build (amd64 && goexperiment.simd) || arm64

package algo

import (
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

func TestFind(t *testing.T) {
	tests := []struct {
		name   string
		slice  []float32
		value  float32
		expect int
	}{
		{"first", []float32{1, 2, 3, 4, 5}, 1, 0},
		{"middle", []float32{1, 2, 3, 4, 5}, 3, 2},
		{"last", []float32{1, 2, 3, 4, 5}, 5, 4},
		{"not_found", []float32{1, 2, 3, 4, 5}, 6, -1},
		{"empty", []float32{}, 1, -1},
		{"single_found", []float32{42}, 42, 0},
		{"single_not_found", []float32{42}, 1, -1},
		{"duplicates", []float32{1, 2, 3, 2, 5}, 2, 1}, // Returns first occurrence
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Find(tt.slice, tt.value)
			if got != tt.expect {
				t.Errorf("Find(%v, %v) = %d, want %d", tt.slice, tt.value, got, tt.expect)
			}
		})
	}
}

func TestFind_Large(t *testing.T) {
	// Test with array larger than vector width
	sizes := []int{15, 16, 17, 31, 32, 33, 100}

	for _, size := range sizes {
		t.Run("size_"+string(rune('0'+size%10)), func(t *testing.T) {
			slice := make([]float32, size)
			for i := range slice {
				slice[i] = float32(i)
			}

			// Find element in tail
			got := Find(slice, float32(size-1))
			if got != size-1 {
				t.Errorf("Find last element: got %d, want %d", got, size-1)
			}

			// Find element in middle
			mid := size / 2
			got = Find(slice, float32(mid))
			if got != mid {
				t.Errorf("Find middle element: got %d, want %d", got, mid)
			}
		})
	}
}

func TestFind_Int32(t *testing.T) {
	slice := []int32{10, 20, 30, 40, 50}
	got := Find(slice, int32(30))
	if got != 2 {
		t.Errorf("Find int32: got %d, want 2", got)
	}

	got = Find(slice, int32(99))
	if got != -1 {
		t.Errorf("Find int32 not found: got %d, want -1", got)
	}
}

func TestFindIf(t *testing.T) {
	slice := []float32{-5, -3, -1, 0, 1, 3, 5}

	// Find first positive
	got := FindIf(slice, func(v hwy.Vec[float32]) hwy.Mask[float32] {
		return hwy.GreaterThan(v, hwy.Zero[float32]())
	})
	if got != 4 { // Index of 1
		t.Errorf("FindIf first positive: got %d, want 4", got)
	}

	// Find first greater than 2
	got = FindIf(slice, func(v hwy.Vec[float32]) hwy.Mask[float32] {
		return hwy.GreaterThan(v, hwy.Set(float32(2)))
	})
	if got != 5 { // Index of 3
		t.Errorf("FindIf > 2: got %d, want 5", got)
	}

	// No match
	got = FindIf(slice, func(v hwy.Vec[float32]) hwy.Mask[float32] {
		return hwy.GreaterThan(v, hwy.Set(float32(100)))
	})
	if got != -1 {
		t.Errorf("FindIf no match: got %d, want -1", got)
	}
}

func TestFindIf_Empty(t *testing.T) {
	var slice []float32
	got := FindIf(slice, func(v hwy.Vec[float32]) hwy.Mask[float32] {
		return hwy.GreaterThan(v, hwy.Zero[float32]())
	})
	if got != -1 {
		t.Errorf("FindIf empty: got %d, want -1", got)
	}
}

func TestCount(t *testing.T) {
	tests := []struct {
		name   string
		slice  []float32
		value  float32
		expect int
	}{
		{"single", []float32{1, 2, 3, 4, 5}, 3, 1},
		{"multiple", []float32{1, 2, 2, 2, 5}, 2, 3},
		{"none", []float32{1, 2, 3, 4, 5}, 6, 0},
		{"all", []float32{7, 7, 7, 7}, 7, 4},
		{"empty", []float32{}, 1, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Count(tt.slice, tt.value)
			if got != tt.expect {
				t.Errorf("Count(%v, %v) = %d, want %d", tt.slice, tt.value, got, tt.expect)
			}
		})
	}
}

func TestCount_Large(t *testing.T) {
	// Create array with known count
	slice := make([]float32, 100)
	for i := range slice {
		if i%3 == 0 {
			slice[i] = 42
		} else {
			slice[i] = float32(i)
		}
	}

	// Should find 34 occurrences of 42 (indices 0, 3, 6, ..., 99)
	got := Count(slice, 42)
	expected := (99 / 3) + 1 // 34
	if got != expected {
		t.Errorf("Count large: got %d, want %d", got, expected)
	}
}

func TestCountIf(t *testing.T) {
	slice := []float32{-5, -3, -1, 0, 1, 3, 5, 7, 9}

	// Count positive
	got := CountIf(slice, func(v hwy.Vec[float32]) hwy.Mask[float32] {
		return hwy.GreaterThan(v, hwy.Zero[float32]())
	})
	if got != 5 { // 1, 3, 5, 7, 9
		t.Errorf("CountIf positive: got %d, want 5", got)
	}

	// Count negative
	got = CountIf(slice, func(v hwy.Vec[float32]) hwy.Mask[float32] {
		return hwy.LessThan(v, hwy.Zero[float32]())
	})
	if got != 3 { // -5, -3, -1
		t.Errorf("CountIf negative: got %d, want 3", got)
	}
}

func TestContains(t *testing.T) {
	slice := []float32{1, 2, 3, 4, 5}

	if !Contains(slice, 3) {
		t.Error("Contains should find 3")
	}

	if Contains(slice, 6) {
		t.Error("Contains should not find 6")
	}
}

func TestAll(t *testing.T) {
	positive := []float32{1, 2, 3, 4, 5}
	mixed := []float32{-1, 2, 3, 4, 5}

	isPositive := func(v hwy.Vec[float32]) hwy.Mask[float32] {
		return hwy.GreaterThan(v, hwy.Zero[float32]())
	}

	if !All(positive, isPositive) {
		t.Error("All should return true for all positive")
	}

	if All(mixed, isPositive) {
		t.Error("All should return false for mixed")
	}

	// Empty slice should return true
	if !All([]float32{}, isPositive) {
		t.Error("All on empty should return true")
	}
}

func TestAny(t *testing.T) {
	positive := []float32{1, 2, 3, 4, 5}
	negative := []float32{-1, -2, -3, -4, -5}
	mixed := []float32{-1, -2, 3, -4, -5}

	isPositive := func(v hwy.Vec[float32]) hwy.Mask[float32] {
		return hwy.GreaterThan(v, hwy.Zero[float32]())
	}

	if !Any(positive, isPositive) {
		t.Error("Any should return true for all positive")
	}

	if Any(negative, isPositive) {
		t.Error("Any should return false for all negative")
	}

	if !Any(mixed, isPositive) {
		t.Error("Any should return true for mixed with one positive")
	}

	// Empty slice should return false
	if Any([]float32{}, isPositive) {
		t.Error("Any on empty should return false")
	}
}

func TestNone(t *testing.T) {
	negative := []float32{-1, -2, -3, -4, -5}
	mixed := []float32{-1, 2, -3, -4, -5}

	isPositive := func(v hwy.Vec[float32]) hwy.Mask[float32] {
		return hwy.GreaterThan(v, hwy.Zero[float32]())
	}

	if !None(negative, isPositive) {
		t.Error("None should return true for all negative")
	}

	if None(mixed, isPositive) {
		t.Error("None should return false for mixed")
	}
}

// Benchmarks

func BenchmarkFind(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(i)
	}
	target := float32(benchSize - 1) // Worst case: last element

	b.ReportAllocs()
	for b.Loop() {
		Find(slice, target)
	}
}

func BenchmarkFind_Stdlib(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(i)
	}
	target := float32(benchSize - 1)

	b.ReportAllocs()
	for b.Loop() {
		for j, v := range slice {
			if v == target {
				_ = j
				break
			}
		}
	}
}

func BenchmarkFind_Early(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(i)
	}
	target := float32(10) // Best case: early element

	b.ReportAllocs()
	for b.Loop() {
		Find(slice, target)
	}
}

func BenchmarkCount(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		if i%10 == 0 {
			slice[i] = 42
		} else {
			slice[i] = float32(i)
		}
	}

	b.ReportAllocs()
	for b.Loop() {
		Count(slice, 42)
	}
}

func BenchmarkCount_Stdlib(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		if i%10 == 0 {
			slice[i] = 42
		} else {
			slice[i] = float32(i)
		}
	}
	target := float32(42)

	b.ReportAllocs()
	for b.Loop() {
		count := 0
		for _, v := range slice {
			if v == target {
				count++
			}
		}
		_ = count
	}
}

func BenchmarkCountIf(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(i) - float32(benchSize/2)
	}

	b.ReportAllocs()
	for b.Loop() {
		CountIf(slice, func(v hwy.Vec[float32]) hwy.Mask[float32] {
			return hwy.GreaterThan(v, hwy.Zero[float32]())
		})
	}
}

func BenchmarkAll(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(i + 1) // All positive
	}

	b.ReportAllocs()
	for b.Loop() {
		All(slice, func(v hwy.Vec[float32]) hwy.Mask[float32] {
			return hwy.GreaterThan(v, hwy.Zero[float32]())
		})
	}
}

func BenchmarkAny(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(-(i + 1)) // All negative
	}
	slice[benchSize-1] = 1 // One positive at the end

	b.ReportAllocs()
	for b.Loop() {
		Any(slice, func(v hwy.Vec[float32]) hwy.Mask[float32] {
			return hwy.GreaterThan(v, hwy.Zero[float32]())
		})
	}
}

// =============================================================================
// Predicate-based API tests
// =============================================================================

func TestFindIfP(t *testing.T) {
	slice := []float32{-5, -3, -1, 0, 1, 3, 5}

	// Find first positive using predicate type
	got := FindIfP(slice, GreaterThan[float32]{Threshold: 0})
	if got != 4 { // Index of 1
		t.Errorf("FindIfP first positive: got %d, want 4", got)
	}

	// Find first greater than 2
	got = FindIfP(slice, GreaterThan[float32]{Threshold: 2})
	if got != 5 { // Index of 3
		t.Errorf("FindIfP > 2: got %d, want 5", got)
	}

	// Find first less than -2
	got = FindIfP(slice, LessThan[float32]{Threshold: -2})
	if got != 0 { // Index of -5
		t.Errorf("FindIfP < -2: got %d, want 0", got)
	}

	// No match
	got = FindIfP(slice, GreaterThan[float32]{Threshold: 100})
	if got != -1 {
		t.Errorf("FindIfP no match: got %d, want -1", got)
	}
}

func TestCountIfP(t *testing.T) {
	slice := []float32{-5, -3, -1, 0, 1, 3, 5, 7, 9}

	// Count positive using predicate type
	got := CountIfP(slice, GreaterThan[float32]{Threshold: 0})
	if got != 5 { // 1, 3, 5, 7, 9
		t.Errorf("CountIfP positive: got %d, want 5", got)
	}

	// Count in range [-2, 2]
	got = CountIfP(slice, InRange[float32]{Min: -2, Max: 2})
	if got != 3 { // -1, 0, 1
		t.Errorf("CountIfP in range: got %d, want 3", got)
	}

	// Count zeros
	got = CountIfP(slice, IsZero[float32]{})
	if got != 1 { // just 0
		t.Errorf("CountIfP zero: got %d, want 1", got)
	}
}

func TestAllP(t *testing.T) {
	positive := []float32{1, 2, 3, 4, 5}
	mixed := []float32{-1, 2, 3, 4, 5}

	// All positive
	if !AllP(positive, GreaterThan[float32]{Threshold: 0}) {
		t.Error("AllP should return true for all positive")
	}

	// Not all positive
	if AllP(mixed, GreaterThan[float32]{Threshold: 0}) {
		t.Error("AllP should return false for mixed")
	}

	// All greater than -10
	if !AllP(mixed, GreaterThan[float32]{Threshold: -10}) {
		t.Error("AllP should return true for all > -10")
	}

	// Empty slice
	if !AllP([]float32{}, GreaterThan[float32]{Threshold: 0}) {
		t.Error("AllP on empty should return true")
	}
}

func TestAnyP(t *testing.T) {
	positive := []float32{1, 2, 3, 4, 5}
	negative := []float32{-1, -2, -3, -4, -5}
	mixed := []float32{-1, -2, 3, -4, -5}

	// Any positive
	if !AnyP(positive, GreaterThan[float32]{Threshold: 0}) {
		t.Error("AnyP should return true for all positive")
	}

	// No positive in all negative
	if AnyP(negative, GreaterThan[float32]{Threshold: 0}) {
		t.Error("AnyP should return false for all negative")
	}

	// One positive in mixed
	if !AnyP(mixed, GreaterThan[float32]{Threshold: 0}) {
		t.Error("AnyP should return true for mixed with one positive")
	}

	// Empty slice
	if AnyP([]float32{}, GreaterThan[float32]{Threshold: 0}) {
		t.Error("AnyP on empty should return false")
	}
}

func TestNoneP(t *testing.T) {
	negative := []float32{-1, -2, -3, -4, -5}
	mixed := []float32{-1, 2, -3, -4, -5}

	// None positive in all negative
	if !NoneP(negative, GreaterThan[float32]{Threshold: 0}) {
		t.Error("NoneP should return true for all negative")
	}

	// Some positive in mixed
	if NoneP(mixed, GreaterThan[float32]{Threshold: 0}) {
		t.Error("NoneP should return false for mixed")
	}
}

func TestPredicateTypes(t *testing.T) {
	slice := []float32{-5, -3, -1, 0, 1, 3, 5}

	// GreaterEqual
	got := CountIfP(slice, GreaterEqual[float32]{Threshold: 0})
	if got != 4 { // 0, 1, 3, 5
		t.Errorf("GreaterEqual >= 0: got %d, want 4", got)
	}

	// LessEqual
	got = CountIfP(slice, LessEqual[float32]{Threshold: 0})
	if got != 4 { // -5, -3, -1, 0
		t.Errorf("LessEqual <= 0: got %d, want 4", got)
	}

	// Equal
	got = CountIfP(slice, Equal[float32]{Value: 3})
	if got != 1 {
		t.Errorf("Equal 3: got %d, want 1", got)
	}

	// NotEqual
	got = CountIfP(slice, NotEqual[float32]{Value: 0})
	if got != 6 { // all except 0
		t.Errorf("NotEqual 0: got %d, want 6", got)
	}

	// OutOfRange
	got = CountIfP(slice, OutOfRange[float32]{Min: -2, Max: 2})
	if got != 4 { // -5, -3, 3, 5
		t.Errorf("OutOfRange [-2, 2]: got %d, want 4", got)
	}

	// IsNonZero
	got = CountIfP(slice, IsNonZero[float32]{})
	if got != 6 { // all except 0
		t.Errorf("IsNonZero: got %d, want 6", got)
	}

	// IsPositive
	got = CountIfP(slice, IsPositive[float32]{})
	if got != 3 { // 1, 3, 5
		t.Errorf("IsPositive: got %d, want 3", got)
	}

	// IsNegative
	got = CountIfP(slice, IsNegative[float32]{})
	if got != 3 { // -5, -3, -1
		t.Errorf("IsNegative: got %d, want 3", got)
	}
}

// Benchmarks for predicate-based API

func BenchmarkFindIfP(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(i)
	}

	pred := GreaterThan[float32]{Threshold: float32(benchSize - 1)}

	b.ReportAllocs()
	for b.Loop() {
		FindIfP(slice, pred)
	}
}

func BenchmarkCountIfP(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(i) - float32(benchSize/2)
	}

	pred := GreaterThan[float32]{Threshold: 0}

	b.ReportAllocs()
	for b.Loop() {
		CountIfP(slice, pred)
	}
}

func BenchmarkAllP(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(i + 1) // All positive
	}

	pred := GreaterThan[float32]{Threshold: 0}

	b.ReportAllocs()
	for b.Loop() {
		AllP(slice, pred)
	}
}

func BenchmarkAnyP(b *testing.B) {
	slice := make([]float32, benchSize)
	for i := range slice {
		slice[i] = float32(-(i + 1)) // All negative
	}
	slice[benchSize-1] = 1 // One positive at the end

	pred := GreaterThan[float32]{Threshold: 0}

	b.ReportAllocs()
	for b.Loop() {
		AnyP(slice, pred)
	}
}
