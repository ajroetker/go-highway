//go:build (amd64 && goexperiment.simd) || arm64

package algo

import (
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

func TestFill(t *testing.T) {
	tests := []struct {
		name  string
		size  int
		value float32
	}{
		{"single", 1, 42.0},
		{"small", 3, 3.14},
		{"vector_aligned", 8, 2.71},
		{"unaligned", 15, 1.41},
		{"large", 100, 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.size)
			Fill(dst, tt.value)

			for i := range dst {
				if dst[i] != tt.value {
					t.Errorf("Fill[%d]: got %v, want %v", i, dst[i], tt.value)
				}
			}
		})
	}
}

func TestFill_Int32(t *testing.T) {
	dst := make([]int32, 17)
	Fill(dst, int32(123))

	for i := range dst {
		if dst[i] != 123 {
			t.Errorf("Fill[%d]: got %v, want 123", i, dst[i])
		}
	}
}

func TestFill_Empty(t *testing.T) {
	var dst []float32
	Fill(dst, 1.0) // Should not panic
}

func TestCopy(t *testing.T) {
	tests := []struct {
		name string
		size int
	}{
		{"single", 1},
		{"small", 3},
		{"vector_aligned", 8},
		{"unaligned", 15},
		{"large", 100},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := make([]float32, tt.size)
			dst := make([]float32, tt.size)

			for i := range src {
				src[i] = float32(i) * 1.5
			}

			copied := Copy(src, dst)

			if copied != tt.size {
				t.Errorf("Copy returned %d, want %d", copied, tt.size)
			}

			for i := range src {
				if dst[i] != src[i] {
					t.Errorf("Copy[%d]: got %v, want %v", i, dst[i], src[i])
				}
			}
		})
	}
}

func TestCopy_DifferentSizes(t *testing.T) {
	// dst smaller than src
	src := make([]float32, 20)
	dst := make([]float32, 10)

	for i := range src {
		src[i] = float32(i)
	}

	copied := Copy(src, dst)

	if copied != 10 {
		t.Errorf("Copy returned %d, want 10", copied)
	}

	for i := range 10 {
		if dst[i] != float32(i) {
			t.Errorf("Copy[%d]: got %v, want %v", i, dst[i], float32(i))
		}
	}

	// src smaller than dst
	src = make([]float32, 10)
	dst = make([]float32, 20)

	for i := range src {
		src[i] = float32(i)
	}

	copied = Copy(src, dst)

	if copied != 10 {
		t.Errorf("Copy returned %d, want 10", copied)
	}
}

func TestCopy_Empty(t *testing.T) {
	var src, dst []float32
	copied := Copy(src, dst)
	if copied != 0 {
		t.Errorf("Copy of empty slices returned %d, want 0", copied)
	}
}

func TestCopyIf(t *testing.T) {
	src := []float32{-2, -1, 0, 1, 2, 3, 4, 5, -3, 6, 7, -4}
	dst := make([]float32, len(src))

	// Copy only positive values
	copied := CopyIf(src, dst, func(v hwy.Vec[float32]) hwy.Mask[float32] {
		return hwy.GreaterThan(v, hwy.Zero[float32]())
	})

	// Expected: 1, 2, 3, 4, 5, 6, 7 = 7 elements
	expected := []float32{1, 2, 3, 4, 5, 6, 7}

	if copied != len(expected) {
		t.Errorf("CopyIf returned %d, want %d", copied, len(expected))
	}

	for i, want := range expected {
		if dst[i] != want {
			t.Errorf("CopyIf[%d]: got %v, want %v", i, dst[i], want)
		}
	}
}

func TestCopyIf_AllPass(t *testing.T) {
	src := []float32{1, 2, 3, 4, 5}
	dst := make([]float32, len(src))

	copied := CopyIf(src, dst, func(v hwy.Vec[float32]) hwy.Mask[float32] {
		// All positive - all pass
		return hwy.GreaterThan(v, hwy.Set(float32(-1)))
	})

	if copied != len(src) {
		t.Errorf("CopyIf returned %d, want %d", copied, len(src))
	}

	for i := range src {
		if dst[i] != src[i] {
			t.Errorf("CopyIf[%d]: got %v, want %v", i, dst[i], src[i])
		}
	}
}

func TestCopyIf_NonePass(t *testing.T) {
	src := []float32{1, 2, 3, 4, 5}
	dst := make([]float32, len(src))

	copied := CopyIf(src, dst, func(v hwy.Vec[float32]) hwy.Mask[float32] {
		// None are negative
		return hwy.LessThan(v, hwy.Zero[float32]())
	})

	if copied != 0 {
		t.Errorf("CopyIf returned %d, want 0", copied)
	}
}

func TestCopyIf_SmallDst(t *testing.T) {
	src := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	dst := make([]float32, 3) // Only space for 3 elements

	copied := CopyIf(src, dst, func(v hwy.Vec[float32]) hwy.Mask[float32] {
		return hwy.GreaterThan(v, hwy.Zero[float32]())
	})

	// Should copy at most 3 elements
	if copied > 3 {
		t.Errorf("CopyIf copied %d elements but dst only has space for 3", copied)
	}
}

// Benchmarks
// Note: benchSize is defined in transform_test.go

func BenchmarkFill(b *testing.B) {
	dst := make([]float32, benchSize)

	b.ReportAllocs()
	for b.Loop() {
		Fill(dst, 42.0)
	}
}

func BenchmarkFill_Stdlib(b *testing.B) {
	dst := make([]float32, benchSize)
	value := float32(42.0)

	b.ReportAllocs()
	for b.Loop() {
		for j := range dst {
			dst[j] = value
		}
	}
}

func BenchmarkCopy(b *testing.B) {
	src := make([]float32, benchSize)
	dst := make([]float32, benchSize)
	for i := range src {
		src[i] = float32(i)
	}

	b.ReportAllocs()
	for b.Loop() {
		Copy(src, dst)
	}
}

func BenchmarkCopy_Stdlib(b *testing.B) {
	src := make([]float32, benchSize)
	dst := make([]float32, benchSize)
	for i := range src {
		src[i] = float32(i)
	}

	b.ReportAllocs()
	for b.Loop() {
		copy(dst, src)
	}
}

func BenchmarkCopyIf(b *testing.B) {
	src := make([]float32, benchSize)
	dst := make([]float32, benchSize)
	for i := range src {
		src[i] = float32(i) - float32(benchSize/2) // Half positive, half negative
	}

	b.ReportAllocs()
	for b.Loop() {
		CopyIf(src, dst, func(v hwy.Vec[float32]) hwy.Mask[float32] {
			return hwy.GreaterThan(v, hwy.Zero[float32]())
		})
	}
}

func BenchmarkCopyIf_Stdlib(b *testing.B) {
	src := make([]float32, benchSize)
	dst := make([]float32, benchSize)
	for i := range src {
		src[i] = float32(i) - float32(benchSize/2)
	}

	b.ReportAllocs()
	for b.Loop() {
		j := 0
		for _, v := range src {
			if v > 0 {
				dst[j] = v
				j++
			}
		}
	}
}

func BenchmarkCopyIfP(b *testing.B) {
	src := make([]float32, benchSize)
	dst := make([]float32, benchSize)
	for i := range src {
		src[i] = float32(i) - float32(benchSize/2) // Half positive, half negative
	}

	b.ReportAllocs()
	for b.Loop() {
		CopyIfP(src, dst, GreaterThan[float32]{Threshold: 0})
	}
}
