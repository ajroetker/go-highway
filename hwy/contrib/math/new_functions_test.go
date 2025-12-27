//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
	"testing"
)

func TestLog2_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{"log2(1) = 0", 1.0, 0.0},
		{"log2(2) = 1", 2.0, 1.0},
		{"log2(4) = 2", 4.0, 2.0},
		{"log2(8) = 3", 8.0, 3.0},
		{"log2(0.5) = -1", 0.5, -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.input)
			result := Log2_AVX2_F32x8(x)
			got := result.ExtractLane(0)

			if stdmath.Abs(float64(got-tt.want)) > 1e-5 {
				t.Errorf("Log2(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestLog10_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{"log10(1) = 0", 1.0, 0.0},
		{"log10(10) = 1", 10.0, 1.0},
		{"log10(100) = 2", 100.0, 2.0},
		{"log10(0.1) = -1", 0.1, -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.input)
			result := Log10_AVX2_F32x8(x)
			got := result.ExtractLane(0)

			if stdmath.Abs(float64(got-tt.want)) > 1e-5 {
				t.Errorf("Log10(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestExp2_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{"2^0 = 1", 0.0, 1.0},
		{"2^1 = 2", 1.0, 2.0},
		{"2^2 = 4", 2.0, 4.0},
		{"2^3 = 8", 3.0, 8.0},
		{"2^-1 = 0.5", -1.0, 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.input)
			result := Exp2_AVX2_F32x8(x)
			got := result.ExtractLane(0)

			if stdmath.Abs(float64(got-tt.want)) > 1e-5 {
				t.Errorf("Exp2(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestSinh_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"sinh(0)", 0.0},
		{"sinh(1)", 1.0},
		{"sinh(-1)", -1.0},
		{"sinh(2)", 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.input)
			result := Sinh_AVX2_F32x8(x)
			got := result.ExtractLane(0)
			want := float32(stdmath.Sinh(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Sinh(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

func TestCosh_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"cosh(0) = 1", 0.0},
		{"cosh(1)", 1.0},
		{"cosh(-1)", -1.0},
		{"cosh(2)", 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.input)
			result := Cosh_AVX2_F32x8(x)
			got := result.ExtractLane(0)
			want := float32(stdmath.Cosh(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Cosh(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

// Float64 tests
// Note: Sqrt tests moved to hwy/ops_test.go since Sqrt is a core op

func TestLog2_AVX2_F64x4(t *testing.T) {
	tests := []struct {
		name  string
		input float64
		want  float64
	}{
		{"log2(1) = 0", 1.0, 0.0},
		{"log2(2) = 1", 2.0, 1.0},
		{"log2(4) = 2", 4.0, 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat64x4(tt.input)
			result := Log2_AVX2_F64x4(x)
			got := result.ExtractLane(0)

			if stdmath.Abs(got-tt.want) > 1e-10 {
				t.Errorf("Log2(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestSinh_AVX2_F64x4(t *testing.T) {
	tests := []struct {
		name  string
		input float64
	}{
		{"sinh(0)", 0.0},
		{"sinh(1)", 1.0},
		{"sinh(2)", 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat64x4(tt.input)
			result := Sinh_AVX2_F64x4(x)
			got := result.ExtractLane(0)
			want := stdmath.Sinh(tt.input)

			if stdmath.Abs(got-want) > 1e-10 {
				t.Errorf("Sinh(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

// Benchmarks

func BenchmarkLog2_AVX2_F32x8(b *testing.B) {
	x := archsimd.BroadcastFloat32x8(2.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Log2_AVX2_F32x8(x)
	}
}

func BenchmarkSinh_AVX2_F32x8(b *testing.B) {
	x := archsimd.BroadcastFloat32x8(1.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Sinh_AVX2_F32x8(x)
	}
}

// Note: Sqrt benchmarks moved to hwy/ops_test.go since Sqrt is a core op
