//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

func TestPow_Base(t *testing.T) {
	tests := []struct {
		name string
		x    float32
		y    float32
		want float32
	}{
		{"2^3 = 8", 2.0, 3.0, 8.0},
		{"2^0 = 1", 2.0, 0.0, 1.0},
		{"2^1 = 2", 2.0, 1.0, 2.0},
		{"3^2 = 9", 3.0, 2.0, 9.0},
		{"10^2 = 100", 10.0, 2.0, 100.0},
		{"2^-1 = 0.5", 2.0, -1.0, 0.5},
		{"4^0.5 = 2", 4.0, 0.5, 2.0},
		{"8^(1/3) = 2", 8.0, 1.0 / 3.0, 2.0},
		{"1^100 = 1", 1.0, 100.0, 1.0},
		{"e^1 = e", float32(stdmath.E), 1.0, float32(stdmath.E)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := hwy.Load([]float32{tt.x, tt.x, tt.x, tt.x})
			y := hwy.Load([]float32{tt.y, tt.y, tt.y, tt.y})
			result := Pow(x, y)
			got := result.Data()[0]

			if stdmath.Abs(float64(got-tt.want)) > 1e-4 {
				t.Errorf("Pow(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

func TestPow_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name string
		x    float32
		y    float32
		want float32
	}{
		{"2^3 = 8", 2.0, 3.0, 8.0},
		{"2^0 = 1", 2.0, 0.0, 1.0},
		{"2^1 = 2", 2.0, 1.0, 2.0},
		{"3^2 = 9", 3.0, 2.0, 9.0},
		{"10^2 = 100", 10.0, 2.0, 100.0},
		{"2^-1 = 0.5", 2.0, -1.0, 0.5},
		{"4^0.5 = 2", 4.0, 0.5, 2.0},
		{"1^100 = 1", 1.0, 100.0, 1.0},
		{"e^1 = e", float32(stdmath.E), 1.0, float32(stdmath.E)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.x)
			y := archsimd.BroadcastFloat32x8(tt.y)
			result := Pow_AVX2_F32x8(x, y)
			got := extractLane32(result)

			if stdmath.Abs(float64(got-tt.want)) > 1e-4 {
				t.Errorf("Pow(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

func TestPow_AVX2_F32x8_SpecialCases(t *testing.T) {
	tests := []struct {
		name    string
		x       float32
		y       float32
		wantInf bool
		wantNaN bool
		want    float32
	}{
		{"0^1 = 0", 0.0, 1.0, false, false, 0.0},
		{"0^0 = 1", 0.0, 0.0, false, false, 1.0},
		{"0^-1 = +Inf", 0.0, -1.0, true, false, 0.0},
		{"1^0 = 1", 1.0, 0.0, false, false, 1.0},
		{"1^NaN = 1", 1.0, float32(stdmath.NaN()), false, false, 1.0},
		{"NaN^0 = 1", float32(stdmath.NaN()), 0.0, false, false, 1.0},
		{"+Inf^1 = +Inf", float32(stdmath.Inf(1)), 1.0, true, false, 0.0},
		{"+Inf^-1 = 0", float32(stdmath.Inf(1)), -1.0, false, false, 0.0},
		{"+Inf^0 = 1", float32(stdmath.Inf(1)), 0.0, false, false, 1.0},
		{"-1^2 = NaN", -1.0, 2.0, false, true, 0.0}, // negative base -> NaN in our impl
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.x)
			y := archsimd.BroadcastFloat32x8(tt.y)
			result := Pow_AVX2_F32x8(x, y)
			got := extractLane32(result)

			if tt.wantInf {
				if !stdmath.IsInf(float64(got), 1) {
					t.Errorf("Pow(%v, %v) = %v, want +Inf", tt.x, tt.y, got)
				}
			} else if tt.wantNaN {
				if !stdmath.IsNaN(float64(got)) {
					t.Errorf("Pow(%v, %v) = %v, want NaN", tt.x, tt.y, got)
				}
			} else {
				if stdmath.Abs(float64(got-tt.want)) > 1e-5 {
					t.Errorf("Pow(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
				}
			}
		})
	}
}

func TestPow_AVX2_F64x4(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		y    float64
		want float64
	}{
		{"2^3 = 8", 2.0, 3.0, 8.0},
		{"2^0 = 1", 2.0, 0.0, 1.0},
		{"2^1 = 2", 2.0, 1.0, 2.0},
		{"3^2 = 9", 3.0, 2.0, 9.0},
		{"10^2 = 100", 10.0, 2.0, 100.0},
		{"2^-1 = 0.5", 2.0, -1.0, 0.5},
		{"4^0.5 = 2", 4.0, 0.5, 2.0},
		{"8^(1/3) = 2", 8.0, 1.0 / 3.0, 2.0},
		{"1^100 = 1", 1.0, 100.0, 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat64x4(tt.x)
			y := archsimd.BroadcastFloat64x4(tt.y)
			result := Pow_AVX2_F64x4(x, y)
			got := extractLane64(result)

			if stdmath.Abs(got-tt.want) > 1e-10 {
				t.Errorf("Pow(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

func TestPow_AVX512_F32x16(t *testing.T) {
	if hwy.CurrentLevel() < hwy.DispatchAVX512 {
		t.Skip("AVX-512 not available")
	}

	tests := []struct {
		name string
		x    float32
		y    float32
		want float32
	}{
		{"2^3 = 8", 2.0, 3.0, 8.0},
		{"2^0 = 1", 2.0, 0.0, 1.0},
		{"2^1 = 2", 2.0, 1.0, 2.0},
		{"3^2 = 9", 3.0, 2.0, 9.0},
		{"10^2 = 100", 10.0, 2.0, 100.0},
		{"2^-1 = 0.5", 2.0, -1.0, 0.5},
		{"4^0.5 = 2", 4.0, 0.5, 2.0},
		{"1^100 = 1", 1.0, 100.0, 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x16(tt.x)
			y := archsimd.BroadcastFloat32x16(tt.y)
			result := Pow_AVX512_F32x16(x, y)

			var buf [16]float32
			result.StoreSlice(buf[:])
			got := buf[0]

			if stdmath.Abs(float64(got-tt.want)) > 1e-4 {
				t.Errorf("Pow(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

func TestPow_AVX512_F64x8(t *testing.T) {
	if hwy.CurrentLevel() < hwy.DispatchAVX512 {
		t.Skip("AVX-512 not available")
	}

	tests := []struct {
		name string
		x    float64
		y    float64
		want float64
	}{
		{"2^3 = 8", 2.0, 3.0, 8.0},
		{"2^0 = 1", 2.0, 0.0, 1.0},
		{"2^1 = 2", 2.0, 1.0, 2.0},
		{"3^2 = 9", 3.0, 2.0, 9.0},
		{"10^2 = 100", 10.0, 2.0, 100.0},
		{"2^-1 = 0.5", 2.0, -1.0, 0.5},
		{"4^0.5 = 2", 4.0, 0.5, 2.0},
		{"8^(1/3) = 2", 8.0, 1.0 / 3.0, 2.0},
		{"1^100 = 1", 1.0, 100.0, 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat64x8(tt.x)
			y := archsimd.BroadcastFloat64x8(tt.y)
			result := Pow_AVX512_F64x8(x, y)

			var buf [8]float64
			result.StoreSlice(buf[:])
			got := buf[0]

			if stdmath.Abs(got-tt.want) > 1e-10 {
				t.Errorf("Pow(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

// Benchmarks

func BenchmarkPow_AVX2_F32x8(b *testing.B) {
	x := archsimd.BroadcastFloat32x8(2.5)
	y := archsimd.BroadcastFloat32x8(3.7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Pow_AVX2_F32x8(x, y)
	}
}

func BenchmarkPow_AVX2_F64x4(b *testing.B) {
	x := archsimd.BroadcastFloat64x4(2.5)
	y := archsimd.BroadcastFloat64x4(3.7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Pow_AVX2_F64x4(x, y)
	}
}

func BenchmarkPow_AVX512_F32x16(b *testing.B) {
	x := archsimd.BroadcastFloat32x16(2.5)
	y := archsimd.BroadcastFloat32x16(3.7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Pow_AVX512_F32x16(x, y)
	}
}

func BenchmarkPow_AVX512_F64x8(b *testing.B) {
	x := archsimd.BroadcastFloat64x8(2.5)
	y := archsimd.BroadcastFloat64x8(3.7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Pow_AVX512_F64x8(x, y)
	}
}

func BenchmarkPow_Scalar(b *testing.B) {
	x := 2.5
	y := 3.7
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = stdmath.Pow(x, y)
	}
}
