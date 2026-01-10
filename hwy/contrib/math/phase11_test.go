//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// Phase 11 tests: Tan, Atan, Atan2, Asin, Acos, Pow, Expm1, Log1p

// ============================================================================
// Tan Tests
// ============================================================================

func TestTan_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"tan(0)", 0.0},
		{"tan(pi/4)", float32(stdmath.Pi / 4)},
		{"tan(-pi/4)", float32(-stdmath.Pi / 4)},
		{"tan(pi/6)", float32(stdmath.Pi / 6)},
		{"tan(0.5)", 0.5},
		{"tan(1.0)", 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.input)
			result := Tan_AVX2_F32x8(x)
			got := extractLane32(result)
			want := float32(stdmath.Tan(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Tan(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

func TestTan_AVX512_F32x16(t *testing.T) {
	if hwy.CurrentLevel() < hwy.DispatchAVX512 {
		t.Skip("AVX-512 not available")
	}

	tests := []struct {
		name  string
		input float32
	}{
		{"tan(0)", 0.0},
		{"tan(pi/4)", float32(stdmath.Pi / 4)},
		{"tan(0.5)", 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x16(tt.input)
			result := Tan_AVX512_F32x16(x)
			var buf [16]float32
			result.StoreSlice(buf[:])
			got := buf[0]
			want := float32(stdmath.Tan(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Tan(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

// ============================================================================
// Atan Tests
// ============================================================================

func TestAtan_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"atan(0)", 0.0},
		{"atan(1)", 1.0},
		{"atan(-1)", -1.0},
		{"atan(0.5)", 0.5},
		{"atan(2)", 2.0},
		{"atan(-2)", -2.0},
		{"atan(10)", 10.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.input)
			result := Atan_AVX2_F32x8(x)
			got := extractLane32(result)
			want := float32(stdmath.Atan(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Atan(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

func TestAtan2_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name string
		y, x float32
	}{
		{"atan2(1, 1)", 1.0, 1.0},
		{"atan2(1, -1)", 1.0, -1.0},
		{"atan2(-1, 1)", -1.0, 1.0},
		{"atan2(-1, -1)", -1.0, -1.0},
		{"atan2(0, 1)", 0.0, 1.0},
		{"atan2(1, 0)", 1.0, 0.0},
		{"atan2(3, 4)", 3.0, 4.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			y := archsimd.BroadcastFloat32x8(tt.y)
			x := archsimd.BroadcastFloat32x8(tt.x)
			result := Atan2_AVX2_F32x8(y, x)
			got := extractLane32(result)
			want := float32(stdmath.Atan2(float64(tt.y), float64(tt.x)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Atan2(%v, %v) = %v, want %v", tt.y, tt.x, got, want)
			}
		})
	}
}

// ============================================================================
// Asin Tests
// ============================================================================

func TestAsin_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"asin(0)", 0.0},
		{"asin(0.5)", 0.5},
		{"asin(-0.5)", -0.5},
		{"asin(0.3)", 0.3},
		{"asin(0.7)", 0.7},
		{"asin(0.9)", 0.9},
		{"asin(1)", 1.0},
		{"asin(-1)", -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.input)
			result := Asin_AVX2_F32x8(x)
			got := extractLane32(result)
			want := float32(stdmath.Asin(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Asin(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

func TestAsin_OutOfRange(t *testing.T) {
	// |x| > 1 should return NaN
	x := archsimd.BroadcastFloat32x8(1.5)
	result := Asin_AVX2_F32x8(x)
	got := extractLane32(result)

	if !stdmath.IsNaN(float64(got)) {
		t.Errorf("Asin(1.5) = %v, want NaN", got)
	}
}

// ============================================================================
// Acos Tests
// ============================================================================

func TestAcos_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"acos(0)", 0.0},
		{"acos(0.5)", 0.5},
		{"acos(-0.5)", -0.5},
		{"acos(1)", 1.0},
		{"acos(-1)", -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.input)
			result := Acos_AVX2_F32x8(x)
			got := extractLane32(result)
			want := float32(stdmath.Acos(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Acos(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

// ============================================================================
// Expm1 Tests
// ============================================================================

func TestExpm1_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"expm1(0)", 0.0},
		{"expm1(0.001)", 0.001},   // Small value - uses Taylor series
		{"expm1(-0.001)", -0.001}, // Small negative
		{"expm1(0.1)", 0.1},
		{"expm1(1)", 1.0},
		{"expm1(-1)", -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.input)
			result := Expm1_AVX2_F32x8(x)
			got := extractLane32(result)
			want := float32(stdmath.Expm1(float64(tt.input)))

			relErr := stdmath.Abs(float64(got-want)) / (stdmath.Abs(float64(want)) + 1e-10)
			if relErr > 1e-5 {
				t.Errorf("Expm1(%v) = %v, want %v (relErr=%v)", tt.input, got, want, relErr)
			}
		})
	}
}

// ============================================================================
// Log1p Tests
// ============================================================================

func TestLog1p_AVX2_F32x8(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"log1p(0)", 0.0},
		{"log1p(0.001)", 0.001},   // Small value - uses Taylor series
		{"log1p(-0.001)", -0.001}, // Small negative
		{"log1p(0.1)", 0.1},
		{"log1p(1)", 1.0},
		{"log1p(10)", 10.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x8(tt.input)
			result := Log1p_AVX2_F32x8(x)
			got := extractLane32(result)
			want := float32(stdmath.Log1p(float64(tt.input)))

			relErr := stdmath.Abs(float64(got-want)) / (stdmath.Abs(float64(want)) + 1e-10)
			if relErr > 1e-5 {
				t.Errorf("Log1p(%v) = %v, want %v (relErr=%v)", tt.input, got, want, relErr)
			}
		})
	}
}

func TestLog1p_SpecialCases(t *testing.T) {
	// log1p(-1) = -Inf
	x := archsimd.BroadcastFloat32x8(-1.0)
	result := Log1p_AVX2_F32x8(x)
	got := extractLane32(result)

	if !stdmath.IsInf(float64(got), -1) {
		t.Errorf("Log1p(-1) = %v, want -Inf", got)
	}

	// log1p(x) = NaN for x < -1
	x = archsimd.BroadcastFloat32x8(-2.0)
	result = Log1p_AVX2_F32x8(x)
	got = extractLane32(result)

	if !stdmath.IsNaN(float64(got)) {
		t.Errorf("Log1p(-2) = %v, want NaN", got)
	}
}

// ============================================================================
// AVX-512 Tests
// ============================================================================

func TestAtan_AVX512_F32x16(t *testing.T) {
	if hwy.CurrentLevel() < hwy.DispatchAVX512 {
		t.Skip("AVX-512 not available")
	}

	tests := []struct {
		name  string
		input float32
	}{
		{"atan(0)", 0.0},
		{"atan(1)", 1.0},
		{"atan(2)", 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x16(tt.input)
			result := Atan_AVX512_F32x16(x)
			var buf [16]float32
			result.StoreSlice(buf[:])
			got := buf[0]
			want := float32(stdmath.Atan(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Atan(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

func TestAsin_AVX512_F32x16(t *testing.T) {
	if hwy.CurrentLevel() < hwy.DispatchAVX512 {
		t.Skip("AVX-512 not available")
	}

	tests := []struct {
		name  string
		input float32
	}{
		{"asin(0)", 0.0},
		{"asin(0.5)", 0.5},
		{"asin(0.9)", 0.9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := archsimd.BroadcastFloat32x16(tt.input)
			result := Asin_AVX512_F32x16(x)
			var buf [16]float32
			result.StoreSlice(buf[:])
			got := buf[0]
			want := float32(stdmath.Asin(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Asin(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

func TestExpm1_AVX512_F32x16(t *testing.T) {
	if hwy.CurrentLevel() < hwy.DispatchAVX512 {
		t.Skip("AVX-512 not available")
	}

	x := archsimd.BroadcastFloat32x16(0.001)
	result := Expm1_AVX512_F32x16(x)
	var buf [16]float32
	result.StoreSlice(buf[:])
	got := buf[0]
	want := float32(stdmath.Expm1(0.001))

	if stdmath.Abs(float64(got-want)) > 1e-7 {
		t.Errorf("Expm1(0.001) = %v, want %v", got, want)
	}
}

func TestLog1p_AVX512_F32x16(t *testing.T) {
	if hwy.CurrentLevel() < hwy.DispatchAVX512 {
		t.Skip("AVX-512 not available")
	}

	x := archsimd.BroadcastFloat32x16(0.001)
	result := Log1p_AVX512_F32x16(x)
	var buf [16]float32
	result.StoreSlice(buf[:])
	got := buf[0]
	want := float32(stdmath.Log1p(0.001))

	if stdmath.Abs(float64(got-want)) > 1e-7 {
		t.Errorf("Log1p(0.001) = %v, want %v", got, want)
	}
}

// ============================================================================
// Benchmarks
// ============================================================================

func BenchmarkTan_AVX2_F32x8(b *testing.B) {
	x := archsimd.BroadcastFloat32x8(0.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Tan_AVX2_F32x8(x)
	}
}

func BenchmarkAtan_AVX2_F32x8(b *testing.B) {
	x := archsimd.BroadcastFloat32x8(0.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Atan_AVX2_F32x8(x)
	}
}

func BenchmarkAsin_AVX2_F32x8(b *testing.B) {
	x := archsimd.BroadcastFloat32x8(0.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Asin_AVX2_F32x8(x)
	}
}

func BenchmarkExpm1_AVX2_F32x8(b *testing.B) {
	x := archsimd.BroadcastFloat32x8(0.001)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Expm1_AVX2_F32x8(x)
	}
}

func BenchmarkLog1p_AVX2_F32x8(b *testing.B) {
	x := archsimd.BroadcastFloat32x8(0.001)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Log1p_AVX2_F32x8(x)
	}
}
