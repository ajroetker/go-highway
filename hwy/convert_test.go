package hwy

import (
	"math"
	"testing"
)

func TestConvertToInt32_Float32(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
		want  []int32
	}{
		{
			name:  "positive integers",
			input: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
			want:  []int32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:  "negative integers",
			input: []float32{-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0},
			want:  []int32{-1, -2, -3, -4, -5, -6, -7, -8},
		},
		{
			name:  "truncate toward zero positive",
			input: []float32{1.9, 2.5, 3.1, 4.7, 5.5, 6.3, 7.8, 8.2},
			want:  []int32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:  "truncate toward zero negative",
			input: []float32{-1.9, -2.5, -3.1, -4.7, -5.5, -6.3, -7.8, -8.2},
			want:  []int32{-1, -2, -3, -4, -5, -6, -7, -8},
		},
		{
			name:  "zeros",
			input: []float32{0.0, -0.0, 0.1, -0.1, 0.9, -0.9, 0.5, -0.5},
			want:  []int32{0, 0, 0, 0, 0, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Load(tt.input)
			result := ConvertToInt32(v)

			for i := 0; i < result.NumLanes() && i < len(tt.want); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("lane %d: got %d, want %d", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestConvertToInt32_Float64(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		want  []int32
	}{
		{
			name:  "positive integers",
			input: []float64{1.0, 2.0, 3.0, 4.0},
			want:  []int32{1, 2, 3, 4},
		},
		{
			name:  "truncate toward zero",
			input: []float64{1.9, -2.9, 3.5, -4.5},
			want:  []int32{1, -2, 3, -4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Load(tt.input)
			result := ConvertToInt32(v)

			for i := 0; i < result.NumLanes() && i < len(tt.want); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("lane %d: got %d, want %d", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestConvertToFloat32_Int32(t *testing.T) {
	tests := []struct {
		name  string
		input []int32
		want  []float32
	}{
		{
			name:  "positive integers",
			input: []int32{1, 2, 3, 4, 5, 6, 7, 8},
			want:  []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
		},
		{
			name:  "negative integers",
			input: []int32{-1, -2, -3, -4, -5, -6, -7, -8},
			want:  []float32{-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0},
		},
		{
			name:  "zeros",
			input: []int32{0, 0, 0, 0, 0, 0, 0, 0},
			want:  []float32{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Load(tt.input)
			result := ConvertToFloat32(v)

			for i := 0; i < result.NumLanes() && i < len(tt.want); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("lane %d: got %f, want %f", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestConvertToFloat32_Int64(t *testing.T) {
	tests := []struct {
		name  string
		input []int64
		want  []float32
	}{
		{
			name:  "small integers",
			input: []int64{1, 2, 3, 4},
			want:  []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:  "negative integers",
			input: []int64{-1, -100, -1000, -10000},
			want:  []float32{-1.0, -100.0, -1000.0, -10000.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Load(tt.input)
			result := ConvertToFloat32(v)

			for i := 0; i < result.NumLanes() && i < len(tt.want); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("lane %d: got %f, want %f", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestConvertToInt64_Float64(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		want  []int64
	}{
		{
			name:  "positive integers",
			input: []float64{1.0, 2.0, 3.0, 4.0},
			want:  []int64{1, 2, 3, 4},
		},
		{
			name:  "truncate toward zero",
			input: []float64{1.9, -2.9, 3.5, -4.5},
			want:  []int64{1, -2, 3, -4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Load(tt.input)
			result := ConvertToInt64(v)

			for i := 0; i < result.NumLanes() && i < len(tt.want); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("lane %d: got %d, want %d", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestConvertToFloat64(t *testing.T) {
	t.Run("from int32", func(t *testing.T) {
		input := []int32{1, -2, 100, -1000}
		v := Load(input)
		result := ConvertToFloat64(v)

		for i := 0; i < result.NumLanes() && i < len(input); i++ {
			want := float64(input[i])
			if result.data[i] != want {
				t.Errorf("lane %d: got %f, want %f", i, result.data[i], want)
			}
		}
	})

	t.Run("from int64", func(t *testing.T) {
		input := []int64{1, -2, 100, -1000}
		v := Load(input)
		result := ConvertToFloat64(v)

		for i := 0; i < result.NumLanes() && i < len(input); i++ {
			want := float64(input[i])
			if result.data[i] != want {
				t.Errorf("lane %d: got %f, want %f", i, result.data[i], want)
			}
		}
	})
}

func TestRound(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		tests := []struct {
			input float32
			want  float32
		}{
			{1.4, 1.0},
			{1.5, 2.0}, // Round half away from zero
			{1.6, 2.0},
			{-1.4, -1.0},
			{-1.5, -2.0},
			{-1.6, -2.0},
			{2.5, 3.0}, // Note: math.Round rounds .5 away from zero
			{0.0, 0.0},
		}

		for _, tt := range tests {
			v := Set[float32](tt.input)
			result := Round(v)

			if result.data[0] != tt.want {
				t.Errorf("Round(%f): got %f, want %f", tt.input, result.data[0], tt.want)
			}
		}
	})

	t.Run("float64", func(t *testing.T) {
		tests := []struct {
			input float64
			want  float64
		}{
			{1.4, 1.0},
			{1.5, 2.0},
			{-1.5, -2.0},
			{2.5, 3.0},
		}

		for _, tt := range tests {
			v := Set[float64](tt.input)
			result := Round(v)

			if result.data[0] != tt.want {
				t.Errorf("Round(%f): got %f, want %f", tt.input, result.data[0], tt.want)
			}
		}
	})
}

func TestTrunc(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		tests := []struct {
			input float32
			want  float32
		}{
			{1.9, 1.0},
			{1.1, 1.0},
			{-1.9, -1.0},
			{-1.1, -1.0},
			{0.9, 0.0},
			{-0.9, 0.0},
		}

		for _, tt := range tests {
			v := Set[float32](tt.input)
			result := Trunc(v)

			if result.data[0] != tt.want {
				t.Errorf("Trunc(%f): got %f, want %f", tt.input, result.data[0], tt.want)
			}
		}
	})

	t.Run("float64", func(t *testing.T) {
		tests := []struct {
			input float64
			want  float64
		}{
			{1.9, 1.0},
			{-1.9, -1.0},
			{0.999, 0.0},
		}

		for _, tt := range tests {
			v := Set[float64](tt.input)
			result := Trunc(v)

			if result.data[0] != tt.want {
				t.Errorf("Trunc(%f): got %f, want %f", tt.input, result.data[0], tt.want)
			}
		}
	})
}

func TestCeil(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		tests := []struct {
			input float32
			want  float32
		}{
			{1.1, 2.0},
			{1.9, 2.0},
			{1.0, 1.0},
			{-1.1, -1.0},
			{-1.9, -1.0},
			{0.1, 1.0},
			{-0.1, 0.0},
		}

		for _, tt := range tests {
			v := Set[float32](tt.input)
			result := Ceil(v)

			if result.data[0] != tt.want {
				t.Errorf("Ceil(%f): got %f, want %f", tt.input, result.data[0], tt.want)
			}
		}
	})

	t.Run("float64", func(t *testing.T) {
		tests := []struct {
			input float64
			want  float64
		}{
			{1.1, 2.0},
			{-1.1, -1.0},
			{0.0, 0.0},
		}

		for _, tt := range tests {
			v := Set[float64](tt.input)
			result := Ceil(v)

			if result.data[0] != tt.want {
				t.Errorf("Ceil(%f): got %f, want %f", tt.input, result.data[0], tt.want)
			}
		}
	})
}

func TestFloor(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		tests := []struct {
			input float32
			want  float32
		}{
			{1.1, 1.0},
			{1.9, 1.0},
			{1.0, 1.0},
			{-1.1, -2.0},
			{-1.9, -2.0},
			{0.1, 0.0},
			{-0.1, -1.0},
		}

		for _, tt := range tests {
			v := Set[float32](tt.input)
			result := Floor(v)

			if result.data[0] != tt.want {
				t.Errorf("Floor(%f): got %f, want %f", tt.input, result.data[0], tt.want)
			}
		}
	})

	t.Run("float64", func(t *testing.T) {
		tests := []struct {
			input float64
			want  float64
		}{
			{1.9, 1.0},
			{-1.1, -2.0},
			{0.0, 0.0},
		}

		for _, tt := range tests {
			v := Set[float64](tt.input)
			result := Floor(v)

			if result.data[0] != tt.want {
				t.Errorf("Floor(%f): got %f, want %f", tt.input, result.data[0], tt.want)
			}
		}
	})
}

func TestNearestInt(t *testing.T) {
	t.Run("float32 banker's rounding", func(t *testing.T) {
		tests := []struct {
			input float32
			want  float32
		}{
			{0.5, 0.0},  // Round to even (0)
			{1.5, 2.0},  // Round to even (2)
			{2.5, 2.0},  // Round to even (2)
			{3.5, 4.0},  // Round to even (4)
			{-0.5, 0.0}, // Round to even (0)
			{-1.5, -2.0},
			{-2.5, -2.0},
			{1.4, 1.0},
			{1.6, 2.0},
		}

		for _, tt := range tests {
			v := Set[float32](tt.input)
			result := NearestInt(v)

			if result.data[0] != tt.want {
				t.Errorf("NearestInt(%f): got %f, want %f", tt.input, result.data[0], tt.want)
			}
		}
	})

	t.Run("float64 banker's rounding", func(t *testing.T) {
		tests := []struct {
			input float64
			want  float64
		}{
			{0.5, 0.0},
			{1.5, 2.0},
			{2.5, 2.0},
			{3.5, 4.0},
		}

		for _, tt := range tests {
			v := Set[float64](tt.input)
			result := NearestInt(v)

			if result.data[0] != tt.want {
				t.Errorf("NearestInt(%f): got %f, want %f", tt.input, result.data[0], tt.want)
			}
		}
	})
}

func TestBitCastF32ToI32(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  int32
	}{
		{"positive one", 1.0, 0x3f800000},
		{"negative one", -1.0, -0x40800000}, // 0xbf800000 as signed
		{"zero", 0.0, 0},
		{"negative zero", float32(math.Copysign(0, -1)), -0x80000000}, // 0x80000000 as signed (min int32)
		{"two", 2.0, 0x40000000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Set[float32](tt.input)
			result := BitCastF32ToI32(v)

			if result.data[0] != tt.want {
				t.Errorf("BitCastF32ToI32(%f): got 0x%08X, want 0x%08X", tt.input, uint32(result.data[0]), uint32(tt.want))
			}
		})
	}
}

func TestBitCastI32ToF32(t *testing.T) {
	tests := []struct {
		name  string
		input int32
		want  float32
	}{
		{"positive one", 0x3f800000, 1.0},
		{"negative one", -0x40800000, -1.0}, // 0xbf800000 as signed
		{"zero", 0, 0.0},
		{"two", 0x40000000, 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Set[int32](tt.input)
			result := BitCastI32ToF32(v)

			if result.data[0] != tt.want {
				t.Errorf("BitCastI32ToF32(0x%08X): got %f, want %f", uint32(tt.input), result.data[0], tt.want)
			}
		})
	}
}

func TestBitCastRoundTrip_F32I32(t *testing.T) {
	values := []float32{0.0, 1.0, -1.0, 3.14159, -2.71828, 1e10, -1e-10}

	for _, val := range values {
		v := Set[float32](val)
		asInt := BitCastF32ToI32(v)
		backToFloat := BitCastI32ToF32(asInt)

		if backToFloat.data[0] != val {
			t.Errorf("BitCast round trip failed for %f: got %f", val, backToFloat.data[0])
		}
	}
}

func TestBitCastF64ToI64(t *testing.T) {
	tests := []struct {
		name  string
		input float64
		want  int64
	}{
		{"positive one", 1.0, 0x3ff0000000000000},
		{"negative one", -1.0, -0x4010000000000000}, // 0xbff0000000000000 as signed
		{"zero", 0.0, 0},
		{"two", 2.0, 0x4000000000000000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Set[float64](tt.input)
			result := BitCastF64ToI64(v)

			if result.data[0] != tt.want {
				t.Errorf("BitCastF64ToI64(%f): got 0x%016X, want 0x%016X", tt.input, uint64(result.data[0]), uint64(tt.want))
			}
		})
	}
}

func TestBitCastI64ToF64(t *testing.T) {
	tests := []struct {
		name  string
		input int64
		want  float64
	}{
		{"positive one", 0x3ff0000000000000, 1.0},
		{"negative one", -0x4010000000000000, -1.0}, // 0xbff0000000000000 as signed
		{"zero", 0, 0.0},
		{"two", 0x4000000000000000, 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Set[int64](tt.input)
			result := BitCastI64ToF64(v)

			if result.data[0] != tt.want {
				t.Errorf("BitCastI64ToF64(0x%016X): got %f, want %f", uint64(tt.input), result.data[0], tt.want)
			}
		})
	}
}

func TestBitCastRoundTrip_F64I64(t *testing.T) {
	values := []float64{0.0, 1.0, -1.0, 3.14159265358979, -2.71828182845904, 1e100, -1e-100}

	for _, val := range values {
		v := Set[float64](val)
		asInt := BitCastF64ToI64(v)
		backToFloat := BitCastI64ToF64(asInt)

		if backToFloat.data[0] != val {
			t.Errorf("BitCast round trip failed for %f: got %f", val, backToFloat.data[0])
		}
	}
}

func TestBitCastU32F32(t *testing.T) {
	t.Run("U32 to F32", func(t *testing.T) {
		v := Set[uint32](0x3f800000)
		result := BitCastU32ToF32(v)
		if result.data[0] != 1.0 {
			t.Errorf("BitCastU32ToF32(0x3f800000): got %f, want 1.0", result.data[0])
		}
	})

	t.Run("F32 to U32", func(t *testing.T) {
		v := Set[float32](1.0)
		result := BitCastF32ToU32(v)
		if result.data[0] != 0x3f800000 {
			t.Errorf("BitCastF32ToU32(1.0): got 0x%08X, want 0x3f800000", result.data[0])
		}
	})
}

func TestBitCastU64F64(t *testing.T) {
	t.Run("U64 to F64", func(t *testing.T) {
		v := Set[uint64](0x3ff0000000000000)
		result := BitCastU64ToF64(v)
		if result.data[0] != 1.0 {
			t.Errorf("BitCastU64ToF64(0x3ff0000000000000): got %f, want 1.0", result.data[0])
		}
	})

	t.Run("F64 to U64", func(t *testing.T) {
		v := Set[float64](1.0)
		result := BitCastF64ToU64(v)
		if result.data[0] != 0x3ff0000000000000 {
			t.Errorf("BitCastF64ToU64(1.0): got 0x%016X, want 0x3ff0000000000000", result.data[0])
		}
	})
}

// Edge case tests

func TestConvertEdgeCases_NaN(t *testing.T) {
	// NaN conversion to int is undefined, but should not panic
	nanF32 := float32(math.NaN())
	v := Set[float32](nanF32)

	// This should not panic
	_ = ConvertToInt32(v)
}

func TestConvertEdgeCases_Inf(t *testing.T) {
	// +Inf and -Inf conversion to int is undefined but should not panic
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))

	vPos := Set[float32](posInf)
	vNeg := Set[float32](negInf)

	// These should not panic
	_ = ConvertToInt32(vPos)
	_ = ConvertToInt32(vNeg)
}

func TestRoundEdgeCases(t *testing.T) {
	t.Run("NaN", func(t *testing.T) {
		nan := float32(math.NaN())
		v := Set[float32](nan)
		result := Round(v)
		if !math.IsNaN(float64(result.data[0])) {
			t.Errorf("Round(NaN) should be NaN, got %f", result.data[0])
		}
	})

	t.Run("positive infinity", func(t *testing.T) {
		inf := float32(math.Inf(1))
		v := Set[float32](inf)
		result := Round(v)
		if !math.IsInf(float64(result.data[0]), 1) {
			t.Errorf("Round(+Inf) should be +Inf, got %f", result.data[0])
		}
	})

	t.Run("negative infinity", func(t *testing.T) {
		inf := float32(math.Inf(-1))
		v := Set[float32](inf)
		result := Round(v)
		if !math.IsInf(float64(result.data[0]), -1) {
			t.Errorf("Round(-Inf) should be -Inf, got %f", result.data[0])
		}
	})
}

func TestTruncEdgeCases(t *testing.T) {
	t.Run("NaN", func(t *testing.T) {
		nan := float64(math.NaN())
		v := Set[float64](nan)
		result := Trunc(v)
		if !math.IsNaN(result.data[0]) {
			t.Errorf("Trunc(NaN) should be NaN")
		}
	})

	t.Run("infinity", func(t *testing.T) {
		inf := float64(math.Inf(1))
		v := Set[float64](inf)
		result := Trunc(v)
		if !math.IsInf(result.data[0], 1) {
			t.Errorf("Trunc(+Inf) should be +Inf")
		}
	})
}

func TestCeilEdgeCases(t *testing.T) {
	t.Run("NaN", func(t *testing.T) {
		nan := float64(math.NaN())
		v := Set[float64](nan)
		result := Ceil(v)
		if !math.IsNaN(result.data[0]) {
			t.Errorf("Ceil(NaN) should be NaN")
		}
	})
}

func TestFloorEdgeCases(t *testing.T) {
	t.Run("NaN", func(t *testing.T) {
		nan := float64(math.NaN())
		v := Set[float64](nan)
		result := Floor(v)
		if !math.IsNaN(result.data[0]) {
			t.Errorf("Floor(NaN) should be NaN")
		}
	})
}

func TestBitCastPreservesBits_NaN(t *testing.T) {
	// Test that NaN bits are preserved through bitcast
	nanBits := uint32(0x7fc00000) // Quiet NaN
	v := Set[uint32](nanBits)
	asFloat := BitCastU32ToF32(v)
	backToU32 := BitCastF32ToU32(asFloat)

	if backToU32.data[0] != nanBits {
		t.Errorf("BitCast round trip failed for NaN bits: got 0x%08X, want 0x%08X", backToU32.data[0], nanBits)
	}
}

func TestBitCastPreservesBits_NegativeZero(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	v := Set[float32](negZero)
	asInt := BitCastF32ToI32(v)

	// Negative zero should have sign bit set (0x80000000 = min int32)
	if asInt.data[0] != -0x80000000 {
		t.Errorf("BitCast of -0.0: got 0x%08X, want 0x80000000", uint32(asInt.data[0]))
	}
}

func TestConvertLargeValues(t *testing.T) {
	t.Run("large float to int32", func(t *testing.T) {
		// Value larger than int32 max - behavior is undefined but should not panic
		largeVal := float32(3e9) // > 2^31-1
		v := Set[float32](largeVal)
		_ = ConvertToInt32(v) // Should not panic
	})

	t.Run("large int64 to float32 precision loss", func(t *testing.T) {
		// Large int64 that cannot be exactly represented as float32
		largeInt := int64(1) << 25 // 2^25, still exact in float32
		v := Set[int64](largeInt)
		result := ConvertToFloat32(v)

		expected := float32(largeInt)
		if result.data[0] != expected {
			t.Errorf("ConvertToFloat32(%d): got %f, want %f", largeInt, result.data[0], expected)
		}
	})
}

// Benchmark tests

func BenchmarkConvertToInt32_F32(b *testing.B) {
	v := Set[float32](3.14159)

	for b.Loop() {
		_ = ConvertToInt32(v)
	}
}

func BenchmarkConvertToFloat32_I32(b *testing.B) {
	v := Set[int32](42)

	for b.Loop() {
		_ = ConvertToFloat32(v)
	}
}

func BenchmarkRound_F32(b *testing.B) {
	v := Set[float32](3.14159)

	for b.Loop() {
		_ = Round(v)
	}
}

func BenchmarkTrunc_F32(b *testing.B) {
	v := Set[float32](3.14159)

	for b.Loop() {
		_ = Trunc(v)
	}
}

func BenchmarkCeil_F32(b *testing.B) {
	v := Set[float32](3.14159)

	for b.Loop() {
		_ = Ceil(v)
	}
}

func BenchmarkFloor_F32(b *testing.B) {
	v := Set[float32](3.14159)

	for b.Loop() {
		_ = Floor(v)
	}
}

func BenchmarkBitCastF32ToI32(b *testing.B) {
	v := Set[float32](3.14159)

	for b.Loop() {
		_ = BitCastF32ToI32(v)
	}
}

func BenchmarkBitCastI32ToF32(b *testing.B) {
	v := Set[int32](0x40490fdb) // pi bits

	for b.Loop() {
		_ = BitCastI32ToF32(v)
	}
}
