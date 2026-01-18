package hwy

import (
	"math"
	"testing"
)

func TestPromoteF32ToF64(t *testing.T) {
	input := Vec[float32]{data: []float32{1.5, 2.25, 3.125, 4.0625}}
	result := PromoteF32ToF64(input)

	for i := 0; i < len(input.data); i++ {
		expected := float64(input.data[i])
		if result.data[i] != expected {
			t.Errorf("PromoteF32ToF64 lane %d: got %v, want %v", i, result.data[i], expected)
		}
	}
}

func TestPromoteLowerF32ToF64(t *testing.T) {
	input := Vec[float32]{data: []float32{1, 2, 3, 4, 5, 6, 7, 8}}
	result := PromoteLowerF32ToF64(input)

	// Should only promote lower 4 lanes
	want := []float64{1, 2, 3, 4}
	if len(result.data) != 4 {
		t.Errorf("PromoteLowerF32ToF64: got %d lanes, want 4", len(result.data))
	}
	for i := 0; i < len(want) && i < len(result.data); i++ {
		if result.data[i] != want[i] {
			t.Errorf("PromoteLowerF32ToF64 lane %d: got %v, want %v", i, result.data[i], want[i])
		}
	}
}

func TestPromoteUpperF32ToF64(t *testing.T) {
	input := Vec[float32]{data: []float32{1, 2, 3, 4, 5, 6, 7, 8}}
	result := PromoteUpperF32ToF64(input)

	// Should only promote upper 4 lanes
	want := []float64{5, 6, 7, 8}
	if len(result.data) != 4 {
		t.Errorf("PromoteUpperF32ToF64: got %d lanes, want 4", len(result.data))
	}
	for i := 0; i < len(want) && i < len(result.data); i++ {
		if result.data[i] != want[i] {
			t.Errorf("PromoteUpperF32ToF64 lane %d: got %v, want %v", i, result.data[i], want[i])
		}
	}
}

func TestDemoteF64ToF32(t *testing.T) {
	input := Vec[float64]{data: []float64{1.5, 2.25, 3.125, 4.0625}}
	result := DemoteF64ToF32(input)

	for i := 0; i < len(input.data); i++ {
		expected := float32(input.data[i])
		if result.data[i] != expected {
			t.Errorf("DemoteF64ToF32 lane %d: got %v, want %v", i, result.data[i], expected)
		}
	}
}

func TestDemoteTwoF64ToF32(t *testing.T) {
	lo := Vec[float64]{data: []float64{1, 2, 3, 4}}
	hi := Vec[float64]{data: []float64{5, 6, 7, 8}}
	result := DemoteTwoF64ToF32(lo, hi)

	want := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	if len(result.data) != 8 {
		t.Errorf("DemoteTwoF64ToF32: got %d lanes, want 8", len(result.data))
	}
	for i := 0; i < len(want) && i < len(result.data); i++ {
		if result.data[i] != want[i] {
			t.Errorf("DemoteTwoF64ToF32 lane %d: got %v, want %v", i, result.data[i], want[i])
		}
	}
}

func TestPromoteDemoteF32F64RoundTrip(t *testing.T) {
	// Values that can be represented exactly in both float32 and float64
	original := Vec[float32]{data: []float32{1.0, 2.0, 0.5, 0.25}}

	promoted := PromoteF32ToF64(original)
	demoted := DemoteF64ToF32(promoted)

	for i := 0; i < len(original.data); i++ {
		if demoted.data[i] != original.data[i] {
			t.Errorf("PromoteDemote round trip lane %d: got %v, want %v",
				i, demoted.data[i], original.data[i])
		}
	}
}

func TestPromoteI8ToI16(t *testing.T) {
	input := Vec[int8]{data: []int8{-128, -1, 0, 1, 127}}
	result := PromoteI8ToI16(input)

	for i := 0; i < len(input.data); i++ {
		expected := int16(input.data[i])
		if result.data[i] != expected {
			t.Errorf("PromoteI8ToI16 lane %d: got %v, want %v", i, result.data[i], expected)
		}
	}
}

func TestPromoteI16ToI32(t *testing.T) {
	input := Vec[int16]{data: []int16{-32768, -1, 0, 1, 32767}}
	result := PromoteI16ToI32(input)

	for i := 0; i < len(input.data); i++ {
		expected := int32(input.data[i])
		if result.data[i] != expected {
			t.Errorf("PromoteI16ToI32 lane %d: got %v, want %v", i, result.data[i], expected)
		}
	}
}

func TestPromoteI32ToI64(t *testing.T) {
	input := Vec[int32]{data: []int32{-2147483648, -1, 0, 1, 2147483647}}
	result := PromoteI32ToI64(input)

	for i := 0; i < len(input.data); i++ {
		expected := int64(input.data[i])
		if result.data[i] != expected {
			t.Errorf("PromoteI32ToI64 lane %d: got %v, want %v", i, result.data[i], expected)
		}
	}
}

func TestPromoteU8ToU16(t *testing.T) {
	input := Vec[uint8]{data: []uint8{0, 1, 127, 128, 255}}
	result := PromoteU8ToU16(input)

	for i := 0; i < len(input.data); i++ {
		expected := uint16(input.data[i])
		if result.data[i] != expected {
			t.Errorf("PromoteU8ToU16 lane %d: got %v, want %v", i, result.data[i], expected)
		}
	}
}

func TestPromoteU16ToU32(t *testing.T) {
	input := Vec[uint16]{data: []uint16{0, 1, 32767, 32768, 65535}}
	result := PromoteU16ToU32(input)

	for i := 0; i < len(input.data); i++ {
		expected := uint32(input.data[i])
		if result.data[i] != expected {
			t.Errorf("PromoteU16ToU32 lane %d: got %v, want %v", i, result.data[i], expected)
		}
	}
}

func TestPromoteU32ToU64(t *testing.T) {
	input := Vec[uint32]{data: []uint32{0, 1, 2147483647, 2147483648, 4294967295}}
	result := PromoteU32ToU64(input)

	for i := 0; i < len(input.data); i++ {
		expected := uint64(input.data[i])
		if result.data[i] != expected {
			t.Errorf("PromoteU32ToU64 lane %d: got %v, want %v", i, result.data[i], expected)
		}
	}
}

func TestDemoteI16ToI8Saturating(t *testing.T) {
	tests := []struct {
		name  string
		input []int16
		want  []int8
	}{
		{
			name:  "within range",
			input: []int16{-128, -1, 0, 1, 127},
			want:  []int8{-128, -1, 0, 1, 127},
		},
		{
			name:  "positive overflow",
			input: []int16{200, 500, 32767},
			want:  []int8{127, 127, 127},
		},
		{
			name:  "negative overflow",
			input: []int16{-200, -500, -32768},
			want:  []int8{-128, -128, -128},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := Vec[int16]{data: tt.input}
			result := DemoteI16ToI8(input)

			for i := 0; i < len(tt.want) && i < len(result.data); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("DemoteI16ToI8 lane %d: got %v, want %v", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestDemoteI32ToI16Saturating(t *testing.T) {
	tests := []struct {
		name  string
		input []int32
		want  []int16
	}{
		{
			name:  "within range",
			input: []int32{-32768, -1, 0, 1, 32767},
			want:  []int16{-32768, -1, 0, 1, 32767},
		},
		{
			name:  "positive overflow",
			input: []int32{40000, 100000, 2147483647},
			want:  []int16{32767, 32767, 32767},
		},
		{
			name:  "negative overflow",
			input: []int32{-40000, -100000, -2147483648},
			want:  []int16{-32768, -32768, -32768},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := Vec[int32]{data: tt.input}
			result := DemoteI32ToI16(input)

			for i := 0; i < len(tt.want) && i < len(result.data); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("DemoteI32ToI16 lane %d: got %v, want %v", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestDemoteI64ToI32Saturating(t *testing.T) {
	tests := []struct {
		name  string
		input []int64
		want  []int32
	}{
		{
			name:  "within range",
			input: []int64{-2147483648, -1, 0, 1, 2147483647},
			want:  []int32{-2147483648, -1, 0, 1, 2147483647},
		},
		{
			name:  "positive overflow",
			input: []int64{3000000000, math.MaxInt64},
			want:  []int32{2147483647, 2147483647},
		},
		{
			name:  "negative overflow",
			input: []int64{-3000000000, math.MinInt64},
			want:  []int32{-2147483648, -2147483648},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := Vec[int64]{data: tt.input}
			result := DemoteI64ToI32(input)

			for i := 0; i < len(tt.want) && i < len(result.data); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("DemoteI64ToI32 lane %d: got %v, want %v", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestDemoteU16ToU8Saturating(t *testing.T) {
	tests := []struct {
		name  string
		input []uint16
		want  []uint8
	}{
		{
			name:  "within range",
			input: []uint16{0, 1, 127, 255},
			want:  []uint8{0, 1, 127, 255},
		},
		{
			name:  "overflow",
			input: []uint16{256, 500, 65535},
			want:  []uint8{255, 255, 255},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := Vec[uint16]{data: tt.input}
			result := DemoteU16ToU8(input)

			for i := 0; i < len(tt.want) && i < len(result.data); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("DemoteU16ToU8 lane %d: got %v, want %v", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestDemoteU32ToU16Saturating(t *testing.T) {
	tests := []struct {
		name  string
		input []uint32
		want  []uint16
	}{
		{
			name:  "within range",
			input: []uint32{0, 1, 32767, 65535},
			want:  []uint16{0, 1, 32767, 65535},
		},
		{
			name:  "overflow",
			input: []uint32{70000, 100000, 4294967295},
			want:  []uint16{65535, 65535, 65535},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := Vec[uint32]{data: tt.input}
			result := DemoteU32ToU16(input)

			for i := 0; i < len(tt.want) && i < len(result.data); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("DemoteU32ToU16 lane %d: got %v, want %v", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestDemoteU64ToU32Saturating(t *testing.T) {
	tests := []struct {
		name  string
		input []uint64
		want  []uint32
	}{
		{
			name:  "within range",
			input: []uint64{0, 1, 2147483647, 4294967295},
			want:  []uint32{0, 1, 2147483647, 4294967295},
		},
		{
			name:  "overflow",
			input: []uint64{5000000000, math.MaxUint64},
			want:  []uint32{4294967295, 4294967295},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := Vec[uint64]{data: tt.input}
			result := DemoteU64ToU32(input)

			for i := 0; i < len(tt.want) && i < len(result.data); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("DemoteU64ToU32 lane %d: got %v, want %v", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestTruncateI16ToI8(t *testing.T) {
	// Truncation just keeps the lower bits
	input := Vec[int16]{data: []int16{256, 512, -256}}
	result := TruncateI16ToI8(input)

	want := []int8{0, 0, 0} // Lower 8 bits of these values
	for i := range want {
		if result.data[i] != want[i] {
			t.Errorf("TruncateI16ToI8 lane %d: got %v, want %v", i, result.data[i], want[i])
		}
	}
}

func TestTruncateI32ToI16(t *testing.T) {
	input := Vec[int32]{data: []int32{65536, 131072, -65536}}
	result := TruncateI32ToI16(input)

	want := []int16{0, 0, 0}
	for i := range want {
		if result.data[i] != want[i] {
			t.Errorf("TruncateI32ToI16 lane %d: got %v, want %v", i, result.data[i], want[i])
		}
	}
}

func TestTruncateI64ToI32(t *testing.T) {
	input := Vec[int64]{data: []int64{4294967296, 8589934592}}
	result := TruncateI64ToI32(input)

	want := []int32{0, 0}
	for i := range want {
		if result.data[i] != want[i] {
			t.Errorf("TruncateI64ToI32 lane %d: got %v, want %v", i, result.data[i], want[i])
		}
	}
}

func TestDemoteTwoI32ToI16(t *testing.T) {
	lo := Vec[int32]{data: []int32{1, 2, 3, 4}}
	hi := Vec[int32]{data: []int32{5, 6, 7, 8}}
	result := DemoteTwoI32ToI16(lo, hi)

	want := []int16{1, 2, 3, 4, 5, 6, 7, 8}
	if len(result.data) != 8 {
		t.Errorf("DemoteTwoI32ToI16: got %d lanes, want 8", len(result.data))
	}
	for i := 0; i < len(want) && i < len(result.data); i++ {
		if result.data[i] != want[i] {
			t.Errorf("DemoteTwoI32ToI16 lane %d: got %v, want %v", i, result.data[i], want[i])
		}
	}
}

func TestDemoteTwoI64ToI32(t *testing.T) {
	lo := Vec[int64]{data: []int64{1, 2, 3, 4}}
	hi := Vec[int64]{data: []int64{5, 6, 7, 8}}
	result := DemoteTwoI64ToI32(lo, hi)

	want := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	if len(result.data) != 8 {
		t.Errorf("DemoteTwoI64ToI32: got %d lanes, want 8", len(result.data))
	}
	for i := 0; i < len(want) && i < len(result.data); i++ {
		if result.data[i] != want[i] {
			t.Errorf("DemoteTwoI64ToI32 lane %d: got %v, want %v", i, result.data[i], want[i])
		}
	}
}

func TestPromoteLowerUpperCombined(t *testing.T) {
	// Test that PromoteLower and PromoteUpper together cover all lanes
	input := Vec[int32]{data: []int32{1, 2, 3, 4, 5, 6, 7, 8}}

	lower := PromoteLowerI32ToI64(input)
	upper := PromoteUpperI32ToI64(input)

	// Lower should have first half
	for i := 0; i < len(lower.data); i++ {
		expected := int64(input.data[i])
		if lower.data[i] != expected {
			t.Errorf("PromoteLowerI32ToI64 lane %d: got %v, want %v", i, lower.data[i], expected)
		}
	}

	// Upper should have second half
	half := len(input.data) / 2
	for i := 0; i < len(upper.data); i++ {
		expected := int64(input.data[half+i])
		if upper.data[i] != expected {
			t.Errorf("PromoteUpperI32ToI64 lane %d: got %v, want %v", i, upper.data[i], expected)
		}
	}
}

// Benchmark tests
func BenchmarkPromoteF32ToF64(b *testing.B) {
	data := make([]float32, 8)
	for i := range data {
		data[i] = float32(i)
	}
	v := Vec[float32]{data: data}

	for b.Loop() {
		_ = PromoteF32ToF64(v)
	}
}

func BenchmarkDemoteF64ToF32(b *testing.B) {
	data := make([]float64, 8)
	for i := range data {
		data[i] = float64(i)
	}
	v := Vec[float64]{data: data}

	for b.Loop() {
		_ = DemoteF64ToF32(v)
	}
}

func BenchmarkPromoteI32ToI64(b *testing.B) {
	data := make([]int32, 8)
	for i := range data {
		data[i] = int32(i)
	}
	v := Vec[int32]{data: data}

	for b.Loop() {
		_ = PromoteI32ToI64(v)
	}
}

func BenchmarkDemoteI64ToI32Saturating(b *testing.B) {
	data := make([]int64, 8)
	for i := range data {
		data[i] = int64(i * 1000000000)
	}
	v := Vec[int64]{data: data}

	for b.Loop() {
		_ = DemoteI64ToI32(v)
	}
}
