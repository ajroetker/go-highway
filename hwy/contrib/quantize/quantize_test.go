// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package quantize

import (
	"math"
	"testing"
)

func TestDequantizeUint8(t *testing.T) {
	tests := []struct {
		name   string
		input  []uint8
		min    float32
		scale  float32
		want   []float32
	}{
		{
			name:  "empty",
			input: []uint8{},
			min:   0, scale: 1,
			want: []float32{},
		},
		{
			name:  "identity scale",
			input: []uint8{0, 1, 2, 255},
			min:   0, scale: 1,
			want: []float32{0, 1, 2, 255},
		},
		{
			name:  "normalize to 0-1",
			input: []uint8{0, 255},
			min:   0, scale: 1.0 / 255.0,
			want: []float32{0, 1},
		},
		{
			name:  "normalize to -1 to 1",
			input: []uint8{0, 128, 255},
			min:   -1.0, scale: 2.0 / 255.0,
			want: []float32{-1.0, -1.0 + 128.0*2.0/255.0, 1.0},
		},
		{
			name:  "all zeros",
			input: []uint8{0, 0, 0, 0},
			min:   0, scale: 1,
			want: []float32{0, 0, 0, 0},
		},
		{
			name:  "all 255",
			input: []uint8{255, 255, 255, 255},
			min:   0, scale: 1,
			want: []float32{255, 255, 255, 255},
		},
		{
			name:  "single element",
			input: []uint8{42},
			min:   0, scale: 1,
			want: []float32{42},
		},
		{
			name:  "non-aligned length 17",
			input: func() []uint8 {
				s := make([]uint8, 17)
				for i := range s {
					s[i] = uint8(i * 15)
				}
				return s
			}(),
			min: 0, scale: 1,
			want: func() []float32 {
				s := make([]float32, 17)
				for i := range s {
					s[i] = float32(uint8(i * 15))
				}
				return s
			}(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := make([]float32, len(tt.input))
			DequantizeUint8(tt.input, got, tt.min, tt.scale)

			if len(got) != len(tt.want) {
				t.Fatalf("length mismatch: got %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if math.Abs(float64(got[i]-tt.want[i])) > 1e-5 {
					t.Errorf("index %d: got %f, want %f", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestQuantizeFloat32(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
		min   float32
		scale float32
		want  []uint8
	}{
		{
			name:  "empty",
			input: []float32{},
			min:   0, scale: 1,
			want: []uint8{},
		},
		{
			name:  "identity scale",
			input: []float32{0, 1, 2, 255},
			min:   0, scale: 1,
			want: []uint8{0, 1, 2, 255},
		},
		{
			name:  "clamp below zero",
			input: []float32{-10, -1, 0},
			min:   0, scale: 1,
			want: []uint8{0, 0, 0},
		},
		{
			name:  "clamp above 255",
			input: []float32{255, 256, 1000},
			min:   0, scale: 1,
			want: []uint8{255, 255, 255},
		},
		{
			name:  "rounding",
			input: []float32{0.4, 0.5, 0.6, 1.5, 2.5},
			min:   0, scale: 1,
			want: []uint8{0, 0, 1, 2, 2},
		},
		{
			name:  "normalize from -1 to 1",
			input: []float32{-1.0, 0.0, 1.0},
			min:   -1.0, scale: 2.0 / 255.0,
			want: []uint8{0, 128, 255},
		},
		{
			name:  "single element",
			input: []float32{42},
			min:   0, scale: 1,
			want: []uint8{42},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := make([]uint8, len(tt.input))
			QuantizeFloat32(tt.input, got, tt.min, tt.scale)

			if len(got) != len(tt.want) {
				t.Fatalf("length mismatch: got %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				// Allow ±1 for rounding differences
				diff := int(got[i]) - int(tt.want[i])
				if diff < -1 || diff > 1 {
					t.Errorf("index %d: got %d, want %d", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestRoundTrip(t *testing.T) {
	// Generate a range of uint8 values
	input := make([]uint8, 256)
	for i := range input {
		input[i] = uint8(i)
	}

	min := float32(-1.0)
	scale := float32(2.0 / 255.0)

	// Dequantize
	floats := make([]float32, len(input))
	DequantizeUint8(input, floats, min, scale)

	// Quantize back
	output := make([]uint8, len(input))
	QuantizeFloat32(floats, output, min, scale)

	// Verify round-trip
	for i := range input {
		diff := int(output[i]) - int(input[i])
		if diff < -1 || diff > 1 {
			t.Errorf("index %d: got %d, want %d (float was %f)", i, output[i], input[i], floats[i])
		}
	}
}

func TestNonAlignedLengths(t *testing.T) {
	// Test various lengths that don't align to SIMD widths
	for _, n := range []int{1, 3, 7, 9, 15, 16, 17, 31, 33, 63, 64, 65, 100, 255, 256, 257} {
		t.Run("", func(t *testing.T) {
			input := make([]uint8, n)
			for i := range input {
				input[i] = uint8(i % 256)
			}

			output := make([]float32, n)
			DequantizeUint8(input, output, 0, 1)

			for i := range output {
				want := float32(input[i])
				if output[i] != want {
					t.Errorf("n=%d, index %d: got %f, want %f", n, i, output[i], want)
				}
			}
		})
	}
}

func BenchmarkDequantizeUint8(b *testing.B) {
	for _, size := range []int{64, 256, 1024, 4096} {
		b.Run("", func(b *testing.B) {
			input := make([]uint8, size)
			for i := range input {
				input[i] = uint8(i % 256)
			}
			output := make([]float32, size)

			b.SetBytes(int64(size))
			b.ResetTimer()
			for range b.N {
				DequantizeUint8(input, output, -1.0, 2.0/255.0)
			}
		})
	}
}

func BenchmarkQuantizeFloat32(b *testing.B) {
	for _, size := range []int{64, 256, 1024, 4096} {
		b.Run("", func(b *testing.B) {
			input := make([]float32, size)
			for i := range input {
				input[i] = float32(i%256) / 255.0
			}
			output := make([]uint8, size)

			b.SetBytes(int64(size) * 4)
			b.ResetTimer()
			for range b.N {
				QuantizeFloat32(input, output, 0, 1.0/255.0)
			}
		})
	}
}
