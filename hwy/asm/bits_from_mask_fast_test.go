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

//go:build !noasm && arm64

package asm

import (
	"testing"
)

func TestBitsFromMaskFast(t *testing.T) {
	tests := []struct {
		name   string
		input  Uint8x16
		want   uint64
	}{
		{
			name:   "all zeros",
			input:  Uint8x16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			want:   0,
		},
		{
			name:   "all ones",
			input:  Uint8x16{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
			want:   0xFFFF,
		},
		{
			name:   "first byte only",
			input:  Uint8x16{0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			want:   0x0001,
		},
		{
			name:   "last byte only",
			input:  Uint8x16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF},
			want:   0x8000,
		},
		{
			name:   "alternating",
			input:  Uint8x16{0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0},
			want:   0x5555,
		},
		{
			name:   "alternating inverse",
			input:  Uint8x16{0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF},
			want:   0xAAAA,
		},
		{
			name:   "first 4 bytes",
			input:  Uint8x16{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			want:   0x000F,
		},
		{
			name:   "byte 8 only (first of high half)",
			input:  Uint8x16{0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0, 0, 0, 0, 0, 0, 0},
			want:   0x0100,
		},
		{
			name:   "low half",
			input:  Uint8x16{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0},
			want:   0x00FF,
		},
		{
			name:   "high half",
			input:  Uint8x16{0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
			want:   0xFF00,
		},
		{
			name:   "12-bit pattern for varint (first 12 bytes)",
			input:  Uint8x16{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0},
			want:   0x0FFF,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BitsFromMaskFast(tt.input)
			if got != tt.want {
				t.Errorf("BitsFromMaskFast() = 0x%04X, want 0x%04X", got, tt.want)
			}

			// Also verify original implementation matches
			original := BitsFromMask(tt.input)
			if uint64(original) != tt.want {
				t.Errorf("BitsFromMask() = 0x%04X, want 0x%04X", original, tt.want)
			}

			// Compare both implementations
			if got != uint64(original) {
				t.Errorf("Mismatch: BitsFromMaskFast=0x%04X, BitsFromMask=0x%04X", got, original)
			}
		})
	}
}

func BenchmarkBitsFromMask(b *testing.B) {
	// Create test input: typical varint boundary pattern
	input := Uint8x16{
		0xFF, 0x00, 0xFF, 0xFF, // bytes 0-3: terminators at 0, 2, 3
		0x00, 0xFF, 0x00, 0xFF, // bytes 4-7: terminators at 5, 7
		0xFF, 0x00, 0xFF, 0xFF, // bytes 8-11: terminators at 8, 10, 11
		0x00, 0x00, 0x00, 0x00, // bytes 12-15: no terminators
	}

	b.Run("Original", func(b *testing.B) {
		b.ReportAllocs()
		var result uint64
		for i := 0; i < b.N; i++ {
			result = BitsFromMask(input)
		}
		_ = result
	})

	b.Run("Fast", func(b *testing.B) {
		b.ReportAllocs()
		var result uint64
		for i := 0; i < b.N; i++ {
			result = BitsFromMaskFast(input)
		}
		_ = result
	})
}
