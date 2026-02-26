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

package gguf

import (
	"fmt"
	"math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// fp16One is the little-endian encoding of fp16 1.0 (0x3C00).
var fp16One = [2]byte{0x00, 0x3C}

// makeQ8_0Block builds a single Q8_0 block with given fp16 scale bytes and 32 int8 quants.
func makeQ8_0Block(scaleLo, scaleHi byte, quants [32]int8) []byte {
	block := make([]byte, BlockSizeQ8_0)
	block[0] = scaleLo
	block[1] = scaleHi
	for i, q := range quants {
		block[2+i] = byte(q)
	}
	return block
}

// makeQ4_0Block builds a single Q4_0/IQ4_NL block with given fp16 scale bytes and 16 nibble bytes.
func makeQ4_0Block(scaleLo, scaleHi byte, nibbles [16]byte) []byte {
	block := make([]byte, BlockSizeQ4_0)
	block[0] = scaleLo
	block[1] = scaleHi
	copy(block[2:], nibbles[:])
	return block
}

func TestDequantizeQ8_0(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want []float32
	}{
		{
			name: "empty",
			data: nil,
			want: nil,
		},
		{
			name: "single block d=1.0 ascending",
			data: func() []byte {
				var quants [32]int8
				for i := range 32 {
					quants[i] = int8(i - 16) // -16..15
				}
				return makeQ8_0Block(fp16One[0], fp16One[1], quants)
			}(),
			want: func() []float32 {
				out := make([]float32, 32)
				for i := range 32 {
					out[i] = float32(i - 16)
				}
				return out
			}(),
		},
		{
			name: "single block d=1.0 all zeros",
			data: makeQ8_0Block(fp16One[0], fp16One[1], [32]int8{}),
			want: make([]float32, 32),
		},
		{
			name: "single block d=0",
			data: makeQ8_0Block(0, 0, [32]int8{1, 2, 3, 127, -128}),
			want: make([]float32, 32),
		},
		{
			name: "two blocks",
			data: func() []byte {
				var q1, q2 [32]int8
				for i := range 32 {
					q1[i] = 1
					q2[i] = -1
				}
				b1 := makeQ8_0Block(fp16One[0], fp16One[1], q1)
				b2 := makeQ8_0Block(fp16One[0], fp16One[1], q2)
				return append(b1, b2...)
			}(),
			want: func() []float32 {
				out := make([]float32, 64)
				for i := range 32 {
					out[i] = 1
					out[32+i] = -1
				}
				return out
			}(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.want == nil {
				got := make([]float32, 0)
				DequantizeQ8_0(tt.data, got)
				return
			}
			got := make([]float32, len(tt.want))
			DequantizeQ8_0(tt.data, got)
			for i := range got {
				if math.Abs(float64(got[i]-tt.want[i])) > 1e-5 {
					t.Errorf("index %d: got %f, want %f", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestDequantizeQ4_0(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want []float32
	}{
		{
			name: "empty",
			data: nil,
			want: nil,
		},
		{
			name: "single block d=1.0 split layout",
			// Each nibble byte: lo=0x0, hi=0xF -> nibbles[0] = 0xF0, etc.
			// Use nibble bytes where lo=i, hi=i for i in 0..15
			data: func() []byte {
				var nibbles [16]byte
				for i := range 16 {
					nibbles[i] = byte(i) | (byte(i) << 4) // lo=i, hi=i
				}
				return makeQ4_0Block(fp16One[0], fp16One[1], nibbles)
			}(),
			want: func() []float32 {
				out := make([]float32, 32)
				for i := range 16 {
					out[i] = float32(i - 8)    // low nibbles: i - 8
					out[16+i] = float32(i - 8) // high nibbles: i - 8
				}
				return out
			}(),
		},
		{
			name: "single block d=1.0 lo!=hi",
			// lo=0, hi=15 for all bytes
			data: func() []byte {
				var nibbles [16]byte
				for i := range 16 {
					nibbles[i] = 0x00 | (0x0F << 4) // lo=0, hi=15
				}
				return makeQ4_0Block(fp16One[0], fp16One[1], nibbles)
			}(),
			want: func() []float32 {
				out := make([]float32, 32)
				for i := range 16 {
					out[i] = float32(0 - 8)   // lo=0 -> -8
					out[16+i] = float32(15-8)  // hi=15 -> 7
				}
				return out
			}(),
		},
		{
			name: "single block d=0",
			data: func() []byte {
				var nibbles [16]byte
				for i := range 16 {
					nibbles[i] = 0xFF
				}
				return makeQ4_0Block(0, 0, nibbles)
			}(),
			want: make([]float32, 32),
		},
		{
			name: "two blocks",
			data: func() []byte {
				var n1, n2 [16]byte
				// Block 1: all nibbles = 8 (lo=8, hi=8) -> value = 0
				for i := range 16 {
					n1[i] = 0x88
				}
				// Block 2: all nibbles = 0 (lo=0, hi=0) -> value = -8
				b1 := makeQ4_0Block(fp16One[0], fp16One[1], n1)
				b2 := makeQ4_0Block(fp16One[0], fp16One[1], n2)
				return append(b1, b2...)
			}(),
			want: func() []float32 {
				out := make([]float32, 64)
				// Block 1: all zeros
				// Block 2: all -8
				for i := range 32 {
					out[32+i] = -8
				}
				return out
			}(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.want == nil {
				got := make([]float32, 0)
				DequantizeQ4_0(tt.data, got)
				return
			}
			got := make([]float32, len(tt.want))
			DequantizeQ4_0(tt.data, got)
			for i := range got {
				if math.Abs(float64(got[i]-tt.want[i])) > 1e-5 {
					t.Errorf("index %d: got %f, want %f", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestDequantizeIQ4NL(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want []float32
	}{
		{
			name: "empty",
			data: nil,
			want: nil,
		},
		{
			name: "single block d=1.0 each nibble 0-15 appears once",
			// 16 nibble bytes: lo=i, hi=i for i in 0..15
			data: func() []byte {
				var nibbles [16]byte
				for i := range 16 {
					nibbles[i] = byte(i) | (byte(i) << 4)
				}
				return makeQ4_0Block(fp16One[0], fp16One[1], nibbles)
			}(),
			want: func() []float32 {
				out := make([]float32, 32)
				for i := range 16 {
					out[i] = float32(kvaluesIQ4NL[i])
					out[16+i] = float32(kvaluesIQ4NL[i])
				}
				return out
			}(),
		},
		{
			name: "lookup table correctness all-zero nibbles",
			data: func() []byte {
				var nibbles [16]byte // all zeros -> lo=0, hi=0
				return makeQ4_0Block(fp16One[0], fp16One[1], nibbles)
			}(),
			want: func() []float32 {
				out := make([]float32, 32)
				for i := range 32 {
					out[i] = float32(kvaluesIQ4NL[0]) // -127
				}
				return out
			}(),
		},
		{
			name: "lookup table correctness all-fifteen nibbles",
			data: func() []byte {
				var nibbles [16]byte
				for i := range 16 {
					nibbles[i] = 0xFF // lo=15, hi=15
				}
				return makeQ4_0Block(fp16One[0], fp16One[1], nibbles)
			}(),
			want: func() []float32 {
				out := make([]float32, 32)
				for i := range 32 {
					out[i] = float32(kvaluesIQ4NL[15]) // 113
				}
				return out
			}(),
		},
		{
			name: "single block d=0",
			data: func() []byte {
				var nibbles [16]byte
				for i := range 16 {
					nibbles[i] = 0xFF
				}
				return makeQ4_0Block(0, 0, nibbles)
			}(),
			want: make([]float32, 32),
		},
		{
			name: "two blocks verify indexing",
			data: func() []byte {
				var n1, n2 [16]byte
				// Block 1: lo=0, hi=0 (kvalues[0] = -127)
				// Block 2: lo=15, hi=15 (kvalues[15] = 113)
				for i := range 16 {
					n2[i] = 0xFF
				}
				b1 := makeQ4_0Block(fp16One[0], fp16One[1], n1)
				b2 := makeQ4_0Block(fp16One[0], fp16One[1], n2)
				return append(b1, b2...)
			}(),
			want: func() []float32 {
				out := make([]float32, 64)
				for i := range 32 {
					out[i] = float32(kvaluesIQ4NL[0])   // -127
					out[32+i] = float32(kvaluesIQ4NL[15]) // 113
				}
				return out
			}(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.want == nil {
				got := make([]float32, 0)
				DequantizeIQ4NL(tt.data, got)
				return
			}
			got := make([]float32, len(tt.want))
			DequantizeIQ4NL(tt.data, got)
			for i := range got {
				if math.Abs(float64(got[i]-tt.want[i])) > 1e-5 {
					t.Errorf("index %d: got %f, want %f", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestDequantizeQ4_0_SplitLayout(t *testing.T) {
	// Explicitly verify that byte[i] low nibble -> output[i] (first 16)
	// and byte[i] high nibble -> output[i+16] (last 16)
	var nibbles [16]byte
	for i := range 16 {
		lo := byte(i)        // 0..15
		hi := byte(15 - i)   // 15..0
		nibbles[i] = lo | (hi << 4)
	}
	data := makeQ4_0Block(fp16One[0], fp16One[1], nibbles)
	got := make([]float32, 32)
	DequantizeQ4_0(data, got)

	for i := range 16 {
		wantLo := float32(i - 8)
		if math.Abs(float64(got[i]-wantLo)) > 1e-5 {
			t.Errorf("low nibble[%d]: got %f, want %f", i, got[i], wantLo)
		}
		wantHi := float32((15 - i) - 8)
		if math.Abs(float64(got[16+i]-wantHi)) > 1e-5 {
			t.Errorf("high nibble[%d]: got %f, want %f", i, got[16+i], wantHi)
		}
	}
}

func TestDequantizeIQ4NL_SplitLayout(t *testing.T) {
	// Verify that low nibble -> first 16 via lookup, high nibble -> last 16 via lookup
	var nibbles [16]byte
	for i := range 16 {
		lo := byte(i)
		hi := byte(15 - i)
		nibbles[i] = lo | (hi << 4)
	}
	data := makeQ4_0Block(fp16One[0], fp16One[1], nibbles)
	got := make([]float32, 32)
	DequantizeIQ4NL(data, got)

	for i := range 16 {
		wantLo := float32(kvaluesIQ4NL[i])
		if math.Abs(float64(got[i]-wantLo)) > 1e-5 {
			t.Errorf("low nibble[%d]: got %f, want %f", i, got[i], wantLo)
		}
		wantHi := float32(kvaluesIQ4NL[15-i])
		if math.Abs(float64(got[16+i]-wantHi)) > 1e-5 {
			t.Errorf("high nibble[%d]: got %f, want %f", i, got[16+i], wantHi)
		}
	}
}

func TestDequantizeQ8_0_DispatchVsFallback(t *testing.T) {
	var quants [32]int8
	for i := range 32 {
		quants[i] = int8(i*8 - 128)
	}
	data := makeQ8_0Block(fp16One[0], fp16One[1], quants)

	dispatch := make([]float32, 32)
	fallback := make([]float32, 32)
	DequantizeQ8_0(data, dispatch)
	BaseDequantizeQ8_0(data, fallback)

	for i := range dispatch {
		if math.Abs(float64(dispatch[i]-fallback[i])) > 1e-5 {
			t.Errorf("index %d: dispatch=%f, fallback=%f", i, dispatch[i], fallback[i])
		}
	}
}

func TestDequantizeQ4_0_DispatchVsFallback(t *testing.T) {
	var nibbles [16]byte
	for i := range 16 {
		nibbles[i] = byte(i) | (byte(15-i) << 4)
	}
	data := makeQ4_0Block(fp16One[0], fp16One[1], nibbles)

	dispatch := make([]float32, 32)
	fallback := make([]float32, 32)
	DequantizeQ4_0(data, dispatch)
	BaseDequantizeQ4_0(data, fallback)

	for i := range dispatch {
		if math.Abs(float64(dispatch[i]-fallback[i])) > 1e-5 {
			t.Errorf("index %d: dispatch=%f, fallback=%f", i, dispatch[i], fallback[i])
		}
	}
}

func TestDequantizeIQ4NL_DispatchVsFallback(t *testing.T) {
	var nibbles [16]byte
	for i := range 16 {
		nibbles[i] = byte(i) | (byte(15-i) << 4)
	}
	data := makeQ4_0Block(fp16One[0], fp16One[1], nibbles)

	dispatch := make([]float32, 32)
	fallback := make([]float32, 32)
	DequantizeIQ4NL(data, dispatch)
	BaseDequantizeIQ4NL(data, fallback)

	for i := range dispatch {
		if math.Abs(float64(dispatch[i]-fallback[i])) > 1e-5 {
			t.Errorf("index %d: dispatch=%f, fallback=%f", i, dispatch[i], fallback[i])
		}
	}
}

func TestDequantizeQ8_0_ScaleConversion(t *testing.T) {
	// Use fp16 encoding of 2.0 (0x4000)
	var quants [32]int8
	for i := range 32 {
		quants[i] = 1
	}
	data := makeQ8_0Block(0x00, 0x40, quants) // fp16 2.0 = 0x4000

	got := make([]float32, 32)
	DequantizeQ8_0(data, got)

	want := hwy.Float16ToFloat32(hwy.Float16(0x4000)) // should be 2.0
	for i := range 32 {
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Errorf("index %d: got %f, want %f", i, got[i], want)
		}
	}
}

func BenchmarkDequantizeQ8_0(b *testing.B) {
	for _, nblocks := range []int{1, 16, 64, 256, 1024} {
		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			data := make([]byte, nblocks*BlockSizeQ8_0)
			for i := range data {
				data[i] = byte(i % 256)
			}
			output := make([]float32, nblocks*QK)

			b.SetBytes(int64(nblocks * BlockSizeQ8_0))
			b.ResetTimer()
			for range b.N {
				DequantizeQ8_0(data, output)
			}
		})
	}
}

func BenchmarkDequantizeQ4_0(b *testing.B) {
	for _, nblocks := range []int{1, 16, 64, 256, 1024} {
		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			data := make([]byte, nblocks*BlockSizeQ4_0)
			for i := range data {
				data[i] = byte(i % 256)
			}
			output := make([]float32, nblocks*QK)

			b.SetBytes(int64(nblocks * BlockSizeQ4_0))
			b.ResetTimer()
			for range b.N {
				DequantizeQ4_0(data, output)
			}
		})
	}
}

func BenchmarkDequantizeIQ4NL(b *testing.B) {
	for _, nblocks := range []int{1, 16, 64, 256, 1024} {
		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			data := make([]byte, nblocks*BlockSizeIQ4NL)
			for i := range data {
				data[i] = byte(i % 256)
			}
			output := make([]float32, nblocks*QK)

			b.SetBytes(int64(nblocks * BlockSizeIQ4NL))
			b.ResetTimer()
			for range b.N {
				DequantizeIQ4NL(data, output)
			}
		})
	}
}
