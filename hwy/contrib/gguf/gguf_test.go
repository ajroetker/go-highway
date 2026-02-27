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

// --- K-quant helpers ---

// fp16Encode encodes a float32 as little-endian fp16 bytes.
func fp16Encode(f float32) [2]byte {
	return [2]byte{byte(hwy.Float32ToFloat16(f)), byte(hwy.Float32ToFloat16(f) >> 8)}
}

// makeQ6KBlock builds a Q6_K super-block (210 bytes).
func makeQ6KBlock(d float32, scales [16]int8, ql [128]byte, qh [64]byte) []byte {
	block := make([]byte, BlockSizeQ6K)
	copy(block[0:128], ql[:])
	copy(block[128:192], qh[:])
	for i, s := range scales {
		block[192+i] = byte(s)
	}
	fp := fp16Encode(d)
	block[208] = fp[0]
	block[209] = fp[1]
	return block
}

// makeQ4KBlock builds a Q4_K super-block (144 bytes).
func makeQ4KBlock(d, dmin float32, scales [12]byte, qs [128]byte) []byte {
	block := make([]byte, BlockSizeQ4K)
	fp := fp16Encode(d)
	block[0] = fp[0]
	block[1] = fp[1]
	fp = fp16Encode(dmin)
	block[2] = fp[0]
	block[3] = fp[1]
	copy(block[4:16], scales[:])
	copy(block[16:144], qs[:])
	return block
}

// makeQ5KBlock builds a Q5_K super-block (176 bytes).
func makeQ5KBlock(d, dmin float32, scales [12]byte, qs [128]byte, qh [32]byte) []byte {
	block := make([]byte, BlockSizeQ5K)
	fp := fp16Encode(d)
	block[0] = fp[0]
	block[1] = fp[1]
	fp = fp16Encode(dmin)
	block[2] = fp[0]
	block[3] = fp[1]
	copy(block[4:16], scales[:])
	copy(block[16:144], qs[:])
	copy(block[144:176], qh[:])
	return block
}

// makeQ2KBlock builds a Q2_K super-block (84 bytes).
func makeQ2KBlock(d, dmin float32, scales [16]byte, qs [64]byte) []byte {
	block := make([]byte, BlockSizeQ2K)
	copy(block[0:16], scales[:])
	fp := fp16Encode(d)
	block[16] = fp[0]
	block[17] = fp[1]
	fp = fp16Encode(dmin)
	block[18] = fp[0]
	block[19] = fp[1]
	copy(block[20:84], qs[:])
	return block
}

// makeQ3KBlock builds a Q3_K super-block (110 bytes).
func makeQ3KBlock(d float32, hmask [32]byte, qs [64]byte, scales [12]byte) []byte {
	block := make([]byte, BlockSizeQ3K)
	copy(block[0:32], hmask[:])
	copy(block[32:96], qs[:])
	copy(block[96:108], scales[:])
	fp := fp16Encode(d)
	block[108] = fp[0]
	block[109] = fp[1]
	return block
}

// --- Q6_K Tests ---

func TestDequantizeQ6K(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		DequantizeQ6K(nil, nil)
	})

	t.Run("single block d=1 uniform scales=1", func(t *testing.T) {
		// All ql=0, all qh=0 → quant=0, value = d * scale * (0 - 32) = -32
		var ql [128]byte
		var qh [64]byte
		var scales [16]int8
		for i := range scales {
			scales[i] = 1
		}
		data := makeQ6KBlock(1.0, scales, ql, qh)
		got := make([]float32, QK_K)
		DequantizeQ6K(data, got)
		for i := range got {
			if math.Abs(float64(got[i]-(-32))) > 1e-3 {
				t.Errorf("index %d: got %f, want %f", i, got[i], float32(-32))
			}
		}
	})

	t.Run("single block verify quant extraction", func(t *testing.T) {
		// Set specific quant values and verify extraction.
		// For value 0 (half=0, l=0): ql[0] low nibble = quant low4, qh[0] bits 0-1 = quant high2
		// Set ql[0] = 0x0F (low=15, high=0), qh[0] = 0x03 (bits 0-1 set)
		// q = 15 | (3 << 4) = 15 | 48 = 63, value = d * scale * (63 - 32) = 31
		var ql [128]byte
		var qh [64]byte
		var scales [16]int8
		scales[0] = 1
		ql[0] = 0x0F // low nibble = 15 for value 0
		qh[0] = 0x03 // bits 0-1 = 3 for value 0
		data := makeQ6KBlock(1.0, scales, ql, qh)
		got := make([]float32, QK_K)
		DequantizeQ6K(data, got)
		want := float32(31) // (15 | (3<<4)) - 32 = 63 - 32 = 31
		if math.Abs(float64(got[0]-want)) > 1e-3 {
			t.Errorf("value 0: got %f, want %f", got[0], want)
		}
	})

	t.Run("two blocks", func(t *testing.T) {
		var ql [128]byte
		var qh [64]byte
		var scales [16]int8
		for i := range scales {
			scales[i] = 1
		}
		b1 := makeQ6KBlock(1.0, scales, ql, qh)
		b2 := makeQ6KBlock(2.0, scales, ql, qh)
		data := append(b1, b2...)
		got := make([]float32, 2*QK_K)
		DequantizeQ6K(data, got)
		// Block 1: all quants=0, scale=1 → value = 1 * 1 * (0-32) = -32
		// Block 2: all quants=0, scale=1 → value = 2 * 1 * (0-32) = -64
		for i := 0; i < QK_K; i++ {
			if math.Abs(float64(got[i]-(-32))) > 1e-3 {
				t.Errorf("block1 index %d: got %f, want %f", i, got[i], float32(-32))
			}
		}
		for i := 0; i < QK_K; i++ {
			if math.Abs(float64(got[QK_K+i]-(-64))) > 1e-3 {
				t.Errorf("block2 index %d: got %f, want %f", i, got[QK_K+i], float32(-64))
			}
		}
	})
}

func TestDequantizeQ6K_DispatchVsFallback(t *testing.T) {
	var ql [128]byte
	var qh [64]byte
	var scales [16]int8
	for i := range ql {
		ql[i] = byte(i * 13 % 256)
	}
	for i := range qh {
		qh[i] = byte(i * 7 % 256)
	}
	for i := range scales {
		scales[i] = int8(i*3 - 8)
	}
	data := makeQ6KBlock(0.5, scales, ql, qh)
	dispatch := make([]float32, QK_K)
	fallback := make([]float32, QK_K)
	DequantizeQ6K(data, dispatch)
	BaseDequantizeQ6K(data, fallback)
	for i := range dispatch {
		if math.Abs(float64(dispatch[i]-fallback[i])) > 1e-3 {
			t.Errorf("index %d: dispatch=%f, fallback=%f", i, dispatch[i], fallback[i])
		}
	}
}

// --- Q4_K Tests ---

func TestDequantizeQ4K(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		DequantizeQ4K(nil, nil)
	})

	t.Run("single block d=1 dmin=0 all scales=1", func(t *testing.T) {
		// With dmin=0, formula reduces to: d * sc * q4
		// All qs=0 → low nibble=0, high nibble=0
		// Sub-block 0: d*sc*0 - 0 = 0; Sub-block 1: d*sc*0 - 0 = 0
		var scales [12]byte
		// For j=0..3: scales[j] = sc & 63 = 1, scales[j+4] = m & 63 = 0
		for j := 0; j < 4; j++ {
			scales[j] = 1
		}
		// For j=4..7: sc from bytes 8-11 low nibble + bytes 0-3 high 2 bits
		// Set scales[8..11] low nibble = 1
		for j := 0; j < 4; j++ {
			scales[8+j] = 1
		}
		var qs [128]byte
		data := makeQ4KBlock(1.0, 0.0, scales, qs)
		got := make([]float32, QK_K)
		DequantizeQ4K(data, got)
		for i := range got {
			if math.Abs(float64(got[i])) > 1e-3 {
				t.Errorf("index %d: got %f, want 0", i, got[i])
			}
		}
	})

	t.Run("scale unpacking", func(t *testing.T) {
		// Verify get_scale_min_k4 logic.
		// Set known scale values and verify dequantization uses correct ones.
		var scales [12]byte
		// j=0: sc = scales[0] & 63, m = scales[4] & 63
		scales[0] = 2   // sc=2
		scales[4] = 3   // m=3
		scales[1] = 0   // rest = 0
		scales[2] = 0
		scales[3] = 0
		scales[5] = 0
		scales[6] = 0
		scales[7] = 0
		scales[8] = 0
		scales[9] = 0
		scales[10] = 0
		scales[11] = 0

		var qs [128]byte
		// First chunk, sub-block 0 (low nibbles of qs[0:32]): set all to 5
		for i := 0; i < 32; i++ {
			qs[i] = 5 // low nibble = 5, high nibble = 0
		}
		data := makeQ4KBlock(1.0, 1.0, scales, qs)
		got := make([]float32, QK_K)
		DequantizeQ4K(data, got)
		// Sub-block 0 (values 0-31): d*sc*q - dmin*m = 1*2*5 - 1*3 = 7
		for i := 0; i < 32; i++ {
			want := float32(7)
			if math.Abs(float64(got[i]-want)) > 1e-3 {
				t.Errorf("index %d: got %f, want %f", i, got[i], want)
			}
		}
	})

	t.Run("two blocks", func(t *testing.T) {
		var scales [12]byte
		for j := 0; j < 4; j++ {
			scales[j] = 1
			scales[8+j] = 1
		}
		var qs [128]byte
		b1 := makeQ4KBlock(1.0, 0.0, scales, qs)
		b2 := makeQ4KBlock(1.0, 0.0, scales, qs)
		data := append(b1, b2...)
		got := make([]float32, 2*QK_K)
		DequantizeQ4K(data, got)
		for i := range got {
			if math.Abs(float64(got[i])) > 1e-3 {
				t.Errorf("index %d: got %f, want 0", i, got[i])
			}
		}
	})
}

func TestDequantizeQ4K_DispatchVsFallback(t *testing.T) {
	var scales [12]byte
	for i := range scales {
		scales[i] = byte(i*17 + 5)
	}
	var qs [128]byte
	for i := range qs {
		qs[i] = byte(i * 13 % 256)
	}
	data := makeQ4KBlock(0.5, 0.25, scales, qs)
	dispatch := make([]float32, QK_K)
	fallback := make([]float32, QK_K)
	DequantizeQ4K(data, dispatch)
	BaseDequantizeQ4K(data, fallback)
	for i := range dispatch {
		if math.Abs(float64(dispatch[i]-fallback[i])) > 1e-3 {
			t.Errorf("index %d: dispatch=%f, fallback=%f", i, dispatch[i], fallback[i])
		}
	}
}

// --- Q5_K Tests ---

func TestDequantizeQ5K(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		DequantizeQ5K(nil, nil)
	})

	t.Run("single block d=1 dmin=0 low nibbles only", func(t *testing.T) {
		// All qh=0 → high bits are 0, so quant = low nibble only (0..15 range)
		var scales [12]byte
		for j := 0; j < 4; j++ {
			scales[j] = 1
			scales[8+j] = 1
		}
		var qs [128]byte
		// Set low nibbles to 7, high nibbles to 0
		for i := range qs {
			qs[i] = 7 // low=7, high=0
		}
		var qh [32]byte // all zeros → no high bits
		data := makeQ5KBlock(1.0, 0.0, scales, qs, qh)
		got := make([]float32, QK_K)
		DequantizeQ5K(data, got)
		// Sub-block 0 (low nibbles): d*sc*7 = 7
		for i := 0; i < 32; i++ {
			want := float32(7)
			if math.Abs(float64(got[i]-want)) > 1e-3 {
				t.Errorf("index %d: got %f, want %f", i, got[i], want)
			}
		}
	})

	t.Run("high bit adds 16", func(t *testing.T) {
		var scales [12]byte
		for j := 0; j < 4; j++ {
			scales[j] = 1
			scales[8+j] = 1
		}
		var qs [128]byte // all zeros → low nibble = 0
		var qh [32]byte
		// Set high bit for sub-block 0 (u1=1, chunk=0): qh[l] bit 0
		for l := 0; l < 32; l++ {
			qh[l] = 1 // bit 0 set → u1 match for chunk 0
		}
		data := makeQ5KBlock(1.0, 0.0, scales, qs, qh)
		got := make([]float32, QK_K)
		DequantizeQ5K(data, got)
		// Sub-block 0: quant = 0 + 16 = 16, value = 1*1*16 = 16
		for i := 0; i < 32; i++ {
			want := float32(16)
			if math.Abs(float64(got[i]-want)) > 1e-3 {
				t.Errorf("index %d: got %f, want %f", i, got[i], want)
			}
		}
	})

	t.Run("two blocks", func(t *testing.T) {
		var scales [12]byte
		for j := 0; j < 4; j++ {
			scales[j] = 1
			scales[8+j] = 1
		}
		var qs [128]byte
		var qh [32]byte
		b1 := makeQ5KBlock(1.0, 0.0, scales, qs, qh)
		b2 := makeQ5KBlock(2.0, 0.0, scales, qs, qh)
		data := append(b1, b2...)
		got := make([]float32, 2*QK_K)
		DequantizeQ5K(data, got)
		for i := 0; i < QK_K; i++ {
			if math.Abs(float64(got[i])) > 1e-3 {
				t.Errorf("block1 index %d: got %f, want 0", i, got[i])
			}
		}
	})
}

func TestDequantizeQ5K_DispatchVsFallback(t *testing.T) {
	var scales [12]byte
	for i := range scales {
		scales[i] = byte(i*11 + 3)
	}
	var qs [128]byte
	for i := range qs {
		qs[i] = byte(i * 7 % 256)
	}
	var qh [32]byte
	for i := range qh {
		qh[i] = byte(i * 19 % 256)
	}
	data := makeQ5KBlock(0.5, 0.25, scales, qs, qh)
	dispatch := make([]float32, QK_K)
	fallback := make([]float32, QK_K)
	DequantizeQ5K(data, dispatch)
	BaseDequantizeQ5K(data, fallback)
	for i := range dispatch {
		if math.Abs(float64(dispatch[i]-fallback[i])) > 1e-3 {
			t.Errorf("index %d: dispatch=%f, fallback=%f", i, dispatch[i], fallback[i])
		}
	}
}

// --- Q2_K Tests ---

func TestDequantizeQ2K(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		DequantizeQ2K(nil, nil)
	})

	t.Run("single block d=1 dmin=0 all quants=3", func(t *testing.T) {
		// scales: sc=1 (low 4 bits), m=0 (high 4 bits)
		var scales [16]byte
		for i := range scales {
			scales[i] = 1 // sc=1, m=0
		}
		// qs: all bits set → quant=3 at every shift
		var qs [64]byte
		for i := range qs {
			qs[i] = 0xFF // all 2-bit values = 3
		}
		data := makeQ2KBlock(1.0, 0.0, scales, qs)
		got := make([]float32, QK_K)
		DequantizeQ2K(data, got)
		// d*sc*3 - dmin*m = 1*1*3 - 0 = 3
		for i := range got {
			want := float32(3)
			if math.Abs(float64(got[i]-want)) > 1e-3 {
				t.Errorf("index %d: got %f, want %f", i, got[i], want)
			}
		}
	})

	t.Run("asymmetric formula with min", func(t *testing.T) {
		// sc=2, m=3
		var scales [16]byte
		for i := range scales {
			scales[i] = 2 | (3 << 4) // sc=2, m=3
		}
		var qs [64]byte
		for i := range qs {
			qs[i] = 0x55 // all 2-bit values = 1 (01 01 01 01)
		}
		data := makeQ2KBlock(1.0, 1.0, scales, qs)
		got := make([]float32, QK_K)
		DequantizeQ2K(data, got)
		// d*sc*q - dmin*m = 1*2*1 - 1*3 = -1
		for i := range got {
			want := float32(-1)
			if math.Abs(float64(got[i]-want)) > 1e-3 {
				t.Errorf("index %d: got %f, want %f", i, got[i], want)
			}
		}
	})

	t.Run("two blocks", func(t *testing.T) {
		var scales [16]byte
		for i := range scales {
			scales[i] = 1
		}
		var qs [64]byte
		b1 := makeQ2KBlock(1.0, 0.0, scales, qs)
		b2 := makeQ2KBlock(1.0, 0.0, scales, qs)
		data := append(b1, b2...)
		got := make([]float32, 2*QK_K)
		DequantizeQ2K(data, got)
		for i := range got {
			if math.Abs(float64(got[i])) > 1e-3 {
				t.Errorf("index %d: got %f, want 0", i, got[i])
			}
		}
	})
}

func TestDequantizeQ2K_DispatchVsFallback(t *testing.T) {
	var scales [16]byte
	for i := range scales {
		scales[i] = byte(i*5 + 17)
	}
	var qs [64]byte
	for i := range qs {
		qs[i] = byte(i * 23 % 256)
	}
	data := makeQ2KBlock(0.5, 0.25, scales, qs)
	dispatch := make([]float32, QK_K)
	fallback := make([]float32, QK_K)
	DequantizeQ2K(data, dispatch)
	BaseDequantizeQ2K(data, fallback)
	for i := range dispatch {
		if math.Abs(float64(dispatch[i]-fallback[i])) > 1e-3 {
			t.Errorf("index %d: dispatch=%f, fallback=%f", i, dispatch[i], fallback[i])
		}
	}
}

// --- Q3_K Tests ---

func TestDequantizeQ3K(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		DequantizeQ3K(nil, nil)
	})

	t.Run("single block d=1 all hmask=1 qs=0", func(t *testing.T) {
		// hmask[l] bit 0 set → high1=1 for chunk0/group0
		// qs=0 → low2=0
		// quant = 0 + 1*4 - 4 = 0
		// value = d * (scale-32) * 0 = 0
		var hmask [32]byte
		for l := range hmask {
			hmask[l] = 0xFF // all hmask bits set
		}
		var qs [64]byte // all zeros
		// Scales: all raw scale values = 32 → scale-32 = 0 → all output = 0
		// Use scale value 33 instead so scale-32 = 1
		var scaleData [12]byte
		// For scales[0..3]: (r[i] & 0x0F) | ((r[8+i] & 0x03) << 4)
		// Want rawScales[j] = 33 for all j. 33 = 0x21
		// For i<4: (r[i]&0x0F) | ((r[8+i]&0x03)<<4) = 1 | (2<<4) = 1|32 = 33 ✓
		// So r[0..3] low nibble = 1, r[8..11] low 2 bits = 2
		for i := 0; i < 4; i++ {
			scaleData[i] = 0x01     // low 4 bits = 1
			scaleData[4+i] = 0x01   // low 4 bits = 1 (for scales 4-7)
			scaleData[8+i] = 0x0A   // bits: 00 00 10 10 → low 2 = 2, bits 2-3 = 2, bits 4-5 = 0, bits 6-7 = 0
		}
		// This gives rawScales[0..3] = 1 | (2<<4) = 33
		// rawScales[4..7] = 1 | ((10>>2 & 3)<<4) = 1 | (2<<4) = 33
		// rawScales[8..11] = ((r[i]>>4)&0x0F) | (((r[8+i]>>4)&0x03)<<4) = 0 | (0<<4) = 0 → NOT 33
		// This is getting complex. Let me simplify by just using zero scales.
		// scale-32 = -32, quant=0 → value = d * (-32) * 0 = 0
		scaleData = [12]byte{} // all zeros → rawScales all 0 → scale-32 = -32
		data := makeQ3KBlock(1.0, hmask, qs, scaleData)
		got := make([]float32, QK_K)
		DequantizeQ3K(data, got)
		// quant = 0 + 1*4 - 4 = 0 (hmask all set, low2=0)
		// value = d * (0-32) * 0 = 0
		for i := range got {
			if math.Abs(float64(got[i])) > 1e-3 {
				t.Errorf("index %d: got %f, want 0", i, got[i])
			}
		}
	})

	t.Run("hmask controls high bit", func(t *testing.T) {
		// hmask all zeros → high1=0 for all
		// qs all zeros → low2=0
		// quant = 0 + 0*4 - 4 = -4
		// Set a known scale to verify
		var hmask [32]byte // all zeros → high1=0
		var qs [64]byte    // all zeros → low2=0
		var scaleData [12]byte
		// Want rawScales[0] = 33 (so scale-32 = 1)
		// rawScales[0] = (scaleData[0]&0x0F) | ((scaleData[8]&0x03)<<4)
		scaleData[0] = 1  // low nibble = 1
		scaleData[8] = 2  // low 2 bits = 2 → (2<<4) = 32 → total = 33
		data := makeQ3KBlock(1.0, hmask, qs, scaleData)
		got := make([]float32, QK_K)
		DequantizeQ3K(data, got)
		// Sub-block 0 (values 0-15): d * (33-32) * (-4) = 1 * 1 * -4 = -4
		for i := 0; i < 16; i++ {
			want := float32(-4)
			if math.Abs(float64(got[i]-want)) > 1e-3 {
				t.Errorf("index %d: got %f, want %f", i, got[i], want)
			}
		}
	})

	t.Run("two blocks", func(t *testing.T) {
		var hmask [32]byte
		for l := range hmask {
			hmask[l] = 0xFF
		}
		var qs [64]byte
		var scaleData [12]byte
		b1 := makeQ3KBlock(1.0, hmask, qs, scaleData)
		b2 := makeQ3KBlock(2.0, hmask, qs, scaleData)
		data := append(b1, b2...)
		got := make([]float32, 2*QK_K)
		DequantizeQ3K(data, got)
		// All quants = 0, all scales = 0-32 = -32
		// value = d * (-32) * 0 = 0
		for i := range got {
			if math.Abs(float64(got[i])) > 1e-3 {
				t.Errorf("index %d: got %f, want 0", i, got[i])
			}
		}
	})
}

func TestDequantizeQ3K_DispatchVsFallback(t *testing.T) {
	var hmask [32]byte
	for i := range hmask {
		hmask[i] = byte(i * 37 % 256)
	}
	var qs [64]byte
	for i := range qs {
		qs[i] = byte(i * 11 % 256)
	}
	var scaleData [12]byte
	for i := range scaleData {
		scaleData[i] = byte(i*29 + 7)
	}
	data := makeQ3KBlock(0.5, hmask, qs, scaleData)
	dispatch := make([]float32, QK_K)
	fallback := make([]float32, QK_K)
	DequantizeQ3K(data, dispatch)
	BaseDequantizeQ3K(data, fallback)
	for i := range dispatch {
		if math.Abs(float64(dispatch[i]-fallback[i])) > 1e-3 {
			t.Errorf("index %d: dispatch=%f, fallback=%f", i, dispatch[i], fallback[i])
		}
	}
}

// --- K-quant Benchmarks ---

func BenchmarkDequantizeQ6K(b *testing.B) {
	for _, nblocks := range []int{1, 4, 16, 64, 256} {
		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			data := make([]byte, nblocks*BlockSizeQ6K)
			for i := range data {
				data[i] = byte(i % 256)
			}
			output := make([]float32, nblocks*QK_K)
			b.SetBytes(int64(nblocks * BlockSizeQ6K))
			b.ResetTimer()
			for range b.N {
				DequantizeQ6K(data, output)
			}
		})
	}
}

func BenchmarkDequantizeQ4K(b *testing.B) {
	for _, nblocks := range []int{1, 4, 16, 64, 256} {
		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			data := make([]byte, nblocks*BlockSizeQ4K)
			for i := range data {
				data[i] = byte(i % 256)
			}
			output := make([]float32, nblocks*QK_K)
			b.SetBytes(int64(nblocks * BlockSizeQ4K))
			b.ResetTimer()
			for range b.N {
				DequantizeQ4K(data, output)
			}
		})
	}
}

func BenchmarkDequantizeQ5K(b *testing.B) {
	for _, nblocks := range []int{1, 4, 16, 64, 256} {
		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			data := make([]byte, nblocks*BlockSizeQ5K)
			for i := range data {
				data[i] = byte(i % 256)
			}
			output := make([]float32, nblocks*QK_K)
			b.SetBytes(int64(nblocks * BlockSizeQ5K))
			b.ResetTimer()
			for range b.N {
				DequantizeQ5K(data, output)
			}
		})
	}
}

func BenchmarkDequantizeQ2K(b *testing.B) {
	for _, nblocks := range []int{1, 4, 16, 64, 256} {
		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			data := make([]byte, nblocks*BlockSizeQ2K)
			for i := range data {
				data[i] = byte(i % 256)
			}
			output := make([]float32, nblocks*QK_K)
			b.SetBytes(int64(nblocks * BlockSizeQ2K))
			b.ResetTimer()
			for range b.N {
				DequantizeQ2K(data, output)
			}
		})
	}
}

func BenchmarkDequantizeQ3K(b *testing.B) {
	for _, nblocks := range []int{1, 4, 16, 64, 256} {
		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			data := make([]byte, nblocks*BlockSizeQ3K)
			for i := range data {
				data[i] = byte(i % 256)
			}
			output := make([]float32, nblocks*QK_K)
			b.SetBytes(int64(nblocks * BlockSizeQ3K))
			b.ResetTimer()
			for range b.N {
				DequantizeQ3K(data, output)
			}
		})
	}
}
