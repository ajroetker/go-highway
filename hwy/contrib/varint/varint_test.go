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

package varint

import (
	"math"
	"testing"
)

// encodeUvarint encodes a uint64 as a standard LEB128 varint.
// Used as a reference implementation for testing.
func encodeUvarint(v uint64) []byte {
	var buf [10]byte
	n := 0
	for v >= 0x80 {
		buf[n] = byte(v) | 0x80
		v >>= 7
		n++
	}
	buf[n] = byte(v)
	return buf[:n+1]
}

// encodeMultipleUvarints encodes multiple values into a single buffer.
func encodeMultipleUvarints(values ...uint64) []byte {
	var result []byte
	for _, v := range values {
		result = append(result, encodeUvarint(v)...)
	}
	return result
}

// ============================================================================
// Tests for varint_base.go
// ============================================================================

func TestBaseFindVarintEnds(t *testing.T) {
	tests := []struct {
		name     string
		input    []byte
		expected uint32
	}{
		{
			name:     "empty buffer",
			input:    []byte{},
			expected: 0,
		},
		{
			name:     "single byte value (terminates at index 0)",
			input:    []byte{0x01},
			expected: 0b00000001, // bit 0 set
		},
		{
			name:     "single byte max (127)",
			input:    []byte{0x7F},
			expected: 0b00000001, // bit 0 set
		},
		{
			name:     "two-byte varint (128)",
			input:    []byte{0x80, 0x01},
			expected: 0b00000010, // bit 1 set (index 1 terminates)
		},
		{
			name:     "two single-byte values",
			input:    []byte{0x01, 0x02},
			expected: 0b00000011, // bits 0 and 1 set
		},
		{
			name:     "continuation byte then terminator",
			input:    []byte{0x80, 0x80, 0x01},
			expected: 0b00000100, // bit 2 set
		},
		{
			name:     "multiple varints mixed",
			input:    []byte{0x01, 0x80, 0x01, 0x7F},
			expected: 0b00001101, // bits 0, 2, 3 set
		},
		{
			name:     "all continuation bytes",
			input:    []byte{0x80, 0x80, 0x80, 0x80},
			expected: 0, // no terminators
		},
		{
			name:     "all terminator bytes",
			input:    []byte{0x01, 0x02, 0x03, 0x04},
			expected: 0b00001111, // all bits set
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseFindVarintEnds(tt.input)
			if got != tt.expected {
				t.Errorf("BaseFindVarintEnds(%v) = %032b, want %032b", tt.input, got, tt.expected)
			}
		})
	}
}

func TestBaseFindVarintEnds_LongBuffer(t *testing.T) {
	// Test with buffer longer than 32 bytes (should only consider first 32)
	input := make([]byte, 64)
	for i := range input {
		input[i] = 0x01 // all terminators
	}
	got := BaseFindVarintEnds(input)
	expected := uint32(0xFFFFFFFF) // only first 32 bits should be set
	if got != expected {
		t.Errorf("BaseFindVarintEnds with 64-byte buffer = %032b, want %032b", got, expected)
	}
}

func TestBaseDecodeUvarint64Batch_SingleByteValues(t *testing.T) {
	tests := []struct {
		name     string
		value    uint64
		encoded  []byte
	}{
		{"zero", 0, []byte{0x00}},
		{"one", 1, []byte{0x01}},
		{"127", 127, []byte{0x7F}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint64, 1)
			decoded, consumed := BaseDecodeUvarint64Batch(tt.encoded, dst, 1)
			if decoded != 1 || consumed != 1 {
				t.Errorf("decoded=%d, consumed=%d, want decoded=1, consumed=1", decoded, consumed)
			}
			if dst[0] != tt.value {
				t.Errorf("got %d, want %d", dst[0], tt.value)
			}
		})
	}
}

func TestBaseDecodeUvarint64Batch_MultiByteValues(t *testing.T) {
	tests := []struct {
		name     string
		value    uint64
		encoded  []byte
	}{
		{"128", 128, []byte{0x80, 0x01}},
		{"300", 300, []byte{0xAC, 0x02}},
		{"16384", 16384, []byte{0x80, 0x80, 0x01}},
		{"maxUint64", math.MaxUint64, []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint64, 1)
			decoded, consumed := BaseDecodeUvarint64Batch(tt.encoded, dst, 1)
			if decoded != 1 || consumed != len(tt.encoded) {
				t.Errorf("decoded=%d, consumed=%d, want decoded=1, consumed=%d", decoded, consumed, len(tt.encoded))
			}
			if dst[0] != tt.value {
				t.Errorf("got %d, want %d", dst[0], tt.value)
			}
		})
	}
}

func TestBaseDecodeUvarint64Batch_MultipleBatch(t *testing.T) {
	// Encode multiple values
	values := []uint64{1, 127, 128, 300, 16384, 0, 255}
	encoded := encodeMultipleUvarints(values...)

	dst := make([]uint64, len(values))
	decoded, consumed := BaseDecodeUvarint64Batch(encoded, dst, len(values))

	if decoded != len(values) {
		t.Errorf("decoded %d values, want %d", decoded, len(values))
	}
	if consumed != len(encoded) {
		t.Errorf("consumed %d bytes, want %d", consumed, len(encoded))
	}

	for i, want := range values {
		if dst[i] != want {
			t.Errorf("dst[%d] = %d, want %d", i, dst[i], want)
		}
	}
}

func TestBaseDecodeUvarint64Batch_EdgeCases(t *testing.T) {
	t.Run("empty buffer", func(t *testing.T) {
		dst := make([]uint64, 10)
		decoded, consumed := BaseDecodeUvarint64Batch([]byte{}, dst, 10)
		if decoded != 0 || consumed != 0 {
			t.Errorf("decoded=%d, consumed=%d, want 0, 0", decoded, consumed)
		}
	})

	t.Run("n=0", func(t *testing.T) {
		dst := make([]uint64, 10)
		decoded, consumed := BaseDecodeUvarint64Batch([]byte{0x01, 0x02}, dst, 0)
		if decoded != 0 || consumed != 0 {
			t.Errorf("decoded=%d, consumed=%d, want 0, 0", decoded, consumed)
		}
	})

	t.Run("dst too small", func(t *testing.T) {
		encoded := encodeMultipleUvarints(1, 2, 3, 4, 5)
		dst := make([]uint64, 3)
		decoded, _ := BaseDecodeUvarint64Batch(encoded, dst, 10)
		if decoded != 3 {
			t.Errorf("decoded=%d, want 3", decoded)
		}
		for i := 0; i < 3; i++ {
			if dst[i] != uint64(i+1) {
				t.Errorf("dst[%d] = %d, want %d", i, dst[i], i+1)
			}
		}
	})

	t.Run("incomplete varint", func(t *testing.T) {
		// Incomplete varint: continuation bytes with no terminator
		dst := make([]uint64, 10)
		decoded, consumed := BaseDecodeUvarint64Batch([]byte{0x80, 0x80}, dst, 10)
		if decoded != 0 || consumed != 0 {
			t.Errorf("decoded=%d, consumed=%d, want 0, 0", decoded, consumed)
		}
	})

	t.Run("partial decode with incomplete at end", func(t *testing.T) {
		// First complete varint, then incomplete
		encoded := append(encodeUvarint(42), 0x80, 0x80)
		dst := make([]uint64, 10)
		decoded, _ := BaseDecodeUvarint64Batch(encoded, dst, 10)
		if decoded != 1 {
			t.Errorf("decoded=%d, want 1", decoded)
		}
		if dst[0] != 42 {
			t.Errorf("dst[0] = %d, want 42", dst[0])
		}
	})

	t.Run("overflow protection", func(t *testing.T) {
		// 11 continuation bytes would overflow uint64
		overflow := make([]byte, 11)
		for i := range overflow {
			overflow[i] = 0x80
		}
		overflow[10] = 0x02 // terminate with value that causes overflow
		dst := make([]uint64, 1)
		decoded, consumed := BaseDecodeUvarint64Batch(overflow, dst, 1)
		if decoded != 0 || consumed != 0 {
			t.Errorf("decoded=%d, consumed=%d, want 0, 0 for overflow", decoded, consumed)
		}
	})
}

func TestBaseDecode2Uvarint64(t *testing.T) {
	tests := []struct {
		name           string
		v1, v2         uint64
		wantV1, wantV2 uint64
		wantConsumed   int
	}{
		{"single byte each", 1, 2, 1, 2, 2},
		{"mixed sizes", 1, 300, 1, 300, 3},
		{"both multi-byte", 128, 256, 128, 256, 4},
		{"zeros", 0, 0, 0, 0, 2},
		{"large values", 16384, 2097152, 16384, 2097152, 7}, // 3 bytes + 4 bytes = 7
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoded := encodeMultipleUvarints(tt.v1, tt.v2)
			v1, v2, consumed := BaseDecode2Uvarint64(encoded)
			if v1 != tt.wantV1 || v2 != tt.wantV2 || consumed != tt.wantConsumed {
				t.Errorf("BaseDecode2Uvarint64 = (%d, %d, %d), want (%d, %d, %d)",
					v1, v2, consumed, tt.wantV1, tt.wantV2, tt.wantConsumed)
			}
		})
	}
}

func TestBaseDecode2Uvarint64_EdgeCases(t *testing.T) {
	t.Run("empty buffer", func(t *testing.T) {
		v1, v2, consumed := BaseDecode2Uvarint64([]byte{})
		if v1 != 0 || v2 != 0 || consumed != 0 {
			t.Errorf("got (%d, %d, %d), want (0, 0, 0)", v1, v2, consumed)
		}
	})

	t.Run("only one varint", func(t *testing.T) {
		v1, v2, c := BaseDecode2Uvarint64(encodeUvarint(42))
		if v1 != 0 || v2 != 0 || c != 0 {
			t.Errorf("got (%d, %d, %d), want (0, 0, 0)", v1, v2, c)
		}
	})

	t.Run("first incomplete", func(t *testing.T) {
		v1, v2, c := BaseDecode2Uvarint64([]byte{0x80, 0x80})
		if v1 != 0 || v2 != 0 || c != 0 {
			t.Errorf("got (%d, %d, %d), want (0, 0, 0)", v1, v2, c)
		}
	})

	t.Run("second incomplete", func(t *testing.T) {
		encoded := append(encodeUvarint(1), 0x80, 0x80)
		v1, v2, c := BaseDecode2Uvarint64(encoded)
		if v1 != 0 || v2 != 0 || c != 0 {
			t.Errorf("got (%d, %d, %d), want (0, 0, 0)", v1, v2, c)
		}
	})
}

func TestBaseDecode5Uvarint64(t *testing.T) {
	tests := []struct {
		name   string
		values [5]uint64
	}{
		{"all single byte", [5]uint64{1, 2, 3, 4, 5}},
		{"mixed sizes", [5]uint64{1, 128, 300, 16384, 0}},
		{"all zeros", [5]uint64{0, 0, 0, 0, 0}},
		{"location fields", [5]uint64{10, 1000, 5000, 5050, 3}}, // typical field, pos, start, end, numArr
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoded := encodeMultipleUvarints(tt.values[0], tt.values[1], tt.values[2], tt.values[3], tt.values[4])
			values, consumed := BaseDecode5Uvarint64(encoded)
			if values != tt.values {
				t.Errorf("values = %v, want %v", values, tt.values)
			}
			// Consumed should equal the length of the encoded data
			if consumed != len(encoded) {
				t.Errorf("consumed = %d, want %d", consumed, len(encoded))
			}
		})
	}
}

func TestBaseDecode5Uvarint64_EdgeCases(t *testing.T) {
	t.Run("empty buffer", func(t *testing.T) {
		values, consumed := BaseDecode5Uvarint64([]byte{})
		if values != [5]uint64{} || consumed != 0 {
			t.Errorf("got (%v, %d), want ([0,0,0,0,0], 0)", values, consumed)
		}
	})

	t.Run("only 4 varints", func(t *testing.T) {
		encoded := encodeMultipleUvarints(1, 2, 3, 4)
		values, consumed := BaseDecode5Uvarint64(encoded)
		if values != [5]uint64{} || consumed != 0 {
			t.Errorf("got (%v, %d), want ([0,0,0,0,0], 0)", values, consumed)
		}
	})

	t.Run("5th varint incomplete", func(t *testing.T) {
		encoded := encodeMultipleUvarints(1, 2, 3, 4)
		encoded = append(encoded, 0x80, 0x80) // incomplete 5th
		values, consumed := BaseDecode5Uvarint64(encoded)
		if values != [5]uint64{} || consumed != 0 {
			t.Errorf("got (%v, %d), want ([0,0,0,0,0], 0)", values, consumed)
		}
	})
}

// ============================================================================
// Tests for groupvarint_base.go
// ============================================================================

func TestBaseDecodeGroupVarint32_RoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		values [4]uint32
	}{
		{"small values", [4]uint32{1, 2, 3, 4}},
		{"mixed sizes", [4]uint32{1, 127, 128, 65535}},
		{"zeros", [4]uint32{0, 0, 0, 0}},
		{"max values", [4]uint32{math.MaxUint32, math.MaxUint32, math.MaxUint32, math.MaxUint32}},
		{"powers of 2", [4]uint32{1, 256, 65536, 16777216}},
		{"example from docs", [4]uint32{300, 5, 1000, 2}},
		{"boundary values", [4]uint32{255, 256, 65535, 65536}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]byte, 17)
			n := BaseEncodeGroupVarint32(tt.values, dst)
			if n == 0 {
				t.Fatal("encode returned 0")
			}

			decoded, consumed := BaseDecodeGroupVarint32(dst[:n])
			if decoded != tt.values {
				t.Errorf("round-trip failed: got %v, want %v", decoded, tt.values)
			}
			if consumed != n {
				t.Errorf("consumed = %d, want %d", consumed, n)
			}
		})
	}
}

func TestBaseDecodeGroupVarint32_KnownEncoding(t *testing.T) {
	// [300, 5, 1000, 2]
	// 300 = 0x12C -> 2 bytes (0x2C, 0x01)
	// 5 = 0x05 -> 1 byte (0x05)
	// 1000 = 0x3E8 -> 2 bytes (0xE8, 0x03)
	// 2 = 0x02 -> 1 byte (0x02)
	// Control: len0-1=1, len1-1=0, len2-1=1, len3-1=0 -> (1<<0)|(0<<2)|(1<<4)|(0<<6) = 0x11
	expected := []byte{0x11, 0x2C, 0x01, 0x05, 0xE8, 0x03, 0x02}

	values := [4]uint32{300, 5, 1000, 2}
	dst := make([]byte, 17)
	n := BaseEncodeGroupVarint32(values, dst)

	if n != len(expected) {
		t.Errorf("encoded length = %d, want %d", n, len(expected))
	}

	for i := 0; i < n; i++ {
		if dst[i] != expected[i] {
			t.Errorf("byte %d: got 0x%02X, want 0x%02X", i, dst[i], expected[i])
		}
	}

	// Decode and verify
	decoded, consumed := BaseDecodeGroupVarint32(expected)
	if decoded != values {
		t.Errorf("decode failed: got %v, want %v", decoded, values)
	}
	if consumed != len(expected) {
		t.Errorf("consumed = %d, want %d", consumed, len(expected))
	}
}

func TestBaseDecodeGroupVarint32_EdgeCases(t *testing.T) {
	t.Run("empty buffer", func(t *testing.T) {
		values, consumed := BaseDecodeGroupVarint32([]byte{})
		if values != [4]uint32{} || consumed != 0 {
			t.Errorf("got (%v, %d), want ([0,0,0,0], 0)", values, consumed)
		}
	})

	t.Run("buffer too short - only control", func(t *testing.T) {
		values, consumed := BaseDecodeGroupVarint32([]byte{0x00}) // control says 4x1-byte but missing data
		if values != [4]uint32{} || consumed != 0 {
			t.Errorf("got (%v, %d), want ([0,0,0,0], 0)", values, consumed)
		}
	})

	t.Run("buffer too short - partial data", func(t *testing.T) {
		// Control 0xFF means 4x4-byte values = 17 bytes total
		values, consumed := BaseDecodeGroupVarint32([]byte{0xFF, 0x01, 0x02, 0x03})
		if values != [4]uint32{} || consumed != 0 {
			t.Errorf("got (%v, %d), want ([0,0,0,0], 0)", values, consumed)
		}
	})

	t.Run("dst too small", func(t *testing.T) {
		values := [4]uint32{1, 2, 3, 4}
		dst := make([]byte, 5) // too small
		n := BaseEncodeGroupVarint32(values, dst)
		if n != 0 {
			t.Errorf("encode to small buffer = %d, want 0", n)
		}
	})
}

func TestBaseDecodeGroupVarint64_RoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		values [4]uint64
	}{
		{"small values", [4]uint64{1, 2, 3, 4}},
		{"mixed sizes", [4]uint64{1, 127, 128, 65535}},
		{"zeros", [4]uint64{0, 0, 0, 0}},
		{"max values", [4]uint64{math.MaxUint64, math.MaxUint64, math.MaxUint64, math.MaxUint64}},
		{"large values", [4]uint64{1 << 40, 1 << 48, 1 << 56, math.MaxUint64}},
		{"32-bit boundary", [4]uint64{math.MaxUint32, math.MaxUint32 + 1, 0, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]byte, 34)
			n := BaseEncodeGroupVarint64(tt.values, dst)
			if n == 0 {
				t.Fatal("encode returned 0")
			}

			decoded, consumed := BaseDecodeGroupVarint64(dst[:n])
			if decoded != tt.values {
				t.Errorf("round-trip failed: got %v, want %v", decoded, tt.values)
			}
			if consumed != n {
				t.Errorf("consumed = %d, want %d", consumed, n)
			}
		})
	}
}

func TestBaseDecodeGroupVarint64_EdgeCases(t *testing.T) {
	t.Run("empty buffer", func(t *testing.T) {
		values, consumed := BaseDecodeGroupVarint64([]byte{})
		if values != [4]uint64{} || consumed != 0 {
			t.Errorf("got (%v, %d), want ([0,0,0,0], 0)", values, consumed)
		}
	})

	t.Run("only one control byte", func(t *testing.T) {
		values, consumed := BaseDecodeGroupVarint64([]byte{0x00})
		if values != [4]uint64{} || consumed != 0 {
			t.Errorf("got (%v, %d), want ([0,0,0,0], 0)", values, consumed)
		}
	})

	t.Run("buffer too short", func(t *testing.T) {
		// Control 0xFF,0x0F means 4x8-byte values = 34 bytes total
		values, consumed := BaseDecodeGroupVarint64([]byte{0xFF, 0x0F, 0x01, 0x02})
		if values != [4]uint64{} || consumed != 0 {
			t.Errorf("got (%v, %d), want ([0,0,0,0], 0)", values, consumed)
		}
	})

	t.Run("dst too small", func(t *testing.T) {
		values := [4]uint64{1, 2, 3, 4}
		dst := make([]byte, 10) // too small
		n := BaseEncodeGroupVarint64(values, dst)
		if n != 0 {
			t.Errorf("encode to small buffer = %d, want 0", n)
		}
	})
}

func TestBaseGroupVarint32Len(t *testing.T) {
	tests := []struct {
		name   string
		values [4]uint32
		want   int
	}{
		{"all 1-byte", [4]uint32{1, 2, 3, 4}, 5},           // 1 + 4*1
		{"all 4-byte", [4]uint32{1 << 24, 1 << 24, 1 << 24, 1 << 24}, 17}, // 1 + 4*4
		{"mixed", [4]uint32{300, 5, 1000, 2}, 7},           // 1 + 2 + 1 + 2 + 1
		{"boundary", [4]uint32{255, 256, 65535, 65536}, 9}, // 1 + 1 + 2 + 2 + 3 (65536 needs 3 bytes)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseGroupVarint32Len(tt.values)
			if got != tt.want {
				t.Errorf("BaseGroupVarint32Len(%v) = %d, want %d", tt.values, got, tt.want)
			}

			// Verify against actual encoding
			dst := make([]byte, 17)
			encoded := BaseEncodeGroupVarint32(tt.values, dst)
			if got != encoded {
				t.Errorf("length prediction %d != actual encoded %d", got, encoded)
			}
		})
	}
}

func TestBaseGroupVarint64Len(t *testing.T) {
	tests := []struct {
		name   string
		values [4]uint64
		want   int
	}{
		{"all 1-byte", [4]uint64{1, 2, 3, 4}, 6},            // 2 + 4*1
		{"all 8-byte", [4]uint64{1 << 56, 1 << 56, 1 << 56, 1 << 56}, 34}, // 2 + 4*8
		{"mixed", [4]uint64{300, 5, 1000, 2}, 8},            // 2 + 2 + 1 + 2 + 1
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseGroupVarint64Len(tt.values)
			if got != tt.want {
				t.Errorf("BaseGroupVarint64Len(%v) = %d, want %d", tt.values, got, tt.want)
			}

			// Verify against actual encoding
			dst := make([]byte, 34)
			encoded := BaseEncodeGroupVarint64(tt.values, dst)
			if got != encoded {
				t.Errorf("length prediction %d != actual encoded %d", got, encoded)
			}
		})
	}
}

func TestBytesNeeded32(t *testing.T) {
	tests := []struct {
		value uint32
		want  int
	}{
		{0, 1},
		{1, 1},
		{255, 1},
		{256, 2},
		{65535, 2},
		{65536, 3},
		{16777215, 3},
		{16777216, 4},
		{math.MaxUint32, 4},
	}

	for _, tt := range tests {
		got := bytesNeeded32(tt.value)
		if got != tt.want {
			t.Errorf("bytesNeeded32(%d) = %d, want %d", tt.value, got, tt.want)
		}
	}
}

func TestBytesNeeded64(t *testing.T) {
	tests := []struct {
		value uint64
		want  int
	}{
		{0, 1},
		{1, 1},
		{255, 1},
		{256, 2},
		{65535, 2},
		{65536, 3},
		{16777215, 3},
		{16777216, 4},
		{math.MaxUint32, 4},
		{math.MaxUint32 + 1, 5},
		{1<<40 - 1, 5},
		{1 << 40, 6},
		{1<<48 - 1, 6},
		{1 << 48, 7},
		{1<<56 - 1, 7},
		{1 << 56, 8},
		{math.MaxUint64, 8},
	}

	for _, tt := range tests {
		got := bytesNeeded64(tt.value)
		if got != tt.want {
			t.Errorf("bytesNeeded64(%d) = %d, want %d", tt.value, got, tt.want)
		}
	}
}

func TestTrailingZeros32(t *testing.T) {
	tests := []struct {
		input uint32
		want  int
	}{
		{0, 32},
		{1, 0},
		{2, 1},
		{4, 2},
		{8, 3},
		{0x80000000, 31},
		{0b00001010, 1}, // from the example in FindVarintEnds
	}

	for _, tt := range tests {
		got := trailingZeros32(tt.input)
		if got != tt.want {
			t.Errorf("trailingZeros32(%032b) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

// ============================================================================
// Benchmark tests
// ============================================================================

func BenchmarkBaseFindVarintEnds(b *testing.B) {
	// Create a buffer with mixed varints
	data := encodeMultipleUvarints(1, 128, 300, 1, 16384, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	// Pad to 32 bytes
	for len(data) < 32 {
		data = append(data, 0x01)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = BaseFindVarintEnds(data)
	}
}

func BenchmarkBaseDecodeUvarint64Batch(b *testing.B) {
	// Pre-encode N varints with mixed sizes
	n := 100
	values := make([]uint64, n)
	for i := 0; i < n; i++ {
		values[i] = uint64(i * 100) // mix of single and multi-byte
	}
	encoded := encodeMultipleUvarints(values...)

	dst := make([]uint64, n)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		BaseDecodeUvarint64Batch(encoded, dst, n)
	}
}

func BenchmarkBaseDecode2Uvarint64(b *testing.B) {
	// Typical freq/norm pair
	encoded := encodeMultipleUvarints(1000, 128)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, _ = BaseDecode2Uvarint64(encoded)
	}
}

func BenchmarkBaseDecode5Uvarint64(b *testing.B) {
	// Typical location fields
	encoded := encodeMultipleUvarints(10, 1000, 5000, 5050, 3)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = BaseDecode5Uvarint64(encoded)
	}
}

func BenchmarkBaseDecodeGroupVarint32(b *testing.B) {
	values := [4]uint32{300, 5, 1000, 2}
	dst := make([]byte, 17)
	n := BaseEncodeGroupVarint32(values, dst)
	encoded := dst[:n]

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = BaseDecodeGroupVarint32(encoded)
	}
}

func BenchmarkBaseDecodeGroupVarint64(b *testing.B) {
	values := [4]uint64{300, 5, 1000, 2}
	dst := make([]byte, 34)
	n := BaseEncodeGroupVarint64(values, dst)
	encoded := dst[:n]

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = BaseDecodeGroupVarint64(encoded)
	}
}

func BenchmarkBaseEncodeGroupVarint32(b *testing.B) {
	values := [4]uint32{300, 5, 1000, 2}
	dst := make([]byte, 17)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = BaseEncodeGroupVarint32(values, dst)
	}
}

func BenchmarkBaseEncodeGroupVarint64(b *testing.B) {
	values := [4]uint64{300, 5, 1000, 2}
	dst := make([]byte, 34)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = BaseEncodeGroupVarint64(values, dst)
	}
}

// Benchmark comparing standard varint batch decode vs group varint
func BenchmarkCompareVarintVsGroupVarint(b *testing.B) {
	// Encode same 4 values both ways
	values32 := [4]uint32{300, 5, 1000, 2}
	values64 := []uint64{300, 5, 1000, 2}

	// Standard varint encoding
	standardEncoded := encodeMultipleUvarints(values64...)
	dst64 := make([]uint64, 4)

	// Group varint encoding
	groupDst := make([]byte, 17)
	groupN := BaseEncodeGroupVarint32(values32, groupDst)
	groupEncoded := groupDst[:groupN]

	b.Run("StandardVarint_4values", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = BaseDecodeUvarint64Batch(standardEncoded, dst64, 4)
		}
	})

	b.Run("GroupVarint32_4values", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = BaseDecodeGroupVarint32(groupEncoded)
		}
	})
}

// ============================================================================
// Stream-VByte Tests
// ============================================================================

func TestBaseEncodeStreamVByte32_RoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		values []uint32
	}{
		{"small_values", []uint32{1, 2, 3, 4}},
		{"mixed_sizes", []uint32{300, 5, 1000, 2}},
		{"all_zeros", []uint32{0, 0, 0, 0}},
		{"max_values", []uint32{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}},
		{"8_values", []uint32{1, 128, 300, 65535, 100000, 1, 2, 3}},
		{"not_multiple_of_4", []uint32{1, 2, 3, 4, 5}}, // Should pad
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			control, data := BaseEncodeStreamVByte32(tt.values)

			// Decode
			n := len(tt.values)
			if n%4 != 0 {
				n = ((n + 3) / 4) * 4 // Round up to multiple of 4
			}
			decoded := BaseDecodeStreamVByte32(control, data, n)

			// Verify (account for padding)
			for i, want := range tt.values {
				if decoded[i] != want {
					t.Errorf("value %d: got %d, want %d", i, decoded[i], want)
				}
			}
		})
	}
}

func TestBaseDecodeStreamVByte32Into(t *testing.T) {
	values := []uint32{300, 5, 1000, 2, 7, 128, 50, 1}
	control, data := BaseEncodeStreamVByte32(values)

	dst := make([]uint32, 8)
	decoded, dataConsumed := BaseDecodeStreamVByte32Into(control, data, dst)

	if decoded != 8 {
		t.Errorf("decoded: got %d, want 8", decoded)
	}
	if dataConsumed != len(data) {
		t.Errorf("dataConsumed: got %d, want %d", dataConsumed, len(data))
	}

	for i, want := range values {
		if dst[i] != want {
			t.Errorf("value %d: got %d, want %d", i, dst[i], want)
		}
	}
}

func TestBaseStreamVByte32DataLen(t *testing.T) {
	values := []uint32{300, 5, 1000, 2}
	control, data := BaseEncodeStreamVByte32(values)

	calcLen := BaseStreamVByte32DataLen(control)
	if calcLen != len(data) {
		t.Errorf("calculated length: got %d, want %d", calcLen, len(data))
	}
}

func TestStreamVByteEncoder(t *testing.T) {
	enc := NewStreamVByteEncoder()
	enc.Add(100)
	enc.Add(200)
	enc.Add(300)
	enc.Add(400)
	enc.Add(500)

	control, data := enc.Finish()

	// Should have 2 control bytes (8 values with padding)
	if len(control) != 2 {
		t.Errorf("control length: got %d, want 2", len(control))
	}

	// Decode and verify
	decoded := BaseDecodeStreamVByte32(control, data, 5)
	expected := []uint32{100, 200, 300, 400, 500}
	for i, want := range expected {
		if decoded[i] != want {
			t.Errorf("value %d: got %d, want %d", i, decoded[i], want)
		}
	}
}

// ============================================================================
// Stream-VByte Benchmarks
// ============================================================================

func BenchmarkBaseEncodeStreamVByte32(b *testing.B) {
	values := make([]uint32, 100)
	for i := range values {
		values[i] = uint32(i * 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		control, data := BaseEncodeStreamVByte32(values)
		_ = control
		_ = data
	}
}

func BenchmarkBaseDecodeStreamVByte32Into(b *testing.B) {
	values := make([]uint32, 100)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	control, data := BaseEncodeStreamVByte32(values)
	dst := make([]uint32, 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BaseDecodeStreamVByte32Into(control, data, dst)
	}
}

func BenchmarkDecodeStreamVByte32Into_Dispatch(b *testing.B) {
	// This benchmark uses the dispatch function which may use SIMD on ARM64
	values := make([]uint32, 100)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	control, data := EncodeStreamVByte32(values)
	dst := make([]uint32, 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DecodeStreamVByte32Into(control, data, dst)
	}
}

func BenchmarkCompareGroupVarintVsStreamVByte(b *testing.B) {
	// Compare Group Varint vs Stream-VByte for 100 values
	values := make([]uint32, 100)
	for i := range values {
		values[i] = uint32(i * 100)
	}

	// Prepare Group Varint encoded data
	groupData := make([]byte, 0, 500)
	for i := 0; i < len(values); i += 4 {
		dst := make([]byte, 17)
		n := BaseEncodeGroupVarint32([4]uint32{values[i], values[i+1], values[i+2], values[i+3]}, dst)
		groupData = append(groupData, dst[:n]...)
	}

	// Prepare Stream-VByte encoded data
	control, streamData := BaseEncodeStreamVByte32(values)

	b.Run("GroupVarint_100values", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			offset := 0
			for j := 0; j < 25; j++ {
				_, consumed := BaseDecodeGroupVarint32(groupData[offset:])
				offset += consumed
			}
		}
	})

	b.Run("StreamVByte_100values", func(b *testing.B) {
		dst := make([]uint32, 100)
		for i := 0; i < b.N; i++ {
			BaseDecodeStreamVByte32Into(control, streamData, dst)
		}
	})

	b.Run("StreamVByte_Dispatch_100values", func(b *testing.B) {
		dst := make([]uint32, 100)
		for i := 0; i < b.N; i++ {
			DecodeStreamVByte32Into(control, streamData, dst)
		}
	})
}
