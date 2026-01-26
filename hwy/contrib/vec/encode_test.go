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

package vec

import (
	"bytes"
	"fmt"
	"math"
	"testing"
)

func TestEncodeDecodeFloat32s(t *testing.T) {
	testCases := []struct {
		name   string
		values []float32
	}{
		{"empty", nil},
		{"single", []float32{1.5}},
		{"small", []float32{1.0, 2.0, 3.0}},
		{"aligned_4", makeTestFloat32s(4)},   // 16 bytes - NEON aligned
		{"aligned_8", makeTestFloat32s(8)},   // 32 bytes - AVX2 aligned
		{"aligned_16", makeTestFloat32s(16)}, // 64 bytes - AVX512 aligned
		{"unaligned_5", makeTestFloat32s(5)},
		{"unaligned_13", makeTestFloat32s(13)},
		{"large_1024", makeTestFloat32s(1024)},
		{"large_4096", makeTestFloat32s(4096)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testEncodeDecode32(t, tc.values)
		})
	}
}

func TestEncodeDecodeFloat32sSpecialValues(t *testing.T) {
	// Test special float values
	special := []float32{
		0,
		-0,
		1,
		-1,
		math.MaxFloat32,
		-math.MaxFloat32,
		math.SmallestNonzeroFloat32,
		-math.SmallestNonzeroFloat32,
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		float32(math.NaN()),
	}

	encoded := make([]byte, len(special)*4)
	EncodeFloat32s(encoded, special)

	decoded := make([]float32, len(special))
	DecodeFloat32s(decoded, encoded)

	for i := range special {
		orig := special[i]
		dec := decoded[i]

		// NaN requires special comparison
		if math.IsNaN(float64(orig)) {
			if !math.IsNaN(float64(dec)) {
				t.Errorf("index %d: expected NaN, got %v", i, dec)
			}
			continue
		}

		// Compare bit patterns for exact equality (handles -0 vs 0)
		origBits := math.Float32bits(orig)
		decBits := math.Float32bits(dec)
		if origBits != decBits {
			t.Errorf("index %d: bit pattern mismatch: got %08x, want %08x (values: %v vs %v)",
				i, decBits, origBits, dec, orig)
		}
	}
}

func TestEncodeDecodeFloat64s(t *testing.T) {
	sizes := []int{0, 1, 2, 4, 8, 13, 64, 1024}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			values := makeTestFloat64s(size)
			testEncodeDecode64(t, values)
		})
	}
}

func TestEncodeDecodeFloat64sSpecialValues(t *testing.T) {
	special := []float64{
		0,
		-0,
		1,
		-1,
		math.MaxFloat64,
		-math.MaxFloat64,
		math.SmallestNonzeroFloat64,
		-math.SmallestNonzeroFloat64,
		math.Inf(1),
		math.Inf(-1),
		math.NaN(),
	}

	encoded := make([]byte, len(special)*8)
	EncodeFloat64s(encoded, special)

	decoded := make([]float64, len(special))
	DecodeFloat64s(decoded, encoded)

	for i := range special {
		orig := special[i]
		dec := decoded[i]

		if math.IsNaN(orig) {
			if !math.IsNaN(dec) {
				t.Errorf("index %d: expected NaN, got %v", i, dec)
			}
			continue
		}

		origBits := math.Float64bits(orig)
		decBits := math.Float64bits(dec)
		if origBits != decBits {
			t.Errorf("index %d: bit pattern mismatch: got %016x, want %016x",
				i, decBits, origBits)
		}
	}
}

func TestEncodeFloat32sPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for short dst")
		}
	}()

	src := make([]float32, 10)
	dst := make([]byte, 10) // too short, need 40
	EncodeFloat32s(dst, src)
}

func TestDecodeFloat32sPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for short src")
		}
	}()

	src := make([]byte, 10) // too short
	dst := make([]float32, 10)
	DecodeFloat32s(dst, src)
}

// testEncodeDecode32 tests round-trip encode/decode for float32
func testEncodeDecode32(t *testing.T, values []float32) {
	t.Helper()

	if len(values) == 0 {
		// Test empty case - just ensure no panic
		encoded := make([]byte, 0)
		decoded := make([]float32, 0)
		EncodeFloat32s(encoded, values)
		DecodeFloat32s(decoded, encoded)
		return
	}

	// Encode
	encoded := make([]byte, len(values)*4)
	EncodeFloat32s(encoded, values)

	// Verify encoding matches scalar implementation
	expectedEncoded := make([]byte, len(values)*4)
	encodeFloat32sScalar(expectedEncoded, values)
	if !bytes.Equal(encoded, expectedEncoded) {
		t.Errorf("encoded bytes don't match scalar implementation")
	}

	// Decode
	decoded := make([]float32, len(values))
	DecodeFloat32s(decoded, encoded)

	// Verify
	for i := range values {
		if decoded[i] != values[i] {
			// Handle NaN specially
			if math.IsNaN(float64(values[i])) && math.IsNaN(float64(decoded[i])) {
				continue
			}
			t.Errorf("index %d: got %v, want %v", i, decoded[i], values[i])
		}
	}
}

// testEncodeDecode64 tests round-trip encode/decode for float64
func testEncodeDecode64(t *testing.T, values []float64) {
	t.Helper()

	if len(values) == 0 {
		encoded := make([]byte, 0)
		decoded := make([]float64, 0)
		EncodeFloat64s(encoded, values)
		DecodeFloat64s(decoded, encoded)
		return
	}

	// Encode
	encoded := make([]byte, len(values)*8)
	EncodeFloat64s(encoded, values)

	// Decode
	decoded := make([]float64, len(values))
	DecodeFloat64s(decoded, encoded)

	// Verify
	for i := range values {
		if decoded[i] != values[i] {
			if math.IsNaN(values[i]) && math.IsNaN(decoded[i]) {
				continue
			}
			t.Errorf("index %d: got %v, want %v", i, decoded[i], values[i])
		}
	}
}

// makeTestFloat32s creates a slice of float32 values for testing
func makeTestFloat32s(n int) []float32 {
	result := make([]float32, n)
	for i := range result {
		result[i] = float32(i) * 1.5
	}
	return result
}

// makeTestFloat64s creates a slice of float64 values for testing
func makeTestFloat64s(n int) []float64 {
	result := make([]float64, n)
	for i := range result {
		result[i] = float64(i) * 1.5
	}
	return result
}

// Benchmarks

func BenchmarkEncodeFloat32s(b *testing.B) {
	sizes := []int{16, 64, 256, 1024, 4096}

	for _, size := range sizes {
		src := makeTestFloat32s(size)
		dst := make([]byte, size*4)

		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			b.SetBytes(int64(size * 4))
			for i := 0; i < b.N; i++ {
				EncodeFloat32s(dst, src)
			}
		})
	}
}

func BenchmarkDecodeFloat32s(b *testing.B) {
	sizes := []int{16, 64, 256, 1024, 4096}

	for _, size := range sizes {
		src := make([]byte, size*4)
		dst := make([]float32, size)

		// Fill with valid float data
		tmp := makeTestFloat32s(size)
		encodeFloat32sScalar(src, tmp)

		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			b.SetBytes(int64(size * 4))
			for i := 0; i < b.N; i++ {
				DecodeFloat32s(dst, src)
			}
		})
	}
}

func BenchmarkEncodeFloat64s(b *testing.B) {
	sizes := []int{16, 64, 256, 1024, 4096}

	for _, size := range sizes {
		src := makeTestFloat64s(size)
		dst := make([]byte, size*8)

		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			b.SetBytes(int64(size * 8))
			for i := 0; i < b.N; i++ {
				EncodeFloat64s(dst, src)
			}
		})
	}
}

func BenchmarkDecodeFloat64s(b *testing.B) {
	sizes := []int{16, 64, 256, 1024, 4096}

	for _, size := range sizes {
		src := make([]byte, size*8)
		dst := make([]float64, size)

		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			b.SetBytes(int64(size * 8))
			for i := 0; i < b.N; i++ {
				DecodeFloat64s(dst, src)
			}
		})
	}
}

// Benchmark comparing SIMD vs scalar (when scalar is available)
func BenchmarkEncodeFloat32sScalar(b *testing.B) {
	sizes := []int{16, 64, 256, 1024, 4096}

	for _, size := range sizes {
		src := makeTestFloat32s(size)
		dst := make([]byte, size*4)

		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			b.SetBytes(int64(size * 4))
			for i := 0; i < b.N; i++ {
				encodeFloat32sScalar(dst, src)
			}
		})
	}
}
