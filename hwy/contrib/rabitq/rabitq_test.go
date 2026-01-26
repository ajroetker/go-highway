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

package rabitq

import (
	"math"
	"math/bits"
	"math/rand/v2"
	"testing"
)

// bitProductReference is a pure Go reference implementation for testing
func bitProductReference(code, q1, q2, q3, q4 []uint64) uint32 {
	var sum uint32
	for i := range code {
		sum += 1 * uint32(bits.OnesCount64(code[i]&q1[i]))
		sum += 2 * uint32(bits.OnesCount64(code[i]&q2[i]))
		sum += 4 * uint32(bits.OnesCount64(code[i]&q3[i]))
		sum += 8 * uint32(bits.OnesCount64(code[i]&q4[i]))
	}
	return sum
}

func TestBitProduct_Basic(t *testing.T) {
	tests := []struct {
		name string
		code []uint64
		q1   []uint64
		q2   []uint64
		q3   []uint64
		q4   []uint64
		want uint32
	}{
		{
			name: "all zeros",
			code: []uint64{0, 0},
			q1:   []uint64{0, 0},
			q2:   []uint64{0, 0},
			q3:   []uint64{0, 0},
			q4:   []uint64{0, 0},
			want: 0,
		},
		{
			name: "all ones code, all ones queries",
			code: []uint64{^uint64(0)},
			q1:   []uint64{^uint64(0)},
			q2:   []uint64{^uint64(0)},
			q3:   []uint64{^uint64(0)},
			q4:   []uint64{^uint64(0)},
			want: 64 * (1 + 2 + 4 + 8), // 64 bits * 15 = 960
		},
		{
			name: "single bit in q1",
			code: []uint64{1},
			q1:   []uint64{1},
			q2:   []uint64{0},
			q3:   []uint64{0},
			q4:   []uint64{0},
			want: 1,
		},
		{
			name: "single bit in q4",
			code: []uint64{1},
			q1:   []uint64{0},
			q2:   []uint64{0},
			q3:   []uint64{0},
			q4:   []uint64{1},
			want: 8,
		},
		{
			name: "mixed weights",
			code: []uint64{0xF}, // 4 bits set (bits 0,1,2,3)
			q1:   []uint64{0x3}, // 2 overlap (bits 0,1)
			q2:   []uint64{0x5}, // 2 overlap (bits 0,2)
			q3:   []uint64{0x9}, // 2 overlap (bits 0,3)
			q4:   []uint64{0x1}, // 1 overlap (bit 0)
			want: 1*2 + 2*2 + 4*2 + 8*1, // 2 + 4 + 8 + 8 = 22
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BitProduct(tt.code, tt.q1, tt.q2, tt.q3, tt.q4)
			if got != tt.want {
				t.Errorf("BitProduct() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestBitProduct_RandomData(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 1))

	for _, size := range []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512} {
		t.Run(sizeToName(size), func(t *testing.T) {
			code := make([]uint64, size)
			q1 := make([]uint64, size)
			q2 := make([]uint64, size)
			q3 := make([]uint64, size)
			q4 := make([]uint64, size)

			for i := range size {
				code[i] = rng.Uint64()
				q1[i] = rng.Uint64()
				q2[i] = rng.Uint64()
				q3[i] = rng.Uint64()
				q4[i] = rng.Uint64()
			}

			got := BitProduct(code, q1, q2, q3, q4)
			want := bitProductReference(code, q1, q2, q3, q4)

			if got != want {
				t.Errorf("BitProduct() = %d, want %d (size=%d)", got, want, size)
			}
		})
	}
}

func TestBitProduct_Empty(t *testing.T) {
	got := BitProduct(nil, nil, nil, nil, nil)
	if got != 0 {
		t.Errorf("BitProduct(nil) = %d, want 0", got)
	}

	got = BitProduct([]uint64{}, []uint64{}, []uint64{}, []uint64{}, []uint64{})
	if got != 0 {
		t.Errorf("BitProduct(empty) = %d, want 0", got)
	}
}

func TestQuantizeVectors_Basic(t *testing.T) {
	dims := 64
	width := CodeWidth(dims)
	sqrtDimsInv := float32(1.0 / math.Sqrt(float64(dims)))

	// Test with a simple vector: all positive
	unitVectors := make([]float32, dims)
	for i := range dims {
		unitVectors[i] = 1.0 / float32(math.Sqrt(float64(dims))) // unit vector
	}

	codes := make([]uint64, width)
	dotProducts := make([]float32, 1)
	codeCounts := make([]uint32, 1)

	QuantizeVectors(unitVectors, codes, dotProducts, codeCounts, sqrtDimsInv, 1, dims, width)

	// All positive values should give all 1s in code
	if codes[0] != ^uint64(0) {
		t.Errorf("Expected all 1s for positive vector, got %064b", codes[0])
	}

	// Code count should be 64 (all bits set)
	if codeCounts[0] != 64 {
		t.Errorf("Expected code count 64, got %d", codeCounts[0])
	}

	// Dot product should be positive
	if dotProducts[0] <= 0 {
		t.Errorf("Expected positive dot product, got %f", dotProducts[0])
	}
}

func TestQuantizeVectors_AllNegative(t *testing.T) {
	dims := 64
	width := CodeWidth(dims)
	sqrtDimsInv := float32(1.0 / math.Sqrt(float64(dims)))

	// Test with all negative values
	unitVectors := make([]float32, dims)
	for i := range dims {
		unitVectors[i] = -1.0 / float32(math.Sqrt(float64(dims)))
	}

	codes := make([]uint64, width)
	dotProducts := make([]float32, 1)
	codeCounts := make([]uint32, 1)

	QuantizeVectors(unitVectors, codes, dotProducts, codeCounts, sqrtDimsInv, 1, dims, width)

	// All negative values should give all 0s in code
	if codes[0] != 0 {
		t.Errorf("Expected all 0s for negative vector, got %064b", codes[0])
	}

	// Code count should be 0
	if codeCounts[0] != 0 {
		t.Errorf("Expected code count 0, got %d", codeCounts[0])
	}
}

func TestQuantizeVectors_Mixed(t *testing.T) {
	dims := 128
	width := CodeWidth(dims)
	sqrtDimsInv := float32(1.0 / math.Sqrt(float64(dims)))

	// Alternate positive and negative
	unitVectors := make([]float32, dims)
	norm := float32(0)
	for i := range dims {
		if i%2 == 0 {
			unitVectors[i] = 1.0
		} else {
			unitVectors[i] = -1.0
		}
		norm += unitVectors[i] * unitVectors[i]
	}
	// Normalize
	norm = float32(math.Sqrt(float64(norm)))
	for i := range dims {
		unitVectors[i] /= norm
	}

	codes := make([]uint64, width)
	dotProducts := make([]float32, 1)
	codeCounts := make([]uint32, 1)

	QuantizeVectors(unitVectors, codes, dotProducts, codeCounts, sqrtDimsInv, 1, dims, width)

	// Half should be set
	if codeCounts[0] != 64 {
		t.Errorf("Expected code count 64 (half of %d), got %d", dims, codeCounts[0])
	}
}

func TestQuantizeVectors_MultipleVectors(t *testing.T) {
	dims := 64
	width := CodeWidth(dims)
	count := 10
	sqrtDimsInv := float32(1.0 / math.Sqrt(float64(dims)))

	rng := rand.New(rand.NewPCG(42, 1))

	// Generate random unit vectors
	unitVectors := make([]float32, count*dims)
	for i := range count {
		var norm float64
		for j := range dims {
			unitVectors[i*dims+j] = rng.Float32()*2 - 1
			norm += float64(unitVectors[i*dims+j] * unitVectors[i*dims+j])
		}
		norm = math.Sqrt(norm)
		for j := range dims {
			unitVectors[i*dims+j] /= float32(norm)
		}
	}

	codes := make([]uint64, count*width)
	dotProducts := make([]float32, count)
	codeCounts := make([]uint32, count)

	QuantizeVectors(unitVectors, codes, dotProducts, codeCounts, sqrtDimsInv, count, dims, width)

	// Verify each vector was processed
	for i := range count {
		// Code count should be between 0 and dims
		if codeCounts[i] > uint32(dims) {
			t.Errorf("Vector %d: code count %d exceeds dims %d", i, codeCounts[i], dims)
		}

		// Dot product should be non-zero for random vectors
		if dotProducts[i] == 0 {
			t.Logf("Vector %d: dot product is zero (possible but unlikely for random data)", i)
		}
	}
}

func TestQuantizeVectors_NonAlignedDims(t *testing.T) {
	// Test with dimensions that don't align to 64
	for _, dims := range []int{17, 33, 65, 100, 127, 200} {
		t.Run(sizeToName(dims), func(t *testing.T) {
			width := CodeWidth(dims)
			sqrtDimsInv := float32(1.0 / math.Sqrt(float64(dims)))

			unitVectors := make([]float32, dims)
			for i := range dims {
				unitVectors[i] = 1.0 / float32(math.Sqrt(float64(dims)))
			}

			codes := make([]uint64, width)
			dotProducts := make([]float32, 1)
			codeCounts := make([]uint32, 1)

			QuantizeVectors(unitVectors, codes, dotProducts, codeCounts, sqrtDimsInv, 1, dims, width)

			// For all positive, code count should equal dims
			if codeCounts[0] != uint32(dims) {
				t.Errorf("dims=%d: expected code count %d, got %d", dims, dims, codeCounts[0])
			}
		})
	}
}

func TestMultiplySigns(t *testing.T) {
	tests := []struct {
		x, y, want float32
	}{
		{1.0, 1.0, 1.0},
		{1.0, -1.0, -1.0},
		{-1.0, 1.0, -1.0},
		{-1.0, -1.0, 1.0},
		{5.0, 3.0, 5.0},
		{5.0, -3.0, -5.0},
		{-5.0, 3.0, -5.0},
		{-5.0, -3.0, 5.0},
		{0.0, 1.0, 0.0},
		{0.0, -1.0, 0.0}, // -0.0 is valid
	}

	for _, tt := range tests {
		got := MultiplySigns(tt.x, tt.y)
		// Check magnitude
		if math.Abs(float64(got)) != math.Abs(float64(tt.want)) {
			t.Errorf("MultiplySigns(%f, %f) magnitude = %f, want %f", tt.x, tt.y, math.Abs(float64(got)), math.Abs(float64(tt.want)))
		}
		// Check sign (handle zero specially)
		if tt.want != 0 {
			gotSign := math.Signbit(float64(got))
			wantSign := math.Signbit(float64(tt.want))
			if gotSign != wantSign {
				t.Errorf("MultiplySigns(%f, %f) sign mismatch: got %f, want %f", tt.x, tt.y, got, tt.want)
			}
		}
	}
}

func TestCodeWidth(t *testing.T) {
	tests := []struct {
		dims int
		want int
	}{
		{1, 1},
		{64, 1},
		{65, 2},
		{128, 2},
		{129, 3},
		{256, 4},
		{384, 6},
		{512, 8},
		{768, 12},
		{1024, 16},
	}

	for _, tt := range tests {
		got := CodeWidth(tt.dims)
		if got != tt.want {
			t.Errorf("CodeWidth(%d) = %d, want %d", tt.dims, got, tt.want)
		}
	}
}

func sizeToName(size int) string {
	return "size_" + itoa(size)
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	s := ""
	for i > 0 {
		s = string(rune('0'+i%10)) + s
		i /= 10
	}
	return s
}

// Benchmarks

func BenchmarkBitProduct(b *testing.B) {
	for _, size := range []int{8, 16, 32, 64, 128, 256, 512} {
		b.Run(sizeToName(size), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(42, 1))

			code := make([]uint64, size)
			q1 := make([]uint64, size)
			q2 := make([]uint64, size)
			q3 := make([]uint64, size)
			q4 := make([]uint64, size)

			for i := range size {
				code[i] = rng.Uint64()
				q1[i] = rng.Uint64()
				q2[i] = rng.Uint64()
				q3[i] = rng.Uint64()
				q4[i] = rng.Uint64()
			}

			b.ResetTimer()
			b.ReportAllocs()

			var result uint32
			for i := 0; i < b.N; i++ {
				result = BitProduct(code, q1, q2, q3, q4)
			}
			_ = result
		})
	}
}

func BenchmarkQuantizeVectors(b *testing.B) {
	for _, dims := range []int{128, 256, 384, 512, 768, 1024} {
		b.Run(sizeToName(dims), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(42, 1))

			count := 100
			width := CodeWidth(dims)
			sqrtDimsInv := float32(1.0 / math.Sqrt(float64(dims)))

			unitVectors := make([]float32, count*dims)
			for i := range count {
				var norm float64
				for j := range dims {
					unitVectors[i*dims+j] = rng.Float32()*2 - 1
					norm += float64(unitVectors[i*dims+j] * unitVectors[i*dims+j])
				}
				norm = math.Sqrt(norm)
				for j := range dims {
					unitVectors[i*dims+j] /= float32(norm)
				}
			}

			codes := make([]uint64, count*width)
			dotProducts := make([]float32, count)
			codeCounts := make([]uint32, count)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				QuantizeVectors(unitVectors, codes, dotProducts, codeCounts, sqrtDimsInv, count, dims, width)
			}
		})
	}
}
