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
)

func TestQuantizeQ8_0_RoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name: "ascending",
			input: func() []float32 {
				v := make([]float32, 32)
				for i := range v {
					v[i] = float32(i-16) * 0.1
				}
				return v
			}(),
		},
		{
			name: "uniform positive",
			input: func() []float32 {
				v := make([]float32, 32)
				for i := range v {
					v[i] = 1.0
				}
				return v
			}(),
		},
		{
			name: "zeros",
			input: make([]float32, 32),
		},
		{
			name: "large values",
			input: func() []float32 {
				v := make([]float32, 32)
				for i := range v {
					v[i] = float32(i) * 100.0
				}
				return v
			}(),
		},
		{
			name: "two blocks",
			input: func() []float32 {
				v := make([]float32, 64)
				for i := range v {
					v[i] = float32(i) - 32.0
				}
				return v
			}(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nblocks := len(tt.input) / QK
			qdata := make([]uint8, nblocks*BlockSizeQ8_0)

			// Quantize.
			QuantizeQ8_0(tt.input, qdata)

			// Dequantize using the existing tested dequantizer.
			roundtrip := make([]float32, len(tt.input))
			DequantizeQ8_0(qdata, roundtrip)

			// Check round-trip error.
			// Q8_0 has 127 levels, so max relative error is ~1/127 ≈ 0.8%.
			// Use absolute tolerance scaled by the block's max value.
			for b := range nblocks {
				// Find max absolute value in this block.
				var amax float32
				for i := 0; i < QK; i++ {
					av := tt.input[b*QK+i]
					if av < 0 {
						av = -av
					}
					if av > amax {
						amax = av
					}
				}
				tol := float64(amax / 127.0 * 1.01) // quantization step + small margin
				if tol < 1e-6 {
					tol = 1e-6
				}

				for i := 0; i < QK; i++ {
					idx := b*QK + i
					diff := math.Abs(float64(roundtrip[idx] - tt.input[idx]))
					if diff > tol {
						t.Errorf("block %d, index %d: input %f, roundtrip %f, diff %f > tol %f",
							b, i, tt.input[idx], roundtrip[idx], diff, tol)
					}
				}
			}
		})
	}
}

func TestQuantizeQ8_0_Empty(t *testing.T) {
	QuantizeQ8_0(nil, nil)
	QuantizeQ8_0([]float32{}, []uint8{})
}

func TestQuantizeQ8_0_DispatchVsFallback(t *testing.T) {
	input := make([]float32, 128) // 4 blocks
	for i := range input {
		input[i] = float32(i%256-128) * 0.5
	}

	nblocks := len(input) / QK
	dispatched := make([]uint8, nblocks*BlockSizeQ8_0)
	fallback := make([]uint8, nblocks*BlockSizeQ8_0)

	QuantizeQ8_0(input, dispatched)
	BaseQuantizeQ8_0_fallback(input, fallback)

	for i := range dispatched {
		if dispatched[i] != fallback[i] {
			t.Errorf("byte %d: dispatched %d, fallback %d", i, dispatched[i], fallback[i])
		}
	}
}

func BenchmarkQuantizeQ8_0(b *testing.B) {
	sizes := []int{1, 4, 16, 64, 256}
	for _, nblocks := range sizes {
		input := make([]float32, nblocks*QK)
		for i := range input {
			input[i] = float32(i%256-128) * 0.01
		}
		output := make([]uint8, nblocks*BlockSizeQ8_0)

		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			b.SetBytes(int64(len(input) * 4))
			for range b.N {
				QuantizeQ8_0(input, output)
			}
		})
	}
}
