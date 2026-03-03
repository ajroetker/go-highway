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

// referenceDot computes the float32 dot product of two vectors.
func referenceDot(a, b []float32) float32 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return float32(sum)
}

func TestVecDotQ4_0Q8_0(t *testing.T) {
	tests := []struct {
		name    string
		nblocks int
	}{
		{"1 block", 1},
		{"4 blocks", 4},
		{"16 blocks", 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nvals := tt.nblocks * QK

			// Create Q4_0 weight blocks with known values.
			// Use split nibble layout: low nibbles 0..15 -> values 0..15,
			// high nibbles 0..15 -> values 16..31.
			wdata := make([]uint8, tt.nblocks*BlockSizeQ4_0)
			for b := range tt.nblocks {
				off := b * BlockSizeQ4_0
				// Scale = 1.0 (fp16)
				wdata[off] = fp16One[0]
				wdata[off+1] = fp16One[1]
				// Nibbles: set each byte to (hi<<4 | lo)
				for i := 0; i < 16; i++ {
					lo := uint8((i + b) % 16)     // 0..15 -> dequant: lo-8
					hi := uint8((15 - i + b) % 16) // reverse -> dequant: hi-8
					wdata[off+2+i] = (hi << 4) | lo
				}
			}

			// Dequantize weights to float32 for reference.
			wFloat := make([]float32, nvals)
			DequantizeQ4_0(wdata, wFloat)

			// Create float32 activations and quantize to Q8_0.
			aFloat := make([]float32, nvals)
			for i := range aFloat {
				aFloat[i] = float32(i%32-16) * 0.1
			}
			adata := make([]uint8, tt.nblocks*BlockSizeQ8_0)
			QuantizeQ8_0(aFloat, adata)

			// Dequantize activations for reference.
			aDeq := make([]float32, nvals)
			DequantizeQ8_0(adata, aDeq)

			// Reference dot product on dequantized values.
			want := referenceDot(wFloat, aDeq)

			// Compute via vec_dot.
			got := VecDotQ4_0Q8_0(wdata, adata, tt.nblocks)

			// Tolerance: quantization introduces error on both sides.
			// Allow 1% relative error or small absolute tolerance.
			relErr := float64(0)
			if want != 0 {
				relErr = math.Abs(float64(got-want)) / math.Abs(float64(want))
			}
			absDiff := math.Abs(float64(got - want))
			if relErr > 0.02 && absDiff > 0.1 {
				t.Errorf("got %f, want %f (relErr=%.4f, absDiff=%.4f)", got, want, relErr, absDiff)
			}
		})
	}
}

func TestVecDotQ8_0Q8_0(t *testing.T) {
	tests := []struct {
		name    string
		nblocks int
	}{
		{"1 block", 1},
		{"4 blocks", 4},
		{"16 blocks", 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nvals := tt.nblocks * QK

			// Create float32 weight and activation vectors.
			wFloat := make([]float32, nvals)
			aFloat := make([]float32, nvals)
			for i := range nvals {
				wFloat[i] = float32(i%32-16) * 0.05
				aFloat[i] = float32((i+7)%32-16) * 0.05
			}

			// Quantize both to Q8_0.
			wdata := make([]uint8, tt.nblocks*BlockSizeQ8_0)
			adata := make([]uint8, tt.nblocks*BlockSizeQ8_0)
			QuantizeQ8_0(wFloat, wdata)
			QuantizeQ8_0(aFloat, adata)

			// Dequantize for reference.
			wDeq := make([]float32, nvals)
			aDeq := make([]float32, nvals)
			DequantizeQ8_0(wdata, wDeq)
			DequantizeQ8_0(adata, aDeq)

			// Reference dot product.
			want := referenceDot(wDeq, aDeq)

			// Compute via vec_dot.
			got := VecDotQ8_0Q8_0(wdata, adata, tt.nblocks)

			relErr := float64(0)
			if want != 0 {
				relErr = math.Abs(float64(got-want)) / math.Abs(float64(want))
			}
			absDiff := math.Abs(float64(got - want))
			if relErr > 0.02 && absDiff > 0.1 {
				t.Errorf("got %f, want %f (relErr=%.4f, absDiff=%.4f)", got, want, relErr, absDiff)
			}
		})
	}
}

func TestVecDotQ4_0Q8_0_DispatchVsFallback(t *testing.T) {
	nblocks := 8
	nvals := nblocks * QK

	wdata := make([]uint8, nblocks*BlockSizeQ4_0)
	for i := range wdata {
		wdata[i] = uint8(i % 256)
	}
	// Set valid fp16 scales at block boundaries.
	for b := range nblocks {
		off := b * BlockSizeQ4_0
		wdata[off] = fp16One[0]
		wdata[off+1] = fp16One[1]
	}

	aFloat := make([]float32, nvals)
	for i := range aFloat {
		aFloat[i] = float32(i%32-16) * 0.1
	}
	adata := make([]uint8, nblocks*BlockSizeQ8_0)
	QuantizeQ8_0(aFloat, adata)

	got := VecDotQ4_0Q8_0(wdata, adata, nblocks)
	want := BaseVecDotQ4_0Q8_0_fallback(wdata, adata, nblocks)

	if got != want {
		t.Errorf("dispatch %f != fallback %f", got, want)
	}
}

func TestVecDotQ8_0Q8_0_DispatchVsFallback(t *testing.T) {
	nblocks := 8
	nvals := nblocks * QK

	wFloat := make([]float32, nvals)
	aFloat := make([]float32, nvals)
	for i := range nvals {
		wFloat[i] = float32(i%32-16) * 0.1
		aFloat[i] = float32((i+7)%32-16) * 0.1
	}
	wdata := make([]uint8, nblocks*BlockSizeQ8_0)
	adata := make([]uint8, nblocks*BlockSizeQ8_0)
	QuantizeQ8_0(wFloat, wdata)
	QuantizeQ8_0(aFloat, adata)

	got := VecDotQ8_0Q8_0(wdata, adata, nblocks)
	want := BaseVecDotQ8_0Q8_0_fallback(wdata, adata, nblocks)

	if got != want {
		t.Errorf("dispatch %f != fallback %f", got, want)
	}
}

func BenchmarkVecDotQ4_0Q8_0(b *testing.B) {
	sizes := []int{1, 4, 16, 64, 128}
	for _, nblocks := range sizes {
		nvals := nblocks * QK
		wdata := make([]uint8, nblocks*BlockSizeQ4_0)
		for i := range wdata {
			wdata[i] = uint8(i % 256)
		}
		for bl := range nblocks {
			off := bl * BlockSizeQ4_0
			wdata[off] = fp16One[0]
			wdata[off+1] = fp16One[1]
		}

		aFloat := make([]float32, nvals)
		for i := range aFloat {
			aFloat[i] = float32(i%32-16) * 0.01
		}
		adata := make([]uint8, nblocks*BlockSizeQ8_0)
		QuantizeQ8_0(aFloat, adata)

		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			b.SetBytes(int64(nblocks * (BlockSizeQ4_0 + BlockSizeQ8_0)))
			for range b.N {
				VecDotQ4_0Q8_0(wdata, adata, nblocks)
			}
		})
	}
}

func BenchmarkVecDotQ8_0Q8_0(b *testing.B) {
	sizes := []int{1, 4, 16, 64, 128}
	for _, nblocks := range sizes {
		nvals := nblocks * QK
		wFloat := make([]float32, nvals)
		aFloat := make([]float32, nvals)
		for i := range nvals {
			wFloat[i] = float32(i%32-16) * 0.01
			aFloat[i] = float32((i+7)%32-16) * 0.01
		}
		wdata := make([]uint8, nblocks*BlockSizeQ8_0)
		adata := make([]uint8, nblocks*BlockSizeQ8_0)
		QuantizeQ8_0(wFloat, wdata)
		QuantizeQ8_0(aFloat, adata)

		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			b.SetBytes(int64(nblocks * 2 * BlockSizeQ8_0))
			for range b.N {
				VecDotQ8_0Q8_0(wdata, adata, nblocks)
			}
		})
	}
}
