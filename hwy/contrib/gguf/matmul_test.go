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
	"runtime"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// referenceMatMul computes C = A @ B^T in float32 (A is [M,K], B is [N,K], C is [M,N]).
func referenceMatMul(a []float32, b []float32, c []float32, M, K, N int) {
	for m := range M {
		for n := range N {
			var sum float64
			for k := range K {
				sum += float64(a[m*K+k]) * float64(b[n*K+k])
			}
			c[m*N+n] = float32(sum)
		}
	}
}

func TestGGUFMatMulQ4_0(t *testing.T) {
	tests := []struct {
		name string
		M, K, N int
	}{
		{"1x32x1", 1, 32, 1},
		{"1x64x2", 1, 64, 2},
		{"2x128x4", 2, 128, 4},
		{"4x256x8", 4, 256, 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nblocks := tt.K / QK

			// Create Q4_0 weight blocks [N, K].
			wdata := make([]uint8, tt.N*nblocks*BlockSizeQ4_0)
			for i := range wdata {
				wdata[i] = uint8(i % 256)
			}
			// Set valid fp16 scale=1.0 at each block boundary.
			for n := range tt.N {
				for b := range nblocks {
					off := n*nblocks*BlockSizeQ4_0 + b*BlockSizeQ4_0
					wdata[off] = fp16One[0]
					wdata[off+1] = fp16One[1]
				}
			}

			// Dequantize weights for reference [N, K].
			wFloat := make([]float32, tt.N*tt.K)
			for n := range tt.N {
				wRow := wdata[n*nblocks*BlockSizeQ4_0 : (n+1)*nblocks*BlockSizeQ4_0]
				DequantizeQ4_0(wRow, wFloat[n*tt.K:(n+1)*tt.K])
			}

			// Create float32 input [M, K].
			input := make([]float32, tt.M*tt.K)
			for i := range input {
				input[i] = float32(i%64-32) * 0.01
			}

			// Reference: dequantize weights, then dense matmul.
			want := make([]float32, tt.M*tt.N)
			referenceMatMul(input, wFloat, want, tt.M, tt.K, tt.N)

			// GGUF matmul on quantized data.
			got := make([]float32, tt.M*tt.N)
			GGUFMatMul(input, wdata, got, tt.M, tt.K, tt.N, TypeQ4_0)

			// Compare with tolerance for double quantization error
			// (input quantized to Q8_0, weights are Q4_0).
			for i := range got {
				absDiff := math.Abs(float64(got[i] - want[i]))
				relErr := float64(0)
				if want[i] != 0 {
					relErr = absDiff / math.Abs(float64(want[i]))
				}
				if relErr > 0.05 && absDiff > 0.5 {
					t.Errorf("output[%d]: got %f, want %f (relErr=%.4f, absDiff=%.4f)",
						i, got[i], want[i], relErr, absDiff)
				}
			}
		})
	}
}

func TestGGUFMatMulQ8_0(t *testing.T) {
	M, K, N := 2, 64, 4
	nblocks := K / QK

	// Create float32 weights and quantize to Q8_0 [N, K].
	wFloat := make([]float32, N*K)
	for i := range wFloat {
		wFloat[i] = float32(i%32-16) * 0.05
	}
	wdata := make([]uint8, N*nblocks*BlockSizeQ8_0)
	for n := range N {
		QuantizeQ8_0(wFloat[n*K:(n+1)*K], wdata[n*nblocks*BlockSizeQ8_0:(n+1)*nblocks*BlockSizeQ8_0])
	}

	// Dequantize weights for reference.
	wDeq := make([]float32, N*K)
	for n := range N {
		DequantizeQ8_0(wdata[n*nblocks*BlockSizeQ8_0:(n+1)*nblocks*BlockSizeQ8_0], wDeq[n*K:(n+1)*K])
	}

	// Create float32 input [M, K].
	input := make([]float32, M*K)
	for i := range input {
		input[i] = float32(i%32-16) * 0.02
	}

	// Reference matmul.
	want := make([]float32, M*N)
	referenceMatMul(input, wDeq, want, M, K, N)

	// GGUF matmul.
	got := make([]float32, M*N)
	GGUFMatMul(input, wdata, got, M, K, N, TypeQ8_0)

	for i := range got {
		absDiff := math.Abs(float64(got[i] - want[i]))
		relErr := float64(0)
		if want[i] != 0 {
			relErr = absDiff / math.Abs(float64(want[i]))
		}
		if relErr > 0.05 && absDiff > 0.5 {
			t.Errorf("output[%d]: got %f, want %f (relErr=%.4f)", i, got[i], want[i], relErr)
		}
	}
}

func TestGGUFMatMul_Empty(t *testing.T) {
	GGUFMatMul(nil, nil, nil, 0, 0, 0, TypeQ4_0)
	GGUFMatMul(nil, nil, nil, 1, 0, 1, TypeQ4_0)
}

func TestParallelGGUFMatMulQ4_0(t *testing.T) {
	pool := workerpool.New(runtime.GOMAXPROCS(0))
	defer pool.Close()

	M, K, N := 4, 128, 8
	nblocks := K / QK

	// Create Q4_0 weight blocks.
	wdata := make([]uint8, N*nblocks*BlockSizeQ4_0)
	for i := range wdata {
		wdata[i] = uint8(i % 256)
	}
	for n := range N {
		for b := range nblocks {
			off := n*nblocks*BlockSizeQ4_0 + b*BlockSizeQ4_0
			wdata[off] = fp16One[0]
			wdata[off+1] = fp16One[1]
		}
	}

	input := make([]float32, M*K)
	for i := range input {
		input[i] = float32(i%64-32) * 0.01
	}

	// Compare serial vs parallel.
	serial := make([]float32, M*N)
	parallel := make([]float32, M*N)

	GGUFMatMul(input, wdata, serial, M, K, N, TypeQ4_0)
	ParallelGGUFMatMul(pool, input, wdata, parallel, M, K, N, TypeQ4_0)

	for i := range serial {
		if serial[i] != parallel[i] {
			t.Errorf("output[%d]: serial %f != parallel %f", i, serial[i], parallel[i])
		}
	}
}

func BenchmarkGGUFMatMulQ4_0(b *testing.B) {
	sizes := []struct {
		M, K, N int
		name    string
	}{
		{1, 4096, 4096, "1x4096x4096_autoregressive"},
		{32, 4096, 4096, "32x4096x4096_prompt"},
		{1, 4096, 11008, "1x4096x11008_mlp"},
	}

	pool := workerpool.New(runtime.GOMAXPROCS(0))
	defer pool.Close()

	for _, s := range sizes {
		nblocks := s.K / QK
		wdata := make([]uint8, s.N*nblocks*BlockSizeQ4_0)
		for i := range wdata {
			wdata[i] = uint8(i % 256)
		}
		for n := range s.N {
			for bl := range nblocks {
				off := n*nblocks*BlockSizeQ4_0 + bl*BlockSizeQ4_0
				wdata[off] = fp16One[0]
				wdata[off+1] = fp16One[1]
			}
		}
		input := make([]float32, s.M*s.K)
		for i := range input {
			input[i] = float32(i%256-128) * 0.001
		}
		output := make([]float32, s.M*s.N)

		b.Run(fmt.Sprintf("serial_%s", s.name), func(b *testing.B) {
			b.SetBytes(int64(s.M * s.N * 4))
			for range b.N {
				GGUFMatMul(input, wdata, output, s.M, s.K, s.N, TypeQ4_0)
			}
		})

		b.Run(fmt.Sprintf("parallel_%s", s.name), func(b *testing.B) {
			b.SetBytes(int64(s.M * s.N * 4))
			for range b.N {
				ParallelGGUFMatMul(pool, input, wdata, output, s.M, s.K, s.N, TypeQ4_0)
			}
		})
	}
}
