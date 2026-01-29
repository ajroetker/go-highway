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

//go:build arm64

package matmul

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// TestBlockMulAddNEONF32 tests the hand-written NEON assembly version.
func TestBlockMulAddNEONF32(t *testing.T) {
	blockSizes := []int{8, 16, 32, 48, 64}

	for _, blockDim := range blockSizes {
		t.Run(sizeStr(blockDim), func(t *testing.T) {
			size := blockDim * blockDim

			a := make([]float32, size)
			b := make([]float32, size)
			c := make([]float32, size)
			expected := make([]float32, size)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}
			for i := range c {
				c[i] = rand.Float32() * 0.1
				expected[i] = c[i]
			}

			aT := transposeBlock(a, blockDim)
			referenceBlockMulAdd(aT, b, expected, blockDim)
			asm.BlockMulAddNEONF32(aT, b, c, blockDim)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := float32(1e-4) * float32(blockDim)
			if maxErr > tolerance {
				t.Errorf("BlockMulAddNEONF32: max error %e exceeds tolerance %e", maxErr, tolerance)
			} else {
				t.Logf("blockDim=%d: max error %e", blockDim, maxErr)
			}
		})
	}
}

// TestBlockMulAddNEONF64 tests the float64 NEON assembly version.
func TestBlockMulAddNEONF64(t *testing.T) {
	blockSizes := []int{8, 16, 32, 48, 64}

	for _, blockDim := range blockSizes {
		t.Run(sizeStr(blockDim), func(t *testing.T) {
			size := blockDim * blockDim

			a := make([]float64, size)
			b := make([]float64, size)
			c := make([]float64, size)
			expected := make([]float64, size)

			for i := range a {
				a[i] = rand.Float64()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float64()*2 - 1
			}
			for i := range c {
				c[i] = rand.Float64() * 0.1
				expected[i] = c[i]
			}

			aT := transposeBlockFloat64(a, blockDim)
			referenceBlockMulAddFloat64(aT, b, expected, blockDim)
			asm.BlockMulAddNEONF64(aT, b, c, blockDim)

			var maxErr float64
			for i := range c {
				err := math.Abs(c[i] - expected[i])
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := 1e-10 * float64(blockDim)
			if maxErr > tolerance {
				t.Errorf("BlockMulAddNEONF64: max error %e exceeds tolerance %e", maxErr, tolerance)
			} else {
				t.Logf("blockDim=%d: max error %e", blockDim, maxErr)
			}
		})
	}
}

// transposeBlockFloat64 transposes a blockDim x blockDim matrix.
func transposeBlockFloat64(m []float64, blockDim int) []float64 {
	result := make([]float64, blockDim*blockDim)
	for i := range blockDim {
		for j := range blockDim {
			result[j*blockDim+i] = m[i*blockDim+j]
		}
	}
	return result
}

// referenceBlockMulAddFloat64 computes C += A * B using naive triple loop for float64.
// aT is the transposed A, b is normal B.
func referenceBlockMulAddFloat64(aT, b, c []float64, blockDim int) {
	for i := range blockDim {
		for j := range blockDim {
			var sum float64
			for k := range blockDim {
				// A[i,k] = aT[k,i]
				aik := aT[k*blockDim+i]
				bkj := b[k*blockDim+j]
				sum += aik * bkj
			}
			c[i*blockDim+j] += sum
		}
	}
}

// BenchmarkBlockMulAddNEONF32 benchmarks the hand-written NEON assembly.
func BenchmarkBlockMulAddNEONF32(b *testing.B) {
	blockSizes := []int{32, 48, 64}

	for _, blockDim := range blockSizes {
		size := blockDim * blockDim

		aT := make([]float32, size)
		bMat := make([]float32, size)
		c := make([]float32, size)

		for i := range aT {
			aT[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*blockDim*blockDim*blockDim) / 1e9

		b.Run(sizeStr(blockDim)+"/NEON", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				asm.BlockMulAddNEONF32(aT, bMat, c, blockDim)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}
