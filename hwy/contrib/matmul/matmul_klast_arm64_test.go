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

//go:build darwin && arm64

package matmul

import (
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// BenchmarkMatMulKLastNEONvsSME compares NEON vs SME at various sizes
func BenchmarkMatMulKLastNEONvsSME(b *testing.B) {
	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, n*k)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		// NEON dot-product (no transpose)
		b.Run(sizeStr(size)+"/NEON", func(b *testing.B) {
			b.SetBytes(int64((m*k + n*k + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				asm.MatMulKLastNEONF32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		// SME with transpose (only if aligned)
		if size%16 == 0 {
			b.Run(sizeStr(size)+"/SME_transpose", func(b *testing.B) {
				// Allocate transpose buffers
				at := make([]float32, k*m)
				bt := make([]float32, k*n)

				b.SetBytes(int64((m*k + n*k + m*n) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					// Transpose both matrices
					Transpose2D(a, m, k, at)
					Transpose2D(bMat, n, k, bt)
					// Call FMOPA
					asm.MatMulFMOPAF32(at, bt, c, m, n, k)
				}
				b.StopTimer()
				elapsed := b.Elapsed().Seconds()
				gflops := flops * float64(b.N) / elapsed
				b.ReportMetric(gflops, "GFLOPS")
			})
		}

		// Dispatch (auto-selects best path)
		b.Run(sizeStr(size)+"/Dispatch", func(b *testing.B) {
			b.SetBytes(int64((m*k + n*k + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulKLastFloat32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}
