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

// BenchmarkMatMulNEONvsSME compares NEON vs SME at various sizes
func BenchmarkMatMulNEONvsSME(b *testing.B) {
	sizes := []int{32, 64, 128, 256, 512}

	for _, size := range sizes {
		m, n, k := size, size, size

		// Standard layout: A [M,K], B [K,N]
		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		// For FMOPA we need AT [K,M]
		at := make([]float32, k*m)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}
		// Transpose A to AT
		Transpose2D(a, m, k, at)

		flops := float64(2*m*n*k) / 1e9

		// NEON streaming (no transpose needed, uses A directly)
		b.Run(sizeStr(size)+"/NEON", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				asm.MatMulNEONF32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		// SME FMOPA (uses pre-transposed AT)
		if size%16 == 0 {
			b.Run(sizeStr(size)+"/SME", func(b *testing.B) {
				b.SetBytes(int64((m*k + k*n + m*n) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					asm.MatMulFMOPAF32(at, bMat, c, m, n, k)
				}
				b.StopTimer()
				elapsed := b.Elapsed().Seconds()
				gflops := flops * float64(b.N) / elapsed
				b.ReportMetric(gflops, "GFLOPS")
			})
		}

		// Dispatch (auto-selects best path)
		b.Run(sizeStr(size)+"/Dispatch", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulFloat32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkBlockedMatMulNEONvsSME compares NEON vs SME for blocked matmul.
// This helps determine the optimal threshold for minDimForBlockedSME.
func BenchmarkBlockedMatMulNEONvsSME(b *testing.B) {
	sizes := []int{32, 48, 64, 128, 256, 512}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		// For FMOPA we need AT [K,M]
		at := make([]float32, k*m)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}
		// Transpose A to AT
		Transpose2D(a, m, k, at)

		flops := float64(2*m*n*k) / 1e9

		// NEON blocked (hwygen-generated) - known to be slow (~2 GFLOPS)
		b.Run(sizeStr(size)+"/NEON_hwygen", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				BaseBlockedMatMul_neon(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		// NEON blocked (GOAT-generated)
		b.Run(sizeStr(size)+"/NEON_GOAT", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				asm.BlockedMatMulNEONF32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		// SME blocked FMOPA (uses pre-transposed AT) - only for 16-aligned sizes
		if size%16 == 0 {
			b.Run(sizeStr(size)+"/SME", func(b *testing.B) {
				b.SetBytes(int64((m*k + k*n + m*n) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					asm.BlockedMatMulFMOPAF32(at, bMat, c, m, n, k)
				}
				b.StopTimer()
				elapsed := b.Elapsed().Seconds()
				gflops := flops * float64(b.N) / elapsed
				b.ReportMetric(gflops, "GFLOPS")
			})

			// SME with transpose included in timing
			b.Run(sizeStr(size)+"/SME_transpose", func(b *testing.B) {
				b.SetBytes(int64((m*k + k*n + m*n) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					Transpose2D(a, m, k, at)
					asm.BlockedMatMulFMOPAF32(at, bMat, c, m, n, k)
				}
				b.StopTimer()
				elapsed := b.Elapsed().Seconds()
				gflops := flops * float64(b.N) / elapsed
				b.ReportMetric(gflops, "GFLOPS")
			})
		}

		// Dispatch (auto-selects best path)
		b.Run(sizeStr(size)+"/Dispatch", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				BlockedMatMulFloat32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}
