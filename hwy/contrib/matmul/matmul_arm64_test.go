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

// BenchmarkMatMulNEON benchmarks NEON streaming matmul at various sizes.
func BenchmarkMatMulNEON(b *testing.B) {
	sizes := []int{32, 64, 128, 256, 512}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size), func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matMulAsmF32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkMatMulSME benchmarks SME multi-tile FMOPA matmul at various sizes.
func BenchmarkMatMulSME(b *testing.B) {
	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)
		at := make([]float32, k*m)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}
		Transpose2D(a, m, k, at)

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size), func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				asm.MultiTileMatMulFMOPAF32(at, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkMatMulDispatch benchmarks auto-dispatched matmul at various sizes.
func BenchmarkMatMulDispatch(b *testing.B) {
	sizes := []int{32, 64, 128, 256, 512}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size), func(b *testing.B) {
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

// BenchmarkBlockedMatMulNEON benchmarks NEON blocked matmul (hwygen and GOAT).
func BenchmarkBlockedMatMulNEON(b *testing.B) {
	sizes := []int{32, 48, 64, 128, 256, 512}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size), func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				blockedMatMulAsmF32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkBlockedMatMulSME benchmarks SME multi-tile FMOPA for blocked matmul.
func BenchmarkBlockedMatMulSME(b *testing.B) {
	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)
		at := make([]float32, k*m)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}
		Transpose2D(a, m, k, at)

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size), func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				asm.MultiTileMatMulFMOPAF32(at, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/transpose", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Transpose2D(a, m, k, at)
				asm.MultiTileMatMulFMOPAF32(at, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkBlockedMatMulDispatch benchmarks auto-dispatched blocked matmul.
func BenchmarkBlockedMatMulDispatch(b *testing.B) {
	sizes := []int{32, 48, 64, 128, 256, 512}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size), func(b *testing.B) {
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
