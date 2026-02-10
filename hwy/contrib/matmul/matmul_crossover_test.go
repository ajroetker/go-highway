// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// BenchmarkParallelCrossover sweeps matrix shapes to find where parallel
// implementations become faster than sequential BlockedMatMul.
//
// Run with:
//   GOEXPERIMENT=simd go test -bench=BenchmarkParallelCrossover -benchmem -timeout=10m ./hwy/contrib/matmul/
func BenchmarkParallelCrossover(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	b.Logf("Dispatch level: %s", hwy.CurrentName())

	// Square matrices: find crossover for NxNxN
	squareSizes := []int{16, 32, 48, 64, 96, 128, 192, 256, 384, 512}

	b.Run("Square", func(b *testing.B) {
		for _, size := range squareSizes {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			bMat := make([]float32, k*n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range bMat {
				bMat[i] = rand.Float32()*2 - 1
			}

			b.Run(fmt.Sprintf("%dx%dx%d/Blocked", m, n, k), func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					BlockedMatMul(a, bMat, c, m, n, k)
				}
			})

			b.Run(fmt.Sprintf("%dx%dx%d/Parallel", m, n, k), func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					ParallelMatMul(pool, a, bMat, c, m, n, k)
				}
			})

			b.Run(fmt.Sprintf("%dx%dx%d/FineGrained", m, n, k), func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					ParallelMatMulFineGrained(pool, a, bMat, c, m, n, k)
				}
			})

			b.Run(fmt.Sprintf("%dx%dx%d/Auto", m, n, k), func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					MatMulAuto(pool, a, bMat, c, m, n, k)
				}
			})
		}
	})

	// Tall-skinny: small M, large N and K (transformer decode-like)
	b.Run("TallSkinny", func(b *testing.B) {
		configs := []struct{ m, n, k int }{
			{1, 512, 512},
			{1, 1024, 1024},
			{4, 512, 512},
			{4, 1024, 1024},
			{8, 512, 512},
			{8, 1024, 1024},
			{11, 1024, 1024},
			{16, 512, 512},
			{16, 1024, 1024},
			{32, 512, 512},
			{32, 1024, 1024},
			{64, 512, 512},
			{64, 1024, 1024},
		}

		for _, cfg := range configs {
			m, n, k := cfg.m, cfg.n, cfg.k

			a := make([]float32, m*k)
			bMat := make([]float32, k*n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range bMat {
				bMat[i] = rand.Float32()*2 - 1
			}

			label := fmt.Sprintf("%dx%dx%d", m, n, k)

			b.Run(label+"/Blocked", func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					BlockedMatMul(a, bMat, c, m, n, k)
				}
			})

			b.Run(label+"/Parallel", func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					ParallelMatMul(pool, a, bMat, c, m, n, k)
				}
			})

			b.Run(label+"/FineGrained", func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					ParallelMatMulFineGrained(pool, a, bMat, c, m, n, k)
				}
			})

			b.Run(label+"/Auto", func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					MatMulAuto(pool, a, bMat, c, m, n, k)
				}
			})
		}
	})

	// Wide: large M, varying N and K
	b.Run("Wide", func(b *testing.B) {
		configs := []struct{ m, n, k int }{
			{256, 64, 256},
			{256, 128, 256},
			{256, 256, 64},
			{512, 64, 512},
			{512, 128, 128},
			{1024, 64, 64},
			{1024, 128, 128},
		}

		for _, cfg := range configs {
			m, n, k := cfg.m, cfg.n, cfg.k

			a := make([]float32, m*k)
			bMat := make([]float32, k*n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range bMat {
				bMat[i] = rand.Float32()*2 - 1
			}

			label := fmt.Sprintf("%dx%dx%d", m, n, k)

			b.Run(label+"/Blocked", func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					BlockedMatMul(a, bMat, c, m, n, k)
				}
			})

			b.Run(label+"/Parallel", func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					ParallelMatMul(pool, a, bMat, c, m, n, k)
				}
			})

			b.Run(label+"/FineGrained", func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					ParallelMatMulFineGrained(pool, a, bMat, c, m, n, k)
				}
			})

			b.Run(label+"/Auto", func(b *testing.B) {
				c := make([]float32, m*n)
				b.SetBytes(int64(2 * m * n * k * 4))
				b.ResetTimer()
				for range b.N {
					MatMulAuto(pool, a, bMat, c, m, n, k)
				}
			})
		}
	})
}
