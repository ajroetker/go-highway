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
	"fmt"
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
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

// BenchmarkMatMulM1 compares SME FMOPA vs NEON for M=1 (decode) at LLM dimensions.
// SME pads M=1→16, transposes A, and wastes 15/16 compute.
// NEON streams through B row-by-row with no overhead.
func BenchmarkMatMulM1(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	type shape struct {
		k, n int
	}
	shapes := []shape{
		{256, 256},
		{512, 512},
		{1024, 1024},
		{2048, 2048},
		{4096, 4096},
		// Asymmetric shapes common in transformers (hidden → 4*hidden)
		{2048, 8192},
		{4096, 11008},
	}

	for _, s := range shapes {
		m, n, k := 1, s.n, s.k

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
		label := fmt.Sprintf("%dx%d", k, n)

		b.Run(label+"/SME", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				blockedMatMulFMOPA(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			b.ReportMetric(flops*float64(b.N)/elapsed, "GFLOPS")
		})

		b.Run(label+"/NEON", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matMulAsmF32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			b.ReportMetric(flops*float64(b.N)/elapsed, "GFLOPS")
		})
	}
}

// BenchmarkMatMulM1Dispatch benchmarks the dispatched path at M=1 to verify
// it picks NEON for small K*N and SME for large K*N.
func BenchmarkMatMulM1Dispatch(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	type shape struct{ k, n int }
	shapes := []shape{
		{512, 512},
		{2048, 2048},
		{4096, 4096},
		{2048, 8192},
	}

	for _, s := range shapes {
		m, n, k := 1, s.n, s.k

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
		label := fmt.Sprintf("%dx%d", k, n)

		b.Run(label, func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				BlockedMatMulFloat32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			b.ReportMetric(flops*float64(b.N)/elapsed, "GFLOPS")
		})
	}
}

// BenchmarkPrefillMatMul benchmarks small-M matmul at LLM prefill shapes.
// Compares: sequential SME (current), per-row NEON parallel, and MatMulAuto.
// Gemma 270M shapes: hidden=640, intermediate=2048, heads=4, kv_heads=1, head_dim=256.
func BenchmarkPrefillMatMul(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	pool := workerpool.New(0) // use all cores
	defer pool.Close()

	type shape struct {
		name   string
		m, k, n int
	}
	shapes := []shape{
		{"Qproj_10x640x1024", 10, 640, 1024},
		{"Kproj_10x640x256", 10, 640, 256},
		{"gate_10x640x2048", 10, 640, 2048},
		{"down_10x2048x640", 10, 2048, 640},
		{"Qproj_13x640x1024", 13, 640, 1024},
		{"gate_13x640x2048", 13, 640, 2048},
	}

	for _, s := range shapes {
		a := make([]float32, s.m*s.k)
		bMat := make([]float32, s.k*s.n)
		c := make([]float32, s.m*s.n)
		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*s.m*s.n*s.k) / 1e9

		// Current path: sequential BlockedMatMul (SME FMOPA)
		b.Run(s.name+"/BlockedSME", func(b *testing.B) {
			b.SetBytes(int64((s.m*s.k + s.k*s.n + s.m*s.n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				BlockedMatMulFloat32(a, bMat, c, s.m, s.n, s.k)
			}
			b.StopTimer()
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds(), "GFLOPS")
		})

		// Proposed: per-row NEON parallel
		b.Run(s.name+"/ParallelNEON", func(b *testing.B) {
			b.SetBytes(int64((s.m*s.k + s.k*s.n + s.m*s.n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ParallelMatMulFineGrained(pool, a, bMat, c, s.m, s.n, s.k)
			}
			b.StopTimer()
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds(), "GFLOPS")
		})

		// MatMulAuto (current dispatch)
		b.Run(s.name+"/Auto", func(b *testing.B) {
			b.SetBytes(int64((s.m*s.k + s.k*s.n + s.m*s.n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulAuto(pool, a, bMat, c, s.m, s.n, s.k)
			}
			b.StopTimer()
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds(), "GFLOPS")
		})

		// Single-core NEON (no parallel dispatch overhead)
		b.Run(s.name+"/NEON_single", func(b *testing.B) {
			b.SetBytes(int64((s.m*s.k + s.k*s.n + s.m*s.n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matMulAsmF32(a, bMat, c, s.m, s.n, s.k)
			}
			b.StopTimer()
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds(), "GFLOPS")
		})

		// NEON blocked kernel (single core)
		b.Run(s.name+"/NEONblocked_single", func(b *testing.B) {
			b.SetBytes(int64((s.m*s.k + s.k*s.n + s.m*s.n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				blockedMatMulAsmF32(a, bMat, c, s.m, s.n, s.k)
			}
			b.StopTimer()
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds(), "GFLOPS")
		})

		// SME FMOPA compute only (pre-transposed, no pad/transpose overhead)
		if s.m <= 16 && s.k%16 == 0 && s.n%16 == 0 {
			at := make([]float32, 16*s.k)
			padA := make([]float32, 16*s.k)
			PadMatrix2D(padA, a, s.m, s.k, 16, s.k)
			Transpose2D(padA, 16, s.k, at)
			cPad := make([]float32, 16*s.n)

			b.Run(s.name+"/SME_compute_only", func(b *testing.B) {
				b.SetBytes(int64((s.m*s.k + s.k*s.n + s.m*s.n) * 4))
				defer hwy.SMEGuard()()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					asm.MultiTileMatMulFMOPAF32(at, bMat, cPad, 16, s.n, s.k)
				}
				b.StopTimer()
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds(), "GFLOPS")
			})
		}

		// Parallel SME: shared transpose, split FMOPA across N tiles (pre-extracted B)
		if s.m <= 16 && s.k%16 == 0 && s.n%16 == 0 {
			paddedM := 16
			at := make([]float32, paddedM*s.k)
			padA := make([]float32, paddedM*s.k)
			PadMatrix2D(padA, a, s.m, s.k, paddedM, s.k)
			Transpose2D(padA, paddedM, s.k, at)

			nTiles := 4
			tileN := AlignUp(s.n/nTiles, 16)

			// Pre-extract B column tiles into contiguous buffers
			bTiles := make([][]float32, nTiles)
			cTiles := make([][]float32, nTiles)
			for t := range nTiles {
				j0 := t * tileN
				j1 := min(j0+tileN, s.n)
				tn := j1 - j0
				bTiles[t] = make([]float32, s.k*tn)
				for kk := range s.k {
					copy(bTiles[t][kk*tn:(kk+1)*tn], bMat[kk*s.n+j0:kk*s.n+j1])
				}
				cTiles[t] = make([]float32, paddedM*tn)
			}

			b.Run(s.name+"/ParallelSME_4tile_extract", func(b *testing.B) {
				b.SetBytes(int64((s.m*s.k + s.k*s.n + s.m*s.n) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					pool.ParallelForAtomic(nTiles, func(t int) {
						j0 := t * tileN
						j1 := min(j0+tileN, s.n)
						tn := j1 - j0
						defer hwy.SMEGuard()()
						asm.MultiTileMatMulFMOPAF32(at, bTiles[t], cTiles[t], paddedM, tn, s.k)
					})
				}
				b.StopTimer()
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds(), "GFLOPS")
			})

			// Parallel SME with NTile kernel (zero-copy B access)
			cPad := make([]float32, paddedM*s.n)
			b.Run(s.name+"/ParallelSME_4tile_ntile", func(b *testing.B) {
				b.SetBytes(int64((s.m*s.k + s.k*s.n + s.m*s.n) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					pool.ParallelForAtomic(nTiles, func(t int) {
						j0 := t * tileN
						j1 := min(j0+tileN, s.n)
						tn := j1 - j0
						defer hwy.SMEGuard()()
						asm.MultiTileMatMulFMOPAF32NTile(at, bMat[j0:], cPad, paddedM, tn, s.k, s.n, s.n, j0)
					})
				}
				b.StopTimer()
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds(), "GFLOPS")
			})
		}
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
