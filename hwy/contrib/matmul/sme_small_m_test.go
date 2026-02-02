// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64

package matmul

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// BenchmarkSMEvsNEONSmallM directly compares the SME FMOPA path (with padding)
// against NEON for small M values to determine if the minDimForBlockedSME=32
// threshold should be lowered.
//
// The SME path pads M up to a tile boundary (16) and handles extraction,
// so M=1 becomes a 16×N×K SME operation with output extraction.
func BenchmarkSMEvsNEONSmallM(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

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

		// NEON path (what FineGrained currently uses per-row)
		b.Run(label+"/NEON", func(b *testing.B) {
			c := make([]float32, m*n)
			b.ResetTimer()
			for range b.N {
				asm.MatMulNEONF32(a, bMat, c, m, n, k)
			}
		})

		// Full SME path with padding/transpose/extract
		b.Run(label+"/SME_full", func(b *testing.B) {
			c := make([]float32, m*n)
			b.ResetTimer()
			for range b.N {
				blockedMatMulFMOPAForBench(a, bMat, c, m, n, k)
			}
		})
	}
}

// blockedMatMulFMOPAForBench is blockedMatMulFMOPA without the minDimForBlockedSME guard.
func blockedMatMulFMOPAForBench(a, b, c []float32, m, n, k int) {
	const tileSize = 16
	paddedM := AlignUp(m, tileSize)
	paddedN := AlignUp(n, tileSize)
	paddedK := AlignUp(k, tileSize)

	defer hwy.SMEGuard()()

	needsPadM := paddedM != m
	needsPadK := paddedK != k
	needsPadN := paddedN != n

	fmopaA := a
	fmopaK := k
	if needsPadM || needsPadK {
		paSize := paddedM * paddedK
		paBuf := make([]float32, paSize)
		PadMatrix2D(paBuf, a, m, k, paddedM, paddedK)
		fmopaA = paBuf
		fmopaK = paddedK
	}

	fmopaB := b
	fmopaN := n
	if needsPadK || needsPadN {
		pbSize := paddedK * paddedN
		pbBuf := make([]float32, pbSize)
		PadMatrix2D(pbBuf, b, k, n, paddedK, paddedN)
		fmopaB = pbBuf
		fmopaN = paddedN
	}

	fmopaM := paddedM

	atSize := fmopaK * fmopaM
	atBuf := make([]float32, atSize)
	transposeMatrix(fmopaA, fmopaM, fmopaK, atBuf)

	if needsPadM || needsPadN {
		pcSize := fmopaM * fmopaN
		paddedC := make([]float32, pcSize)
		asm.MultiTileMatMulFMOPAF32(atBuf, fmopaB, paddedC, fmopaM, fmopaN, fmopaK)
		ExtractMatrix2D(c, paddedC, m, n, fmopaN)
	} else {
		asm.MultiTileMatMulFMOPAF32(atBuf, fmopaB, c, fmopaM, fmopaN, fmopaK)
	}
}

// TestSMESmallMCorrectness verifies that the SME path produces correct results
// for small M values (below current minDimForBlockedSME=32).
func TestSMESmallMCorrectness(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	configs := []struct{ m, n, k int }{
		{1, 64, 64},
		{1, 512, 512},
		{4, 128, 128},
		{8, 256, 256},
		{11, 512, 512},
		{16, 512, 512},
	}

	for _, cfg := range configs {
		m, n, k := cfg.m, cfg.n, cfg.k
		t.Run(fmt.Sprintf("%dx%dx%d", m, n, k), func(t *testing.T) {
			a := make([]float32, m*k)
			b := make([]float32, k*n)
			for i := range a {
				a[i] = float32(i%7-3) * 0.1
			}
			for i := range b {
				b[i] = float32(i%5-2) * 0.1
			}

			cRef := make([]float32, m*n)
			cSME := make([]float32, m*n)

			matmulScalar(a, b, cRef, m, n, k)
			blockedMatMulFMOPAForBench(a, b, cSME, m, n, k)

			var maxErr float32
			for i := range cRef {
				err := float32(math.Abs(float64(cSME[i] - cRef[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tol := float32(1e-4)
			if k >= 256 {
				tol = 1e-3
			}
			if maxErr > tol {
				t.Errorf("max error %v exceeds tolerance %v", maxErr, tol)
			} else {
				t.Logf("max error: %v", maxErr)
			}
		})
	}
}
