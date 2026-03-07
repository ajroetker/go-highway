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
	"math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// TestMultiTileFMOPADirect calls the multi-tile assembly kernel directly.
func TestMultiTileFMOPADirect(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	m, n, k := 32, 32, 32
	// AT is K x M (already transposed)
	at := make([]float32, k*m)
	b := make([]float32, k*n)
	c := make([]float32, m*n)

	// Fill with simple values: identity-like
	for i := range k {
		for j := range m {
			if i == j {
				at[i*m+j] = 1.0
			}
		}
	}
	for i := range k {
		for j := range n {
			b[i*n+j] = float32(i*n + j)
		}
	}

	defer hwy.SMEGuard()()
	asm.MultiTileMatMulFMOPAF32(at, b, c, m, n, k)

	// With AT = identity transposed, C should equal B (first 32 rows)
	var maxErr float32
	for i := range m {
		for j := range n {
			expected := b[i*n+j]
			err := float32(math.Abs(float64(c[i*n+j] - expected)))
			if err > maxErr {
				maxErr = err
			}
		}
	}
	t.Logf("32x32 multi-tile direct: max error %e", maxErr)
	if maxErr > 1e-4 {
		t.Errorf("max error %e exceeds threshold", maxErr)
	}
}

// TestMultiTileFMOPANTile tests the N-tiled FMOPA kernel for correctness.
func TestMultiTileFMOPANTile(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	type shape struct {
		name      string
		m, n, k   int
		nTiles    int
	}
	shapes := []shape{
		{"16x64_2tile", 16, 64, 32, 2},
		{"16x1024_4tile", 16, 1024, 640, 4},
		{"32x2048_4tile", 32, 2048, 640, 4},
		{"16x256_2tile", 16, 256, 640, 2},
	}

	for _, s := range shapes {
		t.Run(s.name, func(t *testing.T) {
			// Create A, B, compute reference C with standard FMOPA
			a := make([]float32, s.m*s.k)
			b := make([]float32, s.k*s.n)
			cRef := make([]float32, s.m*s.n)
			cNTile := make([]float32, s.m*s.n)
			at := make([]float32, s.k*s.m)

			for i := range a {
				a[i] = float32(i%7) * 0.1
			}
			for i := range b {
				b[i] = float32(i%11) * 0.1
			}
			Transpose2D(a, s.m, s.k, at)

			// Reference: single FMOPA call
			func() {
				defer hwy.SMEGuard()()
				asm.MultiTileMatMulFMOPAF32(at, b, cRef, s.m, s.n, s.k)
			}()

			// NTile: split across tiles
			tileN := AlignUp(s.n/s.nTiles, 16)
			for tile := range s.nTiles {
				j0 := tile * tileN
				j1 := min(j0+tileN, s.n)
				tn := j1 - j0
				func() {
					defer hwy.SMEGuard()()
					asm.MultiTileMatMulFMOPAF32NTile(at, b[j0:], cNTile, s.m, tn, s.k, s.n, s.n, j0)
				}()
			}

			// Compare
			var maxErr float32
			for i := range s.m {
				for j := range s.n {
					ref := cRef[i*s.n+j]
					got := cNTile[i*s.n+j]
					err := float32(math.Abs(float64(got - ref)))
					if err > maxErr {
						maxErr = err
					}
				}
			}
			t.Logf("max error: %e", maxErr)
			if maxErr > 1e-3 {
				t.Errorf("max error %e exceeds threshold", maxErr)
			}
		})
	}
}

// TestMultiTileFMOPAF16NTile tests the F16 N-tiled FMOPA kernel for correctness.
func TestMultiTileFMOPAF16NTile(t *testing.T) {
	if !hwy.HasSME() || !hwy.HasARMFP16() {
		t.Skip("SME or FP16 not available")
	}

	type shape struct {
		name    string
		m, n, k int
		nTiles  int
	}
	shapes := []shape{
		{"16x64_2tile", 16, 64, 32, 2},
		{"16x1024_4tile", 16, 1024, 640, 4},
		{"32x2048_4tile", 32, 2048, 640, 4},
	}

	for _, s := range shapes {
		t.Run(s.name, func(t *testing.T) {
			a := make([]hwy.Float16, s.m*s.k)
			b := make([]hwy.Float16, s.k*s.n)
			cRef := make([]hwy.Float16, s.m*s.n)
			cNTile := make([]hwy.Float16, s.m*s.n)
			at := make([]hwy.Float16, s.k*s.m)

			for i := range a {
				a[i] = hwy.NewFloat16(float32(i%7) * 0.1)
			}
			for i := range b {
				b[i] = hwy.NewFloat16(float32(i%11) * 0.1)
			}
			Transpose2D(a, s.m, s.k, at)

			// Reference
			func() {
				defer hwy.SMEGuard()()
				asm.MultiTileMatMulFMOPAF16(at, b, cRef, s.m, s.n, s.k)
			}()

			// NTile
			tileN := AlignUp(s.n/s.nTiles, 16)
			for tile := range s.nTiles {
				j0 := tile * tileN
				j1 := min(j0+tileN, s.n)
				tn := j1 - j0
				func() {
					defer hwy.SMEGuard()()
					asm.MultiTileMatMulFMOPAF16NTile(at, b[j0:], cNTile, s.m, tn, s.k, s.n, s.n, j0)
				}()
			}

			var maxErr float32
			for i := range s.m {
				for j := range s.n {
					ref := cRef[i*s.n+j].Float32()
					got := cNTile[i*s.n+j].Float32()
					err := float32(math.Abs(float64(got - ref)))
					if err > maxErr {
						maxErr = err
					}
				}
			}
			t.Logf("max error: %e", maxErr)
			if maxErr > 1e-2 {
				t.Errorf("max error %e exceeds threshold", maxErr)
			}
		})
	}
}

// TestMultiTileFMOPABF16NTile tests the BF16 N-tiled FMOPA kernel for correctness.
func TestMultiTileFMOPABF16NTile(t *testing.T) {
	if !hwy.HasSME() || !hwy.HasARMBF16() {
		t.Skip("SME or BF16 not available")
	}

	type shape struct {
		name    string
		m, n, k int
		nTiles  int
	}
	shapes := []shape{
		{"16x64_2tile", 16, 64, 32, 2},
		{"16x1024_4tile", 16, 1024, 640, 4},
		{"32x2048_4tile", 32, 2048, 640, 4},
	}

	for _, s := range shapes {
		t.Run(s.name, func(t *testing.T) {
			a := make([]hwy.BFloat16, s.m*s.k)
			b := make([]hwy.BFloat16, s.k*s.n)
			cRef := make([]hwy.BFloat16, s.m*s.n)
			cNTile := make([]hwy.BFloat16, s.m*s.n)
			at := make([]hwy.BFloat16, s.k*s.m)

			for i := range a {
				a[i] = hwy.NewBFloat16(float32(i%7) * 0.1)
			}
			for i := range b {
				b[i] = hwy.NewBFloat16(float32(i%11) * 0.1)
			}
			Transpose2D(a, s.m, s.k, at)

			// Reference
			func() {
				defer hwy.SMEGuard()()
				asm.MultiTileMatMulFMOPABF16(at, b, cRef, s.m, s.n, s.k)
			}()

			// NTile
			tileN := AlignUp(s.n/s.nTiles, 16)
			for tile := range s.nTiles {
				j0 := tile * tileN
				j1 := min(j0+tileN, s.n)
				tn := j1 - j0
				func() {
					defer hwy.SMEGuard()()
					asm.MultiTileMatMulFMOPABF16NTile(at, b[j0:], cNTile, s.m, tn, s.k, s.n, s.n, j0)
				}()
			}

			var maxErr float32
			for i := range s.m {
				for j := range s.n {
					ref := cRef[i*s.n+j].Float32()
					got := cNTile[i*s.n+j].Float32()
					err := float32(math.Abs(float64(got - ref)))
					if err > maxErr {
						maxErr = err
					}
				}
			}
			t.Logf("max error: %e", maxErr)
			if maxErr > 1e-1 { // bf16 has less precision
				t.Errorf("max error %e exceeds threshold", maxErr)
			}
		})
	}
}
