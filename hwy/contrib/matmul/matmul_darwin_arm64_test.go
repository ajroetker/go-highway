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

// TestSMESmallN exercises the 1×4 tile fast path remainder branches (N < 64
// for f32/f16/bf16, N < 32 for f64). These paths previously caused SIGILL when
// clang deferred smstart past the loop entry into remainder-only code paths.
func TestSMESmallN(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	// f32: test N values that skip the 64-col loop (N=16, 32, 48)
	// and values that hit each remainder (N=80 = 64+16, N=96 = 64+32).
	t.Run("f32", func(t *testing.T) {
		for _, n := range []int{16, 32, 48, 80, 96} {
			m, k := 16, 32
			at := make([]float32, k*m)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			for i := range at {
				at[i] = float32(i%7) * 0.1
			}
			for i := range b {
				b[i] = float32(i%11) * 0.1
			}

			// Base kernel
			cRef := make([]float32, m*n)
			func() {
				defer hwy.SMEGuard()()
				asm.MultiTileMatMulFMOPAF32(at, b, cRef, m, n, k)
			}()

			// Strided kernel
			cStr := make([]float32, m*n)
			func() {
				defer hwy.SMEGuard()()
				asm.MultiTileMatMulFMOPAF32Strided(at, b, cStr, m, n, k, n, 0)
			}()

			// NTile kernel
			cNT := make([]float32, m*n)
			func() {
				defer hwy.SMEGuard()()
				asm.MultiTileMatMulFMOPAF32NTile(at, b, cNT, m, n, k, n, n, 0)
			}()

			for _, pair := range []struct {
				name string
				got  []float32
			}{{"strided", cStr}, {"ntile", cNT}} {
				var maxErr float32
				for i := range c {
					e := float32(math.Abs(float64(pair.got[i] - cRef[i])))
					if e > maxErr {
						maxErr = e
					}
				}
				if maxErr > 1e-4 {
					t.Errorf("f32 N=%d %s: max error %e", n, pair.name, maxErr)
				}
			}
		}
	})

	// f64: test N values that skip the 32-col loop (N=8, 16, 24)
	t.Run("f64", func(t *testing.T) {
		for _, n := range []int{8, 16, 24, 40, 48} {
			m, k := 8, 32
			at := make([]float64, k*m)
			b := make([]float64, k*n)
			c := make([]float64, m*n)
			for i := range at {
				at[i] = float64(i%7) * 0.1
			}
			for i := range b {
				b[i] = float64(i%11) * 0.1
			}

			cRef := make([]float64, m*n)
			func() {
				defer hwy.SMEGuard()()
				asm.MultiTileMatMulFMOPAF64(at, b, cRef, m, n, k)
			}()

			cStr := make([]float64, m*n)
			func() {
				defer hwy.SMEGuard()()
				asm.MultiTileMatMulFMOPAF64Strided(at, b, cStr, m, n, k, n, 0)
			}()

			cNT := make([]float64, m*n)
			func() {
				defer hwy.SMEGuard()()
				asm.MultiTileMatMulFMOPAF64NTile(at, b, cNT, m, n, k, n, n, 0)
			}()

			for _, pair := range []struct {
				name string
				got  []float64
			}{{"strided", cStr}, {"ntile", cNT}} {
				var maxErr float64
				for i := range c {
					e := math.Abs(pair.got[i] - cRef[i])
					if e > maxErr {
						maxErr = e
					}
				}
				if maxErr > 1e-10 {
					t.Errorf("f64 N=%d %s: max error %e", n, pair.name, maxErr)
				}
			}
		}
	})

	// f16: small N remainder paths
	if hwy.HasARMFP16() {
		t.Run("f16", func(t *testing.T) {
			for _, n := range []int{16, 32, 48, 80} {
				m, k := 16, 32
				at := make([]hwy.Float16, k*m)
				b := make([]hwy.Float16, k*n)
				for i := range at {
					at[i] = hwy.NewFloat16(float32(i%7) * 0.1)
				}
				for i := range b {
					b[i] = hwy.NewFloat16(float32(i%11) * 0.1)
				}

				cRef := make([]hwy.Float16, m*n)
				func() {
					defer hwy.SMEGuard()()
					asm.MultiTileMatMulFMOPAF16(at, b, cRef, m, n, k)
				}()

				cNT := make([]hwy.Float16, m*n)
				func() {
					defer hwy.SMEGuard()()
					asm.MultiTileMatMulFMOPAF16NTile(at, b, cNT, m, n, k, n, n, 0)
				}()

				var maxErr float32
				for i := range m * n {
					e := float32(math.Abs(float64(cNT[i].Float32() - cRef[i].Float32())))
					if e > maxErr {
						maxErr = e
					}
				}
				if maxErr > 1e-2 {
					t.Errorf("f16 N=%d ntile: max error %e", n, maxErr)
				}
			}
		})
	}

	// bf16: small N remainder paths
	if hwy.HasARMBF16() {
		t.Run("bf16", func(t *testing.T) {
			for _, n := range []int{16, 32, 48, 80} {
				m, k := 16, 32
				at := make([]hwy.BFloat16, k*m)
				b := make([]hwy.BFloat16, k*n)
				for i := range at {
					at[i] = hwy.NewBFloat16(float32(i%7) * 0.1)
				}
				for i := range b {
					b[i] = hwy.NewBFloat16(float32(i%11) * 0.1)
				}

				cRef := make([]hwy.BFloat16, m*n)
				func() {
					defer hwy.SMEGuard()()
					asm.MultiTileMatMulFMOPABF16(at, b, cRef, m, n, k)
				}()

				cNT := make([]hwy.BFloat16, m*n)
				func() {
					defer hwy.SMEGuard()()
					asm.MultiTileMatMulFMOPABF16NTile(at, b, cNT, m, n, k, n, n, 0)
				}()

				var maxErr float32
				for i := range m * n {
					e := float32(math.Abs(float64(cNT[i].Float32() - cRef[i].Float32())))
					if e > maxErr {
						maxErr = e
					}
				}
				if maxErr > 1e-1 {
					t.Errorf("bf16 N=%d ntile: max error %e", n, maxErr)
				}
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
