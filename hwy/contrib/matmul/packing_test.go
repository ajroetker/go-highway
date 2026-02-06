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

package matmul

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// TestKernelDirect tests the micro-kernel directly with known inputs.
func TestKernelDirect(t *testing.T) {
	// Simple 2x2 matmul: C = A * B
	// A = [[1, 2], [3, 4]]  (2x2)
	// B = [[5, 6], [7, 8]]  (2x2)
	// C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

	// Pack A with mr=4 (padding with zeros)
	packedA := []float32{1, 3, 0, 0, 2, 4, 0, 0}

	// Pack B with nr=8 (padding with zeros)
	packedB := []float32{5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0}

	// Output C (n=2, so row stride is 2)
	c := make([]float32, 4*2)
	n := 2

	PackedMicroKernelPartial(packedA, packedB, c, n, 0, 0, 2, 4, 8, 2, 2)

	expected := []float32{19, 22, 43, 50, 0, 0, 0, 0}
	for i := range 4 {
		if c[i] != expected[i] {
			t.Errorf("c[%d] = %f, want %f", i, c[i], expected[i])
		}
	}
}

// TestBaseKernelGeneral directly tests basePackedMicroKernelGeneral.
func TestBaseKernelGeneral(t *testing.T) {
	packedA := []float32{1, 3, 0, 0, 2, 4, 0, 0}
	packedB := []float32{5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0}

	c := make([]float32, 4*8)
	n := 8

	basePackedMicroKernelGeneral(packedA, packedB, c, n, 0, 0, 2, 4, 8)

	if c[0] != 19 {
		t.Errorf("c[0,0] = %f, want 19", c[0])
	}
	if c[1] != 22 {
		t.Errorf("c[0,1] = %f, want 22", c[1])
	}
	if c[8] != 43 {
		t.Errorf("c[1,0] = %f, want 43", c[8])
	}
	if c[9] != 50 {
		t.Errorf("c[1,1] = %f, want 50", c[9])
	}
}

// TestScalarMatmulReference computes the expected result using pure scalar operations.
func TestScalarMatmulReference(t *testing.T) {
	packedA := []float32{1, 3, 0, 0, 2, 4, 0, 0}
	packedB := []float32{5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0}

	mr, nr, kc := 4, 8, 2

	// Compute C manually using scalar loop
	c := make([]float32, 4*8)
	n := 8

	for r := range mr {
		cRowStart := r * n
		for col := range nr {
			var sum float32
			for p := range kc {
				aVal := packedA[p*mr+r]
				bVal := packedB[p*nr+col]
				sum += aVal * bVal
			}
			c[cRowStart+col] += sum
		}
	}

	if c[0] != 19 {
		t.Errorf("Scalar: c[0,0] = %f, want 19", c[0])
	}
	if c[1] != 22 {
		t.Errorf("Scalar: c[0,1] = %f, want 22", c[1])
	}
	if c[8] != 43 {
		t.Errorf("Scalar: c[1,0] = %f, want 43", c[8])
	}
	if c[9] != 50 {
		t.Errorf("Scalar: c[1,1] = %f, want 50", c[9])
	}

	// Compare with general kernel
	c2 := make([]float32, 4*8)
	basePackedMicroKernelGeneral(packedA, packedB, c2, n, 0, 0, kc, mr, nr)

	for i := range 16 {
		if c[i] != c2[i] {
			t.Errorf("Mismatch at c[%d]: scalar=%f, general=%f", i, c[i], c2[i])
		}
	}
}

// TestPackedMatMulSmall tests packed matmul with a small matrix.
func TestPackedMatMulSmall(t *testing.T) {
	m, n, k := 8, 8, 4

	a := make([]float32, m*k)
	for i := range m {
		for j := range k {
			a[i*k+j] = float32(i*k + j + 1)
		}
	}

	b := make([]float32, k*n)
	for i := range k {
		for j := range n {
			b[i*n+j] = float32(i*n + j + 1)
		}
	}

	expected := make([]float32, m*n)
	for i := range m {
		for j := range n {
			var sum float32
			for kk := range k {
				sum += a[i*k+kk] * b[kk*n+j]
			}
			expected[i*n+j] = sum
		}
	}

	c := make([]float32, m*n)
	PackedMatMul(a, b, c, m, n, k)

	var maxErr float32
	for i := range m {
		for j := range n {
			idx := i*n + j
			diff := c[idx] - expected[idx]
			if diff < 0 {
				diff = -diff
			}
			if diff > maxErr {
				maxErr = diff
			}
		}
	}

	if maxErr > 1e-4 {
		t.Errorf("max error %f exceeds threshold", maxErr)
	}
}

// TestMicroKernelEdgePosition is a regression test for the bounds check bug
// where micro-kernels at positions like (ir=12, jr=8) in a 16x16 output
// would fail due to incorrect C slice bounds checking.
func TestMicroKernelEdgePosition(t *testing.T) {
	mr, nr := 4, 8
	m, n, k := 16, 16, 16

	packedA := make([]float32, k*mr)
	for i := range packedA {
		packedA[i] = float32(i + 1)
	}

	packedB := make([]float32, k*nr)
	for i := range packedB {
		packedB[i] = float32(i + 1)
	}

	// Compute expected result using scalar math
	var expected float32
	for p := range k {
		expected += packedA[p*mr+0] * packedB[p*nr+0]
	}

	// Test: ir=12, jr=8 (edge position that triggered the bounds check bug)
	c := make([]float32, m*n)
	PackedMicroKernel(packedA, packedB, c, n, 12, 8, k, mr, nr)

	if c[12*n+8] == 0 && expected != 0 {
		t.Errorf("ir=12, jr=8 produces 0, want %f", expected)
	}
	if c[12*n+8] != expected {
		t.Errorf("ir=12, jr=8: got %f, want %f", c[12*n+8], expected)
	}
}

// TestPackLHS verifies that LHS packing produces the expected layout.
func TestPackLHS(t *testing.T) {
	m, k := 6, 4
	mr := 2
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(i + 1)
	}

	numPanels := (m + mr - 1) / mr
	packed := make([]float32, numPanels*k*mr)

	activeRows := BasePackLHS(a, packed, m, k, 0, 0, m, k, mr)

	// Expected packed layout: [panel, k, mr]
	expected := []float32{
		1, 5, 2, 6, 3, 7, 4, 8, // Panel 0
		9, 13, 10, 14, 11, 15, 12, 16, // Panel 1
		17, 21, 18, 22, 19, 23, 20, 24, // Panel 2
	}

	if activeRows != mr {
		t.Errorf("activeRows = %d, want %d", activeRows, mr)
	}

	for i := range expected {
		if packed[i] != expected[i] {
			t.Errorf("packed[%d] = %f, want %f", i, packed[i], expected[i])
		}
	}
}

// TestPackRHS verifies that RHS packing produces the expected layout.
func TestPackRHS(t *testing.T) {
	k, n := 4, 6
	nr := 2
	b := make([]float32, k*n)
	for i := range b {
		b[i] = float32(i + 1)
	}

	numPanels := (n + nr - 1) / nr
	packed := make([]float32, numPanels*k*nr)

	activeCols := BasePackRHS(b, packed, k, n, 0, 0, k, n, nr)

	// Expected packed layout: [panel, k, nr]
	expected := []float32{
		1, 2, 7, 8, 13, 14, 19, 20, // Panel 0
		3, 4, 9, 10, 15, 16, 21, 22, // Panel 1
		5, 6, 11, 12, 17, 18, 23, 24, // Panel 2
	}

	if activeCols != nr {
		t.Errorf("activeCols = %d, want %d", activeCols, nr)
	}

	for i := range expected {
		if packed[i] != expected[i] {
			t.Errorf("packed[%d] = %f, want %f", i, packed[i], expected[i])
		}
	}
}

// TestPackedMatMul verifies packed matmul produces correct results.
func TestPackedMatMul(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	sizes := []int{16, 32, 48, 64, 96, 128, 256}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			for i := range a {
				a[i] = rng.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rng.Float32()*2 - 1
			}

			matmulReference(a, b, expected, m, n, k)
			PackedMatMul(a, b, c, m, n, k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := float32(1e-4) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// TestPackedMatMulNonSquare verifies packed matmul with non-square matrices.
func TestPackedMatMulNonSquare(t *testing.T) {
	testCases := []struct {
		m, n, k int
	}{
		{64, 128, 32},
		{128, 64, 96},
		{100, 200, 150},
		{37, 53, 41},
		{256, 512, 128},
	}

	for _, tc := range testCases {
		name := sizeStr(tc.m) + "x" + sizeStr(tc.n) + "x" + sizeStr(tc.k)
		t.Run(name, func(t *testing.T) {
			a := make([]float32, tc.m*tc.k)
			b := make([]float32, tc.k*tc.n)
			c := make([]float32, tc.m*tc.n)
			expected := make([]float32, tc.m*tc.n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulReference(a, b, expected, tc.m, tc.n, tc.k)
			PackedMatMul(a, b, c, tc.m, tc.n, tc.k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := float32(1e-4) * float32(tc.k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// TestParallelPackedMatMul verifies parallel packed matmul produces correct results.
func TestParallelPackedMatMul(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	sizes := []int{256, 512}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulReference(a, b, expected, m, n, k)
			ParallelPackedMatMul(pool, a, b, c, m, n, k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := float32(1e-4) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// BenchmarkPackedMatMul benchmarks the packed matmul.
func BenchmarkPackedMatMul(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{128, 256, 512, 1024}

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
				PackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkParallelPackedMatMul benchmarks the parallel packed matmul.
func BenchmarkParallelPackedMatMul(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{256, 512, 1024}

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
				ParallelPackedMatMul(pool, a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkPackedVsBlocked compares packed and blocked matmul side-by-side.
func BenchmarkPackedVsBlocked(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{256, 512, 1024}

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

		b.Run(sizeStr(size)+"/Blocked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				BlockedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Packed", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				PackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/ParallelBlocked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ParallelMatMul(pool, a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/ParallelPacked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ParallelPackedMatMul(pool, a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkAllAlgorithms compares all matmul algorithms.
func BenchmarkAllAlgorithms(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{64, 128, 256, 512, 1024}

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

		b.Run(sizeStr(size)+"/Streaming", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Blocked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				BlockedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Packed", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				PackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Auto", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMulAuto(pool, a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkPacking benchmarks the packing operations themselves.
func BenchmarkPacking(b *testing.B) {
	size := 512
	m, n, k := size, size, size

	a := make([]float32, m*k)
	bMat := make([]float32, k*n)

	for i := range a {
		a[i] = rand.Float32()
	}
	for i := range bMat {
		bMat[i] = rand.Float32()
	}

	params := getCacheParams[float32]()
	packedA := make([]float32, params.PackedASize())
	packedB := make([]float32, params.PackedBSize())

	// Clamp panel sizes to actual matrix dimensions (matching how matmul uses these)
	panelRows := min(params.Mc, m)
	panelK := min(params.Kc, k)
	panelCols := min(params.Nc, n)

	b.Run("PackLHS", func(b *testing.B) {
		b.SetBytes(int64(panelRows * panelK * 4))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			PackLHS(a, packedA, m, k, 0, 0, panelRows, panelK, params.Mr)
		}
	})

	b.Run("PackRHS", func(b *testing.B) {
		b.SetBytes(int64(panelCols * panelK * 4))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			PackRHS(bMat, packedB, k, n, 0, 0, panelK, panelCols, params.Nr)
		}
	})
}
