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

// matmulKLastReference computes C = A * B^T using naive triple loop.
// A is M×K, B is N×K, C is M×N.
// C[i,j] = sum(A[i,p] * B[j,p]) for p in 0..K-1
func matmulKLastReference(a, b, c []float32, m, n, k int) {
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += a[i*k+p] * b[j*k+p]
			}
			c[i*n+j] = sum
		}
	}
}

// matmulKLastReference64 computes C = A * B^T for float64.
func matmulKLastReference64(a, b, c []float64, m, n, k int) {
	for i := range m {
		for j := range n {
			var sum float64
			for p := range k {
				sum += a[i*k+p] * b[j*k+p]
			}
			c[i*n+j] = sum
		}
	}
}

func TestMatMulKLastSmall(t *testing.T) {
	// Test case: 2x3 * 2x3^T = 2x2
	// A = [[1, 2, 3], [4, 5, 6]] (2x3, K=3)
	// B = [[7, 8, 9], [10, 11, 12]] (2x3, N=2)
	// C[0,0] = 1*7 + 2*8 + 3*9 = 7 + 16 + 27 = 50
	// C[0,1] = 1*10 + 2*11 + 3*12 = 10 + 22 + 36 = 68
	// C[1,0] = 4*7 + 5*8 + 6*9 = 28 + 40 + 54 = 122
	// C[1,1] = 4*10 + 5*11 + 6*12 = 40 + 55 + 72 = 167
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{7, 8, 9, 10, 11, 12}
	c := make([]float32, 4)
	expected := make([]float32, 4)

	matmulKLastReference(a, b, expected, 2, 2, 3)
	MatMulKLast(a, b, c, 2, 2, 3)

	t.Logf("Expected: %v", expected)
	t.Logf("Got:      %v", c)

	for i := range c {
		if math.Abs(float64(c[i]-expected[i])) > 1e-5 {
			t.Errorf("c[%d] = %f, want %f", i, c[i], expected[i])
		}
	}
}

func TestMatMulKLastIdentity(t *testing.T) {
	// Test with identity-like pattern
	// A = random, B = I (padded), should give A's first N columns
	n := 8
	k := 8

	a := make([]float32, n*k)
	identity := make([]float32, n*k)
	c := make([]float32, n*n)
	expected := make([]float32, n*n)

	// Fill A with random values
	for i := range a {
		a[i] = rand.Float32()
	}

	// Create identity-ish matrix: B[j, j] = 1
	for j := range n {
		if j < k {
			identity[j*k+j] = 1
		}
	}

	matmulKLastReference(a, identity, expected, n, n, k)
	MatMulKLast(a, identity, c, n, n, k)

	for i := range c {
		if math.Abs(float64(c[i]-expected[i])) > 1e-5 {
			t.Errorf("c[%d] = %f, want %f", i, c[i], expected[i])
		}
	}
}

func TestMatMulKLast(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{16, 32, 64, 128}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, n*k)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			// Fill with random values
			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulKLastReference(a, b, expected, m, n, k)
			MatMulKLast(a, b, c, m, n, k)

			// Check results
			maxErr := float32(0)
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			// Allow some floating point tolerance
			tolerance := float32(1e-4) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %f exceeds tolerance %f", maxErr, tolerance)
			} else {
				t.Logf("size %dx%d: max error %e", size, size, maxErr)
			}
		})
	}
}

func TestMatMulKLastNonSquare(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	testCases := []struct {
		m, n, k int
	}{
		{16, 32, 64},  // M < N < K
		{64, 32, 16},  // M > N > K
		{32, 16, 64},  // Various non-square
		{128, 64, 32}, // Larger non-square
		{1, 128, 256}, // Single row (common for attention)
		{4, 256, 512}, // Small M, large N, K (common for MLP)
	}

	for _, tc := range testCases {
		name := sizeStr(tc.m) + "x" + sizeStr(tc.n) + "x" + sizeStr(tc.k)
		t.Run(name, func(t *testing.T) {
			a := make([]float32, tc.m*tc.k)
			b := make([]float32, tc.n*tc.k)
			c := make([]float32, tc.m*tc.n)
			expected := make([]float32, tc.m*tc.n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulKLastReference(a, b, expected, tc.m, tc.n, tc.k)
			MatMulKLast(a, b, c, tc.m, tc.n, tc.k)

			maxErr := float32(0)
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := float32(1e-4) * float32(tc.k)
			if maxErr > tolerance {
				t.Errorf("max error %f exceeds tolerance %f", maxErr, tolerance)
			}
		})
	}
}

func TestMatMulKLastFloat64(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{16, 32, 64, 128}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float64, m*k)
			b := make([]float64, n*k)
			c := make([]float64, m*n)
			expected := make([]float64, m*n)

			for i := range a {
				a[i] = float64(i%7) + 0.5
			}
			for i := range b {
				b[i] = float64(i%11) + 0.25
			}

			matmulKLastReference64(a, b, expected, m, n, k)
			MatMulKLastFloat64(a, b, c, m, n, k)

			var maxErr float64
			for i := range c {
				err := math.Abs(c[i] - expected[i])
				if err > maxErr {
					maxErr = err
				}
			}
			t.Logf("size %dx%d: max error %e", size, size, maxErr)

			if maxErr > 1e-9 {
				t.Errorf("max error %e exceeds threshold", maxErr)
			}
		})
	}
}

func TestMatMulKLastBlocked(t *testing.T) {
	sizes := []int{64, 128, 256}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, n*k)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			for i := range a {
				a[i] = rand.Float32()
			}
			for i := range b {
				b[i] = rand.Float32()
			}

			matmulKLastReference(a, b, expected, m, n, k)
			MatMulKLastBlocked(a, b, c, m, n, k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			t.Logf("size %dx%d: max error %e", size, size, maxErr)
			tolerance := float32(1e-3) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

func TestParallelMatMulKLast(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	pool := workerpool.New(0)
	defer pool.Close()

	// Test sizes that should trigger parallel path (>= 64^3 ops)
	sizes := []int{128, 256, 512}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, n*k)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulKLastReference(a, b, expected, m, n, k)
			ParallelMatMulKLast(pool, a, b, c, m, n, k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			t.Logf("size %dx%d: max error %e", size, size, maxErr)
			tolerance := float32(1e-3) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

func TestParallelMatMulKLastNonSquare(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	pool := workerpool.New(0)
	defer pool.Close()

	// Test shapes common in LLM inference where batchSize=1 (multi-cross patterns)
	testCases := []struct {
		name    string
		m, n, k int
	}{
		{"QKV", 512, 3072, 768},    // Multi-cross: bsi,oi->bso
		{"MLP_up", 512, 3072, 768}, // MLP up projection
		{"MLP_down", 512, 768, 3072},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a := make([]float32, tc.m*tc.k)
			b := make([]float32, tc.n*tc.k)
			c := make([]float32, tc.m*tc.n)
			expected := make([]float32, tc.m*tc.n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulKLastReference(a, b, expected, tc.m, tc.n, tc.k)
			ParallelMatMulKLast(pool, a, b, c, tc.m, tc.n, tc.k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := float32(1e-3) * float32(tc.k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

func BenchmarkMatMulKLast(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

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

		b.Run(sizeStr(size), func(b *testing.B) {
			b.SetBytes(int64((m*k + n*k + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMulKLast(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

func BenchmarkMatMulKLastVsStandard(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{128, 256, 512}

	for _, size := range sizes {
		m, n, k := size, size, size

		// For KLast: A [M,K], B [N,K]
		aKLast := make([]float32, m*k)
		bKLast := make([]float32, n*k)
		cKLast := make([]float32, m*n)

		// For Standard: A [M,K], B [K,N]
		aStd := make([]float32, m*k)
		bStd := make([]float32, k*n)
		cStd := make([]float32, m*n)

		for i := range aKLast {
			aKLast[i] = rand.Float32()
			aStd[i] = aKLast[i]
		}
		for i := range bKLast {
			bKLast[i] = rand.Float32()
		}
		// Transpose bKLast to get bStd
		for j := 0; j < n; j++ {
			for p := 0; p < k; p++ {
				bStd[p*n+j] = bKLast[j*k+p]
			}
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size)+"/KLast", func(b *testing.B) {
			b.SetBytes(int64((m*k + n*k + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMulKLast(aKLast, bKLast, cKLast, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Standard", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMul(aStd, bStd, cStd, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/TransposeThenStandard", func(b *testing.B) {
			bTransposed := make([]float32, k*n)
			b.SetBytes(int64((m*k + n*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// Transpose B first
				Transpose2D(bKLast, n, k, bTransposed)
				// Then standard matmul
				MatMul(aKLast, bTransposed, cKLast, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkParallelVsBlockedKLast compares parallel vs single-threaded blocked
func BenchmarkParallelVsBlockedKLast(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	pool := workerpool.New(0)
	defer pool.Close()

	// Test sizes that benefit from parallelization
	sizes := []int{256, 512, 1024}

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

		// Single-threaded blocked
		b.Run(sizeStr(size)+"/Blocked", func(b *testing.B) {
			b.SetBytes(int64((m*k + n*k + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulKLastBlocked(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		// Parallel
		b.Run(sizeStr(size)+"/Parallel", func(b *testing.B) {
			b.SetBytes(int64((m*k + n*k + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ParallelMatMulKLast(pool, a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		// Auto (should pick parallel for these sizes)
		b.Run(sizeStr(size)+"/Auto", func(b *testing.B) {
			b.SetBytes(int64((m*k + n*k + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulKLastAuto(pool, a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkMatMulKLastLLMShapes tests shapes common in LLM inference
func BenchmarkMatMulKLastLLMShapes(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	// Common LLM shapes:
	// - Attention QKV projection: [batch*seq, hidden] × [3*hidden, hidden]^T
	// - MLP up projection: [batch*seq, hidden] × [4*hidden, hidden]^T
	// - MLP down projection: [batch*seq, 4*hidden] × [hidden, 4*hidden]^T
	shapes := []struct {
		name    string
		m, n, k int
	}{
		{"QKV_small", 128, 2304, 768},    // GPT-2 small
		{"QKV_medium", 128, 3072, 1024},  // GPT-2 medium
		{"MLP_up_small", 128, 3072, 768}, // GPT-2 small MLP up
		{"MLP_down_small", 128, 768, 3072},
		{"Attention_single", 1, 768, 768}, // Single token
		{"Attention_batch", 32, 768, 768}, // Small batch
	}

	for _, shape := range shapes {
		m, n, k := shape.m, shape.n, shape.k

		a := make([]float32, m*k)
		bMat := make([]float32, n*k)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()*0.1 - 0.05
		}
		for i := range bMat {
			bMat[i] = rand.Float32()*0.1 - 0.05
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(shape.name, func(b *testing.B) {
			b.SetBytes(int64((m*k + n*k + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMulKLast(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}
