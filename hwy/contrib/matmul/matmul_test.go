package matmul

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// matmulReference computes C = A * B using naive triple loop.
// Used as reference for correctness testing.
func matmulReference(a, b, c []float32, m, n, k int) {
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

func TestMatMulSmall(t *testing.T) {
	// 2x3 * 3x2 = 2x2
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{7, 8, 9, 10, 11, 12}
	c := make([]float32, 4)
	expected := make([]float32, 4)

	matmulReference(a, b, expected, 2, 2, 3)
	MatMul(a, b, c, 2, 2, 3)

	for i := range c {
		if math.Abs(float64(c[i]-expected[i])) > 1e-5 {
			t.Errorf("c[%d] = %f, want %f", i, c[i], expected[i])
		}
	}
}

func TestMatMulIdentity(t *testing.T) {
	// Identity matrix multiplication
	n := 4
	a := make([]float32, n*n)
	identity := make([]float32, n*n)
	c := make([]float32, n*n)

	// Fill A with random values
	for i := range a {
		a[i] = rand.Float32()
	}

	// Create identity matrix
	for i := range n {
		identity[i*n+i] = 1
	}

	MatMul(a, identity, c, n, n, n)

	// C should equal A
	for i := range c {
		if math.Abs(float64(c[i]-a[i])) > 1e-5 {
			t.Errorf("c[%d] = %f, want %f", i, c[i], a[i])
		}
	}
}

func TestMatMul256(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	size := 256
	m, n, k := size, size, size

	a := make([]float32, m*k)
	b := make([]float32, k*n)
	c := make([]float32, m*n)
	expected := make([]float32, m*n)

	// Simple test: all 1s, result should be K in each cell
	for i := range a {
		a[i] = 1.0
	}
	for i := range b {
		b[i] = 1.0
	}

	matmulReference(a, b, expected, m, n, k)
	MatMul(a, b, c, m, n, k)

	// Check first few elements
	for i := range 10 {
		t.Logf("c[%d] = %f, expected = %f", i, c[i], expected[i])
	}

	// Check all elements
	maxErr := float32(0)
	for i := range c {
		err := float32(math.Abs(float64(c[i] - expected[i])))
		if err > maxErr {
			maxErr = err
		}
	}

	if maxErr > 1e-3 {
		t.Errorf("max error %f exceeds tolerance", maxErr)
	}
}

func TestMatMulLarge(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{16, 32, 64, 128}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			// Fill with random values
			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulReference(a, b, expected, m, n, k)
			MatMul(a, b, c, m, n, k)

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

func sizeStr(n int) string {
	return string(rune('0'+n/100)) + string(rune('0'+(n/10)%10)) + string(rune('0'+n%10))
}

func BenchmarkMatMul(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		// Fill with random values
		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9 // 2 ops per multiply-add

		b.Run(sizeStr(size), func(b *testing.B) {
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
	}
}

func BenchmarkMatMulScalar(b *testing.B) {
	size := 256
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

	for b.Loop() {
		matmulScalar(a, bMat, c, m, n, k)
	}
}

// matmulReference64 computes C = A * B for float64
func matmulReference64(a, b, c []float64, m, n, k int) {
	for i := range m {
		for j := range n {
			var sum float64
			for p := range k {
				sum += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

func TestMatMulFloat64(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{16, 32, 64, 128}
	for _, size := range sizes {
		m, n, k := size, size, size
		t.Run(sizeStr(size), func(t *testing.T) {
			a := make([]float64, m*k)
			bMat := make([]float64, k*n)
			c := make([]float64, m*n)
			expected := make([]float64, m*n)

			// Fill with test values
			for i := range a {
				a[i] = float64(i%7) + 0.5
			}
			for i := range bMat {
				bMat[i] = float64(i%11) + 0.25
			}

			matmulReference64(a, bMat, expected, m, n, k)
			MatMulFloat64(a, bMat, c, m, n, k)

			var maxErr float64
			for i := range c {
				err := math.Abs(c[i] - expected[i])
				if err > maxErr {
					maxErr = err
				}
			}
			t.Logf("size %dx%d: max error %e", size, size, maxErr)

			// Allow small floating point error
			if maxErr > 1e-9 {
				t.Errorf("max error %e exceeds threshold", maxErr)
			}
		})
	}
}

func BenchmarkMatMulFloat64(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{64, 128, 256}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float64, m*k)
		bMat := make([]float64, k*n)
		c := make([]float64, m*n)

		for i := range a {
			a[i] = rand.Float64()
		}
		for i := range bMat {
			bMat[i] = rand.Float64()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size), func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 8))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMulFloat64(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// TestMatMulDispatch verifies the generic MatMul dispatches correctly for both types.
func TestMatMulDispatch(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		size := 64
		m, n, k := size, size, size

		a := make([]float32, m*k)
		b := make([]float32, k*n)
		c := make([]float32, m*n)
		expected := make([]float32, m*n)

		for i := range a {
			a[i] = 1.0
		}
		for i := range b {
			b[i] = 1.0
		}

		matmulReference(a, b, expected, m, n, k)
		MatMul(a, b, c, m, n, k)

		for i := range c {
			if math.Abs(float64(c[i]-expected[i])) > 1e-5 {
				t.Errorf("c[%d] = %f, want %f", i, c[i], expected[i])
				return
			}
		}
	})

	t.Run("float64", func(t *testing.T) {
		size := 64
		m, n, k := size, size, size

		a := make([]float64, m*k)
		b := make([]float64, k*n)
		c := make([]float64, m*n)
		expected := make([]float64, m*n)

		for i := range a {
			a[i] = 1.0
		}
		for i := range b {
			b[i] = 1.0
		}

		matmulReference64(a, b, expected, m, n, k)
		MatMul(a, b, c, m, n, k)

		for i := range c {
			if math.Abs(c[i]-expected[i]) > 1e-9 {
				t.Errorf("c[%d] = %f, want %f", i, c[i], expected[i])
				return
			}
		}
	})
}

// BenchmarkBlockedMatMul benchmarks the cache-tiled blocked matmul.
func BenchmarkBlockedMatMul(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{64, 128, 256, 512}

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
				BlockedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkStreamingVsBlocked compares streaming and blocked matmul side-by-side.
func BenchmarkStreamingVsBlocked(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{32, 64, 128, 256, 512, 1024}

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

		b.Run(sizeStr(size)+"/Auto", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMulAuto(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// TestBlockedMatMul verifies the blocked matmul produces correct results.
func TestBlockedMatMul(t *testing.T) {
	sizes := []int{16, 32, 48, 64, 96, 128}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			for i := range a {
				a[i] = rand.Float32()
			}
			for i := range b {
				b[i] = rand.Float32()
			}

			matmulReference(a, b, expected, m, n, k)
			BlockedMatMul(a, b, c, m, n, k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			t.Logf("size %dx%d: max error %e", size, size, maxErr)
			if maxErr > 1e-4 {
				t.Errorf("max error %e exceeds threshold 1e-4", maxErr)
			}
		})
	}
}

// TestParallelMatMul verifies the parallel matmul produces correct results.
func TestParallelMatMul(t *testing.T) {
	sizes := []int{128, 256, 512}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			for i := range a {
				a[i] = rand.Float32()
			}
			for i := range b {
				b[i] = rand.Float32()
			}

			matmulReference(a, b, expected, m, n, k)
			ParallelMatMul(a, b, c, m, n, k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			t.Logf("size %dx%d: max error %e", size, size, maxErr)
			tolerance := float32(1e-4) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// BenchmarkParallelMatMul benchmarks the parallel matmul.
func BenchmarkParallelMatMul(b *testing.B) {
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
				ParallelMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkParallelVsBlocked compares parallel and blocked (single-threaded) matmul.
func BenchmarkParallelVsBlocked(b *testing.B) {
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

		b.Run(sizeStr(size)+"/Parallel", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ParallelMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}
