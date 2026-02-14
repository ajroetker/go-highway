// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build cgo && darwin

package matmul

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// matmulRefF64 computes C = A * B using naive triple loop for float64.
func matmulRefF64(a, b, c []float64, m, n, k int) {
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

// matmulKLastRefF64 computes C = A * B^T for float64.
func matmulKLastRefF64(a, b, c []float64, m, n, k int) {
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

func TestAccelerateSgemm(t *testing.T) {
	if accelerateSgemm == nil {
		t.Skip("Accelerate not available (CGO disabled)")
	}

	sizes := []struct {
		m, n, k int
	}{
		{2, 2, 3},
		{4, 4, 4},
		{7, 13, 5},     // non-aligned
		{64, 64, 64},   // medium
		{128, 256, 64}, // rectangular
		{1, 1024, 1024},
		{256, 256, 256},
	}

	for _, sz := range sizes {
		t.Run("", func(t *testing.T) {
			m, n, k := sz.m, sz.n, sz.k
			a := make([]float32, m*k)
			b := make([]float32, k*n)
			cRef := make([]float32, m*n)
			cAcc := make([]float32, m*n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulReference(a, b, cRef, m, n, k)
			accelerateSgemm(a, b, cAcc, m, n, k)

			for i := range cRef {
				diff := math.Abs(float64(cAcc[i] - cRef[i]))
				if diff > 1e-4 {
					t.Errorf("sgemm [%d,%d,%d] index %d: got %f, want %f (diff %e)",
						m, n, k, i, cAcc[i], cRef[i], diff)
					return
				}
			}
		})
	}
}

func TestAccelerateDgemm(t *testing.T) {
	if accelerateDgemm == nil {
		t.Skip("Accelerate not available (CGO disabled)")
	}

	sizes := []struct {
		m, n, k int
	}{
		{2, 2, 3},
		{7, 13, 5},
		{64, 64, 64},
		{128, 256, 64},
	}

	for _, sz := range sizes {
		t.Run("", func(t *testing.T) {
			m, n, k := sz.m, sz.n, sz.k
			a := make([]float64, m*k)
			b := make([]float64, k*n)
			cRef := make([]float64, m*n)
			cAcc := make([]float64, m*n)

			for i := range a {
				a[i] = rand.Float64()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float64()*2 - 1
			}

			matmulRefF64(a, b, cRef, m, n, k)
			accelerateDgemm(a, b, cAcc, m, n, k)

			for i := range cRef {
				diff := math.Abs(cAcc[i] - cRef[i])
				if diff > 1e-10 {
					t.Errorf("dgemm [%d,%d,%d] index %d: got %f, want %f (diff %e)",
						m, n, k, i, cAcc[i], cRef[i], diff)
					return
				}
			}
		})
	}
}

func TestAccelerateSgemmTranspose(t *testing.T) {
	if accelerateSgemmT == nil {
		t.Skip("Accelerate not available (CGO disabled)")
	}

	sizes := []struct {
		m, n, k int
	}{
		{2, 2, 3},
		{7, 13, 5},
		{64, 64, 64},
		{128, 256, 64},
		{1, 1024, 1024},
		{256, 256, 256},
	}

	for _, sz := range sizes {
		t.Run("", func(t *testing.T) {
			m, n, k := sz.m, sz.n, sz.k
			a := make([]float32, m*k)
			b := make([]float32, n*k) // K-last: B is [N,K]
			cRef := make([]float32, m*n)
			cAcc := make([]float32, m*n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulKLastReference(a, b, cRef, m, n, k)
			accelerateSgemmT(a, b, cAcc, m, n, k)

			for i := range cRef {
				diff := math.Abs(float64(cAcc[i] - cRef[i]))
				if diff > 1e-4 {
					t.Errorf("sgemmT [%d,%d,%d] index %d: got %f, want %f (diff %e)",
						m, n, k, i, cAcc[i], cRef[i], diff)
					return
				}
			}
		})
	}
}

func TestAccelerateDgemmTranspose(t *testing.T) {
	if accelerateDgemmT == nil {
		t.Skip("Accelerate not available (CGO disabled)")
	}

	sizes := []struct {
		m, n, k int
	}{
		{2, 2, 3},
		{7, 13, 5},
		{64, 64, 64},
		{128, 256, 64},
	}

	for _, sz := range sizes {
		t.Run("", func(t *testing.T) {
			m, n, k := sz.m, sz.n, sz.k
			a := make([]float64, m*k)
			b := make([]float64, n*k)
			cRef := make([]float64, m*n)
			cAcc := make([]float64, m*n)

			for i := range a {
				a[i] = rand.Float64()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float64()*2 - 1
			}

			matmulKLastRefF64(a, b, cRef, m, n, k)
			accelerateDgemmT(a, b, cAcc, m, n, k)

			for i := range cRef {
				diff := math.Abs(cAcc[i] - cRef[i])
				if diff > 1e-10 {
					t.Errorf("dgemmT [%d,%d,%d] index %d: got %f, want %f (diff %e)",
						m, n, k, i, cAcc[i], cRef[i], diff)
					return
				}
			}
		})
	}
}

// TestMatMulAutoDispatchesAccelerate verifies that MatMulAuto uses Accelerate
// by checking that results match cblas output for float32.
func TestMatMulAutoDispatchesAccelerate(t *testing.T) {
	if accelerateSgemm == nil {
		t.Skip("Accelerate not available (CGO disabled)")
	}

	pool := workerpool.New(runtime.GOMAXPROCS(0))
	defer pool.Close()

	m, n, k := 128, 128, 128
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	cAuto := make([]float32, m*n)
	cRef := make([]float32, m*n)

	for i := range a {
		a[i] = rand.Float32()*2 - 1
	}
	for i := range b {
		b[i] = rand.Float32()*2 - 1
	}

	matmulReference(a, b, cRef, m, n, k)
	MatMulAuto(pool, a, b, cAuto, m, n, k)

	for i := range cRef {
		diff := math.Abs(float64(cAuto[i] - cRef[i]))
		if diff > 1e-4 {
			t.Errorf("MatMulAuto index %d: got %f, want %f (diff %e)",
				i, cAuto[i], cRef[i], diff)
			return
		}
	}
}

// TestMatMulKLastAutoDispatchesAccelerate verifies that MatMulKLastAuto uses Accelerate.
func TestMatMulKLastAutoDispatchesAccelerate(t *testing.T) {
	if accelerateSgemmT == nil {
		t.Skip("Accelerate not available (CGO disabled)")
	}

	pool := workerpool.New(runtime.GOMAXPROCS(0))
	defer pool.Close()

	m, n, k := 128, 128, 128
	a := make([]float32, m*k)
	b := make([]float32, n*k)
	cAuto := make([]float32, m*n)
	cRef := make([]float32, m*n)

	for i := range a {
		a[i] = rand.Float32()*2 - 1
	}
	for i := range b {
		b[i] = rand.Float32()*2 - 1
	}

	matmulKLastReference(a, b, cRef, m, n, k)
	MatMulKLastAuto(pool, a, b, cAuto, m, n, k)

	for i := range cRef {
		diff := math.Abs(float64(cAuto[i] - cRef[i]))
		if diff > 1e-4 {
			t.Errorf("MatMulKLastAuto index %d: got %f, want %f (diff %e)",
				i, cAuto[i], cRef[i], diff)
			return
		}
	}
}

func BenchmarkAccelerateSgemm(b *testing.B) {
	if accelerateSgemm == nil {
		b.Skip("Accelerate not available (CGO disabled)")
	}

	pool := workerpool.New(runtime.GOMAXPROCS(0))
	defer pool.Close()

	for _, size := range []int{128, 256, 512, 1024} {
		m, n, k := size, size, size
		a := make([]float32, m*k)
		bm := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bm {
			bm[i] = rand.Float32()
		}

		b.Run("Accelerate/"+intToStr(size), func(b *testing.B) {
			for range b.N {
				accelerateSgemm(a, bm, c, m, n, k)
			}
		})

		b.Run("GoFMOPA/"+intToStr(size), func(b *testing.B) {
			// Temporarily disable accelerate to benchmark the Go path
			saved := accelerateSgemm
			accelerateSgemm = nil
			defer func() { accelerateSgemm = saved }()

			for range b.N {
				MatMulAuto(pool, a, bm, c, m, n, k)
			}
		})
	}
}

func BenchmarkAccelerateSgemmKLast(b *testing.B) {
	if accelerateSgemmT == nil {
		b.Skip("Accelerate not available (CGO disabled)")
	}

	pool := workerpool.New(runtime.GOMAXPROCS(0))
	defer pool.Close()

	for _, size := range []int{128, 256, 512, 1024} {
		m, n, k := size, size, size
		a := make([]float32, m*k)
		bm := make([]float32, n*k)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bm {
			bm[i] = rand.Float32()
		}

		b.Run("Accelerate/"+intToStr(size), func(b *testing.B) {
			for range b.N {
				accelerateSgemmT(a, bm, c, m, n, k)
			}
		})

		b.Run("GoFMOPA/"+intToStr(size), func(b *testing.B) {
			saved := accelerateSgemmT
			accelerateSgemmT = nil
			defer func() { accelerateSgemmT = saved }()

			for range b.N {
				MatMulKLastAuto(pool, a, bm, c, m, n, k)
			}
		})
	}
}

// intToStr converts an int to a string for benchmark naming without importing strconv.
func intToStr(n int) string {
	if n == 0 {
		return "0"
	}
	buf := make([]byte, 0, 10)
	for n > 0 {
		buf = append(buf, byte('0'+n%10))
		n /= 10
	}
	// reverse
	for i, j := 0, len(buf)-1; i < j; i, j = i+1, j-1 {
		buf[i], buf[j] = buf[j], buf[i]
	}
	return string(buf)
}
