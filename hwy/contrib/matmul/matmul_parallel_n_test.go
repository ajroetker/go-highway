// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"math"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// TestParallelMatMulFineGrained tests the fine-grained parallel matmul for small M.
func TestParallelMatMulFineGrained(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	testCases := []struct {
		name    string
		m, n, k int
	}{
		{"11x1024x1024", 11, 1024, 1024},
		{"1x512x512", 1, 512, 512},
		{"4x256x512", 4, 256, 512},
		{"8x128x256", 8, 128, 256},
		{"15x1024x512", 15, 1024, 512},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m, n, k := tc.m, tc.n, tc.k

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			cParallel := make([]float32, m*n)
			cRef := make([]float32, m*n)

			for i := range a {
				a[i] = float32(i%7 - 3)
			}
			for i := range b {
				b[i] = float32(i%5 - 2)
			}

			matmulScalar(a, b, cRef, m, n, k)
			ParallelMatMulFineGrained(pool, a, b, cParallel, m, n, k)

			var maxErr float32
			for i := range cRef {
				err := float32(math.Abs(float64(cParallel[i] - cRef[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			const tolerance = 1e-4
			if maxErr > tolerance {
				t.Errorf("max error %v exceeds tolerance %v", maxErr, tolerance)
			}
		})
	}
}

// TestParallelMatMulKLastFineGrained tests the fine-grained parallel K-last matmul.
func TestParallelMatMulKLastFineGrained(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	testCases := []struct {
		name    string
		m, n, k int
	}{
		{"11x1024x1024", 11, 1024, 1024},
		{"1x512x512", 1, 512, 512},
		{"4x256x512", 4, 256, 512},
		{"8x128x256", 8, 128, 256},
		{"15x1024x512", 15, 1024, 512},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m, n, k := tc.m, tc.n, tc.k

			a := make([]float32, m*k)
			b := make([]float32, n*k) // B is NxK for K-last
			cParallel := make([]float32, m*n)
			cRef := make([]float32, m*n)

			for i := range a {
				a[i] = float32(i%7 - 3)
			}
			for i := range b {
				b[i] = float32(i%5 - 2)
			}

			// Reference: dot products
			for i := range m {
				for j := range n {
					var sum float32
					for p := range k {
						sum += a[i*k+p] * b[j*k+p]
					}
					cRef[i*n+j] = sum
				}
			}

			ParallelMatMulKLastFineGrained(pool, a, b, cParallel, m, n, k)

			var maxErr float32
			for i := range cRef {
				err := float32(math.Abs(float64(cParallel[i] - cRef[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			const tolerance = 1e-4
			if maxErr > tolerance {
				t.Errorf("max error %v exceeds tolerance %v", maxErr, tolerance)
			}
		})
	}
}

// BenchmarkSmallMParallel benchmarks matmul approaches for small M with large N*K.
func BenchmarkSmallMParallel(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	m, n, k := 11, 1024, 1024

	a := make([]float32, m*k)
	bmat := make([]float32, k*n)
	c := make([]float32, m*n)

	for i := range a {
		a[i] = float32(i%7 - 3)
	}
	for i := range bmat {
		bmat[i] = float32(i%5 - 2)
	}

	b.Run("MatMul_streaming", func(b *testing.B) {
		for range b.N {
			MatMul(a, bmat, c, m, n, k)
		}
	})

	b.Run("BlockedMatMul", func(b *testing.B) {
		for range b.N {
			BlockedMatMul(a, bmat, c, m, n, k)
		}
	})

	b.Run("ParallelMatMul_64strip", func(b *testing.B) {
		for range b.N {
			ParallelMatMul(pool, a, bmat, c, m, n, k)
		}
	})

	b.Run("ParallelMatMulFineGrained", func(b *testing.B) {
		for range b.N {
			ParallelMatMulFineGrained(pool, a, bmat, c, m, n, k)
		}
	})

	b.Run("MatMulAuto", func(b *testing.B) {
		for range b.N {
			MatMulAuto(pool, a, bmat, c, m, n, k)
		}
	})
}

// TestMatMulAutoSmallM verifies that MatMulAuto uses fine-grained parallelism for small M.
func TestMatMulAutoSmallM(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	m, n, k := 11, 1024, 1024

	a := make([]float32, m*k)
	b := make([]float32, k*n)
	cAuto := make([]float32, m*n)
	cRef := make([]float32, m*n)

	for i := range a {
		a[i] = float32(i%7 - 3)
	}
	for i := range b {
		b[i] = float32(i%5 - 2)
	}

	matmulScalar(a, b, cRef, m, n, k)
	MatMulAuto(pool, a, b, cAuto, m, n, k)

	var maxErr float32
	for i := range cRef {
		err := float32(math.Abs(float64(cAuto[i] - cRef[i])))
		if err > maxErr {
			maxErr = err
		}
	}

	const tolerance = 1e-4
	if maxErr > tolerance {
		t.Errorf("max error %v exceeds tolerance %v", maxErr, tolerance)
	}
}

// TestParallelMatMulPool tests the pool-based parallel matmul.
func TestParallelMatMulPool(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	testCases := []struct {
		name    string
		m, n, k int
	}{
		{"11x1024x1024", 11, 1024, 1024},
		{"64x512x512", 64, 512, 512},
		{"128x256x512", 128, 256, 512},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m, n, k := tc.m, tc.n, tc.k

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			cPool := make([]float32, m*n)
			cRef := make([]float32, m*n)

			for i := range a {
				a[i] = float32(i%7 - 3)
			}
			for i := range b {
				b[i] = float32(i%5 - 2)
			}

			matmulScalar(a, b, cRef, m, n, k)
			ParallelMatMul(pool, a, b, cPool, m, n, k)

			var maxErr float32
			for i := range cRef {
				err := float32(math.Abs(float64(cPool[i] - cRef[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			const tolerance = 1e-4
			if maxErr > tolerance {
				t.Errorf("max error %v exceeds tolerance %v", maxErr, tolerance)
			}
		})
	}
}

// BenchmarkPoolReuse simulates transformer inference with 50 matmul ops per "forward pass".
func BenchmarkPoolReuse(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	m, n, k := 11, 1024, 1024

	a := make([]float32, m*k)
	bmat := make([]float32, k*n)
	c := make([]float32, m*n)

	for i := range a {
		a[i] = float32(i%7 - 3)
	}
	for i := range bmat {
		bmat[i] = float32(i%5 - 2)
	}

	const opsPerForwardPass = 50

	b.Run("FineGrained_50ops", func(b *testing.B) {
		for range b.N {
			for range opsPerForwardPass {
				ParallelMatMulFineGrained(pool, a, bmat, c, m, n, k)
			}
		}
	})
}

// BenchmarkPoolKLast benchmarks K-last matmul with pool.
func BenchmarkPoolKLast(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	m, n, k := 11, 1024, 1024

	a := make([]float32, m*k)
	bmat := make([]float32, n*k) // NxK for K-last
	c := make([]float32, m*n)

	for i := range a {
		a[i] = float32(i%7 - 3)
	}
	for i := range bmat {
		bmat[i] = float32(i%5 - 2)
	}

	b.Run("KLast_FineGrained", func(b *testing.B) {
		for range b.N {
			ParallelMatMulKLastFineGrained(pool, a, bmat, c, m, n, k)
		}
	})
}
