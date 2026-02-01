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
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestParallelPackedMatMulV2_Small(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	// Small matrix test: 4x4
	m, n, k := 4, 4, 4

	a := make([]float32, m*k)
	b := make([]float32, k*n)
	c := make([]float32, m*n)
	expected := make([]float32, m*n)

	// Initialize with known values
	for i := range a {
		a[i] = float32(i + 1)
	}
	for i := range b {
		b[i] = float32(i + 1)
	}

	// Compute expected result with naive implementation
	naiveMatMul(a, b, expected, m, n, k)

	// Compute with V2 parallel implementation
	ParallelPackedMatMulV2(pool, a, b, c, m, n, k)

	// Verify
	for i := range expected {
		if math.Abs(float64(c[i]-expected[i])) > 1e-4 {
			t.Errorf("c[%d] = %v, want %v", i, c[i], expected[i])
		}
	}
}

func TestParallelPackedMatMulV2_Medium(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	// Medium matrix to trigger parallel execution
	m, n, k := 128, 128, 128

	a := make([]float32, m*k)
	b := make([]float32, k*n)
	c := make([]float32, m*n)
	expected := make([]float32, m*n)

	// Initialize with random-ish values
	for i := range a {
		a[i] = float32(i%17) / 17.0
	}
	for i := range b {
		b[i] = float32(i%19) / 19.0
	}

	// Compute expected result
	naiveMatMul(a, b, expected, m, n, k)

	// Compute with V2
	ParallelPackedMatMulV2(pool, a, b, c, m, n, k)

	// Verify
	maxDiff := float32(0)
	for i := range expected {
		diff := float32(math.Abs(float64(c[i] - expected[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 1e-3 {
		t.Errorf("max diff = %v, want < 1e-3", maxDiff)
	}
}

func TestParallelPackedMatMulV2_Large(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	// Large matrix to really exercise parallel code
	m, n, k := 256, 256, 256

	a := make([]float32, m*k)
	b := make([]float32, k*n)
	c := make([]float32, m*n)
	expected := make([]float32, m*n)

	// Initialize
	for i := range a {
		a[i] = float32(i%23) / 23.0
	}
	for i := range b {
		b[i] = float32(i%29) / 29.0
	}

	// Compute expected result
	naiveMatMul(a, b, expected, m, n, k)

	// Compute with V2
	ParallelPackedMatMulV2(pool, a, b, c, m, n, k)

	// Verify
	maxDiff := float32(0)
	for i := range expected {
		diff := float32(math.Abs(float64(c[i] - expected[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 1e-2 {
		t.Errorf("max diff = %v, want < 1e-2", maxDiff)
	}
}

func TestParallelPackedMatMulV2_NonSquare(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	// Non-square matrix
	m, n, k := 128, 256, 64

	a := make([]float32, m*k)
	b := make([]float32, k*n)
	c := make([]float32, m*n)
	expected := make([]float32, m*n)

	for i := range a {
		a[i] = float32(i%13) / 13.0
	}
	for i := range b {
		b[i] = float32(i%17) / 17.0
	}

	naiveMatMul(a, b, expected, m, n, k)
	ParallelPackedMatMulV2(pool, a, b, c, m, n, k)

	maxDiff := float32(0)
	for i := range expected {
		diff := float32(math.Abs(float64(c[i] - expected[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 1e-3 {
		t.Errorf("max diff = %v, want < 1e-3", maxDiff)
	}
}

func TestBatchParallelPackedMatMulV2(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batchSize := 4
	m, n, k := 64, 64, 64

	a := make([]float32, batchSize*m*k)
	b := make([]float32, batchSize*k*n)
	c := make([]float32, batchSize*m*n)
	expected := make([]float32, batchSize*m*n)

	// Initialize
	for i := range a {
		a[i] = float32(i%31) / 31.0
	}
	for i := range b {
		b[i] = float32(i%37) / 37.0
	}

	// Compute expected for each batch
	lhsStride := m * k
	rhsStride := k * n
	outStride := m * n
	for batch := 0; batch < batchSize; batch++ {
		naiveMatMul(
			a[batch*lhsStride:(batch+1)*lhsStride],
			b[batch*rhsStride:(batch+1)*rhsStride],
			expected[batch*outStride:(batch+1)*outStride],
			m, n, k,
		)
	}

	// Compute with batched V2
	BatchParallelPackedMatMulV2(pool, a, b, c, batchSize, m, n, k)

	// Verify
	maxDiff := float32(0)
	for i := range expected {
		diff := float32(math.Abs(float64(c[i] - expected[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 1e-3 {
		t.Errorf("max diff = %v, want < 1e-3", maxDiff)
	}
}

// naiveMatMul computes C = A * B using triple loop
func naiveMatMul(a, b, c []float32, m, n, k int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// Benchmarks comparing V1 and V2
func BenchmarkParallelPackedMatMulV1vsV2(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	sizes := []int{256, 512, 1024, 2048}

	for _, size := range sizes {
		m, n, k := size, size, size
		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		// Initialize
		for i := range a {
			a[i] = float32(i%100) / 100.0
		}
		for i := range bMat {
			bMat[i] = float32(i%100) / 100.0
		}

		b.Run("V1/"+sizeStrV2(size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ParallelPackedMatMul(a, bMat, c, m, n, k)
			}
		})

		b.Run("V2/"+sizeStrV2(size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ParallelPackedMatMulV2(pool, a, bMat, c, m, n, k)
			}
		})
	}
}

func sizeStrV2(size int) string {
	switch size {
	case 256:
		return "256x256"
	case 512:
		return "512x512"
	case 1024:
		return "1024x1024"
	case 2048:
		return "2048x2048"
	default:
		return "unknown"
	}
}

func BenchmarkBatchParallelPackedMatMulV2(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	batchSize := 8
	m, n, k := 128, 128, 128

	a := make([]float32, batchSize*m*k)
	bMat := make([]float32, batchSize*k*n)
	c := make([]float32, batchSize*m*n)

	for i := range a {
		a[i] = float32(i%100) / 100.0
	}
	for i := range bMat {
		bMat[i] = float32(i%100) / 100.0
	}

	b.Run("BatchV2", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			BatchParallelPackedMatMulV2(pool, a, bMat, c, batchSize, m, n, k)
		}
	})
}
