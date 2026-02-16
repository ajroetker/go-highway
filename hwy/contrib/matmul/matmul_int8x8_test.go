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
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// testRNGInt8x8 returns a seeded random number generator for reproducible tests.
func testRNGInt8x8() *rand.Rand {
	return rand.New(rand.NewSource(99))
}

// referenceInt8x8MatMul computes a scalar reference int8x8 matmul.
// output[m,n] = sum_k( (int32(a[m,k]) - int32(aZP)) * (int32(b[k,n]) - int32(bZP)) )
func referenceInt8x8MatMul(a, b []uint8, aZP, bZP uint8, M, K, N int) []int32 {
	output := make([]int32, M*N)
	azp := int32(aZP)
	bzp := int32(bZP)

	for m := range M {
		for n := range N {
			var sum int32
			for k := range K {
				aVal := int32(a[m*K+k]) - azp
				bVal := int32(b[k*N+n]) - bzp
				sum += aVal * bVal
			}
			output[m*N+n] = sum
		}
	}
	return output
}

// TestInt8x8MatMulBasicCorrectness verifies the dispatched Int8x8MatMul produces correct results.
func TestInt8x8MatMulBasicCorrectness(t *testing.T) {
	rng := testRNGInt8x8()

	testCases := []struct {
		name    string
		M, K, N int
	}{
		{"small_16x32x48", 16, 32, 48},
		{"medium_32x64x128", 32, 64, 128},
		{"unaligned_17x33x49", 17, 33, 49},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a := make([]uint8, tc.M*tc.K)
			b := make([]uint8, tc.K*tc.N)
			for i := range a {
				a[i] = uint8(rng.Intn(256))
			}
			for i := range b {
				b[i] = uint8(rng.Intn(256))
			}

			aZP := uint8(rng.Intn(256))
			bZP := uint8(rng.Intn(256))

			output := make([]int32, tc.M*tc.N)
			Int8x8MatMul(output, a, b, aZP, bZP, tc.M, tc.K, tc.N)

			ref := referenceInt8x8MatMul(a, b, aZP, bZP, tc.M, tc.K, tc.N)

			for i := range output {
				if output[i] != ref[i] {
					t.Errorf("Mismatch at index %d: got=%d ref=%d", i, output[i], ref[i])
					return
				}
			}
		})
	}
}

// TestInt8x8MatMulFallbackCorrectness verifies the scalar fallback matches the reference exactly.
func TestInt8x8MatMulFallbackCorrectness(t *testing.T) {
	rng := testRNGInt8x8()

	M, K, N := 16, 32, 48
	a := make([]uint8, M*K)
	b := make([]uint8, K*N)
	for i := range a {
		a[i] = uint8(rng.Intn(256))
	}
	for i := range b {
		b[i] = uint8(rng.Intn(256))
	}
	aZP := uint8(128)
	bZP := uint8(128)

	output := make([]int32, M*N)
	BaseInt8x8MatMul_fallback(output, a, b, aZP, bZP, M, K, N)

	ref := referenceInt8x8MatMul(a, b, aZP, bZP, M, K, N)

	for i := range output {
		if output[i] != ref[i] {
			t.Errorf("Mismatch at index %d: fallback=%d ref=%d", i, output[i], ref[i])
			return
		}
	}
}

// TestInt8x8MatMulEdgeCases tests edge cases for Int8x8 matmul.
func TestInt8x8MatMulEdgeCases(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		output := make([]int32, 0)
		Int8x8MatMul(output, nil, nil, 0, 0, 0, 0, 0)
		// Should not panic
	})

	t.Run("single", func(t *testing.T) {
		a := []uint8{200}
		b := []uint8{100}
		output := make([]int32, 1)
		Int8x8MatMul(output, a, b, 0, 0, 1, 1, 1)

		// (200 - 0) * (100 - 0) = 20000
		if output[0] != 20000 {
			t.Errorf("Expected 20000, got %d", output[0])
		}
	})

	t.Run("boundary_values_0_255", func(t *testing.T) {
		// M=1, K=2, N=1: a=[0,255], b=[0,255]
		a := []uint8{0, 255}
		b := []uint8{0, 255}
		output := make([]int32, 1)
		Int8x8MatMul(output, a, b, 0, 0, 1, 2, 1)
		// output[0] = (0-0)*(0-0) + (255-0)*(255-0) = 65025
		if output[0] != 65025 {
			t.Errorf("Expected 65025, got %d", output[0])
		}
	})

	t.Run("boundary_values_with_zp", func(t *testing.T) {
		a := []uint8{0}
		b := []uint8{255}
		output := make([]int32, 1)
		Int8x8MatMul(output, a, b, 128, 128, 1, 1, 1)

		// (0 - 128) * (255 - 128) = -128 * 127 = -16256
		if output[0] != -16256 {
			t.Errorf("Expected -16256, got %d", output[0])
		}
	})
}

// TestInt8x8MatMulZeroPoint tests with symmetric quantization center (128).
func TestInt8x8MatMulZeroPoint(t *testing.T) {
	rng := testRNGInt8x8()

	M, K, N := 32, 64, 48
	a := make([]uint8, M*K)
	b := make([]uint8, K*N)
	for i := range a {
		a[i] = uint8(rng.Intn(256))
	}
	for i := range b {
		b[i] = uint8(rng.Intn(256))
	}

	// Common symmetric quantization center
	aZP := uint8(128)
	bZP := uint8(128)

	output := make([]int32, M*N)
	Int8x8MatMul(output, a, b, aZP, bZP, M, K, N)

	ref := referenceInt8x8MatMul(a, b, aZP, bZP, M, K, N)

	for i := range output {
		if output[i] != ref[i] {
			t.Errorf("Mismatch at index %d: got=%d ref=%d", i, output[i], ref[i])
			return
		}
	}
}

// TestInt8x8MatMulParallel verifies parallel matches serial.
func TestInt8x8MatMulParallel(t *testing.T) {
	rng := testRNGInt8x8()

	M, K, N := 64, 128, 96
	a := make([]uint8, M*K)
	b := make([]uint8, K*N)
	for i := range a {
		a[i] = uint8(rng.Intn(256))
	}
	for i := range b {
		b[i] = uint8(rng.Intn(256))
	}
	aZP := uint8(100)
	bZP := uint8(150)

	// Serial
	serialOutput := make([]int32, M*N)
	Int8x8MatMul(serialOutput, a, b, aZP, bZP, M, K, N)

	// Parallel
	pool := workerpool.New(4)
	defer pool.Close()

	parallelOutput := make([]int32, M*N)
	ParallelInt8x8MatMul(pool, parallelOutput, a, b, aZP, bZP, M, K, N)

	for i := range serialOutput {
		if serialOutput[i] != parallelOutput[i] {
			t.Errorf("Mismatch at index %d: serial=%d parallel=%d", i, serialOutput[i], parallelOutput[i])
			return
		}
	}
}

// BenchmarkInt8x8MatMul benchmarks Int8x8 matmul.
func BenchmarkInt8x8MatMul(b *testing.B) {
	rng := testRNGInt8x8()

	sizes := []struct {
		name    string
		M, K, N int
	}{
		{"32x64x128", 32, 64, 128},
		{"64x128x256", 64, 128, 256},
		{"64x256x512", 64, 256, 512},
	}

	for _, sz := range sizes {
		a := make([]uint8, sz.M*sz.K)
		bmat := make([]uint8, sz.K*sz.N)
		for i := range a {
			a[i] = uint8(rng.Intn(256))
		}
		for i := range bmat {
			bmat[i] = uint8(rng.Intn(256))
		}

		output := make([]int32, sz.M*sz.N)

		b.Run(sz.name, func(b *testing.B) {
			ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Int8x8MatMul(output, a, bmat, 128, 128, sz.M, sz.K, sz.N)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GOPS")
		})
	}
}
