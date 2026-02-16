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

// testRNGInt8x8PerAxis returns a seeded random number generator for reproducible tests.
func testRNGInt8x8PerAxis() *rand.Rand {
	return rand.New(rand.NewSource(101))
}

// referenceInt8x8MatMulPerAxis computes a scalar reference per-axis int8x8 matmul.
// output[m,n] = sum_k( (int32(a[m,k]) - int32(aZP[m])) * (int32(b[k,n]) - int32(bZP[n])) )
func referenceInt8x8MatMulPerAxis(a, b []uint8, aZP, bZP []uint8, M, K, N int) []int32 {
	output := make([]int32, M*N)

	for m := range M {
		azp := int32(aZP[m])
		for n := range N {
			bzp := int32(bZP[n])
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

// TestInt8x8MatMulPerAxisBasicCorrectness verifies the dispatched Int8x8MatMulPerAxis produces correct results.
func TestInt8x8MatMulPerAxisBasicCorrectness(t *testing.T) {
	rng := testRNGInt8x8PerAxis()

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

			aZP := make([]uint8, tc.M)
			bZP := make([]uint8, tc.N)
			for i := range aZP {
				aZP[i] = uint8(rng.Intn(256))
			}
			for i := range bZP {
				bZP[i] = uint8(rng.Intn(256))
			}

			output := make([]int32, tc.M*tc.N)
			Int8x8MatMulPerAxis(output, a, b, aZP, bZP, tc.M, tc.K, tc.N)

			ref := referenceInt8x8MatMulPerAxis(a, b, aZP, bZP, tc.M, tc.K, tc.N)

			for i := range output {
				if output[i] != ref[i] {
					t.Errorf("Mismatch at index %d: got=%d ref=%d", i, output[i], ref[i])
					return
				}
			}
		})
	}
}

// TestInt8x8MatMulPerAxisFallbackCorrectness verifies the scalar fallback matches the reference exactly.
func TestInt8x8MatMulPerAxisFallbackCorrectness(t *testing.T) {
	rng := testRNGInt8x8PerAxis()

	M, K, N := 16, 32, 48
	a := make([]uint8, M*K)
	b := make([]uint8, K*N)
	for i := range a {
		a[i] = uint8(rng.Intn(256))
	}
	for i := range b {
		b[i] = uint8(rng.Intn(256))
	}

	aZP := make([]uint8, M)
	bZP := make([]uint8, N)
	for i := range aZP {
		aZP[i] = uint8(rng.Intn(256))
	}
	for i := range bZP {
		bZP[i] = uint8(rng.Intn(256))
	}

	output := make([]int32, M*N)
	BaseInt8x8MatMulPerAxis_fallback(output, a, b, aZP, bZP, M, K, N)

	ref := referenceInt8x8MatMulPerAxis(a, b, aZP, bZP, M, K, N)

	for i := range output {
		if output[i] != ref[i] {
			t.Errorf("Mismatch at index %d: fallback=%d ref=%d", i, output[i], ref[i])
			return
		}
	}
}

// TestInt8x8MatMulPerAxisEdgeCases tests edge cases for per-axis Int8x8 matmul.
func TestInt8x8MatMulPerAxisEdgeCases(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		output := make([]int32, 0)
		Int8x8MatMulPerAxis(output, nil, nil, nil, nil, 0, 0, 0)
		// Should not panic
	})

	t.Run("single", func(t *testing.T) {
		a := []uint8{200}
		b := []uint8{100}
		aZP := []uint8{0}
		bZP := []uint8{0}
		output := make([]int32, 1)
		Int8x8MatMulPerAxis(output, a, b, aZP, bZP, 1, 1, 1)

		// (200 - 0) * (100 - 0) = 20000
		if output[0] != 20000 {
			t.Errorf("Expected 20000, got %d", output[0])
		}
	})

	t.Run("boundary_values_0_255", func(t *testing.T) {
		// M=1, K=2, N=1: a=[0,255], b=[0,255]
		a := []uint8{0, 255}
		b := []uint8{0, 255}
		aZP := []uint8{0}
		bZP := []uint8{0}
		output := make([]int32, 1)
		Int8x8MatMulPerAxis(output, a, b, aZP, bZP, 1, 2, 1)
		// output[0] = (0-0)*(0-0) + (255-0)*(255-0) = 65025
		if output[0] != 65025 {
			t.Errorf("Expected 65025, got %d", output[0])
		}
	})

	t.Run("boundary_values_with_zp", func(t *testing.T) {
		a := []uint8{0}
		b := []uint8{255}
		aZP := []uint8{128}
		bZP := []uint8{128}
		output := make([]int32, 1)
		Int8x8MatMulPerAxis(output, a, b, aZP, bZP, 1, 1, 1)

		// (0 - 128) * (255 - 128) = -128 * 127 = -16256
		if output[0] != -16256 {
			t.Errorf("Expected -16256, got %d", output[0])
		}
	})
}

// TestInt8x8MatMulPerAxisUniformZP cross-validates: uniform per-axis ZPs must match
// scalar referenceInt8x8MatMul exactly.
func TestInt8x8MatMulPerAxisUniformZP(t *testing.T) {
	rng := testRNGInt8x8PerAxis()

	M, K, N := 32, 64, 48
	a := make([]uint8, M*K)
	b := make([]uint8, K*N)
	for i := range a {
		a[i] = uint8(rng.Intn(256))
	}
	for i := range b {
		b[i] = uint8(rng.Intn(256))
	}

	// Uniform zero points: all rows/cols get the same value
	scalarAZP := uint8(128)
	scalarBZP := uint8(100)

	aZP := make([]uint8, M)
	bZP := make([]uint8, N)
	for i := range aZP {
		aZP[i] = scalarAZP
	}
	for i := range bZP {
		bZP[i] = scalarBZP
	}

	// Per-axis result
	perAxisOutput := make([]int32, M*N)
	Int8x8MatMulPerAxis(perAxisOutput, a, b, aZP, bZP, M, K, N)

	// Scalar reference
	scalarRef := referenceInt8x8MatMul(a, b, scalarAZP, scalarBZP, M, K, N)

	for i := range perAxisOutput {
		if perAxisOutput[i] != scalarRef[i] {
			t.Errorf("Mismatch at index %d: perAxis=%d scalarRef=%d", i, perAxisOutput[i], scalarRef[i])
			return
		}
	}
}

// TestInt8x8MatMulPerAxisParallel verifies parallel matches serial.
func TestInt8x8MatMulPerAxisParallel(t *testing.T) {
	rng := testRNGInt8x8PerAxis()

	M, K, N := 64, 128, 96
	a := make([]uint8, M*K)
	b := make([]uint8, K*N)
	for i := range a {
		a[i] = uint8(rng.Intn(256))
	}
	for i := range b {
		b[i] = uint8(rng.Intn(256))
	}

	aZP := make([]uint8, M)
	bZP := make([]uint8, N)
	for i := range aZP {
		aZP[i] = uint8(rng.Intn(256))
	}
	for i := range bZP {
		bZP[i] = uint8(rng.Intn(256))
	}

	// Serial
	serialOutput := make([]int32, M*N)
	Int8x8MatMulPerAxis(serialOutput, a, b, aZP, bZP, M, K, N)

	// Parallel
	pool := workerpool.New(4)
	defer pool.Close()

	parallelOutput := make([]int32, M*N)
	ParallelInt8x8MatMulPerAxis(pool, parallelOutput, a, b, aZP, bZP, M, K, N)

	for i := range serialOutput {
		if serialOutput[i] != parallelOutput[i] {
			t.Errorf("Mismatch at index %d: serial=%d parallel=%d", i, serialOutput[i], parallelOutput[i])
			return
		}
	}
}

// BenchmarkInt8x8MatMulPerAxis benchmarks per-axis Int8x8 matmul.
func BenchmarkInt8x8MatMulPerAxis(b *testing.B) {
	rng := testRNGInt8x8PerAxis()

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

		aZP := make([]uint8, sz.M)
		bZP := make([]uint8, sz.N)
		for i := range aZP {
			aZP[i] = uint8(rng.Intn(256))
		}
		for i := range bZP {
			bZP[i] = uint8(rng.Intn(256))
		}

		output := make([]int32, sz.M*sz.N)

		b.Run(sz.name, func(b *testing.B) {
			ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Int8x8MatMulPerAxis(output, a, bmat, aZP, bZP, sz.M, sz.K, sz.N)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GOPS")
		})
	}
}
