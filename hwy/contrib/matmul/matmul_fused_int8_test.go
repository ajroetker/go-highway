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
)

// testRNGInt8 returns a seeded random number generator for reproducible tests.
func testRNGInt8() *rand.Rand {
	return rand.New(rand.NewSource(42))
}

// referenceInt8MatMul computes a reference Int8 fused matmul using scalar operations.
func referenceInt8MatMul(input []float32, weights []int8, scales []float32, M, K, N, groupSize int) []float32 {
	output := make([]float32, M*N)
	numGroups := (N + groupSize - 1) / groupSize

	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			groupIdx := n / groupSize
			sum := float32(0)
			for k := 0; k < K; k++ {
				weightIdx := k*N + n
				val := float32(weights[weightIdx])
				scale := scales[k*numGroups+groupIdx]
				weight := val * scale
				sum += input[m*K+k] * weight
			}
			output[m*N+n] = sum
		}
	}
	return output
}

// TestFusedInt8MatMulBasicCorrectness verifies the base Int8 fused matmul produces correct results.
func TestFusedInt8MatMulBasicCorrectness(t *testing.T) {
	rng := testRNGInt8()

	testCases := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		{"small_16x32x48", 16, 32, 48, 16},
		{"medium_32x64x128", 32, 64, 128, 32},
		{"unaligned_17x33x49", 17, 33, 49, 16},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			input := make([]float32, tc.M*tc.K)
			for i := range input {
				input[i] = rng.Float32()*2 - 1
			}

			weights := make([]int8, tc.K*tc.N)
			for i := range weights {
				weights[i] = int8(rng.Intn(256) - 128)
			}

			numGroups := (tc.N + tc.groupSize - 1) / tc.groupSize
			scales := make([]float32, tc.K*numGroups)
			for i := range scales {
				scales[i] = rng.Float32()*0.1 + 0.01
			}

			// Run via dispatch
			fusedOutput := make([]float32, tc.M*tc.N)
			FusedInt8MatMul(input, weights, scales, fusedOutput, tc.M, tc.K, tc.N, tc.groupSize)

			// Run reference
			refOutput := referenceInt8MatMul(input, weights, scales, tc.M, tc.K, tc.N, tc.groupSize)

			// Compare
			maxDiff := float32(0)
			for i := range fusedOutput {
				diff := float32(math.Abs(float64(fusedOutput[i] - refOutput[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			tolerance := float32(1e-4)
			if maxDiff > tolerance {
				t.Errorf("Max difference: %v (tolerance: %v)", maxDiff, tolerance)
			}
		})
	}
}

// TestFusedInt8MatMulFallbackCorrectness verifies the scalar fallback produces correct results.
func TestFusedInt8MatMulFallbackCorrectness(t *testing.T) {
	rng := testRNGInt8()

	M, K, N := 16, 32, 48
	groupSize := 16

	input := make([]float32, M*K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	weights := make([]int8, K*N)
	for i := range weights {
		weights[i] = int8(rng.Intn(256) - 128)
	}

	numGroups := (N + groupSize - 1) / groupSize
	scales := make([]float32, K*numGroups)
	for i := range scales {
		scales[i] = rng.Float32()*0.1 + 0.01
	}

	// Run fallback
	fallbackOutput := make([]float32, M*N)
	BaseFusedInt8MatMul_fallback(input, weights, scales, fallbackOutput, M, K, N, groupSize)

	// Run reference
	refOutput := referenceInt8MatMul(input, weights, scales, M, K, N, groupSize)

	// Should be identical
	for i := range fallbackOutput {
		if fallbackOutput[i] != refOutput[i] {
			t.Errorf("Mismatch at index %d: fallback=%v ref=%v", i, fallbackOutput[i], refOutput[i])
			return
		}
	}
}

// TestFusedInt8MatMulEdgeCases tests edge cases for Int8 fused matmul.
func TestFusedInt8MatMulEdgeCases(t *testing.T) {
	// Empty input
	t.Run("empty", func(t *testing.T) {
		output := make([]float32, 0)
		FusedInt8MatMul(nil, nil, nil, output, 0, 0, 0, 1)
		// Should not panic
	})

	// Single element
	t.Run("single", func(t *testing.T) {
		input := []float32{2.0}
		weights := []int8{3}
		scales := []float32{0.5}
		output := make([]float32, 1)
		FusedInt8MatMul(input, weights, scales, output, 1, 1, 1, 1)

		expected := float32(2.0 * 3.0 * 0.5)
		if math.Abs(float64(output[0]-expected)) > 1e-6 {
			t.Errorf("Expected %v, got %v", expected, output[0])
		}
	})

	// Int8 boundary values
	t.Run("boundary_values", func(t *testing.T) {
		input := []float32{1.0, 1.0}
		weights := []int8{127, -128} // Max and min Int8 values
		scales := []float32{1.0, 1.0}
		output := make([]float32, 2)
		FusedInt8MatMul(input, weights, scales, output, 1, 1, 2, 2)

		if output[0] != 127.0 {
			t.Errorf("Expected 127, got %v", output[0])
		}
		if output[1] != -128.0 {
			t.Errorf("Expected -128, got %v", output[1])
		}
	})
}

// TestFusedInt8MatMulGroupBoundary verifies correctness at group boundaries.
func TestFusedInt8MatMulGroupBoundary(t *testing.T) {
	rng := testRNGInt8()

	// Test where N is not a multiple of groupSize
	testCases := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		{"group_boundary_32x64x100_gs32", 32, 64, 100, 32},
		{"group_boundary_32x64x50_gs16", 32, 64, 50, 16},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			input := make([]float32, tc.M*tc.K)
			for i := range input {
				input[i] = rng.Float32()*2 - 1
			}

			weights := make([]int8, tc.K*tc.N)
			for i := range weights {
				weights[i] = int8(rng.Intn(256) - 128)
			}

			numGroups := (tc.N + tc.groupSize - 1) / tc.groupSize
			scales := make([]float32, tc.K*numGroups)
			for i := range scales {
				scales[i] = rng.Float32()*0.1 + 0.01
			}

			fusedOutput := make([]float32, tc.M*tc.N)
			FusedInt8MatMul(input, weights, scales, fusedOutput, tc.M, tc.K, tc.N, tc.groupSize)

			refOutput := referenceInt8MatMul(input, weights, scales, tc.M, tc.K, tc.N, tc.groupSize)

			maxDiff := float32(0)
			for i := range fusedOutput {
				diff := float32(math.Abs(float64(fusedOutput[i] - refOutput[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			tolerance := float32(1e-4)
			if maxDiff > tolerance {
				t.Errorf("Max difference: %v (tolerance: %v)", maxDiff, tolerance)
			}
		})
	}
}

// BenchmarkFusedInt8MatMul benchmarks Int8 fused matmul.
func BenchmarkFusedInt8MatMul(b *testing.B) {
	rng := testRNGInt8()

	sizes := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		{"32x64x128", 32, 64, 128, 32},
		{"64x128x256", 64, 128, 256, 64},
		{"64x256x512", 64, 256, 512, 128},
	}

	for _, sz := range sizes {
		input := make([]float32, sz.M*sz.K)
		for i := range input {
			input[i] = rng.Float32()*2 - 1
		}

		weights := make([]int8, sz.K*sz.N)
		for i := range weights {
			weights[i] = int8(rng.Intn(256) - 128)
		}

		numGroups := (sz.N + sz.groupSize - 1) / sz.groupSize
		scales := make([]float32, sz.K*numGroups)
		for i := range scales {
			scales[i] = rng.Float32()*0.1 + 0.01
		}

		output := make([]float32, sz.M*sz.N)

		b.Run(sz.name, func(b *testing.B) {
			ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				FusedInt8MatMul(input, weights, scales, output, sz.M, sz.K, sz.N, sz.groupSize)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

// BenchmarkFusedInt8MatMulFallback benchmarks the scalar fallback.
func BenchmarkFusedInt8MatMulFallback(b *testing.B) {
	rng := testRNGInt8()

	sz := struct {
		M, K, N   int
		groupSize int
	}{64, 128, 256, 64}

	input := make([]float32, sz.M*sz.K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	weights := make([]int8, sz.K*sz.N)
	for i := range weights {
		weights[i] = int8(rng.Intn(256) - 128)
	}

	numGroups := (sz.N + sz.groupSize - 1) / sz.groupSize
	scales := make([]float32, sz.K*numGroups)
	for i := range scales {
		scales[i] = rng.Float32()*0.1 + 0.01
	}

	output := make([]float32, sz.M*sz.N)

	b.Run("64x128x256_fallback", func(b *testing.B) {
		ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			BaseFusedInt8MatMul_fallback(input, weights, scales, output, sz.M, sz.K, sz.N, sz.groupSize)
		}
		b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
}
