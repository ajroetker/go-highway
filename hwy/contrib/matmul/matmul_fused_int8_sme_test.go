//go:build !noasm && darwin && arm64

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
)

// testRNGInt8SME returns a seeded random number generator for reproducible tests.
func testRNGInt8SME() *rand.Rand {
	return rand.New(rand.NewSource(42))
}

// TestFusedInt8MatMulSMECorrectness verifies fused Int8 matmul produces correct results.
// Compares SME implementation against scalar fallback.
func TestFusedInt8MatMulSMECorrectness(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	testCases := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		{"64x64x64", 64, 64, 64, 32},
		{"64x128x256", 64, 128, 256, 64},
		{"64x256x512", 64, 256, 512, 128},
		{"128x512x1024", 128, 512, 1024, 128},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rng := testRNGInt8SME()

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

			// Run fused kernel (SME path)
			fusedOutput := make([]float32, tc.M*tc.N)
			FusedInt8MatMul(input, weights, scales, fusedOutput, tc.M, tc.K, tc.N, tc.groupSize)

			// Run reference scalar
			refOutput := make([]float32, tc.M*tc.N)
			BaseFusedInt8MatMul_fallback(input, weights, scales, refOutput, tc.M, tc.K, tc.N, tc.groupSize)

			maxDiff := float32(0)
			avgDiff := float64(0)
			for i := range fusedOutput {
				diff := float32(math.Abs(float64(fusedOutput[i] - refOutput[i])))
				avgDiff += float64(diff)
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			avgDiff /= float64(len(fusedOutput))

			// Allow for floating point differences due to different computation order
			tolerance := float32(1e-4)
			if maxDiff > tolerance {
				t.Errorf("Max difference: %v (tolerance: %v), avg: %v", maxDiff, tolerance, avgDiff)
			} else {
				t.Logf("Max difference: %v, avg: %v", maxDiff, avgDiff)
			}
		})
	}
}

// TestFusedInt8MatMulSMEGroupBoundaryCrossing verifies correctness when tiles cross group boundaries.
func TestFusedInt8MatMulSMEGroupBoundaryCrossing(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	testCases := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		{"64x64x80_cross", 64, 64, 80, 40},
		{"64x128x160_cross", 64, 128, 160, 40},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rng := testRNGInt8SME()

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

			refOutput := make([]float32, tc.M*tc.N)
			BaseFusedInt8MatMul_fallback(input, weights, scales, refOutput, tc.M, tc.K, tc.N, tc.groupSize)

			maxDiff := float32(0)
			maxDiffIdx := 0
			for i := range fusedOutput {
				diff := float32(math.Abs(float64(fusedOutput[i] - refOutput[i])))
				if diff > maxDiff {
					maxDiff = diff
					maxDiffIdx = i
				}
			}

			tolerance := float32(1e-4)
			if maxDiff > tolerance {
				row := maxDiffIdx / tc.N
				col := maxDiffIdx % tc.N
				t.Errorf("Max difference: %v at [%d,%d] (tolerance: %v)", maxDiff, row, col, tolerance)
			}
		})
	}
}

// TestFusedInt8MatMulSMEUnaligned verifies correctness with non-16-aligned dimensions.
// These dimensions force the fallback path since SME requires 16-alignment.
func TestFusedInt8MatMulSMEUnaligned(t *testing.T) {
	testCases := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		// K not 16-aligned
		{"32x33x64_K_unaligned", 32, 33, 64, 32},
		// N not 16-aligned
		{"32x64x33_N_unaligned", 32, 64, 33, 32},
		// Both K and N unaligned
		{"32x33x33_both_unaligned", 32, 33, 33, 16},
		// Small dimensions (below min threshold)
		{"16x32x32_small_M", 16, 32, 32, 16},
		{"32x32x32_small", 32, 32, 32, 16},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rng := testRNGInt8SME()

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

			// Run through FusedInt8MatMul (which should use fallback for unaligned)
			fusedOutput := make([]float32, tc.M*tc.N)
			FusedInt8MatMul(input, weights, scales, fusedOutput, tc.M, tc.K, tc.N, tc.groupSize)

			// Run reference scalar directly
			refOutput := make([]float32, tc.M*tc.N)
			BaseFusedInt8MatMul_fallback(input, weights, scales, refOutput, tc.M, tc.K, tc.N, tc.groupSize)

			maxDiff := float32(0)
			for i := range fusedOutput {
				diff := float32(math.Abs(float64(fusedOutput[i] - refOutput[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			// Should match exactly since both use fallback path
			tolerance := float32(1e-6)
			if maxDiff > tolerance {
				t.Errorf("Max difference: %v (tolerance: %v)", maxDiff, tolerance)
			}
		})
	}
}

// TestParallelFusedInt8MatMulSMECorrectness verifies parallel Int8 matmul produces correct results.
func TestParallelFusedInt8MatMulSMECorrectness(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	testCases := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		{"64x64x64", 64, 64, 64, 32},
		{"64x128x256", 64, 128, 256, 64},
		{"64x256x512", 64, 256, 512, 128},
		{"64x1024x2048", 64, 1024, 2048, 128},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rng := testRNGInt8SME()

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

			parallelOutput := make([]float32, tc.M*tc.N)
			ParallelFusedInt8MatMul(input, weights, scales, parallelOutput, tc.M, tc.K, tc.N, tc.groupSize)

			seqOutput := make([]float32, tc.M*tc.N)
			FusedInt8MatMul(input, weights, scales, seqOutput, tc.M, tc.K, tc.N, tc.groupSize)

			maxDiff := float32(0)
			for i := range parallelOutput {
				diff := float32(math.Abs(float64(parallelOutput[i] - seqOutput[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			// Should be nearly identical (same algorithm, just parallelized)
			tolerance := float32(1e-5)
			if maxDiff > tolerance {
				t.Errorf("Max difference: %v (tolerance: %v)", maxDiff, tolerance)
			}
		})
	}
}

// BenchmarkFusedInt8MatMulSME benchmarks the fused Int8 matmul kernel (SME).
func BenchmarkFusedInt8MatMulSME(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	rng := testRNGInt8SME()

	sizes := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		{"64x256x512", 64, 256, 512, 128},
		{"64x512x1024", 64, 512, 1024, 128},
		{"64x1024x2048", 64, 1024, 2048, 128},
		{"64x4096x4096", 64, 4096, 4096, 128},
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
			b.ReportMetric(b.Elapsed().Seconds()*1000/float64(b.N), "ms/op")
		})
	}
}

// BenchmarkParallelFusedInt8MatMulSME benchmarks parallel Int8 matmul.
func BenchmarkParallelFusedInt8MatMulSME(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	rng := testRNGInt8SME()

	sizes := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		{"64x256x512", 64, 256, 512, 128},
		{"64x512x1024", 64, 512, 1024, 128},
		{"64x1024x2048", 64, 1024, 2048, 128},
		{"64x4096x4096", 64, 4096, 4096, 128},
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

		b.Run(sz.name+"_parallel", func(b *testing.B) {
			ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ParallelFusedInt8MatMul(input, weights, scales, output, sz.M, sz.K, sz.N, sz.groupSize)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			b.ReportMetric(b.Elapsed().Seconds()*1000/float64(b.N), "ms/op")
		})
	}
}

// BenchmarkFusedInt8Comparison directly compares sequential vs parallel performance.
func BenchmarkFusedInt8Comparison(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	rng := testRNGInt8SME()

	sizes := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		{"64x1024x2048", 64, 1024, 2048, 128},
		{"64x4096x4096", 64, 4096, 4096, 128},
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

		b.Run(sz.name+"/sequential", func(b *testing.B) {
			ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				FusedInt8MatMul(input, weights, scales, output, sz.M, sz.K, sz.N, sz.groupSize)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			b.ReportMetric(b.Elapsed().Seconds()*1000/float64(b.N), "ms/op")
		})

		b.Run(sz.name+"/parallel", func(b *testing.B) {
			ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ParallelFusedInt8MatMul(input, weights, scales, output, sz.M, sz.K, sz.N, sz.groupSize)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			b.ReportMetric(b.Elapsed().Seconds()*1000/float64(b.N), "ms/op")
		})
	}
}
