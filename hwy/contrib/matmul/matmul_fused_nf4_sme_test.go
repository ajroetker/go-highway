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

// testRNGSME returns a seeded random number generator for reproducible tests.
func testRNGSME() *rand.Rand {
	return rand.New(rand.NewSource(42))
}

// TestFusedNF4MatMulCorrectness verifies fused NF4 matmul produces correct results.
// Compares SME implementation against scalar fallback.
func TestFusedNF4MatMulCorrectness(t *testing.T) {
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
			rng := testRNGSME()

			input := make([]float32, tc.M*tc.K)
			for i := range input {
				input[i] = rng.Float32()*2 - 1
			}

			packedSize := (tc.K * tc.N + 1) / 2
			packed := make([]uint8, packedSize)
			for i := range packed {
				packed[i] = uint8(rng.Intn(256))
			}

			numGroups := (tc.N + tc.groupSize - 1) / tc.groupSize
			scales := make([]float32, tc.K*numGroups)
			for i := range scales {
				scales[i] = rng.Float32()*2 + 0.1
			}

			// Run fused kernel (SME path)
			fusedOutput := make([]float32, tc.M*tc.N)
			FusedNF4MatMul(input, packed, scales, fusedOutput, tc.M, tc.K, tc.N, tc.groupSize)

			// Run reference scalar
			refOutput := make([]float32, tc.M*tc.N)
			BaseFusedNF4MatMul_fallback(input, packed, scales, refOutput, tc.M, tc.K, tc.N, tc.groupSize)

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
			tolerance := float32(1e-2)
			if maxDiff > tolerance {
				t.Errorf("Max difference: %v (tolerance: %v), avg: %v", maxDiff, tolerance, avgDiff)
			} else {
				t.Logf("Max difference: %v, avg: %v", maxDiff, avgDiff)
			}
		})
	}
}

// TestFusedInt4MatMulCorrectness verifies fused Int4 matmul produces correct results.
func TestFusedInt4MatMulCorrectness(t *testing.T) {
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
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rng := testRNGSME()

			input := make([]float32, tc.M*tc.K)
			for i := range input {
				input[i] = rng.Float32()*2 - 1
			}

			packedSize := (tc.K * tc.N + 1) / 2
			packed := make([]uint8, packedSize)
			for i := range packed {
				packed[i] = uint8(rng.Intn(256))
			}

			numGroups := (tc.N + tc.groupSize - 1) / tc.groupSize
			scales := make([]float32, tc.K*numGroups)
			for i := range scales {
				scales[i] = rng.Float32()*2 + 0.1
			}

			fusedOutput := make([]float32, tc.M*tc.N)
			FusedInt4MatMul(input, packed, scales, fusedOutput, tc.M, tc.K, tc.N, tc.groupSize)

			refOutput := make([]float32, tc.M*tc.N)
			BaseFusedInt4MatMul_fallback(input, packed, scales, refOutput, tc.M, tc.K, tc.N, tc.groupSize)

			maxDiff := float32(0)
			for i := range fusedOutput {
				diff := float32(math.Abs(float64(fusedOutput[i] - refOutput[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			tolerance := float32(1e-2)
			if maxDiff > tolerance {
				t.Errorf("Max difference: %v (tolerance: %v)", maxDiff, tolerance)
			}
		})
	}
}

// TestFusedNF4GroupBoundaryCrossing verifies correctness when tiles cross group boundaries.
func TestFusedNF4GroupBoundaryCrossing(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	// Test groupSize values that force tiles to cross group boundaries
	testCases := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		{"64x64x80_cross", 64, 64, 80, 40},
		{"64x128x160_cross", 64, 128, 160, 40},
		{"64x64x96_cross48", 64, 64, 96, 48},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rng := testRNGSME()

			input := make([]float32, tc.M*tc.K)
			for i := range input {
				input[i] = rng.Float32()*2 - 1
			}

			packedSize := (tc.K * tc.N + 1) / 2
			packed := make([]uint8, packedSize)
			for i := range packed {
				packed[i] = uint8(rng.Intn(256))
			}

			numGroups := (tc.N + tc.groupSize - 1) / tc.groupSize
			scales := make([]float32, tc.K*numGroups)
			for i := range scales {
				scales[i] = rng.Float32()*2 + 0.1
			}

			fusedOutput := make([]float32, tc.M*tc.N)
			FusedNF4MatMul(input, packed, scales, fusedOutput, tc.M, tc.K, tc.N, tc.groupSize)

			refOutput := make([]float32, tc.M*tc.N)
			BaseFusedNF4MatMul_fallback(input, packed, scales, refOutput, tc.M, tc.K, tc.N, tc.groupSize)

			maxDiff := float32(0)
			maxDiffIdx := 0
			for i := range fusedOutput {
				diff := float32(math.Abs(float64(fusedOutput[i] - refOutput[i])))
				if diff > maxDiff {
					maxDiff = diff
					maxDiffIdx = i
				}
			}

			tolerance := float32(1e-2)
			if maxDiff > tolerance {
				row := maxDiffIdx / tc.N
				col := maxDiffIdx % tc.N
				t.Errorf("Max difference: %v at [%d,%d] (tolerance: %v)", maxDiff, row, col, tolerance)
			}
		})
	}
}

// TestFusedInt4GroupBoundaryCrossing verifies Int4 correctness at group boundaries.
func TestFusedInt4GroupBoundaryCrossing(t *testing.T) {
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
			rng := testRNGSME()

			input := make([]float32, tc.M*tc.K)
			for i := range input {
				input[i] = rng.Float32()*2 - 1
			}

			packedSize := (tc.K * tc.N + 1) / 2
			packed := make([]uint8, packedSize)
			for i := range packed {
				packed[i] = uint8(rng.Intn(256))
			}

			numGroups := (tc.N + tc.groupSize - 1) / tc.groupSize
			scales := make([]float32, tc.K*numGroups)
			for i := range scales {
				scales[i] = rng.Float32()*2 + 0.1
			}

			fusedOutput := make([]float32, tc.M*tc.N)
			FusedInt4MatMul(input, packed, scales, fusedOutput, tc.M, tc.K, tc.N, tc.groupSize)

			refOutput := make([]float32, tc.M*tc.N)
			BaseFusedInt4MatMul_fallback(input, packed, scales, refOutput, tc.M, tc.K, tc.N, tc.groupSize)

			maxDiff := float32(0)
			maxDiffIdx := 0
			for i := range fusedOutput {
				diff := float32(math.Abs(float64(fusedOutput[i] - refOutput[i])))
				if diff > maxDiff {
					maxDiff = diff
					maxDiffIdx = i
				}
			}

			tolerance := float32(1e-2)
			if maxDiff > tolerance {
				row := maxDiffIdx / tc.N
				col := maxDiffIdx % tc.N
				t.Errorf("Max difference: %v at [%d,%d] (tolerance: %v)", maxDiff, row, col, tolerance)
			}
		})
	}
}

// TestParallelFusedNF4MatMulCorrectness verifies parallel NF4 matmul produces correct results.
func TestParallelFusedNF4MatMulCorrectness(t *testing.T) {
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
		{"64x1024x2048", 64, 1024, 2048, 128},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rng := testRNGSME()

			input := make([]float32, tc.M*tc.K)
			for i := range input {
				input[i] = rng.Float32()*2 - 1
			}

			packedSize := (tc.K * tc.N + 1) / 2
			packed := make([]uint8, packedSize)
			for i := range packed {
				packed[i] = uint8(rng.Intn(256))
			}

			numGroups := (tc.N + tc.groupSize - 1) / tc.groupSize
			scales := make([]float32, tc.K*numGroups)
			for i := range scales {
				scales[i] = rng.Float32()*2 + 0.1
			}

			parallelOutput := make([]float32, tc.M*tc.N)
			ParallelFusedNF4MatMul(input, packed, scales, parallelOutput, tc.M, tc.K, tc.N, tc.groupSize)

			seqOutput := make([]float32, tc.M*tc.N)
			FusedNF4MatMul(input, packed, scales, seqOutput, tc.M, tc.K, tc.N, tc.groupSize)

			maxDiff := float32(0)
			avgDiff := float64(0)
			for i := range parallelOutput {
				diff := float32(math.Abs(float64(parallelOutput[i] - seqOutput[i])))
				avgDiff += float64(diff)
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			avgDiff /= float64(len(parallelOutput))

			// Should be nearly identical (same algorithm, just parallelized)
			tolerance := float32(1e-5)
			if maxDiff > tolerance {
				t.Errorf("Max difference: %v (tolerance: %v), avg: %v", maxDiff, tolerance, avgDiff)
			} else {
				t.Logf("Max difference: %v, avg: %v", maxDiff, avgDiff)
			}
		})
	}
}

// TestParallelFusedInt4MatMulCorrectness verifies parallel Int4 matmul produces correct results.
func TestParallelFusedInt4MatMulCorrectness(t *testing.T) {
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
			rng := testRNGSME()

			input := make([]float32, tc.M*tc.K)
			for i := range input {
				input[i] = rng.Float32()*2 - 1
			}

			packedSize := (tc.K * tc.N + 1) / 2
			packed := make([]uint8, packedSize)
			for i := range packed {
				packed[i] = uint8(rng.Intn(256))
			}

			numGroups := (tc.N + tc.groupSize - 1) / tc.groupSize
			scales := make([]float32, tc.K*numGroups)
			for i := range scales {
				scales[i] = rng.Float32()*2 + 0.1
			}

			parallelOutput := make([]float32, tc.M*tc.N)
			ParallelFusedInt4MatMul(input, packed, scales, parallelOutput, tc.M, tc.K, tc.N, tc.groupSize)

			seqOutput := make([]float32, tc.M*tc.N)
			FusedInt4MatMul(input, packed, scales, seqOutput, tc.M, tc.K, tc.N, tc.groupSize)

			maxDiff := float32(0)
			for i := range parallelOutput {
				diff := float32(math.Abs(float64(parallelOutput[i] - seqOutput[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			tolerance := float32(1e-5)
			if maxDiff > tolerance {
				t.Errorf("Max difference: %v (tolerance: %v)", maxDiff, tolerance)
			}
		})
	}
}

// BenchmarkFusedNF4MatMul benchmarks the fused NF4 matmul kernel (SME).
func BenchmarkFusedNF4MatMul(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	rng := testRNGSME()

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

		packedSize := (sz.K * sz.N + 1) / 2
		packed := make([]uint8, packedSize)
		for i := range packed {
			packed[i] = uint8(rng.Intn(256))
		}

		numGroups := (sz.N + sz.groupSize - 1) / sz.groupSize
		scales := make([]float32, sz.K*numGroups)
		for i := range scales {
			scales[i] = rng.Float32() + 0.1
		}

		output := make([]float32, sz.M*sz.N)

		b.Run(sz.name, func(b *testing.B) {
			ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				FusedNF4MatMul(input, packed, scales, output, sz.M, sz.K, sz.N, sz.groupSize)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			b.ReportMetric(b.Elapsed().Seconds()*1000/float64(b.N), "ms/op")
		})
	}
}

// BenchmarkParallelFusedNF4MatMul benchmarks parallel NF4 matmul.
func BenchmarkParallelFusedNF4MatMul(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	rng := testRNGSME()

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

		packedSize := (sz.K * sz.N + 1) / 2
		packed := make([]uint8, packedSize)
		for i := range packed {
			packed[i] = uint8(rng.Intn(256))
		}

		numGroups := (sz.N + sz.groupSize - 1) / sz.groupSize
		scales := make([]float32, sz.K*numGroups)
		for i := range scales {
			scales[i] = rng.Float32() + 0.1
		}

		output := make([]float32, sz.M*sz.N)

		b.Run(sz.name+"_parallel", func(b *testing.B) {
			ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ParallelFusedNF4MatMul(input, packed, scales, output, sz.M, sz.K, sz.N, sz.groupSize)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			b.ReportMetric(b.Elapsed().Seconds()*1000/float64(b.N), "ms/op")
		})
	}
}

// BenchmarkFusedNF4Comparison directly compares sequential vs parallel performance.
func BenchmarkFusedNF4Comparison(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	rng := testRNGSME()

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

		packedSize := (sz.K * sz.N + 1) / 2
		packed := make([]uint8, packedSize)
		for i := range packed {
			packed[i] = uint8(rng.Intn(256))
		}

		numGroups := (sz.N + sz.groupSize - 1) / sz.groupSize
		scales := make([]float32, sz.K*numGroups)
		for i := range scales {
			scales[i] = rng.Float32() + 0.1
		}

		output := make([]float32, sz.M*sz.N)

		b.Run(sz.name+"/sequential", func(b *testing.B) {
			ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				FusedNF4MatMul(input, packed, scales, output, sz.M, sz.K, sz.N, sz.groupSize)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			b.ReportMetric(b.Elapsed().Seconds()*1000/float64(b.N), "ms/op")
		})

		b.Run(sz.name+"/parallel", func(b *testing.B) {
			ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ParallelFusedNF4MatMul(input, packed, scales, output, sz.M, sz.K, sz.N, sz.groupSize)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			b.ReportMetric(b.Elapsed().Seconds()*1000/float64(b.N), "ms/op")
		})
	}
}
