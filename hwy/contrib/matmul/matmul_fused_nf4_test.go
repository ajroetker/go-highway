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

// testRNGFusedNF4 returns a seeded random number generator for reproducible tests.
func testRNGFusedNF4() *rand.Rand {
	return rand.New(rand.NewSource(42))
}

// TestFusedNF4FallbackCorrectness verifies the scalar fallback produces correct results.
// This test runs on all platforms.
func TestFusedNF4FallbackCorrectness(t *testing.T) {
	rng := testRNGFusedNF4()

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

			packedSize := (tc.K*tc.N + 1) / 2
			packed := make([]uint8, packedSize)
			for i := range packed {
				packed[i] = uint8(rng.Intn(256))
			}

			numGroups := (tc.N + tc.groupSize - 1) / tc.groupSize
			scales := make([]float32, tc.K*numGroups)
			for i := range scales {
				scales[i] = rng.Float32() + 0.1
			}

			// Run via dispatch (should use fallback for small sizes)
			fusedOutput := make([]float32, tc.M*tc.N)
			FusedNF4MatMul(input, packed, scales, fusedOutput, tc.M, tc.K, tc.N, tc.groupSize)

			// Run scalar directly
			scalarOutput := make([]float32, tc.M*tc.N)
			BaseFusedNF4MatMul_fallback(input, packed, scales, scalarOutput, tc.M, tc.K, tc.N, tc.groupSize)

			// Should be identical when both use fallback
			for i := range fusedOutput {
				if fusedOutput[i] != scalarOutput[i] {
					t.Errorf("Mismatch at index %d: dispatch=%v scalar=%v", i, fusedOutput[i], scalarOutput[i])
					return
				}
			}
		})
	}
}

// TestFusedInt4FallbackCorrectness verifies the scalar Int4 fallback produces correct results.
func TestFusedInt4FallbackCorrectness(t *testing.T) {
	rng := testRNGFusedNF4()

	M, K, N := 16, 32, 48
	groupSize := 16

	input := make([]float32, M*K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	packedSize := (K*N + 1) / 2
	packed := make([]uint8, packedSize)
	for i := range packed {
		packed[i] = uint8(rng.Intn(256))
	}

	numGroups := (N + groupSize - 1) / groupSize
	scales := make([]float32, K*numGroups)
	for i := range scales {
		scales[i] = rng.Float32() + 0.1
	}

	fusedOutput := make([]float32, M*N)
	FusedInt4MatMul(input, packed, scales, fusedOutput, M, K, N, groupSize)

	scalarOutput := make([]float32, M*N)
	BaseFusedInt4MatMul_fallback(input, packed, scales, scalarOutput, M, K, N, groupSize)

	for i := range fusedOutput {
		if fusedOutput[i] != scalarOutput[i] {
			t.Errorf("Mismatch at index %d: dispatch=%v scalar=%v", i, fusedOutput[i], scalarOutput[i])
			return
		}
	}
}

// TestNF4LookupTable verifies the NF4 lookup table has expected properties.
func TestNF4LookupTable(t *testing.T) {
	// Check table size
	if len(nf4LookupTable) != 16 {
		t.Errorf("Expected 16 entries, got %d", len(nf4LookupTable))
	}

	// Check boundary values
	if nf4LookupTable[0] != -1.0 {
		t.Errorf("Expected first entry to be -1.0, got %v", nf4LookupTable[0])
	}
	if nf4LookupTable[15] != 1.0 {
		t.Errorf("Expected last entry to be 1.0, got %v", nf4LookupTable[15])
	}

	// Check zero is in the table
	hasZero := false
	for _, v := range nf4LookupTable {
		if v == 0.0 {
			hasZero = true
			break
		}
	}
	if !hasZero {
		t.Error("Expected NF4 table to contain 0.0")
	}

	// Check values are sorted
	for i := 1; i < len(nf4LookupTable); i++ {
		if nf4LookupTable[i] <= nf4LookupTable[i-1] {
			t.Errorf("Table not sorted at index %d: %v <= %v", i, nf4LookupTable[i], nf4LookupTable[i-1])
		}
	}
}

// TestFusedNF4PackingConsistency verifies that the packing format is consistent.
func TestFusedNF4PackingConsistency(t *testing.T) {
	rng := testRNGFusedNF4()

	// Create known weights and verify unpacking
	K, N := 4, 8
	groupSize := 8

	// Pack known values
	packed := make([]uint8, (K*N+1)/2)
	for i := range packed {
		packed[i] = uint8(rng.Intn(256))
	}

	// Create identity-like input (single row, unit values)
	M := 1
	input := make([]float32, M*K)
	for i := range input {
		input[i] = 1.0
	}

	// Use unit scales
	numGroups := (N + groupSize - 1) / groupSize
	scales := make([]float32, K*numGroups)
	for i := range scales {
		scales[i] = 1.0
	}

	output := make([]float32, M*N)
	BaseFusedNF4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)

	// Verify output is within NF4 table bounds * K
	maxPossible := float32(K) * 1.0  // max table value
	minPossible := float32(K) * -1.0 // min table value

	for i := range N {
		if output[i] > maxPossible || output[i] < minPossible {
			t.Errorf("Output[%d] = %v out of expected range [%v, %v]", i, output[i], minPossible, maxPossible)
		}
	}
}

// TestFusedInt4SymmetricQuantization verifies Int4 [-8,7] range.
func TestFusedInt4SymmetricQuantization(t *testing.T) {
	// Create a single packed byte with known values
	// Packing: low nibble = first value (even index), high nibble = second value (odd index)
	// 0xF0 = high nibble 15, low nibble 0
	packed := []uint8{0xF0} // low=0, high=15

	K, N := 1, 2
	M := 1
	groupSize := 2

	input := []float32{1.0}  // identity
	scales := []float32{1.0} // unit scale
	output := make([]float32, M*N)

	BaseFusedInt4MatMul_fallback(input, packed, scales, output, M, K, N, groupSize)

	// weightIdx=0 (even) uses low nibble = 0 -> (0-8) = -8
	// weightIdx=1 (odd) uses high nibble = 15 -> (15-8) = 7
	if math.Abs(float64(output[0]-(-8.0))) > 1e-6 {
		t.Errorf("Expected output[0] = -8, got %v", output[0])
	}
	if math.Abs(float64(output[1]-7.0)) > 1e-6 {
		t.Errorf("Expected output[1] = 7, got %v", output[1])
	}
}

// BenchmarkFusedNF4Scalar benchmarks the scalar fallback.
func BenchmarkFusedNF4Scalar(b *testing.B) {
	rng := testRNGFusedNF4()

	sizes := []struct {
		name      string
		M, K, N   int
		groupSize int
	}{
		{"32x64x128", 32, 64, 128, 32},
		{"64x128x256", 64, 128, 256, 64},
	}

	for _, sz := range sizes {
		input := make([]float32, sz.M*sz.K)
		for i := range input {
			input[i] = rng.Float32()*2 - 1
		}

		packedSize := (sz.K*sz.N + 1) / 2
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
				BaseFusedNF4MatMul_fallback(input, packed, scales, output, sz.M, sz.K, sz.N, sz.groupSize)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

// BenchmarkFusedInt4Scalar benchmarks the scalar Int4 fallback.
func BenchmarkFusedInt4Scalar(b *testing.B) {
	rng := testRNGFusedNF4()

	sz := struct {
		M, K, N   int
		groupSize int
	}{64, 128, 256, 64}

	input := make([]float32, sz.M*sz.K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	packedSize := (sz.K*sz.N + 1) / 2
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

	b.Run("64x128x256", func(b *testing.B) {
		ops := float64(sz.M) * float64(sz.K) * float64(sz.N) * 2
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			BaseFusedInt4MatMul_fallback(input, packed, scales, output, sz.M, sz.K, sz.N, sz.groupSize)
		}
		b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
}
