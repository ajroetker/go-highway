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

// testRNGActSME returns a seeded random number generator for reproducible tests.
func testRNGActSME() *rand.Rand {
	return rand.New(rand.NewSource(42))
}

// TestFusedNF4MatMulSiLUSME verifies SME implementation of fused NF4 + SiLU.
func TestFusedNF4MatMulSiLUSME(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	rng := testRNGActSME()
	// Use SME-aligned dimensions (multiples of 16, minimum 64)
	M, K, N := 64, 64, 64
	groupSize := 16

	input := make([]float32, M*K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	packedSize := (K * N + 1) / 2
	packed := make([]uint8, packedSize)
	for i := range packed {
		packed[i] = uint8(rng.Intn(256))
	}

	numGroups := (N + groupSize - 1) / groupSize
	scales := make([]float32, K*numGroups)
	for i := range scales {
		scales[i] = rng.Float32() + 0.1
	}

	// Compute with base implementation
	baseOutput := make([]float32, M*N)
	BaseFusedNF4MatMulSiLU(input, packed, scales, baseOutput, M, K, N, groupSize)

	// Compute with SME implementation via dispatch
	smeOutput := make([]float32, M*N)
	FusedNF4MatMulSiLU(input, packed, scales, smeOutput, M, K, N, groupSize)

	// Verify results match (allowing for small numerical differences)
	maxDiff := float64(0)
	for i := range smeOutput {
		diff := math.Abs(float64(smeOutput[i] - baseOutput[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-4 {
			t.Errorf("Index %d: SME=%v, Base=%v (diff=%v)", i, smeOutput[i], baseOutput[i], diff)
		}
	}
	t.Logf("Max diff between SME and Base: %v", maxDiff)
}

// TestFusedNF4MatMulGELUSME verifies SME implementation of fused NF4 + GELU.
func TestFusedNF4MatMulGELUSME(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	rng := testRNGActSME()
	M, K, N := 64, 64, 64
	groupSize := 16

	input := make([]float32, M*K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	packedSize := (K * N + 1) / 2
	packed := make([]uint8, packedSize)
	for i := range packed {
		packed[i] = uint8(rng.Intn(256))
	}

	numGroups := (N + groupSize - 1) / groupSize
	scales := make([]float32, K*numGroups)
	for i := range scales {
		scales[i] = rng.Float32() + 0.1
	}

	baseOutput := make([]float32, M*N)
	BaseFusedNF4MatMulGELU(input, packed, scales, baseOutput, M, K, N, groupSize)

	smeOutput := make([]float32, M*N)
	FusedNF4MatMulGELU(input, packed, scales, smeOutput, M, K, N, groupSize)

	maxDiff := float64(0)
	for i := range smeOutput {
		diff := math.Abs(float64(smeOutput[i] - baseOutput[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-3 { // GELU uses erf which has more approximation error
			t.Errorf("Index %d: SME=%v, Base=%v (diff=%v)", i, smeOutput[i], baseOutput[i], diff)
		}
	}
	t.Logf("Max diff between SME and Base: %v", maxDiff)
}

// TestFusedInt4MatMulSiLUSME verifies SME implementation of fused Int4 + SiLU.
func TestFusedInt4MatMulSiLUSME(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	rng := testRNGActSME()
	M, K, N := 64, 64, 64
	groupSize := 16

	input := make([]float32, M*K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	packedSize := (K * N + 1) / 2
	packed := make([]uint8, packedSize)
	for i := range packed {
		packed[i] = uint8(rng.Intn(256))
	}

	numGroups := (N + groupSize - 1) / groupSize
	scales := make([]float32, K*numGroups)
	for i := range scales {
		scales[i] = rng.Float32() + 0.1
	}

	baseOutput := make([]float32, M*N)
	BaseFusedInt4MatMulSiLU(input, packed, scales, baseOutput, M, K, N, groupSize)

	smeOutput := make([]float32, M*N)
	FusedInt4MatMulSiLU(input, packed, scales, smeOutput, M, K, N, groupSize)

	maxDiff := float64(0)
	for i := range smeOutput {
		diff := math.Abs(float64(smeOutput[i] - baseOutput[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-4 {
			t.Errorf("Index %d: SME=%v, Base=%v (diff=%v)", i, smeOutput[i], baseOutput[i], diff)
		}
	}
	t.Logf("Max diff between SME and Base: %v", maxDiff)
}

// TestFusedInt4MatMulGELUSME verifies SME implementation of fused Int4 + GELU.
func TestFusedInt4MatMulGELUSME(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	rng := testRNGActSME()
	M, K, N := 64, 64, 64
	groupSize := 16

	input := make([]float32, M*K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	packedSize := (K * N + 1) / 2
	packed := make([]uint8, packedSize)
	for i := range packed {
		packed[i] = uint8(rng.Intn(256))
	}

	numGroups := (N + groupSize - 1) / groupSize
	scales := make([]float32, K*numGroups)
	for i := range scales {
		scales[i] = rng.Float32() + 0.1
	}

	baseOutput := make([]float32, M*N)
	BaseFusedInt4MatMulGELU(input, packed, scales, baseOutput, M, K, N, groupSize)

	smeOutput := make([]float32, M*N)
	FusedInt4MatMulGELU(input, packed, scales, smeOutput, M, K, N, groupSize)

	maxDiff := float64(0)
	for i := range smeOutput {
		diff := math.Abs(float64(smeOutput[i] - baseOutput[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-3 {
			t.Errorf("Index %d: SME=%v, Base=%v (diff=%v)", i, smeOutput[i], baseOutput[i], diff)
		}
	}
	t.Logf("Max diff between SME and Base: %v", maxDiff)
}

// TestParallelFusedNF4MatMulSiLUSME verifies parallel SME implementation.
func TestParallelFusedNF4MatMulSiLUSME(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	rng := testRNGActSME()
	// Larger dimensions to trigger parallel execution
	M, K, N := 64, 64, 128 // N >= 64 needed for MinFusedParallelTiles=4
	groupSize := 16

	input := make([]float32, M*K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	packedSize := (K * N + 1) / 2
	packed := make([]uint8, packedSize)
	for i := range packed {
		packed[i] = uint8(rng.Intn(256))
	}

	numGroups := (N + groupSize - 1) / groupSize
	scales := make([]float32, K*numGroups)
	for i := range scales {
		scales[i] = rng.Float32() + 0.1
	}

	// Compute with serial dispatch
	serialOutput := make([]float32, M*N)
	FusedNF4MatMulSiLU(input, packed, scales, serialOutput, M, K, N, groupSize)

	// Compute with parallel dispatch
	parallelOutput := make([]float32, M*N)
	ParallelFusedNF4MatMulSiLU(input, packed, scales, parallelOutput, M, K, N, groupSize)

	// Verify results match
	maxDiff := float64(0)
	for i := range parallelOutput {
		diff := math.Abs(float64(parallelOutput[i] - serialOutput[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-4 {
			t.Errorf("Index %d: Parallel=%v, Serial=%v (diff=%v)", i, parallelOutput[i], serialOutput[i], diff)
		}
	}
	t.Logf("Max diff between Parallel and Serial: %v", maxDiff)
}

// BenchmarkFusedNF4MatMulSiLUSME benchmarks SME fused NF4 + SiLU.
func BenchmarkFusedNF4MatMulSiLUSME(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	rng := testRNGActSME()
	M, K, N := 128, 256, 256
	groupSize := 32

	input := make([]float32, M*K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	packedSize := (K * N + 1) / 2
	packed := make([]uint8, packedSize)
	for i := range packed {
		packed[i] = uint8(rng.Intn(256))
	}

	numGroups := (N + groupSize - 1) / groupSize
	scales := make([]float32, K*numGroups)
	for i := range scales {
		scales[i] = rng.Float32() + 0.1
	}

	output := make([]float32, M*N)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedNF4MatMulSiLU(input, packed, scales, output, M, K, N, groupSize)
	}
}

// BenchmarkFusedNF4MatMulGELUSME benchmarks SME fused NF4 + GELU.
func BenchmarkFusedNF4MatMulGELUSME(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	rng := testRNGActSME()
	M, K, N := 128, 256, 256
	groupSize := 32

	input := make([]float32, M*K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	packedSize := (K * N + 1) / 2
	packed := make([]uint8, packedSize)
	for i := range packed {
		packed[i] = uint8(rng.Intn(256))
	}

	numGroups := (N + groupSize - 1) / groupSize
	scales := make([]float32, K*numGroups)
	for i := range scales {
		scales[i] = rng.Float32() + 0.1
	}

	output := make([]float32, M*N)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedNF4MatMulGELU(input, packed, scales, output, M, K, N, groupSize)
	}
}

// BenchmarkParallelFusedNF4MatMulSiLUSME benchmarks parallel SME fused NF4 + SiLU.
func BenchmarkParallelFusedNF4MatMulSiLUSME(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	rng := testRNGActSME()
	M, K, N := 256, 512, 512
	groupSize := 32

	input := make([]float32, M*K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	packedSize := (K * N + 1) / 2
	packed := make([]uint8, packedSize)
	for i := range packed {
		packed[i] = uint8(rng.Intn(256))
	}

	numGroups := (N + groupSize - 1) / groupSize
	scales := make([]float32, K*numGroups)
	for i := range scales {
		scales[i] = rng.Float32() + 0.1
	}

	output := make([]float32, M*N)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ParallelFusedNF4MatMulSiLU(input, packed, scales, output, M, K, N, groupSize)
	}
}
