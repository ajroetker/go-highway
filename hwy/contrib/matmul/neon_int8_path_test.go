//go:build arm64

package matmul

import (
	"math"
	"testing"
)

func TestNeonGoPath(t *testing.T) {
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

	// Run Go NEON path directly
	neonOutput := make([]float32, M*N)
	BaseFusedInt8MatMul_neon(input, weights, scales, nil, neonOutput, M, K, N, groupSize)

	// Run reference
	refOutput := referenceInt8MatMul(input, weights, scales, M, K, N, groupSize)

	maxDiff := float32(0)
	for i := range neonOutput {
		diff := float32(math.Abs(float64(neonOutput[i] - refOutput[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("Go NEON path max diff: %v", maxDiff)
	if maxDiff > 1e-4 {
		t.Errorf("Go NEON path failed: max diff %v", maxDiff)
	}

	// Run dispatched path (uses asm on arm64)
	asmOutput := make([]float32, M*N)
	FusedInt8MatMul(input, weights, scales, nil, asmOutput, M, K, N, groupSize)
	maxDiffAsm := float32(0)
	for i := range asmOutput {
		diff := float32(math.Abs(float64(asmOutput[i] - refOutput[i])))
		if diff > maxDiffAsm {
			maxDiffAsm = diff
		}
	}
	t.Logf("ASM path max diff: %v", maxDiffAsm)
	if maxDiffAsm > 1e-4 {
		t.Errorf("ASM path failed: max diff %v", maxDiffAsm)
	}

	// Run fallback
	fbOutput := make([]float32, M*N)
	BaseFusedInt8MatMul_fallback(input, weights, scales, nil, fbOutput, M, K, N, groupSize)
	maxDiffFb := float32(0)
	for i := range fbOutput {
		diff := float32(math.Abs(float64(fbOutput[i] - refOutput[i])))
		if diff > maxDiffFb {
			maxDiffFb = diff
		}
	}
	t.Logf("Fallback path max diff: %v", maxDiffFb)
}
