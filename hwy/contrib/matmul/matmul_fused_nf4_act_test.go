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

// testRNGAct returns a seeded random number generator for reproducible tests.
func testRNGAct() *rand.Rand {
	return rand.New(rand.NewSource(42))
}

// referenceSiLU computes SiLU(x) = x * sigmoid(x) using standard library.
func referenceSiLU(x float32) float32 {
	return float32(float64(x) / (1.0 + math.Exp(-float64(x))))
}

// referenceGELU computes exact GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))).
func referenceGELU(x float32) float32 {
	return float32(float64(x) * 0.5 * (1.0 + math.Erf(float64(x)/math.Sqrt2)))
}

// referenceGELUApprox computes approximate GELU(x) = x * sigmoid(1.702 * x).
func referenceGELUApprox(x float32) float32 {
	return float32(float64(x) / (1.0 + math.Exp(-1.702*float64(x))))
}

// referenceReLU computes ReLU(x) = max(0, x).
func referenceReLU(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

// TestActivationFunctions verifies individual activation function implementations.
func TestActivationFunctions(t *testing.T) {
	testValues := []float32{-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10}

	t.Run("SiLU", func(t *testing.T) {
		for _, x := range testValues {
			got := applyActivationScalar(x, ActSiLU)
			want := referenceSiLU(x)
			diff := math.Abs(float64(got - want))
			// Allow slightly larger tolerance for the polynomial approximation
			if diff > 1e-4 {
				t.Errorf("SiLU(%v) = %v, want %v (diff=%v)", x, got, want, diff)
			}
		}
	})

	t.Run("GELU", func(t *testing.T) {
		for _, x := range testValues {
			got := applyActivationScalar(x, ActGELU)
			want := referenceGELU(x)
			diff := math.Abs(float64(got - want))
			// Allow larger tolerance for erf approximation
			if diff > 1e-3 {
				t.Errorf("GELU(%v) = %v, want %v (diff=%v)", x, got, want, diff)
			}
		}
	})

	t.Run("GELUApprox", func(t *testing.T) {
		for _, x := range testValues {
			got := applyActivationScalar(x, ActGELUApprox)
			want := referenceGELUApprox(x)
			diff := math.Abs(float64(got - want))
			if diff > 1e-4 {
				t.Errorf("GELUApprox(%v) = %v, want %v (diff=%v)", x, got, want, diff)
			}
		}
	})

	t.Run("ReLU", func(t *testing.T) {
		for _, x := range testValues {
			got := applyActivationScalar(x, ActReLU)
			want := referenceReLU(x)
			if got != want {
				t.Errorf("ReLU(%v) = %v, want %v", x, got, want)
			}
		}
	})
}

// TestFusedNF4MatMulSiLU verifies fused NF4 matmul with SiLU activation.
func TestFusedNF4MatMulSiLU(t *testing.T) {
	rng := testRNGAct()
	M, K, N := 16, 32, 48
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

	// First compute raw matmul
	rawOutput := make([]float32, M*N)
	BaseFusedNF4MatMul_fallback(input, packed, scales, rawOutput, M, K, N, groupSize)

	// Then compute fused SiLU version
	siluOutput := make([]float32, M*N)
	BaseFusedNF4MatMulSiLU(input, packed, scales, siluOutput, M, K, N, groupSize)

	// Verify: siluOutput should equal SiLU applied to rawOutput
	for i := range siluOutput {
		expected := applyActivationScalar(rawOutput[i], ActSiLU)
		diff := math.Abs(float64(siluOutput[i] - expected))
		if diff > 1e-5 {
			t.Errorf("Index %d: fused=%v, expected SiLU(raw)=%v (diff=%v)", i, siluOutput[i], expected, diff)
		}
	}
}

// TestFusedNF4MatMulGELU verifies fused NF4 matmul with GELU activation.
func TestFusedNF4MatMulGELU(t *testing.T) {
	rng := testRNGAct()
	M, K, N := 16, 32, 48
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

	rawOutput := make([]float32, M*N)
	BaseFusedNF4MatMul_fallback(input, packed, scales, rawOutput, M, K, N, groupSize)

	geluOutput := make([]float32, M*N)
	BaseFusedNF4MatMulGELU(input, packed, scales, geluOutput, M, K, N, groupSize)

	for i := range geluOutput {
		expected := applyActivationScalar(rawOutput[i], ActGELU)
		diff := math.Abs(float64(geluOutput[i] - expected))
		if diff > 1e-5 {
			t.Errorf("Index %d: fused=%v, expected GELU(raw)=%v (diff=%v)", i, geluOutput[i], expected, diff)
		}
	}
}

// TestFusedNF4MatMulReLU verifies fused NF4 matmul with ReLU activation.
func TestFusedNF4MatMulReLU(t *testing.T) {
	rng := testRNGAct()
	M, K, N := 16, 32, 48
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

	rawOutput := make([]float32, M*N)
	BaseFusedNF4MatMul_fallback(input, packed, scales, rawOutput, M, K, N, groupSize)

	reluOutput := make([]float32, M*N)
	BaseFusedNF4MatMulReLU(input, packed, scales, reluOutput, M, K, N, groupSize)

	for i := range reluOutput {
		expected := applyActivationScalar(rawOutput[i], ActReLU)
		diff := math.Abs(float64(reluOutput[i] - expected))
		if diff > 1e-5 {
			t.Errorf("Index %d: fused=%v, expected ReLU(raw)=%v (diff=%v)", i, reluOutput[i], expected, diff)
		}
	}
}

// TestFusedNF4MatMulSwiGLU verifies fused NF4 matmul with SwiGLU activation.
func TestFusedNF4MatMulSwiGLU(t *testing.T) {
	rng := testRNGAct()
	M, K, N := 16, 32, 48
	groupSize := 16

	input := make([]float32, M*K)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	packedSize := (K * N + 1) / 2
	gatePacked := make([]uint8, packedSize)
	upPacked := make([]uint8, packedSize)
	for i := range gatePacked {
		gatePacked[i] = uint8(rng.Intn(256))
		upPacked[i] = uint8(rng.Intn(256))
	}

	numGroups := (N + groupSize - 1) / groupSize
	gateScales := make([]float32, K*numGroups)
	upScales := make([]float32, K*numGroups)
	for i := range gateScales {
		gateScales[i] = rng.Float32() + 0.1
		upScales[i] = rng.Float32() + 0.1
	}

	// Compute gate and up projections separately
	gateOutput := make([]float32, M*N)
	upOutput := make([]float32, M*N)
	BaseFusedNF4MatMul_fallback(input, gatePacked, gateScales, gateOutput, M, K, N, groupSize)
	BaseFusedNF4MatMul_fallback(input, upPacked, upScales, upOutput, M, K, N, groupSize)

	// Compute fused SwiGLU
	swiGLUOutput := make([]float32, M*N)
	BaseFusedNF4MatMulSwiGLU(input, gatePacked, gateScales, upPacked, upScales, swiGLUOutput, M, K, N, groupSize)

	// Verify: SwiGLU = SiLU(gate) * up
	for i := range swiGLUOutput {
		gateSilu := applyActivationScalar(gateOutput[i], ActSiLU)
		expected := gateSilu * upOutput[i]
		diff := math.Abs(float64(swiGLUOutput[i] - expected))
		// Allow larger tolerance for compounded operations
		if diff > 1e-4 {
			t.Errorf("Index %d: fused=%v, expected SiLU(gate)*up=%v (diff=%v)", i, swiGLUOutput[i], expected, diff)
		}
	}
}

// TestFusedInt4MatMulActivations verifies fused Int4 matmul with activations.
func TestFusedInt4MatMulActivations(t *testing.T) {
	rng := testRNGAct()
	M, K, N := 16, 32, 48
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

	// Compute raw Int4 matmul
	rawOutput := make([]float32, M*N)
	BaseFusedInt4MatMul_fallback(input, packed, scales, rawOutput, M, K, N, groupSize)

	testCases := []struct {
		name string
		fn   func([]float32, []uint8, []float32, []float32, int, int, int, int)
		act  ActivationType
	}{
		{"SiLU", BaseFusedInt4MatMulSiLU, ActSiLU},
		{"GELU", BaseFusedInt4MatMulGELU, ActGELU},
		{"GELUApprox", BaseFusedInt4MatMulGELUApprox, ActGELUApprox},
		{"ReLU", BaseFusedInt4MatMulReLU, ActReLU},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actOutput := make([]float32, M*N)
			tc.fn(input, packed, scales, actOutput, M, K, N, groupSize)

			for i := range actOutput {
				expected := applyActivationScalar(rawOutput[i], tc.act)
				diff := math.Abs(float64(actOutput[i] - expected))
				tolerance := float32(1e-5)
				if tc.act == ActGELU {
					tolerance = 1e-4 // erf approximation needs more tolerance
				}
				if diff > float64(tolerance) {
					t.Errorf("Index %d: fused=%v, expected %s(raw)=%v (diff=%v)", i, actOutput[i], tc.name, expected, diff)
				}
			}
		})
	}
}

// TestExpf32Accuracy verifies the expf32 approximation accuracy.
func TestExpf32Accuracy(t *testing.T) {
	testValues := []float32{-80, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 80}

	for _, x := range testValues {
		got := expf32(x)
		want := float32(math.Exp(float64(x)))

		// For very small or very large values, check relative error
		if want != 0 {
			relErr := math.Abs(float64(got-want)) / math.Abs(float64(want))
			if relErr > 0.01 { // 1% relative error tolerance
				t.Errorf("expf32(%v) = %v, want %v (relErr=%v%%)", x, got, want, relErr*100)
			}
		} else {
			if got != 0 {
				t.Errorf("expf32(%v) = %v, want 0", x, got)
			}
		}
	}
}

// TestSigmoidf32Accuracy verifies the sigmoidf32 function accuracy.
func TestSigmoidf32Accuracy(t *testing.T) {
	testValues := []float32{-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10}

	for _, x := range testValues {
		got := sigmoidf32(x)
		want := float32(1.0 / (1.0 + math.Exp(-float64(x))))
		diff := math.Abs(float64(got - want))
		if diff > 1e-5 {
			t.Errorf("sigmoidf32(%v) = %v, want %v (diff=%v)", x, got, want, diff)
		}
	}
}
