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

package nn

import (
	"fmt"
	stdmath "math"
	"testing"
)

func TestLayerNorm(t *testing.T) {
	tests := []struct {
		name     string
		normSize int
		useGamma bool
		useBeta  bool
	}{
		{"normSize=4/no_affine", 4, false, false},
		{"normSize=4/with_affine", 4, true, true},
		{"normSize=8/no_affine", 8, false, false},
		{"normSize=8/with_affine", 8, true, true},
		{"normSize=16/with_affine", 16, true, true},
		{"normSize=64/with_affine", 64, true, true},
		{"normSize=256/with_affine", 256, true, true},
		{"normSize=4/gamma_only", 4, true, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			numGroups := 3
			size := numGroups * tt.normSize
			input := make([]float32, size)
			for i := range input {
				input[i] = float32(i)*0.1 - float32(size)*0.05
			}

			var gamma, beta []float32
			if tt.useGamma {
				gamma = make([]float32, tt.normSize)
				for i := range gamma {
					gamma[i] = 1.0 + float32(i)*0.01
				}
			}
			if tt.useBeta {
				beta = make([]float32, tt.normSize)
				for i := range beta {
					beta[i] = float32(i) * 0.005
				}
			}

			output := make([]float32, size)
			LayerNorm(input, output, tt.normSize, gamma, beta, 1e-5)

			// Verify normalized outputs per group
			for g := range numGroups {
				off := g * tt.normSize

				if !tt.useGamma && !tt.useBeta {
					// Without affine, output should have mean~0, variance~1
					var mean float64
					for i := 0; i < tt.normSize; i++ {
						mean += float64(output[off+i])
					}
					mean /= float64(tt.normSize)

					if stdmath.Abs(mean) > 1e-4 {
						t.Errorf("group %d: mean = %v, want ~0", g, mean)
					}

					var variance float64
					for i := 0; i < tt.normSize; i++ {
						diff := float64(output[off+i]) - mean
						variance += diff * diff
					}
					variance /= float64(tt.normSize)

					if stdmath.Abs(variance-1.0) > 1e-3 {
						t.Errorf("group %d: variance = %v, want ~1", g, variance)
					}
				}
			}
		})
	}
}

func TestLayerNorm64(t *testing.T) {
	normSize := 16
	numGroups := 2
	size := numGroups * normSize
	input := make([]float64, size)
	for i := range input {
		input[i] = float64(i)*0.1 - float64(size)*0.05
	}

	output := make([]float64, size)
	LayerNorm(input, output, normSize, nil, nil, 1e-5)

	for g := range numGroups {
		off := g * normSize

		var mean float64
		for i := range normSize {
			mean += output[off+i]
		}
		mean /= float64(normSize)

		if stdmath.Abs(mean) > 1e-6 {
			t.Errorf("group %d: mean = %v, want ~0", g, mean)
		}

		var variance float64
		for i := range normSize {
			diff := output[off+i] - mean
			variance += diff * diff
		}
		variance /= float64(normSize)

		if stdmath.Abs(variance-1.0) > 1e-4 {
			t.Errorf("group %d: variance = %v, want ~1", g, variance)
		}
	}
}

func TestLayerNormScalarMatch(t *testing.T) {
	normSize := 64
	numGroups := 4
	size := numGroups * normSize

	input := make([]float32, size)
	for i := range input {
		input[i] = float32(i)*0.1 - float32(size)*0.05
	}

	gamma := make([]float32, normSize)
	beta := make([]float32, normSize)
	for i := range gamma {
		gamma[i] = 1.0 + float32(i)*0.01
		beta[i] = float32(i) * 0.005
	}

	simdOutput := make([]float32, size)
	scalarOutput := make([]float32, size)

	LayerNorm(input, simdOutput, normSize, gamma, beta, 1e-5)
	LayerNormScalar(input, scalarOutput, normSize, gamma, beta, 1e-5)

	for i := range simdOutput {
		if stdmath.Abs(float64(simdOutput[i]-scalarOutput[i])) > 1e-4 {
			t.Errorf("SIMD[%d] = %v, scalar[%d] = %v, mismatch", i, simdOutput[i], i, scalarOutput[i])
		}
	}
}

func TestLayerNormEmpty(t *testing.T) {
	// Should not panic
	LayerNorm[float32](nil, nil, 4, nil, nil, 1e-5)
	LayerNorm([]float32{}, []float32{}, 4, nil, nil, 1e-5)
}

func BenchmarkLayerNorm(b *testing.B) {
	sizes := []int{64, 256, 768, 1024}

	for _, normSize := range sizes {
		numGroups := 32
		size := numGroups * normSize

		input := make([]float32, size)
		output := make([]float32, size)
		gamma := make([]float32, normSize)
		beta := make([]float32, normSize)
		for i := range input {
			input[i] = float32(i) * 0.01
		}
		for i := range gamma {
			gamma[i] = 1.0
			beta[i] = 0.0
		}

		b.Run(fmt.Sprintf("SIMD/normSize=%d", normSize), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				LayerNorm(input, output, normSize, gamma, beta, 1e-5)
			}
		})

		b.Run(fmt.Sprintf("Scalar/normSize=%d", normSize), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				LayerNormScalar(input, output, normSize, gamma, beta, 1e-5)
			}
		})
	}
}
