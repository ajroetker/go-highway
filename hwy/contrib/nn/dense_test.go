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

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestDenseAuto(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	tests := []struct {
		name        string
		batchSize   int
		inFeatures  int
		outFeatures int
		useBias     bool
	}{
		{"1x4x4/bias", 1, 4, 4, true},
		{"1x4x4/no_bias", 1, 4, 4, false},
		{"2x8x4/bias", 2, 8, 4, true},
		{"4x16x8/bias", 4, 16, 8, true},
		{"3x7x5/bias", 3, 7, 5, true}, // non-aligned dimensions
		{"8x64x32/bias", 8, 64, 32, true},
		{"1x128x64/bias", 1, 128, 64, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := make([]float32, tt.batchSize*tt.inFeatures)
			weight := make([]float32, tt.outFeatures*tt.inFeatures)
			var bias []float32
			if tt.useBias {
				bias = make([]float32, tt.outFeatures)
				for i := range bias {
					bias[i] = float32(i) * 0.1
				}
			}

			for i := range x {
				x[i] = float32(i)*0.01 - 0.5
			}
			for i := range weight {
				weight[i] = float32(i)*0.005 - 0.25
			}

			autoOutput := make([]float32, tt.batchSize*tt.outFeatures)
			scalarOutput := make([]float32, tt.batchSize*tt.outFeatures)

			DenseAuto(pool, x, weight, bias, autoOutput, tt.batchSize, tt.inFeatures, tt.outFeatures)
			DenseScalar(x, weight, bias, scalarOutput, tt.batchSize, tt.inFeatures, tt.outFeatures)

			for i := range autoOutput {
				diff := stdmath.Abs(float64(autoOutput[i] - scalarOutput[i]))
				relTol := stdmath.Max(1e-4, 1e-4*stdmath.Abs(float64(scalarOutput[i])))
				if diff > relTol {
					t.Errorf("output[%d]: auto=%v, scalar=%v, diff=%v", i, autoOutput[i], scalarOutput[i], diff)
				}
			}
		})
	}
}

func TestDenseAuto64(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batchSize, inFeatures, outFeatures := 2, 16, 8

	x := make([]float64, batchSize*inFeatures)
	weight := make([]float64, outFeatures*inFeatures)
	bias := make([]float64, outFeatures)

	for i := range x {
		x[i] = float64(i)*0.01 - 0.5
	}
	for i := range weight {
		weight[i] = float64(i)*0.005 - 0.25
	}
	for i := range bias {
		bias[i] = float64(i) * 0.1
	}

	autoOutput := make([]float64, batchSize*outFeatures)
	scalarOutput := make([]float64, batchSize*outFeatures)

	DenseAuto(pool, x, weight, bias, autoOutput, batchSize, inFeatures, outFeatures)
	DenseScalar(x, weight, bias, scalarOutput, batchSize, inFeatures, outFeatures)

	for i := range autoOutput {
		if stdmath.Abs(autoOutput[i]-scalarOutput[i]) > 1e-10 {
			t.Errorf("output[%d]: auto=%v, scalar=%v", i, autoOutput[i], scalarOutput[i])
		}
	}
}

func TestBaseDenseScalarMatch(t *testing.T) {
	batchSize, inFeatures, outFeatures := 4, 32, 16

	x := make([]float32, batchSize*inFeatures)
	weight := make([]float32, outFeatures*inFeatures)
	bias := make([]float32, outFeatures)

	for i := range x {
		x[i] = float32(i)*0.01 - 0.5
	}
	for i := range weight {
		weight[i] = float32(i)*0.005 - 0.25
	}
	for i := range bias {
		bias[i] = float32(i) * 0.1
	}

	baseOutput := make([]float32, batchSize*outFeatures)
	scalarOutput := make([]float32, batchSize*outFeatures)

	Dense(x, weight, bias, baseOutput, batchSize, inFeatures, outFeatures)
	DenseScalar(x, weight, bias, scalarOutput, batchSize, inFeatures, outFeatures)

	for i := range baseOutput {
		diff := stdmath.Abs(float64(baseOutput[i] - scalarOutput[i]))
		relTol := stdmath.Max(1e-4, 1e-4*stdmath.Abs(float64(scalarOutput[i])))
		if diff > relTol {
			t.Errorf("Base[%d]=%v, scalar[%d]=%v, diff=%v", i, baseOutput[i], i, scalarOutput[i], diff)
		}
	}
}

func TestDenseActivationAuto(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batchSize, inFeatures, outFeatures := 2, 16, 8

	x := make([]float32, batchSize*inFeatures)
	weight := make([]float32, outFeatures*inFeatures)
	bias := make([]float32, outFeatures)

	for i := range x {
		x[i] = float32(i)*0.01 - 0.5
	}
	for i := range weight {
		weight[i] = float32(i)*0.005 - 0.25
	}
	for i := range bias {
		bias[i] = float32(i) * 0.1
	}

	activations := []struct {
		name string
		act  ActivationType
	}{
		{"None", ActivationNone},
		{"Gelu", ActivationGelu},
		{"Relu", ActivationRelu},
		{"Silu", ActivationSilu},
		{"Tanh", ActivationTanh},
	}

	for _, at := range activations {
		t.Run(at.name, func(t *testing.T) {
			output := make([]float32, batchSize*outFeatures)
			DenseActivationAuto(pool, x, weight, bias, output, batchSize, inFeatures, outFeatures, at.act)

			// Basic sanity: no NaN or Inf
			for i, v := range output {
				if stdmath.IsNaN(float64(v)) || stdmath.IsInf(float64(v), 0) {
					t.Errorf("output[%d] = %v (NaN/Inf)", i, v)
				}
			}

			// ActivationNone should match DenseAuto exactly
			if at.act == ActivationNone {
				expected := make([]float32, batchSize*outFeatures)
				DenseAuto(pool, x, weight, bias, expected, batchSize, inFeatures, outFeatures)
				for i := range output {
					if output[i] != expected[i] {
						t.Errorf("output[%d] = %v, want %v", i, output[i], expected[i])
					}
				}
			}

			// ReLU: all outputs >= 0
			if at.act == ActivationRelu {
				for i, v := range output {
					if v < 0 {
						t.Errorf("ReLU output[%d] = %v, want >= 0", i, v)
					}
				}
			}

			// Tanh: all outputs in [-1, 1]
			if at.act == ActivationTanh {
				for i, v := range output {
					if v < -1 || v > 1 {
						t.Errorf("Tanh output[%d] = %v, want in [-1, 1]", i, v)
					}
				}
			}
		})
	}
}

func BenchmarkDense(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	configs := []struct {
		batch, in, out int
	}{
		{1, 64, 64},
		{1, 256, 256},
		{1, 768, 768},
		{8, 768, 768},
		{32, 768, 3072},
	}

	for _, c := range configs {
		x := make([]float32, c.batch*c.in)
		weight := make([]float32, c.out*c.in)
		bias := make([]float32, c.out)
		output := make([]float32, c.batch*c.out)

		for i := range x {
			x[i] = float32(i) * 0.001
		}
		for i := range weight {
			weight[i] = float32(i) * 0.0005
		}
		for i := range bias {
			bias[i] = float32(i) * 0.01
		}

		label := fmt.Sprintf("b%d_%dx%d", c.batch, c.in, c.out)

		b.Run("Auto/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				DenseAuto(pool, x, weight, bias, output, c.batch, c.in, c.out)
			}
		})

		b.Run("Base/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Dense(x, weight, bias, output, c.batch, c.in, c.out)
			}
		})

		b.Run("Scalar/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				DenseScalar(x, weight, bias, output, c.batch, c.in, c.out)
			}
		})
	}
}
