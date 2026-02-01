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
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// DenseAuto computes a dense (fully-connected) layer using the best available
// matmul implementation: output = x @ weight^T + bias.
//
//   - x is [batchSize, inFeatures] (row-major)
//   - weight is [outFeatures, inFeatures] (row-major, PyTorch format)
//   - bias is [outFeatures] (optional, pass nil to skip)
//   - output is [batchSize, outFeatures] (row-major)
//
// This delegates to MatMulKLastAuto (which dispatches to SME/NEON/AVX as
// appropriate) and then adds bias with SIMD.
func DenseAuto[T hwy.Floats](pool *workerpool.Pool, x, weight, bias, output []T, batchSize, inFeatures, outFeatures int) {
	// Matmul: output = x @ weight^T
	matmul.MatMulKLastAuto(pool, x, weight, output, batchSize, outFeatures, inFeatures)

	// Bias add
	if bias != nil {
		addBias(output, bias, batchSize, outFeatures)
	}
}

// DenseActivationAuto computes a dense layer followed by an activation function.
//
// This is equivalent to:
//
//	DenseAuto(x, weight, bias, output, batchSize, inFeatures, outFeatures)
//	applyActivation(output, act, batchSize*outFeatures)
//
// The activation is applied in-place on the output after the dense computation.
func DenseActivationAuto[T hwy.Floats](pool *workerpool.Pool, x, weight, bias, output []T, batchSize, inFeatures, outFeatures int, act ActivationType) {
	DenseAuto(pool, x, weight, bias, output, batchSize, inFeatures, outFeatures)

	if act != ActivationNone {
		applyActivationInPlace(output[:batchSize*outFeatures], act)
	}
}

// addBias adds bias[j] to output[i*outFeatures+j] for all i using SIMD.
func addBias[T hwy.Floats](output, bias []T, batchSize, outFeatures int) {
	lanes := hwy.MaxLanes[T]()

	for i := range batchSize {
		off := i * outFeatures
		j := 0
		for ; j+lanes <= outFeatures; j += lanes {
			o := hwy.LoadFull(output[off+j:])
			b := hwy.LoadFull(bias[j:])
			hwy.StoreFull(hwy.Add(o, b), output[off+j:])
		}
		for ; j < outFeatures; j++ {
			output[off+j] += bias[j]
		}
	}
}

// applyActivationInPlace applies the given activation function in-place.
func applyActivationInPlace[T hwy.Floats](data []T, act ActivationType) {
	switch act {
	case ActivationGelu:
		activation.GELU(data, data)
	case ActivationRelu:
		activation.ReLU(data, data)
	case ActivationSilu:
		activation.SiLU(data, data)
	case ActivationTanh:
		activation.Tanh(data, data)
	}
}

// DenseScalar is a scalar reference implementation for comparison and testing.
func DenseScalar[T hwy.Floats](x, weight, bias, output []T, batchSize, inFeatures, outFeatures int) {
	for i := range batchSize {
		xOff := i * inFeatures
		oOff := i * outFeatures

		for j := range outFeatures {
			wOff := j * inFeatures
			var sum float64
			for p := range inFeatures {
				sum += float64(x[xOff+p]) * float64(weight[wOff+p])
			}
			if bias != nil {
				sum += float64(bias[j])
			}
			output[oOff+j] = T(sum)
		}
	}
}
