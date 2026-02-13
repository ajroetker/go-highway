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

//go:generate go run ../../../cmd/hwygen -input matmul_fused_int8.go -dispatch fusedint8matmul -output . -targets avx2,avx512,neon:asm,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseFusedInt8MatMul performs fused Int8 dequantization + matrix multiplication + optional bias.
// output[m,n] = sum_k(input[m,k] * (weights[k,n] * scale[k,groupIdx])) + bias[n]
//
// Int8 quantization stores weights as signed 8-bit integers with per-group scales.
// This is more memory efficient than float32 (4x compression) while maintaining
// good accuracy for many model weights.
//
// Parameters:
//   - input: [M, K] float32 input matrix (row-major)
//   - weights: [K, N] int8 quantized weights (row-major)
//   - scales: [K, numGroups] float32 per-group scales
//   - bias: [N] float32 bias vector (nil for no bias)
//   - output: [M, N] float32 output matrix (row-major, pre-allocated)
//   - M, K, N: matrix dimensions
//   - groupSize: number of columns per scale group
func BaseFusedInt8MatMul(input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()

	// Temporary buffer for dequantized weights (one vector width)
	dequantBuf := make([]float32, lanes)

	// Process each output row
	for m := 0; m < M; m++ {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		// Process output columns in groups of lanes
		var n int
		for n = 0; n+lanes <= N; n += lanes {
			// Initialize accumulator
			acc := hwy.Zero[float32]()

			// Accumulate over K dimension
			for k := 0; k < K; k++ {
				// Broadcast input[m, k]
				inputVal := hwy.Set(inputRow[k])

				// Dequantize 'lanes' weights from weights[k, n:n+lanes]
				baseIdx := k * N
				scaleBase := k * numGroups

				for lane := 0; lane < lanes; lane++ {
					colIdx := n + lane
					weightIdx := baseIdx + colIdx

					// Int8 weight value
					val := float32(weights[weightIdx])

					groupIdx := colIdx / groupSize
					scale := scales[scaleBase+groupIdx]
					dequantBuf[lane] = val * scale
				}

				// Load dequantized weights into vector
				dequantWeights := hwy.Load(dequantBuf)

				// FMA: acc += input * weight
				acc = hwy.MulAdd(inputVal, dequantWeights, acc)
			}

			// Add bias
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}

			// Store result
			hwy.Store(acc, outputRow[n:])
		}

		// Handle remaining columns (scalar tail)
		for ; n < N; n++ {
			groupIdx := n / groupSize
			sum := float32(0)
			for k := 0; k < K; k++ {
				weightIdx := k*N + n
				val := float32(weights[weightIdx])
				scale := scales[k*numGroups+groupIdx]
				weight := val * scale
				sum += inputRow[k] * weight
			}
			if bias != nil {
				sum += bias[n]
			}
			outputRow[n] = sum
		}
	}
}
