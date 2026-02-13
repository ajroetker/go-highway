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

	// Accumulator buffer for one output row — fits L1 for typical N
	accBuf := make([]float32, N)

	// Process each output row
	for m := 0; m < M; m++ {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		// Zero accumulators for this row
		for i := 0; i < N; i++ {
			accBuf[i] = 0
		}

		// K-outer, N-inner: sequential weight access, single input broadcast per k
		for k := 0; k < K; k++ {
			inputVal := hwy.Set(inputRow[k])
			baseIdx := k * N
			scaleBase := k * numGroups

			// Vectorized N sweep
			var n int
			for n = 0; n+lanes <= N; n += lanes {
				for lane := 0; lane < lanes; lane++ {
					colIdx := n + lane
					weightIdx := baseIdx + colIdx

					val := float32(weights[weightIdx])

					groupIdx := colIdx / groupSize
					scale := scales[scaleBase+groupIdx]
					dequantBuf[lane] = val * scale
				}

				dequantWeights := hwy.Load(dequantBuf)
				acc := hwy.Load(accBuf[n:])
				acc = hwy.MulAdd(inputVal, dequantWeights, acc)
				hwy.Store(acc, accBuf[n:])
			}

			// Scalar tail
			for ; n < N; n++ {
				weightIdx := baseIdx + n
				val := float32(weights[weightIdx])
				groupIdx := n / groupSize
				scale := scales[scaleBase+groupIdx]
				accBuf[n] += inputRow[k] * val * scale
			}
		}

		// Apply bias and store to output
		var n int
		for n = 0; n+lanes <= N; n += lanes {
			acc := hwy.Load(accBuf[n:])
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}
			hwy.Store(acc, outputRow[n:])
		}
		for ; n < N; n++ {
			val := accBuf[n]
			if bias != nil {
				val += bias[n]
			}
			outputRow[n] = val
		}
	}
}
