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
	tileN := 4 * lanes

	// Temporary buffer for dequantized weights (4 vector widths)
	dequantBuf := make([]float32, tileN)

	// Accumulator buffer for one output row — fits L1 for typical N
	accBuf := make([]float32, N)

	// Process each output row
	for m := range M {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		// Zero accumulators for this row
		for i := range N {
			accBuf[i] = 0
		}

		// K-outer, N-inner: sequential weight access, single input broadcast per k
		for k := range K {
			inputVal := hwy.Set(inputRow[k])
			baseIdx := k * N
			scaleBase := k * numGroups

			// Tiled N sweep — 4 vectors at a time for ILP
			var n int
			for n = 0; n+tileN <= N; n += tileN {
				for lane := range tileN {
					colIdx := n + lane
					weightIdx := baseIdx + colIdx
					val := float32(weights[weightIdx])
					groupIdx := colIdx / groupSize
					scale := scales[scaleBase+groupIdx]
					dequantBuf[lane] = val * scale
				}

				w0 := hwy.Load(dequantBuf[0:])
				w1 := hwy.Load(dequantBuf[lanes:])
				w2 := hwy.Load(dequantBuf[2*lanes:])
				w3 := hwy.Load(dequantBuf[3*lanes:])
				acc0 := hwy.Load(accBuf[n:])
				acc1 := hwy.Load(accBuf[n+lanes:])
				acc2 := hwy.Load(accBuf[n+2*lanes:])
				acc3 := hwy.Load(accBuf[n+3*lanes:])
				acc0 = hwy.MulAdd(inputVal, w0, acc0)
				acc1 = hwy.MulAdd(inputVal, w1, acc1)
				acc2 = hwy.MulAdd(inputVal, w2, acc2)
				acc3 = hwy.MulAdd(inputVal, w3, acc3)
				hwy.Store(acc0, accBuf[n:])
				hwy.Store(acc1, accBuf[n+lanes:])
				hwy.Store(acc2, accBuf[n+2*lanes:])
				hwy.Store(acc3, accBuf[n+3*lanes:])
			}

			// Remainder: single vector
			for ; n+lanes <= N; n += lanes {
				for lane := range lanes {
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
