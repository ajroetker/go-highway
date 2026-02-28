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

package gguf

import (
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// GGUFMatMul computes output = input @ weights^T where weights are in native
// GGUF quantized format. Weights are never dequantized to float32.
//
// Parameters:
//   - input: [M, K] float32 activations (row-major)
//   - weights: [N, K] GGUF-quantized weight data. Each of the N rows is
//     (K / ValuesPerBlock(qt)) contiguous blocks in the given format.
//   - output: [M, N] float32 output (row-major, pre-allocated)
//   - M: number of input rows (batch size * sequence length)
//   - K: hidden dimension (must be a multiple of ValuesPerBlock(qt))
//   - N: output dimension (number of output neurons)
//   - qt: quantization type of the weight data
//
// Activations are quantized to Q8_0 (for Tier 1 types) or Q8_K (for K-quant
// types) on-the-fly. The temporary activation buffer is allocated internally.
func GGUFMatMul(input []float32, weights []uint8, output []float32, M, K, N int, qt QuantType) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	valsPerBlock := ValuesPerBlock(qt)
	nblocks := K / valsPerBlock
	wBlockBytes := BytesPerBlock(qt)
	aBlockBytes := ActivationBlockSize(qt)
	wRowBytes := nblocks * wBlockBytes
	aRowBytes := nblocks * aBlockBytes

	vecDot := getVecDot(qt)
	quantize := getQuantize(qt)

	// Quantize all input rows to activation format.
	qInput := make([]uint8, M*aRowBytes)
	for m := range M {
		quantize(input[m*K:(m+1)*K], qInput[m*aRowBytes:(m+1)*aRowBytes])
	}

	// Compute output[m, n] = vecDot(weights[n], qInput[m], nblocks).
	for m := range M {
		aRow := qInput[m*aRowBytes : (m+1)*aRowBytes]
		for n := range N {
			wRow := weights[n*wRowBytes : (n+1)*wRowBytes]
			output[m*N+n] = vecDot(wRow, aRow, nblocks)
		}
	}
}

// ParallelGGUFMatMul is the parallel version of GGUFMatMul, distributing
// M rows across workers using the provided pool.
func ParallelGGUFMatMul(pool workerpool.Executor, input []float32, weights []uint8, output []float32, M, K, N int, qt QuantType) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	valsPerBlock := ValuesPerBlock(qt)
	nblocks := K / valsPerBlock
	wBlockBytes := BytesPerBlock(qt)
	aBlockBytes := ActivationBlockSize(qt)
	wRowBytes := nblocks * wBlockBytes
	aRowBytes := nblocks * aBlockBytes

	vecDot := getVecDot(qt)
	quantize := getQuantize(qt)

	// Quantize all input rows in parallel.
	qInput := make([]uint8, M*aRowBytes)
	pool.ParallelFor(M, func(mStart, mEnd int) {
		for m := mStart; m < mEnd; m++ {
			quantize(input[m*K:(m+1)*K], qInput[m*aRowBytes:(m+1)*aRowBytes])
		}
	})

	// Compute matmul in parallel across M rows.
	pool.ParallelFor(M, func(mStart, mEnd int) {
		for m := mStart; m < mEnd; m++ {
			aRow := qInput[m*aRowBytes : (m+1)*aRowBytes]
			for n := range N {
				wRow := weights[n*wRowBytes : (n+1)*wRowBytes]
				output[m*N+n] = vecDot(wRow, aRow, nblocks)
			}
		}
	})
}

// getVecDot returns the dispatched vec_dot function for the given quant type.
func getVecDot(qt QuantType) func(wdata, adata []uint8, nblocks int) float32 {
	switch qt {
	case TypeQ4_0:
		return VecDotQ4_0Q8_0
	case TypeQ8_0:
		return VecDotQ8_0Q8_0
	default:
		return nil
	}
}

// getQuantize returns the dispatched quantization function for the given weight type.
func getQuantize(qt QuantType) func(input []float32, output []uint8) {
	switch qt {
	case TypeQ4_0, TypeQ8_0, TypeIQ4NL:
		return QuantizeQ8_0
	default:
		return nil
	}
}
