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
	"fmt"

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

	// SME SMOPA/SUMOPA path for K-quant types.
	if SMEGGUFMatMul != nil && isKQuant(qt) {
		SMEGGUFMatMul(input, weights, output, M, K, N, qt)
		return
	}

	valsPerBlock := ValuesPerBlock(qt)
	if K%valsPerBlock != 0 {
		panic(fmt.Sprintf("gguf: K=%d is not a multiple of ValuesPerBlock=%d for QuantType %d", K, valsPerBlock, qt))
	}

	vecDot := getVecDot(qt)
	quantize := getQuantize(qt)
	if vecDot == nil || quantize == nil {
		panic(fmt.Sprintf("gguf: unsupported QuantType %d for matmul", qt))
	}

	nblocks := K / valsPerBlock
	wBlockBytes := BytesPerBlock(qt)
	aBlockBytes := ActivationBlockSize(qt)
	wRowBytes := nblocks * wBlockBytes
	aRowBytes := nblocks * aBlockBytes

	// Fused quantize+compute per row to avoid allocating the full M*aRowBytes
	// intermediate buffer and to keep the quantized row in cache.
	qRow := make([]uint8, aRowBytes)
	for m := range M {
		quantize(input[m*K:(m+1)*K], qRow)
		for n := range N {
			wRow := weights[n*wRowBytes : (n+1)*wRowBytes]
			output[m*N+n] = vecDot(wRow, qRow, nblocks)
		}
	}
}

// ParallelGGUFMatMul is the parallel version of GGUFMatMul, distributing
// M rows across workers using the provided pool.
func ParallelGGUFMatMul(pool workerpool.Executor, input []float32, weights []uint8, output []float32, M, K, N int, qt QuantType) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	// SME SMOPA/SUMOPA path for K-quant types.
	if SMEParallelGGUFMatMul != nil && isKQuant(qt) {
		SMEParallelGGUFMatMul(pool, input, weights, output, M, K, N, qt)
		return
	}

	valsPerBlock := ValuesPerBlock(qt)
	if K%valsPerBlock != 0 {
		panic(fmt.Sprintf("gguf: K=%d is not a multiple of ValuesPerBlock=%d for QuantType %d", K, valsPerBlock, qt))
	}

	vecDot := getVecDot(qt)
	quantize := getQuantize(qt)
	if vecDot == nil || quantize == nil {
		panic(fmt.Sprintf("gguf: unsupported QuantType %d for matmul", qt))
	}

	nblocks := K / valsPerBlock
	wBlockBytes := BytesPerBlock(qt)
	aBlockBytes := ActivationBlockSize(qt)
	wRowBytes := nblocks * wBlockBytes
	aRowBytes := nblocks * aBlockBytes

	// Fused quantize+compute per worker chunk. Each worker allocates its own
	// qRow buffer to avoid contention.
	pool.ParallelFor(M, func(mStart, mEnd int) {
		qRow := make([]uint8, aRowBytes)
		for m := mStart; m < mEnd; m++ {
			quantize(input[m*K:(m+1)*K], qRow)
			for n := range N {
				wRow := weights[n*wRowBytes : (n+1)*wRowBytes]
				output[m*N+n] = vecDot(wRow, qRow, nblocks)
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
	case TypeQ2_K:
		return VecDotQ2_KQ8_K
	case TypeQ3_K:
		return VecDotQ3_KQ8_K
	case TypeQ4_K:
		return VecDotQ4_KQ8_K
	case TypeQ5_K:
		return VecDotQ5_KQ8_K
	case TypeQ6_K:
		return VecDotQ6_KQ8_K
	default:
		return nil
	}
}

// getQuantize returns the dispatched quantization function for the given weight type.
func getQuantize(qt QuantType) func(input []float32, output []uint8) {
	switch qt {
	case TypeQ4_0, TypeQ8_0, TypeIQ4NL:
		return QuantizeQ8_0
	case TypeQ2_K, TypeQ3_K, TypeQ4_K, TypeQ5_K, TypeQ6_K:
		return QuantizeQ8_K
	default:
		return nil
	}
}

// PreparedGGUFMatMul computes output = input @ weights^T using pre-packed
// weights for maximum throughput. The 4-tile kernel processes 64 output
// columns per kernel call, eliminating per-inference B-panel packing.
//
// Parameters:
//   - input: [M, K] float32 activations (row-major)
//   - pw: pre-packed weights from PrepareWeights
//   - output: [M, pw.N] float32 output (row-major, pre-allocated)
//   - M: number of input rows (batch size * sequence length)
//
// Panics if SME hardware is not available. Use PrepareWeights to create pw.
func PreparedGGUFMatMul(input []float32, pw *PreparedWeights, output []float32, M int) {
	if M == 0 || pw == nil {
		return
	}
	if SMEPreparedGGUFMatMul == nil {
		panic("gguf: PreparedGGUFMatMul requires SME hardware")
	}
	SMEPreparedGGUFMatMul(input, pw, output, M)
}

// ParallelPreparedGGUFMatMul is the parallel version of PreparedGGUFMatMul,
// distributing N-tile-groups across workers using the provided pool.
func ParallelPreparedGGUFMatMul(pool workerpool.Executor, input []float32,
	pw *PreparedWeights, output []float32, M int) {
	if M == 0 || pw == nil {
		return
	}
	if SMEParallelPreparedGGUFMatMul == nil {
		panic("gguf: ParallelPreparedGGUFMatMul requires SME hardware")
	}
	SMEParallelPreparedGGUFMatMul(pool, input, pw, output, M)
}

// isKQuant returns true if the quant type is a K-quant type (uses Q8_K activations).
func isKQuant(qt QuantType) bool {
	switch qt {
	case TypeQ2_K, TypeQ3_K, TypeQ4_K, TypeQ5_K, TypeQ6_K:
		return true
	default:
		return false
	}
}
