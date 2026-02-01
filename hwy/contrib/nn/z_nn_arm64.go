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

//go:build !noasm && arm64

// NOTE: This file is named "z_nn_arm64.go" (starting with 'z')
// to ensure its init() runs AFTER the generated dispatch files.
// Go executes init() functions in lexicographic filename order within a package.
// The generated dispatch sets LayerNorm* etc. to hwygen-generated fallback
// implementations; this file's init() must run afterward to override
// with optimized NEON implementations when available.

package nn

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/nn/asm"
)

// Minimum normSize to use NEON vectorization.
// Below this, the overhead of NEON setup outweighs the benefit.
const minNormSizeForNEON = 8

// layerNormNEONF32 uses GOAT-generated NEON assembly for f32 layer normalization.
func layerNormNEONF32(input, output []float32, normSize int, gamma, beta []float32, epsilon float32) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}

	// Fall back to hwygen-generated code for small normSize
	if normSize < minNormSizeForNEON {
		BaseLayerNorm(input, output, normSize, gamma, beta, epsilon)
		return
	}

	if gamma != nil && beta != nil {
		asm.LayerNormNEONF32(input, output, gamma, beta, size, normSize, epsilon)
	} else {
		asm.LayerNormNEONF32NoAffine(input, output, size, normSize, epsilon)
	}
}

// layerNormNEONF64 uses GOAT-generated NEON assembly for f64 layer normalization.
func layerNormNEONF64(input, output []float64, normSize int, gamma, beta []float64, epsilon float64) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}

	if normSize < minNormSizeForNEON {
		BaseLayerNorm(input, output, normSize, gamma, beta, epsilon)
		return
	}

	if gamma != nil && beta != nil {
		asm.LayerNormNEONF64(input, output, gamma, beta, size, normSize, epsilon)
	} else {
		asm.LayerNormNEONF64NoAffine(input, output, size, normSize, epsilon)
	}
}

// Minimum size to use NEON softmax vectorization.
const minSizeForNEONSoftmax = 8

// softmaxNEONF32 uses GOAT-generated NEON assembly for f32 softmax.
func softmaxNEONF32(input, output []float32) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEONSoftmax {
		BaseSoftmax(input, output)
		return
	}
	asm.SoftmaxNeonF32(input, output, size)
}

// softmaxNEONF64 uses GOAT-generated NEON assembly for f64 softmax.
func softmaxNEONF64(input, output []float64) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEONSoftmax {
		BaseSoftmax(input, output)
		return
	}
	asm.SoftmaxNeonF64(input, output, size)
}

// Minimum dimensions for NEON/SME SDPA acceleration.
const minDimForSDPANEON = 8
const minDimForSDPASME = 32

// sdpaNEONF32 uses GOAT-generated NEON assembly for f32 SDPA.
func sdpaNEONF32(q, k, v, mask, scores, output []float32, seqLen, kvLen, headDim int, scale float32) {
	if seqLen < minDimForSDPANEON || kvLen < minDimForSDPANEON {
		BaseSDPA(q, k, v, mask, scores, output, seqLen, kvLen, headDim, scale)
		return
	}
	asm.SDPANeonF32(q, k, v, mask, scores, output, seqLen, kvLen, headDim, scale)
}

// sdpaNEONF64 uses GOAT-generated NEON assembly for f64 SDPA.
func sdpaNEONF64(q, k, v, mask, scores, output []float64, seqLen, kvLen, headDim int, scale float64) {
	if seqLen < minDimForSDPANEON || kvLen < minDimForSDPANEON {
		BaseSDPA(q, k, v, mask, scores, output, seqLen, kvLen, headDim, scale)
		return
	}
	asm.SDPANeonF64(q, k, v, mask, scores, output, seqLen, kvLen, headDim, scale)
}

// sdpaCausalNEONF32 uses GOAT-generated NEON assembly for f32 causal SDPA.
func sdpaCausalNEONF32(q, k, v, scores, output []float32, seqLen, kvLen, headDim int, scale float32) {
	if seqLen < minDimForSDPANEON || kvLen < minDimForSDPANEON {
		BaseSDPACausal(q, k, v, scores, output, seqLen, kvLen, headDim, scale)
		return
	}
	asm.SDPACausalNeonF32(q, k, v, scores, output, seqLen, kvLen, headDim, scale)
}

// sdpaCausalNEONF64 uses GOAT-generated NEON assembly for f64 causal SDPA.
func sdpaCausalNEONF64(q, k, v, scores, output []float64, seqLen, kvLen, headDim int, scale float64) {
	if seqLen < minDimForSDPANEON || kvLen < minDimForSDPANEON {
		BaseSDPACausal(q, k, v, scores, output, seqLen, kvLen, headDim, scale)
		return
	}
	asm.SDPACausalNeonF64(q, k, v, scores, output, seqLen, kvLen, headDim, scale)
}

// qkvdenseNEONF32 uses GOAT-generated NEON assembly for f32 QKV projection.
func qkvdenseNEONF32(x, wQKV, biasQ, biasK, biasV, q, k, v []float32, batchSize, inFeatures, qDim, kvDim int) {
	asm.QKVDenseNEONF32(x, wQKV, biasQ, biasK, biasV, q, k, v, batchSize, inFeatures, qDim, kvDim)
}

// qkvdenseNEONF64 uses GOAT-generated NEON assembly for f64 QKV projection.
func qkvdenseNEONF64(x, wQKV, biasQ, biasK, biasV, q, k, v []float64, batchSize, inFeatures, qDim, kvDim int) {
	asm.QKVDenseNEONF64(x, wQKV, biasQ, biasK, biasV, q, k, v, batchSize, inFeatures, qDim, kvDim)
}

// =============================================================================
// SME SDPA adapter functions
// =============================================================================

// sdpaSMEF32 decomposes SDPA into multi-tile FMOPA matmul + Go softmax.
// Step 1: scores = Q @ K^T (via MatMulKLast, which uses multi-tile FMOPA)
// Step 2: scale scores, add mask, row-wise softmax (in Go)
// Step 3: output = scores @ V (via BlockedMatMul, which uses multi-tile FMOPA)
func sdpaSMEF32(q, k, v, mask, scores, output []float32, seqLen, kvLen, headDim int, scale float32) {
	if seqLen%16 != 0 || kvLen%16 != 0 || headDim%16 != 0 ||
		seqLen < minDimForSDPASME || kvLen < minDimForSDPASME || headDim < minDimForSDPASME {
		sdpaNEONF32(q, k, v, mask, scores, output, seqLen, kvLen, headDim, scale)
		return
	}

	matmul.MatMulKLastFloat32(q, k, scores, seqLen, kvLen, headDim)
	scaleMaskSoftmax(scores, mask, seqLen, kvLen, scale)
	matmul.BlockedMatMulFloat32(scores, v, output, seqLen, headDim, kvLen)
}

// sdpaSMEF64 decomposes SDPA into multi-tile FMOPA matmul + Go softmax (float64).
func sdpaSMEF64(q, k, v, mask, scores, output []float64, seqLen, kvLen, headDim int, scale float64) {
	if seqLen%8 != 0 || kvLen%8 != 0 || headDim%8 != 0 ||
		seqLen < minDimForSDPASME || kvLen < minDimForSDPASME || headDim < minDimForSDPASME {
		sdpaNEONF64(q, k, v, mask, scores, output, seqLen, kvLen, headDim, scale)
		return
	}

	matmul.MatMulKLastFloat64(q, k, scores, seqLen, kvLen, headDim)
	scaleMaskSoftmax(scores, mask, seqLen, kvLen, scale)
	matmul.BlockedMatMulFloat64(scores, v, output, seqLen, headDim, kvLen)
}

// scaleMaskSoftmax scales scores, adds mask, and applies row-wise softmax in-place.
func scaleMaskSoftmax[T hwy.Floats](scores, mask []T, seqLen, kvLen int, scale T) {
	for i := 0; i < seqLen; i++ {
		row := scores[i*kvLen : (i+1)*kvLen]

		if mask != nil {
			mRow := mask[i*kvLen : (i+1)*kvLen]
			for j := range row {
				row[j] = row[j]*scale + mRow[j]
			}
		} else {
			for j := range row {
				row[j] *= scale
			}
		}

		SoftmaxInPlace(row)
	}
}

// =============================================================================
// SME QKVDense adapter functions
// =============================================================================

// qkvdenseSMEF32 decomposes QKV projection into 3 separate MatMulKLast calls,
// one per projection (Q, K, V). Each call handles its own incremental transpose,
// eliminating the O(inFeatures * totalOut) wQKV transpose buffer.
func qkvdenseSMEF32(x, wQKV, biasQ, biasK, biasV, q, k, v []float32, batchSize, inFeatures, qDim, kvDim int) {
	// wQKV is [totalOut, inFeatures] laid out as [wQ; wK; wV] row-major.
	// MatMulKLast(a, b, c, m, n, k) computes C = A @ B^T.
	// Q: x[batchSize, inFeatures] @ wQ[qDim, inFeatures]^T → q[batchSize, qDim]
	wQ := wQKV[:qDim*inFeatures]
	matmul.MatMulKLastFloat32(x, wQ, q, batchSize, qDim, inFeatures)
	if biasQ != nil {
		addBias(q, biasQ, batchSize, qDim)
	}

	// K: x @ wK^T → k[batchSize, kvDim]
	wK := wQKV[qDim*inFeatures : (qDim+kvDim)*inFeatures]
	matmul.MatMulKLastFloat32(x, wK, k, batchSize, kvDim, inFeatures)
	if biasK != nil {
		addBias(k, biasK, batchSize, kvDim)
	}

	// V: x @ wV^T → v[batchSize, kvDim]
	wV := wQKV[(qDim+kvDim)*inFeatures:]
	matmul.MatMulKLastFloat32(x, wV, v, batchSize, kvDim, inFeatures)
	if biasV != nil {
		addBias(v, biasV, batchSize, kvDim)
	}
}

// qkvdenseSMEF64 decomposes QKV projection into 3 separate MatMulKLast calls (float64).
func qkvdenseSMEF64(x, wQKV, biasQ, biasK, biasV, q, k, v []float64, batchSize, inFeatures, qDim, kvDim int) {
	wQ := wQKV[:qDim*inFeatures]
	matmul.MatMulKLastFloat64(x, wQ, q, batchSize, qDim, inFeatures)
	if biasQ != nil {
		addBias(q, biasQ, batchSize, qDim)
	}

	wK := wQKV[qDim*inFeatures : (qDim+kvDim)*inFeatures]
	matmul.MatMulKLastFloat64(x, wK, k, batchSize, kvDim, inFeatures)
	if biasK != nil {
		addBias(k, biasK, batchSize, kvDim)
	}

	wV := wQKV[(qDim+kvDim)*inFeatures:]
	matmul.MatMulKLastFloat64(x, wV, v, batchSize, kvDim, inFeatures)
	if biasV != nil {
		addBias(v, biasV, batchSize, kvDim)
	}
}

func init() {
	if hwy.NoSimdEnv() {
		return
	}

	// Override LayerNorm dispatch with GOAT NEON implementations
	LayerNormFloat32 = layerNormNEONF32
	LayerNormFloat64 = layerNormNEONF64

	// Override Softmax dispatch with GOAT NEON implementations
	SoftmaxFloat32 = softmaxNEONF32
	SoftmaxFloat64 = softmaxNEONF64

	// Override SDPA and QKVDense dispatch
	if hwy.HasSME() {
		// SME FMOPA provides higher throughput for aligned dimensions.
		// The SME adapters check alignment and fall back to NEON internally.
		SDPAFloat32 = sdpaSMEF32
		SDPAFloat64 = sdpaSMEF64
		QKVDenseFloat32 = qkvdenseSMEF32
		QKVDenseFloat64 = qkvdenseSMEF64
	} else {
		SDPAFloat32 = sdpaNEONF32
		SDPAFloat64 = sdpaNEONF64
		QKVDenseFloat32 = qkvdenseNEONF32
		QKVDenseFloat64 = qkvdenseNEONF64
	}

	// Causal SDPA stays on NEON (no SME causal kernel exists)
	SDPACausalFloat32 = sdpaCausalNEONF32
	SDPACausalFloat64 = sdpaCausalNEONF64

	// Float16/BFloat16 use the hwygen-generated promoted implementations
	// (promote to f32, compute, demote) which are already efficient enough
	// since the promotion is the bottleneck, not the compute.
}
