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

// qkvlinearNEONF32 uses GOAT-generated NEON assembly for f32 QKV projection.
func qkvlinearNEONF32(x, wQKV, biasQ, biasK, biasV, q, k, v []float32, batchSize, inFeatures, qDim, kvDim int) {
	asm.QKVLinearNEONF32(x, wQKV, biasQ, biasK, biasV, q, k, v, batchSize, inFeatures, qDim, kvDim)
}

// qkvlinearNEONF64 uses GOAT-generated NEON assembly for f64 QKV projection.
func qkvlinearNEONF64(x, wQKV, biasQ, biasK, biasV, q, k, v []float64, batchSize, inFeatures, qDim, kvDim int) {
	asm.QKVLinearNEONF64(x, wQKV, biasQ, biasK, biasV, q, k, v, batchSize, inFeatures, qDim, kvDim)
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

	// Override SDPA dispatch with GOAT NEON implementations
	SDPAFloat32 = sdpaNEONF32
	SDPAFloat64 = sdpaNEONF64
	SDPACausalFloat32 = sdpaCausalNEONF32
	SDPACausalFloat64 = sdpaCausalNEONF64

	// Override QKVLinear dispatch with GOAT NEON implementations
	QKVLinearFloat32 = qkvlinearNEONF32
	QKVLinearFloat64 = qkvlinearNEONF64

	// Float16/BFloat16 use the hwygen-generated promoted implementations
	// (promote to f32, compute, demote) which are already efficient enough
	// since the promotion is the bottleneck, not the compute.
}
