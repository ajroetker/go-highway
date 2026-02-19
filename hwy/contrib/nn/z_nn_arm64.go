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
// The z_c_slices_* dispatch files set LayerNorm*, Softmax*, SDPA*, QKVDense*
// to hwygen-generated NEON assembly; this file's init() runs afterward to
// override SDPA and QKVDense with SME implementations when available.

package nn

import (
	"math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/nn/asm"
)

// Minimum dimensions for SME SDPA acceleration.
const minDimForSDPASME = 32

// fillNegInfColumns sets mask[i, kvLen:paddedKvLen] = -inf for all rows.
// This prevents softmax from assigning weight to zero-padded KV positions.
func fillNegInfColumns[T hwy.Floats](m []T, rows, kvLen, paddedKvLen int) {
	var negInf T
	switch any(negInf).(type) {
	case float32:
		negInf = T(float32(math.Inf(-1)))
	case float64:
		negInf = T(math.Inf(-1))
	}
	for i := range rows {
		for j := kvLen; j < paddedKvLen; j++ {
			m[i*paddedKvLen+j] = negInf
		}
	}
}

// buildCausalPaddingMask builds an explicit causal + padding mask for padded SDPA.
// mask[i, j] = 0 if j <= i + offset AND j < kvLen, else -inf.
func buildCausalPaddingMask[T hwy.Floats](m []T, seqLen, kvLen, paddedSeqLen, paddedKvLen int) {
	offset := kvLen - seqLen
	var zero, negInf T
	switch any(zero).(type) {
	case float32:
		negInf = T(float32(math.Inf(-1)))
	case float64:
		negInf = T(math.Inf(-1))
	}
	for i := range paddedSeqLen {
		for j := range paddedKvLen {
			if i < seqLen && j < kvLen && j <= i+offset {
				m[i*paddedKvLen+j] = zero
			} else {
				m[i*paddedKvLen+j] = negInf
			}
		}
	}
}


// =============================================================================
// SME SDPA adapter functions
// =============================================================================

// sdpaSMEF32 uses SME Flash Attention with online softmax via FMOPA.
// Avoids materializing the full [seqLen, kvLen] scores matrix.
// Falls back to NEON for small dimensions; pads unaligned dimensions to tile boundary.
func sdpaSMEF32(q, k, v, mask, scores, output []float32, seqLen, kvLen, headDim int, scale float32) {
	const tileSize = 16
	paddedSeqLen := matmul.AlignUp(seqLen, tileSize)
	paddedKvLen := matmul.AlignUp(kvLen, tileSize)
	paddedHeadDim := matmul.AlignUp(headDim, tileSize)

	if paddedSeqLen < minDimForSDPASME || paddedKvLen < minDimForSDPASME || paddedHeadDim < minDimForSDPASME {
		asm.SDPACF32(q, k, v, mask, scores, output, seqLen, kvLen, headDim, scale)
		return
	}

	needsPadSeq := paddedSeqLen != seqLen
	needsPadKv := paddedKvLen != kvLen
	needsPadHd := paddedHeadDim != headDim

	// Pad Q [seqLen, headDim] → [paddedSeqLen, paddedHeadDim]
	fmopaQ := q
	if needsPadSeq || needsPadHd {
		pq := getTempSlice[float32](paddedSeqLen * paddedHeadDim)
		defer putTempSlice(pq)
		matmul.PadMatrix2D(pq, q, seqLen, headDim, paddedSeqLen, paddedHeadDim)
		fmopaQ = pq
	}

	// Pad K [kvLen, headDim] → [paddedKvLen, paddedHeadDim]
	fmopaK := k
	if needsPadKv || needsPadHd {
		pk := getTempSlice[float32](paddedKvLen * paddedHeadDim)
		defer putTempSlice(pk)
		matmul.PadMatrix2D(pk, k, kvLen, headDim, paddedKvLen, paddedHeadDim)
		fmopaK = pk
	}

	// Pad V [kvLen, headDim] → [paddedKvLen, paddedHeadDim]
	fmopaV := v
	if needsPadKv || needsPadHd {
		pv := getTempSlice[float32](paddedKvLen * paddedHeadDim)
		defer putTempSlice(pv)
		matmul.PadMatrix2D(pv, v, kvLen, headDim, paddedKvLen, paddedHeadDim)
		fmopaV = pv
	}

	// Build mask: when KV is padded, we MUST mask out padded columns with -inf
	// to prevent softmax from assigning attention weight to zero-padded positions.
	fmopaMask := mask
	if needsPadKv {
		pm := getTempSlice[float32](paddedSeqLen * paddedKvLen)
		defer putTempSlice(pm)
		if mask != nil {
			matmul.PadMatrix2D(pm, mask, seqLen, kvLen, paddedSeqLen, paddedKvLen)
		} else {
			clear(pm)
		}
		fillNegInfColumns(pm, paddedSeqLen, kvLen, paddedKvLen)
		fmopaMask = pm
	} else if mask != nil && needsPadSeq {
		pm := getTempSlice[float32](paddedSeqLen * paddedKvLen)
		defer putTempSlice(pm)
		matmul.PadMatrix2D(pm, mask, seqLen, kvLen, paddedSeqLen, paddedKvLen)
		fmopaMask = pm
	}

	// Transpose Q [paddedSeqLen, paddedHeadDim] → qt [paddedHeadDim, paddedSeqLen]
	qt := getTempSlice[float32](paddedHeadDim * paddedSeqLen)
	defer putTempSlice(qt)
	matmul.Transpose2D(fmopaQ, paddedSeqLen, paddedHeadDim, qt)

	// Transpose K [paddedKvLen, paddedHeadDim] → kt [paddedHeadDim, paddedKvLen]
	kt := getTempSlice[float32](paddedHeadDim * paddedKvLen)
	defer putTempSlice(kt)
	matmul.Transpose2D(fmopaK, paddedKvLen, paddedHeadDim, kt)

	if needsPadSeq || needsPadHd {
		// Use padded output, then extract
		paddedOut := getTempSlice[float32](paddedSeqLen * paddedHeadDim)
		defer putTempSlice(paddedOut)
		clear(paddedOut)
		asm.SDPAFMOPAF32(qt, kt, fmopaV, fmopaMask, paddedOut, paddedSeqLen, paddedKvLen, paddedHeadDim, scale)
		matmul.ExtractMatrix2D(output, paddedOut, seqLen, headDim, paddedHeadDim)
	} else {
		asm.SDPAFMOPAF32(qt, kt, fmopaV, fmopaMask, output, paddedSeqLen, paddedKvLen, paddedHeadDim, scale)
	}
}

// sdpaSMEF64 uses SME Flash Attention with online softmax via FMOPA (float64).
func sdpaSMEF64(q, k, v, mask, scores, output []float64, seqLen, kvLen, headDim int, scale float64) {
	const tileSize = 8
	paddedSeqLen := matmul.AlignUp(seqLen, tileSize)
	paddedKvLen := matmul.AlignUp(kvLen, tileSize)
	paddedHeadDim := matmul.AlignUp(headDim, tileSize)

	if paddedSeqLen < minDimForSDPASME || paddedKvLen < minDimForSDPASME || paddedHeadDim < minDimForSDPASME {
		asm.SDPACF64(q, k, v, mask, scores, output, seqLen, kvLen, headDim, scale)
		return
	}

	needsPadSeq := paddedSeqLen != seqLen
	needsPadKv := paddedKvLen != kvLen
	needsPadHd := paddedHeadDim != headDim

	fmopaQ := q
	if needsPadSeq || needsPadHd {
		pq := getTempSlice[float64](paddedSeqLen * paddedHeadDim)
		defer putTempSlice(pq)
		matmul.PadMatrix2D(pq, q, seqLen, headDim, paddedSeqLen, paddedHeadDim)
		fmopaQ = pq
	}

	fmopaK := k
	if needsPadKv || needsPadHd {
		pk := getTempSlice[float64](paddedKvLen * paddedHeadDim)
		defer putTempSlice(pk)
		matmul.PadMatrix2D(pk, k, kvLen, headDim, paddedKvLen, paddedHeadDim)
		fmopaK = pk
	}

	fmopaV := v
	if needsPadKv || needsPadHd {
		pv := getTempSlice[float64](paddedKvLen * paddedHeadDim)
		defer putTempSlice(pv)
		matmul.PadMatrix2D(pv, v, kvLen, headDim, paddedKvLen, paddedHeadDim)
		fmopaV = pv
	}

	// Build mask with -inf for padded KV columns
	fmopaMask := mask
	if needsPadKv {
		pm := getTempSlice[float64](paddedSeqLen * paddedKvLen)
		defer putTempSlice(pm)
		if mask != nil {
			matmul.PadMatrix2D(pm, mask, seqLen, kvLen, paddedSeqLen, paddedKvLen)
		} else {
			clear(pm)
		}
		fillNegInfColumns(pm, paddedSeqLen, kvLen, paddedKvLen)
		fmopaMask = pm
	} else if mask != nil && needsPadSeq {
		pm := getTempSlice[float64](paddedSeqLen * paddedKvLen)
		defer putTempSlice(pm)
		matmul.PadMatrix2D(pm, mask, seqLen, kvLen, paddedSeqLen, paddedKvLen)
		fmopaMask = pm
	}

	qt := getTempSlice[float64](paddedHeadDim * paddedSeqLen)
	defer putTempSlice(qt)
	matmul.Transpose2D(fmopaQ, paddedSeqLen, paddedHeadDim, qt)

	kt := getTempSlice[float64](paddedHeadDim * paddedKvLen)
	defer putTempSlice(kt)
	matmul.Transpose2D(fmopaK, paddedKvLen, paddedHeadDim, kt)

	if needsPadSeq || needsPadHd {
		paddedOut := getTempSlice[float64](paddedSeqLen * paddedHeadDim)
		defer putTempSlice(paddedOut)
		clear(paddedOut)
		asm.SDPAFMOPAF64(qt, kt, fmopaV, fmopaMask, paddedOut, paddedSeqLen, paddedKvLen, paddedHeadDim, scale)
		matmul.ExtractMatrix2D(output, paddedOut, seqLen, headDim, paddedHeadDim)
	} else {
		asm.SDPAFMOPAF64(qt, kt, fmopaV, fmopaMask, output, paddedSeqLen, paddedKvLen, paddedHeadDim, scale)
	}
}

// =============================================================================
// SME Causal SDPA adapter functions
// =============================================================================

// sdpaCausalSMEF32 uses SME Flash Attention with implicit causal masking for float32.
// Falls back to NEON for small dimensions; pads unaligned dimensions to tile boundary.
// When padding is needed, uses the non-causal asm with an explicit combined
// causal+padding mask to correctly handle padded KV positions.
func sdpaCausalSMEF32(q, k, v, scores, output []float32, seqLen, kvLen, headDim int, scale float32) {
	const tileSize = 16
	paddedSeqLen := matmul.AlignUp(seqLen, tileSize)
	paddedKvLen := matmul.AlignUp(kvLen, tileSize)
	paddedHeadDim := matmul.AlignUp(headDim, tileSize)

	if paddedSeqLen < minDimForSDPASME || paddedKvLen < minDimForSDPASME || paddedHeadDim < minDimForSDPASME {
		asm.SDPACausalCF32(q, k, v, scores, output, seqLen, kvLen, headDim, scale)
		return
	}

	needsPadSeq := paddedSeqLen != seqLen
	needsPadKv := paddedKvLen != kvLen
	needsPadHd := paddedHeadDim != headDim

	fmopaQ := q
	if needsPadSeq || needsPadHd {
		pq := getTempSlice[float32](paddedSeqLen * paddedHeadDim)
		defer putTempSlice(pq)
		matmul.PadMatrix2D(pq, q, seqLen, headDim, paddedSeqLen, paddedHeadDim)
		fmopaQ = pq
	}

	fmopaK := k
	if needsPadKv || needsPadHd {
		pk := getTempSlice[float32](paddedKvLen * paddedHeadDim)
		defer putTempSlice(pk)
		matmul.PadMatrix2D(pk, k, kvLen, headDim, paddedKvLen, paddedHeadDim)
		fmopaK = pk
	}

	fmopaV := v
	if needsPadKv || needsPadHd {
		pv := getTempSlice[float32](paddedKvLen * paddedHeadDim)
		defer putTempSlice(pv)
		matmul.PadMatrix2D(pv, v, kvLen, headDim, paddedKvLen, paddedHeadDim)
		fmopaV = pv
	}

	qt := getTempSlice[float32](paddedHeadDim * paddedSeqLen)
	defer putTempSlice(qt)
	matmul.Transpose2D(fmopaQ, paddedSeqLen, paddedHeadDim, qt)

	kt := getTempSlice[float32](paddedHeadDim * paddedKvLen)
	defer putTempSlice(kt)
	matmul.Transpose2D(fmopaK, paddedKvLen, paddedHeadDim, kt)

	if needsPadSeq || needsPadKv {
		// When padding, use non-causal asm with explicit causal+padding mask
		// to correctly mask out padded KV positions.
		cm := getTempSlice[float32](paddedSeqLen * paddedKvLen)
		defer putTempSlice(cm)
		buildCausalPaddingMask(cm, seqLen, kvLen, paddedSeqLen, paddedKvLen)

		paddedOut := getTempSlice[float32](paddedSeqLen * paddedHeadDim)
		defer putTempSlice(paddedOut)
		clear(paddedOut)
		asm.SDPAFMOPAF32(qt, kt, fmopaV, cm, paddedOut, paddedSeqLen, paddedKvLen, paddedHeadDim, scale)
		matmul.ExtractMatrix2D(output, paddedOut, seqLen, headDim, paddedHeadDim)
	} else if needsPadHd {
		paddedOut := getTempSlice[float32](paddedSeqLen * paddedHeadDim)
		defer putTempSlice(paddedOut)
		clear(paddedOut)
		asm.SDPACausalFMOPAF32(qt, kt, fmopaV, paddedOut, paddedSeqLen, paddedKvLen, paddedHeadDim, scale)
		matmul.ExtractMatrix2D(output, paddedOut, seqLen, headDim, paddedHeadDim)
	} else {
		asm.SDPACausalFMOPAF32(qt, kt, fmopaV, output, paddedSeqLen, paddedKvLen, paddedHeadDim, scale)
	}
}

// sdpaCausalSMEF64 uses SME Flash Attention with implicit causal masking for float64.
// When padding is needed, uses the non-causal asm with an explicit combined
// causal+padding mask to correctly handle padded KV positions.
func sdpaCausalSMEF64(q, k, v, scores, output []float64, seqLen, kvLen, headDim int, scale float64) {
	const tileSize = 8
	paddedSeqLen := matmul.AlignUp(seqLen, tileSize)
	paddedKvLen := matmul.AlignUp(kvLen, tileSize)
	paddedHeadDim := matmul.AlignUp(headDim, tileSize)

	if paddedSeqLen < minDimForSDPASME || paddedKvLen < minDimForSDPASME || paddedHeadDim < minDimForSDPASME {
		asm.SDPACausalCF64(q, k, v, scores, output, seqLen, kvLen, headDim, scale)
		return
	}

	needsPadSeq := paddedSeqLen != seqLen
	needsPadKv := paddedKvLen != kvLen
	needsPadHd := paddedHeadDim != headDim

	fmopaQ := q
	if needsPadSeq || needsPadHd {
		pq := getTempSlice[float64](paddedSeqLen * paddedHeadDim)
		defer putTempSlice(pq)
		matmul.PadMatrix2D(pq, q, seqLen, headDim, paddedSeqLen, paddedHeadDim)
		fmopaQ = pq
	}

	fmopaK := k
	if needsPadKv || needsPadHd {
		pk := getTempSlice[float64](paddedKvLen * paddedHeadDim)
		defer putTempSlice(pk)
		matmul.PadMatrix2D(pk, k, kvLen, headDim, paddedKvLen, paddedHeadDim)
		fmopaK = pk
	}

	fmopaV := v
	if needsPadKv || needsPadHd {
		pv := getTempSlice[float64](paddedKvLen * paddedHeadDim)
		defer putTempSlice(pv)
		matmul.PadMatrix2D(pv, v, kvLen, headDim, paddedKvLen, paddedHeadDim)
		fmopaV = pv
	}

	qt := getTempSlice[float64](paddedHeadDim * paddedSeqLen)
	defer putTempSlice(qt)
	matmul.Transpose2D(fmopaQ, paddedSeqLen, paddedHeadDim, qt)

	kt := getTempSlice[float64](paddedHeadDim * paddedKvLen)
	defer putTempSlice(kt)
	matmul.Transpose2D(fmopaK, paddedKvLen, paddedHeadDim, kt)

	if needsPadSeq || needsPadKv {
		// When padding, use non-causal asm with explicit causal+padding mask
		cm := getTempSlice[float64](paddedSeqLen * paddedKvLen)
		defer putTempSlice(cm)
		buildCausalPaddingMask(cm, seqLen, kvLen, paddedSeqLen, paddedKvLen)

		paddedOut := getTempSlice[float64](paddedSeqLen * paddedHeadDim)
		defer putTempSlice(paddedOut)
		clear(paddedOut)
		asm.SDPAFMOPAF64(qt, kt, fmopaV, cm, paddedOut, paddedSeqLen, paddedKvLen, paddedHeadDim, scale)
		matmul.ExtractMatrix2D(output, paddedOut, seqLen, headDim, paddedHeadDim)
	} else if needsPadHd {
		paddedOut := getTempSlice[float64](paddedSeqLen * paddedHeadDim)
		defer putTempSlice(paddedOut)
		clear(paddedOut)
		asm.SDPACausalFMOPAF64(qt, kt, fmopaV, paddedOut, paddedSeqLen, paddedKvLen, paddedHeadDim, scale)
		matmul.ExtractMatrix2D(output, paddedOut, seqLen, headDim, paddedHeadDim)
	} else {
		asm.SDPACausalFMOPAF64(qt, kt, fmopaV, output, paddedSeqLen, paddedKvLen, paddedHeadDim, scale)
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
	if hwy.NoSimdEnv() || !hwy.HasSME() {
		return
	}

	// SME FMOPA provides higher throughput for aligned dimensions.
	// The SME adapters check alignment and fall back to NEON assembly internally.
	SDPAFloat32 = sdpaSMEF32
	SDPAFloat64 = sdpaSMEF64
	SDPACausalFloat32 = sdpaCausalSMEF32
	SDPACausalFloat64 = sdpaCausalSMEF64

	// SME QKVDense decomposes into 3 optimized MatMulKLast calls.
	QKVDenseFloat32 = qkvdenseSMEF32
	QKVDenseFloat64 = qkvdenseSMEF64

	// LayerNorm, Softmax, Dense, and non-SME SDPA/QKVDense use
	// hwygen-generated NEON assembly from z_c_slices_* dispatch files.
	// Float16/BFloat16 use promoted implementations (promote to f32, compute, demote).
}
