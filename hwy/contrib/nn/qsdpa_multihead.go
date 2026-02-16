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
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// MultiHeadQuantizedSDPA computes multi-head quantized scaled dot-product attention
// with optional grouped-query attention (GQA) support.
//
// This uses integer-only matrix multiplication (uint8×uint8→int32) for the
// Q@K^T and attn@V stages, with per-tensor affine quantization.
//
// Layout: all tensors are contiguous BHSD (batch, heads, seq, dim).
//
//   - pool:   worker pool for parallelizing across batch×head (nil = sequential)
//   - q:      [batchSize, numHeads, seqLen, headDim]
//   - k:      [batchSize, numKVHeads, kvLen, headDim]
//   - v:      [batchSize, numKVHeads, kvLen, headDim]
//   - mask:   additive mask, nil for no mask. May be [seqLen, kvLen] (shared),
//     or [batch, numHeads, seqLen, kvLen]. Use maskBatchStride/maskHeadStride
//     to control broadcasting (0 = broadcast).
//   - output: [batchSize, numHeads, seqLen, headDim]
func MultiHeadQuantizedSDPA(
	pool workerpool.Executor,
	q, k, v, mask, output []float32,
	batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim int,
	maskBatchStride, maskHeadStride int,
	scale float32, causal bool,
) {
	if batchSize == 0 || numHeads == 0 || seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	headsPerKVHead := numHeads / numKVHeads
	qHeadStride := seqLen * headDim
	kvHeadStride := kvLen * headDim
	maskSliceLen := seqLen * kvLen
	totalHeads := batchSize * numHeads

	doHead := func(idx int) {
		b := idx / numHeads
		h := idx % numHeads
		kvHead := h / headsPerKVHead

		qOff := (b*numHeads + h) * qHeadStride
		kOff := (b*numKVHeads + kvHead) * kvHeadStride
		vOff := kOff
		oOff := qOff

		qSlice := q[qOff : qOff+qHeadStride]
		kSlice := k[kOff : kOff+kvHeadStride]
		vSlice := v[vOff : vOff+kvHeadStride]
		oSlice := output[oOff : oOff+qHeadStride]

		if causal {
			QuantizedSDPACausal(qSlice, kSlice, vSlice, oSlice,
				seqLen, kvLen, headDim, scale)
		} else {
			var maskSlice []float32
			if mask != nil {
				maskOff := b*maskBatchStride + h*maskHeadStride
				maskSlice = mask[maskOff : maskOff+maskSliceLen]
			}
			QuantizedSDPA(qSlice, kSlice, vSlice, maskSlice, oSlice,
				seqLen, kvLen, headDim, scale)
		}
	}

	if pool != nil {
		pool.ParallelForAtomic(totalHeads, doHead)
	} else {
		for i := range totalHeads {
			doHead(i)
		}
	}
}

// MultiHeadQuantizedSDPAStrided computes multi-head quantized scaled dot-product
// attention with stride-based indexing, supporting both contiguous (BHSD) and
// interleaved (BSHD) memory layouts.
//
// When qSeqStride == headDim and kvSeqStride == headDim (contiguous / BHSD layout),
// this delegates directly to MultiHeadQuantizedSDPA with zero overhead.
//
// For non-contiguous layouts (e.g. BSHD), each head's data is gathered into
// contiguous temp buffers, the quantized single-head SDPA kernel is applied,
// and the result is scattered back.
//
// See MultiHeadSDPAStridedAuto for documentation of stride parameters.
func MultiHeadQuantizedSDPAStrided(
	pool workerpool.Executor,
	q, k, v, mask, output []float32,
	batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim int,
	qBatchStride, qHeadStride, qSeqStride int,
	kvBatchStride, kvHeadStride, kvSeqStride int,
	maskBatchStride, maskHeadStride int,
	scale float32, causal bool,
) {
	if batchSize == 0 || numHeads == 0 || seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	// Fast path: contiguous layout (BHSD) — delegate directly.
	if qSeqStride == headDim && kvSeqStride == headDim {
		MultiHeadQuantizedSDPA(pool, q, k, v, mask, output,
			batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim,
			maskBatchStride, maskHeadStride,
			scale, causal)
		return
	}

	// Strided path: gather per head → quantized SDPA → scatter.
	headsPerKVHead := numHeads / numKVHeads
	maskSliceLen := seqLen * kvLen
	totalHeads := batchSize * numHeads

	doHead := func(idx int) {
		b := idx / numHeads
		h := idx % numHeads
		kvHead := h / headsPerKVHead

		// Gather Q into contiguous temp buffer.
		qTemp := make([]float32, seqLen*headDim)
		qBase := b*qBatchStride + h*qHeadStride
		for s := range seqLen {
			src := qBase + s*qSeqStride
			copy(qTemp[s*headDim:(s+1)*headDim], q[src:src+headDim])
		}

		// Gather K into contiguous temp buffer.
		kTemp := make([]float32, kvLen*headDim)
		kBase := b*kvBatchStride + kvHead*kvHeadStride
		for s := range kvLen {
			src := kBase + s*kvSeqStride
			copy(kTemp[s*headDim:(s+1)*headDim], k[src:src+headDim])
		}

		// Gather V into contiguous temp buffer.
		vTemp := make([]float32, kvLen*headDim)
		vBase := kBase // V uses same layout as K.
		for s := range kvLen {
			src := vBase + s*kvSeqStride
			copy(vTemp[s*headDim:(s+1)*headDim], v[src:src+headDim])
		}

		// Output temp buffer.
		oTemp := make([]float32, seqLen*headDim)

		// Run single-head quantized SDPA on contiguous data.
		if causal {
			QuantizedSDPACausal(qTemp, kTemp, vTemp, oTemp,
				seqLen, kvLen, headDim, scale)
		} else {
			var maskSlice []float32
			if mask != nil {
				maskOff := b*maskBatchStride + h*maskHeadStride
				maskSlice = mask[maskOff : maskOff+maskSliceLen]
			}
			QuantizedSDPA(qTemp, kTemp, vTemp, maskSlice, oTemp,
				seqLen, kvLen, headDim, scale)
		}

		// Scatter output back to strided positions.
		oBase := b*qBatchStride + h*qHeadStride
		for s := range seqLen {
			dst := oBase + s*qSeqStride
			copy(output[dst:dst+headDim], oTemp[s*headDim:(s+1)*headDim])
		}
	}

	if pool != nil {
		pool.ParallelForAtomic(totalHeads, doHead)
	} else {
		for i := range totalHeads {
			doHead(i)
		}
	}
}
