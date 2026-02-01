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
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// SDPAAuto computes single-head scaled dot-product attention using the best
// available implementation.
//
//   - q:      [seqLen, headDim] (queries)
//   - k:      [kvLen, headDim] (keys)
//   - v:      [kvLen, headDim] (values)
//   - mask:   [seqLen, kvLen] (additive mask, nil for no mask)
//   - output: [seqLen, headDim] (result)
//   - scale:  typically 1/sqrt(headDim)
//
// This allocates a scratch buffer for attention scores internally.
func SDPAAuto[T hwy.Floats](
	q, k, v, mask, output []T,
	seqLen, kvLen, headDim int, scale T,
) {
	scores := getTempSlice[T](seqLen * kvLen)
	defer putTempSlice(scores)

	SDPA(q, k, v, mask, scores, output, seqLen, kvLen, headDim, scale)
}

// SDPACausalAuto computes single-head causal scaled dot-product attention
// using the best available implementation.
//
// Parameters are the same as SDPAAuto except mask is implicit (lower-triangular).
func SDPACausalAuto[T hwy.Floats](
	q, k, v, output []T,
	seqLen, kvLen, headDim int, scale T,
) {
	scores := getTempSlice[T](seqLen * kvLen)
	defer putTempSlice(scores)

	SDPACausal(q, k, v, scores, output, seqLen, kvLen, headDim, scale)
}

// MultiHeadSDPAAuto computes multi-head scaled dot-product attention with
// optional grouped-query attention (GQA) support.
//
//   - pool:   worker pool for parallelizing across batch×head (nil = sequential)
//   - q:      [batchSize, numHeads, seqLen, headDim] (queries, contiguous)
//   - k:      [batchSize, numKVHeads, kvLen, headDim] (keys, contiguous)
//   - v:      [batchSize, numKVHeads, kvLen, headDim] (values, contiguous)
//   - mask:   additive mask, nil for no mask. May be [seqLen, kvLen] (shared),
//     [batch, 1, seqLen, kvLen], or [batch, numHeads, seqLen, kvLen].
//     Use maskBatchStride/maskHeadStride to control broadcasting (0 = broadcast).
//   - output: [batchSize, numHeads, seqLen, headDim] (result, contiguous)
//
// maskBatchStride is the number of elements to advance per batch in the mask
// (0 means the same mask is shared across batches). maskHeadStride is the
// number of elements to advance per head (0 means shared across heads).
//
// When numKVHeads < numHeads, grouped-query attention is used: each KV head
// serves numHeads/numKVHeads query heads.
func MultiHeadSDPAAuto[T hwy.Floats](
	pool *workerpool.Pool,
	q, k, v, mask, output []T,
	batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim int,
	maskBatchStride, maskHeadStride int,
	scale T, causal bool,
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
			SDPACausalAuto(qSlice, kSlice, vSlice, oSlice,
				seqLen, kvLen, headDim, scale)
		} else {
			var maskSlice []T
			if mask != nil {
				maskOff := b*maskBatchStride + h*maskHeadStride
				maskSlice = mask[maskOff : maskOff+maskSliceLen]
			}
			SDPAAuto(qSlice, kSlice, vSlice, maskSlice, oSlice,
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
