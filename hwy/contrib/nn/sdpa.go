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
//   - q:      [batchSize, numHeads, seqLen, headDim] (queries, contiguous)
//   - k:      [batchSize, numKVHeads, kvLen, headDim] (keys, contiguous)
//   - v:      [batchSize, numKVHeads, kvLen, headDim] (values, contiguous)
//   - mask:   [seqLen, kvLen] (additive mask shared across heads, nil for no mask)
//   - output: [batchSize, numHeads, seqLen, headDim] (result, contiguous)
//
// When numKVHeads < numHeads, grouped-query attention is used: each KV head
// serves numHeads/numKVHeads query heads.
func MultiHeadSDPAAuto[T hwy.Floats](
	q, k, v, mask, output []T,
	batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim int,
	scale T, causal bool,
) {
	if batchSize == 0 || numHeads == 0 || seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	headsPerKVHead := numHeads / numKVHeads
	qHeadStride := seqLen * headDim
	kvHeadStride := kvLen * headDim

	for b := range batchSize {
		qBatchOff := b * numHeads * qHeadStride
		kBatchOff := b * numKVHeads * kvHeadStride
		vBatchOff := b * numKVHeads * kvHeadStride
		oBatchOff := b * numHeads * qHeadStride

		for h := range numHeads {
			kvHead := h / headsPerKVHead

			qOff := qBatchOff + h*qHeadStride
			kOff := kBatchOff + kvHead*kvHeadStride
			vOff := vBatchOff + kvHead*kvHeadStride
			oOff := oBatchOff + h*qHeadStride

			qSlice := q[qOff : qOff+qHeadStride]
			kSlice := k[kOff : kOff+kvHeadStride]
			vSlice := v[vOff : vOff+kvHeadStride]
			oSlice := output[oOff : oOff+qHeadStride]

			if causal {
				SDPACausalAuto(qSlice, kSlice, vSlice, oSlice,
					seqLen, kvLen, headDim, scale)
			} else {
				SDPAAuto(qSlice, kSlice, vSlice, mask, oSlice,
					seqLen, kvLen, headDim, scale)
			}
		}
	}
}
