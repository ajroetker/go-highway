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
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
)

// QuantizedSDPA computes single-head scaled dot-product attention using
// integer-only matrix multiplication with per-tensor affine quantization.
//
// This quantizes Q, K, V from float32 to uint8, performs the two matmul
// stages (Q@K^T and attn@V) in int32 via Int8x8MatMul, and dequantizes
// back to float32.
//
// Parameters:
//   - q:      [seqLen, headDim] (queries, row-major)
//   - k:      [kvLen, headDim] (keys, row-major)
//   - v:      [kvLen, headDim] (values, row-major)
//   - mask:   [seqLen, kvLen] (additive mask, nil for no mask)
//   - output: [seqLen, headDim] (result)
//   - scale:  typically 1/sqrt(headDim)
//
// Algorithm: output = softmax(dequant(quantize(Q) @ quantize(K)^T) * scale + mask) @ V
func QuantizedSDPA(
	q, k, v, mask, output []float32,
	seqLen, kvLen, headDim int, scale float32,
) {
	if seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	// Step 1: Quantize Q [seqLen, headDim] -> uint8
	qQ := make([]uint8, seqLen*headDim)
	qScale, qZP := QuantizeAffine(q, qQ, seqLen*headDim)

	// Step 2: Quantize K [kvLen, headDim] -> uint8
	kQ := make([]uint8, kvLen*headDim)
	kScale, kZP := QuantizeAffine(k, kQ, kvLen*headDim)

	// Step 3: Transpose kQ [kvLen, headDim] -> kQT [headDim, kvLen]
	kQT := make([]uint8, headDim*kvLen)
	transposeUint8(kQ, kvLen, headDim, kQT)

	// Step 4: Q @ K^T via Int8x8MatMul -> scoresI32 [seqLen, kvLen]
	scoresI32 := make([]int32, seqLen*kvLen)
	matmul.Int8x8MatMul(scoresI32, qQ, kQT, qZP, kZP, seqLen, headDim, kvLen)

	// Step 5: Dequantize scores and apply scale
	scores := make([]float32, seqLen*kvLen)
	combinedScale := qScale * kScale * scale
	DequantizeInt32ToFloat32(scoresI32, scores, seqLen*kvLen, combinedScale)

	// Step 6: Add mask if provided
	if mask != nil {
		for i := range scores {
			scores[i] += mask[i]
		}
	}

	// Step 7: Per-row softmax
	for i := range seqLen {
		SoftmaxInPlaceFloat32(scores[i*kvLen : (i+1)*kvLen])
	}

	// Step 8: Quantize attention weights [seqLen, kvLen] -> uint8
	attnQ := make([]uint8, seqLen*kvLen)
	attnScale, attnZP := QuantizeAffine(scores, attnQ, seqLen*kvLen)

	// Step 9: Quantize V [kvLen, headDim] -> uint8
	vQ := make([]uint8, kvLen*headDim)
	vScale, vZP := QuantizeAffine(v, vQ, kvLen*headDim)

	// Step 10: attn @ V via Int8x8MatMul -> outputI32 [seqLen, headDim]
	outputI32 := make([]int32, seqLen*headDim)
	matmul.Int8x8MatMul(outputI32, attnQ, vQ, attnZP, vZP, seqLen, kvLen, headDim)

	// Step 11: Dequantize output
	DequantizeInt32ToFloat32(outputI32, output, seqLen*headDim, attnScale*vScale)
}

// QuantizedSDPACausal computes single-head causal scaled dot-product attention
// using integer-only matrix multiplication.
//
// Same as QuantizedSDPA but applies a lower-triangular causal mask: for
// position i, only keys at positions j <= i + (kvLen - seqLen) are attended to.
func QuantizedSDPACausal(
	q, k, v, output []float32,
	seqLen, kvLen, headDim int, scale float32,
) {
	if seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	negInf := float32(stdmath.Inf(-1))
	offset := kvLen - seqLen

	// Steps 1-5: same as QuantizedSDPA
	qQ := make([]uint8, seqLen*headDim)
	qScale, qZP := QuantizeAffine(q, qQ, seqLen*headDim)

	kQ := make([]uint8, kvLen*headDim)
	kScale, kZP := QuantizeAffine(k, kQ, kvLen*headDim)

	kQT := make([]uint8, headDim*kvLen)
	transposeUint8(kQ, kvLen, headDim, kQT)

	scoresI32 := make([]int32, seqLen*kvLen)
	matmul.Int8x8MatMul(scoresI32, qQ, kQT, qZP, kZP, seqLen, headDim, kvLen)

	scores := make([]float32, seqLen*kvLen)
	combinedScale := qScale * kScale * scale
	DequantizeInt32ToFloat32(scoresI32, scores, seqLen*kvLen, combinedScale)

	// Apply causal mask
	for i := range seqLen {
		causalEnd := i + offset + 1
		sOff := i * kvLen
		for j := causalEnd; j < kvLen; j++ {
			scores[sOff+j] = negInf
		}
	}

	// Per-row softmax
	for i := range seqLen {
		SoftmaxInPlaceFloat32(scores[i*kvLen : (i+1)*kvLen])
	}

	// Steps 8-11: same as QuantizedSDPA
	attnQ := make([]uint8, seqLen*kvLen)
	attnScale, attnZP := QuantizeAffine(scores, attnQ, seqLen*kvLen)

	vQ := make([]uint8, kvLen*headDim)
	vScale, vZP := QuantizeAffine(v, vQ, kvLen*headDim)

	outputI32 := make([]int32, seqLen*headDim)
	matmul.Int8x8MatMul(outputI32, attnQ, vQ, attnZP, vZP, seqLen, kvLen, headDim)

	DequantizeInt32ToFloat32(outputI32, output, seqLen*headDim, attnScale*vScale)
}

// transposeUint8 transposes a [rows, cols] uint8 matrix to [cols, rows].
func transposeUint8(src []uint8, rows, cols int, dst []uint8) {
	for r := range rows {
		for c := range cols {
			dst[c*rows+r] = src[r*cols+c]
		}
	}
}
