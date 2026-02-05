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

	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go run ../../../cmd/hwygen -input sdpa_base.go -output . -targets avx2,avx512,neon,fallback

// BaseSDPA computes single-head scaled dot-product attention.
//
//   - q:      [seqLen, headDim] (queries, row-major)
//   - k:      [kvLen, headDim] (keys, row-major)
//   - v:      [kvLen, headDim] (values, row-major)
//   - mask:   [seqLen, kvLen] (additive mask, nil for no mask)
//   - scores: [seqLen, kvLen] (scratch buffer for attention weights)
//   - output: [seqLen, headDim] (result)
//   - scale:  typically 1/sqrt(headDim)
//
// Algorithm: output = softmax(Q@K^T * scale + mask) @ V
func BaseSDPA[T hwy.Floats](
	q, k, v, mask, scores, output []T,
	seqLen, kvLen, headDim int, scale T,
) {
	if seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	// Step 1: Q @ K^T -> scores [seqLen, kvLen], scaled
	for i := range seqLen {
		qOff := i * headDim
		sOff := i * kvLen

		for j := range kvLen {
			kOff := j * headDim
			var sum float64
			for p := range headDim {
				sum += float64(q[qOff+p]) * float64(k[kOff+p])
			}
			scores[sOff+j] = T(sum * float64(scale))
		}

		// Add mask if provided
		if mask != nil {
			mOff := i * kvLen
			for j := range kvLen {
				scores[sOff+j] += mask[mOff+j]
			}
		}

		// Per-row softmax
		{
			sRow := scores[sOff : sOff+kvLen]
			maxVal := sRow[0]
			for si := 1; si < kvLen; si++ {
				if sRow[si] > maxVal {
					maxVal = sRow[si]
				}
			}
			var expSum float64
			for si := range sRow {
				sRow[si] = T(stdmath.Exp(float64(sRow[si] - maxVal)))
				expSum += float64(sRow[si])
			}
			invSum := 1.0 / expSum
			for si := range sRow {
				sRow[si] = T(float64(sRow[si]) * invSum)
			}
		}
	}

	// Step 2: scores @ V -> output [seqLen, headDim]
	for i := range seqLen {
		sOff := i * kvLen
		oOff := i * headDim

		for d := range headDim {
			var sum float64
			for j := range kvLen {
				sum += float64(scores[sOff+j]) * float64(v[j*headDim+d])
			}
			output[oOff+d] = T(sum)
		}
	}
}

// BaseSDPACausal computes single-head causal scaled dot-product attention.
// This applies a lower-triangular mask on-the-fly: for position i, only
// keys at positions j <= i + (kvLen - seqLen) are attended to.
//
// Parameters are the same as BaseSDPA except mask is not needed (computed implicitly).
func BaseSDPACausal[T hwy.Floats](
	q, k, v, scores, output []T,
	seqLen, kvLen, headDim int, scale T,
) {
	if seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	negInf := T(stdmath.Inf(-1))
	offset := kvLen - seqLen

	// Step 1: Q @ K^T -> scores [seqLen, kvLen], scaled, with causal mask
	for i := range seqLen {
		qOff := i * headDim
		sOff := i * kvLen
		causalEnd := i + offset + 1 // attend to positions [0, causalEnd)

		for j := range kvLen {
			if j >= causalEnd {
				scores[sOff+j] = negInf
				continue
			}

			kOff := j * headDim
			var sum float64
			for p := range headDim {
				sum += float64(q[qOff+p]) * float64(k[kOff+p])
			}
			scores[sOff+j] = T(sum * float64(scale))
		}

		// Per-row softmax
		{
			sRow := scores[sOff : sOff+kvLen]
			maxVal := sRow[0]
			for si := 1; si < kvLen; si++ {
				if sRow[si] > maxVal {
					maxVal = sRow[si]
				}
			}
			var expSum float64
			for si := range sRow {
				sRow[si] = T(stdmath.Exp(float64(sRow[si] - maxVal)))
				expSum += float64(sRow[si])
			}
			invSum := 1.0 / expSum
			for si := range sRow {
				sRow[si] = T(float64(sRow[si]) * invSum)
			}
		}
	}

	// Step 2: scores @ V -> output [seqLen, headDim]
	for i := range seqLen {
		sOff := i * kvLen
		oOff := i * headDim

		for d := range headDim {
			var sum float64
			for j := range kvLen {
				sum += float64(scores[sOff+j]) * float64(v[j*headDim+d])
			}
			output[oOff+d] = T(sum)
		}
	}
}

// SDPAScalar is a scalar reference implementation for comparison and testing.
func SDPAScalar[T hwy.Floats](
	q, k, v, mask, scores, output []T,
	seqLen, kvLen, headDim int, scale T,
) {
	if seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	// Q @ K^T -> scores, scaled
	for i := range seqLen {
		qOff := i * headDim
		sOff := i * kvLen

		for j := range kvLen {
			kOff := j * headDim
			var sum float64
			for p := range headDim {
				sum += float64(q[qOff+p]) * float64(k[kOff+p])
			}
			scores[sOff+j] = T(sum * float64(scale))
		}

		// Add mask
		if mask != nil {
			mOff := i * kvLen
			for j := range kvLen {
				scores[sOff+j] += mask[mOff+j]
			}
		}

		// Softmax
		scalarSoftmaxRow(scores[sOff : sOff+kvLen])
	}

	// scores @ V -> output
	for i := range seqLen {
		sOff := i * kvLen
		oOff := i * headDim

		for d := range headDim {
			var sum float64
			for j := range kvLen {
				sum += float64(scores[sOff+j]) * float64(v[j*headDim+d])
			}
			output[oOff+d] = T(sum)
		}
	}
}

// SDPACausalScalar is a scalar reference implementation for causal SDPA.
func SDPACausalScalar[T hwy.Floats](
	q, k, v, scores, output []T,
	seqLen, kvLen, headDim int, scale T,
) {
	if seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	negInf := T(stdmath.Inf(-1))
	offset := kvLen - seqLen

	for i := range seqLen {
		qOff := i * headDim
		sOff := i * kvLen
		causalEnd := i + offset + 1

		for j := range kvLen {
			if j >= causalEnd {
				scores[sOff+j] = negInf
				continue
			}
			kOff := j * headDim
			var sum float64
			for p := range headDim {
				sum += float64(q[qOff+p]) * float64(k[kOff+p])
			}
			scores[sOff+j] = T(sum * float64(scale))
		}

		scalarSoftmaxRow(scores[sOff : sOff+kvLen])
	}

	for i := range seqLen {
		sOff := i * kvLen
		oOff := i * headDim

		for d := range headDim {
			var sum float64
			for j := range kvLen {
				sum += float64(scores[sOff+j]) * float64(v[j*headDim+d])
			}
			output[oOff+d] = T(sum)
		}
	}
}

// scalarSoftmaxRow applies softmax in-place using scalar operations.
func scalarSoftmaxRow[T hwy.Floats](row []T) {
	size := len(row)
	if size == 0 {
		return
	}

	maxVal := row[0]
	for i := 1; i < size; i++ {
		if row[i] > maxVal {
			maxVal = row[i]
		}
	}

	var expSum float64
	for i := range row {
		row[i] = T(stdmath.Exp(float64(row[i] - maxVal)))
		expSum += float64(row[i])
	}

	invSum := 1.0 / expSum
	for i := range row {
		row[i] = T(float64(row[i]) * invSum)
	}
}
