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
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

//go:generate go run ../../../cmd/hwygen -input sdpa_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch sdpa

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

	lanes := hwy.MaxLanes[T]()

	for i := range seqLen {
		qOff := i * headDim
		sOff := i * kvLen

		// Step 1: Q[i,:] @ K^T -> scores[i,:], scaled
		// 4x unroll across kvLen: share Q vector load across 4 K rows
		j := 0
		for ; j+4 <= kvLen; j += 4 {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()
			kOff0 := j * headDim
			kOff1 := (j + 1) * headDim
			kOff2 := (j + 2) * headDim
			kOff3 := (j + 3) * headDim
			p := 0
			for ; p+lanes <= headDim; p += lanes {
				vQ := hwy.Load(q[qOff+p:])
				acc0 = hwy.MulAdd(vQ, hwy.Load(k[kOff0+p:]), acc0)
				acc1 = hwy.MulAdd(vQ, hwy.Load(k[kOff1+p:]), acc1)
				acc2 = hwy.MulAdd(vQ, hwy.Load(k[kOff2+p:]), acc2)
				acc3 = hwy.MulAdd(vQ, hwy.Load(k[kOff3+p:]), acc3)
			}
			s0 := hwy.ReduceSum(acc0)
			s1 := hwy.ReduceSum(acc1)
			s2 := hwy.ReduceSum(acc2)
			s3 := hwy.ReduceSum(acc3)
			for ; p < headDim; p++ {
				qp := q[qOff+p]
				s0 += qp * k[kOff0+p]
				s1 += qp * k[kOff1+p]
				s2 += qp * k[kOff2+p]
				s3 += qp * k[kOff3+p]
			}
			scores[sOff+j] = s0 * scale
			scores[sOff+j+1] = s1 * scale
			scores[sOff+j+2] = s2 * scale
			scores[sOff+j+3] = s3 * scale
		}
		// Remainder: 1 at a time
		for ; j < kvLen; j++ {
			kOff := j * headDim
			acc := hwy.Zero[T]()
			p := 0
			for ; p+lanes <= headDim; p += lanes {
				vQ := hwy.Load(q[qOff+p:])
				vK := hwy.Load(k[kOff+p:])
				acc = hwy.MulAdd(vQ, vK, acc)
			}
			sum := hwy.ReduceSum(acc)
			for ; p < headDim; p++ {
				sum += q[qOff+p] * k[kOff+p]
			}
			scores[sOff+j] = sum * scale
		}

		// Add mask if provided
		if mask != nil {
			mOff := i * kvLen
			si := 0
			for ; si+lanes <= kvLen; si += lanes {
				s := hwy.Load(scores[sOff+si:])
				m := hwy.Load(mask[mOff+si:])
				hwy.Store(hwy.Add(s, m), scores[sOff+si:])
			}
			for ; si < kvLen; si++ {
				scores[sOff+si] += mask[mOff+si]
			}
		}

		// Per-row softmax
		maxVal := scores[sOff]
		for j := 1; j < kvLen; j++ {
			if scores[sOff+j] > maxVal {
				maxVal = scores[sOff+j]
			}
		}
		vMax := hwy.Set(maxVal)
		sumAcc := hwy.Zero[T]()
		si := 0
		for ; si+lanes <= kvLen; si += lanes {
			x := hwy.Load(scores[sOff+si:])
			shifted := hwy.Sub(x, vMax)
			expVal := math.BaseExpVec(shifted)
			hwy.Store(expVal, scores[sOff+si:])
			sumAcc = hwy.Add(sumAcc, expVal)
		}
		expSum := hwy.ReduceSum(sumAcc)
		for ; si < kvLen; si++ {
			scores[sOff+si] = T(stdmath.Exp(float64(scores[sOff+si] - maxVal)))
			expSum += scores[sOff+si]
		}
		invSum := T(1.0) / expSum
		vInvSum := hwy.Set(invSum)
		si = 0
		for ; si+lanes <= kvLen; si += lanes {
			x := hwy.Load(scores[sOff+si:])
			hwy.Store(hwy.Mul(x, vInvSum), scores[sOff+si:])
		}
		for ; si < kvLen; si++ {
			scores[sOff+si] = scores[sOff+si] * invSum
		}

		// Step 2: scores[i,:] @ V -> output[i,:] via register-tiled accumulation
		// Tile headDim into 4*lanes strips; accumulators stay in registers across full kvLen loop
		oOff := i * headDim
		tileD := 4 * lanes
		d := 0
		for ; d+tileD <= headDim; d += tileD {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()
			for j := range kvLen {
				vS := hwy.Set(scores[sOff+j])
				vRow := v[j*headDim:]
				acc0 = hwy.MulAdd(vS, hwy.Load(vRow[d:]), acc0)
				acc1 = hwy.MulAdd(vS, hwy.Load(vRow[d+lanes:]), acc1)
				acc2 = hwy.MulAdd(vS, hwy.Load(vRow[d+2*lanes:]), acc2)
				acc3 = hwy.MulAdd(vS, hwy.Load(vRow[d+3*lanes:]), acc3)
			}
			hwy.Store(acc0, output[oOff+d:])
			hwy.Store(acc1, output[oOff+d+lanes:])
			hwy.Store(acc2, output[oOff+d+2*lanes:])
			hwy.Store(acc3, output[oOff+d+3*lanes:])
		}
		// Remainder: single accumulator per remaining vector strip
		for ; d+lanes <= headDim; d += lanes {
			acc := hwy.Zero[T]()
			for j := range kvLen {
				vS := hwy.Set(scores[sOff+j])
				acc = hwy.MulAdd(vS, hwy.Load(v[j*headDim+d:]), acc)
			}
			hwy.Store(acc, output[oOff+d:])
		}
		// Scalar tail
		for ; d < headDim; d++ {
			var sum T
			for j := range kvLen {
				sum += scores[sOff+j] * v[j*headDim+d]
			}
			output[oOff+d] = sum
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

	lanes := hwy.MaxLanes[T]()
	negInf := T(stdmath.Inf(-1))
	offset := kvLen - seqLen

	for i := range seqLen {
		qOff := i * headDim
		sOff := i * kvLen
		causalEnd := i + offset + 1 // attend to positions [0, causalEnd)

		// Step 1: Q[i,:] @ K^T -> scores[i,:], scaled, with causal mask
		// 4x unroll across kvLen while all 4 entries are within causal window
		j := 0
		for ; j+4 <= kvLen && j+4 <= causalEnd; j += 4 {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()
			kOff0 := j * headDim
			kOff1 := (j + 1) * headDim
			kOff2 := (j + 2) * headDim
			kOff3 := (j + 3) * headDim
			p := 0
			for ; p+lanes <= headDim; p += lanes {
				vQ := hwy.Load(q[qOff+p:])
				acc0 = hwy.MulAdd(vQ, hwy.Load(k[kOff0+p:]), acc0)
				acc1 = hwy.MulAdd(vQ, hwy.Load(k[kOff1+p:]), acc1)
				acc2 = hwy.MulAdd(vQ, hwy.Load(k[kOff2+p:]), acc2)
				acc3 = hwy.MulAdd(vQ, hwy.Load(k[kOff3+p:]), acc3)
			}
			s0 := hwy.ReduceSum(acc0)
			s1 := hwy.ReduceSum(acc1)
			s2 := hwy.ReduceSum(acc2)
			s3 := hwy.ReduceSum(acc3)
			for ; p < headDim; p++ {
				qp := q[qOff+p]
				s0 += qp * k[kOff0+p]
				s1 += qp * k[kOff1+p]
				s2 += qp * k[kOff2+p]
				s3 += qp * k[kOff3+p]
			}
			scores[sOff+j] = s0 * scale
			scores[sOff+j+1] = s1 * scale
			scores[sOff+j+2] = s2 * scale
			scores[sOff+j+3] = s3 * scale
		}
		// Remainder: 1 at a time (handles causal boundary and tail)
		for ; j < kvLen; j++ {
			if j >= causalEnd {
				scores[sOff+j] = negInf
				continue
			}
			kOff := j * headDim
			acc := hwy.Zero[T]()
			p := 0
			for ; p+lanes <= headDim; p += lanes {
				vQ := hwy.Load(q[qOff+p:])
				vK := hwy.Load(k[kOff+p:])
				acc = hwy.MulAdd(vQ, vK, acc)
			}
			sum := hwy.ReduceSum(acc)
			for ; p < headDim; p++ {
				sum += q[qOff+p] * k[kOff+p]
			}
			scores[sOff+j] = sum * scale
		}

		// Per-row softmax
		maxVal := scores[sOff]
		for j := 1; j < kvLen; j++ {
			if scores[sOff+j] > maxVal {
				maxVal = scores[sOff+j]
			}
		}
		vMax := hwy.Set(maxVal)
		sumAcc := hwy.Zero[T]()
		si := 0
		for ; si+lanes <= kvLen; si += lanes {
			x := hwy.Load(scores[sOff+si:])
			shifted := hwy.Sub(x, vMax)
			expVal := math.BaseExpVec(shifted)
			hwy.Store(expVal, scores[sOff+si:])
			sumAcc = hwy.Add(sumAcc, expVal)
		}
		expSum := hwy.ReduceSum(sumAcc)
		for ; si < kvLen; si++ {
			scores[sOff+si] = T(stdmath.Exp(float64(scores[sOff+si] - maxVal)))
			expSum += scores[sOff+si]
		}
		invSum := T(1.0) / expSum
		vInvSum := hwy.Set(invSum)
		si = 0
		for ; si+lanes <= kvLen; si += lanes {
			x := hwy.Load(scores[sOff+si:])
			hwy.Store(hwy.Mul(x, vInvSum), scores[sOff+si:])
		}
		for ; si < kvLen; si++ {
			scores[sOff+si] = scores[sOff+si] * invSum
		}

		// Step 2: scores[i,:] @ V -> output[i,:] via register-tiled accumulation
		oOff := i * headDim
		tileD := 4 * lanes
		d := 0
		for ; d+tileD <= headDim; d += tileD {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()
			for j := range kvLen {
				vS := hwy.Set(scores[sOff+j])
				vRow := v[j*headDim:]
				acc0 = hwy.MulAdd(vS, hwy.Load(vRow[d:]), acc0)
				acc1 = hwy.MulAdd(vS, hwy.Load(vRow[d+lanes:]), acc1)
				acc2 = hwy.MulAdd(vS, hwy.Load(vRow[d+2*lanes:]), acc2)
				acc3 = hwy.MulAdd(vS, hwy.Load(vRow[d+3*lanes:]), acc3)
			}
			hwy.Store(acc0, output[oOff+d:])
			hwy.Store(acc1, output[oOff+d+lanes:])
			hwy.Store(acc2, output[oOff+d+2*lanes:])
			hwy.Store(acc3, output[oOff+d+3*lanes:])
		}
		for ; d+lanes <= headDim; d += lanes {
			acc := hwy.Zero[T]()
			for j := range kvLen {
				vS := hwy.Set(scores[sOff+j])
				acc = hwy.MulAdd(vS, hwy.Load(v[j*headDim+d:]), acc)
			}
			hwy.Store(acc, output[oOff+d:])
		}
		for ; d < headDim; d++ {
			var sum T
			for j := range kvLen {
				sum += scores[sOff+j] * v[j*headDim+d]
			}
			output[oOff+d] = sum
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
