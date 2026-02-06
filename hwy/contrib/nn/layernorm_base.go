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

//go:generate go run ../../../cmd/hwygen -input layernorm_base.go -output . -targets avx2,avx512,neon,fallback

// BaseLayerNorm computes layer normalization over groups of normSize elements.
//
// For each group of normSize contiguous elements in input:
//
//	output[i] = (input[i] - mean) / sqrt(variance + epsilon) * gamma[i%normSize] + beta[i%normSize]
//
// The input and output slices must have length that is a multiple of normSize.
// gamma and beta are optional (pass nil to skip affine transform).
// This is the standard layer normalization used in transformers.
func BaseLayerNorm[T hwy.Floats](input, output []T, normSize int, gamma, beta []T, epsilon T) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}

	numGroups := size / normSize
	invN := T(1.0) / T(normSize)
	lanes := hwy.MaxLanes[T]()

	for g := range numGroups {
		off := g * normSize

		// Pass 1: Compute mean using SIMD accumulation
		sumAcc := hwy.Zero[T]()
		ii := 0
		for ; ii+lanes <= normSize; ii += lanes {
			x := hwy.Load(input[off+ii:])
			sumAcc = hwy.Add(sumAcc, x)
		}
		mean := hwy.ReduceSum(sumAcc)
		for i := ii; i < normSize; i++ {
			mean += input[off+i]
		}
		mean *= invN

		// Pass 2: Compute variance using SIMD subtract-square-accumulate
		vMean := hwy.Set(mean)
		varAcc := hwy.Zero[T]()
		ii = 0
		for ; ii+lanes <= normSize; ii += lanes {
			x := hwy.Load(input[off+ii:])
			diff := hwy.Sub(x, vMean)
			varAcc = hwy.MulAdd(diff, diff, varAcc)
		}
		variance := hwy.ReduceSum(varAcc)
		for i := ii; i < normSize; i++ {
			diff := input[off+i] - mean
			variance += diff * diff
		}
		variance *= invN

		// Compute inverse standard deviation
		invStd := T(1.0 / stdmath.Sqrt(float64(variance+epsilon)))
		vInvStd := hwy.Set(invStd)

		// Pass 3: Normalize and optionally apply affine transform
		if gamma != nil && beta != nil {
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				x := hwy.Load(input[off+ii:])
				diff := hwy.Sub(x, vMean)
				normed := hwy.Mul(diff, vInvStd)

				g := hwy.Load(gamma[ii:])
				b := hwy.Load(beta[ii:])
				result := hwy.MulAdd(normed, g, b)
				hwy.Store(result, output[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				normed := (input[off+i] - mean) * invStd
				output[off+i] = normed*gamma[i] + beta[i]
			}
		} else if gamma != nil {
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				x := hwy.Load(input[off+ii:])
				diff := hwy.Sub(x, vMean)
				normed := hwy.Mul(diff, vInvStd)

				g := hwy.Load(gamma[ii:])
				result := hwy.Mul(normed, g)
				hwy.Store(result, output[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				normed := (input[off+i] - mean) * invStd
				output[off+i] = normed * gamma[i]
			}
		} else {
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				x := hwy.Load(input[off+ii:])
				diff := hwy.Sub(x, vMean)
				result := hwy.Mul(diff, vInvStd)
				hwy.Store(result, output[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				output[off+i] = (input[off+i] - mean) * invStd
			}
		}
	}
}
