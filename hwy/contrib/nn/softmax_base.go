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

//go:generate go run ../../../cmd/hwygen -input softmax_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch softmax

// BaseSoftmax computes the softmax function over the input slice.
//
// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// The max subtraction provides numerical stability by preventing overflow
// in the exponential computation.
//
// Uses a fused three-pass algorithm:
//  1. Find max (scalar)
//  2. Fused subtract-max + exp + accumulate sum (SIMD)
//  3. Normalize by 1/sum (SIMD)
func BaseSoftmax[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Pass 1: Find the maximum value for numerical stability
	maxVal := input[0]
	for j := 1; j < size; j++ {
		if input[j] > maxVal {
			maxVal = input[j]
		}
	}

	// Pass 2: Fused subtract-max + exp + accumulate sum
	vMax := hwy.Set(maxVal)
	vSum := hwy.Zero[T]()
	lanes := vSum.NumLanes()

	var ii int
	for ii = 0; ii+lanes <= size; ii += lanes {
		x := hwy.Load(input[ii:])
		shifted := hwy.Sub(x, vMax)
		expVal := math.BaseExpVec(shifted)
		hwy.Store(expVal, output[ii:])
		vSum = hwy.Add(vSum, expVal)
	}
	expSum := hwy.ReduceSum(vSum)
	for ; ii < size; ii++ {
		expVal := T(stdmath.Exp(float64(input[ii] - maxVal)))
		output[ii] = expVal
		expSum += expVal
	}

	// Pass 3: Normalize by dividing by sum
	invSum := T(1.0) / expSum
	vInvSum := hwy.Set(invSum)
	for ii = 0; ii+lanes <= size; ii += lanes {
		v := hwy.Load(output[ii:])
		hwy.Store(hwy.Mul(v, vInvSum), output[ii:])
	}
	for ; ii < size; ii++ {
		output[ii] *= invSum
	}
}

// BaseSoftmaxInPlace applies softmax in-place, modifying the input slice.
func BaseSoftmaxInPlace[T hwy.Floats](x []T) {
	BaseSoftmax(x, x)
}

// BaseLogSoftmax computes the log-softmax function over the input slice.
//
// log_softmax(x_i) = x_i - max(x) - log(sum(exp(x_j - max(x))))
//
// This is more numerically stable than computing log(softmax(x)) directly,
// and is commonly used for negative log-likelihood loss computation.
//
// Uses a fused three-pass algorithm:
//  1. Find max (scalar)
//  2. Fused subtract-max + exp + accumulate sum (SIMD, no intermediate storage)
//  3. output[i] = input[i] - max - log(sum) (SIMD)
func BaseLogSoftmax[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Pass 1: Find the maximum value for numerical stability
	maxVal := input[0]
	for j := 1; j < size; j++ {
		if input[j] > maxVal {
			maxVal = input[j]
		}
	}

	// Pass 2: Compute sum of exp(input[i] - max) without storing exp values
	vMax := hwy.Set(maxVal)
	vSum := hwy.Zero[T]()
	lanes := vSum.NumLanes()

	var ii int
	for ii = 0; ii+lanes <= size; ii += lanes {
		x := hwy.Load(input[ii:])
		shifted := hwy.Sub(x, vMax)
		expVal := math.BaseExpVec(shifted)
		vSum = hwy.Add(vSum, expVal)
	}
	expSum := hwy.ReduceSum(vSum)
	for ; ii < size; ii++ {
		expSum += T(stdmath.Exp(float64(input[ii] - maxVal)))
	}

	// Pass 3: output[i] = input[i] - max - log(sum_exp)
	logSumExp := T(stdmath.Log(float64(expSum)))
	offset := maxVal + logSumExp
	vOffset := hwy.Set(offset)
	for ii = 0; ii+lanes <= size; ii += lanes {
		x := hwy.Load(input[ii:])
		hwy.Store(hwy.Sub(x, vOffset), output[ii:])
	}
	for ; ii < size; ii++ {
		output[ii] = input[ii] - offset
	}
}

// BaseLogSoftmaxInPlace applies log-softmax in-place, modifying the input slice.
func BaseLogSoftmaxInPlace[T hwy.Floats](x []T) {
	BaseLogSoftmax(x, x)
}

// BaseSoftmaxScalar is a scalar reference implementation for comparison and testing.
func BaseSoftmaxScalar[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Find max
	maxVal := input[0]
	for j := 1; j < size; j++ {
		if input[j] > maxVal {
			maxVal = input[j]
		}
	}

	// Compute exp and sum
	var expSum T
	for i := range size {
		output[i] = T(stdmath.Exp(float64(input[i] - maxVal)))
		expSum += output[i]
	}

	// Normalize
	invSum := T(1.0) / expSum
	for i := range size {
		output[i] = output[i] * invSum
	}
}

// BaseSoftmaxWithTemperature computes softmax with a temperature parameter.
//
// softmax(x_i / T) = exp((x_i - max(x)) / T) / sum(exp((x_j - max(x)) / T))
//
// Temperature controls the "sharpness" of the distribution:
//   - T < 1: sharper (more confident, closer to argmax)
//   - T = 1: standard softmax
//   - T > 1: softer (more uniform)
//
// Uses a fused three-pass algorithm:
//  1. Find max (scalar)
//  2. Fused (subtract-max * invTemp) + exp + accumulate sum (SIMD)
//  3. Normalize by 1/sum (SIMD)
func BaseSoftmaxWithTemperature[T hwy.Floats](input, output []T, temperature T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Pass 1: Find the maximum value
	maxVal := input[0]
	for j := 1; j < size; j++ {
		if input[j] > maxVal {
			maxVal = input[j]
		}
	}

	// Pass 2: Fused (x - max) / temperature + exp + accumulate sum
	invTemp := T(1.0) / temperature
	vMax := hwy.Set(maxVal)
	vInvTemp := hwy.Set(invTemp)
	vSum := hwy.Zero[T]()
	lanes := vSum.NumLanes()

	var ii int
	for ii = 0; ii+lanes <= size; ii += lanes {
		x := hwy.Load(input[ii:])
		shifted := hwy.Mul(hwy.Sub(x, vMax), vInvTemp)
		expVal := math.BaseExpVec(shifted)
		hwy.Store(expVal, output[ii:])
		vSum = hwy.Add(vSum, expVal)
	}
	expSum := hwy.ReduceSum(vSum)
	for ; ii < size; ii++ {
		shifted := (input[ii] - maxVal) * invTemp
		expVal := T(stdmath.Exp(float64(shifted)))
		output[ii] = expVal
		expSum += expVal
	}

	// Pass 3: Normalize by dividing by sum
	invSum := T(1.0) / expSum
	vInvSum := hwy.Set(invSum)
	for ii = 0; ii+lanes <= size; ii += lanes {
		v := hwy.Load(output[ii:])
		hwy.Store(hwy.Mul(v, vInvSum), output[ii:])
	}
	for ; ii < size; ii++ {
		output[ii] *= invSum
	}
}
