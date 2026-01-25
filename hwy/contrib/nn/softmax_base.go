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
	"github.com/ajroetker/go-highway/hwy/contrib/algo"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

//go:generate go run ../../../cmd/hwygen -input softmax_base.go -output . -targets avx2,avx512,neon,fallback

// BaseSoftmax computes the softmax function over the input slice.
//
// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// The max subtraction provides numerical stability by preventing overflow
// in the exponential computation.
//
// This function uses SIMD-accelerated exp for efficient processing.
func BaseSoftmax[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Step 1: Find the maximum value for numerical stability
	maxVal := input[0]
	for i := 1; i < size; i++ {
		if input[i] > maxVal {
			maxVal = input[i]
		}
	}

	// Step 2: Subtract max from input (for numerical stability)
	shifted := make([]T, size)
	for i := range size {
		shifted[i] = input[i] - maxVal
	}

	// Step 3: Compute exp(shifted) using SIMD via BaseApply
	algo.BaseApply(shifted, output, math.BaseExpVec[T])

	// Step 4: Compute sum of exponentials
	var expSum T
	for i := range size {
		expSum += output[i]
	}

	// Step 5: Normalize by dividing by sum
	invSum := T(1.0) / expSum
	for i := range size {
		output[i] = output[i] * invSum
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
func BaseLogSoftmax[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Step 1: Find the maximum value for numerical stability
	maxVal := input[0]
	for i := 1; i < size; i++ {
		if input[i] > maxVal {
			maxVal = input[i]
		}
	}

	// Step 2: Compute shifted values and their exp
	shifted := make([]T, size)
	expVals := make([]T, size)
	for i := range size {
		shifted[i] = input[i] - maxVal
	}

	// Step 3: Compute exp(shifted) using SIMD
	algo.BaseApply(shifted, expVals, math.BaseExpVec[T])

	// Step 4: Compute sum of exponentials
	var expSum T
	for i := range size {
		expSum += expVals[i]
	}

	// Step 5: Compute log_softmax = shifted - log(sum_exp)
	logSumExp := T(stdmath.Log(float64(expSum)))
	for i := range size {
		output[i] = shifted[i] - logSumExp
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
	for i := 1; i < size; i++ {
		if input[i] > maxVal {
			maxVal = input[i]
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
func BaseSoftmaxWithTemperature[T hwy.Floats](input, output []T, temperature T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Step 1: Find the maximum value
	maxVal := input[0]
	for i := 1; i < size; i++ {
		if input[i] > maxVal {
			maxVal = input[i]
		}
	}

	// Step 2: Compute (x - max) / temperature
	invTemp := T(1.0) / temperature
	shifted := make([]T, size)
	for i := range size {
		shifted[i] = (input[i] - maxVal) * invTemp
	}

	// Step 3: Compute exp(shifted) using SIMD
	algo.BaseApply(shifted, output, math.BaseExpVec[T])

	// Step 4: Compute sum and normalize
	var expSum T
	for i := range size {
		expSum += output[i]
	}

	invSum := T(1.0) / expSum
	for i := range size {
		output[i] = output[i] * invSum
	}
}
