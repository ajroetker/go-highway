// Package softmax demonstrates the hwygen workflow with a Softmax implementation.
//
// This example shows how to write portable SIMD code that gets transformed
// into target-specific implementations (AVX2, AVX-512, fallback).
//
// Usage:
//
//	go generate ./...
//	GOEXPERIMENT=simd go build
package softmax

//go:generate go run ../../cmd/hwygen -input softmax.go -output . -targets avx2,avx512,neon,fallback

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/algo"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

// BaseSoftmax computes the softmax function over the input slice.
//
// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// The max subtraction provides numerical stability by preventing overflow
// in the exponential computation.
//
// This function uses algo.BaseApply for efficient SIMD processing.
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

// BaseSoftmaxScalar is a scalar reference implementation for comparison.
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
