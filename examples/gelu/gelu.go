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

// Package gelu demonstrates the hwygen workflow with a GELU activation function.
//
// GELU (Gaussian Error Linear Unit) is a popular activation function used
// in transformer models like BERT and GPT.
//
// Usage:
//
//	go generate ./...
//	GOEXPERIMENT=simd go build
package gelu

//go:generate go run ../../cmd/hwygen -input gelu.go -output . -targets avx2,avx512,neon,fallback

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

// BaseGELU computes the Gaussian Error Linear Unit activation function.
//
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//
// This is the exact GELU formula. For a faster approximation, see BaseGELUApprox.
//
// GELU is widely used in transformer architectures (BERT, GPT, etc.) as it
// provides smoother gradients than ReLU while maintaining similar properties.
func BaseGELU[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Constants: 0.5 and 1/sqrt(2) ≈ 0.7071067811865476
	// Use hwy.Const to handle float64-to-T conversion for all float types including Float16/BFloat16
	vHalf := hwy.Const[T](0.5)
	vOne := hwy.Const[T](1.0)
	vInvSqrt2 := hwy.Const[T](0.7071067811865476)

	for ii := 0; ii < size; ii += vOne.NumLanes() {
		remaining := size - ii
		if remaining >= vOne.NumLanes() {
			x := hwy.LoadFull(input[ii:])

			// Compute erf(x / sqrt(2)) = erf(x * invSqrt2)
			xScaled := hwy.Mul(x, vInvSqrt2)
			erfX := math.BaseErfVec(xScaled)

			// Compute 0.5 * (1 + erf(...))
			onePlusErf := hwy.Add(vOne, erfX)
			halfOnePlusErf := hwy.Mul(vHalf, onePlusErf)

			// Compute x * 0.5 * (1 + erf(...))
			result := hwy.Mul(x, halfOnePlusErf)

			hwy.StoreFull(result, output[ii:])
		} else {
			// Handle tail elements with scalar math
			for i := ii; i < size; i++ {
				x := float64(input[i])
				output[i] = T(x * 0.5 * (1.0 + stdmath.Erf(x*0.7071067811865476)))
			}
		}
	}
}

// BaseGELUApprox computes a fast approximation of GELU.
//
// Uses the sigmoid approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
//
// This is faster than the exact formula and is commonly used in practice.
func BaseGELUApprox[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Constant: 1.702 (the approximation coefficient)
	// Use hwy.Const to handle float64-to-T conversion for all float types including Float16/BFloat16
	vCoeff := hwy.Const[T](1.702)

	for ii := 0; ii < size; ii += vCoeff.NumLanes() {
		remaining := size - ii
		if remaining >= vCoeff.NumLanes() {
			x := hwy.LoadFull(input[ii:])

			// Compute sigmoid(1.702 * x)
			xScaled := hwy.Mul(x, vCoeff)
			sigmoidX := math.BaseSigmoidVec(xScaled)

			// Compute x * sigmoid(1.702 * x)
			result := hwy.Mul(x, sigmoidX)

			hwy.StoreFull(result, output[ii:])
		} else {
			// Handle tail elements with scalar math
			for i := ii; i < size; i++ {
				x := float64(input[i])
				sigmoid := 1.0 / (1.0 + stdmath.Exp(-1.702*x))
				output[i] = T(x * sigmoid)
			}
		}
	}
}
