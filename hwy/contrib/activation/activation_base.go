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

package activation

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

//go:generate go run ../../../cmd/hwygen -input activation_base.go -output . -targets avx2,avx512,neon,fallback

// BaseGELU computes the Gaussian Error Linear Unit activation function.
//
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//
// This is the exact GELU formula used in BERT, GPT, and other transformer models.
// For a faster approximation, see BaseGELUApprox.
func BaseGELU[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Constants: 0.5 and 1/sqrt(2) = 0.7071067811865476
	vHalf := hwy.Const[T](0.5)
	vOne := hwy.Const[T](1.0)
	vInvSqrt2 := hwy.Const[T](0.7071067811865476)

	lanes := vOne.NumLanes()
	ii := 0

	// Process full vectors
	for ; ii+lanes <= size; ii += lanes {
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
	}

	// Handle tail elements with scalar math
	for i := ii; i < size; i++ {
		x := float64(input[i])
		output[i] = T(x * 0.5 * (1.0 + stdmath.Erf(x*0.7071067811865476)))
	}
}

// BaseGELUApprox computes a fast approximation of GELU.
//
// Uses the sigmoid approximation: GELU(x) = x * sigmoid(1.702 * x)
//
// This is faster than the exact formula and commonly used in practice.
func BaseGELUApprox[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Constant: 1.702 (the approximation coefficient)
	vCoeff := hwy.Const[T](1.702)

	lanes := vCoeff.NumLanes()
	ii := 0

	// Process full vectors
	for ; ii+lanes <= size; ii += lanes {
		x := hwy.LoadFull(input[ii:])

		// Compute sigmoid(1.702 * x)
		xScaled := hwy.Mul(x, vCoeff)
		sigmoidX := math.BaseSigmoidVec(xScaled)

		// Compute x * sigmoid(1.702 * x)
		result := hwy.Mul(x, sigmoidX)

		hwy.StoreFull(result, output[ii:])
	}

	// Handle tail elements with scalar math
	for i := ii; i < size; i++ {
		x := float64(input[i])
		sigmoid := 1.0 / (1.0 + stdmath.Exp(-1.702*x))
		output[i] = T(x * sigmoid)
	}
}

// BaseReLU computes the Rectified Linear Unit activation: max(0, x).
//
// ReLU is the most common activation function, providing fast computation
// and good gradient flow for positive values.
func BaseReLU[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	vZero := hwy.Const[T](0.0)
	lanes := vZero.NumLanes()
	ii := 0

	// Process full vectors
	for ; ii+lanes <= size; ii += lanes {
		x := hwy.LoadFull(input[ii:])

		// ReLU(x) = max(0, x)
		result := hwy.Max(x, vZero)

		hwy.StoreFull(result, output[ii:])
	}

	// Handle tail elements
	for i := ii; i < size; i++ {
		if input[i] > 0 {
			output[i] = input[i]
		} else {
			output[i] = 0
		}
	}
}

// BaseSiLU computes the Sigmoid Linear Unit (also known as Swish) activation.
//
// SiLU(x) = x * sigmoid(x)
//
// SiLU is used in EfficientNet, GPT-J, and other modern architectures.
// It provides smooth gradients and better optimization than ReLU in some cases.
func BaseSiLU[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	lanes := hwy.MaxLanes[T]()
	ii := 0

	// Process full vectors
	for ; ii+lanes <= size; ii += lanes {
		x := hwy.LoadFull(input[ii:])

		// Compute sigmoid(x)
		sigmoidX := math.BaseSigmoidVec(x)

		// Compute x * sigmoid(x)
		result := hwy.Mul(x, sigmoidX)

		hwy.StoreFull(result, output[ii:])
	}

	// Handle tail elements with scalar math
	for i := ii; i < size; i++ {
		x := float64(input[i])
		sigmoid := 1.0 / (1.0 + stdmath.Exp(-x))
		output[i] = T(x * sigmoid)
	}
}

// BaseLeakyReLU computes the Leaky ReLU activation with a configurable slope.
//
// LeakyReLU(x) = x if x > 0, else alpha * x
//
// This helps prevent "dying ReLU" by allowing small gradients for negative values.
func BaseLeakyReLU[T hwy.Floats](input, output []T, alpha T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	vAlpha := hwy.Set(alpha)
	lanes := hwy.MaxLanes[T]()
	ii := 0

	// Process full vectors
	for ; ii+lanes <= size; ii += lanes {
		x := hwy.LoadFull(input[ii:])

		// Compute alpha * x for the negative part
		negPart := hwy.Mul(x, vAlpha)

		// Select max(x, alpha * x), which gives x for positive, alpha*x for negative
		result := hwy.Max(x, negPart)

		hwy.StoreFull(result, output[ii:])
	}

	// Handle tail elements
	for i := ii; i < size; i++ {
		if input[i] > 0 {
			output[i] = input[i]
		} else {
			output[i] = alpha * input[i]
		}
	}
}

// BaseELU computes the Exponential Linear Unit activation.
//
// ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
//
// ELU has smooth gradients everywhere and can push mean activations toward zero.
func BaseELU[T hwy.Floats](input, output []T, alpha T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	vZero := hwy.Const[T](0.0)
	vOne := hwy.Const[T](1.0)
	vAlpha := hwy.Set(alpha)
	lanes := hwy.MaxLanes[T]()
	ii := 0

	// Process full vectors
	for ; ii+lanes <= size; ii += lanes {
		x := hwy.LoadFull(input[ii:])

		// Compute exp(x) - 1 for negative values
		expX := math.BaseExpVec(x)
		expM1 := hwy.Sub(expX, vOne)
		negPart := hwy.Mul(vAlpha, expM1)

		// Select x for positive, alpha*(exp(x)-1) for negative
		isPositive := hwy.Greater(x, vZero)
		result := hwy.Merge(x, negPart, isPositive)

		hwy.StoreFull(result, output[ii:])
	}

	// Handle tail elements with scalar math
	for i := ii; i < size; i++ {
		if input[i] > 0 {
			output[i] = input[i]
		} else {
			x := float64(input[i])
			output[i] = T(float64(alpha) * (stdmath.Exp(x) - 1.0))
		}
	}
}
