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
	vHalf := hwy.Set(T(0.5))
	vOne := hwy.Set(T(1.0))
	vInvSqrt2 := hwy.Set(T(0.7071067811865476))

	for ii := 0; ii < size; ii += vOne.NumLanes() {
		remaining := size - ii
		if remaining >= vOne.NumLanes() {
			x := hwy.Load(input[ii:])

			// Compute erf(x / sqrt(2)) = erf(x * invSqrt2)
			xScaled := hwy.Mul(x, vInvSqrt2)
			erfX := math.BaseErfVec(xScaled)

			// Compute 0.5 * (1 + erf(...))
			onePlusErf := hwy.Add(vOne, erfX)
			halfOnePlusErf := hwy.Mul(vHalf, onePlusErf)

			// Compute x * 0.5 * (1 + erf(...))
			result := hwy.Mul(x, halfOnePlusErf)

			hwy.Store(result, output[ii:])
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
	vCoeff := hwy.Set(T(1.702))

	for ii := 0; ii < size; ii += vCoeff.NumLanes() {
		remaining := size - ii
		if remaining >= vCoeff.NumLanes() {
			x := hwy.Load(input[ii:])

			// Compute sigmoid(1.702 * x)
			xScaled := hwy.Mul(x, vCoeff)
			sigmoidX := math.BaseSigmoidVec(xScaled)

			// Compute x * sigmoid(1.702 * x)
			result := hwy.Mul(x, sigmoidX)

			hwy.Store(result, output[ii:])
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
