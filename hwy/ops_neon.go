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

//go:build arm64

package hwy

import (
	"github.com/ajroetker/go-highway/hwy/asm"
)

// This file provides NEON SIMD operations that work directly with
// asm vector types. These are needed for operations that aren't available
// as methods on the asm types.

// RSqrtNewtonRaphson_NEON_F32x4 computes 1/sqrt(x) with one Newton-Raphson refinement.
// Provides ~23-bit precision (sufficient for float32).
// Formula: y = y * (1.5 - 0.5 * x * y * y)
func RSqrtNewtonRaphson_NEON_F32x4(x asm.Float32x4) asm.Float32x4 {
	half := asm.BroadcastFloat32x4(0.5)
	threeHalf := asm.BroadcastFloat32x4(1.5)

	// Initial approximation
	y := x.ReciprocalSqrt()

	// One Newton-Raphson iteration: y = y * (1.5 - 0.5 * x * y * y)
	xHalf := x.Mul(half)
	yy := y.Mul(y)
	xyy := xHalf.Mul(yy)
	correction := threeHalf.Sub(xyy)
	return y.Mul(correction)
}

// RSqrtNewtonRaphson_NEON_F64x2 computes 1/sqrt(x) with one Newton-Raphson refinement.
// Formula: y = y * (1.5 - 0.5 * x * y * y)
func RSqrtNewtonRaphson_NEON_F64x2(x asm.Float64x2) asm.Float64x2 {
	half := asm.BroadcastFloat64x2(0.5)
	threeHalf := asm.BroadcastFloat64x2(1.5)

	// Initial approximation (~8-bit precision from vrsqrteq_f64)
	y := x.ReciprocalSqrt()

	// One Newton-Raphson iteration: y = y * (1.5 - 0.5 * x * y * y)
	xHalf := x.Mul(half)
	yy := y.Mul(y)
	xyy := xHalf.Mul(yy)
	correction := threeHalf.Sub(xyy)
	return y.Mul(correction)
}

// RSqrtPrecise_NEON_F32x4 computes precise 1/sqrt(x) via sqrt + reciprocal.
func RSqrtPrecise_NEON_F32x4(x asm.Float32x4) asm.Float32x4 {
	one := asm.BroadcastFloat32x4(1.0)
	sqrtX := x.Sqrt()
	return one.Div(sqrtX)
}

// RSqrtPrecise_NEON_F64x2 computes precise 1/sqrt(x) via sqrt + division.
func RSqrtPrecise_NEON_F64x2(x asm.Float64x2) asm.Float64x2 {
	one := asm.BroadcastFloat64x2(1.0)
	sqrtX := x.Sqrt()
	return one.Div(sqrtX)
}
