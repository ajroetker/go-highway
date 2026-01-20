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

package vec

//go:generate go run ../../../cmd/hwygen -input normalize_base.go -output . -targets avx2,avx512,neon,fallback -dispatch normalize

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseNormalize normalizes a vector in-place to unit length (L2 norm = 1).
// The L2 norm is defined as sqrt(sum of squares): ||v|| = sqrt(Σ v[i]^2).
//
// If the vector is empty or has zero norm (all zeros), it is left unchanged.
// This prevents division by zero while preserving the zero vector.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Example:
//
//	v := []float32{3, 0, 4}
//	BaseNormalize(v)  // v is now [0.6, 0, 0.8] since ||[3,0,4]|| = 5
func BaseNormalize[T hwy.Floats](dst []T) {
	if len(dst) == 0 {
		return
	}

	// Compute squared norm using Dot(v, v) for consistency with how norms are
	// computed elsewhere. This ensures the same precision characteristics.
	squaredNorm := BaseDot(dst, dst)

	// If squared norm is zero, leave unchanged
	if squaredNorm == 0 {
		return
	}

	// Compute scale factor: 1/norm
	// For all types, compute in the native precision then convert
	norm := T(stdmath.Sqrt(float64(squaredNorm)))
	scale := T(1) / norm

	// Create scale vector - hwy.Set takes value of type T
	scaleVec := hwy.Set(scale)
	lanes := scaleVec.NumLanes()

	var i int
	for i = 0; i+lanes <= len(dst); i += lanes {
		vec := hwy.Load(dst[i:])
		result := hwy.Mul(vec, scaleVec)
		hwy.Store(result, dst[i:])
	}

	// Handle tail elements
	for ; i < len(dst); i++ {
		dst[i] *= scale
	}
}

// BaseNormalizeTo normalizes src and stores the result in dst.
// The L2 norm is defined as sqrt(sum of squares): ||v|| = sqrt(Σ v[i]^2).
//
// Uses the minimum of len(dst) and len(src) as the effective length.
// If the source vector is empty or has zero norm, dst is filled with the
// source values unchanged (for zero norm) or left unchanged (for empty).
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Example:
//
//	src := []float32{3, 0, 4}
//	dst := make([]float32, 3)
//	BaseNormalizeTo(dst, src)  // dst is now [0.6, 0, 0.8]
func BaseNormalizeTo[T hwy.Floats](dst, src []T) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}

	// Compute squared norm using Dot(v, v) for consistency with how norms are
	// computed elsewhere. This ensures the same precision characteristics.
	squaredNorm := BaseDot(src[:n], src[:n])

	// If squared norm is zero, copy src to dst unchanged
	if squaredNorm == 0 {
		copy(dst[:n], src[:n])
		return
	}

	// Compute scale factor: 1/norm
	norm := T(stdmath.Sqrt(float64(squaredNorm)))
	scale := T(1) / norm

	// Create scale vector
	scaleVec := hwy.Set(scale)
	lanes := scaleVec.NumLanes()

	var i int
	for i = 0; i+lanes <= n; i += lanes {
		vec := hwy.Load(src[i:])
		result := hwy.Mul(vec, scaleVec)
		hwy.Store(result, dst[i:])
	}

	// Handle tail elements
	for ; i < n; i++ {
		dst[i] = src[i] * scale
	}
}
