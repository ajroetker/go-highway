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

//go:generate go run ../../../cmd/hwygen -input norm_base.go -output . -targets avx2,avx512,neon,fallback -dispatch norm

import (
	"math"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseSquaredNorm computes the squared L2 norm (sum of squares) of a vector
// using hwy primitives.
// The result is equivalent to Dot(v, v): Σ(v[i] * v[i]).
//
// Returns 0 if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Example:
//
//	v := []float32{3, 4}
//	result := SquaredNorm(v)  // 3*3 + 4*4 = 25
func BaseSquaredNorm[T hwy.Floats](v []T) T {
	// Use Dot(v, v) for consistency with how norms are typically computed.
	// This ensures the same precision characteristics as dot product operations.
	return BaseDot(v, v)
}

// BaseNorm computes the L2 norm (Euclidean magnitude) of a vector using hwy
// primitives.
// The result is Sqrt(Σ(v[i] * v[i])).
//
// Returns 0 if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Example:
//
//	v := []float32{3, 4}
//	result := Norm(v)  // Sqrt(3*3 + 4*4) = Sqrt(25) = 5
func BaseNorm[T hwy.Floats](v []T) T {
	squaredNorm := BaseSquaredNorm(v)
	if squaredNorm == 0 {
		return 0
	}

	// Take square root of the squared norm
	// Use standard math library for the final scalar result
	switch any(squaredNorm).(type) {
	case float32:
		return any(float32(math.Sqrt(float64(any(squaredNorm).(float32))))).(T)
	case float64:
		return any(math.Sqrt(any(squaredNorm).(float64))).(T)
	default:
		// For Float16/BFloat16, convert through float32
		return any(float32(math.Sqrt(float64(any(squaredNorm).(float32))))).(T)
	}
}
