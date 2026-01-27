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

//go:generate go run ../../../cmd/hwygen -input distance_base.go -output . -targets avx2,avx512,neon,fallback -dispatch distance

import (
	"math"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseL2SquaredDistance computes the squared Euclidean distance between two slices.
// The result is the sum of squared differences: sum((a[i] - b[i])^2).
//
// If the slices have different lengths, the computation uses the minimum length.
// Returns 0 if either slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Example:
//
//	a := []float32{1, 2, 3}
//	b := []float32{4, 5, 6}
//	result := L2SquaredDistance(a, b)  // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
func BaseL2SquaredDistance[T hwy.Floats](a, b []T) T {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	n := min(len(a), len(b))

	// Use 4 accumulators for better instruction-level parallelism
	sum0 := hwy.Zero[T]()
	sum1 := hwy.Zero[T]()
	sum2 := hwy.Zero[T]()
	sum3 := hwy.Zero[T]()
	lanes := sum0.NumLanes()

	// Process 4 vectors at a time (4x loop unrolling)
	// On ARM NEON, Load4 maps to a single ld1 {v0,v1,v2,v3} instruction
	var i int
	stride := lanes * 4
	for i = 0; i+stride <= n; i += stride {
		// Load 4 consecutive vectors from each slice
		va0, va1, va2, va3 := hwy.Load4(a[i:])
		vb0, vb1, vb2, vb3 := hwy.Load4(b[i:])

		// Compute differences
		diff0 := hwy.Sub(va0, vb0)
		diff1 := hwy.Sub(va1, vb1)
		diff2 := hwy.Sub(va2, vb2)
		diff3 := hwy.Sub(va3, vb3)

		// Accumulate squared differences using FMA: sum += diff * diff
		sum0 = hwy.MulAdd(diff0, diff0, sum0)
		sum1 = hwy.MulAdd(diff1, diff1, sum1)
		sum2 = hwy.MulAdd(diff2, diff2, sum2)
		sum3 = hwy.MulAdd(diff3, diff3, sum3)
	}

	// Process remaining full vectors (1 at a time)
	for i+lanes <= n {
		va := hwy.LoadFull(a[i:])
		vb := hwy.LoadFull(b[i:])
		diff := hwy.Sub(va, vb)
		sum0 = hwy.MulAdd(diff, diff, sum0)
		i += lanes
	}

	// Combine accumulators and reduce to scalar
	sum0 = hwy.Add(sum0, sum1)
	sum2 = hwy.Add(sum2, sum3)
	sum0 = hwy.Add(sum0, sum2)
	result := hwy.ReduceSum(sum0)

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		d := a[i] - b[i]
		result += d * d
	}

	return result
}

// BaseL2Distance computes the Euclidean distance (L2 norm) between two slices.
// The result is the square root of the sum of squared differences: sqrt(sum((a[i] - b[i])^2)).
//
// If the slices have different lengths, the computation uses the minimum length.
// Returns 0 if either slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Example:
//
//	a := []float32{1, 2, 3}
//	b := []float32{4, 5, 6}
//	result := L2Distance(a, b)  // sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt(27) ≈ 5.196
func BaseL2Distance[T hwy.Floats](a, b []T) T {
	sqDist := BaseL2SquaredDistance(a, b)
	return T(math.Sqrt(float64(sqDist)))
}
