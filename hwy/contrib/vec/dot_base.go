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

//go:generate go run ../../../cmd/hwygen -input dot_base.go -output . -targets avx2,avx512,neon,fallback -dispatch dot

import "github.com/ajroetker/go-highway/hwy"

// BaseDot computes the dot product (inner product) of two vectors using hwy primitives.
// The result is the sum of element-wise products: Σ(a[i] * b[i]).
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
//	result := Dot(a, b)  // 1*4 + 2*5 + 3*6 = 32
func BaseDot[T hwy.Floats](a, b []T) T {
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

		// Accumulate products using FMA: sum += a * b
		sum0 = hwy.MulAdd(va0, vb0, sum0)
		sum1 = hwy.MulAdd(va1, vb1, sum1)
		sum2 = hwy.MulAdd(va2, vb2, sum2)
		sum3 = hwy.MulAdd(va3, vb3, sum3)
	}

	// Process remaining full vectors (1 at a time)
	for i+lanes <= n {
		va := hwy.LoadFull(a[i:])
		vb := hwy.LoadFull(b[i:])
		sum0 = hwy.MulAdd(va, vb, sum0)
		i += lanes
	}

	// Combine accumulators and reduce to scalar
	sum0 = hwy.Add(sum0, sum1)
	sum2 = hwy.Add(sum2, sum3)
	sum0 = hwy.Add(sum0, sum2)
	result := hwy.ReduceSum(sum0)

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		result += a[i] * b[i]
	}

	return result
}
