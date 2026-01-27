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

package algo

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input apply_base.go -output . -targets avx2,avx512,neon,fallback

// BaseApply transforms input slice to output slice using the provided vector function.
// Tail elements are handled via buffer-based SIMD processing - no scalar fallback needed.
//
// This is the core primitive for zero-allocation bulk transforms.
// The fn parameter is a Vec[T] -> Vec[T] function that processes one vector at a time.
//
// Example usage:
//
//	Apply(input, output, math.BaseExpVec)
//
func BaseApply[T hwy.Floats](in, out []T, fn func(hwy.Vec[T]) hwy.Vec[T]) {
	n := min(len(in), len(out))
	lanes := hwy.MaxLanes[T]()
	i := 0

	// Process full vectors
	for ; i+lanes <= n; i += lanes {
		x := hwy.LoadFull(in[i:])
		hwy.StoreFull(fn(x), out[i:])
	}

	// Buffer-based tail handling
	if remaining := n - i; remaining > 0 {
		buf := make([]T, lanes)
		copy(buf, in[i:i+remaining])
		x := hwy.Load(buf)
		hwy.Store(fn(x), buf)
		copy(out[i:i+remaining], buf[:remaining])
	}
}
