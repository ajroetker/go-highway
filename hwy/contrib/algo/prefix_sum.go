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

// DeltaEncode computes adjacent differences in place.
// Result[i] = data[i] - data[i-1] for i > 0, Result[0] = data[0] - base.
//
// This is the inverse of DeltaDecode/PrefixSum.
// Commonly used for preparing monotonically increasing data (like positions)
// for more compact encoding.
//
// Example:
//
//	// Document positions: 13, 15, 20, 21
//	// We want to encode as deltas from base=10
//	data := []uint64{13, 15, 20, 21}
//	DeltaEncode(data, 10)
//	// data = [3, 2, 5, 1] (deltas)
//
// If you need to preserve the original, copy first:
//
//	result := slices.Clone(src)
//	DeltaEncode(result, base)
//
// Note: Unlike DeltaDecode which benefits from SIMD prefix sum,
// DeltaEncode uses a scalar reverse-order algorithm. This is because
// computing differences has a boundary dependency between SIMD chunks
// that makes vectorization complex. The scalar reverse-order approach
// is already very fast (no loop-carried dependencies) and allows
// modern CPUs to execute multiple iterations in parallel via ILP.
func DeltaEncode[T hwy.Integers](data []T, base T) {
	n := len(data)
	if n == 0 {
		return
	}

	// Process backwards to avoid overwriting values we still need.
	// This also eliminates loop-carried dependencies, allowing ILP.
	for i := n - 1; i > 0; i-- {
		data[i] -= data[i-1]
	}
	data[0] -= base
}
