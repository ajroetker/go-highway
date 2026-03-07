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

//go:generate go run ../../../cmd/hwygen -input dot_int_base.go -output . -targets neon:asm,fallback -dispatch dotint

import "github.com/ajroetker/go-highway/hwy"

// BaseDotInt computes the integer dot product of two int8 or uint8 slices,
// returning the sum as int32. Uses widening 4-group dot product instructions
// (vdotq_s32 on NEON DOTPROD, VPDPBSSD on AVX-VNNI) when available.
//
// Each group of 4 adjacent elements is multiplied pairwise and summed into
// a single int32, then accumulated across the vector. The scalar tail handles
// remaining elements that don't fill a full SIMD vector.
//
// Example:
//
//	a := []int8{1, 2, 3, 4, 5, 6, 7, 8}
//	b := []int8{1, 1, 1, 1, 1, 1, 1, 1}
//	result := DotInt(a, b)  // 1+2+3+4+5+6+7+8 = 36
//
//hwy:gen T={int8, uint8}
func BaseDotInt[T int8 | uint8](a, b []T) int32 {
	n := min(len(a), len(b))
	lanes := hwy.NumLanes[T]()
	acc := hwy.Zero[int32]()

	var i int
	for i = 0; i+lanes <= n; i += lanes {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		acc = hwy.Add(acc, hwy.DotProduct(va, vb))
	}

	result := hwy.ReduceSum(acc)

	for ; i < n; i++ {
		result += int32(a[i]) * int32(b[i])
	}
	return result
}
