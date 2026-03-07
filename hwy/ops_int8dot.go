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

package hwy

// DotProductSS computes widening 4-group dot product for signed int8.
// Each group of 4 adjacent int8 elements is multiplied and summed into a single int32.
// The result vector has len(a)/4 elements.
//
// Maps to vdotq_s32 (NEON DOTPROD) / VPDPBSSD (AVX-VNNI).
func DotProductSS(a, b Vec[int8]) Vec[int32] {
	n := len(a.data) / 4
	result := make([]int32, n)
	for i := range n {
		base := i * 4
		result[i] = int32(a.data[base])*int32(b.data[base]) +
			int32(a.data[base+1])*int32(b.data[base+1]) +
			int32(a.data[base+2])*int32(b.data[base+2]) +
			int32(a.data[base+3])*int32(b.data[base+3])
	}
	return Vec[int32]{data: result}
}

// DotProductUU computes widening 4-group dot product for unsigned uint8.
// Each group of 4 adjacent uint8 elements is multiplied and summed into a single uint32.
// The result vector has len(a)/4 elements.
//
// Maps to vdotq_u32 (NEON DOTPROD) / VPDPBUUD (AVX-VNNI).
func DotProductUU(a, b Vec[uint8]) Vec[uint32] {
	n := len(a.data) / 4
	result := make([]uint32, n)
	for i := range n {
		base := i * 4
		result[i] = uint32(a.data[base])*uint32(b.data[base]) +
			uint32(a.data[base+1])*uint32(b.data[base+1]) +
			uint32(a.data[base+2])*uint32(b.data[base+2]) +
			uint32(a.data[base+3])*uint32(b.data[base+3])
	}
	return Vec[uint32]{data: result}
}

// DotProduct computes widening 4-group dot product for int8 or uint8 inputs.
// This is the generic wrapper for use in base functions — the C translator maps it
// to vdotq_s32 (int8) or vdotq_u32 (uint8) based on the element type.
//
// Returns int32 for both signed and unsigned inputs. This is safe because
// the maximum per-group value is 4*255*255 = 260100, well within int32 range.
func DotProduct[T int8 | uint8](a, b Vec[T]) Vec[int32] {
	n := len(a.data) / 4
	result := make([]int32, n)
	for i := range n {
		base := i * 4
		result[i] = int32(a.data[base])*int32(b.data[base]) +
			int32(a.data[base+1])*int32(b.data[base+1]) +
			int32(a.data[base+2])*int32(b.data[base+2]) +
			int32(a.data[base+3])*int32(b.data[base+3])
	}
	return Vec[int32]{data: result}
}
