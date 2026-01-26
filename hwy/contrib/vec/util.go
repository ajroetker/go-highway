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

import (
	"math"
)

// Zero sets every element of the given slice to zero.
func Zero[T ~float32 | ~float64](dst []T) {
	clear(dst)
}

// MinIdx returns the index of the minimum value in the input slice.
// If several entries have the minimum value, the first such index is returned.
// It panics if s is zero length.
//
// Deprecated: Use Argmin instead, which supports all float types and uses SIMD acceleration.
func MinIdx(s []float32) int {
	return Argmin(s)
}

// MaxIdx returns the index of the maximum value in the input slice.
// If several entries have the maximum value, the first such index is returned.
// It panics if s is zero length.
//
// Deprecated: Use Argmax instead, which supports all float types and uses SIMD acceleration.
func MaxIdx(s []float32) int {
	return Argmax(s)
}

// MinIdx64 returns the index of the minimum value in the input slice.
//
// Deprecated: Use Argmin instead, which supports all float types and uses SIMD acceleration.
func MinIdx64(s []float64) int {
	return Argmin(s)
}

// MaxIdx64 returns the index of the maximum value in the input slice.
//
// Deprecated: Use Argmax instead, which supports all float types and uses SIMD acceleration.
func MaxIdx64(s []float64) int {
	return Argmax(s)
}

// EncodeFloat32s encodes a slice of float32 values into a byte slice (little-endian).
func EncodeFloat32s(dst []byte, src []float32) {
	if len(dst) < len(src)*4 {
		panic("dst is too short")
	}
	for i, v := range src {
		bits := math.Float32bits(v)
		dst[i*4] = byte(bits)
		dst[i*4+1] = byte(bits >> 8)
		dst[i*4+2] = byte(bits >> 16)
		dst[i*4+3] = byte(bits >> 24)
	}
}

// DecodeFloat32s decodes a byte slice into a slice of float32 values (little-endian).
func DecodeFloat32s(src []byte, dst []float32) {
	if len(src) < len(dst)*4 {
		panic("src is too short")
	}
	for i := range dst {
		bits := uint32(src[i*4]) | uint32(src[i*4+1])<<8 | uint32(src[i*4+2])<<16 | uint32(src[i*4+3])<<24
		dst[i] = math.Float32frombits(bits)
	}
}
