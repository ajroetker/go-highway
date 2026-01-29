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

// This file provides NEON SIMD implementations of shuffle operations.
// These work directly with asm vector types.

// InterleaveLower_NEON_F32x4 interleaves lower halves.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a0,b0,a1,b1]
func InterleaveLower_NEON_F32x4(a, b asm.Float32x4) asm.Float32x4 {
	var dataA, dataB [4]float32
	a.Store(&dataA)
	b.Store(&dataB)
	result := [4]float32{dataA[0], dataB[0], dataA[1], dataB[1]}
	return asm.LoadFloat32x4(&result)
}

// InterleaveLower_NEON_F64x2 interleaves lower halves.
// [a0,a1], [b0,b1] -> [a0,b0]
func InterleaveLower_NEON_F64x2(a, b asm.Float64x2) asm.Float64x2 {
	var dataA, dataB [2]float64
	a.Store(&dataA)
	b.Store(&dataB)
	result := [2]float64{dataA[0], dataB[0]}
	return asm.LoadFloat64x2(&result)
}

// InterleaveUpper_NEON_F32x4 interleaves upper halves.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a2,b2,a3,b3]
func InterleaveUpper_NEON_F32x4(a, b asm.Float32x4) asm.Float32x4 {
	var dataA, dataB [4]float32
	a.Store(&dataA)
	b.Store(&dataB)
	result := [4]float32{dataA[2], dataB[2], dataA[3], dataB[3]}
	return asm.LoadFloat32x4(&result)
}

// InterleaveUpper_NEON_F64x2 interleaves upper halves.
// [a0,a1], [b0,b1] -> [a1,b1]
func InterleaveUpper_NEON_F64x2(a, b asm.Float64x2) asm.Float64x2 {
	var dataA, dataB [2]float64
	a.Store(&dataA)
	b.Store(&dataB)
	result := [2]float64{dataA[1], dataB[1]}
	return asm.LoadFloat64x2(&result)
}
