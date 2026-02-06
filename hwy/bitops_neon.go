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
	"math/bits"
	"unsafe"

	"github.com/ajroetker/go-highway/hwy/asm"
)

// This file provides NEON SIMD implementations of bit manipulation operations.
// NEON doesn't have native SIMD popcount instructions for 64-bit lanes,
// so we use store/scalar/load pattern.

// PopCount_NEON_Uint32x4 counts set bits in each lane (unsigned).
func PopCount_NEON_Uint32x4(v asm.Uint32x4) asm.Uint32x4 {
	data := *(*[4]uint32)(unsafe.Pointer(&v))
	for i := range 4 {
		data[i] = uint32(bits.OnesCount32(data[i]))
	}
	return *(*asm.Uint32x4)(unsafe.Pointer(&data))
}

// PopCount_NEON_Uint64x2 counts set bits in each lane (unsigned).
func PopCount_NEON_Uint64x2(v asm.Uint64x2) asm.Uint64x2 {
	data := *(*[2]uint64)(unsafe.Pointer(&v))
	for i := range 2 {
		data[i] = uint64(bits.OnesCount64(data[i]))
	}
	return *(*asm.Uint64x2)(unsafe.Pointer(&data))
}

// PopCount_NEON_Int32x4 counts set bits in each lane (signed).
func PopCount_NEON_Int32x4(v asm.Int32x4) asm.Int32x4 {
	data := *(*[4]int32)(unsafe.Pointer(&v))
	for i := range 4 {
		data[i] = int32(bits.OnesCount32(uint32(data[i])))
	}
	return *(*asm.Int32x4)(unsafe.Pointer(&data))
}

// PopCount_NEON_Int64x2 counts set bits in each lane (signed).
func PopCount_NEON_Int64x2(v asm.Int64x2) asm.Int64x2 {
	data := *(*[2]int64)(unsafe.Pointer(&v))
	for i := range 2 {
		data[i] = int64(bits.OnesCount64(uint64(data[i])))
	}
	return *(*asm.Int64x2)(unsafe.Pointer(&data))
}
