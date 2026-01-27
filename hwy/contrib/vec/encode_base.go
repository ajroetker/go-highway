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

//go:generate go run ../../../cmd/hwygen -input encode_base.go -output . -targets avx2,avx512,neon,fallback -dispatch encode

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseEncodeFloat32s encodes float32 values to little-endian bytes using SIMD.
// dst must have length >= len(src) * 4.
// Uses unsafe.Slice for zero-cost reinterpretation (ARM64/x86-64 are little-endian).
//
// This is SIMD-accelerated and provides 4-9x speedup over scalar implementations
// (NEON: ~4x, AVX2: ~9x).
func BaseEncodeFloat32s(dst []byte, src []float32) {
	if len(src) == 0 {
		return
	}

	totalBytes := len(src) * 4
	if len(dst) < totalBytes {
		panic("dst is too short")
	}

	// Reinterpret float32 slice as bytes (zero-cost on little-endian)
	srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), totalBytes)

	// SIMD copy using byte vectors
	// Use [:16] slice bounds to hint 128-bit operations to hwygen
	lanes := hwy.NumLanes[uint8]()
	i := 0

	// Process full vectors
	for ; i+lanes <= totalBytes; i += lanes {
		v := hwy.LoadFull[uint8](srcBytes[i:])
		hwy.Store(v, dst[i:])
	}

	// Scalar tail
	for ; i < totalBytes; i++ {
		dst[i] = srcBytes[i]
	}
}

// BaseDecodeFloat32s decodes little-endian bytes to float32 values using SIMD.
// src must have length >= len(dst) * 4.
// Uses unsafe.Slice for zero-cost reinterpretation (ARM64/x86-64 are little-endian).
//
// This is SIMD-accelerated and provides 4-9x speedup over scalar implementations
// (NEON: ~4x, AVX2: ~9x).
func BaseDecodeFloat32s(dst []float32, src []byte) {
	if len(dst) == 0 {
		return
	}

	totalBytes := len(dst) * 4
	if len(src) < totalBytes {
		panic("src is too short")
	}

	// Reinterpret float32 slice as bytes (zero-cost on little-endian)
	dstBytes := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), totalBytes)

	// SIMD copy using byte vectors
	// Use [:16] slice bounds to hint 128-bit operations to hwygen
	lanes := hwy.NumLanes[uint8]()
	i := 0

	// Process full vectors
	for ; i+lanes <= totalBytes; i += lanes {
		v := hwy.LoadFull[uint8](src[i:])
		hwy.Store(v, dstBytes[i:])
	}

	// Scalar tail
	for ; i < totalBytes; i++ {
		dstBytes[i] = src[i]
	}
}

// BaseEncodeFloat64s encodes float64 values to little-endian bytes using SIMD.
// dst must have length >= len(src) * 8.
// Uses unsafe.Slice for zero-cost reinterpretation (ARM64/x86-64 are little-endian).
func BaseEncodeFloat64s(dst []byte, src []float64) {
	if len(src) == 0 {
		return
	}

	totalBytes := len(src) * 8
	if len(dst) < totalBytes {
		panic("dst is too short")
	}

	// Reinterpret float64 slice as bytes (zero-cost on little-endian)
	srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), totalBytes)

	// SIMD copy using byte vectors
	lanes := hwy.NumLanes[uint8]()
	i := 0

	// Process full vectors
	for ; i+lanes <= totalBytes; i += lanes {
		v := hwy.LoadFull[uint8](srcBytes[i:])
		hwy.Store(v, dst[i:])
	}

	// Scalar tail
	for ; i < totalBytes; i++ {
		dst[i] = srcBytes[i]
	}
}

// BaseDecodeFloat64s decodes little-endian bytes to float64 values using SIMD.
// src must have length >= len(dst) * 8.
// Uses unsafe.Slice for zero-cost reinterpretation (ARM64/x86-64 are little-endian).
func BaseDecodeFloat64s(dst []float64, src []byte) {
	if len(dst) == 0 {
		return
	}

	totalBytes := len(dst) * 8
	if len(src) < totalBytes {
		panic("src is too short")
	}

	// Reinterpret float64 slice as bytes (zero-cost on little-endian)
	dstBytes := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), totalBytes)

	// SIMD copy using byte vectors
	lanes := hwy.NumLanes[uint8]()
	i := 0

	// Process full vectors
	for ; i+lanes <= totalBytes; i += lanes {
		v := hwy.LoadFull[uint8](src[i:])
		hwy.Store(v, dstBytes[i:])
	}

	// Scalar tail
	for ; i < totalBytes; i++ {
		dstBytes[i] = src[i]
	}
}
