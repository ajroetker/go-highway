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

//go:build !noasm && arm64

// NEON Encode/Decode wrappers for ARM64
// Uses GoAT-generated assembly for maximum performance (~2.5x faster than hwygen).
package asm

import "unsafe"

// EncodeFloat32s encodes a float32 slice to bytes using NEON SIMD.
// dst must have at least len(src)*4 bytes capacity.
func EncodeFloat32s(dst []byte, src []float32) {
	if len(src) == 0 {
		return
	}
	n := int64(len(src))
	encode_f32_neon(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&n),
	)
}

// DecodeFloat32s decodes bytes to a float32 slice using NEON SIMD.
// src must have at least len(dst)*4 bytes.
func DecodeFloat32s(dst []float32, src []byte) {
	if len(dst) == 0 {
		return
	}
	n := int64(len(dst))
	decode_f32_neon(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&n),
	)
}

// EncodeFloat64s encodes a float64 slice to bytes using NEON SIMD.
// dst must have at least len(src)*8 bytes capacity.
func EncodeFloat64s(dst []byte, src []float64) {
	if len(src) == 0 {
		return
	}
	n := int64(len(src))
	encode_f64_neon(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&n),
	)
}

// DecodeFloat64s decodes bytes to a float64 slice using NEON SIMD.
// src must have at least len(dst)*8 bytes.
func DecodeFloat64s(dst []float64, src []byte) {
	if len(dst) == 0 {
		return
	}
	n := int64(len(dst))
	decode_f64_neon(
		unsafe.Pointer(&src[0]),
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&n),
	)
}
