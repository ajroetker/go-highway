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

// SDPA NEON implementations for ARM64.
// Uses GOAT-transpiled NEON assembly for fused scaled dot-product attention.
package asm

import "unsafe"

// Generate NEON assembly from C source
//go:generate go tool goat ../c/sdpa_neon_arm64.c -O3 --target arm64

// SDPANeonF32 computes scaled dot-product attention using NEON for float32.
func SDPANeonF32(q, k, v, mask, scores, output []float32,
	seqLen, kvLen, headDim int, scale float32) {
	if seqLen <= 0 || kvLen <= 0 || headDim <= 0 {
		return
	}

	var maskPtr unsafe.Pointer
	if mask != nil {
		maskPtr = unsafe.Pointer(&mask[0])
	}

	// Pack dimensions into array (≤8 args for ARM64)
	dims := [3]int64{int64(seqLen), int64(kvLen), int64(headDim)}

	sdpa_neon_f32(
		unsafe.Pointer(&q[0]),
		unsafe.Pointer(&k[0]),
		unsafe.Pointer(&v[0]),
		maskPtr,
		unsafe.Pointer(&scores[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&dims[0]),
		unsafe.Pointer(&scale),
	)
}

// SDPACausalNeonF32 computes causal scaled dot-product attention using NEON for float32.
func SDPACausalNeonF32(q, k, v, scores, output []float32,
	seqLen, kvLen, headDim int, scale float32) {
	if seqLen <= 0 || kvLen <= 0 || headDim <= 0 {
		return
	}

	// Pack dimensions into array (≤8 args for ARM64)
	dims := [3]int64{int64(seqLen), int64(kvLen), int64(headDim)}

	sdpa_causal_neon_f32(
		unsafe.Pointer(&q[0]),
		unsafe.Pointer(&k[0]),
		unsafe.Pointer(&v[0]),
		unsafe.Pointer(&scores[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&dims[0]),
		unsafe.Pointer(&scale),
	)
}

// SDPANeonF64 computes scaled dot-product attention using NEON for float64.
func SDPANeonF64(q, k, v, mask, scores, output []float64,
	seqLen, kvLen, headDim int, scale float64) {
	if seqLen <= 0 || kvLen <= 0 || headDim <= 0 {
		return
	}

	var maskPtr unsafe.Pointer
	if mask != nil {
		maskPtr = unsafe.Pointer(&mask[0])
	}

	// Pack dimensions into array (≤8 args for ARM64)
	dims := [3]int64{int64(seqLen), int64(kvLen), int64(headDim)}

	sdpa_neon_f64(
		unsafe.Pointer(&q[0]),
		unsafe.Pointer(&k[0]),
		unsafe.Pointer(&v[0]),
		maskPtr,
		unsafe.Pointer(&scores[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&dims[0]),
		unsafe.Pointer(&scale),
	)
}

// SDPACausalNeonF64 computes causal scaled dot-product attention using NEON for float64.
func SDPACausalNeonF64(q, k, v, scores, output []float64,
	seqLen, kvLen, headDim int, scale float64) {
	if seqLen <= 0 || kvLen <= 0 || headDim <= 0 {
		return
	}

	// Pack dimensions into array (≤8 args for ARM64)
	dims := [3]int64{int64(seqLen), int64(kvLen), int64(headDim)}

	sdpa_causal_neon_f64(
		unsafe.Pointer(&q[0]),
		unsafe.Pointer(&k[0]),
		unsafe.Pointer(&v[0]),
		unsafe.Pointer(&scores[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&dims[0]),
		unsafe.Pointer(&scale),
	)
}
