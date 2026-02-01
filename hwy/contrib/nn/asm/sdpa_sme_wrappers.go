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

// SDPA SME implementations for ARM64.
// Uses GOAT-transpiled SME FMOPA assembly for Flash Attention with online softmax.
package asm

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// Generate SME assembly from C source.
//
// -fno-builtin prevents clang from optimizing zeroing loops into memset calls,
// and -fno-stack-protector removes stack canary checks. Without these flags,
// the generated SME assembly contains calls to external functions (_memset_pattern16,
// ___arm_sc_memset, ___stack_chk_fail), which forces clang to emit a dynamic
// SVL^2-byte ZA save area (via rdsvl+msub+mov sp) for the TPIDR2_EL0 lazy save
// mechanism. This dynamic stack adjustment is incompatible with Go's fixed-frame
// stack model and causes crashes at runtime.
//go:generate go tool goat ../c/sdpa_sme_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv9-a+sme+sme-f64f64" -e="-fno-builtin" -e="-fno-stack-protector"

// SDPAFMOPAF32 computes scaled dot-product attention using SME Flash Attention for float32.
//
// Uses tiled Flash Attention with online softmax via FMOPA. Avoids materializing
// the full [seqLen, kvLen] scores matrix.
//
// kt is [headDim, kvLen] (pre-transposed K for FMOPA column access).
// mask is [seqLen, kvLen] or nil.
func SDPAFMOPAF32(q []float32, kt []float32, v, mask, output []float32,
	seqLen, kvLen, headDim int, scale float32) {
	if seqLen <= 0 || kvLen <= 0 || headDim <= 0 {
		return
	}
	defer hwy.SMEGuard()()

	var maskPtr unsafe.Pointer
	if mask != nil {
		maskPtr = unsafe.Pointer(&mask[0])
	}

	// Pack dimensions into array (≤8 args for ARM64)
	dims := [3]int64{int64(seqLen), int64(kvLen), int64(headDim)}

	sdpa_fmopa_f32(
		unsafe.Pointer(&q[0]),
		unsafe.Pointer(&kt[0]),
		unsafe.Pointer(&v[0]),
		maskPtr,
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&dims[0]),
		unsafe.Pointer(&scale),
	)
}

// SDPAFMOPAF64 computes scaled dot-product attention using SME Flash Attention for float64.
//
// kt is [headDim, kvLen] (pre-transposed K for FMOPA column access).
func SDPAFMOPAF64(q []float64, kt []float64, v, mask, output []float64,
	seqLen, kvLen, headDim int, scale float64) {
	if seqLen <= 0 || kvLen <= 0 || headDim <= 0 {
		return
	}
	defer hwy.SMEGuard()()

	var maskPtr unsafe.Pointer
	if mask != nil {
		maskPtr = unsafe.Pointer(&mask[0])
	}

	// Pack dimensions into array (≤8 args for ARM64)
	dims := [3]int64{int64(seqLen), int64(kvLen), int64(headDim)}

	sdpa_fmopa_f64(
		unsafe.Pointer(&q[0]),
		unsafe.Pointer(&kt[0]),
		unsafe.Pointer(&v[0]),
		maskPtr,
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&dims[0]),
		unsafe.Pointer(&scale),
	)
}
