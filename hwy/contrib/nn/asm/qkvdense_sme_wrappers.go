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

// QKV Linear SME implementations for ARM64.
// Uses GOAT-transpiled SME FMOPA assembly for fused QKV projection.
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
//go:generate go tool goat ../c/qkvdense_sme_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv9-a+sme+sme-f64f64" -e="-fno-builtin" -e="-fno-stack-protector"

// QKVDenseFMOPAF32 computes fused QKV projection using SME FMOPA for float32.
//
// xt is [inFeatures, batchSize] (pre-transposed x for FMOPA column access).
// wqkv is [inFeatures, totalOut] (pre-transposed wQKV for FMOPA row access).
// Requires batchSize and totalOut to be multiples of 16.
func QKVDenseFMOPAF32(xt, wqkv, biasq, biask, biasv, q, k, v []float32,
	batchSize, inFeatures, qDim, kvDim int) {
	if batchSize <= 0 || inFeatures <= 0 {
		return
	}
	defer hwy.SMEGuard()()

	var biasqPtr, biaskPtr, biasvPtr unsafe.Pointer
	if biasq != nil {
		biasqPtr = unsafe.Pointer(&biasq[0])
	}
	if biask != nil {
		biaskPtr = unsafe.Pointer(&biask[0])
	}
	if biasv != nil {
		biasvPtr = unsafe.Pointer(&biasv[0])
	}

	// Pack v pointer and dimensions into params array (≤8 args for ARM64)
	params := [5]int64{
		int64(uintptr(unsafe.Pointer(&v[0]))),
		int64(batchSize),
		int64(inFeatures),
		int64(qDim),
		int64(kvDim),
	}

	qkvdense_fmopa_f32(
		unsafe.Pointer(&xt[0]),
		unsafe.Pointer(&wqkv[0]),
		biasqPtr,
		biaskPtr,
		biasvPtr,
		unsafe.Pointer(&q[0]),
		unsafe.Pointer(&k[0]),
		unsafe.Pointer(&params[0]),
	)
}

// QKVDenseFMOPAF64 computes fused QKV projection using SME FMOPA for float64.
//
// xt is [inFeatures, batchSize] (pre-transposed x for FMOPA column access).
// wqkv is [inFeatures, totalOut] (pre-transposed wQKV for FMOPA row access).
// Requires batchSize and totalOut to be multiples of 8.
func QKVDenseFMOPAF64(xt, wqkv, biasq, biask, biasv, q, k, v []float64,
	batchSize, inFeatures, qDim, kvDim int) {
	if batchSize <= 0 || inFeatures <= 0 {
		return
	}
	defer hwy.SMEGuard()()

	var biasqPtr, biaskPtr, biasvPtr unsafe.Pointer
	if biasq != nil {
		biasqPtr = unsafe.Pointer(&biasq[0])
	}
	if biask != nil {
		biaskPtr = unsafe.Pointer(&biask[0])
	}
	if biasv != nil {
		biasvPtr = unsafe.Pointer(&biasv[0])
	}

	// Pack v pointer and dimensions into params array (≤8 args for ARM64)
	params := [5]int64{
		int64(uintptr(unsafe.Pointer(&v[0]))),
		int64(batchSize),
		int64(inFeatures),
		int64(qDim),
		int64(kvDim),
	}

	qkvdense_fmopa_f64(
		unsafe.Pointer(&xt[0]),
		unsafe.Pointer(&wqkv[0]),
		biasqPtr,
		biaskPtr,
		biasvPtr,
		unsafe.Pointer(&q[0]),
		unsafe.Pointer(&k[0]),
		unsafe.Pointer(&params[0]),
	)
}
