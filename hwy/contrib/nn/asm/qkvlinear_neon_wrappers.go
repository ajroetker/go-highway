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

// QKV Linear NEON implementations for ARM64.
// Uses GOAT-transpiled NEON assembly for fused matmul + split + bias.
package asm

import "unsafe"

// Generate NEON assembly from C source
//go:generate go tool goat ../c/qkvlinear_neon_arm64.c -O3 --target arm64

// QKVLinearNEONF32 computes fused QKV projection using NEON for float32.
func QKVLinearNEONF32(x, wqkv, biasq, biask, biasv, q, k, v []float32,
	batchSize, inFeatures, qDim, kvDim int) {
	if batchSize <= 0 || inFeatures <= 0 {
		return
	}

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

	qkvlinear_neon_f32(
		unsafe.Pointer(&x[0]),
		unsafe.Pointer(&wqkv[0]),
		biasqPtr,
		biaskPtr,
		biasvPtr,
		unsafe.Pointer(&q[0]),
		unsafe.Pointer(&k[0]),
		unsafe.Pointer(&params[0]),
	)
}

// QKVLinearNEONF64 computes fused QKV projection using NEON for float64.
func QKVLinearNEONF64(x, wqkv, biasq, biask, biasv, q, k, v []float64,
	batchSize, inFeatures, qDim, kvDim int) {
	if batchSize <= 0 || inFeatures <= 0 {
		return
	}

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

	qkvlinear_neon_f64(
		unsafe.Pointer(&x[0]),
		unsafe.Pointer(&wqkv[0]),
		biasqPtr,
		biaskPtr,
		biasvPtr,
		unsafe.Pointer(&q[0]),
		unsafe.Pointer(&k[0]),
		unsafe.Pointer(&params[0]),
	)
}
