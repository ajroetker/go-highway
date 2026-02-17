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

// Packed GEBP Micro-Kernel wrappers for ARM64 NEON
package asm

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// BF16: Requires ARMv8.6-A with BF16 extension
//go:generate go tool goat ../c/packed_kernel_neon_bf16_arm64.c -O3 --target arm64 -e="-march=armv8.6-a+bf16"

// PackedMicroKernelNEONBF16 computes C[mr×nr] += PackedA[mr×kc] * PackedB[kc×nr]
// using NEON with f32 accumulation for bfloat16.
//
// Requires ARMv8.6-A with BF16 extension.
func PackedMicroKernelNEONBF16(packedA, packedB, c []hwy.BFloat16, kc, n, mr, nr int) {
	if kc == 0 || mr == 0 || nr == 0 {
		return
	}
	// Bounds check: need (mr-1)*n + nr elements for C (last row doesn't need full stride)
	if len(packedA) < mr*kc || len(packedB) < kc*nr || len(c) < (mr-1)*n+nr {
		return
	}
	kcVal := int64(kc)
	nVal := int64(n)
	mrVal := int64(mr)
	nrVal := int64(nr)
	packed_microkernel_neon_bf16(
		unsafe.Pointer(&packedA[0]),
		unsafe.Pointer(&packedB[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&kcVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&mrVal),
		unsafe.Pointer(&nrVal),
	)
}
