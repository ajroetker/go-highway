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

// SDOT/TBL dispatch override for Tier 1 vecdot kernels.
//
// The zz_ prefix ensures this init() runs after the hwygen-generated
// z_c_slices_ggufvecdot_neon_arm64.gen.go init(), overriding the float-FMA
// NEON path with integer SDOT kernels.
package gguf

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/gguf/asm"
)

func init() {
	if hwy.NoSimdEnv() {
		return
	}
	VecDotQ4_0Q8_0 = vecDotQ4_0Q8_0_SDOT
	VecDotQ8_0Q8_0 = vecDotQ8_0Q8_0_SDOT
	VecDotIQ4NLQ8_0 = vecDotIQ4NLQ8_0_SDOT
}

func vecDotQ4_0Q8_0_SDOT(wdata, adata []uint8, nblocks int) float32 {
	var result float32
	asm.VecDotQ4_0Q8_0_SDOT(
		unsafe.Pointer(&wdata[0]),
		unsafe.Pointer(&adata[0]),
		int64(nblocks),
		unsafe.Pointer(&result),
	)
	return result
}

func vecDotQ8_0Q8_0_SDOT(wdata, adata []uint8, nblocks int) float32 {
	var result float32
	asm.VecDotQ8_0Q8_0_SDOT(
		unsafe.Pointer(&wdata[0]),
		unsafe.Pointer(&adata[0]),
		int64(nblocks),
		unsafe.Pointer(&result),
	)
	return result
}

func vecDotIQ4NLQ8_0_SDOT(wdata, adata []uint8, nblocks int) float32 {
	var result float32
	asm.VecDotIQ4NLQ8_0_SDOT(
		unsafe.Pointer(&wdata[0]),
		unsafe.Pointer(&adata[0]),
		int64(nblocks),
		unsafe.Pointer(&result),
	)
	return result
}
