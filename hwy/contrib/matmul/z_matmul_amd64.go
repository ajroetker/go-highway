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

//go:build !noasm && amd64

// NOTE: This file is named "z_matmul_amd64.go" (starting with 'z')
// to ensure its init() runs AFTER the generated dispatch files.
// Go executes init() functions in lexicographic filename order within a package.
//
// Override F16/BF16 dispatch to use GoAT-generated AVX assembly.
// Go's archsimd doesn't support Float16/BFloat16, so we use C→assembly via GoAT.
// F32/F64 continue to use the hwygen-generated code with archsimd.

package matmul

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

func init() {
	level := hwy.CurrentLevel()

	// Float16 dispatch
	if level == hwy.DispatchAVX512 && hwy.HasAVX512FP16() {
		// AVX-512 with native FP16 support (Sapphire Rapids+)
		MatMulFloat16 = asm.MatMulAVX512F16
	} else if level >= hwy.DispatchAVX2 && hwy.HasF16C() {
		// AVX2 with F16C for f16<->f32 conversion
		MatMulFloat16 = asm.MatMulAVX2F16
	}

	// BFloat16 dispatch
	if level == hwy.DispatchAVX512 && hwy.HasAVX512BF16() {
		// AVX-512 with native BF16 support (Cooper Lake+)
		MatMulBFloat16 = asm.MatMulAVX512BF16
	} else if level >= hwy.DispatchAVX2 {
		// AVX2 emulates bf16 via f32
		MatMulBFloat16 = asm.MatMulAVX2BF16
	}
}
