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

// NOTE: This file is named "z_activation_arm64.go" (starting with 'z')
// to ensure its init() runs AFTER the generated dispatch files.
// Go executes init() functions in lexicographic filename order within a package.
// The generated dispatch sets GELU* etc. to hwygen-generated implementations;
// this file's init() must run afterward to override with optimized NEON
// implementations when available.

package activation

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/activation/asm"
)

// Minimum size to use NEON vectorization.
// Below this, the overhead of NEON setup outweighs the benefit.
const minSizeForNEON = 8

// geluNEONF32 uses GOAT-generated NEON assembly for f32 exact GELU.
func geluNEONF32(input, output []float32) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEON {
		BaseGELU(input, output)
		return
	}
	asm.GELUNeonF32(input, output, size)
}

// geluNEONF64 uses GOAT-generated NEON assembly for f64 exact GELU.
func geluNEONF64(input, output []float64) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEON {
		BaseGELU(input, output)
		return
	}
	asm.GELUNeonF64(input, output, size)
}

// geluApproxNEONF32 uses GOAT-generated NEON assembly for f32 approximate GELU.
func geluApproxNEONF32(input, output []float32) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEON {
		BaseGELUApprox(input, output)
		return
	}
	asm.GELUApproxNeonF32(input, output, size)
}

// geluApproxNEONF64 uses GOAT-generated NEON assembly for f64 approximate GELU.
func geluApproxNEONF64(input, output []float64) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEON {
		BaseGELUApprox(input, output)
		return
	}
	asm.GELUApproxNeonF64(input, output, size)
}

// siluNEONF32 uses GOAT-generated NEON assembly for f32 SiLU.
func siluNEONF32(input, output []float32) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEON {
		BaseSiLU(input, output)
		return
	}
	asm.SiLUNeonF32(input, output, size)
}

// siluNEONF64 uses GOAT-generated NEON assembly for f64 SiLU.
func siluNEONF64(input, output []float64) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEON {
		BaseSiLU(input, output)
		return
	}
	asm.SiLUNeonF64(input, output, size)
}

// tanhNEONF32 uses GOAT-generated NEON assembly for f32 Tanh.
func tanhNEONF32(input, output []float32) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEON {
		BaseTanh(input, output)
		return
	}
	asm.TanhNeonF32(input, output, size)
}

// tanhNEONF64 uses GOAT-generated NEON assembly for f64 Tanh.
func tanhNEONF64(input, output []float64) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEON {
		BaseTanh(input, output)
		return
	}
	asm.TanhNeonF64(input, output, size)
}

// eluNEONF32 uses GOAT-generated NEON assembly for f32 ELU.
func eluNEONF32(input, output []float32, alpha float32) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEON {
		BaseELU(input, output, alpha)
		return
	}
	asm.ELUNeonF32(input, output, size, alpha)
}

// eluNEONF64 uses GOAT-generated NEON assembly for f64 ELU.
func eluNEONF64(input, output []float64, alpha float64) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}
	if size < minSizeForNEON {
		BaseELU(input, output, alpha)
		return
	}
	asm.ELUNeonF64(input, output, size, alpha)
}

func init() {
	if hwy.NoSimdEnv() {
		return
	}

	// Override GELU dispatch with GOAT NEON implementations
	GELUFloat32 = geluNEONF32
	GELUFloat64 = geluNEONF64
	GELUApproxFloat32 = geluApproxNEONF32
	GELUApproxFloat64 = geluApproxNEONF64

	// Override SiLU dispatch with GOAT NEON implementations
	SiLUFloat32 = siluNEONF32
	SiLUFloat64 = siluNEONF64

	// Override Tanh dispatch with GOAT NEON implementations
	TanhFloat32 = tanhNEONF32
	TanhFloat64 = tanhNEONF64

	// Override ELU dispatch with GOAT NEON implementations
	ELUFloat32 = eluNEONF32
	ELUFloat64 = eluNEONF64

	// Float16/BFloat16 use the hwygen-generated promoted implementations
	// (promote to f32, compute, demote) which are already efficient enough
	// since the promotion is the bottleneck, not the compute.
}
