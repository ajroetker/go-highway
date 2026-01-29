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

// NOTE: This file is named "z_nn_arm64.go" (starting with 'z')
// to ensure its init() runs AFTER the generated dispatch files.
// Go executes init() functions in lexicographic filename order within a package.
// The generated dispatch sets LayerNorm* etc. to hwygen-generated fallback
// implementations; this file's init() must run afterward to override
// with optimized NEON implementations when available.

package nn

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/nn/asm"
)

// Minimum normSize to use NEON vectorization.
// Below this, the overhead of NEON setup outweighs the benefit.
const minNormSizeForNEON = 8

// layerNormNEONF32 uses GOAT-generated NEON assembly for f32 layer normalization.
func layerNormNEONF32(input, output []float32, normSize int, gamma, beta []float32, epsilon float32) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}

	// Fall back to hwygen-generated code for small normSize
	if normSize < minNormSizeForNEON {
		BaseLayerNorm(input, output, normSize, gamma, beta, epsilon)
		return
	}

	if gamma != nil && beta != nil {
		asm.LayerNormNEONF32(input, output, gamma, beta, size, normSize, epsilon)
	} else {
		asm.LayerNormNEONF32NoAffine(input, output, size, normSize, epsilon)
	}
}

// layerNormNEONF64 uses GOAT-generated NEON assembly for f64 layer normalization.
func layerNormNEONF64(input, output []float64, normSize int, gamma, beta []float64, epsilon float64) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}

	if normSize < minNormSizeForNEON {
		BaseLayerNorm(input, output, normSize, gamma, beta, epsilon)
		return
	}

	if gamma != nil && beta != nil {
		asm.LayerNormNEONF64(input, output, gamma, beta, size, normSize, epsilon)
	} else {
		asm.LayerNormNEONF64NoAffine(input, output, size, normSize, epsilon)
	}
}

func init() {
	if hwy.NoSimdEnv() {
		return
	}

	// Override LayerNorm dispatch with GOAT NEON implementations
	LayerNormFloat32 = layerNormNEONF32
	LayerNormFloat64 = layerNormNEONF64

	// Float16/BFloat16 use the hwygen-generated promoted implementations
	// (promote to f32, compute, demote) which are already efficient enough
	// since the promotion is the bottleneck, not the compute.
}
