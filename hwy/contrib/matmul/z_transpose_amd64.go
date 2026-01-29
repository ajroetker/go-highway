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

//go:build amd64 && goexperiment.simd

package matmul

import (
	"github.com/ajroetker/go-highway/hwy"
)

// Minimum size for SIMD transpose (function call and SIMD overhead dominates below this)
// For small matrices, simple scalar loop is faster due to cache efficiency
const minSizeForSIMDTransposeAMD64 = 32

// transposeScalarAMD64 is a simple scalar transpose for small matrices
func transposeScalarAMD64[T any](src []T, m, k int, dst []T) {
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			dst[j*m+i] = src[i*k+j]
		}
	}
}

func init() {
	// Override hwygen-generated dispatch with size-checked versions
	// For small matrices, use scalar to avoid SIMD overhead and lane mismatch issues

	// When HWY_NO_SIMD=1 is set, the fallback SIMD code doesn't work correctly
	// for Float16/BFloat16 (the interleave-based transpose uses SIMD operations
	// that don't behave correctly in pure scalar mode). Always use pure scalar
	// for these types when SIMD is disabled.
	noSimd := hwy.NoSimdEnv()

	simdF32 := Transpose2DFloat32
	Transpose2DFloat32 = func(src []float32, m, k int, dst []float32) {
		if m >= minSizeForSIMDTransposeAMD64 && k >= minSizeForSIMDTransposeAMD64 {
			simdF32(src, m, k, dst)
		} else {
			transposeScalarAMD64(src, m, k, dst)
		}
	}

	simdF64 := Transpose2DFloat64
	Transpose2DFloat64 = func(src []float64, m, k int, dst []float64) {
		if m >= minSizeForSIMDTransposeAMD64 && k >= minSizeForSIMDTransposeAMD64 {
			simdF64(src, m, k, dst)
		} else {
			transposeScalarAMD64(src, m, k, dst)
		}
	}

	simdF16 := Transpose2DFloat16
	Transpose2DFloat16 = func(src []hwy.Float16, m, k int, dst []hwy.Float16) {
		// Float16 fallback SIMD doesn't work correctly - use pure scalar when SIMD disabled
		if noSimd {
			transposeScalarAMD64(src, m, k, dst)
			return
		}
		if m >= minSizeForSIMDTransposeAMD64 && k >= minSizeForSIMDTransposeAMD64 {
			simdF16(src, m, k, dst)
		} else {
			transposeScalarAMD64(src, m, k, dst)
		}
	}

	simdBF16 := Transpose2DBFloat16
	Transpose2DBFloat16 = func(src []hwy.BFloat16, m, k int, dst []hwy.BFloat16) {
		// BFloat16 fallback SIMD doesn't work correctly - use pure scalar when SIMD disabled
		if noSimd {
			transposeScalarAMD64(src, m, k, dst)
			return
		}
		if m >= minSizeForSIMDTransposeAMD64 && k >= minSizeForSIMDTransposeAMD64 {
			simdBF16(src, m, k, dst)
		} else {
			transposeScalarAMD64(src, m, k, dst)
		}
	}
}
