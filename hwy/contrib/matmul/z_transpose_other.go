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

//go:build !arm64 && !(amd64 && goexperiment.simd)

package matmul

import (
	"github.com/ajroetker/go-highway/hwy"
)

// transposeScalarOther is a simple scalar transpose for all matrices on fallback platforms
func transposeScalarOther[T any](src []T, m, k int, dst []T) {
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			dst[j*m+i] = src[i*k+j]
		}
	}
}

func init() {
	// Override hwygen-generated fallback with pure scalar
	// The fallback uses transposeBlockSIMD which has lane mismatch issues for Float16/BFloat16
	Transpose2DFloat32 = func(src []float32, m, k int, dst []float32) {
		transposeScalarOther(src, m, k, dst)
	}

	Transpose2DFloat64 = func(src []float64, m, k int, dst []float64) {
		transposeScalarOther(src, m, k, dst)
	}

	Transpose2DFloat16 = func(src []hwy.Float16, m, k int, dst []hwy.Float16) {
		transposeScalarOther(src, m, k, dst)
	}

	Transpose2DBFloat16 = func(src []hwy.BFloat16, m, k int, dst []hwy.BFloat16) {
		transposeScalarOther(src, m, k, dst)
	}
}
