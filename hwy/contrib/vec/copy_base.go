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

package vec

//go:generate go run ../../../cmd/hwygen -input copy_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch copy

import "github.com/ajroetker/go-highway/hwy"

// BaseCopy copies src into dst: dst[i] = src[i].
//
// If the slices have different lengths, the operation uses the minimum length.
// Returns early if either slice is empty.
//
// This is the BLAS Level 1 COPY operation. Uses SIMD acceleration for
// bulk memory transfer.
func BaseCopy[T hwy.Floats](dst, src []T) {
	if len(dst) == 0 || len(src) == 0 {
		return
	}

	n := min(len(dst), len(src))
	lanes := hwy.Zero[T]().NumLanes()

	var i int
	for i = 0; i+lanes <= n; i += lanes {
		hwy.Store(hwy.Load(src[i:]), dst[i:])
	}

	for ; i < n; i++ {
		dst[i] = src[i]
	}
}
