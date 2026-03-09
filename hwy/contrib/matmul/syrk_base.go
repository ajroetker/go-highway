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

package matmul

//go:generate go run ../../../cmd/hwygen -input syrk_base.go -dispatch syrk -output . -targets avx2,avx512,neon:asm,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseSyrkLN performs the symmetric rank-k update:
//
//	C -= A * A^T    (lower triangle only)
//
// where C is n×n symmetric (row-major, stride ldc) and A is n×k (row-major, stride lda).
// Only the lower triangular part of C (including diagonal) is updated.
//
// This is the BLAS Level 3 SYRK operation with Uplo=Lower, Trans=NoTrans,
// alpha=-1, beta=1.
//
// For each (i,j) with i >= j:
//
//	C[i,j] -= dot(A[i,0:k], A[j,0:k])
//
// Uses SIMD-accelerated dot products for the inner k-loop.
func BaseSyrkLN[T hwy.Floats](c []T, ldc int, a []T, lda int, n, k int) {
	if n == 0 || k == 0 {
		return
	}

	lanes := hwy.Zero[T]().NumLanes()

	for i := range n {
		aRowI := a[i*lda:]

		for j := 0; j <= i; j++ {
			aRowJ := a[j*lda:]

			// Dot product of row i and row j of A
			acc := hwy.Zero[T]()
			var p int
			for p = 0; p+lanes <= k; p += lanes {
				acc = hwy.MulAdd(hwy.Load(aRowI[p:]), hwy.Load(aRowJ[p:]), acc)
			}
			dot := hwy.ReduceSum(acc)
			for ; p < k; p++ {
				dot += aRowI[p] * aRowJ[p]
			}

			c[i*ldc+j] -= dot
		}
	}
}
