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

package matvec

//go:generate go run ../../../cmd/hwygen -input trsv_base.go -dispatch trsv -output . -targets avx2,avx512,neon:asm,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseTrsvLN solves L * x = b for x, overwriting b with the solution.
// L is n×n lower triangular (row-major), b is a vector of length n.
//
// This is the BLAS Level 2 TRSV operation with Uplo=Lower, Trans=NoTrans,
// Diag=NonUnit.
//
// Forward substitution using dot product form (row access, SIMD-friendly):
//
//	x[i] = (b[i] - dot(L[i,0:i], x[0:i])) / L[i,i]
//
// Uses SIMD-accelerated dot products for the inner loop.
func BaseTrsvLN[T hwy.Floats](l []T, b []T, n int) {
	if n == 0 {
		return
	}
	if len(l) < n*n {
		panic("trsv: L slice too short")
	}
	if len(b) < n {
		panic("trsv: b slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()

	for i := range n {
		// b[i] -= dot(L[i,0:i], b[0:i])
		acc := hwy.Zero[T]()
		lRow := l[i*n:]
		var j int
		for j = 0; j+lanes <= i; j += lanes {
			acc = hwy.MulAdd(hwy.Load(lRow[j:]), hwy.Load(b[j:]), acc)
		}
		// Accumulate in float64 for precision (matches production BLAS).
		s := float64(hwy.ReduceSum(acc))
		for ; j < i; j++ {
			s += float64(lRow[j]) * float64(b[j])
		}

		b[i] = T((float64(b[i]) - s) / float64(l[i*n+i]))
	}
}

// BaseTrsvLT solves L^T * x = b for x, overwriting b with the solution.
// L is n×n lower triangular (row-major), b is a vector of length n.
//
// This is the BLAS Level 2 TRSV operation with Uplo=Lower, Trans=Trans,
// Diag=NonUnit.
//
// Backward substitution using AXPY form (row access, SIMD-friendly):
//
//	for i = n-1..0:
//	  x[i] = b[i] / L[i,i]
//	  b[0:i] -= x[i] * L[i,0:i]
//
// The AXPY update accesses row i of L, which is contiguous in row-major.
func BaseTrsvLT[T hwy.Floats](l []T, b []T, n int) {
	if n == 0 {
		return
	}
	if len(l) < n*n {
		panic("trsv: L slice too short")
	}
	if len(b) < n {
		panic("trsv: b slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()

	for i := n - 1; i >= 0; i-- {
		b[i] = T(float64(b[i]) / float64(l[i*n+i]))
		xi := b[i]

		// b[0:i] -= x[i] * L[i,0:i]
		negXi := hwy.Set(-xi)
		lRow := l[i*n:]
		var j int
		for j = 0; j+lanes <= i; j += lanes {
			vb := hwy.Load(b[j:])
			vl := hwy.Load(lRow[j:])
			hwy.Store(hwy.MulAdd(negXi, vl, vb), b[j:])
		}
		for ; j < i; j++ {
			b[j] -= xi * lRow[j]
		}
	}
}
