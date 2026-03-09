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

//go:generate go run ../../../cmd/hwygen -input trsm_base.go -dispatch trsm -output . -targets avx2,avx512,neon:asm,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseTrsmLN solves L * X = B for X, overwriting B with the solution.
// L is n×n lower triangular (row-major), B is n×nrhs (row-major).
//
// This is the BLAS Level 3 TRSM operation with Side=Left, Uplo=Lower,
// Trans=NoTrans, Diag=NonUnit.
//
// Forward substitution, vectorized across nrhs columns:
//
//	for i = 0..n-1:
//	  B[i,:] = (B[i,:] - L[i,0:i] * X[0:i,:]) / L[i,i]
//
// Uses SIMD to process multiple right-hand side columns in parallel.
func BaseTrsmLN[T hwy.Floats](l []T, b []T, n, nrhs int) {
	if n == 0 || nrhs == 0 {
		return
	}
	if len(l) < n*n {
		panic("trsm: L slice too short")
	}
	if len(b) < n*nrhs {
		panic("trsm: B slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()

	for i := range n {
		// B[i,:] -= L[i,j] * B[j,:] for j < i
		for j := range i {
			lij := l[i*n+j]
			negLij := hwy.Set(-lij)

			var k int
			for k = 0; k+lanes <= nrhs; k += lanes {
				vb := hwy.Load(b[i*nrhs+k:])
				vx := hwy.Load(b[j*nrhs+k:])
				hwy.Store(hwy.MulAdd(negLij, vx, vb), b[i*nrhs+k:])
			}
			for ; k < nrhs; k++ {
				b[i*nrhs+k] -= lij * b[j*nrhs+k]
			}
		}

		// B[i,:] /= L[i,i]
		invLii := hwy.Set(T(1) / l[i*n+i])
		var k int
		for k = 0; k+lanes <= nrhs; k += lanes {
			hwy.Store(hwy.Mul(hwy.Load(b[i*nrhs+k:]), invLii), b[i*nrhs+k:])
		}
		s := T(1) / l[i*n+i]
		for ; k < nrhs; k++ {
			b[i*nrhs+k] *= s
		}
	}
}

// BaseTrsmLT solves L^T * X = B for X, overwriting B with the solution.
// L is n×n lower triangular (row-major), B is n×nrhs (row-major).
//
// This is the BLAS Level 3 TRSM operation with Side=Left, Uplo=Lower,
// Trans=Trans, Diag=NonUnit.
//
// Backward substitution, vectorized across nrhs columns:
//
//	for i = n-1..0:
//	  B[i,:] = (B[i,:] - L[i+1:n,i]^T * X[i+1:n,:]) / L[i,i]
//
// Uses SIMD to process multiple right-hand side columns in parallel.
func BaseTrsmLT[T hwy.Floats](l []T, b []T, n, nrhs int) {
	if n == 0 || nrhs == 0 {
		return
	}
	if len(l) < n*n {
		panic("trsm: L slice too short")
	}
	if len(b) < n*nrhs {
		panic("trsm: B slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()

	for i := n - 1; i >= 0; i-- {
		// B[i,:] -= L[j,i] * B[j,:] for j > i
		// Note: L^T[i,j] = L[j,i]
		for j := i + 1; j < n; j++ {
			lji := l[j*n+i]
			negLji := hwy.Set(-lji)

			var k int
			for k = 0; k+lanes <= nrhs; k += lanes {
				vb := hwy.Load(b[i*nrhs+k:])
				vx := hwy.Load(b[j*nrhs+k:])
				hwy.Store(hwy.MulAdd(negLji, vx, vb), b[i*nrhs+k:])
			}
			for ; k < nrhs; k++ {
				b[i*nrhs+k] -= lji * b[j*nrhs+k]
			}
		}

		// B[i,:] /= L[i,i]
		invLii := hwy.Set(T(1) / l[i*n+i])
		var k int
		for k = 0; k+lanes <= nrhs; k += lanes {
			hwy.Store(hwy.Mul(hwy.Load(b[i*nrhs+k:]), invLii), b[i*nrhs+k:])
		}
		s := T(1) / l[i*n+i]
		for ; k < nrhs; k++ {
			b[i*nrhs+k] *= s
		}
	}
}
