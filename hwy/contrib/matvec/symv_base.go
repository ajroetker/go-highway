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

//go:generate go run ../../../cmd/hwygen -input symv_base.go -dispatch symv -output . -targets avx2,avx512,neon:asm,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseSymvLN computes y = A * x where A is symmetric.
//
// Only the lower triangular part of A (n×n, row-major) is read.
// x and y are vectors of length n. y is overwritten (not accumulated).
//
// This is the BLAS Level 2 SYMV operation with Uplo=Lower.
//
// The algorithm processes each row i in two passes:
//  1. y[i] += dot(A[i,0:i], x[0:i]) — below-diagonal (row access, contiguous)
//  2. y[j] += A[i,j] * x[i] for j < i — symmetric scatter (AXPY on y)
//
// Both passes use SIMD on contiguous row data.
func BaseSymvLN[T hwy.Floats](a []T, x, y []T, n int) {
	if n == 0 {
		return
	}
	if len(a) < n*n {
		panic("symv: A slice too short")
	}
	if len(x) < n || len(y) < n {
		panic("symv: vector slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()

	// Zero y
	vZero := hwy.Zero[T]()
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		hwy.Store(vZero, y[i:])
	}
	for ; i < n; i++ {
		y[i] = 0
	}

	for i := range n {
		aRow := a[i*n:]

		// Diagonal contribution
		y[i] += aRow[i] * x[i]

		// Below-diagonal: dot product for y[i], AXPY for y[0:i]
		xi := x[i]
		vxi := hwy.Set(xi)
		acc := hwy.Zero[T]()

		var j int
		for j = 0; j+lanes <= i; j += lanes {
			va := hwy.Load(aRow[j:])
			vx := hwy.Load(x[j:])

			// y[i] += A[i,j] * x[j] (accumulate dot product)
			acc = hwy.MulAdd(va, vx, acc)

			// y[j] += A[i,j] * x[i] (symmetric scatter)
			vy := hwy.Load(y[j:])
			hwy.Store(hwy.MulAdd(vxi, va, vy), y[j:])
		}

		y[i] += hwy.ReduceSum(acc)
		for ; j < i; j++ {
			y[i] += aRow[j] * x[j]
			y[j] += aRow[j] * xi
		}
	}
}
