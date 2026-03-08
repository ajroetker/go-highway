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

// Package linalg provides linear algebra decompositions and solvers
// built on SIMD-accelerated primitives from the vec and matmul packages.
package linalg

import (
	"errors"
	"math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
)

var (
	// ErrNotPositiveDefinite is returned when Cholesky factorization
	// encounters a non-positive diagonal element.
	ErrNotPositiveDefinite = errors.New("linalg: matrix is not positive definite")
)

// sqrt returns the square root of a generic float value.
func sqrt[T hwy.Floats](v T) T {
	switch any(v).(type) {
	case float32:
		return any(float32(math.Sqrt(float64(any(v).(float32))))).(T)
	case float64:
		return any(math.Sqrt(any(v).(float64))).(T)
	default:
		return any(float32(math.Sqrt(float64(any(v).(float32))))).(T)
	}
}

// Cholesky computes the lower Cholesky factorization A = L * L^T in-place.
//
// On entry, A is an n×n symmetric positive definite matrix (row-major).
// Only the lower triangular part is read. On return, the lower triangular
// part of A (including diagonal) contains L. The strict upper triangle
// is not modified.
//
// Inner loops use SIMD-accelerated dot products (vec.Dot).
// Complexity: n³/3 flops.
//
// Returns ErrNotPositiveDefinite if A is not symmetric positive definite.
func Cholesky[T hwy.Floats](a []T, n int) error {
	if len(a) < n*n {
		panic("linalg: A slice too short")
	}

	for j := range n {
		// Diagonal: L[j,j] = sqrt(A[j,j] - ||L[j,0:j]||²)
		diag := a[j*n+j]
		if j > 0 {
			diag -= vec.Dot(a[j*n:j*n+j], a[j*n:j*n+j])
		}
		if diag <= 0 {
			return ErrNotPositiveDefinite
		}
		ljj := sqrt(diag)
		a[j*n+j] = ljj
		invLjj := T(1) / ljj

		// Column j below diagonal:
		// L[i,j] = (A[i,j] - dot(L[i,0:j], L[j,0:j])) / L[j,j]
		for i := j + 1; i < n; i++ {
			val := a[i*n+j]
			if j > 0 {
				val -= vec.Dot(a[i*n:i*n+j], a[j*n:j*n+j])
			}
			a[i*n+j] = val * invLjj
		}
	}

	return nil
}

// CholeskySolve solves A*X = B where A is symmetric positive definite.
//
//   - a: n×n row-major matrix. If factored=false, Cholesky factorization is
//     performed first and the lower triangle is overwritten with L.
//     If factored=true, a must already contain L from a prior Cholesky call.
//   - b: n×nrhs row-major matrix, overwritten with the solution X.
//
// The solve proceeds in two steps using SIMD-accelerated triangular solvers:
//  1. Forward substitution:  L * Y = B  (matmul.TrsmLN)
//  2. Backward substitution: L^T * X = Y (matmul.TrsmLT)
//
// Returns ErrNotPositiveDefinite if A is not positive definite (when factored=false).
func CholeskySolve[T hwy.Floats](a []T, n int, b []T, nrhs int, factored bool) error {
	if !factored {
		if err := Cholesky(a, n); err != nil {
			return err
		}
	}

	matmul.TrsmLN(a, b, n, nrhs)
	matmul.TrsmLT(a, b, n, nrhs)

	return nil
}
