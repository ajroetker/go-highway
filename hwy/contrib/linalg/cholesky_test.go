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

package linalg

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// makeSymPosDef creates a random n×n symmetric positive definite matrix
// by computing A = B^T * B + n*I (the identity scaling ensures positive definiteness).
func makeSymPosDef(n int) []float32 {
	b := make([]float32, n*n)
	for i := range b {
		b[i] = rand.Float32()*2 - 1
	}

	a := make([]float32, n*n)
	// A = B^T * B
	for i := range n {
		for j := range n {
			var s float32
			for k := range n {
				s += b[k*n+i] * b[k*n+j]
			}
			a[i*n+j] = s
		}
	}
	// A += n*I to ensure well-conditioned
	for i := range n {
		a[i*n+i] += float32(n)
	}
	return a
}

func makeSymPosDef64(n int) []float64 {
	b := make([]float64, n*n)
	for i := range b {
		b[i] = rand.Float64()*2 - 1
	}

	a := make([]float64, n*n)
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += b[k*n+i] * b[k*n+j]
			}
			a[i*n+j] = s
		}
	}
	for i := range n {
		a[i*n+i] += float64(n)
	}
	return a
}

func copySlice[T any](src []T) []T {
	dst := make([]T, len(src))
	copy(dst, src)
	return dst
}

// verifyCholesky checks that L * L^T = A (original).
func verifyCholesky(t *testing.T, orig, factored []float32, n int, tol float32) {
	t.Helper()

	var maxErr float32
	for i := range n {
		for j := range n {
			// Reconstruct: (L * L^T)[i,j] = sum(L[i,k] * L[j,k]) for k <= min(i,j)
			var s float32
			upper := min(i, j) + 1
			for k := range upper {
				s += factored[i*n+k] * factored[j*n+k]
			}
			err := float32(math.Abs(float64(s - orig[i*n+j])))
			if err > maxErr {
				maxErr = err
			}
		}
	}

	if maxErr > tol {
		t.Errorf("L*L^T reconstruction error %e exceeds tolerance %e", maxErr, tol)
	}
}

func TestCholeskySmall(t *testing.T) {
	// 3×3 SPD matrix: A = [[4,2,1],[2,5,3],[1,3,9]]
	a := []float32{
		4, 2, 1,
		2, 5, 3,
		1, 3, 9,
	}
	orig := copySlice(a)

	if err := Cholesky(a, 3); err != nil {
		t.Fatalf("Cholesky failed: %v", err)
	}

	// Verify L is lower triangular with positive diagonal
	for i := range 3 {
		if a[i*3+i] <= 0 {
			t.Errorf("L[%d,%d] = %f, want positive", i, i, a[i*3+i])
		}
	}

	verifyCholesky(t, orig, a, 3, 1e-5)
}

func TestCholeskyIdentity(t *testing.T) {
	n := 4
	a := make([]float32, n*n)
	for i := range n {
		a[i*n+i] = 1
	}
	orig := copySlice(a)

	if err := Cholesky(a, n); err != nil {
		t.Fatalf("Cholesky failed: %v", err)
	}

	// L should be identity
	for i := range n {
		for j := range n {
			want := float32(0)
			if i == j {
				want = 1
			}
			if math.Abs(float64(a[i*n+j]-want)) > 1e-7 {
				t.Errorf("L[%d,%d] = %f, want %f", i, j, a[i*n+j], want)
			}
		}
	}

	verifyCholesky(t, orig, a, n, 1e-6)
}

func TestCholeskyRandom(t *testing.T) {
	sizes := []int{1, 2, 4, 8, 16, 32, 64, 128}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("%03d", n), func(t *testing.T) {
			a := makeSymPosDef(n)
			orig := copySlice(a)

			if err := Cholesky(a, n); err != nil {
				t.Fatalf("Cholesky failed: %v", err)
			}

			// Tolerance scales with n due to accumulated floating-point error
			tol := float32(n) * 1e-4
			verifyCholesky(t, orig, a, n, tol)
		})
	}
}

func TestCholesky64Random(t *testing.T) {
	sizes := []int{1, 2, 4, 8, 16, 32, 64, 128}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("%03d", n), func(t *testing.T) {
			a := makeSymPosDef64(n)
			orig := copySlice(a)

			if err := Cholesky(a, n); err != nil {
				t.Fatalf("Cholesky64 failed: %v", err)
			}

			var maxErr float64
			for i := range n {
				for j := range n {
					var s float64
					upper := min(i, j) + 1
					for k := range upper {
						s += a[i*n+k] * a[j*n+k]
					}
					err := math.Abs(s - orig[i*n+j])
					if err > maxErr {
						maxErr = err
					}
				}
			}

			tol := float64(n) * 1e-12
			if maxErr > tol {
				t.Errorf("reconstruction error %e exceeds tolerance %e", maxErr, tol)
			}
		})
	}
}

func TestCholeskyNotPositiveDefinite(t *testing.T) {
	// Not SPD: has negative eigenvalue
	a := []float32{
		1, 2,
		2, 1,
	}

	err := Cholesky(a, 2)
	if err != ErrNotPositiveDefinite {
		t.Errorf("got error %v, want ErrNotPositiveDefinite", err)
	}
}

func TestCholeskySolveSmall(t *testing.T) {
	// A = [[4,2],[2,5]], b = [8,9]
	// Solution: x = [1, 1]  (4+2=6? no: 4*1+2*1=6≠8)
	// Let's compute: A*[1,1] = [6,7], so b=[6,7] → x=[1,1]
	a := []float32{4, 2, 2, 5}
	b := []float32{6, 7}

	if err := CholeskySolve(a, 2, b, 1, false); err != nil {
		t.Fatalf("CholeskySolve failed: %v", err)
	}

	for i, want := range []float32{1, 1} {
		if math.Abs(float64(b[i]-want)) > 1e-5 {
			t.Errorf("x[%d] = %f, want %f", i, b[i], want)
		}
	}
}

func TestCholeskySolveRandom(t *testing.T) {
	sizes := []int{2, 4, 8, 16, 32, 64}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("%03d", n), func(t *testing.T) {
			a := makeSymPosDef(n)
			origA := copySlice(a)

			// Generate random solution, compute b = A*x
			xTrue := make([]float32, n)
			for i := range xTrue {
				xTrue[i] = rand.Float32()*2 - 1
			}

			b := make([]float32, n)
			for i := range n {
				var s float32
				for j := range n {
					s += origA[i*n+j] * xTrue[j]
				}
				b[i] = s
			}

			if err := CholeskySolve(a, n, b, 1, false); err != nil {
				t.Fatalf("CholeskySolve failed: %v", err)
			}

			// Check solution
			var maxErr float32
			for i := range n {
				err := float32(math.Abs(float64(b[i] - xTrue[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tol := float32(n) * 1e-3
			if maxErr > tol {
				t.Errorf("solution error %e exceeds tolerance %e", maxErr, tol)
			}
		})
	}
}

func TestCholeskySolveMultipleRHS(t *testing.T) {
	n := 8
	nrhs := 4

	a := makeSymPosDef(n)
	origA := copySlice(a)

	// Generate random solutions, compute B = A * X
	xTrue := make([]float32, n*nrhs)
	for i := range xTrue {
		xTrue[i] = rand.Float32()*2 - 1
	}

	b := make([]float32, n*nrhs)
	for i := range n {
		for k := range nrhs {
			var s float32
			for j := range n {
				s += origA[i*n+j] * xTrue[j*nrhs+k]
			}
			b[i*nrhs+k] = s
		}
	}

	if err := CholeskySolve(a, n, b, nrhs, false); err != nil {
		t.Fatalf("CholeskySolve failed: %v", err)
	}

	var maxErr float32
	for i := range n * nrhs {
		err := float32(math.Abs(float64(b[i] - xTrue[i])))
		if err > maxErr {
			maxErr = err
		}
	}

	tol := float32(n) * 1e-3
	if maxErr > tol {
		t.Errorf("solution error %e exceeds tolerance %e", maxErr, tol)
	}
}

func BenchmarkCholesky(b *testing.B) {
	sizes := []int{32, 64, 128, 256}

	for _, n := range sizes {
		a := makeSymPosDef(n)

		b.Run(fmt.Sprintf("%03d", n), func(b *testing.B) {
			b.SetBytes(int64(n * n * 4))
			buf := make([]float32, n*n)
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				copy(buf, a)
				if err := Cholesky(buf, n); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

