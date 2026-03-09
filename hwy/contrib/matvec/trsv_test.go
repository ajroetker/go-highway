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

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

func makeLowerTriangular(n int) []float32 {
	l := make([]float32, n*n)
	for i := range n {
		for j := 0; j <= i; j++ {
			l[i*n+j] = rand.Float32()*2 - 1
		}
		// Diagonally dominant: ensures cond(L) = O(1) regardless of n.
		l[i*n+i] = float32(i+1) + 0.5
	}
	return l
}

func TestTrsvLNSmall(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	// L = [[2,0],[3,4]], b = [4,19]
	// L*x = b → x[0]=2, x[1]=(19-3*2)/4=13/4=3.25
	l := []float32{2, 0, 3, 4}
	b := []float32{4, 19}

	TrsvLN(l, b, 2)

	want := []float32{2, 3.25}
	for i := range b {
		if math.Abs(float64(b[i]-want[i])) > 1e-5 {
			t.Errorf("x[%d] = %f, want %f", i, b[i], want[i])
		}
	}
}

func TestTrsvLTSmall(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	// L = [[2,0],[3,4]], L^T = [[2,3],[0,4]]
	// L^T * x = b: x[1]=b[1]/4, x[0]=(b[0]-3*x[1])/2
	l := []float32{2, 0, 3, 4}
	b := []float32{8, 12}

	TrsvLT(l, b, 2)

	// x[1] = 12/4 = 3, x[0] = (8-3*3)/2 = -0.5
	want := []float32{-0.5, 3}
	for i := range b {
		if math.Abs(float64(b[i]-want[i])) > 1e-5 {
			t.Errorf("x[%d] = %f, want %f", i, b[i], want[i])
		}
	}
}

func TestTrsvLNRandom(t *testing.T) {
	sizes := []int{1, 2, 4, 8, 16, 32, 64, 128}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("%03d", n), func(t *testing.T) {
			l := makeLowerTriangular(n)

			// Generate random x, compute b = L*x
			xTrue := make([]float32, n)
			for i := range xTrue {
				xTrue[i] = rand.Float32()*2 - 1
			}

			b := make([]float32, n)
			for i := range n {
				var s float32
				for j := 0; j <= i; j++ {
					s += l[i*n+j] * xTrue[j]
				}
				b[i] = s
			}

			TrsvLN(l, b, n)

			var maxErr float32
			for i := range n {
				err := float32(math.Abs(float64(b[i] - xTrue[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tol := float32(n) * 1e-5
			if maxErr > tol {
				t.Errorf("max error %e exceeds tolerance %e", maxErr, tol)
			}
		})
	}
}

func TestTrsvLTRandom(t *testing.T) {
	sizes := []int{1, 2, 4, 8, 16, 32, 64, 128}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("%03d", n), func(t *testing.T) {
			l := makeLowerTriangular(n)

			// Generate random x, compute b = L^T * x
			xTrue := make([]float32, n)
			for i := range xTrue {
				xTrue[i] = rand.Float32()*2 - 1
			}

			b := make([]float32, n)
			for i := range n {
				var s float32
				for j := i; j < n; j++ {
					// L^T[i,j] = L[j,i]
					s += l[j*n+i] * xTrue[j]
				}
				b[i] = s
			}

			TrsvLT(l, b, n)

			var maxErr float32
			for i := range n {
				err := float32(math.Abs(float64(b[i] - xTrue[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tol := float32(n) * 1e-5
			if maxErr > tol {
				t.Errorf("max error %e exceeds tolerance %e", maxErr, tol)
			}
		})
	}
}

