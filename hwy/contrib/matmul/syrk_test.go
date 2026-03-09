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

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// syrkReference computes C -= A * A^T (lower triangle) using naive triple loop.
func syrkReference(c []float32, ldc int, a []float32, lda int, n, k int) {
	for i := range n {
		for j := 0; j <= i; j++ {
			var s float32
			for p := range k {
				s += a[i*lda+p] * a[j*lda+p]
			}
			c[i*ldc+j] -= s
		}
	}
}

func TestSyrkLNSmall(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	// A = [[1,2],[3,4]], C = [[10,0],[5,10]]
	// C -= A*A^T = [[1*1+2*2, _],[3*1+4*2, 3*3+4*4]] = [[5, _],[11, 25]]
	// C_lower = [[10-5, 0],[5-11, 10-25]] = [[5, 0],[-6, -15]]
	a := []float32{1, 2, 3, 4}
	c := []float32{10, 0, 5, 10}

	SyrkLN(c, 2, a, 2, 2, 2)

	expected := []float32{5, 0, -6, -15}
	for i := range 2 {
		for j := 0; j <= i; j++ {
			if math.Abs(float64(c[i*2+j]-expected[i*2+j])) > 1e-5 {
				t.Errorf("C[%d,%d] = %f, want %f", i, j, c[i*2+j], expected[i*2+j])
			}
		}
	}
}

func TestSyrkLNRandom(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []struct{ n, k int }{
		{1, 1}, {2, 2}, {4, 4}, {8, 8}, {16, 16},
		{4, 8}, {8, 4}, {16, 32}, {32, 16}, {64, 64},
	}

	for _, sz := range sizes {
		name := fmt.Sprintf("%03d",sz.n) + "x" + fmt.Sprintf("%03d",sz.k)
		t.Run(name, func(t *testing.T) {
			a := make([]float32, sz.n*sz.k)
			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}

			c := make([]float32, sz.n*sz.n)
			cRef := make([]float32, sz.n*sz.n)
			for i := range c {
				c[i] = rand.Float32()*2 - 1
				cRef[i] = c[i]
			}

			syrkReference(cRef, sz.n, a, sz.k, sz.n, sz.k)
			SyrkLN(c, sz.n, a, sz.k, sz.n, sz.k)

			var maxErr float32
			for i := range sz.n {
				for j := 0; j <= i; j++ {
					err := float32(math.Abs(float64(c[i*sz.n+j] - cRef[i*sz.n+j])))
					if err > maxErr {
						maxErr = err
					}
				}
			}

			tol := float32(sz.k) * 1e-5
			if maxErr > tol {
				t.Errorf("max error %e exceeds tolerance %e", maxErr, tol)
			}
		})
	}
}

func TestSyrkLNFloat64(t *testing.T) {
	n, k := 16, 32
	a := make([]float64, n*k)
	for i := range a {
		a[i] = rand.Float64()*2 - 1
	}

	c := make([]float64, n*n)
	cRef := make([]float64, n*n)
	for i := range c {
		c[i] = rand.Float64()*2 - 1
		cRef[i] = c[i]
	}

	// Reference
	for i := range n {
		for j := 0; j <= i; j++ {
			var s float64
			for p := range k {
				s += a[i*k+p] * a[j*k+p]
			}
			cRef[i*n+j] -= s
		}
	}

	SyrkLN(c, n, a, k, n, k)

	var maxErr float64
	for i := range n {
		for j := 0; j <= i; j++ {
			err := math.Abs(c[i*n+j] - cRef[i*n+j])
			if err > maxErr {
				maxErr = err
			}
		}
	}

	if maxErr > 1e-12 {
		t.Errorf("max error %e exceeds tolerance", maxErr)
	}
}

func BenchmarkSyrkLN(b *testing.B) {
	sizes := []struct{ n, k int }{
		{32, 32}, {64, 64}, {128, 128},
	}

	for _, sz := range sizes {
		name := fmt.Sprintf("%03d",sz.n) + "x" + fmt.Sprintf("%03d",sz.k)
		a := make([]float32, sz.n*sz.k)
		for i := range a {
			a[i] = rand.Float32()
		}
		c := make([]float32, sz.n*sz.n)
		buf := make([]float32, len(c))

		b.Run(name, func(b *testing.B) {
			flops := float64(sz.n*(sz.n+1)/2*sz.k*2) / 1e9
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(buf, c)
				SyrkLN(buf, sz.n, a, sz.k, sz.n, sz.k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}
