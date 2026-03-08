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
	"math"
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// trsmLNReference solves L*X = B by naive forward substitution.
func trsmLNReference(l, b []float32, n, nrhs int) {
	for i := range n {
		for j := range i {
			for k := range nrhs {
				b[i*nrhs+k] -= l[i*n+j] * b[j*nrhs+k]
			}
		}
		for k := range nrhs {
			b[i*nrhs+k] /= l[i*n+i]
		}
	}
}

// trsmLTReference solves L^T*X = B by naive backward substitution.
func trsmLTReference(l, b []float32, n, nrhs int) {
	for i := n - 1; i >= 0; i-- {
		for j := i + 1; j < n; j++ {
			for k := range nrhs {
				b[i*nrhs+k] -= l[j*n+i] * b[j*nrhs+k]
			}
		}
		for k := range nrhs {
			b[i*nrhs+k] /= l[i*n+i]
		}
	}
}

// makeLowerTriangular creates a random n×n lower triangular matrix
// with positive diagonal (guaranteed non-singular).
func makeLowerTriangular(n int) []float32 {
	l := make([]float32, n*n)
	for i := range n {
		for j := 0; j <= i; j++ {
			l[i*n+j] = rand.Float32()*2 - 1
		}
		// Ensure positive diagonal
		l[i*n+i] = rand.Float32()*2 + 0.5
	}
	return l
}

func TestTrsmLNSmall(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	// L = [[2,0],[3,4]], B = [4,19]
	// L*X = B → X = [2, 13/4]
	l := []float32{2, 0, 3, 4}
	expected := []float32{4, 19}
	trsmLNReference(l, expected, 2, 1)

	got := []float32{4, 19}
	TrsmLN(l, got, 2, 1)

	for i := range got {
		if math.Abs(float64(got[i]-expected[i])) > 1e-5 {
			t.Errorf("x[%d] = %f, want %f", i, got[i], expected[i])
		}
	}
}

func TestTrsmLNRandom(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []struct{ n, nrhs int }{
		{1, 1}, {2, 1}, {4, 1}, {8, 1}, {16, 1},
		{4, 4}, {8, 8}, {16, 4}, {32, 8}, {64, 16},
	}

	for _, sz := range sizes {
		name := sizeStr(sz.n) + "x" + sizeStr(sz.nrhs)
		t.Run(name, func(t *testing.T) {
			l := makeLowerTriangular(sz.n)
			b := make([]float32, sz.n*sz.nrhs)
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			expected := make([]float32, len(b))
			copy(expected, b)
			trsmLNReference(l, expected, sz.n, sz.nrhs)

			got := make([]float32, len(b))
			copy(got, b)
			TrsmLN(l, got, sz.n, sz.nrhs)

			var maxErr float32
			for i := range got {
				err := float32(math.Abs(float64(got[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tol := float32(sz.n) * 1e-5
			if maxErr > tol {
				t.Errorf("max error %e exceeds tolerance %e", maxErr, tol)
			}
		})
	}
}

func TestTrsmLTRandom(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []struct{ n, nrhs int }{
		{1, 1}, {2, 1}, {4, 1}, {8, 1}, {16, 1},
		{4, 4}, {8, 8}, {16, 4}, {32, 8}, {64, 16},
	}

	for _, sz := range sizes {
		name := sizeStr(sz.n) + "x" + sizeStr(sz.nrhs)
		t.Run(name, func(t *testing.T) {
			l := makeLowerTriangular(sz.n)
			b := make([]float32, sz.n*sz.nrhs)
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			expected := make([]float32, len(b))
			copy(expected, b)
			trsmLTReference(l, expected, sz.n, sz.nrhs)

			got := make([]float32, len(b))
			copy(got, b)
			TrsmLT(l, got, sz.n, sz.nrhs)

			var maxErr float32
			for i := range got {
				err := float32(math.Abs(float64(got[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tol := float32(sz.n) * 1e-5
			if maxErr > tol {
				t.Errorf("max error %e exceeds tolerance %e", maxErr, tol)
			}
		})
	}
}

func TestTrsmLNFloat64(t *testing.T) {
	n, nrhs := 16, 4
	l := make([]float64, n*n)
	for i := range n {
		for j := 0; j <= i; j++ {
			l[i*n+j] = rand.Float64()*2 - 1
		}
		l[i*n+i] = rand.Float64()*2 + 0.5
	}
	b := make([]float64, n*nrhs)
	for i := range b {
		b[i] = rand.Float64()*2 - 1
	}

	// Reference
	expected := make([]float64, len(b))
	copy(expected, b)
	for i := range n {
		for j := range i {
			for k := range nrhs {
				expected[i*nrhs+k] -= l[i*n+j] * expected[j*nrhs+k]
			}
		}
		for k := range nrhs {
			expected[i*nrhs+k] /= l[i*n+i]
		}
	}

	got := make([]float64, len(b))
	copy(got, b)
	TrsmLN(l, got, n, nrhs)

	var maxErr float64
	for i := range got {
		err := math.Abs(got[i] - expected[i])
		if err > maxErr {
			maxErr = err
		}
	}

	if maxErr > 1e-12 {
		t.Errorf("max error %e exceeds tolerance", maxErr)
	}
}

func BenchmarkTrsmLN(b *testing.B) {
	sizes := []struct{ n, nrhs int }{
		{32, 1}, {64, 1}, {128, 1},
		{32, 32}, {64, 64},
	}

	for _, sz := range sizes {
		name := sizeStr(sz.n) + "x" + sizeStr(sz.nrhs)
		l := makeLowerTriangular(sz.n)
		bMat := make([]float32, sz.n*sz.nrhs)
		for i := range bMat {
			bMat[i] = rand.Float32()
		}
		buf := make([]float32, len(bMat))

		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(sz.n*sz.n+sz.n*sz.nrhs) * 4)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(buf, bMat)
				TrsmLN(l, buf, sz.n, sz.nrhs)
			}
		})
	}
}
