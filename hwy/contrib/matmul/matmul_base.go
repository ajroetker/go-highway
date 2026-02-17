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

//go:generate go run ../../../cmd/hwygen -input matmul_base.go -dispatch matmul -output . -targets avx2,avx512,neon:asm,fallback

import "github.com/ajroetker/go-highway/hwy"

// matmulScalar is the pure Go scalar implementation.
// C[i,j] = sum(A[i,p] * B[p,j]) for p in 0..K-1
// This is kept for reference and benchmarking; the generated BaseMatMul_fallback
// is used as the actual fallback implementation.
func matmulScalar(a, b, c []float32, m, n, k int) {
	// Clear output
	for i := range c[:m*n] {
		c[i] = 0
	}

	// Standard triple-loop matrix multiply
	for i := range m {
		for p := range k {
			aip := a[i*k+p]
			for j := range n {
				c[i*n+j] += aip * b[p*n+j]
			}
		}
	}
}

// matmulScalar64 is the pure Go scalar implementation for float64.
func matmulScalar64(a, b, c []float64, m, n, k int) {
	// Clear output
	for i := range c[:m*n] {
		c[i] = 0
	}

	// Standard triple-loop matrix multiply
	for i := range m {
		for p := range k {
			aip := a[i*k+p]
			for j := range n {
				c[i*n+j] += aip * b[p*n+j]
			}
		}
	}
}

// BaseMatMul computes C = A * B where:
//   - A is M x K (row-major)
//   - B is K x N (row-major)
//   - C is M x N (row-major)
//
// Uses register-blocked accumulators: the J dimension is tiled into groups
// of 4 vector widths, with accumulators held in registers across the full
// K loop. This eliminates K-1 redundant loads and stores of C per element.
//
// This function is designed for code generation by hwygen.
// It will be specialized for AVX2, AVX-512, NEON, and fallback targets.
func BaseMatMul[T hwy.Floats](a, b, c []T, m, n, k int) {
	if len(a) < m*k {
		panic("matmul: A slice too short")
	}
	if len(b) < k*n {
		panic("matmul: B slice too short")
	}
	if len(c) < m*n {
		panic("matmul: C slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()
	tileJ := 4 * lanes

	// For each row i of C
	for i := range m {
		cRow := c[i*n : (i+1)*n]

		// Tiled J loop — 4 accumulators held in registers across full K loop
		var j int
		for j = 0; j+tileJ <= n; j += tileJ {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()
			for p := range k {
				vA := hwy.Set(a[i*k+p])
				bRow := b[p*n:]
				acc0 = hwy.MulAdd(vA, hwy.Load(bRow[j:]), acc0)
				acc1 = hwy.MulAdd(vA, hwy.Load(bRow[j+lanes:]), acc1)
				acc2 = hwy.MulAdd(vA, hwy.Load(bRow[j+2*lanes:]), acc2)
				acc3 = hwy.MulAdd(vA, hwy.Load(bRow[j+3*lanes:]), acc3)
			}
			hwy.Store(acc0, cRow[j:])
			hwy.Store(acc1, cRow[j+lanes:])
			hwy.Store(acc2, cRow[j+2*lanes:])
			hwy.Store(acc3, cRow[j+3*lanes:])
		}

		// Remainder: single accumulator per remaining vector strip
		for ; j+lanes <= n; j += lanes {
			acc := hwy.Zero[T]()
			for p := range k {
				vA := hwy.Set(a[i*k+p])
				acc = hwy.MulAdd(vA, hwy.Load(b[p*n+j:]), acc)
			}
			hwy.Store(acc, cRow[j:])
		}

		// Scalar tail
		for ; j < n; j++ {
			var sum T
			for p := range k {
				sum += a[i*k+p] * b[p*n+j]
			}
			cRow[j] = sum
		}
	}
}
