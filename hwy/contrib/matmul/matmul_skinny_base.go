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

//go:generate go run ../../../cmd/hwygen -input matmul_skinny_base.go -dispatch matmul_skinny -output . -targets neon:asm,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseSkinnyMatMul computes C = A * B for small M (1-64 rows).
//
// Uses BLIS-style register blocking: Mr=4 rows × Nr=4 vector widths,
// with accumulators held in registers across the full K dimension.
// Unlike BaseBlockedMatMul which tiles I and J in 48×48 blocks
// (re-reading A for each J block), this kernel streams through N
// without I/J blocking. A is small enough to stay in L1 cache.
//
//   - A is M x K (row-major)
//   - B is K x N (row-major)
//   - C is M x N (row-major)
//
// The inner loop processes 4 rows × 4 NEON vectors (16 floats) = 16
// accumulators per micro-tile, fitting comfortably in NEON's 32 registers.
// Each C element is written exactly once (vs K times in the streaming kernel).
func BaseSkinnyMatMul[T hwy.Floats](a, b, c []T, m, n, k int) {
	lanes := hwy.Zero[T]().NumLanes()
	nr := lanes * 4 // 4 vector widths per column strip

	// Zero output
	vZero := hwy.Zero[T]()
	total := m * n
	var idx int
	for idx = 0; idx+lanes <= total; idx += lanes {
		hwy.Store(vZero, c[idx:])
	}
	for ; idx < total; idx++ {
		c[idx] = 0
	}

	// Process 4 rows at a time (Mr=4)
	var i int
	for i = 0; i+4 <= m; i += 4 {
		// Row offsets into A and C
		aOff0 := i * k
		aOff1 := (i + 1) * k
		aOff2 := (i + 2) * k
		aOff3 := (i + 3) * k
		cOff0 := i * n
		cOff1 := (i + 1) * n
		cOff2 := (i + 2) * n
		cOff3 := (i + 3) * n

		// Column loop: Nr = 4 vector widths
		var j int
		for j = 0; j+nr <= n; j += nr {
			// 16 accumulators (4 rows × 4 column vectors)
			acc00 := hwy.Zero[T]()
			acc01 := hwy.Zero[T]()
			acc02 := hwy.Zero[T]()
			acc03 := hwy.Zero[T]()
			acc10 := hwy.Zero[T]()
			acc11 := hwy.Zero[T]()
			acc12 := hwy.Zero[T]()
			acc13 := hwy.Zero[T]()
			acc20 := hwy.Zero[T]()
			acc21 := hwy.Zero[T]()
			acc22 := hwy.Zero[T]()
			acc23 := hwy.Zero[T]()
			acc30 := hwy.Zero[T]()
			acc31 := hwy.Zero[T]()
			acc32 := hwy.Zero[T]()
			acc33 := hwy.Zero[T]()

			// K loop: accumulators stay in registers
			for p := range k {
				// Broadcast A values for 4 rows
				vA0 := hwy.Set(a[aOff0+p])
				vA1 := hwy.Set(a[aOff1+p])
				vA2 := hwy.Set(a[aOff2+p])
				vA3 := hwy.Set(a[aOff3+p])

				// Load 4 B vectors
				bOff := p*n + j
				vB0 := hwy.Load(b[bOff:])
				vB1 := hwy.Load(b[bOff+lanes:])
				vB2 := hwy.Load(b[bOff+2*lanes:])
				vB3 := hwy.Load(b[bOff+3*lanes:])

				// 16 FMA operations
				acc00 = hwy.MulAdd(vA0, vB0, acc00)
				acc01 = hwy.MulAdd(vA0, vB1, acc01)
				acc02 = hwy.MulAdd(vA0, vB2, acc02)
				acc03 = hwy.MulAdd(vA0, vB3, acc03)
				acc10 = hwy.MulAdd(vA1, vB0, acc10)
				acc11 = hwy.MulAdd(vA1, vB1, acc11)
				acc12 = hwy.MulAdd(vA1, vB2, acc12)
				acc13 = hwy.MulAdd(vA1, vB3, acc13)
				acc20 = hwy.MulAdd(vA2, vB0, acc20)
				acc21 = hwy.MulAdd(vA2, vB1, acc21)
				acc22 = hwy.MulAdd(vA2, vB2, acc22)
				acc23 = hwy.MulAdd(vA2, vB3, acc23)
				acc30 = hwy.MulAdd(vA3, vB0, acc30)
				acc31 = hwy.MulAdd(vA3, vB1, acc31)
				acc32 = hwy.MulAdd(vA3, vB2, acc32)
				acc33 = hwy.MulAdd(vA3, vB3, acc33)
			}

			// Store once — the whole point vs streaming kernel
			hwy.Store(acc00, c[cOff0+j:])
			hwy.Store(acc01, c[cOff0+j+lanes:])
			hwy.Store(acc02, c[cOff0+j+2*lanes:])
			hwy.Store(acc03, c[cOff0+j+3*lanes:])
			hwy.Store(acc10, c[cOff1+j:])
			hwy.Store(acc11, c[cOff1+j+lanes:])
			hwy.Store(acc12, c[cOff1+j+2*lanes:])
			hwy.Store(acc13, c[cOff1+j+3*lanes:])
			hwy.Store(acc20, c[cOff2+j:])
			hwy.Store(acc21, c[cOff2+j+lanes:])
			hwy.Store(acc22, c[cOff2+j+2*lanes:])
			hwy.Store(acc23, c[cOff2+j+3*lanes:])
			hwy.Store(acc30, c[cOff3+j:])
			hwy.Store(acc31, c[cOff3+j+lanes:])
			hwy.Store(acc32, c[cOff3+j+2*lanes:])
			hwy.Store(acc33, c[cOff3+j+3*lanes:])
		}

		// Remainder: 2 vector widths (Nr=8)
		if j+2*lanes <= n {
			acc00 := hwy.Zero[T]()
			acc01 := hwy.Zero[T]()
			acc10 := hwy.Zero[T]()
			acc11 := hwy.Zero[T]()
			acc20 := hwy.Zero[T]()
			acc21 := hwy.Zero[T]()
			acc30 := hwy.Zero[T]()
			acc31 := hwy.Zero[T]()

			for p := range k {
				vA0 := hwy.Set(a[aOff0+p])
				vA1 := hwy.Set(a[aOff1+p])
				vA2 := hwy.Set(a[aOff2+p])
				vA3 := hwy.Set(a[aOff3+p])
				bOff := p*n + j
				vB0 := hwy.Load(b[bOff:])
				vB1 := hwy.Load(b[bOff+lanes:])
				acc00 = hwy.MulAdd(vA0, vB0, acc00)
				acc01 = hwy.MulAdd(vA0, vB1, acc01)
				acc10 = hwy.MulAdd(vA1, vB0, acc10)
				acc11 = hwy.MulAdd(vA1, vB1, acc11)
				acc20 = hwy.MulAdd(vA2, vB0, acc20)
				acc21 = hwy.MulAdd(vA2, vB1, acc21)
				acc30 = hwy.MulAdd(vA3, vB0, acc30)
				acc31 = hwy.MulAdd(vA3, vB1, acc31)
			}

			hwy.Store(acc00, c[cOff0+j:])
			hwy.Store(acc01, c[cOff0+j+lanes:])
			hwy.Store(acc10, c[cOff1+j:])
			hwy.Store(acc11, c[cOff1+j+lanes:])
			hwy.Store(acc20, c[cOff2+j:])
			hwy.Store(acc21, c[cOff2+j+lanes:])
			hwy.Store(acc30, c[cOff3+j:])
			hwy.Store(acc31, c[cOff3+j+lanes:])
			j += 2 * lanes
		}

		// Remainder: single vector width
		if j+lanes <= n {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()

			for p := range k {
				vB := hwy.Load(b[p*n+j:])
				acc0 = hwy.MulAdd(hwy.Set(a[aOff0+p]), vB, acc0)
				acc1 = hwy.MulAdd(hwy.Set(a[aOff1+p]), vB, acc1)
				acc2 = hwy.MulAdd(hwy.Set(a[aOff2+p]), vB, acc2)
				acc3 = hwy.MulAdd(hwy.Set(a[aOff3+p]), vB, acc3)
			}

			hwy.Store(acc0, c[cOff0+j:])
			hwy.Store(acc1, c[cOff1+j:])
			hwy.Store(acc2, c[cOff2+j:])
			hwy.Store(acc3, c[cOff3+j:])
			j += lanes
		}

		// Scalar tail
		for ; j < n; j++ {
			var s0, s1, s2, s3 T
			for p := range k {
				bp := b[p*n+j]
				s0 += a[aOff0+p] * bp
				s1 += a[aOff1+p] * bp
				s2 += a[aOff2+p] * bp
				s3 += a[aOff3+p] * bp
			}
			c[cOff0+j] = s0
			c[cOff1+j] = s1
			c[cOff2+j] = s2
			c[cOff3+j] = s3
		}
	}

	// Handle 2 remaining rows
	if i+2 <= m {
		aOff0 := i * k
		aOff1 := (i + 1) * k
		cOff0 := i * n
		cOff1 := (i + 1) * n

		var j int
		for j = 0; j+nr <= n; j += nr {
			acc00 := hwy.Zero[T]()
			acc01 := hwy.Zero[T]()
			acc02 := hwy.Zero[T]()
			acc03 := hwy.Zero[T]()
			acc10 := hwy.Zero[T]()
			acc11 := hwy.Zero[T]()
			acc12 := hwy.Zero[T]()
			acc13 := hwy.Zero[T]()

			for p := range k {
				vA0 := hwy.Set(a[aOff0+p])
				vA1 := hwy.Set(a[aOff1+p])
				bOff := p*n + j
				vB0 := hwy.Load(b[bOff:])
				vB1 := hwy.Load(b[bOff+lanes:])
				vB2 := hwy.Load(b[bOff+2*lanes:])
				vB3 := hwy.Load(b[bOff+3*lanes:])
				acc00 = hwy.MulAdd(vA0, vB0, acc00)
				acc01 = hwy.MulAdd(vA0, vB1, acc01)
				acc02 = hwy.MulAdd(vA0, vB2, acc02)
				acc03 = hwy.MulAdd(vA0, vB3, acc03)
				acc10 = hwy.MulAdd(vA1, vB0, acc10)
				acc11 = hwy.MulAdd(vA1, vB1, acc11)
				acc12 = hwy.MulAdd(vA1, vB2, acc12)
				acc13 = hwy.MulAdd(vA1, vB3, acc13)
			}

			hwy.Store(acc00, c[cOff0+j:])
			hwy.Store(acc01, c[cOff0+j+lanes:])
			hwy.Store(acc02, c[cOff0+j+2*lanes:])
			hwy.Store(acc03, c[cOff0+j+3*lanes:])
			hwy.Store(acc10, c[cOff1+j:])
			hwy.Store(acc11, c[cOff1+j+lanes:])
			hwy.Store(acc12, c[cOff1+j+2*lanes:])
			hwy.Store(acc13, c[cOff1+j+3*lanes:])
		}

		for ; j+lanes <= n; j += lanes {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()

			for p := range k {
				vB := hwy.Load(b[p*n+j:])
				acc0 = hwy.MulAdd(hwy.Set(a[aOff0+p]), vB, acc0)
				acc1 = hwy.MulAdd(hwy.Set(a[aOff1+p]), vB, acc1)
			}

			hwy.Store(acc0, c[cOff0+j:])
			hwy.Store(acc1, c[cOff1+j:])
		}

		for ; j < n; j++ {
			var s0, s1 T
			for p := range k {
				bp := b[p*n+j]
				s0 += a[aOff0+p] * bp
				s1 += a[aOff1+p] * bp
			}
			c[cOff0+j] = s0
			c[cOff1+j] = s1
		}
		i += 2
	}

	// Handle final single row
	if i < m {
		aOff := i * k
		cOff := i * n

		var j int
		for j = 0; j+nr <= n; j += nr {
			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()

			for p := range k {
				vA := hwy.Set(a[aOff+p])
				bOff := p*n + j
				acc0 = hwy.MulAdd(vA, hwy.Load(b[bOff:]), acc0)
				acc1 = hwy.MulAdd(vA, hwy.Load(b[bOff+lanes:]), acc1)
				acc2 = hwy.MulAdd(vA, hwy.Load(b[bOff+2*lanes:]), acc2)
				acc3 = hwy.MulAdd(vA, hwy.Load(b[bOff+3*lanes:]), acc3)
			}

			hwy.Store(acc0, c[cOff+j:])
			hwy.Store(acc1, c[cOff+j+lanes:])
			hwy.Store(acc2, c[cOff+j+2*lanes:])
			hwy.Store(acc3, c[cOff+j+3*lanes:])
		}

		for ; j+lanes <= n; j += lanes {
			acc := hwy.Zero[T]()
			for p := range k {
				acc = hwy.MulAdd(hwy.Set(a[aOff+p]), hwy.Load(b[p*n+j:]), acc)
			}
			hwy.Store(acc, c[cOff+j:])
		}

		for ; j < n; j++ {
			var s T
			for p := range k {
				s += a[aOff+p] * b[p*n+j]
			}
			c[cOff+j] = s
		}
	}
}
