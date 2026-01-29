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

//go:generate go run ../../../cmd/hwygen -input matmul_blocked.go -dispatch matmul_blocked -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

// Block size tuned for L1 cache (32KB typical).
// 3 blocks of 48x48 float32 = 3 * 48 * 48 * 4 = 27KB < 32KB L1.
// Must be a multiple of 16 for AVX-512 alignment.
const (
	BlockSize = 48
)

// BaseBlockedMatMul computes C = A * B using cache-tiled blocking with register accumulation.
//
//   - A is M x K (row-major)
//   - B is K x N (row-major)
//   - C is M x N (row-major)
//
// This implementation uses register blocking: accumulators are held in registers
// across the entire K dimension to minimize memory traffic. Each micro-tile
// processes 4 rows × 2 vector widths of output.
func BaseBlockedMatMul[T hwy.Floats](a, b, c []T, m, n, k int) {
	if len(a) < m*k {
		panic("matmul: A slice too short")
	}
	if len(b) < k*n {
		panic("matmul: B slice too short")
	}
	if len(c) < m*n {
		panic("matmul: C slice too short")
	}

	// Zero output first using SIMD
	vZero := hwy.Zero[T]()
	lanes := vZero.NumLanes()
	total := m * n
	var idx int
	for idx = 0; idx+lanes <= total; idx += lanes {
		hwy.Store(vZero, c[idx:])
	}
	for ; idx < total; idx++ {
		c[idx] = 0
	}

	// Micro-tile dimensions
	mr := 4         // Rows per micro-tile
	nr := lanes * 2 // Columns per micro-tile (2 vector widths)

	// Block over output dimensions (i, j) for cache locality.
	// Process full K dimension per (i,j) block to maximize register reuse.
	for i0 := 0; i0 < m; i0 += BlockSize {
		iEnd := min(i0+BlockSize, m)

		for j0 := 0; j0 < n; j0 += BlockSize {
			jEnd := min(j0+BlockSize, n)

			// Process micro-tiles within this block
			var i int
			for i = i0; i+mr <= iEnd; i += mr {
				// Process columns in groups of Nr (2 vector widths)
				var j int
				for j = j0; j+nr <= jEnd; j += nr {
					// Initialize 8 accumulators (4 rows × 2 column strips)
					acc00 := hwy.Zero[T]()
					acc01 := hwy.Zero[T]()
					acc10 := hwy.Zero[T]()
					acc11 := hwy.Zero[T]()
					acc20 := hwy.Zero[T]()
					acc21 := hwy.Zero[T]()
					acc30 := hwy.Zero[T]()
					acc31 := hwy.Zero[T]()

					// K-loop: accumulate in registers (full K dimension)
					for p := 0; p < k; p++ {
						// Load A values for 4 rows
						a0p := a[i*k+p]
						a1p := a[(i+1)*k+p]
						a2p := a[(i+2)*k+p]
						a3p := a[(i+3)*k+p]

						vA0 := hwy.Set(a0p)
						vA1 := hwy.Set(a1p)
						vA2 := hwy.Set(a2p)
						vA3 := hwy.Set(a3p)

						// Load B values (2 vector widths)
						bRowStart := p * n
						vB0 := hwy.Load(b[bRowStart+j:])
						vB1 := hwy.Load(b[bRowStart+j+lanes:])

						// Accumulate: 8 FMA operations
						acc00 = hwy.MulAdd(vA0, vB0, acc00)
						acc01 = hwy.MulAdd(vA0, vB1, acc01)
						acc10 = hwy.MulAdd(vA1, vB0, acc10)
						acc11 = hwy.MulAdd(vA1, vB1, acc11)
						acc20 = hwy.MulAdd(vA2, vB0, acc20)
						acc21 = hwy.MulAdd(vA2, vB1, acc21)
						acc30 = hwy.MulAdd(vA3, vB0, acc30)
						acc31 = hwy.MulAdd(vA3, vB1, acc31)
					}

					// Write back accumulators to C
					cRow0 := i * n
					cRow1 := (i + 1) * n
					cRow2 := (i + 2) * n
					cRow3 := (i + 3) * n

					hwy.Store(acc00, c[cRow0+j:])
					hwy.Store(acc01, c[cRow0+j+lanes:])
					hwy.Store(acc10, c[cRow1+j:])
					hwy.Store(acc11, c[cRow1+j+lanes:])
					hwy.Store(acc20, c[cRow2+j:])
					hwy.Store(acc21, c[cRow2+j+lanes:])
					hwy.Store(acc30, c[cRow3+j:])
					hwy.Store(acc31, c[cRow3+j+lanes:])
				}

				// Handle remaining columns (less than Nr)
				for ; j < jEnd; j += lanes {
					remaining := jEnd - j
					if remaining >= lanes {
						// Full vector width, single column strip
						acc0 := hwy.Zero[T]()
						acc1 := hwy.Zero[T]()
						acc2 := hwy.Zero[T]()
						acc3 := hwy.Zero[T]()

						for p := 0; p < k; p++ {
							vA0 := hwy.Set(a[i*k+p])
							vA1 := hwy.Set(a[(i+1)*k+p])
							vA2 := hwy.Set(a[(i+2)*k+p])
							vA3 := hwy.Set(a[(i+3)*k+p])

							vB := hwy.Load(b[p*n+j:])
							acc0 = hwy.MulAdd(vA0, vB, acc0)
							acc1 = hwy.MulAdd(vA1, vB, acc1)
							acc2 = hwy.MulAdd(vA2, vB, acc2)
							acc3 = hwy.MulAdd(vA3, vB, acc3)
						}

						hwy.Store(acc0, c[i*n+j:])
						hwy.Store(acc1, c[(i+1)*n+j:])
						hwy.Store(acc2, c[(i+2)*n+j:])
						hwy.Store(acc3, c[(i+3)*n+j:])
					} else {
						// Scalar tail
						for jj := j; jj < jEnd; jj++ {
							var sum0, sum1, sum2, sum3 T
							for p := 0; p < k; p++ {
								bpj := b[p*n+jj]
								sum0 += a[i*k+p] * bpj
								sum1 += a[(i+1)*k+p] * bpj
								sum2 += a[(i+2)*k+p] * bpj
								sum3 += a[(i+3)*k+p] * bpj
							}
							c[i*n+jj] = sum0
							c[(i+1)*n+jj] = sum1
							c[(i+2)*n+jj] = sum2
							c[(i+3)*n+jj] = sum3
						}
						break
					}
				}
			}

			// Handle remaining rows - process pairs when possible for SIMD efficiency
			// This avoids the per-row overhead when M % 4 != 0

			// Process pairs of remaining rows with SIMD
			for i+2 <= iEnd {
				cRow0 := i * n
				cRow1 := (i + 1) * n

				var j int
				for j = j0; j+lanes <= jEnd; j += lanes {
					acc0 := hwy.Zero[T]()
					acc1 := hwy.Zero[T]()

					for p := 0; p < k; p++ {
						vA0 := hwy.Set(a[i*k+p])
						vA1 := hwy.Set(a[(i+1)*k+p])
						vB := hwy.Load(b[p*n+j:])
						acc0 = hwy.MulAdd(vA0, vB, acc0)
						acc1 = hwy.MulAdd(vA1, vB, acc1)
					}

					hwy.Store(acc0, c[cRow0+j:])
					hwy.Store(acc1, c[cRow1+j:])
				}

				// Scalar tail for remaining columns
				for ; j < jEnd; j++ {
					var sum0, sum1 T
					for p := 0; p < k; p++ {
						bp := b[p*n+j]
						sum0 += a[i*k+p] * bp
						sum1 += a[(i+1)*k+p] * bp
					}
					c[cRow0+j] = sum0
					c[cRow1+j] = sum1
				}

				i += 2
			}

			// Handle final single row if M % 2 == 1
			for ; i < iEnd; i++ {
				cRowStart := i * n

				var j int
				for j = j0; j+lanes <= jEnd; j += lanes {
					acc := hwy.Zero[T]()
					for p := 0; p < k; p++ {
						vA := hwy.Set(a[i*k+p])
						vB := hwy.Load(b[p*n+j:])
						acc = hwy.MulAdd(vA, vB, acc)
					}
					hwy.Store(acc, c[cRowStart+j:])
				}

				// Scalar tail for remaining columns
				for ; j < jEnd; j++ {
					var sum T
					for p := 0; p < k; p++ {
						sum += a[i*k+p] * b[p*n+j]
					}
					c[cRowStart+j] = sum
				}
			}
		}
	}
}
