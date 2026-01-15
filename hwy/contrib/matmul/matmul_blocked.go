package matmul

//go:generate go run ../../../cmd/hwygen -input matmul_blocked.go -dispatch blockedmatmul -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

// Block size tuned for L1 cache (32KB typical).
// 3 blocks of 48x48 float32 = 3 * 48 * 48 * 4 = 27KB < 32KB L1.
// Must be a multiple of 16 for AVX-512 alignment.
const (
	BlockSize = 48
)

// BaseBlockedMatMul computes C = A * B using cache-tiled blocking.
// Uses "broadcast A, stream B" within each block to avoid horizontal reduction.
//
//   - A is M x K (row-major)
//   - B is K x N (row-major)
//   - C is M x N (row-major)
//
// This is more efficient than streaming for large matrices where data
// doesn't fit in cache, as it ensures each block of A, B, C stays in L1
// while being processed.
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

	// Block over all three dimensions.
	// Loop order: i-blocks, j-blocks, k-blocks ensures output block stays in cache
	// while we accumulate all contributions from K dimension.
	for i0 := 0; i0 < m; i0 += BlockSize {
		iEnd := min(i0+BlockSize, m)

		for j0 := 0; j0 < n; j0 += BlockSize {
			jEnd := min(j0+BlockSize, n)
			blockN := jEnd - j0

			for p0 := 0; p0 < k; p0 += BlockSize {
				pEnd := min(p0+BlockSize, k)

				// Multiply block: C[i0:iEnd, j0:jEnd] += A[i0:iEnd, p0:pEnd] * B[p0:pEnd, j0:jEnd]
				// Inlined "broadcast A, stream B" algorithm
				for i := i0; i < iEnd; i++ {
					cRowStart := i*n + j0

					for p := p0; p < pEnd; p++ {
						aip := a[i*k+p]
						vA := hwy.Set(aip) // Broadcast A[i,p] to all lanes
						bRowStart := p*n + j0

						// Vectorized: C[i, j0:jEnd] += A[i,p] * B[p, j0:jEnd]
						var j int
						for j = 0; j+lanes <= blockN; j += lanes {
							vB := hwy.Load(b[bRowStart+j:])
							vC := hwy.Load(c[cRowStart+j:])
							vC = hwy.MulAdd(vA, vB, vC)
							hwy.Store(vC, c[cRowStart+j:])
						}

						// Scalar tail for non-aligned block widths
						for ; j < blockN; j++ {
							c[cRowStart+j] += aip * b[bRowStart+j]
						}
					}
				}
			}
		}
	}
}
