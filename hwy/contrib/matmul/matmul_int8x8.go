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

//go:generate go run ../../../cmd/hwygen -input matmul_int8x8.go -dispatch int8x8matmul -output . -targets avx2,avx512,neon:asm,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseInt8x8MatMul performs integer matrix multiplication of two uint8 matrices
// with zero-point subtraction, accumulating into int32.
//
// output[m,n] = sum_k( (int32(a[m,k]) - int32(aZP)) * (int32(b[k,n]) - int32(bZP)) )
//
// This is useful for quantized SDPA where both operands are quantized uint8
// tensors with per-tensor affine quantization (scale + zero point).
//
// Parameters:
//   - output: [M, N] int32 output matrix (row-major, pre-allocated)
//   - a: [M, K] uint8 input matrix (row-major)
//   - b: [K, N] uint8 input matrix (row-major)
//   - aZP: zero point for a
//   - bZP: zero point for b
//   - M, K, N: matrix dimensions
func BaseInt8x8MatMul(output []int32, a, b []uint8, aZP, bZP uint8, M, K, N int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	lanes := hwy.Zero[int32]().NumLanes()

	// Temporary buffer for dequantized b values (one vector width)
	dequantBuf := make([]int32, lanes)

	// Accumulator buffer for one output row
	accBuf := make([]int32, N)

	azp := int32(aZP)
	bzp := int32(bZP)

	// Process each output row
	for m := range M {
		// Zero accumulators for this row
		for i := range N {
			accBuf[i] = 0
		}

		// K-outer, N-inner: sequential weight access, single input broadcast per k
		for k := range K {
			aVal := int32(a[m*K+k]) - azp
			aVec := hwy.Set(aVal)
			baseIdx := k * N

			// Vectorized N sweep
			var n int
			for n = 0; n+lanes <= N; n += lanes {
				for lane := range lanes {
					dequantBuf[lane] = int32(b[baseIdx+n+lane]) - bzp
				}

				bVec := hwy.Load(dequantBuf)
				acc := hwy.Load(accBuf[n:])
				acc = hwy.Add(hwy.Mul(aVec, bVec), acc)
				hwy.Store(acc, accBuf[n:])
			}

			// Scalar tail
			for ; n < N; n++ {
				accBuf[n] += aVal * (int32(b[baseIdx+n]) - bzp)
			}
		}

		// Copy accBuf to output
		copy(output[m*N:(m+1)*N], accBuf)
	}
}
