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

package nn

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input qkvdense_base.go -output . -targets avx2,avx512,neon,fallback

// BaseQKVDense computes a fused QKV projection: a single matmul against stacked
// Q/K/V weights, then splits and adds per-segment biases.
//
//   - x:      [batchSize, inFeatures] (row-major)
//   - wQKV:   [(qDim + 2*kvDim), inFeatures] (row-major, stacked Q, K, V weights)
//   - biasQ:  [qDim] (optional, pass nil to skip)
//   - biasK:  [kvDim] (optional, pass nil to skip)
//   - biasV:  [kvDim] (optional, pass nil to skip)
//   - q:      [batchSize, qDim] output
//   - k:      [batchSize, kvDim] output
//   - v:      [batchSize, kvDim] output
//
// This fuses the matmul, scatter, and bias-add into a single pass, avoiding
// a temporary buffer and separate scatter copy. Each output row is computed
// via SIMD dot-product accumulation with 4-row unrolling on batchSize.
func BaseQKVDense[T hwy.Floats](
	x, wQKV, biasQ, biasK, biasV, q, k, v []T,
	batchSize, inFeatures, qDim, kvDim int,
) {
	totalOut := qDim + 2*kvDim

	if len(x) < batchSize*inFeatures {
		panic("qkvdense: x slice too short")
	}
	if len(wQKV) < totalOut*inFeatures {
		panic("qkvdense: wQKV slice too short")
	}
	if len(q) < batchSize*qDim {
		panic("qkvdense: q slice too short")
	}
	if len(k) < batchSize*kvDim {
		panic("qkvdense: k slice too short")
	}
	if len(v) < batchSize*kvDim {
		panic("qkvdense: v slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()

	// Process 4 batch rows at a time for better register utilization
	var i int
	for i = 0; i+3 < batchSize; i += 4 {
		xRow0 := i * inFeatures
		xRow1 := (i + 1) * inFeatures
		xRow2 := (i + 2) * inFeatures
		xRow3 := (i + 3) * inFeatures

		// Compute dot products for all output columns, writing directly to q/k/v
		for j := range totalOut {
			wRow := j * inFeatures

			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()

			var p int
			for p = 0; p+lanes <= inFeatures; p += lanes {
				vW := hwy.LoadSlice(wQKV[wRow+p:])

				vX0 := hwy.LoadSlice(x[xRow0+p:])
				vX1 := hwy.LoadSlice(x[xRow1+p:])
				vX2 := hwy.LoadSlice(x[xRow2+p:])
				vX3 := hwy.LoadSlice(x[xRow3+p:])

				acc0 = hwy.MulAdd(vX0, vW, acc0)
				acc1 = hwy.MulAdd(vX1, vW, acc1)
				acc2 = hwy.MulAdd(vX2, vW, acc2)
				acc3 = hwy.MulAdd(vX3, vW, acc3)
			}

			sum0 := hwy.ReduceSum(acc0)
			sum1 := hwy.ReduceSum(acc1)
			sum2 := hwy.ReduceSum(acc2)
			sum3 := hwy.ReduceSum(acc3)

			for ; p < inFeatures; p++ {
				sum0 += x[xRow0+p] * wQKV[wRow+p]
				sum1 += x[xRow1+p] * wQKV[wRow+p]
				sum2 += x[xRow2+p] * wQKV[wRow+p]
				sum3 += x[xRow3+p] * wQKV[wRow+p]
			}

			// Write directly to the correct output segment with bias
			if j < qDim {
				// Q segment
				if biasQ != nil {
					b := biasQ[j]
					sum0 += b
					sum1 += b
					sum2 += b
					sum3 += b
				}
				q[i*qDim+j] = sum0
				q[(i+1)*qDim+j] = sum1
				q[(i+2)*qDim+j] = sum2
				q[(i+3)*qDim+j] = sum3
			} else if j < qDim+kvDim {
				// K segment
				kj := j - qDim
				if biasK != nil {
					b := biasK[kj]
					sum0 += b
					sum1 += b
					sum2 += b
					sum3 += b
				}
				k[i*kvDim+kj] = sum0
				k[(i+1)*kvDim+kj] = sum1
				k[(i+2)*kvDim+kj] = sum2
				k[(i+3)*kvDim+kj] = sum3
			} else {
				// V segment
				vj := j - qDim - kvDim
				if biasV != nil {
					b := biasV[vj]
					sum0 += b
					sum1 += b
					sum2 += b
					sum3 += b
				}
				v[i*kvDim+vj] = sum0
				v[(i+1)*kvDim+vj] = sum1
				v[(i+2)*kvDim+vj] = sum2
				v[(i+3)*kvDim+vj] = sum3
			}
		}
	}

	// Handle remaining rows (0-3)
	for ; i < batchSize; i++ {
		xRow := i * inFeatures

		for j := range totalOut {
			wRow := j * inFeatures
			acc := hwy.Zero[T]()

			var p int
			for p = 0; p+lanes <= inFeatures; p += lanes {
				vX := hwy.LoadSlice(x[xRow+p:])
				vW := hwy.LoadSlice(wQKV[wRow+p:])
				acc = hwy.MulAdd(vX, vW, acc)
			}

			sum := hwy.ReduceSum(acc)
			for ; p < inFeatures; p++ {
				sum += x[xRow+p] * wQKV[wRow+p]
			}

			if j < qDim {
				if biasQ != nil {
					sum += biasQ[j]
				}
				q[i*qDim+j] = sum
			} else if j < qDim+kvDim {
				kj := j - qDim
				if biasK != nil {
					sum += biasK[kj]
				}
				k[i*kvDim+kj] = sum
			} else {
				vj := j - qDim - kvDim
				if biasV != nil {
					sum += biasV[vj]
				}
				v[i*kvDim+vj] = sum
			}
		}
	}
}
