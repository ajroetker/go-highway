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

//go:generate go run ../../../cmd/hwygen -input dense_base.go -output . -targets avx2,avx512,neon,fallback

// BaseDense computes a dense (fully-connected) layer: output = x @ weight^T + bias.
//
//   - x is [batchSize, inFeatures] (row-major)
//   - weight is [outFeatures, inFeatures] (row-major, PyTorch format)
//   - bias is [outFeatures] (optional, pass nil to skip)
//   - output is [batchSize, outFeatures] (row-major)
//
// This uses SIMD dot-product accumulation along inFeatures with 4-row unrolling,
// matching the BaseMatMulKLast pattern, plus an optional SIMD bias add.
func BaseDense[T hwy.Floats](x, weight, bias, output []T, batchSize, inFeatures, outFeatures int) {
	if len(x) < batchSize*inFeatures {
		panic("dense: x slice too short")
	}
	if len(weight) < outFeatures*inFeatures {
		panic("dense: weight slice too short")
	}
	if len(output) < batchSize*outFeatures {
		panic("dense: output slice too short")
	}
	if bias != nil && len(bias) < outFeatures {
		panic("dense: bias slice too short")
	}

	lanes := hwy.Zero[T]().NumLanes()

	// Process 4 rows of x at a time for better register utilization
	var i int
	for i = 0; i+3 < batchSize; i += 4 {
		xRow0 := i * inFeatures
		xRow1 := (i + 1) * inFeatures
		xRow2 := (i + 2) * inFeatures
		xRow3 := (i + 3) * inFeatures

		oRow0 := i * outFeatures
		oRow1 := (i + 1) * outFeatures
		oRow2 := (i + 2) * outFeatures
		oRow3 := (i + 3) * outFeatures

		for j := range outFeatures {
			wRow := j * inFeatures

			acc0 := hwy.Zero[T]()
			acc1 := hwy.Zero[T]()
			acc2 := hwy.Zero[T]()
			acc3 := hwy.Zero[T]()

			var p int
			for p = 0; p+lanes <= inFeatures; p += lanes {
				vW := hwy.LoadSlice(weight[wRow+p:])

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
				sum0 += x[xRow0+p] * weight[wRow+p]
				sum1 += x[xRow1+p] * weight[wRow+p]
				sum2 += x[xRow2+p] * weight[wRow+p]
				sum3 += x[xRow3+p] * weight[wRow+p]
			}

			if bias != nil {
				b := bias[j]
				sum0 += b
				sum1 += b
				sum2 += b
				sum3 += b
			}

			output[oRow0+j] = sum0
			output[oRow1+j] = sum1
			output[oRow2+j] = sum2
			output[oRow3+j] = sum3
		}
	}

	// Handle remaining rows (0-3)
	for ; i < batchSize; i++ {
		xRow := i * inFeatures
		oRow := i * outFeatures

		for j := range outFeatures {
			wRow := j * inFeatures
			acc := hwy.Zero[T]()

			var p int
			for p = 0; p+lanes <= inFeatures; p += lanes {
				vX := hwy.LoadSlice(x[xRow+p:])
				vW := hwy.LoadSlice(weight[wRow+p:])
				acc = hwy.MulAdd(vX, vW, acc)
			}

			sum := hwy.ReduceSum(acc)
			for ; p < inFeatures; p++ {
				sum += x[xRow+p] * weight[wRow+p]
			}

			if bias != nil {
				sum += bias[j]
			}

			output[oRow+j] = sum
		}
	}
}
