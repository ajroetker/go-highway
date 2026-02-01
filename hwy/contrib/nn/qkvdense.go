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

import (
	"sync"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// QKVDenseAuto computes a fused QKV projection using the best available
// matmul implementation, then scatters and adds biases.
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
// This delegates to MatMulKLastAuto for the fused matmul, then scatters
// the results into separate Q, K, V buffers and adds biases.
//
// For asymmetric dims (GQA-style where kvDim << qDim), it automatically
// falls back to 3× separate DenseAuto calls when the problem is small enough,
// as this gives better cache utilization.

// SmallFusedThreshold is the total element count (batchSize * totalOut * inFeatures)
// below which asymmetric QKV dims (qDim != kvDim) use 3× separate DenseAuto calls
// instead of the fused matmul path.
const SmallFusedThreshold = 768 * (256 + 2*64)

func QKVDenseAuto[T hwy.Floats](
	pool *workerpool.Pool,
	x, wQKV, biasQ, biasK, biasV, q, k, v []T,
	batchSize, inFeatures, qDim, kvDim int,
) {
	totalOut := qDim + 2*kvDim

	// For asymmetric dims (GQA-style where kvDim << qDim), three separate
	// DenseAuto calls are faster because each fits cache better.
	if qDim != kvDim && batchSize*totalOut*inFeatures < SmallFusedThreshold {
		wQ := wQKV[:qDim*inFeatures]
		wK := wQKV[qDim*inFeatures : (qDim+kvDim)*inFeatures]
		wV := wQKV[(qDim+kvDim)*inFeatures:]
		DenseAuto(pool, x, wQ, biasQ, q, batchSize, inFeatures, qDim)
		DenseAuto(pool, x, wK, biasK, k, batchSize, inFeatures, kvDim)
		DenseAuto(pool, x, wV, biasV, v, batchSize, inFeatures, kvDim)
		return
	}

	// Get temp buffer from pool
	temp := getTempSlice[T](batchSize * totalOut)
	defer putTempSlice(temp)

	// Fused matmul: temp = x @ wQKV^T
	matmul.MatMulKLastAuto(pool, x, wQKV, temp, batchSize, totalOut, inFeatures)

	// Scatter + bias add
	lanes := hwy.MaxLanes[T]()

	for i := range batchSize {
		tOff := i * totalOut
		qOff := i * qDim
		kOff := i * kvDim
		vOff := i * kvDim

		// Copy Q segment + bias
		scatterWithBias(temp[tOff:tOff+qDim], q[qOff:qOff+qDim], biasQ, qDim, lanes)

		// Copy K segment + bias
		scatterWithBias(temp[tOff+qDim:tOff+qDim+kvDim], k[kOff:kOff+kvDim], biasK, kvDim, lanes)

		// Copy V segment + bias
		scatterWithBias(temp[tOff+qDim+kvDim:tOff+totalOut], v[vOff:vOff+kvDim], biasV, kvDim, lanes)
	}
}

// scatterWithBias copies src to dst with optional SIMD bias add.
func scatterWithBias[T hwy.Floats](src, dst, bias []T, dim, lanes int) {
	if bias != nil {
		j := 0
		for ; j+lanes <= dim; j += lanes {
			s := hwy.LoadFull(src[j:])
			b := hwy.LoadFull(bias[j:])
			hwy.StoreFull(hwy.Add(s, b), dst[j:])
		}
		for ; j < dim; j++ {
			dst[j] = src[j] + bias[j]
		}
	} else {
		copy(dst[:dim], src[:dim])
	}
}

// QKVDenseScalar is a scalar reference implementation for comparison and testing.
func QKVDenseScalar[T hwy.Floats](
	x, wQKV, biasQ, biasK, biasV, q, k, v []T,
	batchSize, inFeatures, qDim, kvDim int,
) {
	totalOut := qDim + 2*kvDim

	for i := range batchSize {
		xOff := i * inFeatures

		for j := range totalOut {
			wOff := j * inFeatures
			var sum float64
			for p := range inFeatures {
				sum += float64(x[xOff+p]) * float64(wQKV[wOff+p])
			}

			if j < qDim {
				if biasQ != nil {
					sum += float64(biasQ[j])
				}
				q[i*qDim+j] = T(sum)
			} else if j < qDim+kvDim {
				kj := j - qDim
				if biasK != nil {
					sum += float64(biasK[kj])
				}
				k[i*kvDim+kj] = T(sum)
			} else {
				vj := j - qDim - kvDim
				if biasV != nil {
					sum += float64(biasV[vj])
				}
				v[i*kvDim+vj] = T(sum)
			}
		}
	}
}

// Pool for temporary float32 slices.
var tempPoolF32 = sync.Pool{
	New: func() any { return &[]float32{} },
}

// Pool for temporary float64 slices.
var tempPoolF64 = sync.Pool{
	New: func() any { return &[]float64{} },
}

// getTempSlice gets a temporary slice of at least the given size from a pool.
func getTempSlice[T hwy.Floats](size int) []T {
	var zero T
	switch any(zero).(type) {
	case float32:
		p := tempPoolF32.Get().(*[]float32)
		if cap(*p) < size {
			*p = make([]float32, size)
		}
		*p = (*p)[:size]
		return any(*p).([]T)
	case float64:
		p := tempPoolF64.Get().(*[]float64)
		if cap(*p) < size {
			*p = make([]float64, size)
		}
		*p = (*p)[:size]
		return any(*p).([]T)
	default:
		return make([]T, size)
	}
}

// putTempSlice returns a temporary slice to its pool.
func putTempSlice[T hwy.Floats](s []T) {
	var zero T
	switch any(zero).(type) {
	case float32:
		f := any(s).([]float32)
		tempPoolF32.Put(&f)
	case float64:
		f := any(s).([]float64)
		tempPoolF64.Put(&f)
	}
}
