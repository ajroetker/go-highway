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
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// ---------------------------------------------------------------------------
// Parallel softmax variants (row-batched)
// ---------------------------------------------------------------------------

// ParallelSoftmax applies Softmax independently to each row of a [rows, cols]
// matrix in parallel.
func ParallelSoftmax[T hwy.Floats](pool workerpool.Executor, input, output []T, rows, cols int) {
	activation.ParallelApplyRows(pool, input, output, rows, cols, func(in, out []T) {
		Softmax(in, out)
	})
}

// ParallelLogSoftmax applies LogSoftmax independently to each row of a
// [rows, cols] matrix in parallel.
func ParallelLogSoftmax[T hwy.Floats](pool workerpool.Executor, input, output []T, rows, cols int) {
	activation.ParallelApplyRows(pool, input, output, rows, cols, func(in, out []T) {
		LogSoftmax(in, out)
	})
}

// ParallelSoftmaxWithTemperature applies SoftmaxWithTemperature independently to
// each row of a [rows, cols] matrix in parallel.
func ParallelSoftmaxWithTemperature[T hwy.Floats](pool workerpool.Executor, input, output []T, rows, cols int, temperature T) {
	activation.ParallelApplyRows(pool, input, output, rows, cols, func(in, out []T) {
		SoftmaxWithTemperature(in, out, temperature)
	})
}

// ---------------------------------------------------------------------------
// Parallel LayerNorm
// ---------------------------------------------------------------------------

// ParallelLayerNorm applies LayerNorm in parallel across normalization groups.
// The input and output are flat slices of length numGroups*normSize, where each
// contiguous group of normSize elements is normalized independently.
func ParallelLayerNorm[T hwy.Floats](pool workerpool.Executor, input, output []T, normSize int, gamma, beta []T, epsilon T) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}
	numGroups := size / normSize

	if pool == nil || numGroups*normSize < activation.MinParallelActivationOps {
		LayerNorm(input, output, normSize, gamma, beta, epsilon)
		return
	}

	pool.ParallelForAtomicBatched(numGroups, activation.ActivationRowBatch, func(start, end int) {
		inSlice := input[start*normSize : end*normSize]
		outSlice := output[start*normSize : end*normSize]
		LayerNorm(inSlice, outSlice, normSize, gamma, beta, epsilon)
	})
}

