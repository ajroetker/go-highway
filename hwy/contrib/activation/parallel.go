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

package activation

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// Parallel tuning parameters for row-parallel activation operations.
const (
	// MinParallelActivationOps is the minimum total element count before
	// parallelizing memory-bound activation operations.
	// Benchmarked on M4 Max (14 cores): parallel overhead is ~3.5µs, so
	// parallelism pays off above ~10K elements. 16384 gives a clear win
	// across GELU, ReLU, SiLU, Tanh, Softmax, and LayerNorm.
	MinParallelActivationOps = 16384

	// ActivationRowBatch is the number of rows handed to each worker in a
	// single batch via ParallelForAtomicBatched.
	ActivationRowBatch = 4
)

// ---------------------------------------------------------------------------
// Generic row-parallel helper
// ---------------------------------------------------------------------------

// ParallelApplyRows applies fn to each row of a [rows, cols] matrix in
// parallel. fn receives the input and output slices for a single row.
//
// Falls back to sequential execution when pool is nil or the total element
// count is below MinParallelActivationOps.
func ParallelApplyRows[T hwy.Floats](pool workerpool.Executor, input, output []T, rows, cols int, fn func(input, output []T)) {
	if pool == nil || rows*cols < MinParallelActivationOps {
		for r := range rows {
			off := r * cols
			fn(input[off:off+cols], output[off:off+cols])
		}
		return
	}

	pool.ParallelForAtomicBatched(rows, ActivationRowBatch, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * cols
			fn(input[off:off+cols], output[off:off+cols])
		}
	})
}

// ---------------------------------------------------------------------------
// Parallel activations
// ---------------------------------------------------------------------------

// ParallelGELU applies GELU element-wise across a [rows, cols] matrix in
// parallel.
func ParallelGELU[T hwy.Floats](pool workerpool.Executor, input, output []T, rows, cols int) {
	ParallelApplyRows(pool, input, output, rows, cols, func(in, out []T) {
		GELU(in, out)
	})
}

// ParallelGELUApprox applies the fast approximate GELU across a [rows, cols]
// matrix in parallel.
func ParallelGELUApprox[T hwy.Floats](pool workerpool.Executor, input, output []T, rows, cols int) {
	ParallelApplyRows(pool, input, output, rows, cols, func(in, out []T) {
		GELUApprox(in, out)
	})
}

// ParallelReLU applies ReLU element-wise across a [rows, cols] matrix in
// parallel.
func ParallelReLU[T hwy.Floats](pool workerpool.Executor, input, output []T, rows, cols int) {
	ParallelApplyRows(pool, input, output, rows, cols, func(in, out []T) {
		ReLU(in, out)
	})
}

// ParallelSiLU applies SiLU (Swish) element-wise across a [rows, cols] matrix
// in parallel.
func ParallelSiLU[T hwy.Floats](pool workerpool.Executor, input, output []T, rows, cols int) {
	ParallelApplyRows(pool, input, output, rows, cols, func(in, out []T) {
		SiLU(in, out)
	})
}

// ParallelTanh applies Tanh element-wise across a [rows, cols] matrix in
// parallel.
func ParallelTanh[T hwy.Floats](pool workerpool.Executor, input, output []T, rows, cols int) {
	ParallelApplyRows(pool, input, output, rows, cols, func(in, out []T) {
		Tanh(in, out)
	})
}

// ParallelLeakyReLU applies LeakyReLU(alpha) element-wise across a
// [rows, cols] matrix in parallel.
func ParallelLeakyReLU[T hwy.Floats](pool workerpool.Executor, input, output []T, rows, cols int, alpha T) {
	ParallelApplyRows(pool, input, output, rows, cols, func(in, out []T) {
		LeakyReLU(in, out, alpha)
	})
}

// ParallelELU applies ELU(alpha) element-wise across a [rows, cols] matrix in
// parallel.
func ParallelELU[T hwy.Floats](pool workerpool.Executor, input, output []T, rows, cols int, alpha T) {
	ParallelApplyRows(pool, input, output, rows, cols, func(in, out []T) {
		ELU(in, out, alpha)
	})
}

