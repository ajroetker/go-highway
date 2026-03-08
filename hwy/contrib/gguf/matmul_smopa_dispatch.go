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

package gguf

import "github.com/ajroetker/go-highway/hwy/contrib/workerpool"

// SMEGGUFMatMul is the SME-accelerated GGUF matmul using SMOPA/SUMOPA integer
// outer products. Set to non-nil by init() on SME-capable ARM64 hardware.
var SMEGGUFMatMul func(input []float32, weights []uint8, output []float32,
	M, K, N int, qt QuantType)

// SMEParallelGGUFMatMul is the parallel SME-accelerated GGUF matmul.
var SMEParallelGGUFMatMul func(pool workerpool.Executor, input []float32,
	weights []uint8, output []float32, M, K, N int, qt QuantType)

// SMEPreparedGGUFMatMul is the SME-accelerated GGUF matmul using pre-packed
// weights. The 4-tile kernel eliminates per-inference B-panel packing overhead.
// Set to non-nil by init() on SME-capable ARM64 hardware.
var SMEPreparedGGUFMatMul func(input []float32, pw *PreparedWeights,
	output []float32, M int)

// SMEParallelPreparedGGUFMatMul is the parallel version of SMEPreparedGGUFMatMul.
var SMEParallelPreparedGGUFMatMul func(pool workerpool.Executor, input []float32,
	pw *PreparedWeights, output []float32, M int)
