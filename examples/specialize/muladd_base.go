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

// Package specialize demonstrates the //hwy:specializes and //hwy:targets
// directives for architecture-specific template specialization.
//
// This example shows a fused multiply-add (MulAdd) with a primary generic
// implementation and a NEON:asm specialization for half-precision types:
//
//   - Primary (this file): handles float32/float64 on all targets using
//     generic hwy operations. On NEON:asm this compiles to assembly via
//     GOAT; on AVX2/AVX-512 it uses Go's simd package; on fallback it
//     uses scalar code.
//
//   - Specialization (muladd_half_base.go): adds Float16/BFloat16 on
//     NEON:asm only, providing a body that the GOAT transpiler compiles
//     to native fp16/bf16 instructions. These types are NOT available on
//     AVX2/AVX-512/fallback targets.
//
// The dispatch group "MulAdd" unifies both under a single MulAdd[T]() entry point.
//
// Usage:
//
//	go generate ./...
//	GOEXPERIMENT=simd go build
package specialize

//go:generate go run ../../cmd/hwygen -input muladd_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch specialize

import "github.com/ajroetker/go-highway/hwy"

// BaseMulAdd computes element-wise fused multiply-add: out[i] += x[i] * y[i].
//
// Uses SIMD FMA instructions for vectorized throughput across all targets.
// This primary generates for float32 and float64 only. Float16/BFloat16
// are added by the specialization in muladd_half_base.go (NEON:asm only).
//
//hwy:gen T={float32, float64}
func BaseMulAdd[T hwy.Floats](x, y, out []T) {
	size := min(len(x), min(len(y), len(out)))
	if size == 0 {
		return
	}
	lanes := hwy.Zero[T]().NumLanes()

	var i int
	for ; i+lanes <= size; i += lanes {
		vx := hwy.Load(x[i:])
		vy := hwy.Load(y[i:])
		vo := hwy.Load(out[i:])
		hwy.Store(hwy.MulAdd(vx, vy, vo), out[i:])
	}
	// Scalar tail
	for ; i < size; i++ {
		out[i] += x[i] * y[i]
	}
}
