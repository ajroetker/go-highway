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

package specialize

import "github.com/ajroetker/go-highway/hwy"

// BaseMulAddHalf computes element-wise fused multiply-add for half-precision types.
//
// Float16 and BFloat16 aren't native to Go's simd package, so this
// specialization restricts to NEON assembly where the GOAT transpiler
// can emit native fp16/bf16 instructions. The function loads half-precision
// values, widens to float32 for the FMA computation, then narrows back.
//
// On non-NEON targets (AVX2, AVX-512, fallback), Float16/BFloat16 are
// not generated -- callers should promote to float32 before dispatch.
//
//hwy:gen T={hwy.Float16, hwy.BFloat16}
//hwy:specializes MulAdd
//hwy:targets neon:asm
func BaseMulAddHalf[T hwy.Floats](x, y, out []T) {
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
