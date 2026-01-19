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

//go:build !noasm && arm64

package algo

import (
	"os"

	"github.com/ajroetker/go-highway/hwy/contrib/algo/asm"
)

// ARM64 optimized prefix sum and delta decode implementations.
//
// These override the hwygen-generated dispatch with C code compiled via GoAT
// (clang -O3 → Go assembly). This approach provides significant speedups:
//
// Performance vs Go scalar (1024 elements, Apple M4 Max):
//
//	| Type    | GoAT    | Go Scalar | Speedup |
//	|---------|---------|-----------|---------|
//	| Int64   | 346 ns  | 482 ns    | 1.4x    |
//	| Uint64  | 362 ns  | 482 ns    | 1.3x    |
//	| Int32   | 428 ns  | -         | NEON    |
//	| Uint32  | 427 ns  | -         | NEON    |
//	| Float32 | 362 ns  | 2313 ns   | 6.4x    |
//
// Implementation strategy by type:
//
//   - 32-bit types (int32, uint32, float32): Use NEON SIMD with 4 lanes.
//     The Hillis-Steele algorithm computes prefix sum within each 4-element
//     vector, then propagates carry across vectors.
//
//   - 64-bit integers (int64, uint64): Use scalar C loop compiled via GoAT.
//     With only 2 NEON lanes, SIMD overhead (vector↔scalar transfers for carry)
//     exceeds the parallelism benefit. Clang -O3 generates tighter scalar code
//     than Go's compiler (ARM post-increment addressing, better register alloc).
//
//   - 64-bit floats (float64): Use NEON with 2 lanes. Despite limited lanes,
//     NEON still wins because scalar float64 operations are more expensive.
//
// References:
//   - Hillis-Steele parallel prefix: https://en.wikipedia.org/wiki/Prefix_sum
//   - SIMD prefix sum analysis: https://arxiv.org/html/2312.14874v1

func init() {
	// Respect HWY_NO_SIMD to allow fallback testing
	if os.Getenv("HWY_NO_SIMD") != "" {
		return
	}

	// Override generated dispatch with GoAT-compiled implementations.
	// These run after the generated init() in dispatch_prefix_sum_arm64.gen.go
	// due to file naming (z_ sorts after dispatch_).
	PrefixSumFloat32 = asm.PrefixSumFloat32
	PrefixSumFloat64 = asm.PrefixSumFloat64
	PrefixSumInt32 = asm.PrefixSumInt32
	PrefixSumInt64 = asm.PrefixSumInt64
	PrefixSumUint32 = asm.PrefixSumUint32
	PrefixSumUint64 = asm.PrefixSumUint64

	DeltaDecodeInt32 = asm.DeltaDecodeInt32
	DeltaDecodeInt64 = asm.DeltaDecodeInt64
	DeltaDecodeUint32 = asm.DeltaDecodeUint32
	DeltaDecodeUint64 = asm.DeltaDecodeUint64
}
