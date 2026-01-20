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

package varint

import (
	"os"
	"runtime"

	"github.com/ajroetker/go-highway/hwy/contrib/varint/asm"
)

// ARM64 optimized varint operations.
//
// These override the hwygen-generated dispatch with C code compiled via GoAT
// (clang -O3 → Go assembly).
//
// The NEON implementation provides:
//   - Vectorized boundary detection (16 bytes at a time)
//   - Optimized scalar loops compiled with -O3
//
// Performance benefits:
//   - FindVarintEnds: NEON loads 16 bytes and uses vector comparison to detect
//     all varint boundaries in parallel, vs scalar byte-at-a-time processing.
//   - Batch decoders: Clang -O3 generates tighter scalar code than Go's compiler
//     (ARM post-increment addressing, better register allocation, loop unrolling).
//   - Group varint: Optimized shuffle-based decoding using NEON TBL instruction.

func init() {
	// Respect HWY_NO_SIMD to allow fallback testing
	if os.Getenv("HWY_NO_SIMD") != "" {
		return
	}

	// Override generated dispatch with GoAT-compiled implementations.
	// These run after the generated init() in dispatch_varint_arm64.gen.go
	// and dispatch_groupvarint_arm64.gen.go due to file naming (z_ sorts after dispatch_).

	// Standard varint operations
	FindVarintEnds = wrapFindVarintEnds
	DecodeUvarint64Batch = asm.DecodeUvarint64Batch
	Decode2Uvarint64 = asm.Decode2Uvarint64
	Decode5Uvarint64 = asm.Decode5Uvarint64

	// Group varint operations
	DecodeGroupVarint32 = asm.DecodeGroupVarint32
	DecodeGroupVarint64 = asm.DecodeGroupVarint64

	// Stream-VByte SIMD decode
	// Note: Skip on Linux ARM64 due to frame pointer unwinding crash.
	// The StreamVByte function uses all available registers, so clang uses
	// x29/x30 as scratch registers. When Go's runtime sends an async preemption
	// signal while x29 is corrupted, the frame pointer unwinder crashes.
	// See: https://github.com/golang/go/issues/63830
	// GoAT attempts to substitute x29/x30 with free registers, but this
	// function has no free registers available.
	if runtime.GOOS != "linux" {
		DecodeStreamVByte32Into = asm.DecodeStreamVByte32Into
	}
}

// wrapFindVarintEnds adapts the asm function signature.
// The asm version returns uint64 (64 bits for up to 64 byte positions),
// but the dispatch function returns uint32 (32 bits for up to 32 positions).
func wrapFindVarintEnds(src []byte) uint32 {
	return uint32(asm.FindVarintEnds(src))
}
