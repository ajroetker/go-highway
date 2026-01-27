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

// GoAT-accelerated encode/decode for ARM64
// These use hand-optimized NEON assembly for maximum throughput (~2.5x faster than hwygen).
package vec

import (
	"github.com/ajroetker/go-highway/hwy/contrib/vec/asm"
)

// GoATEncodeFloat32s_neon encodes a float32 slice to bytes using GoAT NEON assembly.
func GoATEncodeFloat32s_neon(dst []byte, src []float32) {
	if len(src) == 0 {
		return
	}
	requiredBytes := len(src) * 4
	if len(dst) < requiredBytes {
		panic("dst is too short")
	}
	asm.EncodeFloat32s(dst, src)
}

// GoATDecodeFloat32s_neon decodes bytes to a float32 slice using GoAT NEON assembly.
func GoATDecodeFloat32s_neon(dst []float32, src []byte) {
	if len(dst) == 0 {
		return
	}
	requiredBytes := len(dst) * 4
	if len(src) < requiredBytes {
		panic("src is too short")
	}
	asm.DecodeFloat32s(dst, src)
}

// GoATEncodeFloat64s_neon encodes a float64 slice to bytes using GoAT NEON assembly.
func GoATEncodeFloat64s_neon(dst []byte, src []float64) {
	if len(src) == 0 {
		return
	}
	requiredBytes := len(src) * 8
	if len(dst) < requiredBytes {
		panic("dst is too short")
	}
	asm.EncodeFloat64s(dst, src)
}

// GoATDecodeFloat64s_neon decodes bytes to a float64 slice using GoAT NEON assembly.
func GoATDecodeFloat64s_neon(dst []float64, src []byte) {
	if len(dst) == 0 {
		return
	}
	requiredBytes := len(dst) * 8
	if len(src) < requiredBytes {
		panic("src is too short")
	}
	asm.DecodeFloat64s(dst, src)
}
