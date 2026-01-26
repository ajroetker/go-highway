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

// RaBitQ SIMD Operations for ARM64
// Uses NEON assembly for all ARM64 platforms.
package asm

import (
	"fmt"
	"unsafe"
)

// Generate NEON assembly from C source
//go:generate go tool goat ../c/rabitq_neon_arm64.c -O3 --target arm64

// Generate SME assembly from C source (for Apple M4+)
//go:generate go tool goat ../c/rabitq_sme_arm64.c -O3 --target arm64 -march=armv9-a+sme

// BitProductNEON computes the RaBitQ bit product using NEON.
// Returns: 1*popcount(code & q1) + 2*popcount(code & q2) + 4*popcount(code & q3) + 8*popcount(code & q4)
func BitProductNEON(code, query1, query2, query3, query4 []uint64) uint32 {
	if len(code) != len(query1) || len(code) != len(query2) ||
		len(code) != len(query3) || len(code) != len(query4) {
		panic(fmt.Errorf("BitProductNEON: mismatched lengths"))
	}
	if len(code) == 0 {
		panic(fmt.Errorf("BitProductNEON: zero length"))
	}

	// Defensive nil-pointer checks for unsafe.SliceData to prevent segfaults
	if unsafe.SliceData(code) == nil || unsafe.SliceData(query1) == nil ||
		unsafe.SliceData(query2) == nil || unsafe.SliceData(query3) == nil ||
		unsafe.SliceData(query4) == nil {
		panic(fmt.Errorf("BitProductNEON: nil slice data pointer detected"))
	}

	l := int64(len(code))

	var sum uint64
	rabitq_bit_product_neon(
		unsafe.Pointer(unsafe.SliceData(code)),
		unsafe.Pointer(unsafe.SliceData(query1)),
		unsafe.Pointer(unsafe.SliceData(query2)),
		unsafe.Pointer(unsafe.SliceData(query3)),
		unsafe.Pointer(unsafe.SliceData(query4)),
		unsafe.Pointer(&sum),
		unsafe.Pointer(&l),
	)
	return uint32(sum)
}
