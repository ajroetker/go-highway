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

package asm

import "encoding/binary"

// BitsFromMaskFast extracts bit 0 from each byte into a 16-bit mask.
// This is an optimized implementation that uses bit manipulation instead of
// loading constants from memory, which is ~2x faster than the USHL+ADDV approach.
//
// Input: comparison result where each byte is either 0xFF (true) or 0x00 (false)
// Output: bit i is set if byte i had bit 0 set (i.e., was 0xFF)
//
// Algorithm:
// 1. Extract low and high 64 bits
// 2. Mask bit 0 of each byte: AND with 0x0101010101010101
// 3. Multiply by magic constant 0x0102040810204080 to gather bits
// 4. The high byte now contains all 8 bits packed together
// 5. Combine low and high halves
func BitsFromMaskFast(v Uint8x16) uint64 {
	lo := binary.LittleEndian.Uint64(v[0:8])
	hi := binary.LittleEndian.Uint64(v[8:16])

	// Mask to extract bit 0 of each byte (0xFF has bit 0 set, 0x00 doesn't)
	const mask = 0x0101010101010101

	lo = lo & mask
	hi = hi & mask

	// Magic multiplier: 2^0 + 2^8 + 2^16 + 2^24 + 2^32 + 2^40 + 2^48 + 2^56
	// This shifts each byte's bit 0 to its final position and sums them in the high byte
	// Byte 0's bit ends up at bit 56
	// Byte 1's bit ends up at bit 57
	// ...
	// Byte 7's bit ends up at bit 63
	// After >> 56, we get the 8-bit mask
	const magic = 0x0102040810204080

	lo = (lo * magic) >> 56
	hi = (hi * magic) >> 56

	return lo | (hi << 8)
}
