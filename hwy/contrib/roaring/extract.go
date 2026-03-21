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

package roaring

import "math/bits"

// ExtractBitPositions extracts the positions of all set bits in bitmap into out.
// Each position is a uint16 offset within the bitmap (0..len(bitmap)*64-1).
// Returns the number of positions written.
//
// This is a drop-in replacement for roaring's bitmapContainer.fillLeastSignificant16bits
// and is significantly faster because it uses TrailingZeros64 (single instruction on
// modern CPUs) instead of popcount(t-1).
//
// The caller must ensure out is large enough to hold all set bits. For a roaring
// bitmap container (1024 uint64 words), the maximum is 65536 positions.
var ExtractBitPositions func(bitmap []uint64, out []uint16) int = extractBitPositions

func extractBitPositions(bitmap []uint64, out []uint16) int {
	pos := 0
	for k, bitset := range bitmap {
		base := uint16(k * 64)
		for bitset != 0 {
			tz := bits.TrailingZeros64(bitset)
			out[pos] = base + uint16(tz)
			pos++
			bitset &= bitset - 1 // clear lowest set bit
		}
	}
	return pos
}

// ExtractBitPositionsAND extracts the positions of all set bits in (a & b) into out.
// Returns the number of positions written.
//
// This is a drop-in replacement for roaring's fillArrayAND, fusing the AND
// operation with bit extraction in a single pass.
var ExtractBitPositionsAND func(out []uint16, a, b []uint64) int = extractBitPositionsAND

func extractBitPositionsAND(out []uint16, a, b []uint64) int {
	n := min(len(a), len(b))
	pos := 0
	for k := 0; k < n; k++ {
		bitset := a[k] & b[k]
		base := uint16(k * 64)
		for bitset != 0 {
			tz := bits.TrailingZeros64(bitset)
			out[pos] = base + uint16(tz)
			pos++
			bitset &= bitset - 1
		}
	}
	return pos
}

// ExtractBitPositionsANDNOT extracts the positions of all set bits in (a &^ b) into out.
// Returns the number of positions written.
//
// This is a drop-in replacement for roaring's fillArrayANDNOT, fusing the ANDNOT
// operation with bit extraction in a single pass.
var ExtractBitPositionsANDNOT func(out []uint16, a, b []uint64) int = extractBitPositionsANDNOT

func extractBitPositionsANDNOT(out []uint16, a, b []uint64) int {
	n := min(len(a), len(b))
	pos := 0
	for k := 0; k < n; k++ {
		bitset := a[k] &^ b[k]
		base := uint16(k * 64)
		for bitset != 0 {
			tz := bits.TrailingZeros64(bitset)
			out[pos] = base + uint16(tz)
			pos++
			bitset &= bitset - 1
		}
	}
	return pos
}

// ExtractBitPositionsXOR extracts the positions of all set bits in (a ^ b) into out.
// Returns the number of positions written.
//
// This is a drop-in replacement for roaring's fillArrayXOR, fusing the XOR
// operation with bit extraction in a single pass.
var ExtractBitPositionsXOR func(out []uint16, a, b []uint64) int = extractBitPositionsXOR

func extractBitPositionsXOR(out []uint16, a, b []uint64) int {
	n := min(len(a), len(b))
	pos := 0
	for k := 0; k < n; k++ {
		bitset := a[k] ^ b[k]
		base := uint16(k * 64)
		for bitset != 0 {
			tz := bits.TrailingZeros64(bitset)
			out[pos] = base + uint16(tz)
			pos++
			bitset &= bitset - 1
		}
	}
	return pos
}
