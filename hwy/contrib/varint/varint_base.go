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

//go:generate go run ../../../cmd/hwygen -input varint_base.go -output . -targets avx2,avx512,neon,fallback -dispatch varint

// Package varint provides SIMD-accelerated batch varint (LEB128) decoding.
//
// Varints are variable-length integers commonly used in full-text search
// engines (like Bleve/zapx) and protocol buffers. Each byte stores 7 bits
// of data, with the high bit indicating whether more bytes follow.
//
// This package accelerates varint decoding by:
//   - Using SIMD to detect varint boundaries in parallel
//   - Providing batch decoders optimized for common patterns
//   - Offering fixed-count decoders for known tuple sizes
//
// Example usage:
//
//	// Batch decode varints
//	values := make([]uint64, 100)
//	decoded, consumed := varint.DecodeUvarint64Batch(data, values, 100)
//
//	// Decode freq/norm pair (2 varints)
//	freq, norm, n := varint.Decode2Uvarint64(data)
//
//	// Decode location fields (5 varints)
//	loc, n := varint.Decode5Uvarint64(data)
package varint

import "github.com/ajroetker/go-highway/hwy"

// BaseFindVarintEnds examines up to 32 bytes and returns a bitmask where
// bit i is set if src[i] is the last byte of a varint (i.e., src[i] < 0x80).
// This enables SIMD-accelerated boundary detection.
//
// The returned bitmask can be used with bits.TrailingZeros32 to find the
// position of the first varint end, or with bits.OnesCount32 to count
// how many complete varints are in the window.
//
// SIMD implementations load 16-32 bytes at once and use vector comparisons
// to detect all varint boundaries in parallel.
//
// Example:
//
//	// Find boundaries in a buffer
//	mask := BaseFindVarintEnds(data[:32])
//	// If mask = 0b00001010, bytes 1 and 3 are varint terminators
//	// (they have high bit clear)
func BaseFindVarintEnds(src []byte) uint32 {
	if len(src) == 0 {
		return 0
	}

	// Limit to 32 bytes (one uint32 bitmask)
	n := min(len(src), 32)

	// SIMD path: process 32 bytes using two 16-byte operations
	// This works for both NEON (16-byte vectors) and AVX2/AVX512 (which can also use 16-byte ops)
	// Use LoadSlice since we're explicitly working with 16-byte chunks regardless of vector width
	if n == 32 {
		// Create threshold for comparison: bytes < 0x80 are terminators
		threshold := hwy.Set[uint8](0x80)

		// Process first 16 bytes
		v0 := hwy.LoadSlice[uint8](src[:16])
		isTerminator0 := hwy.LessThan(v0, threshold)
		mask0 := uint32(hwy.BitsFromMask(isTerminator0))

		// Process second 16 bytes
		v1 := hwy.LoadSlice[uint8](src[16:32])
		isTerminator1 := hwy.LessThan(v1, threshold)
		mask1 := uint32(hwy.BitsFromMask(isTerminator1))

		// Combine masks: lower 16 bits from first half, upper 16 bits from second half
		return mask0 | (mask1 << 16)
	}

	// Scalar fallback for partial buffers
	var mask uint32
	for i := range n {
		if src[i] < 0x80 {
			mask |= 1 << uint(i)
		}
	}
	return mask
}

// BaseDecodeUvarint64Batch decodes up to n varints from src into dst.
// Returns (values decoded, bytes consumed).
//
// This is more efficient than calling binary.Uvarint N times because:
//   - It can use SIMD to find varint boundaries
//   - It minimizes bounds checking overhead
//   - It provides predictable memory access patterns
//
// The function stops when:
//   - n values have been decoded
//   - The end of src is reached
//   - An incomplete varint is encountered
//
// Example:
//
//	dst := make([]uint64, 1000)
//	decoded, consumed := BaseDecodeUvarint64Batch(data, dst, 1000)
//	values := dst[:decoded]
//	remaining := data[consumed:]
func BaseDecodeUvarint64Batch(src []byte, dst []uint64, n int) (decoded int, consumed int) {
	if len(src) == 0 || n == 0 || len(dst) == 0 {
		return 0, 0
	}

	maxDecode := min(n, len(dst))
	pos := 0

	for decoded < maxDecode && pos < len(src) {
		val, bytesRead := decodeOneUvarint64(src[pos:])
		if bytesRead == 0 {
			// Incomplete varint
			break
		}
		dst[decoded] = val
		decoded++
		pos += bytesRead
	}

	return decoded, pos
}

// BaseDecode2Uvarint64 decodes exactly 2 varints from src.
// Returns (v1, v2, bytes consumed).
//
// This is optimized for decoding freq/norm pairs in posting lists,
// a common pattern in full-text search engines like Bleve/zapx.
//
// If the buffer doesn't contain 2 complete varints, returns (0, 0, 0).
//
// Example:
//
//	// Decode frequency and norm from posting list
//	freq, norm, n := BaseDecode2Uvarint64(postingData)
//	if n > 0 {
//	    processPosting(freq, norm)
//	    postingData = postingData[n:]
//	}
func BaseDecode2Uvarint64(src []byte) (v1, v2 uint64, consumed int) {
	if len(src) == 0 {
		return 0, 0, 0
	}

	// Decode first varint
	val1, n1 := decodeOneUvarint64(src)
	if n1 == 0 {
		return 0, 0, 0
	}

	// Decode second varint
	if n1 >= len(src) {
		return 0, 0, 0
	}
	val2, n2 := decodeOneUvarint64(src[n1:])
	if n2 == 0 {
		return 0, 0, 0
	}

	return val1, val2, n1 + n2
}

// BaseDecode5Uvarint64 decodes exactly 5 varints from src.
// Returns (values array, bytes consumed).
//
// This is optimized for decoding location fields in full-text search:
//   - values[0]: field number
//   - values[1]: position
//   - values[2]: start offset
//   - values[3]: end offset
//   - values[4]: number of array positions
//
// If the buffer doesn't contain 5 complete varints, returns (zero array, 0).
//
// Example:
//
//	// Decode location from posting list
//	loc, n := BaseDecode5Uvarint64(locData)
//	if n > 0 {
//	    field, pos, start, end, numArr := loc[0], loc[1], loc[2], loc[3], loc[4]
//	    locData = locData[n:]
//	}
func BaseDecode5Uvarint64(src []byte) (values [5]uint64, consumed int) {
	if len(src) == 0 {
		return [5]uint64{}, 0
	}

	pos := 0
	for i := range 5 {
		if pos >= len(src) {
			return [5]uint64{}, 0
		}
		val, n := decodeOneUvarint64(src[pos:])
		if n == 0 {
			return [5]uint64{}, 0
		}
		values[i] = val
		pos += n
	}

	return values, pos
}

// decodeOneUvarint64 decodes a single unsigned varint from src.
// Returns (value, bytes consumed). Returns (0, 0) if the varint is incomplete.
//
// LEB128/Varint format:
//   - Each byte contains 7 bits of data (bits 0-6)
//   - Bit 7 (high bit) indicates continuation: 1 = more bytes, 0 = final byte
//   - Bytes are stored little-endian (least significant first)
func decodeOneUvarint64(src []byte) (uint64, int) {
	var x uint64
	var s uint

	for i, b := range src {
		if i >= 10 {
			// Varint too long for uint64 (max 10 bytes for 64-bit)
			return 0, 0
		}
		if b < 0x80 {
			// Final byte: high bit is clear
			if i == 9 && b > 1 {
				// Overflow: 10th byte can only be 0 or 1 for uint64
				return 0, 0
			}
			return x | uint64(b)<<s, i + 1
		}
		// Continuation byte: extract 7 bits, set continuation
		x |= uint64(b&0x7f) << s
		s += 7
	}

	// Reached end of buffer without finding terminator
	return 0, 0
}

// BaseDecodeUvarint64BatchWithMask uses the boundary mask from BaseFindVarintEnds
// to accelerate batch decoding. This is useful when you've already computed
// boundaries for filtering or other purposes.
//
// The mask should have bits set for varint end positions (bytes < 0x80).
// This function processes varints whose end positions are marked in the mask.
//
// Returns (values decoded, bytes consumed).
func BaseDecodeUvarint64BatchWithMask(src []byte, dst []uint64, mask uint32, n int) (decoded int, consumed int) {
	if mask == 0 || len(src) == 0 || n == 0 || len(dst) == 0 {
		return 0, 0
	}

	maxDecode := min(n, len(dst))
	pos := 0
	startPos := 0

	// Process each set bit in the mask (each varint end)
	for mask != 0 && decoded < maxDecode {
		// Find position of next varint end
		endPos := trailingZeros32(mask)
		if endPos >= len(src) {
			break
		}

		// Decode the varint from startPos to endPos (inclusive)
		val, bytesRead := decodeOneUvarint64(src[startPos : endPos+1])
		if bytesRead == 0 || startPos+bytesRead-1 != endPos {
			// Decoding failed or position mismatch
			break
		}

		dst[decoded] = val
		decoded++
		pos = endPos + 1
		startPos = pos

		// Clear the bit we just processed
		mask &= mask - 1
	}

	return decoded, pos
}

// trailingZeros32 returns the number of trailing zero bits in x.
// Returns 32 if x is 0.
func trailingZeros32(x uint32) int {
	if x == 0 {
		return 32
	}
	n := 0
	for x&1 == 0 {
		n++
		x >>= 1
	}
	return n
}
