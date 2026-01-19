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

package varint

import "github.com/ajroetker/go-highway/hwy"

// Group Varint encoding is a SIMD-friendly alternative to standard LEB128 varint encoding.
// It encodes 4 values with a single control byte, providing ~4x throughput compared to
// byte-at-a-time LEB128 encoding.
//
// Format for uint32 (1-byte control):
//   [control byte][value0 bytes][value1 bytes][value2 bytes][value3 bytes]
//
// Control byte: 2 bits per value indicating byte length minus 1
//   - Bits 0-1: length of value0 minus 1 (0=1 byte, 1=2 bytes, 2=3 bytes, 3=4 bytes)
//   - Bits 2-3: length of value1 minus 1
//   - Bits 4-5: length of value2 minus 1
//   - Bits 6-7: length of value3 minus 1
//
// Format for uint64 (2-byte control):
//   [control byte 0][control byte 1][value0 bytes][value1 bytes][value2 bytes][value3 bytes]
//
// Control uses 3 bits per value (supporting 1-8 bytes per value):
//   - Bits 0-2:  length of value0 minus 1
//   - Bits 3-5:  length of value1 minus 1
//   - Bits 6-8:  length of value2 minus 1
//   - Bits 9-11: length of value3 minus 1

// Lookup tables for decoding uint32 group varint.
// For each control byte (0-255), stores total encoded length and per-value offsets.
var (
	// groupVarint32TotalLen[control] = 1 + sum of value lengths
	groupVarint32TotalLen [256]uint8

	// groupVarint32Offsets[control][i] = offset to value i within encoded data
	groupVarint32Offsets [256][4]uint8
)

func init() {
	// Build lookup tables for uint32 group varint decoding
	for control := 0; control < 256; control++ {
		len0 := ((control >> 0) & 0x3) + 1
		len1 := ((control >> 2) & 0x3) + 1
		len2 := ((control >> 4) & 0x3) + 1
		len3 := ((control >> 6) & 0x3) + 1

		groupVarint32TotalLen[control] = uint8(1 + len0 + len1 + len2 + len3)

		groupVarint32Offsets[control][0] = 1 // value0 starts after control byte
		groupVarint32Offsets[control][1] = uint8(1 + len0)
		groupVarint32Offsets[control][2] = uint8(1 + len0 + len1)
		groupVarint32Offsets[control][3] = uint8(1 + len0 + len1 + len2)
	}
}

// bytesNeeded32 returns the number of bytes needed to encode a uint32 (1-4).
func bytesNeeded32(v uint32) int {
	switch {
	case v < 1<<8:
		return 1
	case v < 1<<16:
		return 2
	case v < 1<<24:
		return 3
	default:
		return 4
	}
}

// bytesNeeded64 returns the number of bytes needed to encode a uint64 (1-8).
func bytesNeeded64(v uint64) int {
	switch {
	case v < 1<<8:
		return 1
	case v < 1<<16:
		return 2
	case v < 1<<24:
		return 3
	case v < 1<<32:
		return 4
	case v < 1<<40:
		return 5
	case v < 1<<48:
		return 6
	case v < 1<<56:
		return 7
	default:
		return 8
	}
}

// BaseDecodeGroupVarint32 decodes 4 uint32 values from group-varint format.
// Returns the 4 values and number of bytes consumed (1 + sum of value lengths).
//
// The src slice must contain at least 1 byte (the control byte) and enough
// subsequent bytes for all encoded values. If src is too short, returns
// zeros and 0 bytes consumed.
//
// Example:
//
//	// Encoded: control=0x11, then 2+1+2+1=6 value bytes
//	src := []byte{0x11, 0x2C, 0x01, 0x05, 0xE8, 0x03, 0x02}
//	values, consumed := BaseDecodeGroupVarint32(src)
//	// values = [300, 5, 1000, 2], consumed = 7
func BaseDecodeGroupVarint32(src []byte) (values [4]uint32, consumed int) {
	if len(src) < 1 {
		return [4]uint32{}, 0
	}

	control := src[0]
	totalLen := int(groupVarint32TotalLen[control])

	if len(src) < totalLen {
		return [4]uint32{}, 0
	}

	offsets := groupVarint32Offsets[control]

	// Decode each value using the precomputed offsets
	values[0] = decodeValue32(src, int(offsets[0]), ((int(control)>>0)&0x3)+1)
	values[1] = decodeValue32(src, int(offsets[1]), ((int(control)>>2)&0x3)+1)
	values[2] = decodeValue32(src, int(offsets[2]), ((int(control)>>4)&0x3)+1)
	values[3] = decodeValue32(src, int(offsets[3]), ((int(control)>>6)&0x3)+1)

	return values, totalLen
}

// decodeValue32 reads a little-endian uint32 of the specified byte length.
func decodeValue32(src []byte, offset, length int) uint32 {
	var v uint32
	for i := 0; i < length; i++ {
		v |= uint32(src[offset+i]) << (8 * i)
	}
	return v
}

// BaseDecodeGroupVarint64 decodes 4 uint64 values from group-varint format.
// For uint64, each value can be 1-8 bytes. The control uses 2 bytes (12 bits = 4 * 3 bits).
// Returns the 4 values and number of bytes consumed (2 + sum of value lengths).
//
// Control format (little-endian 12-bit value across 2 bytes):
//   - Bits 0-2:  length of value0 minus 1 (0-7 = 1-8 bytes)
//   - Bits 3-5:  length of value1 minus 1
//   - Bits 6-8:  length of value2 minus 1
//   - Bits 9-11: length of value3 minus 1
//
// Example:
//
//	src := []byte{...}  // encoded data
//	values, consumed := BaseDecodeGroupVarint64(src)
func BaseDecodeGroupVarint64(src []byte) (values [4]uint64, consumed int) {
	if len(src) < 2 {
		return [4]uint64{}, 0
	}

	// Read 12-bit control from 2 bytes (only lower 12 bits used)
	control := uint16(src[0]) | (uint16(src[1]) << 8)

	// Extract lengths (3 bits each, value is length-1)
	len0 := int((control>>0)&0x7) + 1
	len1 := int((control>>3)&0x7) + 1
	len2 := int((control>>6)&0x7) + 1
	len3 := int((control>>9)&0x7) + 1

	totalLen := 2 + len0 + len1 + len2 + len3

	if len(src) < totalLen {
		return [4]uint64{}, 0
	}

	// Compute offsets and decode
	offset := 2
	values[0] = decodeValue64(src, offset, len0)
	offset += len0
	values[1] = decodeValue64(src, offset, len1)
	offset += len1
	values[2] = decodeValue64(src, offset, len2)
	offset += len2
	values[3] = decodeValue64(src, offset, len3)

	return values, totalLen
}

// decodeValue64 reads a little-endian uint64 of the specified byte length.
func decodeValue64(src []byte, offset, length int) uint64 {
	var v uint64
	for i := 0; i < length; i++ {
		v |= uint64(src[offset+i]) << (8 * i)
	}
	return v
}

// BaseEncodeGroupVarint32 encodes 4 uint32 values to group-varint format.
// dst must have at least 17 bytes capacity (1 control + 4*4 max).
// Returns number of bytes written.
//
// Example:
//
//	values := [4]uint32{300, 5, 1000, 2}
//	dst := make([]byte, 17)
//	n := BaseEncodeGroupVarint32(values, dst)
//	// dst[:n] contains the encoded data
func BaseEncodeGroupVarint32(values [4]uint32, dst []byte) int {
	if len(dst) < 17 {
		return 0
	}

	// Compute byte lengths
	len0 := bytesNeeded32(values[0])
	len1 := bytesNeeded32(values[1])
	len2 := bytesNeeded32(values[2])
	len3 := bytesNeeded32(values[3])

	// Build control byte: 2 bits per value (length - 1)
	control := byte((len0-1)<<0 | (len1-1)<<2 | (len2-1)<<4 | (len3-1)<<6)
	dst[0] = control

	// Encode values in little-endian format
	offset := 1
	offset += encodeValue32(values[0], dst[offset:], len0)
	offset += encodeValue32(values[1], dst[offset:], len1)
	offset += encodeValue32(values[2], dst[offset:], len2)
	offset += encodeValue32(values[3], dst[offset:], len3)

	return offset
}

// encodeValue32 writes a uint32 in little-endian format using the specified byte length.
// Returns the number of bytes written.
func encodeValue32(v uint32, dst []byte, length int) int {
	for i := 0; i < length; i++ {
		dst[i] = byte(v >> (8 * i))
	}
	return length
}

// BaseEncodeGroupVarint64 encodes 4 uint64 values to group-varint format.
// dst must have at least 34 bytes capacity (2 control + 4*8 max).
// Returns number of bytes written.
//
// Example:
//
//	values := [4]uint64{300, 5, 1000, 2}
//	dst := make([]byte, 34)
//	n := BaseEncodeGroupVarint64(values, dst)
//	// dst[:n] contains the encoded data
func BaseEncodeGroupVarint64(values [4]uint64, dst []byte) int {
	if len(dst) < 34 {
		return 0
	}

	// Compute byte lengths
	len0 := bytesNeeded64(values[0])
	len1 := bytesNeeded64(values[1])
	len2 := bytesNeeded64(values[2])
	len3 := bytesNeeded64(values[3])

	// Build 12-bit control: 3 bits per value (length - 1)
	control := uint16((len0-1)<<0 | (len1-1)<<3 | (len2-1)<<6 | (len3-1)<<9)
	dst[0] = byte(control)
	dst[1] = byte(control >> 8)

	// Encode values in little-endian format
	offset := 2
	offset += encodeValue64(values[0], dst[offset:], len0)
	offset += encodeValue64(values[1], dst[offset:], len1)
	offset += encodeValue64(values[2], dst[offset:], len2)
	offset += encodeValue64(values[3], dst[offset:], len3)

	return offset
}

// encodeValue64 writes a uint64 in little-endian format using the specified byte length.
// Returns the number of bytes written.
func encodeValue64(v uint64, dst []byte, length int) int {
	for i := 0; i < length; i++ {
		dst[i] = byte(v >> (8 * i))
	}
	return length
}

// BaseGroupVarint32Len returns the encoded length for 4 uint32 values.
// This is useful for pre-allocating destination buffers.
//
// Example:
//
//	values := [4]uint32{300, 5, 1000, 2}
//	length := BaseGroupVarint32Len(values)  // Returns 7
func BaseGroupVarint32Len(values [4]uint32) int {
	return 1 + bytesNeeded32(values[0]) + bytesNeeded32(values[1]) +
		bytesNeeded32(values[2]) + bytesNeeded32(values[3])
}

// BaseGroupVarint64Len returns the encoded length for 4 uint64 values.
// This is useful for pre-allocating destination buffers.
//
// Example:
//
//	values := [4]uint64{300, 5, 1000, 2}
//	length := BaseGroupVarint64Len(values)  // Returns 8
func BaseGroupVarint64Len(values [4]uint64) int {
	return 2 + bytesNeeded64(values[0]) + bytesNeeded64(values[1]) +
		bytesNeeded64(values[2]) + bytesNeeded64(values[3])
}

// The following are placeholder references to hwy package to ensure
// the import is used. SIMD-accelerated implementations will use these
// for shuffle-based decoding.
var _ = hwy.Zero[uint32]
