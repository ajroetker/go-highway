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

// Group Varint encoding functions. These are pure Go implementations.
// Decode functions are in groupvarint_base.go and have GoAT-compiled
// assembly overrides on ARM64 via z_varint_neon_arm64.go.

// EncodeGroupVarint32 encodes 4 uint32 values to group-varint format.
// dst must have at least 17 bytes capacity (1 control + 4*4 max).
// Returns number of bytes written.
//
// Example:
//
//	values := [4]uint32{300, 5, 1000, 2}
//	dst := make([]byte, 17)
//	n := EncodeGroupVarint32(values, dst)
//	// dst[:n] contains the encoded data
func EncodeGroupVarint32(values [4]uint32, dst []byte) int {
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

// EncodeGroupVarint64 encodes 4 uint64 values to group-varint format.
// dst must have at least 34 bytes capacity (2 control + 4*8 max).
// Returns number of bytes written.
//
// Example:
//
//	values := [4]uint64{300, 5, 1000, 2}
//	dst := make([]byte, 34)
//	n := EncodeGroupVarint64(values, dst)
//	// dst[:n] contains the encoded data
func EncodeGroupVarint64(values [4]uint64, dst []byte) int {
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

// GroupVarint32Len returns the encoded length for 4 uint32 values.
// This is useful for pre-allocating destination buffers.
//
// Example:
//
//	values := [4]uint32{300, 5, 1000, 2}
//	length := GroupVarint32Len(values)  // Returns 7
func GroupVarint32Len(values [4]uint32) int {
	return 1 + bytesNeeded32(values[0]) + bytesNeeded32(values[1]) +
		bytesNeeded32(values[2]) + bytesNeeded32(values[3])
}

// GroupVarint64Len returns the encoded length for 4 uint64 values.
// This is useful for pre-allocating destination buffers.
//
// Example:
//
//	values := [4]uint64{300, 5, 1000, 2}
//	length := GroupVarint64Len(values)  // Returns 8
func GroupVarint64Len(values [4]uint64) int {
	return 2 + bytesNeeded64(values[0]) + bytesNeeded64(values[1]) +
		bytesNeeded64(values[2]) + bytesNeeded64(values[3])
}
