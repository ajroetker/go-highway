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

// Dispatch function variables.
// These are initialized to the base (pure Go) implementations and may be
// overridden by architecture-specific optimized implementations in init().

// Standard varint functions
var (
	// FindVarintEnds returns a bitmask where bit i is set if src[i] < 0x80
	FindVarintEnds func(src []byte) uint32

	// DecodeUvarint64Batch decodes up to n varints from src into dst
	DecodeUvarint64Batch func(src []byte, dst []uint64, n int) (decoded, consumed int)

	// Decode2Uvarint64 decodes exactly 2 varints (freq/norm pattern)
	Decode2Uvarint64 func(src []byte) (v1, v2 uint64, consumed int)

	// Decode5Uvarint64 decodes exactly 5 varints (location pattern)
	Decode5Uvarint64 func(src []byte) (values [5]uint64, consumed int)
)

// Group varint functions
var (
	// DecodeGroupVarint32 decodes 4 uint32 values from group-varint format
	DecodeGroupVarint32 func(src []byte) (values [4]uint32, consumed int)

	// DecodeGroupVarint64 decodes 4 uint64 values from group-varint format
	DecodeGroupVarint64 func(src []byte) (values [4]uint64, consumed int)

	// EncodeGroupVarint32 encodes 4 uint32 values to group-varint format
	EncodeGroupVarint32 func(values [4]uint32, dst []byte) int

	// EncodeGroupVarint64 encodes 4 uint64 values to group-varint format
	EncodeGroupVarint64 func(values [4]uint64, dst []byte) int

	// GroupVarint32Len returns the encoded length for 4 uint32 values
	GroupVarint32Len func(values [4]uint32) int

	// GroupVarint64Len returns the encoded length for 4 uint64 values
	GroupVarint64Len func(values [4]uint64) int
)

// Stream-VByte functions (SIMD-friendly separated control/data streams)
var (
	// EncodeStreamVByte32 encodes uint32 values to Stream-VByte format
	// Returns (control bytes, data bytes)
	EncodeStreamVByte32 func(values []uint32) (control, data []byte)

	// DecodeStreamVByte32 decodes uint32 values from Stream-VByte format
	// n is the number of values to decode
	DecodeStreamVByte32 func(control, data []byte, n int) []uint32

	// DecodeStreamVByte32Into decodes into pre-allocated dst slice
	// Returns (values decoded, data bytes consumed)
	DecodeStreamVByte32Into func(control, data []byte, dst []uint32) (decoded, dataConsumed int)

	// StreamVByte32DataLen returns total data length from control bytes
	StreamVByte32DataLen func(control []byte) int
)

func init() {
	// Initialize with base (pure Go) implementations.
	// These may be overridden by architecture-specific init() functions
	// in z_*.go files (e.g., z_varint_neon_arm64.go for ARM64 NEON).

	// Standard varint
	FindVarintEnds = BaseFindVarintEnds
	DecodeUvarint64Batch = BaseDecodeUvarint64Batch
	Decode2Uvarint64 = BaseDecode2Uvarint64
	Decode5Uvarint64 = BaseDecode5Uvarint64

	// Group varint
	DecodeGroupVarint32 = BaseDecodeGroupVarint32
	DecodeGroupVarint64 = BaseDecodeGroupVarint64
	EncodeGroupVarint32 = BaseEncodeGroupVarint32
	EncodeGroupVarint64 = BaseEncodeGroupVarint64
	GroupVarint32Len = BaseGroupVarint32Len
	GroupVarint64Len = BaseGroupVarint64Len

	// Stream-VByte
	EncodeStreamVByte32 = BaseEncodeStreamVByte32
	DecodeStreamVByte32 = BaseDecodeStreamVByte32
	DecodeStreamVByte32Into = BaseDecodeStreamVByte32Into
	StreamVByte32DataLen = BaseStreamVByte32DataLen
}
