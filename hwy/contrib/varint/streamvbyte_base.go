// Copyright 2025 The Go Highway Authors
// SPDX-License-Identifier: Apache-2.0
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

// Package varint provides variable-length integer encoding/decoding with SIMD acceleration.
package varint

import (
	_ "github.com/ajroetker/go-highway/hwy" // for future SIMD use
)

// streamVByte32DataLen[control] = sum of 4 value lengths for this control byte.
// Each 2-bit field encodes (length - 1), so we add 1 to each.
var streamVByte32DataLen [256]uint8

func init() {
	for control := 0; control < 256; control++ {
		len0 := ((control >> 0) & 0x3) + 1
		len1 := ((control >> 2) & 0x3) + 1
		len2 := ((control >> 4) & 0x3) + 1
		len3 := ((control >> 6) & 0x3) + 1
		streamVByte32DataLen[control] = uint8(len0 + len1 + len2 + len3)
	}
}

// StreamVByteEncoder accumulates uint32 values and encodes them.
type StreamVByteEncoder struct {
	control []byte     // control bytes
	data    []byte     // data bytes
	pending [4]uint32  // pending values (0-3)
	count   int        // count of pending values
}

// NewStreamVByteEncoder creates a new encoder.
func NewStreamVByteEncoder() *StreamVByteEncoder {
	return &StreamVByteEncoder{
		control: make([]byte, 0, 64),
		data:    make([]byte, 0, 256),
	}
}

// Add adds a value to the encoder.
func (e *StreamVByteEncoder) Add(v uint32) {
	e.pending[e.count] = v
	e.count++
	if e.count == 4 {
		e.flushPending()
	}
}

// AddBatch adds multiple values.
func (e *StreamVByteEncoder) AddBatch(values []uint32) {
	for _, v := range values {
		e.Add(v)
	}
}

// flushPending encodes the 4 pending values.
func (e *StreamVByteEncoder) flushPending() {
	var ctrl byte
	for i := 0; i < 4; i++ {
		v := e.pending[i]
		length := encodedLength(v)
		ctrl |= byte(length-1) << (i * 2)
		e.appendValue(v, length)
	}
	e.control = append(e.control, ctrl)
	e.count = 0
}

// appendValue appends a value as little-endian bytes.
func (e *StreamVByteEncoder) appendValue(v uint32, length int) {
	switch length {
	case 1:
		e.data = append(e.data, byte(v))
	case 2:
		e.data = append(e.data, byte(v), byte(v>>8))
	case 3:
		e.data = append(e.data, byte(v), byte(v>>8), byte(v>>16))
	case 4:
		e.data = append(e.data, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
	}
}

// encodedLength returns the number of bytes needed to encode v.
func encodedLength(v uint32) int {
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

// Finish finalizes encoding and returns (control bytes, data bytes).
// If there are pending values (not a multiple of 4), pads with zeros.
func (e *StreamVByteEncoder) Finish() (control, data []byte) {
	// Pad pending values with zeros if needed
	if e.count > 0 {
		for i := e.count; i < 4; i++ {
			e.pending[i] = 0
		}
		e.flushPending()
	}
	return e.control, e.data
}

// Reset clears the encoder for reuse.
func (e *StreamVByteEncoder) Reset() {
	e.control = e.control[:0]
	e.data = e.data[:0]
	e.count = 0
}

// BaseEncodeStreamVByte32 encodes uint32 values to Stream-VByte format.
// Returns (control bytes, data bytes).
// If len(values) is not a multiple of 4, pads with zeros.
func BaseEncodeStreamVByte32(values []uint32) (control, data []byte) {
	if len(values) == 0 {
		return nil, nil
	}

	// Calculate control bytes needed
	numGroups := (len(values) + 3) / 4
	control = make([]byte, numGroups)

	// Estimate data size (worst case: 4 bytes per value)
	data = make([]byte, 0, len(values)*4)

	for g := 0; g < numGroups; g++ {
		var ctrl byte
		baseIdx := g * 4

		for i := 0; i < 4; i++ {
			var v uint32
			if baseIdx+i < len(values) {
				v = values[baseIdx+i]
			}
			// else v = 0 (padding)

			length := encodedLength(v)
			ctrl |= byte(length-1) << (i * 2)

			// Append value bytes (little-endian)
			switch length {
			case 1:
				data = append(data, byte(v))
			case 2:
				data = append(data, byte(v), byte(v>>8))
			case 3:
				data = append(data, byte(v), byte(v>>8), byte(v>>16))
			case 4:
				data = append(data, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
			}
		}
		control[g] = ctrl
	}

	return control, data
}

// BaseDecodeStreamVByte32 decodes uint32 values from Stream-VByte format.
// control contains the control bytes, data contains the value bytes.
// n is the number of values to decode (must be <= len(control)*4).
// Returns the decoded values.
func BaseDecodeStreamVByte32(control, data []byte, n int) []uint32 {
	if n <= 0 || len(control) == 0 {
		return nil
	}

	result := make([]uint32, n)
	BaseDecodeStreamVByte32Into(control, data, result)
	return result
}

// BaseDecodeStreamVByte32Into decodes into a pre-allocated dst slice.
// Returns number of values decoded and data bytes consumed.
func BaseDecodeStreamVByte32Into(control, data []byte, dst []uint32) (decoded, dataConsumed int) {
	if len(dst) == 0 || len(control) == 0 {
		return 0, 0
	}

	dataPos := 0
	dstPos := 0
	n := len(dst)

	for _, ctrl := range control {
		if dstPos >= n {
			break
		}

		// Decode up to 4 values from this control byte
		for i := 0; i < 4 && dstPos < n; i++ {
			length := int(((ctrl >> (i * 2)) & 0x3) + 1)

			if dataPos+length > len(data) {
				// Not enough data
				return dstPos, dataPos
			}

			var v uint32
			switch length {
			case 1:
				v = uint32(data[dataPos])
			case 2:
				v = uint32(data[dataPos]) | uint32(data[dataPos+1])<<8
			case 3:
				v = uint32(data[dataPos]) | uint32(data[dataPos+1])<<8 | uint32(data[dataPos+2])<<16
			case 4:
				v = uint32(data[dataPos]) | uint32(data[dataPos+1])<<8 | uint32(data[dataPos+2])<<16 | uint32(data[dataPos+3])<<24
			}

			dst[dstPos] = v
			dstPos++
			dataPos += length
		}
	}

	return dstPos, dataPos
}

// BaseStreamVByte32DataLen returns the data length for given control bytes.
// This is useful for allocating the right buffer size.
func BaseStreamVByte32DataLen(control []byte) int {
	total := 0
	for _, ctrl := range control {
		total += int(streamVByte32DataLen[ctrl])
	}
	return total
}
