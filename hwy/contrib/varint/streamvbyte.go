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

package varint

// Stream-VByte encoding functions.
// The main encode functions (EncodeStreamVByte32, EncodeStreamVByte32Into) are
// dispatched in dispatch_streamvbyte_*.gen.go and use SIMD acceleration.
// This file contains the streaming encoder API.

// StreamVByteEncoder accumulates uint32 values and encodes them.
type StreamVByteEncoder struct {
	control []byte    // control bytes
	data    []byte    // data bytes
	pending [4]uint32 // pending values (0-3)
	count   int       // count of pending values
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
	for i := range 4 {
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

// StreamVByte32DataLen returns the data length for given control bytes.
// This is useful for allocating the right buffer size.
func StreamVByte32DataLen(control []byte) int {
	total := 0
	for _, ctrl := range control {
		total += int(streamVByte32DataLen[ctrl])
	}
	return total
}

// DecodeStreamVByte32 decodes uint32 values from Stream-VByte format.
// control contains the control bytes, data contains the value bytes.
// n is the number of values to decode (must be <= len(control)*4).
// Returns the decoded values.
func DecodeStreamVByte32(control []byte, data []uint8, n int) []uint32 {
	if n <= 0 || len(control) == 0 {
		return nil
	}

	result := make([]uint32, n)
	DecodeStreamVByte32Into(control, data, result)
	return result
}

// EncodeStreamVByte32 encodes uint32 values to Stream-VByte format.
// Returns (control bytes, data bytes).
func EncodeStreamVByte32(values []uint32) (control, data []byte) {
	if len(values) == 0 {
		return nil, nil
	}

	numGroups := (len(values) + 3) / 4
	control = make([]byte, numGroups)
	data = make([]byte, 0, len(values)*4)

	// Scratch buffer for SIMD group encode (needs at least 16 bytes)
	var buf [16]byte

	for g := range numGroups {
		baseIdx := g * 4
		remaining := len(values) - baseIdx

		if remaining >= 4 {
			// Full group - use SIMD into scratch buffer
			ctrl, n := EncodeStreamVByte32Group(values[baseIdx:baseIdx+4], buf[:])
			control[g] = ctrl
			data = append(data, buf[:n]...)
		} else {
			// Partial group - scalar with padding
			var group [4]uint32
			copy(group[:], values[baseIdx:])
			ctrl, n := encodeGroupScalarInto(group[:], buf[:])
			control[g] = ctrl
			data = append(data, buf[:n]...)
		}
	}

	return control, data
}

// EncodeStreamVByte32Into encodes into pre-allocated buffers.
// Returns sliced control and data buffers.
func EncodeStreamVByte32Into(values []uint32, controlBuf, dataBuf []byte) (control, data []byte) {
	if len(values) == 0 {
		return nil, nil
	}

	numGroups := (len(values) + 3) / 4

	// Ensure buffers are large enough
	if cap(controlBuf) < numGroups {
		controlBuf = make([]byte, numGroups)
	} else {
		controlBuf = controlBuf[:numGroups]
	}

	maxDataLen := len(values) * 4
	if cap(dataBuf) < maxDataLen {
		dataBuf = make([]byte, maxDataLen)
	} else {
		dataBuf = dataBuf[:maxDataLen]
	}

	dataPos := 0
	for g := range numGroups {
		baseIdx := g * 4
		remaining := len(values) - baseIdx

		if remaining >= 4 && dataPos+16 <= len(dataBuf) {
			// Full group - use SIMD directly into buffer
			ctrl, n := EncodeStreamVByte32Group(values[baseIdx:baseIdx+4], dataBuf[dataPos:])
			controlBuf[g] = ctrl
			dataPos += n
		} else {
			// Partial group - scalar with padding
			var group [4]uint32
			copy(group[:], values[baseIdx:])
			ctrl, n := encodeGroupScalarInto(group[:], dataBuf[dataPos:])
			controlBuf[g] = ctrl
			dataPos += n
		}
	}

	return controlBuf, dataBuf[:dataPos]
}
