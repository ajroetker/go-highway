/*
 * Copyright 2025 go-highway Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// NEON-optimized varint (LEB128) operations for ARM64
// These provide SIMD acceleration for finding varint boundaries and decoding.
//
// Varints use the high bit (bit 7) as a continuation flag:
//   - If bit 7 is set (byte >= 0x80): more bytes follow
//   - If bit 7 is clear (byte < 0x80): this is the final byte

#include <arm_neon.h>

// ============================================================================
// SIMD Varint Boundary Detection
// ============================================================================

// find_varint_ends_u8: Find positions where varints end (byte < 0x80)
// Examines up to 64 bytes and returns bitmask where bit i = 1 if src[i] < 0x80
//
// This is the key SIMD operation for accelerated varint decoding:
//   - Load 16 bytes into NEON vector
//   - Check high bit of each byte
//   - Convert to bitmask for fast boundary lookup
void find_varint_ends_u8(unsigned char *src, int64_t n, int64_t *result) {
    int64_t mask = 0;

    if (n <= 0) {
        *result = 0;
        return;
    }

    // Limit to 64 bits (max bitmask size)
    if (n > 64) {
        n = 64;
    }

    int64_t i = 0;

    // Process 16 bytes at a time with NEON
    for (; i + 16 <= n; i += 16) {
        uint8x16_t v = vld1q_u8(src + i);

        // Get the high bit of each byte by shifting right 7 positions
        // highBit[j] = 1 if byte[j] >= 0x80 (continuation), 0 if < 0x80 (terminator)
        uint8x16_t highBit = vshrq_n_u8(v, 7);

        // We need the inverse: terminator positions should be 1
        // XOR with 1 to flip: notHighBit[j] = 1 if byte[j] < 0x80
        uint8x16_t ones = vdupq_n_u8(1);
        uint8x16_t terminator = veorq_u8(highBit, ones);

        // Extract to scalar bitmask
        // NEON doesn't have movemask, so we accumulate lane by lane
        // Each lane is either 0 or 1, so we shift and OR
        uint64_t laneMask = 0;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 0) << 0;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 1) << 1;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 2) << 2;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 3) << 3;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 4) << 4;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 5) << 5;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 6) << 6;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 7) << 7;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 8) << 8;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 9) << 9;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 10) << 10;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 11) << 11;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 12) << 12;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 13) << 13;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 14) << 14;
        laneMask |= (uint64_t)vgetq_lane_u8(terminator, 15) << 15;

        mask |= laneMask << i;
    }

    // Handle remaining bytes with scalar loop
    for (; i < n; i++) {
        if (src[i] < 0x80) {
            mask |= 1LL << i;
        }
    }

    *result = mask;
}

// ============================================================================
// Batch Varint Decoding
// ============================================================================

// decode_uvarint64_batch: Decode multiple unsigned varints
// Decodes up to n varints from src into dst.
// Returns number of values decoded in *decoded, bytes consumed in *consumed.
//
// LEB128 format:
//   - Each byte stores 7 bits of data (bits 0-6)
//   - Bit 7 indicates continuation (1 = more bytes follow)
//   - Little-endian: least significant bytes first
void decode_uvarint64_batch(
    unsigned char *src, int64_t src_len,
    unsigned long long *dst, int64_t dst_len,
    int64_t n,
    int64_t *decoded, int64_t *consumed
) {
    *decoded = 0;
    *consumed = 0;

    if (src_len <= 0) {
        return;
    }
    if (dst_len <= 0) {
        return;
    }
    if (n <= 0) {
        return;
    }

    int64_t maxDecode = n;
    if (maxDecode > dst_len) {
        maxDecode = dst_len;
    }

    int64_t pos = 0;
    int64_t count = 0;

    for (; count < maxDecode; count++) {
        if (pos >= src_len) {
            break;
        }

        // Decode one varint
        unsigned long long val = 0;
        int64_t shift = 0;
        int64_t bytesRead = 0;

        for (;;) {
            if (pos + bytesRead >= src_len) {
                // Incomplete varint - stop decoding
                *decoded = count;
                *consumed = pos;
                return;
            }

            unsigned char b = src[pos + bytesRead];
            bytesRead++;

            // Check for overflow (max 10 bytes for uint64)
            if (bytesRead > 10) {
                // Varint too long
                *decoded = count;
                *consumed = pos;
                return;
            }

            // Check for overflow on 10th byte
            if (bytesRead == 10) {
                if (b > 1) {
                    // Overflow
                    *decoded = count;
                    *consumed = pos;
                    return;
                }
            }

            val |= ((unsigned long long)(b & 0x7f)) << shift;
            shift += 7;

            if (b < 0x80) {
                // Final byte - high bit is clear
                break;
            }
        }

        dst[count] = val;
        pos += bytesRead;
    }

    *decoded = count;
    *consumed = pos;
}

// ============================================================================
// Group Varint Decoding (SIMD-friendly format)
// ============================================================================

// decode_group_varint32: Decode 4 uint32 values from group varint format
// Group varint uses a control byte followed by variable-length values.
//
// Control byte (2 bits per value):
//   - Bits 0-1: length of value0 minus 1 (0=1 byte, 1=2 bytes, etc.)
//   - Bits 2-3: length of value1 minus 1
//   - Bits 4-5: length of value2 minus 1
//   - Bits 6-7: length of value3 minus 1
//
// Returns bytes consumed in *consumed (0 if error)
void decode_group_varint32(
    unsigned char *src, int64_t src_len,
    unsigned int *values,
    int64_t *consumed
) {
    *consumed = 0;

    if (src_len < 1) {
        return;
    }

    unsigned char control = src[0];

    // Extract lengths from control byte (2 bits each, value is length-1)
    int64_t len0 = ((control >> 0) & 0x3) + 1;
    int64_t len1 = ((control >> 2) & 0x3) + 1;
    int64_t len2 = ((control >> 4) & 0x3) + 1;
    int64_t len3 = ((control >> 6) & 0x3) + 1;

    int64_t totalLen = 1 + len0 + len1 + len2 + len3;

    if (src_len < totalLen) {
        return;
    }

    // Decode value0 (little-endian)
    int64_t offset = 1;
    unsigned int v0 = 0;
    for (int64_t j = 0; j < len0; j++) {
        v0 |= ((unsigned int)src[offset + j]) << (8 * j);
    }
    values[0] = v0;
    offset += len0;

    // Decode value1
    unsigned int v1 = 0;
    for (int64_t j = 0; j < len1; j++) {
        v1 |= ((unsigned int)src[offset + j]) << (8 * j);
    }
    values[1] = v1;
    offset += len1;

    // Decode value2
    unsigned int v2 = 0;
    for (int64_t j = 0; j < len2; j++) {
        v2 |= ((unsigned int)src[offset + j]) << (8 * j);
    }
    values[2] = v2;
    offset += len2;

    // Decode value3
    unsigned int v3 = 0;
    for (int64_t j = 0; j < len3; j++) {
        v3 |= ((unsigned int)src[offset + j]) << (8 * j);
    }
    values[3] = v3;

    *consumed = totalLen;
}

// decode_group_varint64: Decode 4 uint64 values from group varint format
// Uses 2-byte control (12 bits = 4 * 3 bits) for 1-8 bytes per value.
//
// Control bits (3 bits per value):
//   - Bits 0-2:  length of value0 minus 1 (0-7 = 1-8 bytes)
//   - Bits 3-5:  length of value1 minus 1
//   - Bits 6-8:  length of value2 minus 1
//   - Bits 9-11: length of value3 minus 1
//
// Returns bytes consumed in *consumed (0 if error)
void decode_group_varint64(
    unsigned char *src, int64_t src_len,
    unsigned long long *values,
    int64_t *consumed
) {
    *consumed = 0;

    if (src_len < 2) {
        return;
    }

    // Read 12-bit control from 2 bytes (little-endian)
    unsigned int control = (unsigned int)src[0] | ((unsigned int)src[1] << 8);

    // Extract lengths (3 bits each, value is length-1)
    int64_t len0 = ((control >> 0) & 0x7) + 1;
    int64_t len1 = ((control >> 3) & 0x7) + 1;
    int64_t len2 = ((control >> 6) & 0x7) + 1;
    int64_t len3 = ((control >> 9) & 0x7) + 1;

    int64_t totalLen = 2 + len0 + len1 + len2 + len3;

    if (src_len < totalLen) {
        return;
    }

    // Decode value0 (little-endian)
    int64_t offset = 2;
    unsigned long long v0 = 0;
    for (int64_t j = 0; j < len0; j++) {
        v0 |= ((unsigned long long)src[offset + j]) << (8 * j);
    }
    values[0] = v0;
    offset += len0;

    // Decode value1
    unsigned long long v1 = 0;
    for (int64_t j = 0; j < len1; j++) {
        v1 |= ((unsigned long long)src[offset + j]) << (8 * j);
    }
    values[1] = v1;
    offset += len1;

    // Decode value2
    unsigned long long v2 = 0;
    for (int64_t j = 0; j < len2; j++) {
        v2 |= ((unsigned long long)src[offset + j]) << (8 * j);
    }
    values[2] = v2;
    offset += len2;

    // Decode value3
    unsigned long long v3 = 0;
    for (int64_t j = 0; j < len3; j++) {
        v3 |= ((unsigned long long)src[offset + j]) << (8 * j);
    }
    values[3] = v3;

    *consumed = totalLen;
}

// ============================================================================
// Single Varint Decoding (optimized scalar)
// ============================================================================

// decode_uvarint64: Decode a single unsigned varint
// Returns value in *value, bytes consumed in *consumed (0 if incomplete/error)
void decode_uvarint64(
    unsigned char *src, int64_t src_len,
    unsigned long long *value, int64_t *consumed
) {
    *value = 0;
    *consumed = 0;

    if (src_len <= 0) {
        return;
    }

    unsigned long long val = 0;
    int64_t shift = 0;

    for (int64_t i = 0; i < src_len; i++) {
        unsigned char b = src[i];

        // Check for overflow (max 10 bytes for uint64)
        if (i >= 10) {
            return;
        }

        // Check for overflow on 10th byte
        if (i == 9) {
            if (b > 1) {
                return;
            }
        }

        val |= ((unsigned long long)(b & 0x7f)) << shift;
        shift += 7;

        if (b < 0x80) {
            // Final byte
            *value = val;
            *consumed = i + 1;
            return;
        }
    }

    // Incomplete varint
}

// decode_2uvarint64: Decode exactly 2 unsigned varints (freq/norm pair)
// Returns values in *v1 and *v2, bytes consumed in *consumed (0 if error)
void decode_2uvarint64(
    unsigned char *src, int64_t src_len,
    unsigned long long *v1, unsigned long long *v2, int64_t *consumed
) {
    *v1 = 0;
    *v2 = 0;
    *consumed = 0;

    if (src_len <= 0) {
        return;
    }

    // Decode first varint
    unsigned long long val1 = 0;
    int64_t shift1 = 0;
    int64_t pos = 0;

    for (;;) {
        if (pos >= src_len) {
            return;
        }
        if (pos >= 10) {
            return;
        }

        unsigned char b = src[pos];
        pos++;

        if (pos == 10) {
            if (b > 1) {
                return;
            }
        }

        val1 |= ((unsigned long long)(b & 0x7f)) << shift1;
        shift1 += 7;

        if (b < 0x80) {
            break;
        }
    }

    // Decode second varint
    unsigned long long val2 = 0;
    int64_t shift2 = 0;
    int64_t start2 = pos;

    for (;;) {
        if (pos >= src_len) {
            return;
        }
        if (pos - start2 >= 10) {
            return;
        }

        unsigned char b = src[pos];
        pos++;

        if (pos - start2 == 10) {
            if (b > 1) {
                return;
            }
        }

        val2 |= ((unsigned long long)(b & 0x7f)) << shift2;
        shift2 += 7;

        if (b < 0x80) {
            break;
        }
    }

    *v1 = val1;
    *v2 = val2;
    *consumed = pos;
}

// decode_5uvarint64: Decode exactly 5 unsigned varints (location fields)
// Returns values in values array (5 elements), bytes consumed in *consumed
void decode_5uvarint64(
    unsigned char *src, int64_t src_len,
    unsigned long long *values, int64_t *consumed
) {
    values[0] = 0;
    values[1] = 0;
    values[2] = 0;
    values[3] = 0;
    values[4] = 0;
    *consumed = 0;

    if (src_len <= 0) {
        return;
    }

    int64_t pos = 0;

    // Decode 5 varints
    for (int64_t vi = 0; vi < 5; vi++) {
        unsigned long long val = 0;
        int64_t shift = 0;
        int64_t startPos = pos;

        for (;;) {
            if (pos >= src_len) {
                // Reset all on failure
                values[0] = 0;
                values[1] = 0;
                values[2] = 0;
                values[3] = 0;
                values[4] = 0;
                *consumed = 0;
                return;
            }
            if (pos - startPos >= 10) {
                values[0] = 0;
                values[1] = 0;
                values[2] = 0;
                values[3] = 0;
                values[4] = 0;
                *consumed = 0;
                return;
            }

            unsigned char b = src[pos];
            pos++;

            if (pos - startPos == 10) {
                if (b > 1) {
                    values[0] = 0;
                    values[1] = 0;
                    values[2] = 0;
                    values[3] = 0;
                    values[4] = 0;
                    *consumed = 0;
                    return;
                }
            }

            val |= ((unsigned long long)(b & 0x7f)) << shift;
            shift += 7;

            if (b < 0x80) {
                values[vi] = val;
                break;
            }
        }
    }

    *consumed = pos;
}
