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

// NEON Encode/Decode Operations for go-highway
// Compile with: -march=armv8-a -O3
//
// Implements efficient float32/float64 to byte slice encoding using NEON SIMD.
// On little-endian systems (ARM64), this is essentially a SIMD memcpy with
// type reinterpretation.
//
// GoAT generates Go assembly from this file via:
//   go tool goat encode_neon_arm64.c -O3 --target arm64

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// =============================================================================
// encode_f32_neon: Encode float32 slice to bytes
// =============================================================================
// dst must have at least len*4 bytes capacity
void encode_f32_neon(float* src, unsigned char* dst, long *plen) {
    long len = *plen;
    if (len <= 0) {
        return;
    }

    long i = 0;

    // Process 4 floats (16 bytes) at a time
    for (; i + 4 <= len; i += 4) {
        // Load 4 float32 values into a NEON register
        float32x4_t float_vec = vld1q_f32(src + i);

        // Reinterpret the bit pattern of the floats as bytes and store
        vst1q_u8(dst + (i * 4), vreinterpretq_u8_f32(float_vec));
    }

    // Handle remaining elements (scalar)
    for (; i < len; i++) {
        unsigned int bits = *(unsigned int*)&src[i];
        dst[i*4] = bits & 0xFF;
        dst[i*4+1] = (bits >> 8) & 0xFF;
        dst[i*4+2] = (bits >> 16) & 0xFF;
        dst[i*4+3] = (bits >> 24) & 0xFF;
    }
}

// =============================================================================
// decode_f32_neon: Decode bytes to float32 slice
// =============================================================================
// src must have at least len*4 bytes
void decode_f32_neon(unsigned char* src, float* dst, long *plen) {
    long len = *plen;
    if (len <= 0) {
        return;
    }

    long i = 0;

    // Process 4 floats (16 bytes) at a time
    for (; i + 4 <= len; i += 4) {
        // Load 16 bytes into a NEON register
        uint8x16_t byte_vec = vld1q_u8(src + (i * 4));

        // Reinterpret the bit pattern as float32 and store
        vst1q_f32(dst + i, vreinterpretq_f32_u8(byte_vec));
    }

    // Handle remainder of 2 elements using SIMD
    if (i + 2 <= len) {
        // Load 8 bytes (2 floats worth) into a NEON register
        uint8x8_t byte_vec = vld1_u8(src + (i * 4));
        // Reinterpret as float32 values and store
        vst1_f32(dst + i, vreinterpret_f32_u8(byte_vec));
        i += 2;
    }

    // Handle final single element if needed
    if (i < len) {
        // Load 4 bytes into a 32-bit integer using NEON
        unsigned int bits = vget_lane_u32(vreinterpret_u32_u8(vld1_u8(src + (i * 4))), 0);
        // Store as float using NEON
        vst1_lane_f32(dst + i, vreinterpret_f32_u32(vdup_n_u32(bits)), 0);
    }
}

// =============================================================================
// encode_f64_neon: Encode float64 slice to bytes
// =============================================================================
// dst must have at least len*8 bytes capacity
void encode_f64_neon(double* src, unsigned char* dst, long *plen) {
    long len = *plen;
    if (len <= 0) {
        return;
    }

    long i = 0;

    // Process 2 doubles (16 bytes) at a time
    for (; i + 2 <= len; i += 2) {
        // Load 2 float64 values into a NEON register
        float64x2_t float_vec = vld1q_f64(src + i);

        // Reinterpret the bit pattern as bytes and store
        vst1q_u8(dst + (i * 8), vreinterpretq_u8_f64(float_vec));
    }

    // Handle remaining single element (scalar)
    if (i < len) {
        unsigned long bits = *(unsigned long*)&src[i];
        dst[i*8] = bits & 0xFF;
        dst[i*8+1] = (bits >> 8) & 0xFF;
        dst[i*8+2] = (bits >> 16) & 0xFF;
        dst[i*8+3] = (bits >> 24) & 0xFF;
        dst[i*8+4] = (bits >> 32) & 0xFF;
        dst[i*8+5] = (bits >> 40) & 0xFF;
        dst[i*8+6] = (bits >> 48) & 0xFF;
        dst[i*8+7] = (bits >> 56) & 0xFF;
    }
}

// =============================================================================
// decode_f64_neon: Decode bytes to float64 slice
// =============================================================================
// src must have at least len*8 bytes
void decode_f64_neon(unsigned char* src, double* dst, long *plen) {
    long len = *plen;
    if (len <= 0) {
        return;
    }

    long i = 0;

    // Process 2 doubles (16 bytes) at a time
    for (; i + 2 <= len; i += 2) {
        // Load 16 bytes into a NEON register
        uint8x16_t byte_vec = vld1q_u8(src + (i * 8));

        // Reinterpret the bit pattern as float64 and store
        vst1q_f64(dst + i, vreinterpretq_f64_u8(byte_vec));
    }

    // Handle remaining single element
    if (i < len) {
        // Load 8 bytes
        uint8x8_t byte_vec = vld1_u8(src + (i * 8));
        // Reinterpret as float64 and store
        vst1_lane_f64(dst + i, vreinterpret_f64_u8(byte_vec), 0);
    }
}
