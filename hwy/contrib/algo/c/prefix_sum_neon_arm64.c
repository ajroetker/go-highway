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

// NEON-optimized in-place prefix sum and delta decode algorithms for ARM64
// These process entire slices with minimal overhead by keeping carry in registers
// and avoiding per-vector function call overhead.

#include <arm_neon.h>

// ============================================================================
// In-place Prefix Sum Operations
// ============================================================================

// prefix_sum_inplace_f32: Compute prefix sum of float32 slice in-place
void prefix_sum_inplace_f32(float *data, int64_t n) {
    float32x4_t zero = vdupq_n_f32(0);
    float32x4_t carry = zero;
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t s1 = vextq_f32(zero, v, 3);
        v = vaddq_f32(v, s1);
        float32x4_t s2 = vextq_f32(zero, v, 2);
        v = vaddq_f32(v, s2);
        v = vaddq_f32(v, carry);
        vst1q_f32(data + i, v);
        carry = vdupq_laneq_f32(v, 3);
    }

    float c = vgetq_lane_f32(carry, 0);
    for (; i < n; i++) {
        c += data[i];
        data[i] = c;
    }
}

// prefix_sum_inplace_f64: Compute prefix sum of float64 slice in-place
void prefix_sum_inplace_f64(double *data, int64_t n) {
    float64x2_t zero = vdupq_n_f64(0);
    float64x2_t carry = zero;
    int64_t i = 0;

    for (; i + 2 <= n; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        float64x2_t s1 = vextq_f64(zero, v, 1);
        v = vaddq_f64(v, s1);
        v = vaddq_f64(v, carry);
        vst1q_f64(data + i, v);
        carry = vdupq_laneq_f64(v, 1);
    }

    double c = vgetq_lane_f64(carry, 0);
    for (; i < n; i++) {
        c += data[i];
        data[i] = c;
    }
}

// prefix_sum_inplace_i32: Compute prefix sum of int32 slice in-place
void prefix_sum_inplace_i32(int *data, int64_t n) {
    int32x4_t zero = vdupq_n_s32(0);
    int32x4_t carry = zero;
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        int32x4_t v = vld1q_s32(data + i);
        int32x4_t s1 = vextq_s32(zero, v, 3);
        v = vaddq_s32(v, s1);
        int32x4_t s2 = vextq_s32(zero, v, 2);
        v = vaddq_s32(v, s2);
        v = vaddq_s32(v, carry);
        vst1q_s32(data + i, v);
        carry = vdupq_laneq_s32(v, 3);
    }

    int c = vgetq_lane_s32(carry, 0);
    for (; i < n; i++) {
        c += data[i];
        data[i] = c;
    }
}

// prefix_sum_inplace_i64: Compute prefix sum of int64 slice in-place
// For 64-bit types with only 2 NEON lanes, simple scalar is competitive.
// This version uses scalar to avoid vector↔scalar transfer overhead.
void prefix_sum_inplace_i64(long long *data, int64_t n) {
    long long acc = 0;
    for (int64_t i = 0; i < n; i++) {
        acc += data[i];
        data[i] = acc;
    }
}

// prefix_sum_inplace_u32: Compute prefix sum of uint32 slice in-place
void prefix_sum_inplace_u32(unsigned int *data, int64_t n) {
    uint32x4_t zero = vdupq_n_u32(0);
    uint32x4_t carry = zero;
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        uint32x4_t v = vld1q_u32(data + i);
        uint32x4_t s1 = vextq_u32(zero, v, 3);
        v = vaddq_u32(v, s1);
        uint32x4_t s2 = vextq_u32(zero, v, 2);
        v = vaddq_u32(v, s2);
        v = vaddq_u32(v, carry);
        vst1q_u32(data + i, v);
        carry = vdupq_laneq_u32(v, 3);
    }

    unsigned int c = vgetq_lane_u32(carry, 0);
    for (; i < n; i++) {
        c += data[i];
        data[i] = c;
    }
}

// prefix_sum_inplace_u64: Compute prefix sum of uint64 slice in-place
// For 64-bit types with only 2 NEON lanes, simple scalar is competitive.
void prefix_sum_inplace_u64(unsigned long long *data, int64_t n) {
    unsigned long long acc = 0;
    for (int64_t i = 0; i < n; i++) {
        acc += data[i];
        data[i] = acc;
    }
}

// ============================================================================
// In-place Delta Decode Operations (Prefix Sum with initial base)
// ============================================================================

// delta_decode_inplace_i32: Decode delta-encoded int32 slice in-place
void delta_decode_inplace_i32(int *data, int64_t n, int32_t base) {
    int32x4_t zero = vdupq_n_s32(0);
    int32x4_t carry = vdupq_n_s32(base);
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        int32x4_t v = vld1q_s32(data + i);
        int32x4_t s1 = vextq_s32(zero, v, 3);
        v = vaddq_s32(v, s1);
        int32x4_t s2 = vextq_s32(zero, v, 2);
        v = vaddq_s32(v, s2);
        v = vaddq_s32(v, carry);
        vst1q_s32(data + i, v);
        carry = vdupq_laneq_s32(v, 3);
    }

    int c = vgetq_lane_s32(carry, 0);
    for (; i < n; i++) {
        c += data[i];
        data[i] = c;
    }
}

// delta_decode_inplace_i64: Decode delta-encoded int64 slice in-place
// For 64-bit types with only 2 NEON lanes, simple scalar is competitive.
void delta_decode_inplace_i64(long long *data, int64_t n, long long base) {
    long long acc = base;
    for (int64_t i = 0; i < n; i++) {
        acc += data[i];
        data[i] = acc;
    }
}

// delta_decode_inplace_u32: Decode delta-encoded uint32 slice in-place
void delta_decode_inplace_u32(unsigned int *data, int64_t n, int32_t base) {
    uint32x4_t zero = vdupq_n_u32(0);
    uint32x4_t carry = vdupq_n_u32((unsigned int)base);
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        uint32x4_t v = vld1q_u32(data + i);
        uint32x4_t s1 = vextq_u32(zero, v, 3);
        v = vaddq_u32(v, s1);
        uint32x4_t s2 = vextq_u32(zero, v, 2);
        v = vaddq_u32(v, s2);
        v = vaddq_u32(v, carry);
        vst1q_u32(data + i, v);
        carry = vdupq_laneq_u32(v, 3);
    }

    unsigned int c = vgetq_lane_u32(carry, 0);
    for (; i < n; i++) {
        c += data[i];
        data[i] = c;
    }
}

// delta_decode_inplace_u64: Decode delta-encoded uint64 slice in-place
// For 64-bit types with only 2 NEON lanes, simple scalar is competitive.
// Note: base passed as long long due to GoAT limitations, cast in Go wrapper
void delta_decode_inplace_u64(unsigned long long *data, int64_t n, long long base) {
    unsigned long long acc = (unsigned long long)base;
    for (int64_t i = 0; i < n; i++) {
        acc += data[i];
        data[i] = acc;
    }
}
