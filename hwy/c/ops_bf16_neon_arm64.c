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

// BFloat16 SIMD operations for ARM64 with BF16 extension (ARMv8.6-A+)
// Used with GOAT to generate Go assembly
// Compile with: -march=armv8.6-a+bf16
#include <arm_neon.h>

// ============================================================================
// BFloat16 Conversions
// ============================================================================
// BFloat16 is simpler than Float16: it's just float32 with lower 16 bits truncated.
// No special hardware conversion instructions needed - just bit manipulation.

// Promote bfloat16 to float32: zero-extend and shift left 16 bits
// BF16 format: 1-bit sign, 8-bit exponent, 7-bit mantissa
// F32 format:  1-bit sign, 8-bit exponent, 23-bit mantissa
// Conversion: just shift BF16 bits to upper 16 bits of F32
void promote_bf16_to_f32_neon(unsigned short *a, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 bfloat16 -> 32 float32 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        // Load 8 BF16 values as uint16
        uint16x8_t bf0 = vld1q_u16(a + i);
        uint16x8_t bf1 = vld1q_u16(a + i + 8);
        uint16x8_t bf2 = vld1q_u16(a + i + 16);
        uint16x8_t bf3 = vld1q_u16(a + i + 24);

        // Shift left by 16 to get float32 bit pattern
        // vshll_n_u16 widens u16->u32 and shifts left
        uint32x4_t lo0 = vshll_n_u16(vget_low_u16(bf0), 16);
        uint32x4_t hi0 = vshll_n_u16(vget_high_u16(bf0), 16);
        uint32x4_t lo1 = vshll_n_u16(vget_low_u16(bf1), 16);
        uint32x4_t hi1 = vshll_n_u16(vget_high_u16(bf1), 16);
        uint32x4_t lo2 = vshll_n_u16(vget_low_u16(bf2), 16);
        uint32x4_t hi2 = vshll_n_u16(vget_high_u16(bf2), 16);
        uint32x4_t lo3 = vshll_n_u16(vget_low_u16(bf3), 16);
        uint32x4_t hi3 = vshll_n_u16(vget_high_u16(bf3), 16);

        // Reinterpret as float32 and store
        vst1q_f32(result + i, vreinterpretq_f32_u32(lo0));
        vst1q_f32(result + i + 4, vreinterpretq_f32_u32(hi0));
        vst1q_f32(result + i + 8, vreinterpretq_f32_u32(lo1));
        vst1q_f32(result + i + 12, vreinterpretq_f32_u32(hi1));
        vst1q_f32(result + i + 16, vreinterpretq_f32_u32(lo2));
        vst1q_f32(result + i + 20, vreinterpretq_f32_u32(hi2));
        vst1q_f32(result + i + 24, vreinterpretq_f32_u32(lo3));
        vst1q_f32(result + i + 28, vreinterpretq_f32_u32(hi3));
    }

    // Process 8 bfloat16 -> 8 float32 at a time
    for (; i + 7 < n; i += 8) {
        uint16x8_t bf = vld1q_u16(a + i);
        uint32x4_t lo = vshll_n_u16(vget_low_u16(bf), 16);
        uint32x4_t hi = vshll_n_u16(vget_high_u16(bf), 16);
        vst1q_f32(result + i, vreinterpretq_f32_u32(lo));
        vst1q_f32(result + i + 4, vreinterpretq_f32_u32(hi));
    }

    // Process 4 at a time
    for (; i + 3 < n; i += 4) {
        uint16x4_t bf = vld1_u16(a + i);
        uint32x4_t f32bits = vshll_n_u16(bf, 16);
        vst1q_f32(result + i, vreinterpretq_f32_u32(f32bits));
    }

    // Scalar remainder using NEON for single element
    for (; i < n; i++) {
        uint16x4_t bf = vld1_dup_u16(a + i);
        uint32x4_t f32bits = vshll_n_u16(bf, 16);
        vst1q_lane_f32(result + i, vreinterpretq_f32_u32(f32bits), 0);
    }
}

// Demote float32 to bfloat16: round-to-nearest-even and shift right 16 bits
// This implements proper rounding (not just truncation) for better accuracy
void demote_f32_to_bf16_neon(float *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Rounding constant: 0x7FFF (for round-to-nearest-even with bias)
    // Add this before truncating to implement rounding
    uint32x4_t round_const = vdupq_n_u32(0x7FFF);
    // For round-to-nearest-even, we also need to check the LSB of result
    uint32x4_t one = vdupq_n_u32(1);

    // Process 32 float32 -> 32 bfloat16 at a time
    for (; i + 31 < n; i += 32) {
        float32x4_t f0 = vld1q_f32(a + i);
        float32x4_t f1 = vld1q_f32(a + i + 4);
        float32x4_t f2 = vld1q_f32(a + i + 8);
        float32x4_t f3 = vld1q_f32(a + i + 12);
        float32x4_t f4 = vld1q_f32(a + i + 16);
        float32x4_t f5 = vld1q_f32(a + i + 20);
        float32x4_t f6 = vld1q_f32(a + i + 24);
        float32x4_t f7 = vld1q_f32(a + i + 28);

        // Reinterpret as uint32
        uint32x4_t u0 = vreinterpretq_u32_f32(f0);
        uint32x4_t u1 = vreinterpretq_u32_f32(f1);
        uint32x4_t u2 = vreinterpretq_u32_f32(f2);
        uint32x4_t u3 = vreinterpretq_u32_f32(f3);
        uint32x4_t u4 = vreinterpretq_u32_f32(f4);
        uint32x4_t u5 = vreinterpretq_u32_f32(f5);
        uint32x4_t u6 = vreinterpretq_u32_f32(f6);
        uint32x4_t u7 = vreinterpretq_u32_f32(f7);

        // Round-to-nearest-even: add rounding bias, then check for tie-breaking
        // bias = 0x7FFF + ((u >> 16) & 1)  to break ties to even
        uint32x4_t bias0 = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u0, 16), one));
        uint32x4_t bias1 = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u1, 16), one));
        uint32x4_t bias2 = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u2, 16), one));
        uint32x4_t bias3 = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u3, 16), one));
        uint32x4_t bias4 = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u4, 16), one));
        uint32x4_t bias5 = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u5, 16), one));
        uint32x4_t bias6 = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u6, 16), one));
        uint32x4_t bias7 = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u7, 16), one));

        // Add bias and shift right 16 to get BF16
        uint32x4_t r0 = vshrq_n_u32(vaddq_u32(u0, bias0), 16);
        uint32x4_t r1 = vshrq_n_u32(vaddq_u32(u1, bias1), 16);
        uint32x4_t r2 = vshrq_n_u32(vaddq_u32(u2, bias2), 16);
        uint32x4_t r3 = vshrq_n_u32(vaddq_u32(u3, bias3), 16);
        uint32x4_t r4 = vshrq_n_u32(vaddq_u32(u4, bias4), 16);
        uint32x4_t r5 = vshrq_n_u32(vaddq_u32(u5, bias5), 16);
        uint32x4_t r6 = vshrq_n_u32(vaddq_u32(u6, bias6), 16);
        uint32x4_t r7 = vshrq_n_u32(vaddq_u32(u7, bias7), 16);

        // Narrow from u32 to u16 and store
        // vmovn_u32 takes lower 16 bits of each u32 lane
        uint16x4_t h0 = vmovn_u32(r0);
        uint16x4_t h1 = vmovn_u32(r1);
        uint16x4_t h2 = vmovn_u32(r2);
        uint16x4_t h3 = vmovn_u32(r3);
        uint16x4_t h4 = vmovn_u32(r4);
        uint16x4_t h5 = vmovn_u32(r5);
        uint16x4_t h6 = vmovn_u32(r6);
        uint16x4_t h7 = vmovn_u32(r7);

        vst1q_u16(result + i, vcombine_u16(h0, h1));
        vst1q_u16(result + i + 8, vcombine_u16(h2, h3));
        vst1q_u16(result + i + 16, vcombine_u16(h4, h5));
        vst1q_u16(result + i + 24, vcombine_u16(h6, h7));
    }

    // Process 8 float32 -> 8 bfloat16 at a time
    for (; i + 7 < n; i += 8) {
        float32x4_t f_lo = vld1q_f32(a + i);
        float32x4_t f_hi = vld1q_f32(a + i + 4);

        uint32x4_t u_lo = vreinterpretq_u32_f32(f_lo);
        uint32x4_t u_hi = vreinterpretq_u32_f32(f_hi);

        uint32x4_t bias_lo = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_lo, 16), one));
        uint32x4_t bias_hi = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_hi, 16), one));

        uint32x4_t r_lo = vshrq_n_u32(vaddq_u32(u_lo, bias_lo), 16);
        uint32x4_t r_hi = vshrq_n_u32(vaddq_u32(u_hi, bias_hi), 16);

        uint16x4_t h_lo = vmovn_u32(r_lo);
        uint16x4_t h_hi = vmovn_u32(r_hi);

        vst1q_u16(result + i, vcombine_u16(h_lo, h_hi));
    }

    // Process 4 at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t f = vld1q_f32(a + i);
        uint32x4_t u = vreinterpretq_u32_f32(f);
        uint32x4_t bias = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u, 16), one));
        uint32x4_t r = vshrq_n_u32(vaddq_u32(u, bias), 16);
        uint16x4_t h = vmovn_u32(r);
        vst1_u16(result + i, h);
    }

    // Scalar remainder using NEON for single element
    for (; i < n; i++) {
        float32x4_t f = vld1q_dup_f32(a + i);
        uint32x4_t u = vreinterpretq_u32_f32(f);
        uint32x4_t bias = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u, 16), one));
        uint32x4_t r = vshrq_n_u32(vaddq_u32(u, bias), 16);
        uint16x4_t h = vmovn_u32(r);
        vst1_lane_u16(result + i, h, 0);
    }
}

// ============================================================================
// BFloat16 Dot Product (ARMv8.6-A BFDOT instruction)
// ============================================================================
// The BFDOT instruction computes dot product of BF16 pairs, accumulating to F32.
// vbfdotq_f32(acc, a, b) computes: acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]
// Each F32 accumulator lane receives the dot product of 2 BF16 pairs.

// Dot product: acc += sum(a[i] * b[i]) for i in 0..len
// Uses BFDOT for main loop, promotes to F32 for remainder
void dot_bf16_neon(unsigned short *a, unsigned short *b, float *acc, long *len) {
    long n = *len;
    long i = 0;

    // Initialize accumulator vectors
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    // Process 32 BF16 pairs at a time using BFDOT
    // Each vbfdotq_f32 processes 8 BF16 values (4 pairs) -> 4 F32 partial sums
    for (; i + 31 < n; i += 32) {
        // Load as bfloat16x8_t (reinterpret from uint16)
        bfloat16x8_t av0 = vld1q_bf16((bfloat16_t*)(a + i));
        bfloat16x8_t av1 = vld1q_bf16((bfloat16_t*)(a + i + 8));
        bfloat16x8_t av2 = vld1q_bf16((bfloat16_t*)(a + i + 16));
        bfloat16x8_t av3 = vld1q_bf16((bfloat16_t*)(a + i + 24));

        bfloat16x8_t bv0 = vld1q_bf16((bfloat16_t*)(b + i));
        bfloat16x8_t bv1 = vld1q_bf16((bfloat16_t*)(b + i + 8));
        bfloat16x8_t bv2 = vld1q_bf16((bfloat16_t*)(b + i + 16));
        bfloat16x8_t bv3 = vld1q_bf16((bfloat16_t*)(b + i + 24));

        // BFDOT: each call processes 8 BF16 -> 4 F32 partial sums
        sum0 = vbfdotq_f32(sum0, av0, bv0);
        sum1 = vbfdotq_f32(sum1, av1, bv1);
        sum2 = vbfdotq_f32(sum2, av2, bv2);
        sum3 = vbfdotq_f32(sum3, av3, bv3);
    }

    // Process 8 BF16 pairs at a time
    for (; i + 7 < n; i += 8) {
        bfloat16x8_t av = vld1q_bf16((bfloat16_t*)(a + i));
        bfloat16x8_t bv = vld1q_bf16((bfloat16_t*)(b + i));
        sum0 = vbfdotq_f32(sum0, av, bv);
    }

    // Horizontal sum of vector accumulators
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);

    // Reduce 4 lanes to scalar
    float32x2_t sum_lo = vget_low_f32(sum0);
    float32x2_t sum_hi = vget_high_f32(sum0);
    float32x2_t sum_pair = vadd_f32(sum_lo, sum_hi);
    float total = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    // Scalar remainder: promote to F32, multiply, accumulate
    for (; i < n; i++) {
        // Load BF16, promote to F32 by shifting left 16
        uint16x4_t a_bf = vld1_dup_u16(a + i);
        uint16x4_t b_bf = vld1_dup_u16(b + i);
        uint32x4_t a_u32 = vshll_n_u16(a_bf, 16);
        uint32x4_t b_u32 = vshll_n_u16(b_bf, 16);
        float32x4_t a_f32 = vreinterpretq_f32_u32(a_u32);
        float32x4_t b_f32 = vreinterpretq_f32_u32(b_u32);
        float32x4_t prod = vmulq_f32(a_f32, b_f32);
        total = total + vgetq_lane_f32(prod, 0);
    }

    // Add to accumulator
    *acc = *acc + total;
}

// ============================================================================
// BFloat16 Matrix-Matrix Multiply (BFMMLA instruction - ARMv8.6-A)
// ============================================================================
// BFMMLA performs a 2x4 by 4x2 matrix multiply of BF16 values, accumulating
// into a 2x2 F32 result. This is highly efficient for tiled GEMM operations.
//
// vbfmmlaq_f32(acc, a, b) where:
//   acc: float32x4_t (2x2 matrix stored as [c00, c01, c10, c11])
//   a: bfloat16x8_t (2x4 matrix, row-major: [a00,a01,a02,a03, a10,a11,a12,a13])
//   b: bfloat16x8_t (4x2 matrix, col-major: [b00,b10,b20,b30, b01,b11,b21,b31])
//
// Result: C[i,j] += sum_k(A[i,k] * B[k,j]) for 2x2 output

// Tiled matrix multiply: C += A * B
// A is MxK (row-major), B is KxN (row-major), C is MxN (row-major)
// For simplicity, this processes 2x2 output tiles using BFMMLA
void matmul_bf16_neon(unsigned short *a, unsigned short *b, float *c,
                      long *m_ptr, long *n_ptr, long *k_ptr,
                      long *lda_ptr, long *ldb_ptr, long *ldc_ptr) {
    long M = *m_ptr;
    long N = *n_ptr;
    long K = *k_ptr;
    long lda = *lda_ptr;
    long ldb = *ldb_ptr;
    long ldc = *ldc_ptr;

    // Process 2x2 output tiles
    long i = 0;
    for (; i + 1 < M; i += 2) {
        long j = 0;
        for (; j + 1 < N; j += 2) {
            // Initialize 2x2 accumulator
            float32x4_t acc = vdupq_n_f32(0.0f);

            // Process K dimension in chunks of 4 for BFMMLA
            long k = 0;
            for (; k + 3 < K; k += 4) {
                // Load 2x4 tile from A (rows i and i+1, columns k to k+3)
                bfloat16x4_t a_row0 = vld1_bf16((bfloat16_t*)(a + i * lda + k));
                bfloat16x4_t a_row1 = vld1_bf16((bfloat16_t*)(a + (i + 1) * lda + k));
                bfloat16x8_t a_tile = vcombine_bf16(a_row0, a_row1);

                // Load 4x2 tile from B (rows k to k+3, columns j and j+1)
                // Need column-major for BFMMLA: [b[k,j], b[k+1,j], b[k+2,j], b[k+3,j],
                //                                b[k,j+1], b[k+1,j+1], b[k+2,j+1], b[k+3,j+1]]
                bfloat16_t b_col0_0 = *((bfloat16_t*)(b + k * ldb + j));
                bfloat16_t b_col0_1 = *((bfloat16_t*)(b + (k + 1) * ldb + j));
                bfloat16_t b_col0_2 = *((bfloat16_t*)(b + (k + 2) * ldb + j));
                bfloat16_t b_col0_3 = *((bfloat16_t*)(b + (k + 3) * ldb + j));
                bfloat16_t b_col1_0 = *((bfloat16_t*)(b + k * ldb + j + 1));
                bfloat16_t b_col1_1 = *((bfloat16_t*)(b + (k + 1) * ldb + j + 1));
                bfloat16_t b_col1_2 = *((bfloat16_t*)(b + (k + 2) * ldb + j + 1));
                bfloat16_t b_col1_3 = *((bfloat16_t*)(b + (k + 3) * ldb + j + 1));

                // Create column-major B tile - use scalar stores to build vector
                bfloat16_t b_arr[8];
                b_arr[0] = b_col0_0;
                b_arr[1] = b_col0_1;
                b_arr[2] = b_col0_2;
                b_arr[3] = b_col0_3;
                b_arr[4] = b_col1_0;
                b_arr[5] = b_col1_1;
                b_arr[6] = b_col1_2;
                b_arr[7] = b_col1_3;
                bfloat16x8_t b_tile = vld1q_bf16(b_arr);

                // BFMMLA: accumulate 2x2 result
                acc = vbfmmlaq_f32(acc, a_tile, b_tile);
            }

            // Store 2x2 result
            // acc layout: [c00, c01, c10, c11]
            c[i * ldc + j] = c[i * ldc + j] + vgetq_lane_f32(acc, 0);
            c[i * ldc + j + 1] = c[i * ldc + j + 1] + vgetq_lane_f32(acc, 1);
            c[(i + 1) * ldc + j] = c[(i + 1) * ldc + j] + vgetq_lane_f32(acc, 2);
            c[(i + 1) * ldc + j + 1] = c[(i + 1) * ldc + j + 1] + vgetq_lane_f32(acc, 3);

            // Handle K remainder with scalar
            for (; k < K; k++) {
                uint16x4_t a0_bf = vld1_dup_u16(a + i * lda + k);
                uint16x4_t a1_bf = vld1_dup_u16(a + (i + 1) * lda + k);
                uint16x4_t b0_bf = vld1_dup_u16(b + k * ldb + j);
                uint16x4_t b1_bf = vld1_dup_u16(b + k * ldb + j + 1);

                // Promote to F32
                float a0_f32 = vgetq_lane_f32(vreinterpretq_f32_u32(vshll_n_u16(a0_bf, 16)), 0);
                float a1_f32 = vgetq_lane_f32(vreinterpretq_f32_u32(vshll_n_u16(a1_bf, 16)), 0);
                float b0_f32 = vgetq_lane_f32(vreinterpretq_f32_u32(vshll_n_u16(b0_bf, 16)), 0);
                float b1_f32 = vgetq_lane_f32(vreinterpretq_f32_u32(vshll_n_u16(b1_bf, 16)), 0);

                c[i * ldc + j] = c[i * ldc + j] + a0_f32 * b0_f32;
                c[i * ldc + j + 1] = c[i * ldc + j + 1] + a0_f32 * b1_f32;
                c[(i + 1) * ldc + j] = c[(i + 1) * ldc + j] + a1_f32 * b0_f32;
                c[(i + 1) * ldc + j + 1] = c[(i + 1) * ldc + j + 1] + a1_f32 * b1_f32;
            }
        }

        // Handle odd N (single column remainder)
        if (j < N) {
            long k = 0;
            for (; k < K; k++) {
                uint16x4_t a0_bf = vld1_dup_u16(a + i * lda + k);
                uint16x4_t a1_bf = vld1_dup_u16(a + (i + 1) * lda + k);
                uint16x4_t b0_bf = vld1_dup_u16(b + k * ldb + j);

                float a0_f32 = vgetq_lane_f32(vreinterpretq_f32_u32(vshll_n_u16(a0_bf, 16)), 0);
                float a1_f32 = vgetq_lane_f32(vreinterpretq_f32_u32(vshll_n_u16(a1_bf, 16)), 0);
                float b0_f32 = vgetq_lane_f32(vreinterpretq_f32_u32(vshll_n_u16(b0_bf, 16)), 0);

                c[i * ldc + j] = c[i * ldc + j] + a0_f32 * b0_f32;
                c[(i + 1) * ldc + j] = c[(i + 1) * ldc + j] + a1_f32 * b0_f32;
            }
        }
    }

    // Handle odd M (single row remainder)
    if (i < M) {
        long j = 0;
        for (; j < N; j++) {
            long k = 0;
            for (; k < K; k++) {
                uint16x4_t a0_bf = vld1_dup_u16(a + i * lda + k);
                uint16x4_t b0_bf = vld1_dup_u16(b + k * ldb + j);

                float a0_f32 = vgetq_lane_f32(vreinterpretq_f32_u32(vshll_n_u16(a0_bf, 16)), 0);
                float b0_f32 = vgetq_lane_f32(vreinterpretq_f32_u32(vshll_n_u16(b0_bf, 16)), 0);

                c[i * ldc + j] = c[i * ldc + j] + a0_f32 * b0_f32;
            }
        }
    }
}

// ============================================================================
// BFloat16 Load Operations (for vec operations)
// ============================================================================

// Load4: Load 4 consecutive bfloat16x8 vectors (32 bfloat16 values = 64 bytes)
// Uses vld1q_bf16_x4 which loads 64 bytes in a single instruction
void load4_bf16x8(unsigned short *ptr,
                  bfloat16x8_t *out0, bfloat16x8_t *out1,
                  bfloat16x8_t *out2, bfloat16x8_t *out3) {
    bfloat16x8x4_t v = vld1q_bf16_x4((bfloat16_t*)ptr);
    *out0 = v.val[0];
    *out1 = v.val[1];
    *out2 = v.val[2];
    *out3 = v.val[3];
}

// Store4: Store 4 consecutive bfloat16x8 vectors (32 bfloat16 values = 64 bytes)
// Uses vst1q_bf16_x4 which stores 64 bytes in a single instruction
void store4_bf16x8(unsigned short *ptr,
                   bfloat16x8_t v0, bfloat16x8_t v1,
                   bfloat16x8_t v2, bfloat16x8_t v3) {
    bfloat16x8x4_t v;
    v.val[0] = v0;
    v.val[1] = v1;
    v.val[2] = v2;
    v.val[3] = v3;
    vst1q_bf16_x4((bfloat16_t*)ptr, v);
}

// ============================================================================
// BFloat16x8 Single-Vector Operations (register-resident)
// ============================================================================
// BFloat16 does NOT have native SIMD arithmetic. These operations use the
// promote-to-F32 -> compute -> demote-to-BF16 pattern with proper rounding.
// For optimal performance in ML workloads, prefer using BFDOT/BFMMLA with
// F32 accumulators instead of these general arithmetic operations.

// Broadcast a scalar bfloat16 to all 8 lanes
bfloat16x8_t broadcast_bf16x8(unsigned short *val) {
    return vld1q_dup_bf16((bfloat16_t*)val);
}

// Helper macros for promote/demote (inlined since GOAT doesn't support static inline)
// Promote BF16 to F32: shift left by 16 bits
// Demote F32 to BF16: round-to-nearest-even and shift right by 16 bits

bfloat16x8_t add_bf16x8(bfloat16x8_t a, bfloat16x8_t b) {
    // Promote to F32
    uint16x8_t ua = vreinterpretq_u16_bf16(a);
    uint16x8_t ub = vreinterpretq_u16_bf16(b);
    uint32x4_t a_lo_u32 = vshll_n_u16(vget_low_u16(ua), 16);
    uint32x4_t a_hi_u32 = vshll_n_u16(vget_high_u16(ua), 16);
    uint32x4_t b_lo_u32 = vshll_n_u16(vget_low_u16(ub), 16);
    uint32x4_t b_hi_u32 = vshll_n_u16(vget_high_u16(ub), 16);
    float32x4_t a_lo = vreinterpretq_f32_u32(a_lo_u32);
    float32x4_t a_hi = vreinterpretq_f32_u32(a_hi_u32);
    float32x4_t b_lo = vreinterpretq_f32_u32(b_lo_u32);
    float32x4_t b_hi = vreinterpretq_f32_u32(b_hi_u32);

    // Compute in F32
    float32x4_t r_lo = vaddq_f32(a_lo, b_lo);
    float32x4_t r_hi = vaddq_f32(a_hi, b_hi);

    // Demote to BF16 with round-to-nearest-even
    uint32x4_t u_lo = vreinterpretq_u32_f32(r_lo);
    uint32x4_t u_hi = vreinterpretq_u32_f32(r_hi);
    uint32x4_t round_const = vdupq_n_u32(0x7FFF);
    uint32x4_t one = vdupq_n_u32(1);
    uint32x4_t bias_lo = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_lo, 16), one));
    uint32x4_t bias_hi = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_hi, 16), one));
    uint32x4_t rr_lo = vshrq_n_u32(vaddq_u32(u_lo, bias_lo), 16);
    uint32x4_t rr_hi = vshrq_n_u32(vaddq_u32(u_hi, bias_hi), 16);
    uint16x4_t h_lo = vmovn_u32(rr_lo);
    uint16x4_t h_hi = vmovn_u32(rr_hi);
    return vreinterpretq_bf16_u16(vcombine_u16(h_lo, h_hi));
}

bfloat16x8_t sub_bf16x8(bfloat16x8_t a, bfloat16x8_t b) {
    uint16x8_t ua = vreinterpretq_u16_bf16(a);
    uint16x8_t ub = vreinterpretq_u16_bf16(b);
    uint32x4_t a_lo_u32 = vshll_n_u16(vget_low_u16(ua), 16);
    uint32x4_t a_hi_u32 = vshll_n_u16(vget_high_u16(ua), 16);
    uint32x4_t b_lo_u32 = vshll_n_u16(vget_low_u16(ub), 16);
    uint32x4_t b_hi_u32 = vshll_n_u16(vget_high_u16(ub), 16);
    float32x4_t a_lo = vreinterpretq_f32_u32(a_lo_u32);
    float32x4_t a_hi = vreinterpretq_f32_u32(a_hi_u32);
    float32x4_t b_lo = vreinterpretq_f32_u32(b_lo_u32);
    float32x4_t b_hi = vreinterpretq_f32_u32(b_hi_u32);

    float32x4_t r_lo = vsubq_f32(a_lo, b_lo);
    float32x4_t r_hi = vsubq_f32(a_hi, b_hi);

    uint32x4_t u_lo = vreinterpretq_u32_f32(r_lo);
    uint32x4_t u_hi = vreinterpretq_u32_f32(r_hi);
    uint32x4_t round_const = vdupq_n_u32(0x7FFF);
    uint32x4_t one = vdupq_n_u32(1);
    uint32x4_t bias_lo = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_lo, 16), one));
    uint32x4_t bias_hi = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_hi, 16), one));
    uint32x4_t rr_lo = vshrq_n_u32(vaddq_u32(u_lo, bias_lo), 16);
    uint32x4_t rr_hi = vshrq_n_u32(vaddq_u32(u_hi, bias_hi), 16);
    uint16x4_t h_lo = vmovn_u32(rr_lo);
    uint16x4_t h_hi = vmovn_u32(rr_hi);
    return vreinterpretq_bf16_u16(vcombine_u16(h_lo, h_hi));
}

bfloat16x8_t mul_bf16x8(bfloat16x8_t a, bfloat16x8_t b) {
    uint16x8_t ua = vreinterpretq_u16_bf16(a);
    uint16x8_t ub = vreinterpretq_u16_bf16(b);
    uint32x4_t a_lo_u32 = vshll_n_u16(vget_low_u16(ua), 16);
    uint32x4_t a_hi_u32 = vshll_n_u16(vget_high_u16(ua), 16);
    uint32x4_t b_lo_u32 = vshll_n_u16(vget_low_u16(ub), 16);
    uint32x4_t b_hi_u32 = vshll_n_u16(vget_high_u16(ub), 16);
    float32x4_t a_lo = vreinterpretq_f32_u32(a_lo_u32);
    float32x4_t a_hi = vreinterpretq_f32_u32(a_hi_u32);
    float32x4_t b_lo = vreinterpretq_f32_u32(b_lo_u32);
    float32x4_t b_hi = vreinterpretq_f32_u32(b_hi_u32);

    float32x4_t r_lo = vmulq_f32(a_lo, b_lo);
    float32x4_t r_hi = vmulq_f32(a_hi, b_hi);

    uint32x4_t u_lo = vreinterpretq_u32_f32(r_lo);
    uint32x4_t u_hi = vreinterpretq_u32_f32(r_hi);
    uint32x4_t round_const = vdupq_n_u32(0x7FFF);
    uint32x4_t one = vdupq_n_u32(1);
    uint32x4_t bias_lo = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_lo, 16), one));
    uint32x4_t bias_hi = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_hi, 16), one));
    uint32x4_t rr_lo = vshrq_n_u32(vaddq_u32(u_lo, bias_lo), 16);
    uint32x4_t rr_hi = vshrq_n_u32(vaddq_u32(u_hi, bias_hi), 16);
    uint16x4_t h_lo = vmovn_u32(rr_lo);
    uint16x4_t h_hi = vmovn_u32(rr_hi);
    return vreinterpretq_bf16_u16(vcombine_u16(h_lo, h_hi));
}

bfloat16x8_t div_bf16x8(bfloat16x8_t a, bfloat16x8_t b) {
    uint16x8_t ua = vreinterpretq_u16_bf16(a);
    uint16x8_t ub = vreinterpretq_u16_bf16(b);
    uint32x4_t a_lo_u32 = vshll_n_u16(vget_low_u16(ua), 16);
    uint32x4_t a_hi_u32 = vshll_n_u16(vget_high_u16(ua), 16);
    uint32x4_t b_lo_u32 = vshll_n_u16(vget_low_u16(ub), 16);
    uint32x4_t b_hi_u32 = vshll_n_u16(vget_high_u16(ub), 16);
    float32x4_t a_lo = vreinterpretq_f32_u32(a_lo_u32);
    float32x4_t a_hi = vreinterpretq_f32_u32(a_hi_u32);
    float32x4_t b_lo = vreinterpretq_f32_u32(b_lo_u32);
    float32x4_t b_hi = vreinterpretq_f32_u32(b_hi_u32);

    float32x4_t r_lo = vdivq_f32(a_lo, b_lo);
    float32x4_t r_hi = vdivq_f32(a_hi, b_hi);

    uint32x4_t u_lo = vreinterpretq_u32_f32(r_lo);
    uint32x4_t u_hi = vreinterpretq_u32_f32(r_hi);
    uint32x4_t round_const = vdupq_n_u32(0x7FFF);
    uint32x4_t one = vdupq_n_u32(1);
    uint32x4_t bias_lo = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_lo, 16), one));
    uint32x4_t bias_hi = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_hi, 16), one));
    uint32x4_t rr_lo = vshrq_n_u32(vaddq_u32(u_lo, bias_lo), 16);
    uint32x4_t rr_hi = vshrq_n_u32(vaddq_u32(u_hi, bias_hi), 16);
    uint16x4_t h_lo = vmovn_u32(rr_lo);
    uint16x4_t h_hi = vmovn_u32(rr_hi);
    return vreinterpretq_bf16_u16(vcombine_u16(h_lo, h_hi));
}

// Fused multiply-add: a * b + c (using F32 FMA for precision)
bfloat16x8_t fma_bf16x8(bfloat16x8_t a, bfloat16x8_t b, bfloat16x8_t c) {
    uint16x8_t ua = vreinterpretq_u16_bf16(a);
    uint16x8_t ub = vreinterpretq_u16_bf16(b);
    uint16x8_t uc = vreinterpretq_u16_bf16(c);
    uint32x4_t a_lo_u32 = vshll_n_u16(vget_low_u16(ua), 16);
    uint32x4_t a_hi_u32 = vshll_n_u16(vget_high_u16(ua), 16);
    uint32x4_t b_lo_u32 = vshll_n_u16(vget_low_u16(ub), 16);
    uint32x4_t b_hi_u32 = vshll_n_u16(vget_high_u16(ub), 16);
    uint32x4_t c_lo_u32 = vshll_n_u16(vget_low_u16(uc), 16);
    uint32x4_t c_hi_u32 = vshll_n_u16(vget_high_u16(uc), 16);
    float32x4_t a_lo = vreinterpretq_f32_u32(a_lo_u32);
    float32x4_t a_hi = vreinterpretq_f32_u32(a_hi_u32);
    float32x4_t b_lo = vreinterpretq_f32_u32(b_lo_u32);
    float32x4_t b_hi = vreinterpretq_f32_u32(b_hi_u32);
    float32x4_t c_lo = vreinterpretq_f32_u32(c_lo_u32);
    float32x4_t c_hi = vreinterpretq_f32_u32(c_hi_u32);

    float32x4_t r_lo = vfmaq_f32(c_lo, a_lo, b_lo);
    float32x4_t r_hi = vfmaq_f32(c_hi, a_hi, b_hi);

    uint32x4_t u_lo = vreinterpretq_u32_f32(r_lo);
    uint32x4_t u_hi = vreinterpretq_u32_f32(r_hi);
    uint32x4_t round_const = vdupq_n_u32(0x7FFF);
    uint32x4_t one = vdupq_n_u32(1);
    uint32x4_t bias_lo = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_lo, 16), one));
    uint32x4_t bias_hi = vaddq_u32(round_const, vandq_u32(vshrq_n_u32(u_hi, 16), one));
    uint32x4_t rr_lo = vshrq_n_u32(vaddq_u32(u_lo, bias_lo), 16);
    uint32x4_t rr_hi = vshrq_n_u32(vaddq_u32(u_hi, bias_hi), 16);
    uint16x4_t h_lo = vmovn_u32(rr_lo);
    uint16x4_t h_hi = vmovn_u32(rr_hi);
    return vreinterpretq_bf16_u16(vcombine_u16(h_lo, h_hi));
}

// ============================================================================
// BFloat16x8 In-Place Operations (avoid return allocation overhead)
// ============================================================================

void add_bf16x8_ip(bfloat16x8_t a, bfloat16x8_t b, bfloat16x8_t *result) {
    *result = add_bf16x8(a, b);
}

void sub_bf16x8_ip(bfloat16x8_t a, bfloat16x8_t b, bfloat16x8_t *result) {
    *result = sub_bf16x8(a, b);
}

void mul_bf16x8_ip(bfloat16x8_t a, bfloat16x8_t b, bfloat16x8_t *result) {
    *result = mul_bf16x8(a, b);
}

void div_bf16x8_ip(bfloat16x8_t a, bfloat16x8_t b, bfloat16x8_t *result) {
    *result = div_bf16x8(a, b);
}

// Fused multiply-add with BF16 accumulator: *acc = a * b + *acc
// Note: For ML workloads, prefer BFDOT with F32 accumulator for better precision
void muladd_bf16x8_acc(bfloat16x8_t a, bfloat16x8_t b, bfloat16x8_t *acc) {
    *acc = fma_bf16x8(a, b, *acc);
}

// Fused multiply-add to output: *result = a * b + c
void muladd_bf16x8_ip(bfloat16x8_t a, bfloat16x8_t b, bfloat16x8_t c, bfloat16x8_t *result) {
    *result = fma_bf16x8(a, b, c);
}

// ============================================================================
// BFloat16 Dot Product to F32 Accumulator (single vector pair)
// ============================================================================
// This is the preferred pattern for ML: keep accumulators in F32, only use BF16
// for storage. Uses BFDOT instruction for optimal performance.

// Accumulates dot product of two BF16x8 vectors into F32x4 accumulator
// Each F32 lane receives dot product of 2 BF16 pairs:
//   acc[0] += a[0]*b[0] + a[1]*b[1]
//   acc[1] += a[2]*b[2] + a[3]*b[3]
//   acc[2] += a[4]*b[4] + a[5]*b[5]
//   acc[3] += a[6]*b[6] + a[7]*b[7]
void bfdot_bf16x8_f32x4_acc(bfloat16x8_t a, bfloat16x8_t b, float32x4_t *acc) {
    *acc = vbfdotq_f32(*acc, a, b);
}
