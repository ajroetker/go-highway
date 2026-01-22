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

// Blocked/Cache-Tiled NEON Matrix Multiplication for go-highway
// Compile with: -march=armv8.2-a+fp16 -march=armv8.6-a+bf16
//
// Implements cache-efficient blocked matrix multiplication using NEON SIMD.
// Block sizes tuned for 32KB L1 cache:
//   - 3 blocks of 48x48 float32 = 27KB < 32KB
//   - For f16: same block count, 2x elements per block
//   - For bf16: same approach with f32 accumulation

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// Block size for cache tiling (same as Go implementation)
#define BLOCK_SIZE 48

// =============================================================================
// blocked_matmul_neon_f16: Cache-tiled NEON matrix multiply for float16
// =============================================================================
// Computes C = A * B with cache-efficient blocking.
// Uses native NEON f16 FMA instructions (ARMv8.2-A FP16).
//
// Block structure:
//   - Outer loops iterate over output blocks
//   - K dimension blocked for cache reuse of A and B panels
//   - Inner kernel uses "broadcast A, stream B" with NEON f16
//
// NEON f16: 8 elements per 128-bit vector
//
// func blocked_matmul_neon_f16(a, b, c unsafe.Pointer, m, n, k int64)
void blocked_matmul_neon_f16(__fp16 *a, __fp16 *b, __fp16 *c,
                              long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // First, zero the output matrix
    for (long i = 0; i < m * n; i++) {
        c[i] = (__fp16)0.0f;
    }

    // Block over i (M dimension)
    for (long bi = 0; bi < m; bi += BLOCK_SIZE) {
        long i_end = (bi + BLOCK_SIZE < m) ? bi + BLOCK_SIZE : m;

        // Block over j (N dimension)
        for (long bj = 0; bj < n; bj += BLOCK_SIZE) {
            long j_end = (bj + BLOCK_SIZE < n) ? bj + BLOCK_SIZE : n;

            // Block over k (contracting dimension)
            for (long bk = 0; bk < k; bk += BLOCK_SIZE) {
                long k_end = (bk + BLOCK_SIZE < k) ? bk + BLOCK_SIZE : k;

                // Inner kernel: process block
                for (long i = bi; i < i_end; i++) {
                    // Process output columns in chunks of 8 (NEON f16 vector width)
                    long j;
                    for (j = bj; j + 8 <= j_end; j += 8) {
                        // Load current accumulator
                        float16x8_t acc = vld1q_f16(c + i * n + j);

                        // Accumulate over K block
                        for (long p = bk; p < k_end; p++) {
                            // Broadcast A[i,p] to all lanes
                            float16x8_t a_val = vdupq_n_f16(a[i * k + p]);

                            // Load B[p,j:j+8]
                            float16x8_t b_row = vld1q_f16(b + p * n + j);

                            // FMA: acc += a_val * b_row
                            acc = vfmaq_f16(acc, a_val, b_row);
                        }

                        // Store result
                        vst1q_f16(c + i * n + j, acc);
                    }

                    // Handle remainder (less than 8 elements)
                    for (; j < j_end; j++) {
                        __fp16 sum = c[i * n + j];
                        for (long p = bk; p < k_end; p++) {
                            sum += a[i * k + p] * b[p * n + j];
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

// =============================================================================
// blocked_matmul_neon_bf16: Cache-tiled NEON matrix multiply for bfloat16
// =============================================================================
// Computes C = A * B with cache-efficient blocking.
// Uses f32 accumulation for precision, converts at store.
//
// BFDOT (BFloat16 DOT product) processes pairs of bf16, accumulating into f32.
// For blocking, we accumulate in f32 and convert back to bf16 at the end.
//
// func blocked_matmul_neon_bf16(a, b, c unsafe.Pointer, m, n, k int64)
void blocked_matmul_neon_bf16(__bf16 *a, __bf16 *b, __bf16 *c,
                               long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // First, zero the output matrix
    for (long i = 0; i < m * n; i++) {
        // Zero bf16 is just 0x0000
        unsigned short zero = 0;
        __builtin_memcpy(&c[i], &zero, sizeof(zero));
    }

    // Block over i (M dimension)
    for (long bi = 0; bi < m; bi += BLOCK_SIZE) {
        long i_end = (bi + BLOCK_SIZE < m) ? bi + BLOCK_SIZE : m;

        // Block over j (N dimension)
        for (long bj = 0; bj < n; bj += BLOCK_SIZE) {
            long j_end = (bj + BLOCK_SIZE < n) ? bj + BLOCK_SIZE : n;

            // Block over k (contracting dimension)
            for (long bk = 0; bk < k; bk += BLOCK_SIZE) {
                long k_end = (bk + BLOCK_SIZE < k) ? bk + BLOCK_SIZE : k;

                // Inner kernel: process block
                for (long i = bi; i < i_end; i++) {
                    // Process output columns in chunks of 4 (f32 accumulator width)
                    long j;
                    for (j = bj; j + 4 <= j_end; j += 4) {
                        // Load current accumulator (convert bf16 to f32)
                        // Use a temporary array and vld1q to load
                        float acc_arr[4];
                        for (int col = 0; col < 4; col++) {
                            unsigned short bf16_bits;
                            __builtin_memcpy(&bf16_bits, &c[i * n + j + col], sizeof(bf16_bits));
                            unsigned int f32_bits = ((unsigned int)bf16_bits) << 16;
                            __builtin_memcpy(&acc_arr[col], &f32_bits, sizeof(float));
                        }
                        float32x4_t acc = vld1q_f32(acc_arr);

                        // Accumulate over K block using BFDOT
                        // Process K in pairs for BFDOT
                        long p;
                        for (p = bk; p + 2 <= k_end; p += 2) {
                            // Load 2 consecutive A elements and broadcast
                            bfloat16x8_t a_pair = vld1q_bf16(a + i * k + p);

                            // Load B[p:p+2, j:j+4]
                            bfloat16x4_t b_row0 = vld1_bf16(b + p * n + j);
                            bfloat16x4_t b_row1 = vld1_bf16(b + (p + 1) * n + j);
                            bfloat16x8_t b_combined = vcombine_bf16(b_row0, b_row1);

                            // BFDOT accumulate
                            acc = vbfdotq_f32(acc, a_pair, b_combined);
                        }

                        // Handle odd K element - store acc to array, process, reload
                        if (p < k_end) {
                            vst1q_f32(acc_arr, acc);

                            unsigned short a_bits;
                            __builtin_memcpy(&a_bits, &a[i * k + p], sizeof(a_bits));
                            unsigned int a_f32_bits = ((unsigned int)a_bits) << 16;
                            float a_val;
                            __builtin_memcpy(&a_val, &a_f32_bits, sizeof(a_val));

                            for (int col = 0; col < 4; col++) {
                                unsigned short b_bits;
                                __builtin_memcpy(&b_bits, &b[p * n + j + col], sizeof(b_bits));
                                unsigned int b_f32_bits = ((unsigned int)b_bits) << 16;
                                float b_val;
                                __builtin_memcpy(&b_val, &b_f32_bits, sizeof(b_val));

                                acc_arr[col] += a_val * b_val;
                            }

                            acc = vld1q_f32(acc_arr);
                        }

                        // Convert f32 accumulator back to bf16 and store
                        bfloat16x4_t result = vcvt_bf16_f32(acc);
                        vst1_bf16(c + i * n + j, result);
                    }

                    // Handle remainder (less than 4 elements)
                    for (; j < j_end; j++) {
                        // Scalar bf16 accumulation via f32
                        unsigned short c_bits;
                        __builtin_memcpy(&c_bits, &c[i * n + j], sizeof(c_bits));
                        unsigned int c_f32_bits = ((unsigned int)c_bits) << 16;
                        float sum;
                        __builtin_memcpy(&sum, &c_f32_bits, sizeof(sum));

                        for (long p = bk; p < k_end; p++) {
                            unsigned short a_bits, b_bits;
                            __builtin_memcpy(&a_bits, &a[i * k + p], sizeof(a_bits));
                            __builtin_memcpy(&b_bits, &b[p * n + j], sizeof(b_bits));

                            unsigned int a_f32_bits = ((unsigned int)a_bits) << 16;
                            unsigned int b_f32_bits = ((unsigned int)b_bits) << 16;
                            float a_val, b_val;
                            __builtin_memcpy(&a_val, &a_f32_bits, sizeof(a_val));
                            __builtin_memcpy(&b_val, &b_f32_bits, sizeof(b_val));

                            sum += a_val * b_val;
                        }

                        // Convert f32 to bf16 with rounding
                        unsigned int sum_bits;
                        __builtin_memcpy(&sum_bits, &sum, sizeof(sum_bits));
                        unsigned int rounding = 0x7FFF + ((sum_bits >> 16) & 1);
                        sum_bits += rounding;
                        unsigned short bf16_result = (unsigned short)(sum_bits >> 16);
                        __builtin_memcpy(&c[i * n + j], &bf16_result, sizeof(bf16_result));
                    }
                }
            }
        }
    }
}
