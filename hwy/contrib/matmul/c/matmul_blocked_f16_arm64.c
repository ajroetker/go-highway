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

// Blocked/Cache-Tiled NEON Matrix Multiplication for go-highway - Float16
// Compile with: -march=armv8.2-a+fp16
//
// Implements cache-efficient blocked matrix multiplication using NEON SIMD.
// Block sizes tuned for 32KB L1 cache.
//
// For bf16: see matmul_blocked_bf16_arm64.c

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
// Requires ARMv8.2-A with FP16 extension.
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
