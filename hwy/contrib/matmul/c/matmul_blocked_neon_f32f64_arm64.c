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

// Blocked/Cache-Tiled NEON Matrix Multiplication for go-highway (F32/F64 only)
// Compile with: -march=armv8-a
//
// Implements cache-efficient blocked matrix multiplication using NEON SIMD.
// Block sizes tuned for 32KB L1 cache:
//   - 3 blocks of 48x48 float32 = 27KB < 32KB

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// Block size for cache tiling (same as Go implementation)
#define BLOCK_SIZE 48

// =============================================================================
// blocked_matmul_neon_f32: Cache-tiled NEON matrix multiply for float32
// =============================================================================
// Computes C = A * B with cache-efficient blocking.
// Uses NEON f32 FMA instructions.
//
// NEON f32: 4 elements per 128-bit vector
//
// func blocked_matmul_neon_f32(a, b, c unsafe.Pointer, m, n, k int64)
void blocked_matmul_neon_f32(float *a, float *b, float *c,
                              long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // First, zero the output matrix
    for (long i = 0; i < m * n; i++) {
        c[i] = 0.0f;
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
                    // Process output columns in chunks of 4 (NEON f32 vector width)
                    long j;
                    for (j = bj; j + 4 <= j_end; j += 4) {
                        // Load current accumulator
                        float32x4_t acc = vld1q_f32(c + i * n + j);

                        // Accumulate over K block
                        for (long p = bk; p < k_end; p++) {
                            // Broadcast A[i,p] to all lanes
                            float32x4_t a_val = vdupq_n_f32(a[i * k + p]);

                            // Load B[p,j:j+4]
                            float32x4_t b_row = vld1q_f32(b + p * n + j);

                            // FMA: acc += a_val * b_row
                            acc = vfmaq_f32(acc, a_val, b_row);
                        }

                        // Store result
                        vst1q_f32(c + i * n + j, acc);
                    }

                    // Handle remainder (less than 4 elements)
                    for (; j < j_end; j++) {
                        float sum = c[i * n + j];
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
// blocked_matmul_neon_f64: Cache-tiled NEON matrix multiply for float64
// =============================================================================
// Computes C = A * B with cache-efficient blocking.
// Uses NEON f64 FMA instructions.
//
// NEON f64: 2 elements per 128-bit vector
//
// func blocked_matmul_neon_f64(a, b, c unsafe.Pointer, m, n, k int64)
void blocked_matmul_neon_f64(double *a, double *b, double *c,
                              long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // First, zero the output matrix
    for (long i = 0; i < m * n; i++) {
        c[i] = 0.0;
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
                    // Process output columns in chunks of 2 (NEON f64 vector width)
                    long j;
                    for (j = bj; j + 2 <= j_end; j += 2) {
                        // Load current accumulator
                        float64x2_t acc = vld1q_f64(c + i * n + j);

                        // Accumulate over K block
                        for (long p = bk; p < k_end; p++) {
                            // Broadcast A[i,p] to all lanes
                            float64x2_t a_val = vdupq_n_f64(a[i * k + p]);

                            // Load B[p,j:j+2]
                            float64x2_t b_row = vld1q_f64(b + p * n + j);

                            // FMA: acc += a_val * b_row
                            acc = vfmaq_f64(acc, a_val, b_row);
                        }

                        // Store result
                        vst1q_f64(c + i * n + j, acc);
                    }

                    // Handle remainder (1 element)
                    for (; j < j_end; j++) {
                        double sum = c[i * n + j];
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
