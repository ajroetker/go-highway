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

// NEON Block Kernel for go-highway
// Compile with: -O3 --target arm64
//
// Computes C += A^T * B for square blocks using NEON SIMD.
// aT is pre-transposed A (rows are original A columns).
// b is normal row-major B.
//
// Uses "broadcast A, stream B" pattern:
//   For each k:
//     C[i,:] += aT[k,i] * B[k,:]
//
// Processes 4 rows (f32) or 2 rows (f64) at a time for register reuse.

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// =============================================================================
// block_muladd_neon_f32: C += A^T * B using NEON for float32
// =============================================================================
// aT is blockDim x blockDim (pre-transposed A, row-major)
// b is blockDim x blockDim (row-major)
// c is blockDim x blockDim (row-major, accumulated into)
//
// func block_muladd_neon_f32(aT, b, c unsafe.Pointer, blockDim int64)
void block_muladd_neon_f32(float *aT, float *b, float *c, long *pblockDim) {
    long n = *pblockDim;

    // Process 4 rows at a time
    long i;
    for (i = 0; i + 4 <= n; i += 4) {
        // Process 4 columns at a time (NEON f32 vector width)
        long j;
        for (j = 0; j + 4 <= n; j += 4) {
            // Load C accumulators for 4 rows
            float32x4_t c0 = vld1q_f32(c + (i + 0) * n + j);
            float32x4_t c1 = vld1q_f32(c + (i + 1) * n + j);
            float32x4_t c2 = vld1q_f32(c + (i + 2) * n + j);
            float32x4_t c3 = vld1q_f32(c + (i + 3) * n + j);

            // K-loop: accumulate outer products
            for (long k = 0; k < n; k++) {
                // Load aT[k, i:i+4] = A[i:i+4, k] (contiguous in aT)
                // Broadcast each A value
                float32x4_t a0 = vdupq_n_f32(aT[k * n + i + 0]);
                float32x4_t a1 = vdupq_n_f32(aT[k * n + i + 1]);
                float32x4_t a2 = vdupq_n_f32(aT[k * n + i + 2]);
                float32x4_t a3 = vdupq_n_f32(aT[k * n + i + 3]);

                // Load B[k, j:j+4]
                float32x4_t b_row = vld1q_f32(b + k * n + j);

                // FMA: C[i+r, j:j+4] += A[i+r, k] * B[k, j:j+4]
                c0 = vfmaq_f32(c0, a0, b_row);
                c1 = vfmaq_f32(c1, a1, b_row);
                c2 = vfmaq_f32(c2, a2, b_row);
                c3 = vfmaq_f32(c3, a3, b_row);
            }

            // Store back
            vst1q_f32(c + (i + 0) * n + j, c0);
            vst1q_f32(c + (i + 1) * n + j, c1);
            vst1q_f32(c + (i + 2) * n + j, c2);
            vst1q_f32(c + (i + 3) * n + j, c3);
        }

        // Scalar tail for remaining columns
        for (; j < n; j++) {
            float s0 = c[(i + 0) * n + j];
            float s1 = c[(i + 1) * n + j];
            float s2 = c[(i + 2) * n + j];
            float s3 = c[(i + 3) * n + j];
            for (long k = 0; k < n; k++) {
                float bv = b[k * n + j];
                s0 += aT[k * n + i + 0] * bv;
                s1 += aT[k * n + i + 1] * bv;
                s2 += aT[k * n + i + 2] * bv;
                s3 += aT[k * n + i + 3] * bv;
            }
            c[(i + 0) * n + j] = s0;
            c[(i + 1) * n + j] = s1;
            c[(i + 2) * n + j] = s2;
            c[(i + 3) * n + j] = s3;
        }
    }

    // Remaining rows (less than 4)
    for (; i < n; i++) {
        long j;
        for (j = 0; j + 4 <= n; j += 4) {
            float32x4_t acc = vld1q_f32(c + i * n + j);
            for (long k = 0; k < n; k++) {
                float32x4_t a_bcast = vdupq_n_f32(aT[k * n + i]);
                float32x4_t b_row = vld1q_f32(b + k * n + j);
                acc = vfmaq_f32(acc, a_bcast, b_row);
            }
            vst1q_f32(c + i * n + j, acc);
        }
        // Scalar tail
        for (; j < n; j++) {
            float sum = c[i * n + j];
            for (long k = 0; k < n; k++) {
                sum += aT[k * n + i] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// =============================================================================
// block_muladd_neon_f64: C += A^T * B using NEON for float64
// =============================================================================
// Same algorithm but with 2-wide vectors (128-bit NEON holds 2 doubles).
//
// func block_muladd_neon_f64(aT, b, c unsafe.Pointer, blockDim int64)
void block_muladd_neon_f64(double *aT, double *b, double *c, long *pblockDim) {
    long n = *pblockDim;

    // Process 2 rows at a time
    long i;
    for (i = 0; i + 2 <= n; i += 2) {
        long j;
        for (j = 0; j + 2 <= n; j += 2) {
            // Load C accumulators for 2 rows
            float64x2_t c0 = vld1q_f64(c + (i + 0) * n + j);
            float64x2_t c1 = vld1q_f64(c + (i + 1) * n + j);

            // K-loop
            for (long k = 0; k < n; k++) {
                float64x2_t a0 = vdupq_n_f64(aT[k * n + i + 0]);
                float64x2_t a1 = vdupq_n_f64(aT[k * n + i + 1]);

                float64x2_t b_row = vld1q_f64(b + k * n + j);

                c0 = vfmaq_f64(c0, a0, b_row);
                c1 = vfmaq_f64(c1, a1, b_row);
            }

            vst1q_f64(c + (i + 0) * n + j, c0);
            vst1q_f64(c + (i + 1) * n + j, c1);
        }

        // Scalar tail for remaining columns
        for (; j < n; j++) {
            double s0 = c[(i + 0) * n + j];
            double s1 = c[(i + 1) * n + j];
            for (long k = 0; k < n; k++) {
                double bv = b[k * n + j];
                s0 += aT[k * n + i + 0] * bv;
                s1 += aT[k * n + i + 1] * bv;
            }
            c[(i + 0) * n + j] = s0;
            c[(i + 1) * n + j] = s1;
        }
    }

    // Remaining single row
    for (; i < n; i++) {
        long j;
        for (j = 0; j + 2 <= n; j += 2) {
            float64x2_t acc = vld1q_f64(c + i * n + j);
            for (long k = 0; k < n; k++) {
                float64x2_t a_bcast = vdupq_n_f64(aT[k * n + i]);
                float64x2_t b_row = vld1q_f64(b + k * n + j);
                acc = vfmaq_f64(acc, a_bcast, b_row);
            }
            vst1q_f64(c + i * n + j, acc);
        }
        // Scalar tail
        for (; j < n; j++) {
            double sum = c[i * n + j];
            for (long k = 0; k < n; k++) {
                sum += aT[k * n + i] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
