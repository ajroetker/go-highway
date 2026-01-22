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

// NEON Matrix Multiplication for go-highway - Float16, Float32, Float64
// Compile with: -march=armv8.2-a+fp16
//
// This file contains matmul implementations that require ARMv8.2-A FP16
// extension or only basic NEON (F32/F64).
//
// For bf16: uses BFMMLA matrix multiply accumulate (ARMv8.6-A BF16)
// See matmul_neon_bf16_arm64.c for BF16 implementation.

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// =============================================================================
// matmul_neon_f16: NEON matrix multiply for float16
// =============================================================================
// Computes C = A * B where:
//   A is M x K (row-major)
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// Uses "broadcast A, stream B" algorithm:
//   For each row i of A:
//     For each element A[i,p]:
//       C[i,:] += A[i,p] * B[p,:]
//
// NEON f16: 8 elements per 128-bit vector
// Requires ARMv8.2-A with FP16 extension.
//
// func matmul_neon_f16(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_neon_f16(__fp16 *a, __fp16 *b, __fp16 *c,
                      long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 8 (NEON f16 vector width)
        for (long j = 0; j < n; j += 8) {
            // Initialize accumulator
            float16x8_t acc = vdupq_n_f16((__fp16)0.0f);

            // Accumulate: acc += A[i,p] * B[p,j:j+8] for all p
            for (long p = 0; p < k; p++) {
                // Broadcast A[i,p] to all lanes
                float16x8_t a_val = vdupq_n_f16(a[i * k + p]);

                // Load B[p,j:j+8]
                float16x8_t b_row = vld1q_f16(b + p * n + j);

                // FMA: acc += a_val * b_row
                acc = vfmaq_f16(acc, a_val, b_row);
            }

            // Store result to C[i,j:j+8]
            vst1q_f16(c + i * n + j, acc);
        }
    }
}

// =============================================================================
// matmul_neon_f32: NEON matrix multiply for float32
// =============================================================================
// Standard NEON f32 matmul - works on all ARMv8-A CPUs.
//
// func matmul_neon_f32(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_neon_f32(float *a, float *b, float *c,
                      long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 4 (NEON f32 vector width)
        for (long j = 0; j < n; j += 4) {
            // Initialize accumulator
            float32x4_t acc = vdupq_n_f32(0.0f);

            // Accumulate: acc += A[i,p] * B[p,j:j+4] for all p
            for (long p = 0; p < k; p++) {
                // Broadcast A[i,p] to all lanes
                float32x4_t a_val = vdupq_n_f32(a[i * k + p]);

                // Load B[p,j:j+4]
                float32x4_t b_row = vld1q_f32(b + p * n + j);

                // FMA: acc += a_val * b_row
                acc = vfmaq_f32(acc, a_val, b_row);
            }

            // Store result to C[i,j:j+4]
            vst1q_f32(c + i * n + j, acc);
        }
    }
}

// =============================================================================
// matmul_neon_f64: NEON matrix multiply for float64
// =============================================================================
// NEON f64: 2 elements per 128-bit vector - works on all ARMv8-A CPUs.
//
// func matmul_neon_f64(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_neon_f64(double *a, double *b, double *c,
                      long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 2 (NEON f64 vector width)
        for (long j = 0; j < n; j += 2) {
            // Initialize accumulator
            float64x2_t acc = vdupq_n_f64(0.0);

            // Accumulate: acc += A[i,p] * B[p,j:j+2] for all p
            for (long p = 0; p < k; p++) {
                // Broadcast A[i,p] to all lanes
                float64x2_t a_val = vdupq_n_f64(a[i * k + p]);

                // Load B[p,j:j+2]
                float64x2_t b_row = vld1q_f64(b + p * n + j);

                // FMA: acc += a_val * b_row
                acc = vfmaq_f64(acc, a_val, b_row);
            }

            // Store result to C[i,j:j+2]
            vst1q_f64(c + i * n + j, acc);
        }
    }
}
