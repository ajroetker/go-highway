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

// NEON Matrix Multiplication for go-highway - BFloat16
// Compile with: -march=armv8.6-a+bf16
//
// This file contains BF16 matmul that requires ARMv8.6-A BF16 extension.
// Uses BFDOT for bf16 computation with f32 accumulation.
//
// For f16/f32/f64: see matmul_neon_f16_arm64.c

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// =============================================================================
// matmul_neon_bf16: NEON matrix multiply for bfloat16
// =============================================================================
// Computes C = A * B using BFDOT (BFloat16 DOT product)
//
// BFDOT processes pairs of bf16 elements, accumulating into f32.
// This is different from standard matmul - we process 2 elements at a time.
//
// Requires ARMv8.6-A with BF16 extension (FEAT_BF16).
//
// func matmul_neon_bf16(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_neon_bf16(__bf16 *a, __bf16 *b, __bf16 *c,
                       long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 4 (f32 accumulator width)
        for (long j = 0; j < n; j += 4) {
            // Initialize f32 accumulators (bf16 math uses f32 accumulation)
            float32x4_t acc = vdupq_n_f32(0.0f);

            // Process K dimension in pairs (BFDOT processes 2 bf16 at a time)
            for (long p = 0; p < k; p += 2) {
                // Load 2 consecutive A elements: A[i,p], A[i,p+1]
                bfloat16x8_t a_pair = vld1q_bf16(a + i * k + p);
                // We only use the first 2 elements, but load 8 for BFDOT format

                // Load B[p:p+2, j:j+4] - need to gather 2 rows of 4 elements each
                // B[p,j:j+4] and B[p+1,j:j+4]
                bfloat16x4_t b_row0 = vld1_bf16(b + p * n + j);
                bfloat16x4_t b_row1 = vld1_bf16(b + (p + 1) * n + j);

                // Combine into 8 elements for BFDOT: [b0, b1, b0, b1, ...]
                bfloat16x8_t b_combined = vcombine_bf16(b_row0, b_row1);

                // BFDOT: acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]
                // This computes dot products of bf16 pairs into f32
                acc = vbfdotq_f32(acc, a_pair, b_combined);
            }

            // Convert f32 accumulator back to bf16 and store
            bfloat16x4_t result = vcvt_bf16_f32(acc);
            vst1_bf16(c + i * n + j, result);
        }
    }
}
