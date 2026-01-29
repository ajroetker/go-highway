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

// MatMulKLast NEON implementation for ARM64
// Computes C = A * B^T where A is [M,K] and B is [N,K] (K-last layout)
//
// This is optimized for the dot-product pattern:
//   C[i,j] = sum_k(A[i,k] * B[j,k])
//
// Uses tiled computation to:
// 1. Reuse A and B loads across multiple output elements
// 2. Keep accumulators in registers across the K dimension
// 3. Only do horizontal sums at tile boundaries

#include <arm_neon.h>

// =============================================================================
// matmul_klast_neon_f32: Tiled dot-product matmul for K-last layout
// =============================================================================
// Processes 4 rows of A × 4 rows of B = 16 output elements per tile
// This gives 8 loads per K iteration (4 from A, 4 from B) and 16 FMAs
// Horizontal sums only happen once per 16 outputs
//
// func matmul_klast_neon_f32(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_klast_neon_f32(float *a, float *b, float *c,
                            long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process 4×4 output tiles
    for (long i = 0; i < m; i += 4) {
        long iEnd = i + 4;
        if (iEnd > m) iEnd = m;
        long iCount = iEnd - i;

        for (long j = 0; j < n; j += 4) {
            long jEnd = j + 4;
            if (jEnd > n) jEnd = n;
            long jCount = jEnd - j;

            // 16 accumulators for 4×4 tile (use all even if tile is smaller)
            float32x4_t acc00 = vdupq_n_f32(0.0f);
            float32x4_t acc01 = vdupq_n_f32(0.0f);
            float32x4_t acc02 = vdupq_n_f32(0.0f);
            float32x4_t acc03 = vdupq_n_f32(0.0f);
            float32x4_t acc10 = vdupq_n_f32(0.0f);
            float32x4_t acc11 = vdupq_n_f32(0.0f);
            float32x4_t acc12 = vdupq_n_f32(0.0f);
            float32x4_t acc13 = vdupq_n_f32(0.0f);
            float32x4_t acc20 = vdupq_n_f32(0.0f);
            float32x4_t acc21 = vdupq_n_f32(0.0f);
            float32x4_t acc22 = vdupq_n_f32(0.0f);
            float32x4_t acc23 = vdupq_n_f32(0.0f);
            float32x4_t acc30 = vdupq_n_f32(0.0f);
            float32x4_t acc31 = vdupq_n_f32(0.0f);
            float32x4_t acc32 = vdupq_n_f32(0.0f);
            float32x4_t acc33 = vdupq_n_f32(0.0f);

            // Vectorized accumulation along K (4 floats at a time)
            long p = 0;
            for (; p + 4 <= k; p += 4) {
                // Load 4 vectors from A rows (4 elements each)
                float32x4_t a0 = vld1q_f32(a + (i + 0) * k + p);
                float32x4_t a1 = vld1q_f32(a + (i + 1) * k + p);
                float32x4_t a2 = vld1q_f32(a + (i + 2) * k + p);
                float32x4_t a3 = vld1q_f32(a + (i + 3) * k + p);

                // Load 4 vectors from B rows
                float32x4_t b0 = vld1q_f32(b + (j + 0) * k + p);
                float32x4_t b1 = vld1q_f32(b + (j + 1) * k + p);
                float32x4_t b2 = vld1q_f32(b + (j + 2) * k + p);
                float32x4_t b3 = vld1q_f32(b + (j + 3) * k + p);

                // 16 FMAs: each A row × each B row
                acc00 = vfmaq_f32(acc00, a0, b0);
                acc01 = vfmaq_f32(acc01, a0, b1);
                acc02 = vfmaq_f32(acc02, a0, b2);
                acc03 = vfmaq_f32(acc03, a0, b3);

                acc10 = vfmaq_f32(acc10, a1, b0);
                acc11 = vfmaq_f32(acc11, a1, b1);
                acc12 = vfmaq_f32(acc12, a1, b2);
                acc13 = vfmaq_f32(acc13, a1, b3);

                acc20 = vfmaq_f32(acc20, a2, b0);
                acc21 = vfmaq_f32(acc21, a2, b1);
                acc22 = vfmaq_f32(acc22, a2, b2);
                acc23 = vfmaq_f32(acc23, a2, b3);

                acc30 = vfmaq_f32(acc30, a3, b0);
                acc31 = vfmaq_f32(acc31, a3, b1);
                acc32 = vfmaq_f32(acc32, a3, b2);
                acc33 = vfmaq_f32(acc33, a3, b3);
            }

            // Horizontal sums for the 16 accumulators
            float s00 = vaddvq_f32(acc00);
            float s01 = vaddvq_f32(acc01);
            float s02 = vaddvq_f32(acc02);
            float s03 = vaddvq_f32(acc03);
            float s10 = vaddvq_f32(acc10);
            float s11 = vaddvq_f32(acc11);
            float s12 = vaddvq_f32(acc12);
            float s13 = vaddvq_f32(acc13);
            float s20 = vaddvq_f32(acc20);
            float s21 = vaddvq_f32(acc21);
            float s22 = vaddvq_f32(acc22);
            float s23 = vaddvq_f32(acc23);
            float s30 = vaddvq_f32(acc30);
            float s31 = vaddvq_f32(acc31);
            float s32 = vaddvq_f32(acc32);
            float s33 = vaddvq_f32(acc33);

            // Scalar tail for remaining K elements
            for (; p < k; p++) {
                float a0s = a[(i + 0) * k + p];
                float a1s = a[(i + 1) * k + p];
                float a2s = a[(i + 2) * k + p];
                float a3s = a[(i + 3) * k + p];

                float b0s = b[(j + 0) * k + p];
                float b1s = b[(j + 1) * k + p];
                float b2s = b[(j + 2) * k + p];
                float b3s = b[(j + 3) * k + p];

                s00 += a0s * b0s;
                s01 += a0s * b1s;
                s02 += a0s * b2s;
                s03 += a0s * b3s;

                s10 += a1s * b0s;
                s11 += a1s * b1s;
                s12 += a1s * b2s;
                s13 += a1s * b3s;

                s20 += a2s * b0s;
                s21 += a2s * b1s;
                s22 += a2s * b2s;
                s23 += a2s * b3s;

                s30 += a3s * b0s;
                s31 += a3s * b1s;
                s32 += a3s * b2s;
                s33 += a3s * b3s;
            }

            // Store results (only valid elements based on tile size)
            if (iCount > 0) {
                if (jCount > 0) c[(i + 0) * n + (j + 0)] = s00;
                if (jCount > 1) c[(i + 0) * n + (j + 1)] = s01;
                if (jCount > 2) c[(i + 0) * n + (j + 2)] = s02;
                if (jCount > 3) c[(i + 0) * n + (j + 3)] = s03;
            }
            if (iCount > 1) {
                if (jCount > 0) c[(i + 1) * n + (j + 0)] = s10;
                if (jCount > 1) c[(i + 1) * n + (j + 1)] = s11;
                if (jCount > 2) c[(i + 1) * n + (j + 2)] = s12;
                if (jCount > 3) c[(i + 1) * n + (j + 3)] = s13;
            }
            if (iCount > 2) {
                if (jCount > 0) c[(i + 2) * n + (j + 0)] = s20;
                if (jCount > 1) c[(i + 2) * n + (j + 1)] = s21;
                if (jCount > 2) c[(i + 2) * n + (j + 2)] = s22;
                if (jCount > 3) c[(i + 2) * n + (j + 3)] = s23;
            }
            if (iCount > 3) {
                if (jCount > 0) c[(i + 3) * n + (j + 0)] = s30;
                if (jCount > 1) c[(i + 3) * n + (j + 1)] = s31;
                if (jCount > 2) c[(i + 3) * n + (j + 2)] = s32;
                if (jCount > 3) c[(i + 3) * n + (j + 3)] = s33;
            }
        }
    }
}

// =============================================================================
// matmul_klast_neon_f32_aligned: Fast path for 4-aligned dimensions
// =============================================================================
// When M, N are multiples of 4, we skip the boundary checks
//
// func matmul_klast_neon_f32_aligned(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_klast_neon_f32_aligned(float *a, float *b, float *c,
                                    long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process 4×4 output tiles (no boundary checks needed)
    for (long i = 0; i < m; i += 4) {
        for (long j = 0; j < n; j += 4) {
            // 16 accumulators
            float32x4_t acc00 = vdupq_n_f32(0.0f);
            float32x4_t acc01 = vdupq_n_f32(0.0f);
            float32x4_t acc02 = vdupq_n_f32(0.0f);
            float32x4_t acc03 = vdupq_n_f32(0.0f);
            float32x4_t acc10 = vdupq_n_f32(0.0f);
            float32x4_t acc11 = vdupq_n_f32(0.0f);
            float32x4_t acc12 = vdupq_n_f32(0.0f);
            float32x4_t acc13 = vdupq_n_f32(0.0f);
            float32x4_t acc20 = vdupq_n_f32(0.0f);
            float32x4_t acc21 = vdupq_n_f32(0.0f);
            float32x4_t acc22 = vdupq_n_f32(0.0f);
            float32x4_t acc23 = vdupq_n_f32(0.0f);
            float32x4_t acc30 = vdupq_n_f32(0.0f);
            float32x4_t acc31 = vdupq_n_f32(0.0f);
            float32x4_t acc32 = vdupq_n_f32(0.0f);
            float32x4_t acc33 = vdupq_n_f32(0.0f);

            // Main loop: 4 elements at a time
            long p = 0;
            for (; p + 4 <= k; p += 4) {
                float32x4_t a0 = vld1q_f32(a + (i + 0) * k + p);
                float32x4_t a1 = vld1q_f32(a + (i + 1) * k + p);
                float32x4_t a2 = vld1q_f32(a + (i + 2) * k + p);
                float32x4_t a3 = vld1q_f32(a + (i + 3) * k + p);

                float32x4_t b0 = vld1q_f32(b + (j + 0) * k + p);
                float32x4_t b1 = vld1q_f32(b + (j + 1) * k + p);
                float32x4_t b2 = vld1q_f32(b + (j + 2) * k + p);
                float32x4_t b3 = vld1q_f32(b + (j + 3) * k + p);

                acc00 = vfmaq_f32(acc00, a0, b0);
                acc01 = vfmaq_f32(acc01, a0, b1);
                acc02 = vfmaq_f32(acc02, a0, b2);
                acc03 = vfmaq_f32(acc03, a0, b3);

                acc10 = vfmaq_f32(acc10, a1, b0);
                acc11 = vfmaq_f32(acc11, a1, b1);
                acc12 = vfmaq_f32(acc12, a1, b2);
                acc13 = vfmaq_f32(acc13, a1, b3);

                acc20 = vfmaq_f32(acc20, a2, b0);
                acc21 = vfmaq_f32(acc21, a2, b1);
                acc22 = vfmaq_f32(acc22, a2, b2);
                acc23 = vfmaq_f32(acc23, a2, b3);

                acc30 = vfmaq_f32(acc30, a3, b0);
                acc31 = vfmaq_f32(acc31, a3, b1);
                acc32 = vfmaq_f32(acc32, a3, b2);
                acc33 = vfmaq_f32(acc33, a3, b3);
            }

            // Horizontal sums
            float s00 = vaddvq_f32(acc00);
            float s01 = vaddvq_f32(acc01);
            float s02 = vaddvq_f32(acc02);
            float s03 = vaddvq_f32(acc03);
            float s10 = vaddvq_f32(acc10);
            float s11 = vaddvq_f32(acc11);
            float s12 = vaddvq_f32(acc12);
            float s13 = vaddvq_f32(acc13);
            float s20 = vaddvq_f32(acc20);
            float s21 = vaddvq_f32(acc21);
            float s22 = vaddvq_f32(acc22);
            float s23 = vaddvq_f32(acc23);
            float s30 = vaddvq_f32(acc30);
            float s31 = vaddvq_f32(acc31);
            float s32 = vaddvq_f32(acc32);
            float s33 = vaddvq_f32(acc33);

            // Scalar tail
            for (; p < k; p++) {
                float a0s = a[(i + 0) * k + p];
                float a1s = a[(i + 1) * k + p];
                float a2s = a[(i + 2) * k + p];
                float a3s = a[(i + 3) * k + p];

                float b0s = b[(j + 0) * k + p];
                float b1s = b[(j + 1) * k + p];
                float b2s = b[(j + 2) * k + p];
                float b3s = b[(j + 3) * k + p];

                s00 += a0s * b0s; s01 += a0s * b1s; s02 += a0s * b2s; s03 += a0s * b3s;
                s10 += a1s * b0s; s11 += a1s * b1s; s12 += a1s * b2s; s13 += a1s * b3s;
                s20 += a2s * b0s; s21 += a2s * b1s; s22 += a2s * b2s; s23 += a2s * b3s;
                s30 += a3s * b0s; s31 += a3s * b1s; s32 += a3s * b2s; s33 += a3s * b3s;
            }

            // Store 4×4 tile
            c[(i + 0) * n + (j + 0)] = s00;
            c[(i + 0) * n + (j + 1)] = s01;
            c[(i + 0) * n + (j + 2)] = s02;
            c[(i + 0) * n + (j + 3)] = s03;

            c[(i + 1) * n + (j + 0)] = s10;
            c[(i + 1) * n + (j + 1)] = s11;
            c[(i + 1) * n + (j + 2)] = s12;
            c[(i + 1) * n + (j + 3)] = s13;

            c[(i + 2) * n + (j + 0)] = s20;
            c[(i + 2) * n + (j + 1)] = s21;
            c[(i + 2) * n + (j + 2)] = s22;
            c[(i + 2) * n + (j + 3)] = s23;

            c[(i + 3) * n + (j + 0)] = s30;
            c[(i + 3) * n + (j + 1)] = s31;
            c[(i + 3) * n + (j + 2)] = s32;
            c[(i + 3) * n + (j + 3)] = s33;
        }
    }
}

// =============================================================================
// matmul_klast_neon_f64: Tiled dot-product matmul for float64
// =============================================================================
// Uses 2-wide vectors (float64x2), processes 2×2 output tiles
//
// func matmul_klast_neon_f64(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_klast_neon_f64(double *a, double *b, double *c,
                            long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process 2×2 output tiles (float64x2 = 2 doubles)
    for (long i = 0; i < m; i += 2) {
        long iEnd = i + 2;
        if (iEnd > m) iEnd = m;
        long iCount = iEnd - i;

        for (long j = 0; j < n; j += 2) {
            long jEnd = j + 2;
            if (jEnd > n) jEnd = n;
            long jCount = jEnd - j;

            // 4 accumulators for 2×2 tile
            float64x2_t acc00 = vdupq_n_f64(0.0);
            float64x2_t acc01 = vdupq_n_f64(0.0);
            float64x2_t acc10 = vdupq_n_f64(0.0);
            float64x2_t acc11 = vdupq_n_f64(0.0);

            long p = 0;
            for (; p + 2 <= k; p += 2) {
                float64x2_t a0 = vld1q_f64(a + (i + 0) * k + p);
                float64x2_t a1 = vld1q_f64(a + (i + 1) * k + p);

                float64x2_t b0 = vld1q_f64(b + (j + 0) * k + p);
                float64x2_t b1 = vld1q_f64(b + (j + 1) * k + p);

                acc00 = vfmaq_f64(acc00, a0, b0);
                acc01 = vfmaq_f64(acc01, a0, b1);
                acc10 = vfmaq_f64(acc10, a1, b0);
                acc11 = vfmaq_f64(acc11, a1, b1);
            }

            // Horizontal sums
            double s00 = vaddvq_f64(acc00);
            double s01 = vaddvq_f64(acc01);
            double s10 = vaddvq_f64(acc10);
            double s11 = vaddvq_f64(acc11);

            // Scalar tail
            for (; p < k; p++) {
                double a0s = a[(i + 0) * k + p];
                double a1s = a[(i + 1) * k + p];
                double b0s = b[(j + 0) * k + p];
                double b1s = b[(j + 1) * k + p];

                s00 += a0s * b0s;
                s01 += a0s * b1s;
                s10 += a1s * b0s;
                s11 += a1s * b1s;
            }

            // Store results
            if (iCount > 0) {
                if (jCount > 0) c[(i + 0) * n + (j + 0)] = s00;
                if (jCount > 1) c[(i + 0) * n + (j + 1)] = s01;
            }
            if (iCount > 1) {
                if (jCount > 0) c[(i + 1) * n + (j + 0)] = s10;
                if (jCount > 1) c[(i + 1) * n + (j + 1)] = s11;
            }
        }
    }
}

// =============================================================================
// matmul_klast_neon_f16: Tiled dot-product matmul for float16
// =============================================================================
// Uses 8-wide vectors (float16x8), processes 4×4 output tiles
// Accumulates in f16 using FMLA f16 instructions
//
// func matmul_klast_neon_f16(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_klast_neon_f16(__fp16 *a, __fp16 *b, __fp16 *c,
                            long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process 4×4 output tiles (float16x8 = 8 halfs)
    for (long i = 0; i < m; i += 4) {
        long iEnd = i + 4;
        if (iEnd > m) iEnd = m;
        long iCount = iEnd - i;

        for (long j = 0; j < n; j += 4) {
            long jEnd = j + 4;
            if (jEnd > n) jEnd = n;
            long jCount = jEnd - j;

            // 16 accumulators for 4×4 tile (accumulate in f32 for precision)
            float32x4_t acc00 = vdupq_n_f32(0.0f);
            float32x4_t acc01 = vdupq_n_f32(0.0f);
            float32x4_t acc02 = vdupq_n_f32(0.0f);
            float32x4_t acc03 = vdupq_n_f32(0.0f);
            float32x4_t acc10 = vdupq_n_f32(0.0f);
            float32x4_t acc11 = vdupq_n_f32(0.0f);
            float32x4_t acc12 = vdupq_n_f32(0.0f);
            float32x4_t acc13 = vdupq_n_f32(0.0f);
            float32x4_t acc20 = vdupq_n_f32(0.0f);
            float32x4_t acc21 = vdupq_n_f32(0.0f);
            float32x4_t acc22 = vdupq_n_f32(0.0f);
            float32x4_t acc23 = vdupq_n_f32(0.0f);
            float32x4_t acc30 = vdupq_n_f32(0.0f);
            float32x4_t acc31 = vdupq_n_f32(0.0f);
            float32x4_t acc32 = vdupq_n_f32(0.0f);
            float32x4_t acc33 = vdupq_n_f32(0.0f);

            // Process 4 f16 elements at a time, widening to f32
            long p = 0;
            for (; p + 4 <= k; p += 4) {
                // Load f16 and widen to f32
                float16x4_t a0_h = vld1_f16(a + (i + 0) * k + p);
                float16x4_t a1_h = vld1_f16(a + (i + 1) * k + p);
                float16x4_t a2_h = vld1_f16(a + (i + 2) * k + p);
                float16x4_t a3_h = vld1_f16(a + (i + 3) * k + p);

                float32x4_t a0 = vcvt_f32_f16(a0_h);
                float32x4_t a1 = vcvt_f32_f16(a1_h);
                float32x4_t a2 = vcvt_f32_f16(a2_h);
                float32x4_t a3 = vcvt_f32_f16(a3_h);

                float16x4_t b0_h = vld1_f16(b + (j + 0) * k + p);
                float16x4_t b1_h = vld1_f16(b + (j + 1) * k + p);
                float16x4_t b2_h = vld1_f16(b + (j + 2) * k + p);
                float16x4_t b3_h = vld1_f16(b + (j + 3) * k + p);

                float32x4_t b0 = vcvt_f32_f16(b0_h);
                float32x4_t b1 = vcvt_f32_f16(b1_h);
                float32x4_t b2 = vcvt_f32_f16(b2_h);
                float32x4_t b3 = vcvt_f32_f16(b3_h);

                // 16 FMAs in f32
                acc00 = vfmaq_f32(acc00, a0, b0);
                acc01 = vfmaq_f32(acc01, a0, b1);
                acc02 = vfmaq_f32(acc02, a0, b2);
                acc03 = vfmaq_f32(acc03, a0, b3);

                acc10 = vfmaq_f32(acc10, a1, b0);
                acc11 = vfmaq_f32(acc11, a1, b1);
                acc12 = vfmaq_f32(acc12, a1, b2);
                acc13 = vfmaq_f32(acc13, a1, b3);

                acc20 = vfmaq_f32(acc20, a2, b0);
                acc21 = vfmaq_f32(acc21, a2, b1);
                acc22 = vfmaq_f32(acc22, a2, b2);
                acc23 = vfmaq_f32(acc23, a2, b3);

                acc30 = vfmaq_f32(acc30, a3, b0);
                acc31 = vfmaq_f32(acc31, a3, b1);
                acc32 = vfmaq_f32(acc32, a3, b2);
                acc33 = vfmaq_f32(acc33, a3, b3);
            }

            // Horizontal sums
            float s00 = vaddvq_f32(acc00);
            float s01 = vaddvq_f32(acc01);
            float s02 = vaddvq_f32(acc02);
            float s03 = vaddvq_f32(acc03);
            float s10 = vaddvq_f32(acc10);
            float s11 = vaddvq_f32(acc11);
            float s12 = vaddvq_f32(acc12);
            float s13 = vaddvq_f32(acc13);
            float s20 = vaddvq_f32(acc20);
            float s21 = vaddvq_f32(acc21);
            float s22 = vaddvq_f32(acc22);
            float s23 = vaddvq_f32(acc23);
            float s30 = vaddvq_f32(acc30);
            float s31 = vaddvq_f32(acc31);
            float s32 = vaddvq_f32(acc32);
            float s33 = vaddvq_f32(acc33);

            // Scalar tail
            for (; p < k; p++) {
                float a0s = (float)a[(i + 0) * k + p];
                float a1s = (float)a[(i + 1) * k + p];
                float a2s = (float)a[(i + 2) * k + p];
                float a3s = (float)a[(i + 3) * k + p];

                float b0s = (float)b[(j + 0) * k + p];
                float b1s = (float)b[(j + 1) * k + p];
                float b2s = (float)b[(j + 2) * k + p];
                float b3s = (float)b[(j + 3) * k + p];

                s00 += a0s * b0s; s01 += a0s * b1s; s02 += a0s * b2s; s03 += a0s * b3s;
                s10 += a1s * b0s; s11 += a1s * b1s; s12 += a1s * b2s; s13 += a1s * b3s;
                s20 += a2s * b0s; s21 += a2s * b1s; s22 += a2s * b2s; s23 += a2s * b3s;
                s30 += a3s * b0s; s31 += a3s * b1s; s32 += a3s * b2s; s33 += a3s * b3s;
            }

            // Store results (convert back to f16)
            if (iCount > 0) {
                if (jCount > 0) c[(i + 0) * n + (j + 0)] = (__fp16)s00;
                if (jCount > 1) c[(i + 0) * n + (j + 1)] = (__fp16)s01;
                if (jCount > 2) c[(i + 0) * n + (j + 2)] = (__fp16)s02;
                if (jCount > 3) c[(i + 0) * n + (j + 3)] = (__fp16)s03;
            }
            if (iCount > 1) {
                if (jCount > 0) c[(i + 1) * n + (j + 0)] = (__fp16)s10;
                if (jCount > 1) c[(i + 1) * n + (j + 1)] = (__fp16)s11;
                if (jCount > 2) c[(i + 1) * n + (j + 2)] = (__fp16)s12;
                if (jCount > 3) c[(i + 1) * n + (j + 3)] = (__fp16)s13;
            }
            if (iCount > 2) {
                if (jCount > 0) c[(i + 2) * n + (j + 0)] = (__fp16)s20;
                if (jCount > 1) c[(i + 2) * n + (j + 1)] = (__fp16)s21;
                if (jCount > 2) c[(i + 2) * n + (j + 2)] = (__fp16)s22;
                if (jCount > 3) c[(i + 2) * n + (j + 3)] = (__fp16)s23;
            }
            if (iCount > 3) {
                if (jCount > 0) c[(i + 3) * n + (j + 0)] = (__fp16)s30;
                if (jCount > 1) c[(i + 3) * n + (j + 1)] = (__fp16)s31;
                if (jCount > 2) c[(i + 3) * n + (j + 2)] = (__fp16)s32;
                if (jCount > 3) c[(i + 3) * n + (j + 3)] = (__fp16)s33;
            }
        }
    }
}

// =============================================================================
// matmul_klast_neon_bf16: Tiled dot-product matmul for bfloat16
// =============================================================================
// Uses widening to f32 for computation (like f16 version)
// BFDOT is designed for matrix multiplication, not K-last dot products
// Accumulates in f32 for precision
//
// func matmul_klast_neon_bf16(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_klast_neon_bf16(__bf16 *a, __bf16 *b, __bf16 *c,
                             long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process 4×4 output tiles
    for (long i = 0; i < m; i += 4) {
        long iEnd = i + 4;
        if (iEnd > m) iEnd = m;
        long iCount = iEnd - i;

        for (long j = 0; j < n; j += 4) {
            long jEnd = j + 4;
            if (jEnd > n) jEnd = n;
            long jCount = jEnd - j;

            // 16 accumulators in f32
            float32x4_t acc00 = vdupq_n_f32(0.0f);
            float32x4_t acc01 = vdupq_n_f32(0.0f);
            float32x4_t acc02 = vdupq_n_f32(0.0f);
            float32x4_t acc03 = vdupq_n_f32(0.0f);
            float32x4_t acc10 = vdupq_n_f32(0.0f);
            float32x4_t acc11 = vdupq_n_f32(0.0f);
            float32x4_t acc12 = vdupq_n_f32(0.0f);
            float32x4_t acc13 = vdupq_n_f32(0.0f);
            float32x4_t acc20 = vdupq_n_f32(0.0f);
            float32x4_t acc21 = vdupq_n_f32(0.0f);
            float32x4_t acc22 = vdupq_n_f32(0.0f);
            float32x4_t acc23 = vdupq_n_f32(0.0f);
            float32x4_t acc30 = vdupq_n_f32(0.0f);
            float32x4_t acc31 = vdupq_n_f32(0.0f);
            float32x4_t acc32 = vdupq_n_f32(0.0f);
            float32x4_t acc33 = vdupq_n_f32(0.0f);

            // Process 4 bf16 elements at a time, widening to f32
            long p = 0;
            for (; p + 4 <= k; p += 4) {
                // Load bf16 and widen to f32 using vcvt_f32_bf16
                bfloat16x4_t a0_bf = vld1_bf16(a + (i + 0) * k + p);
                bfloat16x4_t a1_bf = vld1_bf16(a + (i + 1) * k + p);
                bfloat16x4_t a2_bf = vld1_bf16(a + (i + 2) * k + p);
                bfloat16x4_t a3_bf = vld1_bf16(a + (i + 3) * k + p);

                float32x4_t a0 = vcvt_f32_bf16(a0_bf);
                float32x4_t a1 = vcvt_f32_bf16(a1_bf);
                float32x4_t a2 = vcvt_f32_bf16(a2_bf);
                float32x4_t a3 = vcvt_f32_bf16(a3_bf);

                bfloat16x4_t b0_bf = vld1_bf16(b + (j + 0) * k + p);
                bfloat16x4_t b1_bf = vld1_bf16(b + (j + 1) * k + p);
                bfloat16x4_t b2_bf = vld1_bf16(b + (j + 2) * k + p);
                bfloat16x4_t b3_bf = vld1_bf16(b + (j + 3) * k + p);

                float32x4_t b0 = vcvt_f32_bf16(b0_bf);
                float32x4_t b1 = vcvt_f32_bf16(b1_bf);
                float32x4_t b2 = vcvt_f32_bf16(b2_bf);
                float32x4_t b3 = vcvt_f32_bf16(b3_bf);

                // 16 FMAs in f32
                acc00 = vfmaq_f32(acc00, a0, b0);
                acc01 = vfmaq_f32(acc01, a0, b1);
                acc02 = vfmaq_f32(acc02, a0, b2);
                acc03 = vfmaq_f32(acc03, a0, b3);

                acc10 = vfmaq_f32(acc10, a1, b0);
                acc11 = vfmaq_f32(acc11, a1, b1);
                acc12 = vfmaq_f32(acc12, a1, b2);
                acc13 = vfmaq_f32(acc13, a1, b3);

                acc20 = vfmaq_f32(acc20, a2, b0);
                acc21 = vfmaq_f32(acc21, a2, b1);
                acc22 = vfmaq_f32(acc22, a2, b2);
                acc23 = vfmaq_f32(acc23, a2, b3);

                acc30 = vfmaq_f32(acc30, a3, b0);
                acc31 = vfmaq_f32(acc31, a3, b1);
                acc32 = vfmaq_f32(acc32, a3, b2);
                acc33 = vfmaq_f32(acc33, a3, b3);
            }

            // Horizontal sums
            float s00 = vaddvq_f32(acc00);
            float s01 = vaddvq_f32(acc01);
            float s02 = vaddvq_f32(acc02);
            float s03 = vaddvq_f32(acc03);
            float s10 = vaddvq_f32(acc10);
            float s11 = vaddvq_f32(acc11);
            float s12 = vaddvq_f32(acc12);
            float s13 = vaddvq_f32(acc13);
            float s20 = vaddvq_f32(acc20);
            float s21 = vaddvq_f32(acc21);
            float s22 = vaddvq_f32(acc22);
            float s23 = vaddvq_f32(acc23);
            float s30 = vaddvq_f32(acc30);
            float s31 = vaddvq_f32(acc31);
            float s32 = vaddvq_f32(acc32);
            float s33 = vaddvq_f32(acc33);

            // Scalar tail
            for (; p < k; p++) {
                // bf16 to f32 conversion
                float a0s = vcvtah_f32_bf16(a[(i + 0) * k + p]);
                float a1s = vcvtah_f32_bf16(a[(i + 1) * k + p]);
                float a2s = vcvtah_f32_bf16(a[(i + 2) * k + p]);
                float a3s = vcvtah_f32_bf16(a[(i + 3) * k + p]);

                float b0s = vcvtah_f32_bf16(b[(j + 0) * k + p]);
                float b1s = vcvtah_f32_bf16(b[(j + 1) * k + p]);
                float b2s = vcvtah_f32_bf16(b[(j + 2) * k + p]);
                float b3s = vcvtah_f32_bf16(b[(j + 3) * k + p]);

                s00 += a0s * b0s; s01 += a0s * b1s; s02 += a0s * b2s; s03 += a0s * b3s;
                s10 += a1s * b0s; s11 += a1s * b1s; s12 += a1s * b2s; s13 += a1s * b3s;
                s20 += a2s * b0s; s21 += a2s * b1s; s22 += a2s * b2s; s23 += a2s * b3s;
                s30 += a3s * b0s; s31 += a3s * b1s; s32 += a3s * b2s; s33 += a3s * b3s;
            }

            // Store results (convert back to bf16)
            if (iCount > 0) {
                if (jCount > 0) c[(i + 0) * n + (j + 0)] = vcvth_bf16_f32(s00);
                if (jCount > 1) c[(i + 0) * n + (j + 1)] = vcvth_bf16_f32(s01);
                if (jCount > 2) c[(i + 0) * n + (j + 2)] = vcvth_bf16_f32(s02);
                if (jCount > 3) c[(i + 0) * n + (j + 3)] = vcvth_bf16_f32(s03);
            }
            if (iCount > 1) {
                if (jCount > 0) c[(i + 1) * n + (j + 0)] = vcvth_bf16_f32(s10);
                if (jCount > 1) c[(i + 1) * n + (j + 1)] = vcvth_bf16_f32(s11);
                if (jCount > 2) c[(i + 1) * n + (j + 2)] = vcvth_bf16_f32(s12);
                if (jCount > 3) c[(i + 1) * n + (j + 3)] = vcvth_bf16_f32(s13);
            }
            if (iCount > 2) {
                if (jCount > 0) c[(i + 2) * n + (j + 0)] = vcvth_bf16_f32(s20);
                if (jCount > 1) c[(i + 2) * n + (j + 1)] = vcvth_bf16_f32(s21);
                if (jCount > 2) c[(i + 2) * n + (j + 2)] = vcvth_bf16_f32(s22);
                if (jCount > 3) c[(i + 2) * n + (j + 3)] = vcvth_bf16_f32(s23);
            }
            if (iCount > 3) {
                if (jCount > 0) c[(i + 3) * n + (j + 0)] = vcvth_bf16_f32(s30);
                if (jCount > 1) c[(i + 3) * n + (j + 1)] = vcvth_bf16_f32(s31);
                if (jCount > 2) c[(i + 3) * n + (j + 2)] = vcvth_bf16_f32(s32);
                if (jCount > 3) c[(i + 3) * n + (j + 3)] = vcvth_bf16_f32(s33);
            }
        }
    }
}
