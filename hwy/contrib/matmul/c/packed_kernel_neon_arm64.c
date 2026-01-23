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

// Packed GEBP Micro-Kernel for ARM NEON
//
// These micro-kernels read from pre-packed A and B matrices and accumulate
// into C. The packing is done in Go, but the innermost compute kernel is
// optimized NEON assembly via GOAT.
//
// Packed memory layout (K-first for sequential access):
//   - Packed A: [Mr] elements per K-step, total Mr * Kc elements
//   - Packed B: [Nr] elements per K-step, total Kc * Nr elements
//
// Micro-kernel computes: C[Mr×Nr] += PackedA[Mr×Kc] * PackedB[Kc×Nr]

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// =============================================================================
// Float32 Packed Micro-Kernel (Mr=4, Nr=8)
// =============================================================================
// func packed_microkernel_neon_f32(packedA, packedB, c unsafe.Pointer,
//                                   kc, n, mr, nr int64)
//
// Computes C[mr×nr] += PackedA[mr×kc] * PackedB[kc×nr]
// where mr <= 4 (MR) and nr <= 8 (NR).
//
// PackedA layout: [kc][mr] - for each k, mr consecutive A elements
// PackedB layout: [kc][nr] - for each k, nr consecutive B elements
// C is row-major: C[i][j] at c + i*n + j
//
void packed_microkernel_neon_f32(float *packedA, float *packedB, float *c,
                                  long *pkc, long *pn, long *pmr, long *pnr) {
    long kc = *pkc;
    long n = *pn;
    long mr = *pmr;
    long nr = *pnr;

    // Accumulators: 4 rows × 2 vectors (8 columns)
    float32x4_t acc00, acc01;  // Row 0: columns 0-3, 4-7
    float32x4_t acc10, acc11;  // Row 1
    float32x4_t acc20, acc21;  // Row 2
    float32x4_t acc30, acc31;  // Row 3

    // Load existing C values into accumulators
    // Handle partial micro-tiles (mr < 4 or nr < 8)
    if (mr >= 1) {
        acc00 = (nr >= 4) ? vld1q_f32(c + 0*n + 0) : vdupq_n_f32(0);
        acc01 = (nr >= 8) ? vld1q_f32(c + 0*n + 4) : vdupq_n_f32(0);
    } else {
        acc00 = vdupq_n_f32(0);
        acc01 = vdupq_n_f32(0);
    }
    if (mr >= 2) {
        acc10 = (nr >= 4) ? vld1q_f32(c + 1*n + 0) : vdupq_n_f32(0);
        acc11 = (nr >= 8) ? vld1q_f32(c + 1*n + 4) : vdupq_n_f32(0);
    } else {
        acc10 = vdupq_n_f32(0);
        acc11 = vdupq_n_f32(0);
    }
    if (mr >= 3) {
        acc20 = (nr >= 4) ? vld1q_f32(c + 2*n + 0) : vdupq_n_f32(0);
        acc21 = (nr >= 8) ? vld1q_f32(c + 2*n + 4) : vdupq_n_f32(0);
    } else {
        acc20 = vdupq_n_f32(0);
        acc21 = vdupq_n_f32(0);
    }
    if (mr >= 4) {
        acc30 = (nr >= 4) ? vld1q_f32(c + 3*n + 0) : vdupq_n_f32(0);
        acc31 = (nr >= 8) ? vld1q_f32(c + 3*n + 4) : vdupq_n_f32(0);
    } else {
        acc30 = vdupq_n_f32(0);
        acc31 = vdupq_n_f32(0);
    }

    // Main K-loop
    for (long k = 0; k < kc; k++) {
        // Load packed A: mr elements at packedA[k*mr : k*mr + mr]
        // Sequential access - great for prefetching
        float32x4_t a_col = vld1q_f32(packedA + k * 4);

        // Load packed B: nr elements at packedB[k*nr : k*nr + nr]
        float32x4_t b0 = vld1q_f32(packedB + k * 8 + 0);
        float32x4_t b1 = vld1q_f32(packedB + k * 8 + 4);

        // Broadcast each A element and FMA with B row
        // Row 0: C[0,:] += A[0,k] * B[k,:]
        acc00 = vfmaq_laneq_f32(acc00, b0, a_col, 0);
        acc01 = vfmaq_laneq_f32(acc01, b1, a_col, 0);

        // Row 1: C[1,:] += A[1,k] * B[k,:]
        acc10 = vfmaq_laneq_f32(acc10, b0, a_col, 1);
        acc11 = vfmaq_laneq_f32(acc11, b1, a_col, 1);

        // Row 2: C[2,:] += A[2,k] * B[k,:]
        acc20 = vfmaq_laneq_f32(acc20, b0, a_col, 2);
        acc21 = vfmaq_laneq_f32(acc21, b1, a_col, 2);

        // Row 3: C[3,:] += A[3,k] * B[k,:]
        acc30 = vfmaq_laneq_f32(acc30, b0, a_col, 3);
        acc31 = vfmaq_laneq_f32(acc31, b1, a_col, 3);
    }

    // Store accumulators back to C (handle partial tiles)
    if (mr >= 1 && nr >= 4) vst1q_f32(c + 0*n + 0, acc00);
    if (mr >= 1 && nr >= 8) vst1q_f32(c + 0*n + 4, acc01);
    if (mr >= 2 && nr >= 4) vst1q_f32(c + 1*n + 0, acc10);
    if (mr >= 2 && nr >= 8) vst1q_f32(c + 1*n + 4, acc11);
    if (mr >= 3 && nr >= 4) vst1q_f32(c + 2*n + 0, acc20);
    if (mr >= 3 && nr >= 8) vst1q_f32(c + 2*n + 4, acc21);
    if (mr >= 4 && nr >= 4) vst1q_f32(c + 3*n + 0, acc30);
    if (mr >= 4 && nr >= 8) vst1q_f32(c + 3*n + 4, acc31);

    // Handle partial nr (1-3 or 5-7 columns) with scalar stores
    if (nr > 0 && nr < 4) {
        float acc0[4], acc1[4], acc2[4], acc3[4];
        vst1q_f32(acc0, acc00);
        vst1q_f32(acc1, acc10);
        vst1q_f32(acc2, acc20);
        vst1q_f32(acc3, acc30);
        for (long j = 0; j < nr; j++) {
            if (mr >= 1) c[0*n + j] = acc0[j];
            if (mr >= 2) c[1*n + j] = acc1[j];
            if (mr >= 3) c[2*n + j] = acc2[j];
            if (mr >= 4) c[3*n + j] = acc3[j];
        }
    }
    if (nr > 4 && nr < 8) {
        float acc0[4], acc1[4], acc2[4], acc3[4];
        vst1q_f32(acc0, acc01);
        vst1q_f32(acc1, acc11);
        vst1q_f32(acc2, acc21);
        vst1q_f32(acc3, acc31);
        for (long j = 4; j < nr; j++) {
            if (mr >= 1) c[0*n + j] = acc0[j-4];
            if (mr >= 2) c[1*n + j] = acc1[j-4];
            if (mr >= 3) c[2*n + j] = acc2[j-4];
            if (mr >= 4) c[3*n + j] = acc3[j-4];
        }
    }
}

// =============================================================================
// Float64 Packed Micro-Kernel (Mr=4, Nr=4)
// =============================================================================
// For float64, NEON vectors hold 2 elements, so Nr=4 (2 vectors)
//
void packed_microkernel_neon_f64(double *packedA, double *packedB, double *c,
                                  long *pkc, long *pn, long *pmr, long *pnr) {
    long kc = *pkc;
    long n = *pn;
    long mr = *pmr;
    long nr = *pnr;

    // Accumulators: 4 rows × 2 vectors (4 columns)
    float64x2_t acc00, acc01;  // Row 0: columns 0-1, 2-3
    float64x2_t acc10, acc11;  // Row 1
    float64x2_t acc20, acc21;  // Row 2
    float64x2_t acc30, acc31;  // Row 3

    // Load existing C values
    if (mr >= 1) {
        acc00 = (nr >= 2) ? vld1q_f64(c + 0*n + 0) : vdupq_n_f64(0);
        acc01 = (nr >= 4) ? vld1q_f64(c + 0*n + 2) : vdupq_n_f64(0);
    } else {
        acc00 = vdupq_n_f64(0);
        acc01 = vdupq_n_f64(0);
    }
    if (mr >= 2) {
        acc10 = (nr >= 2) ? vld1q_f64(c + 1*n + 0) : vdupq_n_f64(0);
        acc11 = (nr >= 4) ? vld1q_f64(c + 1*n + 2) : vdupq_n_f64(0);
    } else {
        acc10 = vdupq_n_f64(0);
        acc11 = vdupq_n_f64(0);
    }
    if (mr >= 3) {
        acc20 = (nr >= 2) ? vld1q_f64(c + 2*n + 0) : vdupq_n_f64(0);
        acc21 = (nr >= 4) ? vld1q_f64(c + 2*n + 2) : vdupq_n_f64(0);
    } else {
        acc20 = vdupq_n_f64(0);
        acc21 = vdupq_n_f64(0);
    }
    if (mr >= 4) {
        acc30 = (nr >= 2) ? vld1q_f64(c + 3*n + 0) : vdupq_n_f64(0);
        acc31 = (nr >= 4) ? vld1q_f64(c + 3*n + 2) : vdupq_n_f64(0);
    } else {
        acc30 = vdupq_n_f64(0);
        acc31 = vdupq_n_f64(0);
    }

    // Main K-loop
    for (long k = 0; k < kc; k++) {
        // Load packed A: 4 elements (but only mr valid)
        // For f64, load as 2 vectors of 2
        float64x2_t a01 = vld1q_f64(packedA + k * 4 + 0);
        float64x2_t a23 = vld1q_f64(packedA + k * 4 + 2);

        // Load packed B: nr elements (up to 4)
        float64x2_t b0 = vld1q_f64(packedB + k * 4 + 0);
        float64x2_t b1 = vld1q_f64(packedB + k * 4 + 2);

        // Row 0: C[0,:] += A[0,k] * B[k,:]
        acc00 = vfmaq_laneq_f64(acc00, b0, a01, 0);
        acc01 = vfmaq_laneq_f64(acc01, b1, a01, 0);

        // Row 1: C[1,:] += A[1,k] * B[k,:]
        acc10 = vfmaq_laneq_f64(acc10, b0, a01, 1);
        acc11 = vfmaq_laneq_f64(acc11, b1, a01, 1);

        // Row 2: C[2,:] += A[2,k] * B[k,:]
        acc20 = vfmaq_laneq_f64(acc20, b0, a23, 0);
        acc21 = vfmaq_laneq_f64(acc21, b1, a23, 0);

        // Row 3: C[3,:] += A[3,k] * B[k,:]
        acc30 = vfmaq_laneq_f64(acc30, b0, a23, 1);
        acc31 = vfmaq_laneq_f64(acc31, b1, a23, 1);
    }

    // Store accumulators back to C
    if (mr >= 1 && nr >= 2) vst1q_f64(c + 0*n + 0, acc00);
    if (mr >= 1 && nr >= 4) vst1q_f64(c + 0*n + 2, acc01);
    if (mr >= 2 && nr >= 2) vst1q_f64(c + 1*n + 0, acc10);
    if (mr >= 2 && nr >= 4) vst1q_f64(c + 1*n + 2, acc11);
    if (mr >= 3 && nr >= 2) vst1q_f64(c + 2*n + 0, acc20);
    if (mr >= 3 && nr >= 4) vst1q_f64(c + 2*n + 2, acc21);
    if (mr >= 4 && nr >= 2) vst1q_f64(c + 3*n + 0, acc30);
    if (mr >= 4 && nr >= 4) vst1q_f64(c + 3*n + 2, acc31);

    // Handle partial nr with scalar stores
    if (nr == 1) {
        if (mr >= 1) c[0*n + 0] = vgetq_lane_f64(acc00, 0);
        if (mr >= 2) c[1*n + 0] = vgetq_lane_f64(acc10, 0);
        if (mr >= 3) c[2*n + 0] = vgetq_lane_f64(acc20, 0);
        if (mr >= 4) c[3*n + 0] = vgetq_lane_f64(acc30, 0);
    }
    if (nr == 3) {
        if (mr >= 1) c[0*n + 2] = vgetq_lane_f64(acc01, 0);
        if (mr >= 2) c[1*n + 2] = vgetq_lane_f64(acc11, 0);
        if (mr >= 3) c[2*n + 2] = vgetq_lane_f64(acc21, 0);
        if (mr >= 4) c[3*n + 2] = vgetq_lane_f64(acc31, 0);
    }
}
