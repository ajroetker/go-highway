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

// Packed GEBP Micro-Kernel for ARM NEON - Float16 (FP16)
// Requires ARMv8.2-A with FP16 extension
// Compile with: -march=armv8.2-a+fp16

#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// =============================================================================
// Float16 Packed Micro-Kernel (Mr=4, Nr=16)
// =============================================================================
// For float16, NEON vectors hold 8 elements, so Nr=16 (2 vectors)
//
// func packed_microkernel_neon_f16(packedA, packedB, c unsafe.Pointer,
//                                   kc, n, mr, nr int64)
//
void packed_microkernel_neon_f16(__fp16 *packedA, __fp16 *packedB, __fp16 *c,
                                  long *pkc, long *pn, long *pmr, long *pnr) {
    long kc = *pkc;
    long n = *pn;
    long mr = *pmr;
    long nr = *pnr;

    // Accumulators: 4 rows × 2 vectors (16 columns)
    float16x8_t acc00, acc01;  // Row 0: columns 0-7, 8-15
    float16x8_t acc10, acc11;  // Row 1
    float16x8_t acc20, acc21;  // Row 2
    float16x8_t acc30, acc31;  // Row 3

    // Load existing C values
    __fp16 zero = (__fp16)0.0f;
    if (mr >= 1) {
        acc00 = (nr >= 8)  ? vld1q_f16(c + 0*n + 0) : vdupq_n_f16(zero);
        acc01 = (nr >= 16) ? vld1q_f16(c + 0*n + 8) : vdupq_n_f16(zero);
    } else {
        acc00 = vdupq_n_f16(zero);
        acc01 = vdupq_n_f16(zero);
    }
    if (mr >= 2) {
        acc10 = (nr >= 8)  ? vld1q_f16(c + 1*n + 0) : vdupq_n_f16(zero);
        acc11 = (nr >= 16) ? vld1q_f16(c + 1*n + 8) : vdupq_n_f16(zero);
    } else {
        acc10 = vdupq_n_f16(zero);
        acc11 = vdupq_n_f16(zero);
    }
    if (mr >= 3) {
        acc20 = (nr >= 8)  ? vld1q_f16(c + 2*n + 0) : vdupq_n_f16(zero);
        acc21 = (nr >= 16) ? vld1q_f16(c + 2*n + 8) : vdupq_n_f16(zero);
    } else {
        acc20 = vdupq_n_f16(zero);
        acc21 = vdupq_n_f16(zero);
    }
    if (mr >= 4) {
        acc30 = (nr >= 8)  ? vld1q_f16(c + 3*n + 0) : vdupq_n_f16(zero);
        acc31 = (nr >= 16) ? vld1q_f16(c + 3*n + 8) : vdupq_n_f16(zero);
    } else {
        acc30 = vdupq_n_f16(zero);
        acc31 = vdupq_n_f16(zero);
    }

    // Main K-loop
    for (long k = 0; k < kc; k++) {
        // Load packed A: 4 elements (mr valid)
        // Use float16x4_t for 4 elements
        float16x4_t a_col = vld1_f16(packedA + k * 4);

        // Load packed B: nr elements (up to 16)
        float16x8_t b0 = vld1q_f16(packedB + k * 16 + 0);
        float16x8_t b1 = vld1q_f16(packedB + k * 16 + 8);

        // Row 0: C[0,:] += A[0,k] * B[k,:]
        acc00 = vfmaq_lane_f16(acc00, b0, a_col, 0);
        acc01 = vfmaq_lane_f16(acc01, b1, a_col, 0);

        // Row 1: C[1,:] += A[1,k] * B[k,:]
        acc10 = vfmaq_lane_f16(acc10, b0, a_col, 1);
        acc11 = vfmaq_lane_f16(acc11, b1, a_col, 1);

        // Row 2: C[2,:] += A[2,k] * B[k,:]
        acc20 = vfmaq_lane_f16(acc20, b0, a_col, 2);
        acc21 = vfmaq_lane_f16(acc21, b1, a_col, 2);

        // Row 3: C[3,:] += A[3,k] * B[k,:]
        acc30 = vfmaq_lane_f16(acc30, b0, a_col, 3);
        acc31 = vfmaq_lane_f16(acc31, b1, a_col, 3);
    }

    // Store accumulators back to C
    if (mr >= 1 && nr >= 8)  vst1q_f16(c + 0*n + 0, acc00);
    if (mr >= 1 && nr >= 16) vst1q_f16(c + 0*n + 8, acc01);
    if (mr >= 2 && nr >= 8)  vst1q_f16(c + 1*n + 0, acc10);
    if (mr >= 2 && nr >= 16) vst1q_f16(c + 1*n + 8, acc11);
    if (mr >= 3 && nr >= 8)  vst1q_f16(c + 2*n + 0, acc20);
    if (mr >= 3 && nr >= 16) vst1q_f16(c + 2*n + 8, acc21);
    if (mr >= 4 && nr >= 8)  vst1q_f16(c + 3*n + 0, acc30);
    if (mr >= 4 && nr >= 16) vst1q_f16(c + 3*n + 8, acc31);

    // Handle partial nr (1-7 or 9-15) with scalar stores
    if (nr > 0 && nr < 8) {
        __fp16 acc0[8], acc1[8], acc2[8], acc3[8];
        vst1q_f16(acc0, acc00);
        vst1q_f16(acc1, acc10);
        vst1q_f16(acc2, acc20);
        vst1q_f16(acc3, acc30);
        for (long j = 0; j < nr; j++) {
            if (mr >= 1) c[0*n + j] = acc0[j];
            if (mr >= 2) c[1*n + j] = acc1[j];
            if (mr >= 3) c[2*n + j] = acc2[j];
            if (mr >= 4) c[3*n + j] = acc3[j];
        }
    }
    if (nr > 8 && nr < 16) {
        __fp16 acc0[8], acc1[8], acc2[8], acc3[8];
        vst1q_f16(acc0, acc01);
        vst1q_f16(acc1, acc11);
        vst1q_f16(acc2, acc21);
        vst1q_f16(acc3, acc31);
        for (long j = 8; j < nr; j++) {
            if (mr >= 1) c[0*n + j] = acc0[j-8];
            if (mr >= 2) c[1*n + j] = acc1[j-8];
            if (mr >= 3) c[2*n + j] = acc2[j-8];
            if (mr >= 4) c[3*n + j] = acc3[j-8];
        }
    }
}
