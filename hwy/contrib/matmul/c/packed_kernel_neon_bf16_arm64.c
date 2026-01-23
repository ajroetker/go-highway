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

// Packed GEBP Micro-Kernel for ARM NEON - BFloat16
// Requires ARMv8.6-A with BF16 extension
// Compile with: -march=armv8.6-a+bf16
//
// Uses f32 accumulation for precision, with BFMMLA or BFDOT instructions.

#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// =============================================================================
// BFloat16 Packed Micro-Kernel (Mr=4, Nr=8)
// =============================================================================
// For bfloat16, we use f32 accumulation (4 elements per vector)
// Input is bf16, output is bf16, accumulation in f32
//
// func packed_microkernel_neon_bf16(packedA, packedB, c unsafe.Pointer,
//                                    kc, n, mr, nr int64)
//
void packed_microkernel_neon_bf16(__bf16 *packedA, __bf16 *packedB, __bf16 *c,
                                   long *pkc, long *pn, long *pmr, long *pnr) {
    long kc = *pkc;
    long n = *pn;
    long mr = *pmr;
    long nr = *pnr;

    // Accumulators in f32: 4 rows × 2 vectors (8 columns)
    float32x4_t acc00, acc01;  // Row 0: columns 0-3, 4-7
    float32x4_t acc10, acc11;  // Row 1
    float32x4_t acc20, acc21;  // Row 2
    float32x4_t acc30, acc31;  // Row 3

    // Helper to convert bf16 to f32
    #define BF16_TO_F32(bf16_val) ({ \
        unsigned short bits; \
        __builtin_memcpy(&bits, &(bf16_val), sizeof(bits)); \
        unsigned int f32_bits = ((unsigned int)bits) << 16; \
        float result; \
        __builtin_memcpy(&result, &f32_bits, sizeof(result)); \
        result; \
    })

    // Load existing C values (convert bf16 to f32)
    // For simplicity, load via temporary arrays
    float c_f32[4][8];
    for (long i = 0; i < mr && i < 4; i++) {
        for (long j = 0; j < nr && j < 8; j++) {
            c_f32[i][j] = BF16_TO_F32(c[i*n + j]);
        }
        for (long j = nr; j < 8; j++) {
            c_f32[i][j] = 0.0f;
        }
    }
    for (long i = mr; i < 4; i++) {
        for (long j = 0; j < 8; j++) {
            c_f32[i][j] = 0.0f;
        }
    }

    acc00 = vld1q_f32(&c_f32[0][0]);
    acc01 = vld1q_f32(&c_f32[0][4]);
    acc10 = vld1q_f32(&c_f32[1][0]);
    acc11 = vld1q_f32(&c_f32[1][4]);
    acc20 = vld1q_f32(&c_f32[2][0]);
    acc21 = vld1q_f32(&c_f32[2][4]);
    acc30 = vld1q_f32(&c_f32[3][0]);
    acc31 = vld1q_f32(&c_f32[3][4]);

    // Main K-loop - process 2 K elements at a time for BFDOT
    long k = 0;
    for (; k + 2 <= kc; k += 2) {
        // Load packed A: 4 elements × 2 K values = 8 bf16
        // Layout: [k0: a0,a1,a2,a3, k1: a0,a1,a2,a3]
        bfloat16x8_t a_pair = vld1q_bf16(packedA + k * 4);

        // Load packed B: 8 elements × 2 K values = 16 bf16
        // But we process as pairs for BFDOT
        // B layout: [k0: b0..b7, k1: b0..b7]
        bfloat16x8_t b0_pair = vld1q_bf16(packedB + k * 8 + 0);  // k0: b0-3, k1: b0-3
        bfloat16x8_t b1_pair = vld1q_bf16(packedB + k * 8 + 8);  // k0: b4-7, k1: b4-7

        // For BFDOT, we need to reorganize:
        // BFDOT does: acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]
        // But our layout is different, so we use scalar approach for correctness

        // Extract individual values and do FMA
        // This is less optimal but correct
        for (int kk = 0; kk < 2; kk++) {
            long kidx = k + kk;
            // Get A values for this k
            float a0 = BF16_TO_F32(packedA[kidx * 4 + 0]);
            float a1 = BF16_TO_F32(packedA[kidx * 4 + 1]);
            float a2 = BF16_TO_F32(packedA[kidx * 4 + 2]);
            float a3 = BF16_TO_F32(packedA[kidx * 4 + 3]);

            // Load B as f32
            float b[8];
            for (int j = 0; j < 8; j++) {
                b[j] = BF16_TO_F32(packedB[kidx * 8 + j]);
            }
            float32x4_t b_lo = vld1q_f32(&b[0]);
            float32x4_t b_hi = vld1q_f32(&b[4]);

            // FMA for each row
            acc00 = vfmaq_n_f32(acc00, b_lo, a0);
            acc01 = vfmaq_n_f32(acc01, b_hi, a0);
            acc10 = vfmaq_n_f32(acc10, b_lo, a1);
            acc11 = vfmaq_n_f32(acc11, b_hi, a1);
            acc20 = vfmaq_n_f32(acc20, b_lo, a2);
            acc21 = vfmaq_n_f32(acc21, b_hi, a2);
            acc30 = vfmaq_n_f32(acc30, b_lo, a3);
            acc31 = vfmaq_n_f32(acc31, b_hi, a3);
        }
    }

    // Handle remaining K (if kc is odd)
    for (; k < kc; k++) {
        float a0 = BF16_TO_F32(packedA[k * 4 + 0]);
        float a1 = BF16_TO_F32(packedA[k * 4 + 1]);
        float a2 = BF16_TO_F32(packedA[k * 4 + 2]);
        float a3 = BF16_TO_F32(packedA[k * 4 + 3]);

        float b[8];
        for (int j = 0; j < 8; j++) {
            b[j] = BF16_TO_F32(packedB[k * 8 + j]);
        }
        float32x4_t b_lo = vld1q_f32(&b[0]);
        float32x4_t b_hi = vld1q_f32(&b[4]);

        acc00 = vfmaq_n_f32(acc00, b_lo, a0);
        acc01 = vfmaq_n_f32(acc01, b_hi, a0);
        acc10 = vfmaq_n_f32(acc10, b_lo, a1);
        acc11 = vfmaq_n_f32(acc11, b_hi, a1);
        acc20 = vfmaq_n_f32(acc20, b_lo, a2);
        acc21 = vfmaq_n_f32(acc21, b_hi, a2);
        acc30 = vfmaq_n_f32(acc30, b_lo, a3);
        acc31 = vfmaq_n_f32(acc31, b_hi, a3);
    }

    // Convert f32 accumulators back to bf16 and store
    // vcvt_bf16_f32 converts 4 f32 to 4 bf16
    bfloat16x4_t out00 = vcvt_bf16_f32(acc00);
    bfloat16x4_t out01 = vcvt_bf16_f32(acc01);
    bfloat16x4_t out10 = vcvt_bf16_f32(acc10);
    bfloat16x4_t out11 = vcvt_bf16_f32(acc11);
    bfloat16x4_t out20 = vcvt_bf16_f32(acc20);
    bfloat16x4_t out21 = vcvt_bf16_f32(acc21);
    bfloat16x4_t out30 = vcvt_bf16_f32(acc30);
    bfloat16x4_t out31 = vcvt_bf16_f32(acc31);

    // Store results (handle partial tiles)
    if (mr >= 1 && nr >= 4) vst1_bf16(c + 0*n + 0, out00);
    if (mr >= 1 && nr >= 8) vst1_bf16(c + 0*n + 4, out01);
    if (mr >= 2 && nr >= 4) vst1_bf16(c + 1*n + 0, out10);
    if (mr >= 2 && nr >= 8) vst1_bf16(c + 1*n + 4, out11);
    if (mr >= 3 && nr >= 4) vst1_bf16(c + 2*n + 0, out20);
    if (mr >= 3 && nr >= 8) vst1_bf16(c + 2*n + 4, out21);
    if (mr >= 4 && nr >= 4) vst1_bf16(c + 3*n + 0, out30);
    if (mr >= 4 && nr >= 8) vst1_bf16(c + 3*n + 4, out31);

    // Handle partial nr with scalar stores
    if (nr > 0 && nr < 4) {
        __bf16 o00[4], o10[4], o20[4], o30[4];
        vst1_bf16(o00, out00);
        vst1_bf16(o10, out10);
        vst1_bf16(o20, out20);
        vst1_bf16(o30, out30);
        for (long j = 0; j < nr; j++) {
            if (mr >= 1) c[0*n + j] = o00[j];
            if (mr >= 2) c[1*n + j] = o10[j];
            if (mr >= 3) c[2*n + j] = o20[j];
            if (mr >= 4) c[3*n + j] = o30[j];
        }
    }
    if (nr > 4 && nr < 8) {
        __bf16 o01[4], o11[4], o21[4], o31[4];
        vst1_bf16(o01, out01);
        vst1_bf16(o11, out11);
        vst1_bf16(o21, out21);
        vst1_bf16(o31, out31);
        for (long j = 4; j < nr; j++) {
            if (mr >= 1) c[0*n + j] = o01[j-4];
            if (mr >= 2) c[1*n + j] = o11[j-4];
            if (mr >= 3) c[2*n + j] = o21[j-4];
            if (mr >= 4) c[3*n + j] = o31[j-4];
        }
    }

    #undef BF16_TO_F32
}
