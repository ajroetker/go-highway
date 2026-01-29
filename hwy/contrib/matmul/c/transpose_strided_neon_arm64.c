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

// NEON Strided Transpose for ARM64
// Transposes rows [rowStart, rowEnd) with dstM as the destination stride.
// This enables parallel transpose by processing row strips independently.
// Compile with: -march=armv8.2-a+fp16

#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// ============================================================================
// 4x4 float32 strided transpose
// ============================================================================
// func transpose_strided_neon_f32(src, dst unsafe.Pointer, rowStart, rowEnd, k, dstM *int64)
void transpose_strided_neon_f32(const float *src, float *dst,
                                 long *pRowStart, long *pRowEnd, long *pk, long *pDstM) {
    long rowStart = *pRowStart;
    long rowEnd = *pRowEnd;
    long k = *pk;
    long dstM = *pDstM;

    // Round rowStart up to 4-aligned, rowEnd down to 4-aligned for SIMD blocks
    long blockRowStart = ((rowStart + 3) / 4) * 4;
    long blockRowEnd = (rowEnd / 4) * 4;
    long blockK = (k / 4) * 4;

    // Process 4x4 blocks
    for (long i = blockRowStart; i < blockRowEnd; i += 4) {
        for (long j = 0; j < blockK; j += 4) {
            // Load 4 rows
            float32x4_t r0 = vld1q_f32(src + i*k + j);
            float32x4_t r1 = vld1q_f32(src + (i+1)*k + j);
            float32x4_t r2 = vld1q_f32(src + (i+2)*k + j);
            float32x4_t r3 = vld1q_f32(src + (i+3)*k + j);

            // Level 1: transpose pairs of 32-bit elements
            float32x4_t t0 = vtrn1q_f32(r0, r1);
            float32x4_t t1 = vtrn2q_f32(r0, r1);
            float32x4_t t2 = vtrn1q_f32(r2, r3);
            float32x4_t t3 = vtrn2q_f32(r2, r3);

            // Level 2: transpose pairs of 64-bit elements
            float32x4_t d0 = vreinterpretq_f32_f64(vtrn1q_f64(
                vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
            float32x4_t d1 = vreinterpretq_f32_f64(vtrn1q_f64(
                vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
            float32x4_t d2 = vreinterpretq_f32_f64(vtrn2q_f64(
                vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
            float32x4_t d3 = vreinterpretq_f32_f64(vtrn2q_f64(
                vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));

            // Store with dstM stride
            vst1q_f32(dst + j*dstM + i, d0);
            vst1q_f32(dst + (j+1)*dstM + i, d1);
            vst1q_f32(dst + (j+2)*dstM + i, d2);
            vst1q_f32(dst + (j+3)*dstM + i, d3);
        }
    }

    // Top edge: rows [rowStart, blockRowStart) not covered by SIMD
    for (long i = rowStart; i < blockRowStart; i++) {
        for (long j = 0; j < blockK; j++) {
            dst[j*dstM + i] = src[i*k + j];
        }
    }

    // Bottom edge: rows [blockRowEnd, rowEnd) not covered by SIMD
    for (long i = blockRowEnd; i < rowEnd; i++) {
        for (long j = 0; j < blockK; j++) {
            dst[j*dstM + i] = src[i*k + j];
        }
    }

    // Right edge: columns [blockK, k) for all rows in range
    for (long i = rowStart; i < rowEnd; i++) {
        for (long j = blockK; j < k; j++) {
            dst[j*dstM + i] = src[i*k + j];
        }
    }
}

// ============================================================================
// 2x2 float64 strided transpose
// ============================================================================
// func transpose_strided_neon_f64(src, dst unsafe.Pointer, rowStart, rowEnd, k, dstM *int64)
void transpose_strided_neon_f64(const double *src, double *dst,
                                 long *pRowStart, long *pRowEnd, long *pk, long *pDstM) {
    long rowStart = *pRowStart;
    long rowEnd = *pRowEnd;
    long k = *pk;
    long dstM = *pDstM;

    long blockRowStart = ((rowStart + 1) / 2) * 2;
    long blockRowEnd = (rowEnd / 2) * 2;
    long blockK = (k / 2) * 2;

    for (long i = blockRowStart; i < blockRowEnd; i += 2) {
        for (long j = 0; j < blockK; j += 2) {
            float64x2_t r0 = vld1q_f64(src + i*k + j);
            float64x2_t r1 = vld1q_f64(src + (i+1)*k + j);

            float64x2_t d0 = vtrn1q_f64(r0, r1);
            float64x2_t d1 = vtrn2q_f64(r0, r1);

            vst1q_f64(dst + j*dstM + i, d0);
            vst1q_f64(dst + (j+1)*dstM + i, d1);
        }
    }

    // Edges (scalar)
    for (long i = rowStart; i < blockRowStart; i++) {
        for (long j = 0; j < blockK; j++) {
            dst[j*dstM + i] = src[i*k + j];
        }
    }
    for (long i = blockRowEnd; i < rowEnd; i++) {
        for (long j = 0; j < blockK; j++) {
            dst[j*dstM + i] = src[i*k + j];
        }
    }
    for (long i = rowStart; i < rowEnd; i++) {
        for (long j = blockK; j < k; j++) {
            dst[j*dstM + i] = src[i*k + j];
        }
    }
}

// ============================================================================
// 8x8 float16 strided transpose
// ============================================================================
// func transpose_strided_neon_f16(src, dst unsafe.Pointer, rowStart, rowEnd, k, dstM *int64)
void transpose_strided_neon_f16(__fp16 *src, __fp16 *dst,
                                 long *pRowStart, long *pRowEnd, long *pk, long *pDstM) {
    long rowStart = *pRowStart;
    long rowEnd = *pRowEnd;
    long k = *pk;
    long dstM = *pDstM;

    long blockRowStart = ((rowStart + 7) / 8) * 8;
    long blockRowEnd = (rowEnd / 8) * 8;
    long blockK = (k / 8) * 8;

    for (long i = blockRowStart; i < blockRowEnd; i += 8) {
        for (long j = 0; j < blockK; j += 8) {
            // Load 8 rows
            float16x8_t r0 = vld1q_f16(src + i*k + j);
            float16x8_t r1 = vld1q_f16(src + (i+1)*k + j);
            float16x8_t r2 = vld1q_f16(src + (i+2)*k + j);
            float16x8_t r3 = vld1q_f16(src + (i+3)*k + j);
            float16x8_t r4 = vld1q_f16(src + (i+4)*k + j);
            float16x8_t r5 = vld1q_f16(src + (i+5)*k + j);
            float16x8_t r6 = vld1q_f16(src + (i+6)*k + j);
            float16x8_t r7 = vld1q_f16(src + (i+7)*k + j);

            // Level 1: 16-bit interleave
            float16x8_t t0 = vtrn1q_f16(r0, r1);
            float16x8_t t1 = vtrn2q_f16(r0, r1);
            float16x8_t t2 = vtrn1q_f16(r2, r3);
            float16x8_t t3 = vtrn2q_f16(r2, r3);
            float16x8_t t4 = vtrn1q_f16(r4, r5);
            float16x8_t t5 = vtrn2q_f16(r4, r5);
            float16x8_t t6 = vtrn1q_f16(r6, r7);
            float16x8_t t7 = vtrn2q_f16(r6, r7);

            // Level 2: 32-bit interleave
            float32x4_t s0 = vtrn1q_f32(vreinterpretq_f32_f16(t0), vreinterpretq_f32_f16(t2));
            float32x4_t s1 = vtrn2q_f32(vreinterpretq_f32_f16(t0), vreinterpretq_f32_f16(t2));
            float32x4_t s2 = vtrn1q_f32(vreinterpretq_f32_f16(t1), vreinterpretq_f32_f16(t3));
            float32x4_t s3 = vtrn2q_f32(vreinterpretq_f32_f16(t1), vreinterpretq_f32_f16(t3));
            float32x4_t s4 = vtrn1q_f32(vreinterpretq_f32_f16(t4), vreinterpretq_f32_f16(t6));
            float32x4_t s5 = vtrn2q_f32(vreinterpretq_f32_f16(t4), vreinterpretq_f32_f16(t6));
            float32x4_t s6 = vtrn1q_f32(vreinterpretq_f32_f16(t5), vreinterpretq_f32_f16(t7));
            float32x4_t s7 = vtrn2q_f32(vreinterpretq_f32_f16(t5), vreinterpretq_f32_f16(t7));

            // Level 3: 64-bit interleave
            float16x8_t d0 = vreinterpretq_f16_f64(vtrn1q_f64(vreinterpretq_f64_f32(s0), vreinterpretq_f64_f32(s4)));
            float16x8_t d1 = vreinterpretq_f16_f64(vtrn1q_f64(vreinterpretq_f64_f32(s2), vreinterpretq_f64_f32(s6)));
            float16x8_t d2 = vreinterpretq_f16_f64(vtrn1q_f64(vreinterpretq_f64_f32(s1), vreinterpretq_f64_f32(s5)));
            float16x8_t d3 = vreinterpretq_f16_f64(vtrn1q_f64(vreinterpretq_f64_f32(s3), vreinterpretq_f64_f32(s7)));
            float16x8_t d4 = vreinterpretq_f16_f64(vtrn2q_f64(vreinterpretq_f64_f32(s0), vreinterpretq_f64_f32(s4)));
            float16x8_t d5 = vreinterpretq_f16_f64(vtrn2q_f64(vreinterpretq_f64_f32(s2), vreinterpretq_f64_f32(s6)));
            float16x8_t d6 = vreinterpretq_f16_f64(vtrn2q_f64(vreinterpretq_f64_f32(s1), vreinterpretq_f64_f32(s5)));
            float16x8_t d7 = vreinterpretq_f16_f64(vtrn2q_f64(vreinterpretq_f64_f32(s3), vreinterpretq_f64_f32(s7)));

            // Store with dstM stride
            vst1q_f16(dst + j*dstM + i, d0);
            vst1q_f16(dst + (j+1)*dstM + i, d1);
            vst1q_f16(dst + (j+2)*dstM + i, d2);
            vst1q_f16(dst + (j+3)*dstM + i, d3);
            vst1q_f16(dst + (j+4)*dstM + i, d4);
            vst1q_f16(dst + (j+5)*dstM + i, d5);
            vst1q_f16(dst + (j+6)*dstM + i, d6);
            vst1q_f16(dst + (j+7)*dstM + i, d7);
        }
    }

    // Edges (scalar)
    for (long i = rowStart; i < blockRowStart; i++) {
        for (long j = 0; j < blockK; j++) {
            dst[j*dstM + i] = src[i*k + j];
        }
    }
    for (long i = blockRowEnd; i < rowEnd; i++) {
        for (long j = 0; j < blockK; j++) {
            dst[j*dstM + i] = src[i*k + j];
        }
    }
    for (long i = rowStart; i < rowEnd; i++) {
        for (long j = blockK; j < k; j++) {
            dst[j*dstM + i] = src[i*k + j];
        }
    }
}

// BFloat16 uses same 8x8 pattern
// func transpose_strided_neon_bf16(src, dst unsafe.Pointer, rowStart, rowEnd, k, dstM *int64)
void transpose_strided_neon_bf16(void *src, void *dst,
                                  long *pRowStart, long *pRowEnd, long *pk, long *pDstM) {
    transpose_strided_neon_f16((__fp16*)src, (__fp16*)dst, pRowStart, pRowEnd, pk, pDstM);
}
