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

// SME Transpose for Apple Silicon M4+
// Key insight: Load rows into ZA columns (vertical), store ZA rows (horizontal).
// The matrix tile handles the data reorganization at memory bandwidth speed.
// Compile with: -march=armv9-a+sme+sme-f64f64+sme-f16f16

#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

// SME tile size depends on SVL (Streaming Vector Length)
// Apple M4: SVL = 512 bits = 16 float32 = 8 float64 = 32 float16

// ============================================================================
// SME 16x16 float32 transpose
// ============================================================================
// Load rows into ZA tile columns, store tile rows to output
// This achieves transpose "for free" via the matrix coprocessor

// func transpose_sme_f32(src, dst unsafe.Pointer, m, k *int64)
void transpose_sme_f32(const float *src, float *dst, long *pm, long *pk) __arm_streaming __arm_out("za") {
    long m = *pm;
    long k = *pk;

    long blockM = (m / 16) * 16;
    long blockK = (k / 16) * 16;

    // Process 16x16 tiles with SME
    for (long i = 0; i < blockM; i += 16) {
        for (long j = 0; j < blockK; j += 16) {
            // Zero the ZA tile
            svzero_za();

            svbool_t pg = svptrue_b32();  // Predicate for 16 f32 elements

            // Load 16 source rows into ZA tile columns (vertical writes)
            // ZA column c gets source row c
            svfloat32_t row0 = svld1_f32(pg, src + (i+0)*k + j);
            svfloat32_t row1 = svld1_f32(pg, src + (i+1)*k + j);
            svfloat32_t row2 = svld1_f32(pg, src + (i+2)*k + j);
            svfloat32_t row3 = svld1_f32(pg, src + (i+3)*k + j);
            svfloat32_t row4 = svld1_f32(pg, src + (i+4)*k + j);
            svfloat32_t row5 = svld1_f32(pg, src + (i+5)*k + j);
            svfloat32_t row6 = svld1_f32(pg, src + (i+6)*k + j);
            svfloat32_t row7 = svld1_f32(pg, src + (i+7)*k + j);
            svfloat32_t row8 = svld1_f32(pg, src + (i+8)*k + j);
            svfloat32_t row9 = svld1_f32(pg, src + (i+9)*k + j);
            svfloat32_t row10 = svld1_f32(pg, src + (i+10)*k + j);
            svfloat32_t row11 = svld1_f32(pg, src + (i+11)*k + j);
            svfloat32_t row12 = svld1_f32(pg, src + (i+12)*k + j);
            svfloat32_t row13 = svld1_f32(pg, src + (i+13)*k + j);
            svfloat32_t row14 = svld1_f32(pg, src + (i+14)*k + j);
            svfloat32_t row15 = svld1_f32(pg, src + (i+15)*k + j);

            svwrite_ver_za32_f32_m(0, 0, pg, row0);
            svwrite_ver_za32_f32_m(0, 1, pg, row1);
            svwrite_ver_za32_f32_m(0, 2, pg, row2);
            svwrite_ver_za32_f32_m(0, 3, pg, row3);
            svwrite_ver_za32_f32_m(0, 4, pg, row4);
            svwrite_ver_za32_f32_m(0, 5, pg, row5);
            svwrite_ver_za32_f32_m(0, 6, pg, row6);
            svwrite_ver_za32_f32_m(0, 7, pg, row7);
            svwrite_ver_za32_f32_m(0, 8, pg, row8);
            svwrite_ver_za32_f32_m(0, 9, pg, row9);
            svwrite_ver_za32_f32_m(0, 10, pg, row10);
            svwrite_ver_za32_f32_m(0, 11, pg, row11);
            svwrite_ver_za32_f32_m(0, 12, pg, row12);
            svwrite_ver_za32_f32_m(0, 13, pg, row13);
            svwrite_ver_za32_f32_m(0, 14, pg, row14);
            svwrite_ver_za32_f32_m(0, 15, pg, row15);

            // Store ZA tile rows (horizontal reads) to destination
            // ZA row r is column r of source = row r of transposed output
            svfloat32_t col0 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 0);
            svfloat32_t col1 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 1);
            svfloat32_t col2 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 2);
            svfloat32_t col3 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 3);
            svfloat32_t col4 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 4);
            svfloat32_t col5 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 5);
            svfloat32_t col6 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 6);
            svfloat32_t col7 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 7);
            svfloat32_t col8 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 8);
            svfloat32_t col9 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 9);
            svfloat32_t col10 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 10);
            svfloat32_t col11 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 11);
            svfloat32_t col12 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 12);
            svfloat32_t col13 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 13);
            svfloat32_t col14 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 14);
            svfloat32_t col15 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, 15);

            svst1_f32(pg, dst + (j+0)*m + i, col0);
            svst1_f32(pg, dst + (j+1)*m + i, col1);
            svst1_f32(pg, dst + (j+2)*m + i, col2);
            svst1_f32(pg, dst + (j+3)*m + i, col3);
            svst1_f32(pg, dst + (j+4)*m + i, col4);
            svst1_f32(pg, dst + (j+5)*m + i, col5);
            svst1_f32(pg, dst + (j+6)*m + i, col6);
            svst1_f32(pg, dst + (j+7)*m + i, col7);
            svst1_f32(pg, dst + (j+8)*m + i, col8);
            svst1_f32(pg, dst + (j+9)*m + i, col9);
            svst1_f32(pg, dst + (j+10)*m + i, col10);
            svst1_f32(pg, dst + (j+11)*m + i, col11);
            svst1_f32(pg, dst + (j+12)*m + i, col12);
            svst1_f32(pg, dst + (j+13)*m + i, col13);
            svst1_f32(pg, dst + (j+14)*m + i, col14);
            svst1_f32(pg, dst + (j+15)*m + i, col15);
        }
    }

    // Fall back to scalar for edges (SME streaming mode overhead not worth it)
    // Right edge
    for (long ii = 0; ii < m; ii++) {
        for (long jj = blockK; jj < k; jj++) {
            dst[jj*m + ii] = src[ii*k + jj];
        }
    }
    // Bottom edge
    for (long ii = blockM; ii < m; ii++) {
        for (long jj = 0; jj < blockK; jj++) {
            dst[jj*m + ii] = src[ii*k + jj];
        }
    }
}

// ============================================================================
// SME 8x8 float64 transpose
// ============================================================================
// func transpose_sme_f64(src, dst unsafe.Pointer, m, k *int64)
void transpose_sme_f64(const double *src, double *dst, long *pm, long *pk) __arm_streaming __arm_out("za") {
    long m = *pm;
    long k = *pk;

    long blockM = (m / 8) * 8;
    long blockK = (k / 8) * 8;

    for (long i = 0; i < blockM; i += 8) {
        for (long j = 0; j < blockK; j += 8) {
            svzero_za();

            svbool_t pg = svptrue_b64();  // Predicate for 8 f64 elements

            // Load 8 source rows into ZA tile columns
            svfloat64_t row0 = svld1_f64(pg, src + (i+0)*k + j);
            svfloat64_t row1 = svld1_f64(pg, src + (i+1)*k + j);
            svfloat64_t row2 = svld1_f64(pg, src + (i+2)*k + j);
            svfloat64_t row3 = svld1_f64(pg, src + (i+3)*k + j);
            svfloat64_t row4 = svld1_f64(pg, src + (i+4)*k + j);
            svfloat64_t row5 = svld1_f64(pg, src + (i+5)*k + j);
            svfloat64_t row6 = svld1_f64(pg, src + (i+6)*k + j);
            svfloat64_t row7 = svld1_f64(pg, src + (i+7)*k + j);

            svwrite_ver_za64_f64_m(0, 0, pg, row0);
            svwrite_ver_za64_f64_m(0, 1, pg, row1);
            svwrite_ver_za64_f64_m(0, 2, pg, row2);
            svwrite_ver_za64_f64_m(0, 3, pg, row3);
            svwrite_ver_za64_f64_m(0, 4, pg, row4);
            svwrite_ver_za64_f64_m(0, 5, pg, row5);
            svwrite_ver_za64_f64_m(0, 6, pg, row6);
            svwrite_ver_za64_f64_m(0, 7, pg, row7);

            // Store ZA tile rows
            svfloat64_t col0 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, 0);
            svfloat64_t col1 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, 1);
            svfloat64_t col2 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, 2);
            svfloat64_t col3 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, 3);
            svfloat64_t col4 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, 4);
            svfloat64_t col5 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, 5);
            svfloat64_t col6 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, 6);
            svfloat64_t col7 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, 7);

            svst1_f64(pg, dst + (j+0)*m + i, col0);
            svst1_f64(pg, dst + (j+1)*m + i, col1);
            svst1_f64(pg, dst + (j+2)*m + i, col2);
            svst1_f64(pg, dst + (j+3)*m + i, col3);
            svst1_f64(pg, dst + (j+4)*m + i, col4);
            svst1_f64(pg, dst + (j+5)*m + i, col5);
            svst1_f64(pg, dst + (j+6)*m + i, col6);
            svst1_f64(pg, dst + (j+7)*m + i, col7);
        }
    }

    // Edges
    for (long ii = 0; ii < m; ii++) {
        for (long jj = blockK; jj < k; jj++) {
            dst[jj*m + ii] = src[ii*k + jj];
        }
    }
    for (long ii = blockM; ii < m; ii++) {
        for (long jj = 0; jj < blockK; jj++) {
            dst[jj*m + ii] = src[ii*k + jj];
        }
    }
}

// ============================================================================
// SME 32x32 float16 transpose (SVL=512 means 32 f16 per vector)
// ============================================================================
// func transpose_sme_f16(src, dst unsafe.Pointer, m, k *int64)
void transpose_sme_f16(const __fp16 *src, __fp16 *dst, long *pm, long *pk) __arm_streaming __arm_out("za") {
    long m = *pm;
    long k = *pk;

    long blockM = (m / 32) * 32;
    long blockK = (k / 32) * 32;

    for (long i = 0; i < blockM; i += 32) {
        for (long j = 0; j < blockK; j += 32) {
            svzero_za();

            svbool_t pg = svptrue_b16();  // 32 f16 elements

            // Load 32 source rows into ZA tile columns
            for (long r = 0; r < 32; r++) {
                svfloat16_t row = svld1_f16(pg, src + (i+r)*k + j);
                svwrite_ver_za16_f16_m(0, r, pg, row);
            }

            // Store 32 ZA tile rows
            for (long c = 0; c < 32; c++) {
                svfloat16_t col = svread_hor_za16_f16_m(svundef_f16(), pg, 0, c);
                svst1_f16(pg, dst + (j+c)*m + i, col);
            }
        }
    }

    // Edges
    for (long ii = 0; ii < m; ii++) {
        for (long jj = blockK; jj < k; jj++) {
            dst[jj*m + ii] = src[ii*k + jj];
        }
    }
    for (long ii = blockM; ii < m; ii++) {
        for (long jj = 0; jj < blockK; jj++) {
            dst[jj*m + ii] = src[ii*k + jj];
        }
    }
}

// BFloat16
// func transpose_sme_bf16(src, dst unsafe.Pointer, m, k *int64)
void transpose_sme_bf16(void *src, void *dst, long *pm, long *pk) __arm_streaming __arm_out("za") {
    transpose_sme_f16((const __fp16*)src, (__fp16*)dst, pm, pk);
}
