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

// Multi-Tile SME FMOPA Matrix Multiplication for go-highway
// Compile with: -march=armv9-a+sme+sme-f64f64+sme-f16f16+bf16
//
// Uses all 4 ZA tiles (ZA0-ZA3) in a 2x2 arrangement to process 32x32
// output blocks (f32) or 16x16 output blocks (f64).
//
// 2x2 tile layout (f32, 32x32 output block):
//                 cols 0-15    cols 16-31
//   rows 0-15:      ZA0          ZA2
//   rows 16-31:     ZA1          ZA3
//
// Per K iteration: load a0, a1 (2 A cols) + b0, b1 (2 B rows), then 4 FMOPAs.
// Ratio: 1.0 FMOPA/load (vs 0.5 for single-tile).
//
// Within each cache block's i-loop: process 32-row chunks with 4-tile,
// fall back to single-tile for 16-row remainder. Same for N dimension.
//
// IMPORTANT: Requires M and N to be multiples of 16 (Go handles padding).
// M and N do NOT need to be multiples of 32 -- the kernel handles the
// 16-row and 16-col remainders internally with single-tile fallback.

#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

#define BLOCK_SIZE 48

// =============================================================================
// multitile_fmopa_at_f32: Multi-tile blocked FMOPA matmul (float32)
// =============================================================================
// Computes C = AT^T * B where:
//   AT is K x M (A transposed, row-major)
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// func multitile_fmopa_at_f32(at, b, c unsafe.Pointer, m, n, k int64)
void multitile_fmopa_at_f32(float *at, float *b, float *c,
                             long *pm, long *pn, long *pk)
    __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    svbool_t pg = svptrue_b32();

    for (long i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
        long iEnd = i0 + BLOCK_SIZE;
        if (iEnd > m) {
            iEnd = m;
        }

        for (long j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            long jEnd = j0 + BLOCK_SIZE;
            if (jEnd > n) {
                jEnd = n;
            }

            // Process 32x32 chunks with 4-tile FMOPA
            long ti = i0;
            for (; ti + 32 <= iEnd; ti += 32) {
                long tj = j0;
                for (; tj + 32 <= jEnd; tj += 32) {
                    // 2x2 tile: ZA0(0-15,0-15) ZA2(0-15,16-31)
                    //            ZA1(16-31,0-15) ZA3(16-31,16-31)
                    svzero_za();

                    for (long kk = 0; kk < k; kk++) {
                        // Use svld1_vnum to hint consecutive VL-offset loads,
                        // enabling the compiler to emit contiguous group loads
                        // (LD1W {z0.s-z1.s}) for ~5-10% throughput improvement.
                        float *a_base = at + kk * m + ti;
                        svfloat32_t a0 = svld1_vnum_f32(pg, a_base, 0);
                        svfloat32_t a1 = svld1_vnum_f32(pg, a_base, 1);
                        float *b_base = b + kk * n + tj;
                        svfloat32_t b0 = svld1_vnum_f32(pg, b_base, 0);
                        svfloat32_t b1 = svld1_vnum_f32(pg, b_base, 1);

                        svmopa_za32_f32_m(0, pg, pg, a0, b0);
                        svmopa_za32_f32_m(1, pg, pg, a1, b0);
                        svmopa_za32_f32_m(2, pg, pg, a0, b1);
                        svmopa_za32_f32_m(3, pg, pg, a1, b1);
                    }

                    // Store using svst1_vnum for consecutive tile pairs
                    for (int row = 0; row < 16; row++) {
                        float *c_row = c + (ti + row) * n + tj;
                        svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, row);
                        svst1_vnum_f32(pg, c_row, 0, r0);
                        svfloat32_t r2 = svread_hor_za32_f32_m(svundef_f32(), pg, 2, row);
                        svst1_vnum_f32(pg, c_row, 1, r2);
                    }
                    for (int row = 0; row < 16; row++) {
                        float *c_row = c + (ti + 16 + row) * n + tj;
                        svfloat32_t r1 = svread_hor_za32_f32_m(svundef_f32(), pg, 1, row);
                        svst1_vnum_f32(pg, c_row, 0, r1);
                        svfloat32_t r3 = svread_hor_za32_f32_m(svundef_f32(), pg, 3, row);
                        svst1_vnum_f32(pg, c_row, 1, r3);
                    }
                }

                // N remainder: 16-col strip with single tile (ZA0)
                if (tj < jEnd) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat32_t a0 = svld1_f32(pg, at + kk * m + ti);
                        svfloat32_t b0 = svld1_f32(pg, b + kk * n + tj);
                        svmopa_za32_f32_m(0, pg, pg, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, row);
                        svst1_f32(pg, c + (ti + row) * n + tj, r0);
                    }

                    // Second row block of the N remainder
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat32_t a1 = svld1_f32(pg, at + kk * m + ti + 16);
                        svfloat32_t b0 = svld1_f32(pg, b + kk * n + tj);
                        svmopa_za32_f32_m(0, pg, pg, a1, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, row);
                        svst1_f32(pg, c + (ti + 16 + row) * n + tj, r0);
                    }
                }
            }

            // M remainder: 16-row strip with single tile
            if (ti < iEnd) {
                for (long tj = j0; tj < jEnd; tj += 16) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat32_t a0 = svld1_f32(pg, at + kk * m + ti);
                        svfloat32_t b0 = svld1_f32(pg, b + kk * n + tj);
                        svmopa_za32_f32_m(0, pg, pg, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, row);
                        svst1_f32(pg, c + (ti + row) * n + tj, r0);
                    }
                }
            }
        }
    }
}

// =============================================================================
// multitile_fmopa_at_f32_strided: Same as above but with separate ldc for C
// =============================================================================
// Computes C = AT^T * B where C has leading dimension ldc (row stride).
// B has leading dimension n (row stride). This enables writing output strips
// directly into a larger output matrix without scatter copies.
//
// func multitile_fmopa_at_f32_strided(at, b, c unsafe.Pointer, m, n, k, ldc, coff int64)
void multitile_fmopa_at_f32_strided(float *at, float *b, float *c,
                                     long *pm, long *pn, long *pk,
                                     long *pldc, long *pcoff)
    __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;
    long ldc = *pldc;
    long coff = *pcoff;

    svbool_t pg = svptrue_b32();

    for (long i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
        long iEnd = i0 + BLOCK_SIZE;
        if (iEnd > m) {
            iEnd = m;
        }

        for (long j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            long jEnd = j0 + BLOCK_SIZE;
            if (jEnd > n) {
                jEnd = n;
            }

            long ti = i0;
            for (; ti + 32 <= iEnd; ti += 32) {
                long tj = j0;
                for (; tj + 32 <= jEnd; tj += 32) {
                    svzero_za();

                    for (long kk = 0; kk < k; kk++) {
                        float *a_base = at + kk * m + ti;
                        svfloat32_t a0 = svld1_vnum_f32(pg, a_base, 0);
                        svfloat32_t a1 = svld1_vnum_f32(pg, a_base, 1);
                        float *b_base = b + kk * n + tj;
                        svfloat32_t b0 = svld1_vnum_f32(pg, b_base, 0);
                        svfloat32_t b1 = svld1_vnum_f32(pg, b_base, 1);

                        svmopa_za32_f32_m(0, pg, pg, a0, b0);
                        svmopa_za32_f32_m(1, pg, pg, a1, b0);
                        svmopa_za32_f32_m(2, pg, pg, a0, b1);
                        svmopa_za32_f32_m(3, pg, pg, a1, b1);
                    }

                    for (int row = 0; row < 16; row++) {
                        float *c_row = c + (ti + row) * ldc + coff + tj;
                        svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, row);
                        svst1_vnum_f32(pg, c_row, 0, r0);
                        svfloat32_t r2 = svread_hor_za32_f32_m(svundef_f32(), pg, 2, row);
                        svst1_vnum_f32(pg, c_row, 1, r2);
                    }
                    for (int row = 0; row < 16; row++) {
                        float *c_row = c + (ti + 16 + row) * ldc + coff + tj;
                        svfloat32_t r1 = svread_hor_za32_f32_m(svundef_f32(), pg, 1, row);
                        svst1_vnum_f32(pg, c_row, 0, r1);
                        svfloat32_t r3 = svread_hor_za32_f32_m(svundef_f32(), pg, 3, row);
                        svst1_vnum_f32(pg, c_row, 1, r3);
                    }
                }

                if (tj < jEnd) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat32_t a0 = svld1_f32(pg, at + kk * m + ti);
                        svfloat32_t b0 = svld1_f32(pg, b + kk * n + tj);
                        svmopa_za32_f32_m(0, pg, pg, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, row);
                        svst1_f32(pg, c + (ti + row) * ldc + coff + tj, r0);
                    }

                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat32_t a1 = svld1_f32(pg, at + kk * m + ti + 16);
                        svfloat32_t b0 = svld1_f32(pg, b + kk * n + tj);
                        svmopa_za32_f32_m(0, pg, pg, a1, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, row);
                        svst1_f32(pg, c + (ti + 16 + row) * ldc + coff + tj, r0);
                    }
                }
            }

            if (ti < iEnd) {
                for (long tj = j0; tj < jEnd; tj += 16) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat32_t a0 = svld1_f32(pg, at + kk * m + ti);
                        svfloat32_t b0 = svld1_f32(pg, b + kk * n + tj);
                        svmopa_za32_f32_m(0, pg, pg, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg, 0, row);
                        svst1_f32(pg, c + (ti + row) * ldc + coff + tj, r0);
                    }
                }
            }
        }
    }
}

// =============================================================================
// multitile_fmopa_at_f64_strided: Same as f64 but with separate ldc for C
// =============================================================================
//
// func multitile_fmopa_at_f64_strided(at, b, c unsafe.Pointer, m, n, k, ldc, coff int64)
void multitile_fmopa_at_f64_strided(double *at, double *b, double *c,
                                     long *pm, long *pn, long *pk,
                                     long *pldc, long *pcoff)
    __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;
    long ldc = *pldc;
    long coff = *pcoff;

    svbool_t pg = svptrue_b64();

    for (long i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
        long iEnd = i0 + BLOCK_SIZE;
        if (iEnd > m) {
            iEnd = m;
        }

        for (long j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            long jEnd = j0 + BLOCK_SIZE;
            if (jEnd > n) {
                jEnd = n;
            }

            long ti = i0;
            for (; ti + 16 <= iEnd; ti += 16) {
                long tj = j0;
                for (; tj + 16 <= jEnd; tj += 16) {
                    svzero_za();

                    for (long kk = 0; kk < k; kk++) {
                        double *a_base = at + kk * m + ti;
                        svfloat64_t a0 = svld1_vnum_f64(pg, a_base, 0);
                        svfloat64_t a1 = svld1_vnum_f64(pg, a_base, 1);
                        double *b_base = b + kk * n + tj;
                        svfloat64_t b0 = svld1_vnum_f64(pg, b_base, 0);
                        svfloat64_t b1 = svld1_vnum_f64(pg, b_base, 1);

                        svmopa_za64_f64_m(0, pg, pg, a0, b0);
                        svmopa_za64_f64_m(1, pg, pg, a1, b0);
                        svmopa_za64_f64_m(2, pg, pg, a0, b1);
                        svmopa_za64_f64_m(3, pg, pg, a1, b1);
                    }

                    for (int row = 0; row < 8; row++) {
                        double *c_row = c + (ti + row) * ldc + coff + tj;
                        svfloat64_t r0 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, row);
                        svst1_vnum_f64(pg, c_row, 0, r0);
                        svfloat64_t r2 = svread_hor_za64_f64_m(svundef_f64(), pg, 2, row);
                        svst1_vnum_f64(pg, c_row, 1, r2);
                    }
                    for (int row = 0; row < 8; row++) {
                        double *c_row = c + (ti + 8 + row) * ldc + coff + tj;
                        svfloat64_t r1 = svread_hor_za64_f64_m(svundef_f64(), pg, 1, row);
                        svst1_vnum_f64(pg, c_row, 0, r1);
                        svfloat64_t r3 = svread_hor_za64_f64_m(svundef_f64(), pg, 3, row);
                        svst1_vnum_f64(pg, c_row, 1, r3);
                    }
                }

                if (tj < jEnd) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat64_t a0 = svld1_f64(pg, at + kk * m + ti);
                        svfloat64_t b0 = svld1_f64(pg, b + kk * n + tj);
                        svmopa_za64_f64_m(0, pg, pg, a0, b0);
                    }
                    for (int row = 0; row < 8; row++) {
                        svfloat64_t r0 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, row);
                        svst1_f64(pg, c + (ti + row) * ldc + coff + tj, r0);
                    }

                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat64_t a1 = svld1_f64(pg, at + kk * m + ti + 8);
                        svfloat64_t b0 = svld1_f64(pg, b + kk * n + tj);
                        svmopa_za64_f64_m(0, pg, pg, a1, b0);
                    }
                    for (int row = 0; row < 8; row++) {
                        svfloat64_t r0 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, row);
                        svst1_f64(pg, c + (ti + 8 + row) * ldc + coff + tj, r0);
                    }
                }
            }

            if (ti < iEnd) {
                for (long tj = j0; tj < jEnd; tj += 8) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat64_t a0 = svld1_f64(pg, at + kk * m + ti);
                        svfloat64_t b0 = svld1_f64(pg, b + kk * n + tj);
                        svmopa_za64_f64_m(0, pg, pg, a0, b0);
                    }
                    for (int row = 0; row < 8; row++) {
                        svfloat64_t r0 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, row);
                        svst1_f64(pg, c + (ti + row) * ldc + coff + tj, r0);
                    }
                }
            }
        }
    }
}

// =============================================================================
// multitile_fmopa_at_f64: Multi-tile blocked FMOPA matmul (float64)
// =============================================================================
// Same algorithm with 8×8 tiles per ZA, so 2x2 = 16×16 output block.
// Requires M, N to be multiples of 8.
//
// func multitile_fmopa_at_f64(at, b, c unsafe.Pointer, m, n, k int64)
void multitile_fmopa_at_f64(double *at, double *b, double *c,
                             long *pm, long *pn, long *pk)
    __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    svbool_t pg = svptrue_b64();

    for (long i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
        long iEnd = i0 + BLOCK_SIZE;
        if (iEnd > m) {
            iEnd = m;
        }

        for (long j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            long jEnd = j0 + BLOCK_SIZE;
            if (jEnd > n) {
                jEnd = n;
            }

            // Process 16x16 chunks with 4-tile FMOPA (8x8 per tile)
            long ti = i0;
            for (; ti + 16 <= iEnd; ti += 16) {
                long tj = j0;
                for (; tj + 16 <= jEnd; tj += 16) {
                    svzero_za();

                    for (long kk = 0; kk < k; kk++) {
                        double *a_base = at + kk * m + ti;
                        svfloat64_t a0 = svld1_vnum_f64(pg, a_base, 0);
                        svfloat64_t a1 = svld1_vnum_f64(pg, a_base, 1);
                        double *b_base = b + kk * n + tj;
                        svfloat64_t b0 = svld1_vnum_f64(pg, b_base, 0);
                        svfloat64_t b1 = svld1_vnum_f64(pg, b_base, 1);

                        svmopa_za64_f64_m(0, pg, pg, a0, b0);
                        svmopa_za64_f64_m(1, pg, pg, a1, b0);
                        svmopa_za64_f64_m(2, pg, pg, a0, b1);
                        svmopa_za64_f64_m(3, pg, pg, a1, b1);
                    }

                    // Store using svst1_vnum for consecutive tile pairs
                    for (int row = 0; row < 8; row++) {
                        double *c_row = c + (ti + row) * n + tj;
                        svfloat64_t r0 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, row);
                        svst1_vnum_f64(pg, c_row, 0, r0);
                        svfloat64_t r2 = svread_hor_za64_f64_m(svundef_f64(), pg, 2, row);
                        svst1_vnum_f64(pg, c_row, 1, r2);
                    }
                    for (int row = 0; row < 8; row++) {
                        double *c_row = c + (ti + 8 + row) * n + tj;
                        svfloat64_t r1 = svread_hor_za64_f64_m(svundef_f64(), pg, 1, row);
                        svst1_vnum_f64(pg, c_row, 0, r1);
                        svfloat64_t r3 = svread_hor_za64_f64_m(svundef_f64(), pg, 3, row);
                        svst1_vnum_f64(pg, c_row, 1, r3);
                    }
                }

                // N remainder: 8-col strip with single tile
                if (tj < jEnd) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat64_t a0 = svld1_f64(pg, at + kk * m + ti);
                        svfloat64_t b0 = svld1_f64(pg, b + kk * n + tj);
                        svmopa_za64_f64_m(0, pg, pg, a0, b0);
                    }
                    for (int row = 0; row < 8; row++) {
                        svfloat64_t r0 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, row);
                        svst1_f64(pg, c + (ti + row) * n + tj, r0);
                    }

                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat64_t a1 = svld1_f64(pg, at + kk * m + ti + 8);
                        svfloat64_t b0 = svld1_f64(pg, b + kk * n + tj);
                        svmopa_za64_f64_m(0, pg, pg, a1, b0);
                    }
                    for (int row = 0; row < 8; row++) {
                        svfloat64_t r0 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, row);
                        svst1_f64(pg, c + (ti + 8 + row) * n + tj, r0);
                    }
                }
            }

            // M remainder: 8-row strip with single tile
            if (ti < iEnd) {
                for (long tj = j0; tj < jEnd; tj += 8) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svfloat64_t a0 = svld1_f64(pg, at + kk * m + ti);
                        svfloat64_t b0 = svld1_f64(pg, b + kk * n + tj);
                        svmopa_za64_f64_m(0, pg, pg, a0, b0);
                    }
                    for (int row = 0; row < 8; row++) {
                        svfloat64_t r0 = svread_hor_za64_f64_m(svundef_f64(), pg, 0, row);
                        svst1_f64(pg, c + (ti + row) * n + tj, r0);
                    }
                }
            }
        }
    }
}

// =============================================================================
// multitile_fmopa_at_f16: Multi-tile FMOPA matmul for float16
// =============================================================================
// Uses widening approach: f16 -> f32 -> FMOPA -> f32 -> f16
// 2x2 tile layout (32x32 output blocks via f32 accumulator):
//   ZA0(0-15,0-15) ZA2(0-15,16-31)
//   ZA1(16-31,0-15) ZA3(16-31,16-31)
//
// scratch: unused (kept for API compatibility)
//
// func multitile_fmopa_at_f16(at, b, c unsafe.Pointer, m, n, k int64, scratch unsafe.Pointer)
void multitile_fmopa_at_f16(__fp16 *at, __fp16 *b, __fp16 *c,
                             long *pm, long *pn, long *pk,
                             float *scratch)
    __arm_streaming __arm_out("za") {
    (void)scratch;
    long m = *pm;
    long n = *pn;
    long k = *pk;

    svbool_t pg32 = svptrue_b32();
    svbool_t pg16 = svptrue_pat_b16(SV_VL16);
    svuint32_t exp_adjust = svdup_n_u32(112 << 23);

    for (long i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
        long iEnd = i0 + BLOCK_SIZE;
        if (iEnd > m) {
            iEnd = m;
        }

        for (long j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            long jEnd = j0 + BLOCK_SIZE;
            if (jEnd > n) {
                jEnd = n;
            }

            // Process 32x32 chunks with 4-tile FMOPA
            long ti = i0;
            for (; ti + 32 <= iEnd; ti += 32) {
                long tj = j0;
                for (; tj + 32 <= jEnd; tj += 32) {
                    svzero_za();

                    for (long kk = 0; kk < k; kk++) {
                        // Load and widen A columns (2x16 f16 -> 2x16 f32)
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 13);
                        a0_u32 = svadd_u32_x(pg32, a0_u32, exp_adjust);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t a1_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti + 16));
                        svuint32_t a1_u32 = svunpklo_u32(a1_u16);
                        a1_u32 = svlsl_n_u32_x(pg32, a1_u32, 13);
                        a1_u32 = svadd_u32_x(pg32, a1_u32, exp_adjust);
                        svfloat32_t a1 = svreinterpret_f32_u32(a1_u32);

                        // Load and widen B rows (2x16 f16 -> 2x16 f32)
                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 13);
                        b0_u32 = svadd_u32_x(pg32, b0_u32, exp_adjust);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svuint16_t b1_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj + 16));
                        svuint32_t b1_u32 = svunpklo_u32(b1_u16);
                        b1_u32 = svlsl_n_u32_x(pg32, b1_u32, 13);
                        b1_u32 = svadd_u32_x(pg32, b1_u32, exp_adjust);
                        svfloat32_t b1 = svreinterpret_f32_u32(b1_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                        svmopa_za32_f32_m(1, pg32, pg32, a1, b0);
                        svmopa_za32_f32_m(2, pg32, pg32, a0, b1);
                        svmopa_za32_f32_m(3, pg32, pg32, a1, b1);
                    }

                    // Store ZA0: rows 0-15, cols 0-15 (f32 -> f16)
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * n + tj), bits);
                    }
                    // Store ZA2: rows 0-15, cols 16-31
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * n + tj + 16), bits);
                    }
                    // Store ZA1: rows 16-31, cols 0-15
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * n + tj), bits);
                    }
                    // Store ZA3: rows 16-31, cols 16-31
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * n + tj + 16), bits);
                    }
                }

                // N remainder: 16-col strip with single tile (ZA0)
                if (tj < jEnd) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 13);
                        a0_u32 = svadd_u32_x(pg32, a0_u32, exp_adjust);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 13);
                        b0_u32 = svadd_u32_x(pg32, b0_u32, exp_adjust);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * n + tj), bits);
                    }

                    // Second row block of the N remainder
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a1_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti + 16));
                        svuint32_t a1_u32 = svunpklo_u32(a1_u16);
                        a1_u32 = svlsl_n_u32_x(pg32, a1_u32, 13);
                        a1_u32 = svadd_u32_x(pg32, a1_u32, exp_adjust);
                        svfloat32_t a1 = svreinterpret_f32_u32(a1_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 13);
                        b0_u32 = svadd_u32_x(pg32, b0_u32, exp_adjust);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a1, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * n + tj), bits);
                    }
                }
            }

            // M remainder: 16-row strip with single tile
            if (ti < iEnd) {
                for (long tj = j0; tj < jEnd; tj += 16) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 13);
                        a0_u32 = svadd_u32_x(pg32, a0_u32, exp_adjust);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 13);
                        b0_u32 = svadd_u32_x(pg32, b0_u32, exp_adjust);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * n + tj), bits);
                    }
                }
            }
        }
    }
}

// =============================================================================
// multitile_fmopa_at_f16_strided: Strided multi-tile F16 FMOPA matmul
// =============================================================================
// Same as multitile_fmopa_at_f16 but writes to C with leading dimension ldc
// at column offset coff.
//
// func multitile_fmopa_at_f16_strided(at, b, c, pm, pn, pk, pldc, pcoff, scratch unsafe.Pointer)
void multitile_fmopa_at_f16_strided(__fp16 *at, __fp16 *b, __fp16 *c,
                                     long *pm, long *pn, long *pk,
                                     long *pldc, long *pcoff,
                                     float *scratch)
    __arm_streaming __arm_out("za") {
    (void)scratch;
    long m = *pm;
    long n = *pn;
    long k = *pk;
    long ldc = *pldc;
    long coff = *pcoff;

    svbool_t pg32 = svptrue_b32();
    svbool_t pg16 = svptrue_pat_b16(SV_VL16);
    svuint32_t exp_adjust = svdup_n_u32(112 << 23);

    for (long i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
        long iEnd = i0 + BLOCK_SIZE;
        if (iEnd > m) {
            iEnd = m;
        }

        for (long j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            long jEnd = j0 + BLOCK_SIZE;
            if (jEnd > n) {
                jEnd = n;
            }

            long ti = i0;
            for (; ti + 32 <= iEnd; ti += 32) {
                long tj = j0;
                for (; tj + 32 <= jEnd; tj += 32) {
                    svzero_za();

                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 13);
                        a0_u32 = svadd_u32_x(pg32, a0_u32, exp_adjust);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t a1_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti + 16));
                        svuint32_t a1_u32 = svunpklo_u32(a1_u16);
                        a1_u32 = svlsl_n_u32_x(pg32, a1_u32, 13);
                        a1_u32 = svadd_u32_x(pg32, a1_u32, exp_adjust);
                        svfloat32_t a1 = svreinterpret_f32_u32(a1_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 13);
                        b0_u32 = svadd_u32_x(pg32, b0_u32, exp_adjust);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svuint16_t b1_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj + 16));
                        svuint32_t b1_u32 = svunpklo_u32(b1_u16);
                        b1_u32 = svlsl_n_u32_x(pg32, b1_u32, 13);
                        b1_u32 = svadd_u32_x(pg32, b1_u32, exp_adjust);
                        svfloat32_t b1 = svreinterpret_f32_u32(b1_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                        svmopa_za32_f32_m(1, pg32, pg32, a1, b0);
                        svmopa_za32_f32_m(2, pg32, pg32, a0, b1);
                        svmopa_za32_f32_m(3, pg32, pg32, a1, b1);
                    }

                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * ldc + coff + tj), bits);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * ldc + coff + tj + 16), bits);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * ldc + coff + tj), bits);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * ldc + coff + tj + 16), bits);
                    }
                }

                if (tj < jEnd) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 13);
                        a0_u32 = svadd_u32_x(pg32, a0_u32, exp_adjust);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 13);
                        b0_u32 = svadd_u32_x(pg32, b0_u32, exp_adjust);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * ldc + coff + tj), bits);
                    }

                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a1_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti + 16));
                        svuint32_t a1_u32 = svunpklo_u32(a1_u16);
                        a1_u32 = svlsl_n_u32_x(pg32, a1_u32, 13);
                        a1_u32 = svadd_u32_x(pg32, a1_u32, exp_adjust);
                        svfloat32_t a1 = svreinterpret_f32_u32(a1_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 13);
                        b0_u32 = svadd_u32_x(pg32, b0_u32, exp_adjust);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a1, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * ldc + coff + tj), bits);
                    }
                }
            }

            if (ti < iEnd) {
                for (long tj = j0; tj < jEnd; tj += 16) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 13);
                        a0_u32 = svadd_u32_x(pg32, a0_u32, exp_adjust);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 13);
                        b0_u32 = svadd_u32_x(pg32, b0_u32, exp_adjust);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        bits = svsub_u32_x(pg32, bits, exp_adjust);
                        svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                        round_bit = svand_n_u32_x(pg32, round_bit, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 13);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * ldc + coff + tj), bits);
                    }
                }
            }
        }
    }
}

// =============================================================================
// multitile_bfmopa_at_bf16: Multi-tile FMOPA matmul for bfloat16
// =============================================================================
// Uses widening approach: bf16 -> f32 -> FMOPA -> f32 -> bf16
// BF16 is simply the upper 16 bits of F32:
//   bf16→f32: shift left 16
//   f32→bf16: round-to-nearest-even, shift right 16
//
// scratch: unused (kept for API compatibility)
//
// func multitile_bfmopa_at_bf16(at, b, c unsafe.Pointer, m, n, k int64, scratch unsafe.Pointer)
void multitile_bfmopa_at_bf16(__bf16 *at, __bf16 *b, __bf16 *c,
                               long *pm, long *pn, long *pk,
                               float *scratch)
    __arm_streaming __arm_out("za") {
    (void)scratch;
    long m = *pm;
    long n = *pn;
    long k = *pk;

    svbool_t pg32 = svptrue_b32();
    svbool_t pg16 = svptrue_pat_b16(SV_VL16);

    for (long i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
        long iEnd = i0 + BLOCK_SIZE;
        if (iEnd > m) {
            iEnd = m;
        }

        for (long j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            long jEnd = j0 + BLOCK_SIZE;
            if (jEnd > n) {
                jEnd = n;
            }

            long ti = i0;
            for (; ti + 32 <= iEnd; ti += 32) {
                long tj = j0;
                for (; tj + 32 <= jEnd; tj += 32) {
                    svzero_za();

                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 16);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t a1_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti + 16));
                        svuint32_t a1_u32 = svunpklo_u32(a1_u16);
                        a1_u32 = svlsl_n_u32_x(pg32, a1_u32, 16);
                        svfloat32_t a1 = svreinterpret_f32_u32(a1_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 16);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svuint16_t b1_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj + 16));
                        svuint32_t b1_u32 = svunpklo_u32(b1_u16);
                        b1_u32 = svlsl_n_u32_x(pg32, b1_u32, 16);
                        svfloat32_t b1 = svreinterpret_f32_u32(b1_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                        svmopa_za32_f32_m(1, pg32, pg32, a1, b0);
                        svmopa_za32_f32_m(2, pg32, pg32, a0, b1);
                        svmopa_za32_f32_m(3, pg32, pg32, a1, b1);
                    }

                    // Store ZA0-ZA3 with f32->bf16 conversion
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * n + tj), bits);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * n + tj + 16), bits);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * n + tj), bits);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * n + tj + 16), bits);
                    }
                }

                // N remainder
                if (tj < jEnd) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 16);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 16);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * n + tj), bits);
                    }

                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a1_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti + 16));
                        svuint32_t a1_u32 = svunpklo_u32(a1_u16);
                        a1_u32 = svlsl_n_u32_x(pg32, a1_u32, 16);
                        svfloat32_t a1 = svreinterpret_f32_u32(a1_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 16);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a1, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * n + tj), bits);
                    }
                }
            }

            // M remainder
            if (ti < iEnd) {
                for (long tj = j0; tj < jEnd; tj += 16) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 16);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 16);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * n + tj), bits);
                    }
                }
            }
        }
    }
}

// =============================================================================
// multitile_bfmopa_at_bf16_strided: Strided multi-tile BF16 FMOPA matmul
// =============================================================================
//
// func multitile_bfmopa_at_bf16_strided(at, b, c, pm, pn, pk, pldc, pcoff, scratch unsafe.Pointer)
void multitile_bfmopa_at_bf16_strided(__bf16 *at, __bf16 *b, __bf16 *c,
                                       long *pm, long *pn, long *pk,
                                       long *pldc, long *pcoff,
                                       float *scratch)
    __arm_streaming __arm_out("za") {
    (void)scratch;
    long m = *pm;
    long n = *pn;
    long k = *pk;
    long ldc = *pldc;
    long coff = *pcoff;

    svbool_t pg32 = svptrue_b32();
    svbool_t pg16 = svptrue_pat_b16(SV_VL16);

    for (long i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
        long iEnd = i0 + BLOCK_SIZE;
        if (iEnd > m) {
            iEnd = m;
        }

        for (long j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            long jEnd = j0 + BLOCK_SIZE;
            if (jEnd > n) {
                jEnd = n;
            }

            long ti = i0;
            for (; ti + 32 <= iEnd; ti += 32) {
                long tj = j0;
                for (; tj + 32 <= jEnd; tj += 32) {
                    svzero_za();

                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 16);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t a1_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti + 16));
                        svuint32_t a1_u32 = svunpklo_u32(a1_u16);
                        a1_u32 = svlsl_n_u32_x(pg32, a1_u32, 16);
                        svfloat32_t a1 = svreinterpret_f32_u32(a1_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 16);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svuint16_t b1_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj + 16));
                        svuint32_t b1_u32 = svunpklo_u32(b1_u16);
                        b1_u32 = svlsl_n_u32_x(pg32, b1_u32, 16);
                        svfloat32_t b1 = svreinterpret_f32_u32(b1_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                        svmopa_za32_f32_m(1, pg32, pg32, a1, b0);
                        svmopa_za32_f32_m(2, pg32, pg32, a0, b1);
                        svmopa_za32_f32_m(3, pg32, pg32, a1, b1);
                    }

                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * ldc + coff + tj), bits);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * ldc + coff + tj + 16), bits);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * ldc + coff + tj), bits);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * ldc + coff + tj + 16), bits);
                    }
                }

                if (tj < jEnd) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 16);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 16);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * ldc + coff + tj), bits);
                    }

                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a1_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti + 16));
                        svuint32_t a1_u32 = svunpklo_u32(a1_u16);
                        a1_u32 = svlsl_n_u32_x(pg32, a1_u32, 16);
                        svfloat32_t a1 = svreinterpret_f32_u32(a1_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 16);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a1, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + 16 + row) * ldc + coff + tj), bits);
                    }
                }
            }

            if (ti < iEnd) {
                for (long tj = j0; tj < jEnd; tj += 16) {
                    svzero_za();
                    for (long kk = 0; kk < k; kk++) {
                        svuint16_t a0_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                        svuint32_t a0_u32 = svunpklo_u32(a0_u16);
                        a0_u32 = svlsl_n_u32_x(pg32, a0_u32, 16);
                        svfloat32_t a0 = svreinterpret_f32_u32(a0_u32);

                        svuint16_t b0_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                        svuint32_t b0_u32 = svunpklo_u32(b0_u16);
                        b0_u32 = svlsl_n_u32_x(pg32, b0_u32, 16);
                        svfloat32_t b0 = svreinterpret_f32_u32(b0_u32);

                        svmopa_za32_f32_m(0, pg32, pg32, a0, b0);
                    }
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                        svuint32_t bits = svreinterpret_u32_f32(zrow);
                        svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                        bit16 = svand_n_u32_x(pg32, bit16, 1);
                        svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                        bits = svadd_u32_x(pg32, bits, rounding);
                        bits = svlsr_n_u32_x(pg32, bits, 16);
                        svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * ldc + coff + tj), bits);
                    }
                }
            }
        }
    }
}
