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

// Blocked/Cache-Tiled SME FMOPA Matrix Multiplication for go-highway
// Compile with: -march=armv9-a+sme+sme-f64f64
//
// Combines cache-tiled blocking strategy (48×48 blocks for L1 cache) with
// SME FMOPA outer product tiles (16×16 for f32, 8×8 for f64).
//
// Key optimizations:
//   - Single streaming mode entry/exit for the entire operation
//   - Pre-transposed A for contiguous column loads during FMOPA
//   - Cache-tiled to keep working set in L1 for large matrices
//
// IMPORTANT: Only blocks over M and N, NOT K. This is critical for SME:
// - Blocking over K would require loading C into ZA for every k-block
// - Instead, we process ALL of K for each tile, accumulating in ZA
// - M/N blocking ensures C tiles stay in cache
//
// Apple M4 SVL = 512 bits:
//   - f32: 16 lanes, 16×16 tiles, 512 FLOPs per FMOPA
//   - f64: 8 lanes, 8×8 tiles, 128 FLOPs per FMOPA

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

// Block size for cache tiling (tuned for L1 cache, multiple of 16 for FMOPA)
#define BLOCK_SIZE 48

// =============================================================================
// blockedmatmul_fmopa_at_f32: Blocked FMOPA matmul with transposed A (float32)
// =============================================================================
// Computes C = AT^T * B = A * B where:
//   AT is K x M (A transposed, row-major)
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// Processes in 48×48 cache blocks, each containing 16×16 FMOPA tiles.
// Requires M, N to be multiples of 16.
//
// func blockedmatmul_fmopa_at_f32(at, b, c unsafe.Pointer, m, n, k int64)
void blockedmatmul_fmopa_at_f32(float *at, float *b, float *c,
                                 long *pm, long *pn, long *pk) __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    svbool_t pg = svptrue_b32();

    // Block over M and N (48×48 cache blocks)
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

            // Tile within block (16×16)
            for (long ti = i0; ti < iEnd; ti += 16) {
                for (long tj = j0; tj < jEnd; tj += 16) {
                    // Zero ZA for accumulation (process ALL of K)
                    svzero_za();

                    // K loop: accumulate over entire K dimension
                    for (long kk = 0; kk < k; kk++) {
                        // Load A column: AT[kk, ti:ti+16] (contiguous)
                        svfloat32_t a_col = svld1_f32(pg, at + kk * m + ti);
                        // Load B row: B[kk, tj:tj+16] (contiguous)
                        svfloat32_t b_row = svld1_f32(pg, b + kk * n + tj);
                        // Outer product accumulate
                        svmopa_za32_f32_m(0, pg, pg, a_col, b_row);
                    }

                    // Store ZA tile directly to C (overwrite)
                    for (int row = 0; row < 16; row++) {
                        svfloat32_t za_row = svread_hor_za32_f32_m(svundef_f32(), pg, 0, row);
                        svst1_f32(pg, c + (ti + row) * n + tj, za_row);
                    }
                }
            }
        }
    }
}

// =============================================================================
// blockedmatmul_fmopa_at_f64: Blocked FMOPA matmul with transposed A (float64)
// =============================================================================
// Same algorithm but with 8×8 tiles for float64.
// Apple M4 SVL = 512 bits = 8 × float64.
// Requires M, N to be multiples of 8.
//
// func blockedmatmul_fmopa_at_f64(at, b, c unsafe.Pointer, m, n, k int64)
void blockedmatmul_fmopa_at_f64(double *at, double *b, double *c,
                                 long *pm, long *pn, long *pk) __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    svbool_t pg = svptrue_b64();

    // Block over M and N (48×48 cache blocks)
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

            // Tile within block (8×8 for f64)
            for (long ti = i0; ti < iEnd; ti += 8) {
                for (long tj = j0; tj < jEnd; tj += 8) {
                    svzero_za();

                    for (long kk = 0; kk < k; kk++) {
                        svfloat64_t a_col = svld1_f64(pg, at + kk * m + ti);
                        svfloat64_t b_row = svld1_f64(pg, b + kk * n + tj);
                        svmopa_za64_f64_m(0, pg, pg, a_col, b_row);
                    }

                    for (int row = 0; row < 8; row++) {
                        svfloat64_t za_row = svread_hor_za64_f64_m(svundef_f64(), pg, 0, row);
                        svst1_f64(pg, c + (ti + row) * n + tj, za_row);
                    }
                }
            }
        }
    }
}
