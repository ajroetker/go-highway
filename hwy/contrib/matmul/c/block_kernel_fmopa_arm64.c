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

// SME FMOPA Block Kernel for go-highway
// Compile with: -march=armv9-a+sme+sme-f64f64
//
// Computes C += A^T * B for square blocks using SME FMOPA outer product.
// aT is pre-transposed A (rows are original A columns).
// b is normal row-major B.
//
// Uses FMOPA outer product accumulate with ZA tiles:
//   - f32: 16×16 tiles (SVL=512 bits on Apple M4)
//   - f64: 8×8 tiles
//
// Each FMOPA computes a full tile update in one instruction.
// Results are ADDED to existing C values (C += ...).

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

// =============================================================================
// block_muladd_fmopa_f32: C += A^T * B using SME FMOPA for float32
// =============================================================================
// Processes 16×16 tiles using outer product accumulate.
// Requires blockDim to be a multiple of 16.
//
// For each 16×16 tile:
//   1. Zero ZA accumulator
//   2. For each k: ZA += aT[k, tile_rows] ⊗ b[k, tile_cols]
//   3. C[tile] += ZA
//
// func block_muladd_fmopa_f32(aT, b, c unsafe.Pointer, blockDim int64)
void block_muladd_fmopa_f32(float * restrict aT, float * restrict b, float * restrict c,
                             long blockDim) __arm_streaming __arm_out("za") {
    long n = blockDim;

    // Predicate for 16-element operations
    // Note: Clang optimizes two ptrue calls to use same register (p0)
    // Using separate predicates (p0/m, p1/m) like handwritten version would require
    // inline assembly since clang's optimizer merges identical ptrue values.
    svbool_t pg = svptrue_b32();

    // Process 16×16 tiles
    for (long ti = 0; ti < n; ti += 16) {
        for (long tj = 0; tj < n; tj += 16) {
            // Zero ZA accumulator
            svzero_za();

            // K-loop: accumulate outer products into ZA
            // Note: Clang generates tight code that doesn't hide memory latency well.
            // Handwritten assembly achieves ~15-25% higher throughput by interleaving
            // address calculations between loads for latency hiding.
            long stride = n;
            float *aT_ptr = aT + ti;
            float *b_ptr = b + tj;
            for (long k = 0; k < n; k++) {
                svfloat32_t a_col = svld1_f32(pg, aT_ptr);
                svfloat32_t b_row = svld1_f32(pg, b_ptr);
                svmopa_za32_f32_m(0, pg, pg, a_col, b_row);
                aT_ptr += stride;
                b_ptr += stride;
            }

            // Store: C[tile] += ZA
            // Use pointer incrementing to avoid expensive offset pre-computation
            float *c_ptr = c + ti * n + tj;
            for (int row = 0; row < 16; row++) {
                // Read ZA row
                svfloat32_t za_row = svread_hor_za32_f32_m(svundef_f32(), pg, 0, row);
                // Load existing C row
                svfloat32_t c_row = svld1_f32(pg, c_ptr);
                // Add and store
                c_row = svadd_f32_x(pg, c_row, za_row);
                svst1_f32(pg, c_ptr, c_row);
                c_ptr += n;  // Move to next row
            }
        }
    }
}

// =============================================================================
// block_muladd_fmopa_f64: C += A^T * B using SME FMOPA for float64
// =============================================================================
// Same algorithm with 8×8 tiles (SVL=512 bits = 8 × float64).
// Requires blockDim to be a multiple of 8.
//
// func block_muladd_fmopa_f64(aT, b, c unsafe.Pointer, blockDim int64)
void block_muladd_fmopa_f64(double * restrict aT, double * restrict b, double * restrict c,
                             long blockDim) __arm_streaming __arm_out("za") {
    long n = blockDim;

    // Predicate for 8-element operations
    svbool_t pg = svptrue_b64();

    // Process 8×8 tiles
    for (long ti = 0; ti < n; ti += 8) {
        for (long tj = 0; tj < n; tj += 8) {
            // Zero ZA accumulator
            svzero_za();

            // K-loop: same latency hiding limitation as f32
            long stride = n;
            double *aT_ptr = aT + ti;
            double *b_ptr = b + tj;
            for (long k = 0; k < n; k++) {
                svfloat64_t a_col = svld1_f64(pg, aT_ptr);
                svfloat64_t b_row = svld1_f64(pg, b_ptr);
                svmopa_za64_f64_m(0, pg, pg, a_col, b_row);
                aT_ptr += stride;
                b_ptr += stride;
            }

            // Store: C[tile] += ZA
            // Use pointer incrementing to avoid expensive offset pre-computation
            double *c_ptr = c + ti * n + tj;
            for (int row = 0; row < 8; row++) {
                svfloat64_t za_row = svread_hor_za64_f64_m(svundef_f64(), pg, 0, row);
                svfloat64_t c_row = svld1_f64(pg, c_ptr);
                c_row = svadd_f64_x(pg, c_row, za_row);
                svst1_f64(pg, c_ptr, c_row);
                c_ptr += n;  // Move to next row
            }
        }
    }
}
