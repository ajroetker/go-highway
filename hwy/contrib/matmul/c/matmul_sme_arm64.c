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

// SME Matrix Multiplication for go-highway
// Compile with: -march=armv9-a+sme
//
// This implements the same algorithms as the hand-written assembly in
// hwy/contrib/matmul/matmul_fmopa_at_arm64.s
//
// The generated assembly can be compared against the hand-written version
// to verify correctness and optimize.

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif


// =============================================================================
// matmul_fmopa_at_f32: FMOPA-based matrix multiply with transposed A
// =============================================================================
// Computes C = A * B where:
//   AT is K x M (A transposed, row-major) - for contiguous column access
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// Uses 16x16 tile processing with FMOPA outer product accumulate.
// Requires M, N to be multiples of 16.
//
// This mirrors: hwy/contrib/matmul/matmul_fmopa_at_arm64.s
//
// func matmul_fmopa_at_f32(at, b, c unsafe.Pointer, m, n, k int64)
void matmul_fmopa_at_f32(float *at, float *b, float *c,
                          long *pm, long *pn, long *pk) __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process output in 16x16 tiles
    for (long ti = 0; ti < m; ti += 16) {
        for (long tj = 0; tj < n; tj += 16) {
            // Zero accumulator tile
            svzero_za();

            // Accumulate over K dimension
            for (long kk = 0; kk < k; kk++) {
                // Load A column from transposed AT: AT[kk, ti:ti+16]
                // This is contiguous in memory!
                svfloat32_t za_col = svld1_f32(svptrue_b32(), at + kk * m + ti);

                // Load B row: B[kk, tj:tj+16]
                svfloat32_t zb_row = svld1_f32(svptrue_b32(), b + kk * n + tj);

                // Outer product accumulate: ZA0 += za_col * zb_row^T
                // This computes a 16x16 tile contribution in one instruction
                svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), za_col, zb_row);
            }

            // Store result tile to C[ti:ti+16, tj:tj+16]
            for (int row = 0; row < 16; row++) {
                svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, row);
                svst1_f32(svptrue_b32(), c + (ti + row) * n + tj, zrow);
            }
        }
    }
}

// =============================================================================
// matmul_fmopa_at_f64: FMOPA-based matrix multiply for float64
// =============================================================================
// Same algorithm but with 8x8 tiles for float64.
// Apple M4 SVL = 512 bits = 8 x float64
//
// This mirrors: hwy/contrib/matmul/matmul_fmopa_at_f64_arm64.s
//
// func matmul_fmopa_at_f64(at, b, c unsafe.Pointer, m, n, k int64)
void matmul_fmopa_at_f64(double *at, double *b, double *c,
                          long *pm, long *pn, long *pk) __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process output in 8x8 tiles (float64 uses half the lanes)
    for (long ti = 0; ti < m; ti += 8) {
        for (long tj = 0; tj < n; tj += 8) {
            // Zero accumulator tile
            svzero_za();

            // Accumulate over K dimension
            for (long kk = 0; kk < k; kk++) {
                // Load A column from transposed AT: AT[kk, ti:ti+8]
                svfloat64_t za_col = svld1_f64(svptrue_b64(), at + kk * m + ti);

                // Load B row: B[kk, tj:tj+8]
                svfloat64_t zb_row = svld1_f64(svptrue_b64(), b + kk * n + tj);

                // Outer product accumulate for float64
                svmopa_za64_f64_m(0, svptrue_b64(), svptrue_b64(), za_col, zb_row);
            }

            // Store result tile to C[ti:ti+8, tj:tj+8]
            for (int row = 0; row < 8; row++) {
                svfloat64_t zrow = svread_hor_za64_f64_m(svundef_f64(), svptrue_b64(), 0, row);
                svst1_f64(svptrue_b64(), c + (ti + row) * n + tj, zrow);
            }
        }
    }
}

// =============================================================================
// matmul_fmopa_at_f16: FMOPA-based matrix multiply for float16
// =============================================================================
// Uses widening approach: f16 -> f32 -> FMOPA -> f32 -> f16
// Apple M4 doesn't support FEAT_SME_F16F16 (native f16 FMOPA), so we
// convert to f32, use f32 FMOPA with 16x16 tiles, then convert back.
//
// Uses SVE bit manipulation for f16<->f32 conversion (faster than scalar loops).
// FCVT intrinsics have predication issues on M4, so we use explicit bit ops:
//   f16→f32: shift left 13, add exponent bias (112 << 23)
//   f32→f16: subtract bias, round, shift right 13
// This works for normalized f16 values (the typical case for matmul).
//
// scratch: unused (kept for API compatibility)
//
// func matmul_fmopa_at_f16(at, b, c unsafe.Pointer, m, n, k int64, scratch unsafe.Pointer)
void matmul_fmopa_at_f16(__fp16 *at, __fp16 *b, __fp16 *c,
                          long *pm, long *pn, long *pk,
                          float *scratch) __arm_streaming __arm_out("za") {
    (void)scratch;  // unused - kept for API compatibility
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Predicates for operations
    svbool_t pg32 = svptrue_b32();              // All 16 f32 lanes
    svbool_t pg16 = svptrue_pat_b16(SV_VL16);   // First 16 f16 lanes

    // Constants for f16->f32 conversion
    // Exponent adjustment: f32_bias - f16_bias = 127 - 15 = 112
    svuint32_t exp_adjust = svdup_n_u32(112 << 23);  // 112 in f32 exponent position

    for (long ti = 0; ti < m; ti += 16) {
        for (long tj = 0; tj < n; tj += 16) {
            // Zero accumulator tile
            svzero_za();

            // Accumulate over K dimension
            for (long kk = 0; kk < k; kk++) {
                // Load A as u16, convert to f32 via bit manipulation
                svuint16_t a_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                svuint32_t a_u32 = svunpklo_u32(a_u16);
                // f16→f32: shift left 13 to align mantissa, then adjust exponent
                // This works for normalized f16 values (not denormals/inf/nan)
                a_u32 = svlsl_n_u32_x(pg32, a_u32, 13);
                a_u32 = svadd_u32_x(pg32, a_u32, exp_adjust);
                svfloat32_t za_col = svreinterpret_f32_u32(a_u32);

                // Load B as u16, convert to f32
                svuint16_t b_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                svuint32_t b_u32 = svunpklo_u32(b_u16);
                b_u32 = svlsl_n_u32_x(pg32, b_u32, 13);
                b_u32 = svadd_u32_x(pg32, b_u32, exp_adjust);
                svfloat32_t zb_row = svreinterpret_f32_u32(b_u32);

                // Outer product accumulate in f32
                svmopa_za32_f32_m(0, pg32, pg32, za_col, zb_row);
            }

            // Read f32 tiles, convert to f16, and store
            for (int row = 0; row < 16; row++) {
                svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
                svuint32_t bits = svreinterpret_u32_f32(zrow);

                // f32→f16: subtract exponent adjustment, round, shift right 13
                bits = svsub_u32_x(pg32, bits, exp_adjust);
                // Add rounding bias (bit 12 for round-to-nearest)
                svuint32_t round_bit = svlsr_n_u32_x(pg32, bits, 13);
                round_bit = svand_n_u32_x(pg32, round_bit, 1);
                svuint32_t rounding = svadd_n_u32_x(pg32, round_bit, 0xFFF);  // 2^12 - 1
                bits = svadd_u32_x(pg32, bits, rounding);
                bits = svlsr_n_u32_x(pg32, bits, 13);

                // Store as 16-bit
                svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * n + tj), bits);
            }
        }
    }
}

// =============================================================================
// matmul_bfmopa_at_bf16: FMOPA-based matrix multiply for bfloat16
// =============================================================================
// Uses widening approach: bf16 -> f32 -> FMOPA -> f32 -> bf16
// Apple M4's BFMOPA expects 32 bf16 elements per vector, but our tiles are 16 wide.
// We use SVE bit manipulation for bf16<->f32 conversion, then f32 FMOPA with 16x16 tiles.
//
// BF16 is simply the upper 16 bits of F32, so conversion is trivial:
//   bf16→f32: load as u16, unpack to u32, shift left 16, reinterpret as f32
//   f32→bf16: reinterpret as u32, round-to-nearest-even, shift right 16, store as u16
//
// scratch: unused (kept for API compatibility)
//
// func matmul_bfmopa_at_bf16(at, b, c unsafe.Pointer, m, n, k int64, scratch unsafe.Pointer)
void matmul_bfmopa_at_bf16(__bf16 *at, __bf16 *b, __bf16 *c,
                            long *pm, long *pn, long *pk,
                            float *scratch) __arm_streaming __arm_out("za") {
    (void)scratch;  // unused - kept for API compatibility
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Predicates for operations
    // Use ptrue with VL16 pattern instead of whilelt (whilelt may not work outside streaming mode on M4)
    svbool_t pg32 = svptrue_b32();              // All 16 u32/f32 lanes
    svbool_t pg16 = svptrue_pat_b16(SV_VL16);   // First 16 u16 lanes (VL pattern)

    for (long ti = 0; ti < m; ti += 16) {
        for (long tj = 0; tj < n; tj += 16) {
            // Zero accumulator tile
            svzero_za();

            // Accumulate over K dimension
            for (long kk = 0; kk < k; kk++) {
                // Load A column as bf16, convert to f32 using bit manipulation
                // bf16→f32: load u16 (only 16 elements), unpack to u32, shift left 16, reinterpret as f32
                svuint16_t a_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                svuint32_t a_u32 = svunpklo_u32(a_u16);  // unpack low 16 u16 → 16 u32
                a_u32 = svlsl_n_u32_x(pg32, a_u32, 16);
                svfloat32_t za_col = svreinterpret_f32_u32(a_u32);

                // Load B row as bf16, convert to f32
                svuint16_t b_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                svuint32_t b_u32 = svunpklo_u32(b_u16);  // unpack low 16 u16 → 16 u32
                b_u32 = svlsl_n_u32_x(pg32, b_u32, 16);
                svfloat32_t zb_row = svreinterpret_f32_u32(b_u32);

                // Outer product accumulate in f32
                svmopa_za32_f32_m(0, pg32, pg32, za_col, zb_row);
            }

            // Read f32 tiles, convert to bf16 using bit manipulation, and store
            for (int row = 0; row < 16; row++) {
                svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);

                // f32→bf16: reinterpret as u32, add rounding, shift right 16, truncate
                svuint32_t bits = svreinterpret_u32_f32(zrow);
                // Round-to-nearest-even: add 0x7FFF + (bit16 & 1)
                svuint32_t bit16 = svlsr_n_u32_x(pg32, bits, 16);
                bit16 = svand_n_u32_x(pg32, bit16, 1);
                svuint32_t rounding = svadd_n_u32_x(pg32, bit16, 0x7FFF);
                bits = svadd_u32_x(pg32, bits, rounding);
                // Shift right 16 to get bf16 bits in lower 16 bits
                bits = svlsr_n_u32_x(pg32, bits, 16);
                // Truncate to u16 and store (using svst1h for 16-bit store)
                svst1h_u32(pg32, (unsigned short*)(c + (ti + row) * n + tj), bits);
            }
        }
    }
}

// =============================================================================
// matmul_fmopa_at_f16_strided: Strided F16 FMOPA matmul
// =============================================================================
// Same as matmul_fmopa_at_f16 but writes to C with leading dimension ldc
// at column offset coff. Enables incremental B transpose: transpose a strip
// of B, call this function to write directly into the correct columns of
// the full output.
//
// func matmul_fmopa_at_f16_strided(at, b, c, pm, pn, pk, pldc, pcoff, scratch unsafe.Pointer)
void matmul_fmopa_at_f16_strided(__fp16 *at, __fp16 *b, __fp16 *c,
                                  long *pm, long *pn, long *pk,
                                  long *pldc, long *pcoff,
                                  float *scratch) __arm_streaming __arm_out("za") {
    (void)scratch;
    long m = *pm;
    long n = *pn;
    long k = *pk;
    long ldc = *pldc;
    long coff = *pcoff;

    svbool_t pg32 = svptrue_b32();
    svbool_t pg16 = svptrue_pat_b16(SV_VL16);
    svuint32_t exp_adjust = svdup_n_u32(112 << 23);

    for (long ti = 0; ti < m; ti += 16) {
        for (long tj = 0; tj < n; tj += 16) {
            svzero_za();

            for (long kk = 0; kk < k; kk++) {
                svuint16_t a_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                svuint32_t a_u32 = svunpklo_u32(a_u16);
                a_u32 = svlsl_n_u32_x(pg32, a_u32, 13);
                a_u32 = svadd_u32_x(pg32, a_u32, exp_adjust);
                svfloat32_t za_col = svreinterpret_f32_u32(a_u32);

                svuint16_t b_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                svuint32_t b_u32 = svunpklo_u32(b_u16);
                b_u32 = svlsl_n_u32_x(pg32, b_u32, 13);
                b_u32 = svadd_u32_x(pg32, b_u32, exp_adjust);
                svfloat32_t zb_row = svreinterpret_f32_u32(b_u32);

                svmopa_za32_f32_m(0, pg32, pg32, za_col, zb_row);
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

// =============================================================================
// matmul_bfmopa_at_bf16_strided: Strided BF16 BFMOPA matmul
// =============================================================================
// Same as matmul_bfmopa_at_bf16 but writes to C with leading dimension ldc
// at column offset coff.
//
// func matmul_bfmopa_at_bf16_strided(at, b, c, pm, pn, pk, pldc, pcoff, scratch unsafe.Pointer)
void matmul_bfmopa_at_bf16_strided(__bf16 *at, __bf16 *b, __bf16 *c,
                                    long *pm, long *pn, long *pk,
                                    long *pldc, long *pcoff,
                                    float *scratch) __arm_streaming __arm_out("za") {
    (void)scratch;
    long m = *pm;
    long n = *pn;
    long k = *pk;
    long ldc = *pldc;
    long coff = *pcoff;

    svbool_t pg32 = svptrue_b32();
    svbool_t pg16 = svptrue_pat_b16(SV_VL16);

    for (long ti = 0; ti < m; ti += 16) {
        for (long tj = 0; tj < n; tj += 16) {
            svzero_za();

            for (long kk = 0; kk < k; kk++) {
                svuint16_t a_u16 = svld1_u16(pg16, (unsigned short*)(at + kk * m + ti));
                svuint32_t a_u32 = svunpklo_u32(a_u16);
                a_u32 = svlsl_n_u32_x(pg32, a_u32, 16);
                svfloat32_t za_col = svreinterpret_f32_u32(a_u32);

                svuint16_t b_u16 = svld1_u16(pg16, (unsigned short*)(b + kk * n + tj));
                svuint32_t b_u32 = svunpklo_u32(b_u16);
                b_u32 = svlsl_n_u32_x(pg32, b_u32, 16);
                svfloat32_t zb_row = svreinterpret_f32_u32(b_u32);

                svmopa_za32_f32_m(0, pg32, pg32, za_col, zb_row);
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
