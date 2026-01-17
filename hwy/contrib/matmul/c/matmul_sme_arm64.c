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
// scratch: 16-element f32 buffer passed from Go to avoid SVE-dependent stack allocation
//
// func matmul_fmopa_at_f16(at, b, c unsafe.Pointer, m, n, k int64, scratch unsafe.Pointer)
void matmul_fmopa_at_f16(__fp16 *at, __fp16 *b, __fp16 *c,
                          long *pm, long *pn, long *pk,
                          float *scratch) __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process output in 16x16 tiles (f32 accumulator size)
    //
    // Strategy: Load f16 data via scratch buffer to avoid SVE/fcvt complexity.
    // This is slower but avoids streaming-mode compatibility issues.
    svbool_t pg32 = svptrue_b32();  // All 16 f32 lanes

    for (long ti = 0; ti < m; ti += 16) {
        for (long tj = 0; tj < n; tj += 16) {
            // Zero accumulator tile
            svzero_za();

            // Accumulate over K dimension
            for (long kk = 0; kk < k; kk++) {
                // Load A column: convert f16 to f32 via scratch buffer
                // Copy 16 f16 elements and convert to f32
                for (int i = 0; i < 16; i++) {
                    scratch[i] = (float)at[kk * m + ti + i];
                }
                svfloat32_t za_col = svld1_f32(pg32, scratch);

                // Load B row: convert f16 to f32 via scratch buffer
                for (int i = 0; i < 16; i++) {
                    scratch[i] = (float)b[kk * n + tj + i];
                }
                svfloat32_t zb_row = svld1_f32(pg32, scratch);

                // Outer product accumulate in f32
                svmopa_za32_f32_m(0, pg32, pg32, za_col, zb_row);
            }

            // Read f32 tiles and store via scratch buffer with scalar conversion to f16
            // Scratch buffer passed from Go avoids SVE-dependent stack allocation
            for (int row = 0; row < 16; row++) {
                svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, row);
                // Store f32 to scratch buffer
                svst1_f32(svptrue_b32(), scratch, zrow);
                // Convert each f32 to f16 and store
                for (int col = 0; col < 16; col++) {
                    c[(ti + row) * n + tj + col] = (__fp16)scratch[col];
                }
            }
        }
    }
}

// =============================================================================
// matmul_bfmopa_at_bf16: FMOPA-based matrix multiply for bfloat16
// =============================================================================
// Uses widening approach: bf16 -> f32 -> FMOPA -> f32 -> bf16
// Apple M4's BFMOPA expects 32 bf16 elements per vector, but our tiles are 16 wide.
// So we use the same scalar conversion approach as F16: load bf16, convert to f32,
// use f32 FMOPA with 16x16 tiles, then convert back to bf16.
//
// scratch: 16-element f32 buffer passed from Go to avoid SVE-dependent stack allocation
//
// func matmul_bfmopa_at_bf16(at, b, c unsafe.Pointer, m, n, k int64, scratch unsafe.Pointer)
void matmul_bfmopa_at_bf16(__bf16 *at, __bf16 *b, __bf16 *c,
                            long *pm, long *pn, long *pk,
                            float *scratch) __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process output in 16x16 tiles (f32 accumulator size)
    //
    // Strategy: Load bf16 data via scratch buffer to avoid SVE vector width issues.
    // BFMOPA expects 32 bf16 elements per vector on SVL=512, but our tiles are 16 wide.
    // Using scalar conversion ensures we only read 16 elements as intended.
    svbool_t pg32 = svptrue_b32();  // All 16 f32 lanes

    for (long ti = 0; ti < m; ti += 16) {
        for (long tj = 0; tj < n; tj += 16) {
            // Zero accumulator tile
            svzero_za();

            // Accumulate over K dimension
            for (long kk = 0; kk < k; kk++) {
                // Load A column: convert bf16 to f32 via scratch buffer
                // Copy 16 bf16 elements and convert to f32
                for (int i = 0; i < 16; i++) {
                    // BF16 to F32: shift left by 16 bits (bf16 is upper 16 bits of f32)
                    unsigned short bf16_bits;
                    __builtin_memcpy(&bf16_bits, &at[kk * m + ti + i], sizeof(bf16_bits));
                    unsigned int f32_bits = ((unsigned int)bf16_bits) << 16;
                    __builtin_memcpy(&scratch[i], &f32_bits, sizeof(f32_bits));
                }
                svfloat32_t za_col = svld1_f32(pg32, scratch);

                // Load B row: convert bf16 to f32 via scratch buffer
                for (int i = 0; i < 16; i++) {
                    unsigned short bf16_bits;
                    __builtin_memcpy(&bf16_bits, &b[kk * n + tj + i], sizeof(bf16_bits));
                    unsigned int f32_bits = ((unsigned int)bf16_bits) << 16;
                    __builtin_memcpy(&scratch[i], &f32_bits, sizeof(f32_bits));
                }
                svfloat32_t zb_row = svld1_f32(pg32, scratch);

                // Outer product accumulate in f32
                svmopa_za32_f32_m(0, pg32, pg32, za_col, zb_row);
            }

            // Read f32 tiles and store via scratch buffer with scalar conversion to bf16
            // Scratch buffer passed from Go avoids SVE-dependent stack allocation
            for (int row = 0; row < 16; row++) {
                svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, row);
                // Store f32 to scratch buffer
                svst1_f32(svptrue_b32(), scratch, zrow);
                // Convert each f32 to bf16 and store
                // BF16 conversion: truncate lower 16 bits with rounding
                for (int col = 0; col < 16; col++) {
                    unsigned int bits;
                    __builtin_memcpy(&bits, &scratch[col], sizeof(bits));
                    unsigned int rounding = 0x7FFF + ((bits >> 16) & 1);
                    bits += rounding;
                    unsigned short bf16_bits = (unsigned short)(bits >> 16);
                    __builtin_memcpy(&c[(ti + row) * n + tj + col], &bf16_bits, sizeof(bf16_bits));
                }
            }
        }
    }
}
