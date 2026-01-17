// SME Matrix-Vector Multiplication for go-highway
// Compile with: -march=armv9-a+sme
//
// This implements the same algorithms as the hand-written assembly in
// hwy/contrib/matvec/matvec_sme_arm64.s
//
// The generated assembly can be compared against the hand-written version
// to verify correctness and optimize.

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif


// =============================================================================
// matvec_sme_f32: FMOPA-based matrix-vector multiplication
// =============================================================================
// Computes result = M * v where:
//   MT is cols x rows (M transposed, row-major) - for contiguous column access
//   v is cols (input vector)
//   result is rows (output vector)
//
// With M transposed, loading M columns becomes contiguous!
//   M[row:row+16, k] = MT[k, row:row+16] = contiguous in memory
//
// Algorithm (using outer product like matmul):
//   For each 16-row tile:
//     zero ZA
//     for k in 0..cols:
//       z2 = MT[k, row:row+16]  (contiguous load - like A column in matmul)
//       z0 = broadcast(v[k])   (like B row, but broadcast scalar)
//       FMOPA za0 += z2 ⊗ z0
//     Extract za0 rows and store first element of each to result
//
// Apple M4 SVL = 512 bits = 16 × float32
// ZA tiles = 16×16 = 256 float32 elements
//
// This mirrors: hwy/contrib/matvec/matvec_sme_arm64.s
//
// func matvec_sme_f32(mt, v, result unsafe.Pointer, rows, cols int64)
void matvec_sme_f32(float *mt, float *v, float *result,
                     long *prows, long *pcols) __arm_streaming __arm_out("za") {
    long rows = *prows;
    long cols = *pcols;

    // Process 16 rows at a time
    for (long row = 0; row < rows; row += 16) {
        // Zero accumulator tile
        svzero_za();

        // Inner loop: accumulate over columns
        for (long k = 0; k < cols; k++) {
            // Load M column: M[row:row+16, k] from transposed MT[k, row:row+16]
            // This is contiguous at: MT_base + k * rows + row
            svfloat32_t z_col = svld1_f32(svptrue_b32(), mt + k * rows + row);

            // Broadcast v[k] to all lanes
            svfloat32_t z_v = svdup_f32(v[k]);

            // Outer product accumulate: ZA0 += z_col * z_v^T
            // Since z_v is broadcast, za[i][j] = sum_k(M[row+i, k] * v[k]) for all j
            svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), z_col, z_v);
        }

        // Extract horizontal slices from za0 and store first element of each
        // After accumulation, za[i][j] = dot(row i, v) for all j (all columns same)
        for (int i = 0; i < 16; i++) {
            svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, i);
            // Extract first element and store
            result[row + i] = svlasta_f32(svptrue_b32(), zrow);
        }
    }
}

// =============================================================================
// matvec_sme_f64: FMOPA-based matrix-vector multiplication for float64
// =============================================================================
// Same algorithm but with 8-element tiles for float64.
// Apple M4 SVL = 512 bits = 8 × float64
// ZA tiles = 8×8 = 64 float64 elements
//
// This mirrors: hwy/contrib/matvec/matvec_sme_arm64.s (f64 version)
//
// func matvec_sme_f64(mt, v, result unsafe.Pointer, rows, cols int64)
void matvec_sme_f64(double *mt, double *v, double *result,
                     long *prows, long *pcols) __arm_streaming __arm_out("za") {
    long rows = *prows;
    long cols = *pcols;

    // Process 8 rows at a time (float64 has half the lanes)
    for (long row = 0; row < rows; row += 8) {
        // Zero accumulator tile
        svzero_za();

        // Inner loop: accumulate over columns
        for (long k = 0; k < cols; k++) {
            // Load M column from transposed MT[k, row:row+8]
            svfloat64_t z_col = svld1_f64(svptrue_b64(), mt + k * rows + row);

            // Broadcast v[k] to all lanes
            svfloat64_t z_v = svdup_f64(v[k]);

            // Outer product accumulate for float64
            svmopa_za64_f64_m(0, svptrue_b64(), svptrue_b64(), z_col, z_v);
        }

        // Extract and store first element of each row
        for (int i = 0; i < 8; i++) {
            svfloat64_t zrow = svread_hor_za64_f64_m(svundef_f64(), svptrue_b64(), 0, i);
            // Extract first element and store
            result[row + i] = svlasta_f64(svptrue_b64(), zrow);
        }
    }
}

// =============================================================================
// matvec_sme_f16: FMOPA-based matrix-vector multiplication for float16
// =============================================================================
// Uses widening approach: f16 -> f32 -> FMOPA -> f32 -> f16
// Apple M4 doesn't support FEAT_SME_F16F16 (native f16 FMOPA), so we
// convert to f32, use f32 FMOPA with 16-element tiles, then convert back.
//
// scratch: 16-element f32 buffer passed from Go to avoid SVE-dependent stack allocation
//
// func matvec_sme_f16(mt, v, result unsafe.Pointer, rows, cols int64, scratch unsafe.Pointer)
void matvec_sme_f16(__fp16 *mt, __fp16 *v, __fp16 *result,
                     long *prows, long *pcols,
                     float *scratch) __arm_streaming __arm_out("za") {
    long rows = *prows;
    long cols = *pcols;

    // Key insight: svcvt_f32_f16 reads from EVEN-indexed h elements (0, 2, 4, ...)
    // because .s element i corresponds to .h element 2i.
    //
    // We must load exactly 16 f16 elements (not 32) to avoid reading past array bounds.
    // svptrue_pat_b16(SV_VL32) = 32 bytes = 16 × 2-byte f16 elements
    //
    // Strategy: Use svzip1 to duplicate each f16 element into consecutive positions:
    //   Original: z.h = [e0, e1, ..., e15, garbage...]
    //   After zip1(z,z): z.h = [e0, e0, e1, e1, ..., e15, e15]
    //   After fcvt (reads even indices): z.s = [e0, e1, ..., e15] as f32
    svbool_t pg16_vl32 = svptrue_pat_b16(SV_VL32);  // 32 bytes = 16 f16 elements
    svbool_t pg32 = svptrue_b32();  // All 16 f32 lanes for ops

    // Process 16 rows at a time (f32 accumulator size)
    for (long row = 0; row < rows; row += 16) {
        // Zero accumulator tile
        svzero_za();

        // Inner loop: accumulate over columns
        for (long k = 0; k < cols; k++) {
            // Load M column as f16: MT[k, row:row+16] - exactly 16 elements
            svfloat16_t z_col_f16 = svld1_f16(pg16_vl32, mt + k * rows + row);
            // Interleave with self: [e0,e1,...,e15] -> [e0,e0,e1,e1,...,e15,e15]
            svfloat16_t z_col_interleaved = svzip1_f16(z_col_f16, z_col_f16);
            // Convert: reads even indices -> gets consecutive original elements
            svfloat32_t z_col = svcvt_f32_f16_x(pg32, z_col_interleaved);

            // Load v[k] as f16, convert to f32, and broadcast
            float v_f32 = (float)v[k];
            svfloat32_t z_v = svdup_f32(v_f32);

            // Outer product accumulate in f32
            svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), z_col, z_v);
        }

        // Extract f32 rows via scratch buffer and convert to f16
        // Scratch buffer passed from Go avoids SVE-dependent stack allocation
        for (int i = 0; i < 16; i++) {
            svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, i);
            // Store to scratch buffer and extract first element
            svst1_f32(svptrue_b32(), scratch, zrow);
            // Convert to f16 and store
            result[row + i] = (__fp16)scratch[0];
        }
    }
}

// =============================================================================
// matvec_sme_bf16: BFMOPA-based matrix-vector multiplication for bfloat16
// =============================================================================
// Uses widening BFMOPA: bf16 inputs accumulate to f32, then convert back.
// Apple M4 SVL = 512 bits = 32 × bfloat16 (input), 16 × float32 (accumulator)
//
// Uses 16-element tiles (same as f32) since accumulator is f32.
//
// scratch: 16-element f32 buffer passed from Go to avoid SVE-dependent stack allocation
//
// func matvec_sme_bf16(mt, v, result unsafe.Pointer, rows, cols int64, scratch unsafe.Pointer)
void matvec_sme_bf16(__bf16 *mt, __bf16 *v, __bf16 *result,
                      long *prows, long *pcols,
                      float *scratch) __arm_streaming __arm_out("za") {
    long rows = *prows;
    long cols = *pcols;

    // Strategy: Load bf16 with all lanes active
    // svptrue_b16() loads 32 bf16 elements on SVL=512
    // BFMOPA uses the first 16 elements for the 16x16 f32 accumulator
    svbool_t pg16 = svptrue_b16();  // All 32 bf16 lanes for loads

    // Process 16 rows at a time (f32 accumulator size)
    for (long row = 0; row < rows; row += 16) {
        // Zero accumulator tile
        svzero_za();

        // Inner loop: accumulate over columns
        for (long k = 0; k < cols; k++) {
            // Load M column from transposed MT[k, row:row+16] (loads 32, uses first 16)
            svbfloat16_t z_col = svld1_bf16(pg16, mt + k * rows + row);

            // Broadcast v[k] - need to convert bf16 scalar to bf16 vector
            // Load single bf16 and broadcast
            svbfloat16_t z_v = svdup_bf16(v[k]);

            // Widening outer product: bf16 inputs -> f32 accumulator
            svmopa_za32_bf16_m(0, svptrue_b32(), svptrue_b32(), z_col, z_v);
        }

        // Extract f32 rows via scratch buffer and convert to bf16
        // Scratch buffer passed from Go avoids SVE-dependent stack allocation
        for (int i = 0; i < 16; i++) {
            svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, i);
            // Store to scratch buffer and extract first element
            svst1_f32(svptrue_b32(), scratch, zrow);
            // Convert to bf16 and store
            // Manual conversion: truncate lower 16 bits with rounding
            unsigned int bits;
            __builtin_memcpy(&bits, &scratch[0], sizeof(bits));
            unsigned int rounding = 0x7FFF + ((bits >> 16) & 1);
            bits += rounding;
            unsigned short bf16_bits = (unsigned short)(bits >> 16);
            __builtin_memcpy(&result[row + i], &bf16_bits, sizeof(bf16_bits));
        }
    }
}
