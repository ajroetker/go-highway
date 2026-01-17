// NEON Matrix Multiplication for go-highway
// Compile with: -march=armv8.2-a+fp16 -march=armv8.6-a+bf16
//
// Implements matrix multiply using NEON SIMD instructions.
// For f16: uses native half-precision FMA (ARMv8.2-A FP16)
// For bf16: uses BFMMLA matrix multiply accumulate (ARMv8.6-A BF16)

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// =============================================================================
// matmul_neon_f16: NEON matrix multiply for float16
// =============================================================================
// Computes C = A * B where:
//   A is M x K (row-major)
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// Uses "broadcast A, stream B" algorithm:
//   For each row i of A:
//     For each element A[i,p]:
//       C[i,:] += A[i,p] * B[p,:]
//
// NEON f16: 8 elements per 128-bit vector
//
// func matmul_neon_f16(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_neon_f16(__fp16 *a, __fp16 *b, __fp16 *c,
                      long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 8 (NEON f16 vector width)
        for (long j = 0; j < n; j += 8) {
            // Initialize accumulator
            float16x8_t acc = vdupq_n_f16((__fp16)0.0f);

            // Accumulate: acc += A[i,p] * B[p,j:j+8] for all p
            for (long p = 0; p < k; p++) {
                // Broadcast A[i,p] to all lanes
                float16x8_t a_val = vdupq_n_f16(a[i * k + p]);

                // Load B[p,j:j+8]
                float16x8_t b_row = vld1q_f16(b + p * n + j);

                // FMA: acc += a_val * b_row
                acc = vfmaq_f16(acc, a_val, b_row);
            }

            // Store result to C[i,j:j+8]
            vst1q_f16(c + i * n + j, acc);
        }
    }
}

// =============================================================================
// matmul_neon_bf16: NEON matrix multiply for bfloat16
// =============================================================================
// Computes C = A * B using BFMMLA (BFloat16 Matrix Multiply Accumulate)
//
// BFMMLA computes a 2x2 output tile from 2x4 and 4x2 inputs:
//   C[2x2] += A[2x4] * B[4x2]
//
// This is different from standard matmul - we need to restructure the
// computation around 2x4 blocks.
//
// For simplicity, this implementation uses BFDOT (dot product) instead,
// which accumulates bf16 pairs into f32, similar to the "broadcast A, stream B"
// pattern but processes 2 elements at a time.
//
// func matmul_neon_bf16(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_neon_bf16(__bf16 *a, __bf16 *b, __bf16 *c,
                       long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 4 (f32 accumulator width)
        for (long j = 0; j < n; j += 4) {
            // Initialize f32 accumulators (bf16 math uses f32 accumulation)
            float32x4_t acc = vdupq_n_f32(0.0f);

            // Process K dimension in pairs (BFDOT processes 2 bf16 at a time)
            for (long p = 0; p < k; p += 2) {
                // Load 2 consecutive A elements: A[i,p], A[i,p+1]
                bfloat16x8_t a_pair = vld1q_bf16(a + i * k + p);
                // We only use the first 2 elements, but load 8 for BFDOT format

                // Load B[p:p+2, j:j+4] - need to gather 2 rows of 4 elements each
                // B[p,j:j+4] and B[p+1,j:j+4]
                bfloat16x4_t b_row0 = vld1_bf16(b + p * n + j);
                bfloat16x4_t b_row1 = vld1_bf16(b + (p + 1) * n + j);

                // Combine into 8 elements for BFDOT: [b0, b1, b0, b1, ...]
                bfloat16x8_t b_combined = vcombine_bf16(b_row0, b_row1);

                // BFDOT: acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]
                // This computes dot products of bf16 pairs into f32
                acc = vbfdotq_f32(acc, a_pair, b_combined);
            }

            // Convert f32 accumulator back to bf16 and store
            bfloat16x4_t result = vcvt_bf16_f32(acc);
            vst1_bf16(c + i * n + j, result);
        }
    }
}

// =============================================================================
// matmul_neon_f32: NEON matrix multiply for float32
// =============================================================================
// Standard NEON f32 matmul for comparison
//
// func matmul_neon_f32(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_neon_f32(float *a, float *b, float *c,
                      long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 4 (NEON f32 vector width)
        for (long j = 0; j < n; j += 4) {
            // Initialize accumulator
            float32x4_t acc = vdupq_n_f32(0.0f);

            // Accumulate: acc += A[i,p] * B[p,j:j+4] for all p
            for (long p = 0; p < k; p++) {
                // Broadcast A[i,p] to all lanes
                float32x4_t a_val = vdupq_n_f32(a[i * k + p]);

                // Load B[p,j:j+4]
                float32x4_t b_row = vld1q_f32(b + p * n + j);

                // FMA: acc += a_val * b_row
                acc = vfmaq_f32(acc, a_val, b_row);
            }

            // Store result to C[i,j:j+4]
            vst1q_f32(c + i * n + j, acc);
        }
    }
}

// =============================================================================
// matmul_neon_f64: NEON matrix multiply for float64
// =============================================================================
// NEON f64: 2 elements per 128-bit vector
//
// func matmul_neon_f64(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_neon_f64(double *a, double *b, double *c,
                      long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 2 (NEON f64 vector width)
        for (long j = 0; j < n; j += 2) {
            // Initialize accumulator
            float64x2_t acc = vdupq_n_f64(0.0);

            // Accumulate: acc += A[i,p] * B[p,j:j+2] for all p
            for (long p = 0; p < k; p++) {
                // Broadcast A[i,p] to all lanes
                float64x2_t a_val = vdupq_n_f64(a[i * k + p]);

                // Load B[p,j:j+2]
                float64x2_t b_row = vld1q_f64(b + p * n + j);

                // FMA: acc += a_val * b_row
                acc = vfmaq_f64(acc, a_val, b_row);
            }

            // Store result to C[i,j:j+2]
            vst1q_f64(c + i * n + j, acc);
        }
    }
}
