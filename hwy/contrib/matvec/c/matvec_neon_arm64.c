// NEON Matrix-Vector Multiplication for go-highway
// Compile with: -march=armv8.2-a+fp16 -march=armv8.6-a+bf16
//
// Implements matrix-vector multiply using NEON SIMD instructions.
// For f16: uses native half-precision FMA (ARMv8.2-A FP16)
// For bf16: uses BFDOT dot product accumulate (ARMv8.6-A BF16)

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// =============================================================================
// matvec_neon_f16: NEON matrix-vector multiply for float16
// =============================================================================
// Computes result = M * v where:
//   M is rows x cols (row-major)
//   v is cols (input vector)
//   result is rows (output vector)
//
// NEON f16: 8 elements per 128-bit vector
//
// func matvec_neon_f16(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_neon_f16(__fp16 *m, __fp16 *v, __fp16 *result,
                      long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        // Initialize accumulators
        float16x8_t acc = vdupq_n_f16((__fp16)0.0f);

        // Accumulate: acc += M[i,j:j+8] * v[j:j+8] for all j
        long j;
        for (j = 0; j + 8 <= cols; j += 8) {
            // Load M row chunk
            float16x8_t m_row = vld1q_f16(m + i * cols + j);
            // Load v chunk
            float16x8_t v_chunk = vld1q_f16(v + j);
            // FMA: acc += m_row * v_chunk
            acc = vfmaq_f16(acc, m_row, v_chunk);
        }

        // Horizontal sum of accumulator
        // float16x4_t sum4 = vadd_f16(vget_low_f16(acc), vget_high_f16(acc));
        // Reduce to scalar - need to extract and sum manually
        __fp16 sum = 0.0f;
        for (int k = 0; k < 8; k++) {
            sum += acc[k];
        }

        // Handle remaining elements (if cols not multiple of 8)
        for (; j < cols; j++) {
            sum += m[i * cols + j] * v[j];
        }

        result[i] = sum;
    }
}

// =============================================================================
// matvec_neon_bf16: NEON matrix-vector multiply for bfloat16
// =============================================================================
// Uses BFDOT for dot product: accumulates bf16 pairs into f32
//
// func matvec_neon_bf16(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_neon_bf16(__bf16 *m, __bf16 *v, __bf16 *result,
                       long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        // Initialize f32 accumulator (bf16 math uses f32 accumulation)
        float32x4_t acc = vdupq_n_f32(0.0f);

        // Process columns in chunks of 8 (BFDOT processes 8 bf16 -> 4 f32)
        long j;
        for (j = 0; j + 8 <= cols; j += 8) {
            // Load M row chunk (8 bf16)
            bfloat16x8_t m_row = vld1q_bf16(m + i * cols + j);
            // Load v chunk (8 bf16)
            bfloat16x8_t v_chunk = vld1q_bf16(v + j);
            // BFDOT: acc[i] += m[2i]*v[2i] + m[2i+1]*v[2i+1]
            acc = vbfdotq_f32(acc, m_row, v_chunk);
        }

        // Horizontal sum of f32 accumulator
        float sum = vaddvq_f32(acc);

        // Handle remaining elements (if cols not multiple of 8)
        for (; j < cols; j++) {
            // Manual bf16 to f32 conversion and multiply
            float m_val = *(unsigned short*)&m[i * cols + j];
            m_val = *(float*)&(unsigned int){(unsigned int)*(unsigned short*)&m[i * cols + j] << 16};
            float v_val = *(float*)&(unsigned int){(unsigned int)*(unsigned short*)&v[j] << 16};
            sum += m_val * v_val;
        }

        // Convert f32 back to bf16 and store
        unsigned int bits = *(unsigned int*)&sum;
        unsigned int rounding = 0x7FFF + ((bits >> 16) & 1);
        bits += rounding;
        result[i] = (__bf16)(bits >> 16);
    }
}

// =============================================================================
// matvec_neon_f32: NEON matrix-vector multiply for float32
// =============================================================================
// Standard NEON f32 matvec for comparison
//
// func matvec_neon_f32(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_neon_f32(float *m, float *v, float *result,
                      long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        // Initialize accumulators
        float32x4_t acc = vdupq_n_f32(0.0f);

        // Accumulate: acc += M[i,j:j+4] * v[j:j+4] for all j
        long j;
        for (j = 0; j + 4 <= cols; j += 4) {
            // Load M row chunk
            float32x4_t m_row = vld1q_f32(m + i * cols + j);
            // Load v chunk
            float32x4_t v_chunk = vld1q_f32(v + j);
            // FMA: acc += m_row * v_chunk
            acc = vfmaq_f32(acc, m_row, v_chunk);
        }

        // Horizontal sum of accumulator
        float sum = vaddvq_f32(acc);

        // Handle remaining elements
        for (; j < cols; j++) {
            sum += m[i * cols + j] * v[j];
        }

        result[i] = sum;
    }
}

// =============================================================================
// matvec_neon_f64: NEON matrix-vector multiply for float64
// =============================================================================
// NEON f64: 2 elements per 128-bit vector
//
// func matvec_neon_f64(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_neon_f64(double *m, double *v, double *result,
                      long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        // Initialize accumulators
        float64x2_t acc = vdupq_n_f64(0.0);

        // Accumulate: acc += M[i,j:j+2] * v[j:j+2] for all j
        long j;
        for (j = 0; j + 2 <= cols; j += 2) {
            // Load M row chunk
            float64x2_t m_row = vld1q_f64(m + i * cols + j);
            // Load v chunk
            float64x2_t v_chunk = vld1q_f64(v + j);
            // FMA: acc += m_row * v_chunk
            acc = vfmaq_f64(acc, m_row, v_chunk);
        }

        // Horizontal sum of accumulator
        double sum = vaddvq_f64(acc);

        // Handle remaining elements
        for (; j < cols; j++) {
            sum += m[i * cols + j] * v[j];
        }

        result[i] = sum;
    }
}
