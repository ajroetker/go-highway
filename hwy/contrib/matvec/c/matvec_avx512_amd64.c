// AVX-512 Matrix-Vector Multiplication for go-highway
// Compile with: -mavx512f -mavx512fp16 -mavx512bf16
//
// Implements matrix-vector multiply using AVX-512 SIMD instructions.
// For f16: uses AVX-512 FP16 native arithmetic (Sapphire Rapids+)
// For bf16: uses AVX-512 BF16 VDPBF16PS dot product (Cooper Lake+)

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <immintrin.h>
#endif

// =============================================================================
// matvec_avx512_f16: AVX-512 FP16 matrix-vector multiply for float16
// =============================================================================
// Uses native AVX-512 FP16 arithmetic
// 32 f16 elements per 512-bit vector
//
// func matvec_avx512_f16(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_avx512_f16(_Float16 *m, _Float16 *v, _Float16 *result,
                        long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        __m512h acc = _mm512_setzero_ph();

        long j;
        for (j = 0; j + 32 <= cols; j += 32) {
            __m512h m_row = _mm512_loadu_ph(m + i * cols + j);
            __m512h v_chunk = _mm512_loadu_ph(v + j);
            acc = _mm512_fmadd_ph(m_row, v_chunk, acc);
        }

        // Horizontal sum using reduction
        _Float16 sum = _mm512_reduce_add_ph(acc);

        // Handle remaining elements
        for (; j < cols; j++) {
            sum += m[i * cols + j] * v[j];
        }

        result[i] = sum;
    }
}

// =============================================================================
// matvec_avx512_bf16: AVX-512 BF16 matrix-vector multiply for bfloat16
// =============================================================================
// Uses VDPBF16PS: bf16 dot product accumulate to f32
//
// func matvec_avx512_bf16(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_avx512_bf16(__bf16 *m, __bf16 *v, __bf16 *result,
                         long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        __m512 acc = _mm512_setzero_ps();

        // Process in chunks of 32 bf16 (for 16 f32 dot products)
        long j;
        for (j = 0; j + 32 <= cols; j += 32) {
            // Load M row chunk (32 bf16)
            __m512i m_bf16 = _mm512_loadu_si512(m + i * cols + j);
            // Load v chunk (32 bf16)
            __m512i v_bf16 = _mm512_loadu_si512(v + j);

            // DPBF16PS: acc[i] += m[2i]*v[2i] + m[2i+1]*v[2i+1]
            acc = _mm512_dpbf16_ps(acc, (__m512bh)m_bf16, (__m512bh)v_bf16);
        }

        // Horizontal sum of f32 accumulator
        float sum = _mm512_reduce_add_ps(acc);

        // Handle remaining elements
        for (; j < cols; j++) {
            unsigned int m_u32 = (unsigned int)*(unsigned short*)&m[i * cols + j] << 16;
            unsigned int v_u32 = (unsigned int)*(unsigned short*)&v[j] << 16;
            sum += *(float*)&m_u32 * *(float*)&v_u32;
        }

        // Convert f32 back to bf16 with rounding
        __m128 sum_vec = _mm_set_ss(sum);
        __m128bh result_bf16 = _mm_cvtneps_pbh(sum_vec);
        result[i] = ((__bf16*)&result_bf16)[0];
    }
}

// =============================================================================
// matvec_avx512_f32: AVX-512 matrix-vector multiply for float32
// =============================================================================
// func matvec_avx512_f32(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_avx512_f32(float *m, float *v, float *result,
                        long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        __m512 acc = _mm512_setzero_ps();

        long j;
        for (j = 0; j + 16 <= cols; j += 16) {
            __m512 m_row = _mm512_loadu_ps(m + i * cols + j);
            __m512 v_chunk = _mm512_loadu_ps(v + j);
            acc = _mm512_fmadd_ps(m_row, v_chunk, acc);
        }

        // Horizontal sum
        float sum = _mm512_reduce_add_ps(acc);

        // Handle remaining elements
        for (; j < cols; j++) {
            sum += m[i * cols + j] * v[j];
        }

        result[i] = sum;
    }
}

// =============================================================================
// matvec_avx512_f64: AVX-512 matrix-vector multiply for float64
// =============================================================================
// func matvec_avx512_f64(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_avx512_f64(double *m, double *v, double *result,
                        long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        __m512d acc = _mm512_setzero_pd();

        long j;
        for (j = 0; j + 8 <= cols; j += 8) {
            __m512d m_row = _mm512_loadu_pd(m + i * cols + j);
            __m512d v_chunk = _mm512_loadu_pd(v + j);
            acc = _mm512_fmadd_pd(m_row, v_chunk, acc);
        }

        // Horizontal sum
        double sum = _mm512_reduce_add_pd(acc);

        // Handle remaining elements
        for (; j < cols; j++) {
            sum += m[i * cols + j] * v[j];
        }

        result[i] = sum;
    }
}
