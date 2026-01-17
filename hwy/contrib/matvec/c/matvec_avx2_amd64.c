// AVX2 Matrix-Vector Multiplication for go-highway
// Compile with: -mavx2 -mfma -mf16c
//
// Implements matrix-vector multiply using AVX2 SIMD instructions.
// For f16: uses F16C for conversion, compute in f32
// For bf16: emulates via f32 conversion

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <immintrin.h>
#endif

// =============================================================================
// matvec_avx2_f16: AVX2 matrix-vector multiply for float16
// =============================================================================
// Uses F16C for conversion: f16 -> f32 -> compute -> f32 -> f16
//
// func matvec_avx2_f16(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_avx2_f16(unsigned short *m, unsigned short *v, unsigned short *result,
                      long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        // Initialize f32 accumulator
        __m256 acc = _mm256_setzero_ps();

        // Accumulate in chunks of 8
        long j;
        for (j = 0; j + 8 <= cols; j += 8) {
            // Load M row chunk as f16 and convert to f32
            __m128i m_f16 = _mm_loadu_si128((__m128i*)(m + i * cols + j));
            __m256 m_f32 = _mm256_cvtph_ps(m_f16);

            // Load v chunk as f16 and convert to f32
            __m128i v_f16 = _mm_loadu_si128((__m128i*)(v + j));
            __m256 v_f32 = _mm256_cvtph_ps(v_f16);

            // FMA: acc += m_f32 * v_f32
            acc = _mm256_fmadd_ps(m_f32, v_f32, acc);
        }

        // Horizontal sum of accumulator
        // acc = [a0,a1,a2,a3,a4,a5,a6,a7]
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        // sum128 = [a0+a4, a1+a5, a2+a6, a3+a7]
        sum128 = _mm_hadd_ps(sum128, sum128);
        // sum128 = [a0+a4+a1+a5, a2+a6+a3+a7, ...]
        sum128 = _mm_hadd_ps(sum128, sum128);
        // sum128 = [total, total, total, total]
        float sum = _mm_cvtss_f32(sum128);

        // Handle remaining elements
        for (; j < cols; j++) {
            // Manual f16 to f32 conversion and multiply
            __m128i m_scalar = _mm_set1_epi16(m[i * cols + j]);
            __m128 m_f32 = _mm_cvtph_ps(m_scalar);
            __m128i v_scalar = _mm_set1_epi16(v[j]);
            __m128 v_f32 = _mm_cvtph_ps(v_scalar);
            sum += _mm_cvtss_f32(_mm_mul_ss(m_f32, v_f32));
        }

        // Convert f32 back to f16 and store
        __m128 sum_vec = _mm_set_ss(sum);
        __m128i result_f16 = _mm_cvtps_ph(sum_vec, _MM_FROUND_TO_NEAREST_INT);
        result[i] = (unsigned short)_mm_extract_epi16(result_f16, 0);
    }
}

// =============================================================================
// matvec_avx2_bf16: AVX2 matrix-vector multiply for bfloat16
// =============================================================================
// Emulates via f32 conversion (no native bf16 in AVX2)
//
// func matvec_avx2_bf16(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_avx2_bf16(unsigned short *m, unsigned short *v, unsigned short *result,
                       long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        // Initialize f32 accumulator
        __m256 acc = _mm256_setzero_ps();

        // Accumulate in chunks of 8
        long j;
        for (j = 0; j + 8 <= cols; j += 8) {
            // Load M row chunk as bf16 and convert to f32
            __m128i m_bf16 = _mm_loadu_si128((__m128i*)(m + i * cols + j));
            __m256i m_u32 = _mm256_cvtepu16_epi32(m_bf16);
            m_u32 = _mm256_slli_epi32(m_u32, 16);
            __m256 m_f32 = _mm256_castsi256_ps(m_u32);

            // Load v chunk as bf16 and convert to f32
            __m128i v_bf16 = _mm_loadu_si128((__m128i*)(v + j));
            __m256i v_u32 = _mm256_cvtepu16_epi32(v_bf16);
            v_u32 = _mm256_slli_epi32(v_u32, 16);
            __m256 v_f32 = _mm256_castsi256_ps(v_u32);

            // FMA: acc += m_f32 * v_f32
            acc = _mm256_fmadd_ps(m_f32, v_f32, acc);
        }

        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);

        // Handle remaining elements
        for (; j < cols; j++) {
            unsigned int m_u32 = (unsigned int)m[i * cols + j] << 16;
            unsigned int v_u32 = (unsigned int)v[j] << 16;
            sum += *(float*)&m_u32 * *(float*)&v_u32;
        }

        // Convert f32 back to bf16 with rounding and store
        unsigned int bits = *(unsigned int*)&sum;
        unsigned int rounding = 0x7FFF + ((bits >> 16) & 1);
        bits += rounding;
        result[i] = (unsigned short)(bits >> 16);
    }
}

// =============================================================================
// matvec_avx2_f32: AVX2 matrix-vector multiply for float32
// =============================================================================
// func matvec_avx2_f32(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_avx2_f32(float *m, float *v, float *result,
                      long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        __m256 acc = _mm256_setzero_ps();

        long j;
        for (j = 0; j + 8 <= cols; j += 8) {
            __m256 m_row = _mm256_loadu_ps(m + i * cols + j);
            __m256 v_chunk = _mm256_loadu_ps(v + j);
            acc = _mm256_fmadd_ps(m_row, v_chunk, acc);
        }

        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);

        // Handle remaining elements
        for (; j < cols; j++) {
            sum += m[i * cols + j] * v[j];
        }

        result[i] = sum;
    }
}

// =============================================================================
// matvec_avx2_f64: AVX2 matrix-vector multiply for float64
// =============================================================================
// func matvec_avx2_f64(m, v, result unsafe.Pointer, rows, cols int64)
void matvec_avx2_f64(double *m, double *v, double *result,
                      long *prows, long *pcols) {
    long rows = *prows;
    long cols = *pcols;

    // Process each row
    for (long i = 0; i < rows; i++) {
        __m256d acc = _mm256_setzero_pd();

        long j;
        for (j = 0; j + 4 <= cols; j += 4) {
            __m256d m_row = _mm256_loadu_pd(m + i * cols + j);
            __m256d v_chunk = _mm256_loadu_pd(v + j);
            acc = _mm256_fmadd_pd(m_row, v_chunk, acc);
        }

        // Horizontal sum
        __m128d lo = _mm256_castpd256_pd128(acc);
        __m128d hi = _mm256_extractf128_pd(acc, 1);
        __m128d sum128 = _mm_add_pd(lo, hi);
        sum128 = _mm_hadd_pd(sum128, sum128);
        double sum = _mm_cvtsd_f64(sum128);

        // Handle remaining elements
        for (; j < cols; j++) {
            sum += m[i * cols + j] * v[j];
        }

        result[i] = sum;
    }
}
