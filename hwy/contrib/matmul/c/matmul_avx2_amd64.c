// AVX2 Matrix Multiplication for go-highway
// Compile with: -mavx2 -mfma -mf16c
//
// Implements matrix multiply using AVX2 SIMD instructions.
// For f16: uses F16C for conversion, compute in f32 (AVX2 has no native f16 FMA)
// For bf16: emulates via f32 conversion (no native bf16 support in AVX2)
// For f32/f64: native AVX2 FMA

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <immintrin.h>
#endif

// =============================================================================
// matmul_avx2_f16: AVX2 matrix multiply for float16
// =============================================================================
// Uses F16C for conversion: f16 -> f32 -> compute -> f32 -> f16
// AVX2 has no native f16 FMA, so we use f32 intermediate
//
// VCVTPH2PS: 8 f16 -> 8 f32 (256-bit)
// VCVTPS2PH: 8 f32 -> 8 f16 (128-bit output)
//
// func matmul_avx2_f16(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_avx2_f16(unsigned short *a, unsigned short *b, unsigned short *c,
                      long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 8 (AVX2 f32 vector width)
        for (long j = 0; j < n; j += 8) {
            // Initialize f32 accumulator
            __m256 acc = _mm256_setzero_ps();

            // Accumulate: acc += A[i,p] * B[p,j:j+8] for all p
            for (long p = 0; p < k; p++) {
                // Load A[i,p] as f16 and convert to f32, broadcast
                unsigned short a_f16 = a[i * k + p];
                __m128i a_vec = _mm_set1_epi16(a_f16);
                __m256 a_f32 = _mm256_cvtph_ps(a_vec);

                // Load B[p,j:j+8] as f16 and convert to f32
                __m128i b_f16 = _mm_loadu_si128((__m128i*)(b + p * n + j));
                __m256 b_f32 = _mm256_cvtph_ps(b_f16);

                // FMA: acc += a_f32 * b_f32
                acc = _mm256_fmadd_ps(a_f32, b_f32, acc);
            }

            // Convert f32 accumulator back to f16 and store
            __m128i result = _mm256_cvtps_ph(acc, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i*)(c + i * n + j), result);
        }
    }
}

// =============================================================================
// matmul_avx2_bf16: AVX2 matrix multiply for bfloat16
// =============================================================================
// AVX2 has no native bf16 support, emulate via f32 conversion
// bf16 to f32: shift left by 16 bits
// f32 to bf16: shift right by 16 bits (with rounding)
//
// func matmul_avx2_bf16(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_avx2_bf16(unsigned short *a, unsigned short *b, unsigned short *c,
                       long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 8
        for (long j = 0; j < n; j += 8) {
            // Initialize f32 accumulator
            __m256 acc = _mm256_setzero_ps();

            // Accumulate: acc += A[i,p] * B[p,j:j+8] for all p
            for (long p = 0; p < k; p++) {
                // Load A[i,p] as bf16 and convert to f32, broadcast
                unsigned short a_bf16 = a[i * k + p];
                unsigned int a_u32 = (unsigned int)a_bf16 << 16;
                float a_f32_scalar = *(float*)&a_u32;
                __m256 a_f32 = _mm256_set1_ps(a_f32_scalar);

                // Load B[p,j:j+8] as bf16 and convert to f32
                __m128i b_bf16 = _mm_loadu_si128((__m128i*)(b + p * n + j));
                // Unpack bf16 to f32: shift left by 16
                __m256i b_u32 = _mm256_cvtepu16_epi32(b_bf16);
                b_u32 = _mm256_slli_epi32(b_u32, 16);
                __m256 b_f32 = _mm256_castsi256_ps(b_u32);

                // FMA: acc += a_f32 * b_f32
                acc = _mm256_fmadd_ps(a_f32, b_f32, acc);
            }

            // Convert f32 accumulator back to bf16 with rounding
            __m256i acc_u32 = _mm256_castps_si256(acc);
            // Add rounding bias: 0x7FFF + bit 16
            __m256i bias = _mm256_and_si256(_mm256_srli_epi32(acc_u32, 16), _mm256_set1_epi32(1));
            bias = _mm256_add_epi32(bias, _mm256_set1_epi32(0x7FFF));
            acc_u32 = _mm256_add_epi32(acc_u32, bias);
            // Shift right by 16 to get bf16
            acc_u32 = _mm256_srli_epi32(acc_u32, 16);
            // Pack 32-bit to 16-bit
            __m128i result_lo = _mm256_castsi256_si128(acc_u32);
            __m128i result_hi = _mm256_extracti128_si256(acc_u32, 1);
            __m128i result = _mm_packus_epi32(result_lo, result_hi);
            _mm_storeu_si128((__m128i*)(c + i * n + j), result);
        }
    }
}

// =============================================================================
// matmul_avx2_f32: AVX2 matrix multiply for float32
// =============================================================================
// Standard AVX2 FMA matmul
//
// func matmul_avx2_f32(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_avx2_f32(float *a, float *b, float *c,
                      long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 8 (AVX2 f32 vector width)
        for (long j = 0; j < n; j += 8) {
            // Initialize accumulator
            __m256 acc = _mm256_setzero_ps();

            // Accumulate: acc += A[i,p] * B[p,j:j+8] for all p
            for (long p = 0; p < k; p++) {
                // Broadcast A[i,p] to all lanes
                __m256 a_val = _mm256_set1_ps(a[i * k + p]);

                // Load B[p,j:j+8]
                __m256 b_row = _mm256_loadu_ps(b + p * n + j);

                // FMA: acc += a_val * b_row
                acc = _mm256_fmadd_ps(a_val, b_row, acc);
            }

            // Store result to C[i,j:j+8]
            _mm256_storeu_ps(c + i * n + j, acc);
        }
    }
}

// =============================================================================
// matmul_avx2_f64: AVX2 matrix multiply for float64
// =============================================================================
// AVX2 f64: 4 elements per 256-bit vector
//
// func matmul_avx2_f64(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_avx2_f64(double *a, double *b, double *c,
                      long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 4 (AVX2 f64 vector width)
        for (long j = 0; j < n; j += 4) {
            // Initialize accumulator
            __m256d acc = _mm256_setzero_pd();

            // Accumulate: acc += A[i,p] * B[p,j:j+4] for all p
            for (long p = 0; p < k; p++) {
                // Broadcast A[i,p] to all lanes
                __m256d a_val = _mm256_set1_pd(a[i * k + p]);

                // Load B[p,j:j+4]
                __m256d b_row = _mm256_loadu_pd(b + p * n + j);

                // FMA: acc += a_val * b_row
                acc = _mm256_fmadd_pd(a_val, b_row, acc);
            }

            // Store result to C[i,j:j+4]
            _mm256_storeu_pd(c + i * n + j, acc);
        }
    }
}
