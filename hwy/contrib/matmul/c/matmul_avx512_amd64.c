// AVX-512 Matrix Multiplication for go-highway
// Compile with: -mavx512f -mavx512fp16 -mavx512bf16
//
// Implements matrix multiply using AVX-512 SIMD instructions.
// For f16: uses AVX-512 FP16 native arithmetic (Sapphire Rapids+)
// For bf16: uses AVX-512 BF16 VDPBF16PS dot product (Cooper Lake+)
// For f32/f64: native AVX-512 FMA

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <immintrin.h>
#endif

// =============================================================================
// matmul_avx512_f16: AVX-512 FP16 matrix multiply for float16
// =============================================================================
// Uses native AVX-512 FP16 arithmetic (Intel Sapphire Rapids, AMD Zen5+)
// 32 f16 elements per 512-bit vector
//
// func matmul_avx512_f16(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_avx512_f16(_Float16 *a, _Float16 *b, _Float16 *c,
                        long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 32 (AVX-512 FP16 vector width)
        for (long j = 0; j < n; j += 32) {
            // Initialize accumulator
            __m512h acc = _mm512_setzero_ph();

            // Accumulate: acc += A[i,p] * B[p,j:j+32] for all p
            for (long p = 0; p < k; p++) {
                // Broadcast A[i,p] to all lanes
                __m512h a_val = _mm512_set1_ph(a[i * k + p]);

                // Load B[p,j:j+32]
                __m512h b_row = _mm512_loadu_ph(b + p * n + j);

                // Native FP16 FMA: acc += a_val * b_row
                acc = _mm512_fmadd_ph(a_val, b_row, acc);
            }

            // Store result to C[i,j:j+32]
            _mm512_storeu_ph(c + i * n + j, acc);
        }
    }
}

// =============================================================================
// matmul_avx512_bf16: AVX-512 BF16 matrix multiply for bfloat16
// =============================================================================
// Uses VDPBF16PS: bf16 dot product accumulate to f32
// Intel Cooper Lake (2020), AMD Zen4+ (2022)
//
// VDPBF16PS: Each f32 result = dot product of 2 bf16 pairs
//   result[i] = src1[2i]*src2[2i] + src1[2i+1]*src2[2i+1]
//
// func matmul_avx512_bf16(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_avx512_bf16(__bf16 *a, __bf16 *b, __bf16 *c,
                         long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 16 (f32 accumulator width)
        for (long j = 0; j < n; j += 16) {
            // Initialize f32 accumulator
            __m512 acc = _mm512_setzero_ps();

            // Process K dimension in pairs (DPBF16 processes 2 bf16 at a time)
            for (long p = 0; p < k; p += 2) {
                // Load 2 consecutive A elements and broadcast as pairs
                // A[i,p:p+2] broadcast to all lanes
                unsigned int a_pair = *(unsigned int*)(a + i * k + p);
                __m512i a_bcast = _mm512_set1_epi32(a_pair);
                __m512bh a_bf16 = (__m512bh)a_bcast;

                // Load B[p:p+2, j:j+16] - need to interleave 2 rows
                // Load B[p,j:j+16] and B[p+1,j:j+16]
                __m256i b_row0 = _mm256_loadu_si256((__m256i*)(b + p * n + j));
                __m256i b_row1 = _mm256_loadu_si256((__m256i*)(b + (p + 1) * n + j));
                // Interleave: [b0[0],b1[0], b0[1],b1[1], ...]
                __m512i b_interleaved = _mm512_inserti64x4(_mm512_castsi256_si512(b_row0), b_row1, 1);
                // Rearrange for DPBF16PS format
                __m512bh b_bf16 = (__m512bh)b_interleaved;

                // DPBF16PS: acc += dot(a_pair, b_pair) for each position
                acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
            }

            // Convert f32 accumulator back to bf16
            __m256i result = _mm512_cvtneps_pbh(acc);
            _mm256_storeu_si256((__m256i*)(c + i * n + j), result);
        }
    }
}

// =============================================================================
// matmul_avx512_f32: AVX-512 matrix multiply for float32
// =============================================================================
// Standard AVX-512 FMA matmul
// 16 f32 elements per 512-bit vector
//
// func matmul_avx512_f32(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_avx512_f32(float *a, float *b, float *c,
                        long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 16 (AVX-512 f32 vector width)
        for (long j = 0; j < n; j += 16) {
            // Initialize accumulator
            __m512 acc = _mm512_setzero_ps();

            // Accumulate: acc += A[i,p] * B[p,j:j+16] for all p
            for (long p = 0; p < k; p++) {
                // Broadcast A[i,p] to all lanes
                __m512 a_val = _mm512_set1_ps(a[i * k + p]);

                // Load B[p,j:j+16]
                __m512 b_row = _mm512_loadu_ps(b + p * n + j);

                // FMA: acc += a_val * b_row
                acc = _mm512_fmadd_ps(a_val, b_row, acc);
            }

            // Store result to C[i,j:j+16]
            _mm512_storeu_ps(c + i * n + j, acc);
        }
    }
}

// =============================================================================
// matmul_avx512_f64: AVX-512 matrix multiply for float64
// =============================================================================
// AVX-512 f64: 8 elements per 512-bit vector
//
// func matmul_avx512_f64(a, b, c unsafe.Pointer, m, n, k int64)
void matmul_avx512_f64(double *a, double *b, double *c,
                        long *pm, long *pn, long *pk) {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process each row of the output
    for (long i = 0; i < m; i++) {
        // Process output columns in chunks of 8 (AVX-512 f64 vector width)
        for (long j = 0; j < n; j += 8) {
            // Initialize accumulator
            __m512d acc = _mm512_setzero_pd();

            // Accumulate: acc += A[i,p] * B[p,j:j+8] for all p
            for (long p = 0; p < k; p++) {
                // Broadcast A[i,p] to all lanes
                __m512d a_val = _mm512_set1_pd(a[i * k + p]);

                // Load B[p,j:j+8]
                __m512d b_row = _mm512_loadu_pd(b + p * n + j);

                // FMA: acc += a_val * b_row
                acc = _mm512_fmadd_pd(a_val, b_row, acc);
            }

            // Store result to C[i,j:j+8]
            _mm512_storeu_pd(c + i * n + j, acc);
        }
    }
}
