// Native Float16 SIMD operations for x86-64 with AVX-512 FP16 extension
// Used with GoAT to generate Go assembly
// Compile with: -mavx512fp16
//
// Requires: Intel Sapphire Rapids (2023+), AMD Zen5+
// Provides: Native float16 arithmetic with 32 lanes per 512-bit ZMM register!
//
// This is different from F16C (conversion-only) - AVX-512 FP16 has native
// float16 arithmetic instructions: VADDPH, VSUBPH, VMULPH, VDIVPH, VFMADD132PH, etc.

#include <immintrin.h>

// ============================================================================
// Native Float16 Arithmetic Operations (32 lanes per ZMM register!)
// ============================================================================

// Vector addition: result[i] = a[i] + b[i]
void add_f16_avx512(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 128 float16 at a time (4 ZMM registers)
    for (; i + 127 < n; i += 128) {
        __m512h a0 = _mm512_loadu_ph(a + i);
        __m512h a1 = _mm512_loadu_ph(a + i + 32);
        __m512h a2 = _mm512_loadu_ph(a + i + 64);
        __m512h a3 = _mm512_loadu_ph(a + i + 96);

        __m512h b0 = _mm512_loadu_ph(b + i);
        __m512h b1 = _mm512_loadu_ph(b + i + 32);
        __m512h b2 = _mm512_loadu_ph(b + i + 64);
        __m512h b3 = _mm512_loadu_ph(b + i + 96);

        _mm512_storeu_ph(result + i, _mm512_add_ph(a0, b0));        // VADDPH zmm
        _mm512_storeu_ph(result + i + 32, _mm512_add_ph(a1, b1));
        _mm512_storeu_ph(result + i + 64, _mm512_add_ph(a2, b2));
        _mm512_storeu_ph(result + i + 96, _mm512_add_ph(a3, b3));
    }

    // Process 32 float16 at a time (1 ZMM register)
    for (; i + 31 < n; i += 32) {
        __m512h av = _mm512_loadu_ph(a + i);
        __m512h bv = _mm512_loadu_ph(b + i);
        _mm512_storeu_ph(result + i, _mm512_add_ph(av, bv));
    }

    // Process 16 float16 at a time (1 YMM register)
    for (; i + 15 < n; i += 16) {
        __m256h av = _mm256_loadu_ph(a + i);
        __m256h bv = _mm256_loadu_ph(b + i);
        _mm256_storeu_ph(result + i, _mm256_add_ph(av, bv));
    }

    // Scalar remainder using single-element operations
    for (; i < n; i++) {
        __m128h av = _mm_load_sh(a + i);
        __m128h bv = _mm_load_sh(b + i);
        __m128h rv = _mm_add_sh(av, bv);
        _mm_store_sh(result + i, rv);
    }
}

// Vector subtraction: result[i] = a[i] - b[i]
void sub_f16_avx512(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 128 float16 at a time (4 ZMM registers)
    for (; i + 127 < n; i += 128) {
        __m512h a0 = _mm512_loadu_ph(a + i);
        __m512h a1 = _mm512_loadu_ph(a + i + 32);
        __m512h a2 = _mm512_loadu_ph(a + i + 64);
        __m512h a3 = _mm512_loadu_ph(a + i + 96);

        __m512h b0 = _mm512_loadu_ph(b + i);
        __m512h b1 = _mm512_loadu_ph(b + i + 32);
        __m512h b2 = _mm512_loadu_ph(b + i + 64);
        __m512h b3 = _mm512_loadu_ph(b + i + 96);

        _mm512_storeu_ph(result + i, _mm512_sub_ph(a0, b0));        // VSUBPH zmm
        _mm512_storeu_ph(result + i + 32, _mm512_sub_ph(a1, b1));
        _mm512_storeu_ph(result + i + 64, _mm512_sub_ph(a2, b2));
        _mm512_storeu_ph(result + i + 96, _mm512_sub_ph(a3, b3));
    }

    // Process 32 float16 at a time (1 ZMM register)
    for (; i + 31 < n; i += 32) {
        __m512h av = _mm512_loadu_ph(a + i);
        __m512h bv = _mm512_loadu_ph(b + i);
        _mm512_storeu_ph(result + i, _mm512_sub_ph(av, bv));
    }

    // Process 16 float16 at a time (1 YMM register)
    for (; i + 15 < n; i += 16) {
        __m256h av = _mm256_loadu_ph(a + i);
        __m256h bv = _mm256_loadu_ph(b + i);
        _mm256_storeu_ph(result + i, _mm256_sub_ph(av, bv));
    }

    // Scalar remainder using single-element operations
    for (; i < n; i++) {
        __m128h av = _mm_load_sh(a + i);
        __m128h bv = _mm_load_sh(b + i);
        __m128h rv = _mm_sub_sh(av, bv);
        _mm_store_sh(result + i, rv);
    }
}

// Vector multiplication: result[i] = a[i] * b[i]
void mul_f16_avx512(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 128 float16 at a time (4 ZMM registers)
    for (; i + 127 < n; i += 128) {
        __m512h a0 = _mm512_loadu_ph(a + i);
        __m512h a1 = _mm512_loadu_ph(a + i + 32);
        __m512h a2 = _mm512_loadu_ph(a + i + 64);
        __m512h a3 = _mm512_loadu_ph(a + i + 96);

        __m512h b0 = _mm512_loadu_ph(b + i);
        __m512h b1 = _mm512_loadu_ph(b + i + 32);
        __m512h b2 = _mm512_loadu_ph(b + i + 64);
        __m512h b3 = _mm512_loadu_ph(b + i + 96);

        _mm512_storeu_ph(result + i, _mm512_mul_ph(a0, b0));        // VMULPH zmm
        _mm512_storeu_ph(result + i + 32, _mm512_mul_ph(a1, b1));
        _mm512_storeu_ph(result + i + 64, _mm512_mul_ph(a2, b2));
        _mm512_storeu_ph(result + i + 96, _mm512_mul_ph(a3, b3));
    }

    // Process 32 float16 at a time (1 ZMM register)
    for (; i + 31 < n; i += 32) {
        __m512h av = _mm512_loadu_ph(a + i);
        __m512h bv = _mm512_loadu_ph(b + i);
        _mm512_storeu_ph(result + i, _mm512_mul_ph(av, bv));
    }

    // Process 16 float16 at a time (1 YMM register)
    for (; i + 15 < n; i += 16) {
        __m256h av = _mm256_loadu_ph(a + i);
        __m256h bv = _mm256_loadu_ph(b + i);
        _mm256_storeu_ph(result + i, _mm256_mul_ph(av, bv));
    }

    // Scalar remainder using single-element operations
    for (; i < n; i++) {
        __m128h av = _mm_load_sh(a + i);
        __m128h bv = _mm_load_sh(b + i);
        __m128h rv = _mm_mul_sh(av, bv);
        _mm_store_sh(result + i, rv);
    }
}

// Vector division: result[i] = a[i] / b[i]
void div_f16_avx512(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 128 float16 at a time (4 ZMM registers)
    for (; i + 127 < n; i += 128) {
        __m512h a0 = _mm512_loadu_ph(a + i);
        __m512h a1 = _mm512_loadu_ph(a + i + 32);
        __m512h a2 = _mm512_loadu_ph(a + i + 64);
        __m512h a3 = _mm512_loadu_ph(a + i + 96);

        __m512h b0 = _mm512_loadu_ph(b + i);
        __m512h b1 = _mm512_loadu_ph(b + i + 32);
        __m512h b2 = _mm512_loadu_ph(b + i + 64);
        __m512h b3 = _mm512_loadu_ph(b + i + 96);

        _mm512_storeu_ph(result + i, _mm512_div_ph(a0, b0));        // VDIVPH zmm
        _mm512_storeu_ph(result + i + 32, _mm512_div_ph(a1, b1));
        _mm512_storeu_ph(result + i + 64, _mm512_div_ph(a2, b2));
        _mm512_storeu_ph(result + i + 96, _mm512_div_ph(a3, b3));
    }

    // Process 32 float16 at a time (1 ZMM register)
    for (; i + 31 < n; i += 32) {
        __m512h av = _mm512_loadu_ph(a + i);
        __m512h bv = _mm512_loadu_ph(b + i);
        _mm512_storeu_ph(result + i, _mm512_div_ph(av, bv));
    }

    // Process 16 float16 at a time (1 YMM register)
    for (; i + 15 < n; i += 16) {
        __m256h av = _mm256_loadu_ph(a + i);
        __m256h bv = _mm256_loadu_ph(b + i);
        _mm256_storeu_ph(result + i, _mm256_div_ph(av, bv));
    }

    // Scalar remainder using single-element operations
    for (; i < n; i++) {
        __m128h av = _mm_load_sh(a + i);
        __m128h bv = _mm_load_sh(b + i);
        __m128h rv = _mm_div_sh(av, bv);
        _mm_store_sh(result + i, rv);
    }
}

// Fused multiply-add: result[i] = a[i] * b[i] + c[i]
void fma_f16_avx512(unsigned short *a, unsigned short *b, unsigned short *c, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 128 float16 at a time (4 ZMM registers)
    for (; i + 127 < n; i += 128) {
        __m512h a0 = _mm512_loadu_ph(a + i);
        __m512h a1 = _mm512_loadu_ph(a + i + 32);
        __m512h a2 = _mm512_loadu_ph(a + i + 64);
        __m512h a3 = _mm512_loadu_ph(a + i + 96);

        __m512h b0 = _mm512_loadu_ph(b + i);
        __m512h b1 = _mm512_loadu_ph(b + i + 32);
        __m512h b2 = _mm512_loadu_ph(b + i + 64);
        __m512h b3 = _mm512_loadu_ph(b + i + 96);

        __m512h c0 = _mm512_loadu_ph(c + i);
        __m512h c1 = _mm512_loadu_ph(c + i + 32);
        __m512h c2 = _mm512_loadu_ph(c + i + 64);
        __m512h c3 = _mm512_loadu_ph(c + i + 96);

        // VFMADD132PH: result = a * b + c
        _mm512_storeu_ph(result + i, _mm512_fmadd_ph(a0, b0, c0));
        _mm512_storeu_ph(result + i + 32, _mm512_fmadd_ph(a1, b1, c1));
        _mm512_storeu_ph(result + i + 64, _mm512_fmadd_ph(a2, b2, c2));
        _mm512_storeu_ph(result + i + 96, _mm512_fmadd_ph(a3, b3, c3));
    }

    // Process 32 float16 at a time (1 ZMM register)
    for (; i + 31 < n; i += 32) {
        __m512h av = _mm512_loadu_ph(a + i);
        __m512h bv = _mm512_loadu_ph(b + i);
        __m512h cv = _mm512_loadu_ph(c + i);
        _mm512_storeu_ph(result + i, _mm512_fmadd_ph(av, bv, cv));
    }

    // Process 16 float16 at a time (1 YMM register)
    for (; i + 15 < n; i += 16) {
        __m256h av = _mm256_loadu_ph(a + i);
        __m256h bv = _mm256_loadu_ph(b + i);
        __m256h cv = _mm256_loadu_ph(c + i);
        _mm256_storeu_ph(result + i, _mm256_fmadd_ph(av, bv, cv));
    }

    // Scalar remainder using single-element operations
    for (; i < n; i++) {
        __m128h av = _mm_load_sh(a + i);
        __m128h bv = _mm_load_sh(b + i);
        __m128h cv = _mm_load_sh(c + i);
        __m128h rv = _mm_fmadd_sh(av, bv, cv);
        _mm_store_sh(result + i, rv);
    }
}

// Square root: result[i] = sqrt(a[i])
void sqrt_f16_avx512(unsigned short *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 128 float16 at a time (4 ZMM registers)
    for (; i + 127 < n; i += 128) {
        __m512h a0 = _mm512_loadu_ph(a + i);
        __m512h a1 = _mm512_loadu_ph(a + i + 32);
        __m512h a2 = _mm512_loadu_ph(a + i + 64);
        __m512h a3 = _mm512_loadu_ph(a + i + 96);

        _mm512_storeu_ph(result + i, _mm512_sqrt_ph(a0));           // VSQRTPH zmm
        _mm512_storeu_ph(result + i + 32, _mm512_sqrt_ph(a1));
        _mm512_storeu_ph(result + i + 64, _mm512_sqrt_ph(a2));
        _mm512_storeu_ph(result + i + 96, _mm512_sqrt_ph(a3));
    }

    // Process 32 float16 at a time (1 ZMM register)
    for (; i + 31 < n; i += 32) {
        __m512h av = _mm512_loadu_ph(a + i);
        _mm512_storeu_ph(result + i, _mm512_sqrt_ph(av));
    }

    // Process 16 float16 at a time (1 YMM register)
    for (; i + 15 < n; i += 16) {
        __m256h av = _mm256_loadu_ph(a + i);
        _mm256_storeu_ph(result + i, _mm256_sqrt_ph(av));
    }

    // Scalar remainder using single-element operations
    for (; i < n; i++) {
        __m128h av = _mm_load_sh(a + i);
        __m128h rv = _mm_sqrt_sh(av, av);
        _mm_store_sh(result + i, rv);
    }
}

// Vector minimum: result[i] = min(a[i], b[i])
void min_f16_avx512(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 128 float16 at a time (4 ZMM registers)
    for (; i + 127 < n; i += 128) {
        __m512h a0 = _mm512_loadu_ph(a + i);
        __m512h a1 = _mm512_loadu_ph(a + i + 32);
        __m512h a2 = _mm512_loadu_ph(a + i + 64);
        __m512h a3 = _mm512_loadu_ph(a + i + 96);

        __m512h b0 = _mm512_loadu_ph(b + i);
        __m512h b1 = _mm512_loadu_ph(b + i + 32);
        __m512h b2 = _mm512_loadu_ph(b + i + 64);
        __m512h b3 = _mm512_loadu_ph(b + i + 96);

        _mm512_storeu_ph(result + i, _mm512_min_ph(a0, b0));        // VMINPH zmm
        _mm512_storeu_ph(result + i + 32, _mm512_min_ph(a1, b1));
        _mm512_storeu_ph(result + i + 64, _mm512_min_ph(a2, b2));
        _mm512_storeu_ph(result + i + 96, _mm512_min_ph(a3, b3));
    }

    // Process 32 float16 at a time (1 ZMM register)
    for (; i + 31 < n; i += 32) {
        __m512h av = _mm512_loadu_ph(a + i);
        __m512h bv = _mm512_loadu_ph(b + i);
        _mm512_storeu_ph(result + i, _mm512_min_ph(av, bv));
    }

    // Process 16 float16 at a time (1 YMM register)
    for (; i + 15 < n; i += 16) {
        __m256h av = _mm256_loadu_ph(a + i);
        __m256h bv = _mm256_loadu_ph(b + i);
        _mm256_storeu_ph(result + i, _mm256_min_ph(av, bv));
    }

    // Scalar remainder using single-element operations
    for (; i < n; i++) {
        __m128h av = _mm_load_sh(a + i);
        __m128h bv = _mm_load_sh(b + i);
        __m128h rv = _mm_min_sh(av, bv);
        _mm_store_sh(result + i, rv);
    }
}

// Vector maximum: result[i] = max(a[i], b[i])
void max_f16_avx512(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 128 float16 at a time (4 ZMM registers)
    for (; i + 127 < n; i += 128) {
        __m512h a0 = _mm512_loadu_ph(a + i);
        __m512h a1 = _mm512_loadu_ph(a + i + 32);
        __m512h a2 = _mm512_loadu_ph(a + i + 64);
        __m512h a3 = _mm512_loadu_ph(a + i + 96);

        __m512h b0 = _mm512_loadu_ph(b + i);
        __m512h b1 = _mm512_loadu_ph(b + i + 32);
        __m512h b2 = _mm512_loadu_ph(b + i + 64);
        __m512h b3 = _mm512_loadu_ph(b + i + 96);

        _mm512_storeu_ph(result + i, _mm512_max_ph(a0, b0));        // VMAXPH zmm
        _mm512_storeu_ph(result + i + 32, _mm512_max_ph(a1, b1));
        _mm512_storeu_ph(result + i + 64, _mm512_max_ph(a2, b2));
        _mm512_storeu_ph(result + i + 96, _mm512_max_ph(a3, b3));
    }

    // Process 32 float16 at a time (1 ZMM register)
    for (; i + 31 < n; i += 32) {
        __m512h av = _mm512_loadu_ph(a + i);
        __m512h bv = _mm512_loadu_ph(b + i);
        _mm512_storeu_ph(result + i, _mm512_max_ph(av, bv));
    }

    // Process 16 float16 at a time (1 YMM register)
    for (; i + 15 < n; i += 16) {
        __m256h av = _mm256_loadu_ph(a + i);
        __m256h bv = _mm256_loadu_ph(b + i);
        _mm256_storeu_ph(result + i, _mm256_max_ph(av, bv));
    }

    // Scalar remainder using single-element operations
    for (; i < n; i++) {
        __m128h av = _mm_load_sh(a + i);
        __m128h bv = _mm_load_sh(b + i);
        __m128h rv = _mm_max_sh(av, bv);
        _mm_store_sh(result + i, rv);
    }
}

// ============================================================================
// Float16 <-> Float32 Conversions using AVX-512 FP16
// ============================================================================

// Promote float16 to float32: result[i] = (float32)a[i]
// Uses native AVX-512 FP16 conversion (faster than F16C for large arrays)
void promote_f16_to_f32_avx512(unsigned short *a, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 64 float16 -> 64 float32 at a time (2 ZMM inputs -> 4 ZMM outputs)
    for (; i + 63 < n; i += 64) {
        __m512h h0 = _mm512_loadu_ph(a + i);
        __m512h h1 = _mm512_loadu_ph(a + i + 32);

        // Each 32-lane F16 vector produces two 16-lane F32 vectors
        _mm512_storeu_ps(result + i, _mm512_cvtxph_ps(_mm512_castph512_ph256(h0)));
        _mm512_storeu_ps(result + i + 16, _mm512_cvtxph_ps(_mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(h0), 1))));
        _mm512_storeu_ps(result + i + 32, _mm512_cvtxph_ps(_mm512_castph512_ph256(h1)));
        _mm512_storeu_ps(result + i + 48, _mm512_cvtxph_ps(_mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(h1), 1))));
    }

    // Process 32 float16 -> 32 float32 at a time (1 ZMM input -> 2 ZMM outputs)
    for (; i + 31 < n; i += 32) {
        __m512h hv = _mm512_loadu_ph(a + i);
        _mm512_storeu_ps(result + i, _mm512_cvtxph_ps(_mm512_castph512_ph256(hv)));
        _mm512_storeu_ps(result + i + 16, _mm512_cvtxph_ps(_mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(hv), 1))));
    }

    // Process 16 float16 -> 16 float32 at a time (1 YMM input -> 1 ZMM output)
    for (; i + 15 < n; i += 16) {
        __m256h hv = _mm256_loadu_ph(a + i);
        _mm512_storeu_ps(result + i, _mm512_cvtxph_ps(hv));
    }

    // Process 8 float16 -> 8 float32 at a time
    for (; i + 7 < n; i += 8) {
        __m128h hv = _mm_loadu_ph(a + i);
        _mm256_storeu_ps(result + i, _mm256_cvtxph_ps(hv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        __m128h hv = _mm_load_sh(a + i);
        __m128 fv = _mm_cvtsh_ss(_mm_setzero_ps(), hv);
        result[i] = _mm_cvtss_f32(fv);
    }
}

// Demote float32 to float16: result[i] = (float16)a[i]
// Uses native AVX-512 FP16 conversion
void demote_f32_to_f16_avx512(float *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 64 float32 -> 64 float16 at a time (4 ZMM inputs -> 2 ZMM outputs)
    for (; i + 63 < n; i += 64) {
        __m512 f0 = _mm512_loadu_ps(a + i);
        __m512 f1 = _mm512_loadu_ps(a + i + 16);
        __m512 f2 = _mm512_loadu_ps(a + i + 32);
        __m512 f3 = _mm512_loadu_ps(a + i + 48);

        // Convert pairs of F32 vectors to F16
        __m256h h0 = _mm512_cvtxps_ph(f0);
        __m256h h1 = _mm512_cvtxps_ph(f1);
        __m256h h2 = _mm512_cvtxps_ph(f2);
        __m256h h3 = _mm512_cvtxps_ph(f3);

        // Combine into 512-bit vectors and store
        __m512h out0 = _mm512_castph256_ph512(h0);
        out0 = _mm512_insertf64x4(_mm512_castph_pd(out0), _mm256_castph_pd(h1), 1);
        _mm512_storeu_ph(result + i, _mm512_castpd_ph(out0));

        __m512h out1 = _mm512_castph256_ph512(h2);
        out1 = _mm512_insertf64x4(_mm512_castph_pd(out1), _mm256_castph_pd(h3), 1);
        _mm512_storeu_ph(result + i + 32, _mm512_castpd_ph(out1));
    }

    // Process 32 float32 -> 32 float16 at a time (2 ZMM inputs -> 1 ZMM output)
    for (; i + 31 < n; i += 32) {
        __m512 f0 = _mm512_loadu_ps(a + i);
        __m512 f1 = _mm512_loadu_ps(a + i + 16);

        __m256h h0 = _mm512_cvtxps_ph(f0);
        __m256h h1 = _mm512_cvtxps_ph(f1);

        __m512h out = _mm512_castph256_ph512(h0);
        out = _mm512_insertf64x4(_mm512_castph_pd(out), _mm256_castph_pd(h1), 1);
        _mm512_storeu_ph(result + i, _mm512_castpd_ph(out));
    }

    // Process 16 float32 -> 16 float16 at a time (1 ZMM input -> 1 YMM output)
    for (; i + 15 < n; i += 16) {
        __m512 fv = _mm512_loadu_ps(a + i);
        __m256h hv = _mm512_cvtxps_ph(fv);
        _mm256_storeu_ph(result + i, hv);
    }

    // Process 8 float32 -> 8 float16 at a time
    for (; i + 7 < n; i += 8) {
        __m256 fv = _mm256_loadu_ps(a + i);
        __m128h hv = _mm256_cvtxps_ph(fv);
        _mm_storeu_ph(result + i, hv);
    }

    // Scalar remainder
    for (; i < n; i++) {
        __m128 fv = _mm_set_ss(a[i]);
        __m128h hv = _mm_cvtss_sh(_mm_setzero_ph(), fv);
        _mm_store_sh(result + i, hv);
    }
}
