// Float16 conversions for x86-64 with F16C extension (AVX2)
// Used with GoAT to generate Go assembly
// Compile with: -mf16c -mavx2
//
// NOTE: x86 F16C provides conversion-only instructions (VCVTPH2PS, VCVTPS2PH).
// There is NO native float16 arithmetic on F16C - arithmetic must use the
// promote -> compute (float32) -> demote pattern.

#include <immintrin.h>

// ============================================================================
// Float16 <-> Float32 Conversions using F16C
// ============================================================================

// Promote float16 to float32 using F16C (VCVTPH2PS)
// a: array of uint16 (float16 bit patterns)
// result: array of float32
// len: pointer to array length
void promote_f16_to_f32_f16c(unsigned short *a, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 -> 32 float32 at a time (4 AVX2 operations)
    for (; i + 31 < n; i += 32) {
        // Load 8 float16 (128-bit) and convert to 8 float32 (256-bit)
        __m128i h0 = _mm_loadu_si128((__m128i*)(a + i));
        __m128i h1 = _mm_loadu_si128((__m128i*)(a + i + 8));
        __m128i h2 = _mm_loadu_si128((__m128i*)(a + i + 16));
        __m128i h3 = _mm_loadu_si128((__m128i*)(a + i + 24));

        __m256 f0 = _mm256_cvtph_ps(h0);  // VCVTPH2PS ymm, xmm
        __m256 f1 = _mm256_cvtph_ps(h1);
        __m256 f2 = _mm256_cvtph_ps(h2);
        __m256 f3 = _mm256_cvtph_ps(h3);

        _mm256_storeu_ps(result + i, f0);
        _mm256_storeu_ps(result + i + 8, f1);
        _mm256_storeu_ps(result + i + 16, f2);
        _mm256_storeu_ps(result + i + 24, f3);
    }

    // Process 8 float16 -> 8 float32 at a time (AVX2)
    for (; i + 7 < n; i += 8) {
        __m128i h = _mm_loadu_si128((__m128i*)(a + i));
        __m256 f = _mm256_cvtph_ps(h);  // VCVTPH2PS ymm, xmm
        _mm256_storeu_ps(result + i, f);
    }

    // Process 4 float16 -> 4 float32 at a time (SSE)
    for (; i + 3 < n; i += 4) {
        __m128i h = _mm_loadl_epi64((__m128i*)(a + i));
        __m128 f = _mm_cvtph_ps(h);  // VCVTPH2PS xmm, xmm
        _mm_storeu_ps(result + i, f);
    }

    // Scalar remainder - inline bit manipulation conversion
    for (; i < n; i++) {
        unsigned int bits = a[i];
        unsigned int sign = (bits >> 15) & 1;
        unsigned int exp = (bits >> 10) & 0x1F;
        unsigned int mant = bits & 0x3FF;

        unsigned int f32_bits;

        // Zero case (positive or negative)
        if (exp == 0) {
            if (mant == 0) {
                f32_bits = sign << 31;
            }
            // Denormal -> convert to normalized float32
            // For simplicity, treat small denormals as zero
            if (mant != 0) {
                // Denormal: find leading 1 and normalize
                unsigned int e = 1;
                unsigned int m = mant;
                for (; (m & 0x400) == 0; ) {
                    m <<= 1;
                    e--;
                }
                m &= 0x3FF;
                f32_bits = (sign << 31) | ((e + 127 - 15) << 23) | (m << 13);
            }
            if (mant == 0) {
                *(unsigned int*)(result + i) = sign << 31;
            }
            if (mant != 0) {
                unsigned int e = 1;
                unsigned int m = mant;
                for (; (m & 0x400) == 0; ) {
                    m <<= 1;
                    e--;
                }
                m &= 0x3FF;
                *(unsigned int*)(result + i) = (sign << 31) | ((e + 127 - 15) << 23) | (m << 13);
            }
        }

        // Infinity or NaN case
        if (exp == 31) {
            if (mant == 0) {
                // Infinity
                f32_bits = (sign << 31) | 0x7F800000;
            }
            if (mant != 0) {
                // NaN - preserve some mantissa bits
                f32_bits = (sign << 31) | 0x7FC00000 | (mant << 13);
            }
            *(unsigned int*)(result + i) = (sign << 31) | 0x7F800000 | (mant << 13);
        }

        // Normal case: rebias exponent from 15 to 127
        if (exp > 0) {
            if (exp < 31) {
                f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                *(unsigned int*)(result + i) = f32_bits;
            }
        }
    }
}

// Demote float32 to float16 using F16C (VCVTPS2PH)
// a: array of float32
// result: array of uint16 (float16 bit patterns)
// len: pointer to array length
// Uses round-to-nearest-even (imm8 = 0)
void demote_f32_to_f16_f16c(float *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float32 -> 32 float16 at a time (4 AVX2 operations)
    for (; i + 31 < n; i += 32) {
        __m256 f0 = _mm256_loadu_ps(a + i);
        __m256 f1 = _mm256_loadu_ps(a + i + 8);
        __m256 f2 = _mm256_loadu_ps(a + i + 16);
        __m256 f3 = _mm256_loadu_ps(a + i + 24);

        // imm8 = 0: round to nearest even
        __m128i h0 = _mm256_cvtps_ph(f0, 0);  // VCVTPS2PH xmm, ymm, imm8
        __m128i h1 = _mm256_cvtps_ph(f1, 0);
        __m128i h2 = _mm256_cvtps_ph(f2, 0);
        __m128i h3 = _mm256_cvtps_ph(f3, 0);

        _mm_storeu_si128((__m128i*)(result + i), h0);
        _mm_storeu_si128((__m128i*)(result + i + 8), h1);
        _mm_storeu_si128((__m128i*)(result + i + 16), h2);
        _mm_storeu_si128((__m128i*)(result + i + 24), h3);
    }

    // Process 8 float32 -> 8 float16 at a time (AVX2)
    for (; i + 7 < n; i += 8) {
        __m256 f = _mm256_loadu_ps(a + i);
        __m128i h = _mm256_cvtps_ph(f, 0);  // VCVTPS2PH xmm, ymm, imm8
        _mm_storeu_si128((__m128i*)(result + i), h);
    }

    // Process 4 float32 -> 4 float16 at a time (SSE)
    for (; i + 3 < n; i += 4) {
        __m128 f = _mm_loadu_ps(a + i);
        __m128i h = _mm_cvtps_ph(f, 0);  // VCVTPS2PH xmm, xmm, imm8
        _mm_storel_epi64((__m128i*)(result + i), h);
    }

    // Scalar remainder - inline bit manipulation conversion
    for (; i < n; i++) {
        unsigned int bits = *(unsigned int*)(a + i);
        unsigned int sign = (bits >> 16) & 0x8000;
        int exp = ((bits >> 23) & 0xFF) - 127 + 15;  // Rebias from 127 to 15
        unsigned int mant = bits & 0x7FFFFF;

        unsigned short h;

        // Underflow to zero
        if (exp <= 0) {
            if (exp < -10) {
                // Too small even for denormal
                h = sign;
                result[i] = h;
            }
            if (exp >= -10) {
                // Denormalized result
                mant = (mant | 0x800000) >> (1 - exp);
                // Round to nearest even
                if ((mant & 0x1000) != 0) {
                    if ((mant & 0x2FFF) != 0) {
                        mant += 0x2000;
                    }
                }
                h = sign | (mant >> 13);
                result[i] = h;
            }
        }

        // Overflow to infinity
        if (exp >= 31) {
            // Check if original was NaN
            if ((bits & 0x7F800000) == 0x7F800000) {
                if (mant != 0) {
                    // NaN
                    h = sign | 0x7E00 | (mant >> 13);
                    result[i] = h;
                }
                if (mant == 0) {
                    // Infinity
                    h = sign | 0x7C00;
                    result[i] = h;
                }
            }
            if ((bits & 0x7F800000) != 0x7F800000) {
                // Overflow to infinity
                h = sign | 0x7C00;
                result[i] = h;
            }
        }

        // Normal case
        if (exp > 0) {
            if (exp < 31) {
                // Round to nearest even
                if ((mant & 0x1000) != 0) {
                    if ((mant & 0x2FFF) != 0) {
                        mant += 0x2000;
                        if ((mant & 0x800000) != 0) {
                            mant = 0;
                            exp++;
                            if (exp >= 31) {
                                h = sign | 0x7C00;
                                result[i] = h;
                            }
                        }
                    }
                }
                if (exp < 31) {
                    h = sign | (exp << 10) | (mant >> 13);
                    result[i] = h;
                }
            }
        }
    }
}
