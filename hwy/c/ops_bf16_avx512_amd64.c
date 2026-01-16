// BFloat16 SIMD operations for x86-64 with AVX-512 BF16 extension
// Used with GoAT to generate Go assembly
// Compile with: -mavx512bf16
//
// Requires: Intel Cooper Lake (2020+), AMD Zen4+
// Provides: BFloat16 conversions and dot product with F32 accumulator
//
// AVX-512 BF16 does NOT provide native BF16 arithmetic (add, sub, mul, div).
// It provides:
// - VCVTNE2PS2BF16: Convert two F32 vectors to one BF16 vector
// - VCVTNEPS2BF16: Convert F32 to BF16 (round to nearest even)
// - VDPBF16PS: BF16 dot product with F32 accumulator (KEY for ML!)
//
// For BF16 arithmetic, use the promote -> compute (F32) -> demote pattern.

#include <immintrin.h>

// ============================================================================
// BFloat16 <-> Float32 Conversions
// ============================================================================

// Demote float32 to bfloat16 using VCVTNEPS2BF16
// a: array of float32
// result: array of uint16 (bfloat16 bit patterns)
// len: pointer to array length
void demote_f32_to_bf16_avx512(float *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 64 float32 -> 64 bfloat16 at a time (4 ZMM inputs -> 2 ZMM outputs)
    for (; i + 63 < n; i += 64) {
        __m512 f0 = _mm512_loadu_ps(a + i);
        __m512 f1 = _mm512_loadu_ps(a + i + 16);
        __m512 f2 = _mm512_loadu_ps(a + i + 32);
        __m512 f3 = _mm512_loadu_ps(a + i + 48);

        // VCVTNEPS2BF16 zmm -> ymm (16 F32 -> 16 BF16)
        __m256bh h0 = _mm512_cvtneps_pbh(f0);
        __m256bh h1 = _mm512_cvtneps_pbh(f1);
        __m256bh h2 = _mm512_cvtneps_pbh(f2);
        __m256bh h3 = _mm512_cvtneps_pbh(f3);

        // Store combined results
        _mm256_storeu_si256((__m256i*)(result + i), (__m256i)h0);
        _mm256_storeu_si256((__m256i*)(result + i + 16), (__m256i)h1);
        _mm256_storeu_si256((__m256i*)(result + i + 32), (__m256i)h2);
        _mm256_storeu_si256((__m256i*)(result + i + 48), (__m256i)h3);
    }

    // Process 32 float32 -> 32 bfloat16 at a time (2 ZMM inputs -> 1 ZMM output)
    for (; i + 31 < n; i += 32) {
        __m512 f0 = _mm512_loadu_ps(a + i);
        __m512 f1 = _mm512_loadu_ps(a + i + 16);

        __m256bh h0 = _mm512_cvtneps_pbh(f0);
        __m256bh h1 = _mm512_cvtneps_pbh(f1);

        _mm256_storeu_si256((__m256i*)(result + i), (__m256i)h0);
        _mm256_storeu_si256((__m256i*)(result + i + 16), (__m256i)h1);
    }

    // Process 16 float32 -> 16 bfloat16 at a time (1 ZMM input -> 1 YMM output)
    for (; i + 15 < n; i += 16) {
        __m512 fv = _mm512_loadu_ps(a + i);
        __m256bh hv = _mm512_cvtneps_pbh(fv);  // VCVTNEPS2BF16 zmm, ymm
        _mm256_storeu_si256((__m256i*)(result + i), (__m256i)hv);
    }

    // Process 8 float32 -> 8 bfloat16 at a time (using AVX2 fallback)
    for (; i + 7 < n; i += 8) {
        __m256 fv = _mm256_loadu_ps(a + i);
        __m128bh hv = _mm256_cvtneps_pbh(fv);  // VCVTNEPS2BF16 ymm, xmm
        _mm_storeu_si128((__m128i*)(result + i), (__m128i)hv);
    }

    // Scalar remainder: BF16 = upper 16 bits of F32 with rounding
    for (; i < n; i++) {
        unsigned int bits = *(unsigned int*)(a + i);

        // Check for NaN
        if ((bits & 0x7FFFFFFF) > 0x7F800000) {
            // NaN: preserve sign, set quiet NaN in BF16
            result[i] = (unsigned short)((bits >> 16) | 0x0040);
        }

        // Normal conversion with round-to-nearest-even
        if ((bits & 0x7FFFFFFF) <= 0x7F800000) {
            // Add rounding bias: 0x7FFF + bit 16 (for round-to-nearest-even)
            unsigned int rounding = 0x7FFF + ((bits >> 16) & 1);
            bits += rounding;
            result[i] = (unsigned short)(bits >> 16);
        }
    }
}

// Demote two float32 vectors to one bfloat16 vector using VCVTNE2PS2BF16
// lo: lower half float32 values
// hi: upper half float32 values
// result: array of uint16 (bfloat16 bit patterns)
// len: pointer to length (number of F32 pairs, result will have 2x elements)
//
// This packs two F32 vectors into one BF16 vector:
//   result[0..15] = demote(lo[0..15])
//   result[16..31] = demote(hi[0..15])
void demote_two_f32_to_bf16_avx512(float *lo, float *hi, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 64 float32 pairs -> 64 bfloat16 at a time (2 ZMM pairs -> 2 ZMM outputs)
    for (; i + 31 < n; i += 32) {
        __m512 lo0 = _mm512_loadu_ps(lo + i);
        __m512 hi0 = _mm512_loadu_ps(hi + i);
        __m512 lo1 = _mm512_loadu_ps(lo + i + 16);
        __m512 hi1 = _mm512_loadu_ps(hi + i + 16);

        // VCVTNE2PS2BF16 zmm, zmm, zmm: interleaves lo and hi conversions
        __m512bh r0 = _mm512_cvtne2ps_pbh(hi0, lo0);
        __m512bh r1 = _mm512_cvtne2ps_pbh(hi1, lo1);

        _mm512_storeu_si512(result + i * 2, (__m512i)r0);
        _mm512_storeu_si512(result + i * 2 + 32, (__m512i)r1);
    }

    // Process 16 float32 pairs -> 32 bfloat16 at a time (1 ZMM pair -> 1 ZMM output)
    for (; i + 15 < n; i += 16) {
        __m512 lo_v = _mm512_loadu_ps(lo + i);
        __m512 hi_v = _mm512_loadu_ps(hi + i);

        // VCVTNE2PS2BF16: result = [cvt(lo), cvt(hi)] interleaved
        __m512bh rv = _mm512_cvtne2ps_pbh(hi_v, lo_v);
        _mm512_storeu_si512(result + i * 2, (__m512i)rv);
    }

    // Process 8 float32 pairs -> 16 bfloat16 at a time (1 YMM pair -> 1 YMM output)
    for (; i + 7 < n; i += 8) {
        __m256 lo_v = _mm256_loadu_ps(lo + i);
        __m256 hi_v = _mm256_loadu_ps(hi + i);

        __m256bh rv = _mm256_cvtne2ps_pbh(hi_v, lo_v);
        _mm256_storeu_si256((__m256i*)(result + i * 2), (__m256i)rv);
    }

    // Scalar remainder
    for (; i < n; i++) {
        unsigned int lo_bits = *(unsigned int*)(lo + i);
        unsigned int hi_bits = *(unsigned int*)(hi + i);

        // Convert lo
        if ((lo_bits & 0x7FFFFFFF) > 0x7F800000) {
            result[i * 2] = (unsigned short)((lo_bits >> 16) | 0x0040);
        }
        if ((lo_bits & 0x7FFFFFFF) <= 0x7F800000) {
            unsigned int rounding = 0x7FFF + ((lo_bits >> 16) & 1);
            lo_bits += rounding;
            result[i * 2] = (unsigned short)(lo_bits >> 16);
        }

        // Convert hi
        if ((hi_bits & 0x7FFFFFFF) > 0x7F800000) {
            result[i * 2 + 1] = (unsigned short)((hi_bits >> 16) | 0x0040);
        }
        if ((hi_bits & 0x7FFFFFFF) <= 0x7F800000) {
            unsigned int rounding = 0x7FFF + ((hi_bits >> 16) & 1);
            hi_bits += rounding;
            result[i * 2 + 1] = (unsigned short)(hi_bits >> 16);
        }
    }
}

// Promote bfloat16 to float32 (simple bit shift, no special instruction needed)
// a: array of uint16 (bfloat16 bit patterns)
// result: array of float32
// len: pointer to array length
//
// BFloat16 to Float32 is trivial: just left-shift by 16 bits.
// No special AVX-512 BF16 instruction exists for this - it's just a shuffle/shift.
void promote_bf16_to_f32_avx512(unsigned short *a, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 bfloat16 -> 32 float32 at a time
    // We unpack BF16 by zero-extending and shifting
    for (; i + 31 < n; i += 32) {
        // Load 32 BF16 values as 16-bit integers
        __m256i h0 = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i h1 = _mm256_loadu_si256((__m256i*)(a + i + 16));

        // Unpack to 32-bit and shift left by 16
        // Low halves
        __m256i lo0_16 = _mm256_unpacklo_epi16(_mm256_setzero_si256(), h0);
        __m256i hi0_16 = _mm256_unpackhi_epi16(_mm256_setzero_si256(), h0);
        __m256i lo1_16 = _mm256_unpacklo_epi16(_mm256_setzero_si256(), h1);
        __m256i hi1_16 = _mm256_unpackhi_epi16(_mm256_setzero_si256(), h1);

        // Store as float32 (reinterpret cast)
        // Need to fix lane ordering due to AVX2 lane semantics
        __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        __m256i f0_lo = _mm256_permutevar8x32_epi32(lo0_16, perm);
        __m256i f0_hi = _mm256_permutevar8x32_epi32(hi0_16, perm);
        __m256i f1_lo = _mm256_permutevar8x32_epi32(lo1_16, perm);
        __m256i f1_hi = _mm256_permutevar8x32_epi32(hi1_16, perm);

        _mm256_storeu_si256((__m256i*)(result + i), f0_lo);
        _mm256_storeu_si256((__m256i*)(result + i + 8), f0_hi);
        _mm256_storeu_si256((__m256i*)(result + i + 16), f1_lo);
        _mm256_storeu_si256((__m256i*)(result + i + 24), f1_hi);
    }

    // Process 16 bfloat16 -> 16 float32 at a time
    for (; i + 15 < n; i += 16) {
        __m256i hv = _mm256_loadu_si256((__m256i*)(a + i));

        __m256i lo_16 = _mm256_unpacklo_epi16(_mm256_setzero_si256(), hv);
        __m256i hi_16 = _mm256_unpackhi_epi16(_mm256_setzero_si256(), hv);

        __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        __m256i f_lo = _mm256_permutevar8x32_epi32(lo_16, perm);
        __m256i f_hi = _mm256_permutevar8x32_epi32(hi_16, perm);

        _mm256_storeu_si256((__m256i*)(result + i), f_lo);
        _mm256_storeu_si256((__m256i*)(result + i + 8), f_hi);
    }

    // Process 8 bfloat16 -> 8 float32 at a time
    for (; i + 7 < n; i += 8) {
        __m128i hv = _mm_loadu_si128((__m128i*)(a + i));

        // Unpack low and high parts
        __m128i lo = _mm_unpacklo_epi16(_mm_setzero_si128(), hv);
        __m128i hi = _mm_unpackhi_epi16(_mm_setzero_si128(), hv);

        _mm_storeu_si128((__m128i*)(result + i), lo);
        _mm_storeu_si128((__m128i*)(result + i + 4), hi);
    }

    // Scalar remainder
    for (; i < n; i++) {
        *(unsigned int*)(result + i) = (unsigned int)a[i] << 16;
    }
}

// ============================================================================
// BFloat16 Dot Product with Float32 Accumulator (VDPBF16PS)
// This is the KEY operation for ML inference and training!
// ============================================================================

// BF16 dot product with F32 accumulator using VDPBF16PS
// Computes: acc[lane] += sum(a[i*2:i*2+1] * b[i*2:i*2+1]) for each lane
//
// VDPBF16PS processes pairs of BF16 values:
// - Each 512-bit operation processes 32 BF16 pairs (64 BF16 values total)
// - Each pair is multiplied and accumulated into 16 F32 lanes
// - Result: 16 F32 accumulator lanes
//
// a: array of uint16 (bfloat16 bit patterns), must be even length
// b: array of uint16 (bfloat16 bit patterns), must be even length
// acc: array of float32 accumulators (16 elements for full utilization)
// len: pointer to number of BF16 elements to process (will be rounded down to even)
void dot_bf16_avx512(unsigned short *a, unsigned short *b, float *acc, long *len) {
    long n = *len;
    long i = 0;

    // Initialize accumulator from input
    __m512 sum = _mm512_loadu_ps(acc);

    // Process 128 BF16 pairs at a time (4 ZMM register pairs)
    for (; i + 127 < n; i += 128) {
        __m512bh a0 = (__m512bh)_mm512_loadu_si512(a + i);
        __m512bh b0 = (__m512bh)_mm512_loadu_si512(b + i);
        __m512bh a1 = (__m512bh)_mm512_loadu_si512(a + i + 32);
        __m512bh b1 = (__m512bh)_mm512_loadu_si512(b + i + 32);
        __m512bh a2 = (__m512bh)_mm512_loadu_si512(a + i + 64);
        __m512bh b2 = (__m512bh)_mm512_loadu_si512(b + i + 64);
        __m512bh a3 = (__m512bh)_mm512_loadu_si512(a + i + 96);
        __m512bh b3 = (__m512bh)_mm512_loadu_si512(b + i + 96);

        // VDPBF16PS: dot product of BF16 pairs, accumulate to F32
        sum = _mm512_dpbf16_ps(sum, a0, b0);
        sum = _mm512_dpbf16_ps(sum, a1, b1);
        sum = _mm512_dpbf16_ps(sum, a2, b2);
        sum = _mm512_dpbf16_ps(sum, a3, b3);
    }

    // Process 32 BF16 pairs at a time (1 ZMM register pair)
    for (; i + 31 < n; i += 32) {
        __m512bh av = (__m512bh)_mm512_loadu_si512(a + i);
        __m512bh bv = (__m512bh)_mm512_loadu_si512(b + i);

        // VDPBF16PS zmm, zmm, zmm
        sum = _mm512_dpbf16_ps(sum, av, bv);
    }

    // Process 16 BF16 pairs at a time (1 YMM register pair)
    for (; i + 15 < n; i += 16) {
        __m256bh av = (__m256bh)_mm256_loadu_si256((__m256i*)(a + i));
        __m256bh bv = (__m256bh)_mm256_loadu_si256((__m256i*)(b + i));

        // Need to use 256-bit version and widen accumulator
        __m256 sum256 = _mm512_castps512_ps256(sum);
        sum256 = _mm256_dpbf16_ps(sum256, av, bv);
        sum = _mm512_castps256_ps512(sum256);
        // Keep upper 256 bits unchanged
        sum = _mm512_insertf32x8(sum, _mm512_extractf32x8_ps(sum, 1), 1);
    }

    // Store accumulator result
    _mm512_storeu_ps(acc, sum);

    // Scalar remainder: promote to F32, multiply, accumulate
    for (; i + 1 < n; i += 2) {
        float a0_f = *(float*)&(unsigned int){(unsigned int)a[i] << 16};
        float a1_f = *(float*)&(unsigned int){(unsigned int)a[i + 1] << 16};
        float b0_f = *(float*)&(unsigned int){(unsigned int)b[i] << 16};
        float b1_f = *(float*)&(unsigned int){(unsigned int)b[i + 1] << 16};

        acc[0] += a0_f * b0_f + a1_f * b1_f;
    }
}

// ============================================================================
// BFloat16 Arithmetic via Promote-Compute-Demote Pattern
// AVX-512 BF16 does NOT have native arithmetic; these use F32 intermediate.
// ============================================================================

// Vector addition: result[i] = a[i] + b[i]
// Uses promote -> F32 add -> demote pattern
void add_bf16_avx512(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 bfloat16 at a time
    for (; i + 31 < n; i += 32) {
        // Load and promote to F32
        __m256i ah0 = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i ah1 = _mm256_loadu_si256((__m256i*)(a + i + 16));
        __m256i bh0 = _mm256_loadu_si256((__m256i*)(b + i));
        __m256i bh1 = _mm256_loadu_si256((__m256i*)(b + i + 16));

        // Unpack BF16 to F32 (zero-extend and shift)
        __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        __m256i a0_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), ah0), perm);
        __m256i a0_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), ah0), perm);
        __m256i a1_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), ah1), perm);
        __m256i a1_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), ah1), perm);

        __m256i b0_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), bh0), perm);
        __m256i b0_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), bh0), perm);
        __m256i b1_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), bh1), perm);
        __m256i b1_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), bh1), perm);

        // Add as F32
        __m256 r0_lo = _mm256_add_ps(_mm256_castsi256_ps(a0_lo), _mm256_castsi256_ps(b0_lo));
        __m256 r0_hi = _mm256_add_ps(_mm256_castsi256_ps(a0_hi), _mm256_castsi256_ps(b0_hi));
        __m256 r1_lo = _mm256_add_ps(_mm256_castsi256_ps(a1_lo), _mm256_castsi256_ps(b1_lo));
        __m256 r1_hi = _mm256_add_ps(_mm256_castsi256_ps(a1_hi), _mm256_castsi256_ps(b1_hi));

        // Convert results from 4x8 F32 to 2x16 BF16 using AVX-512 instructions
        __m512 f0 = _mm512_insertf32x8(_mm512_castps256_ps512(r0_lo), r0_hi, 1);
        __m512 f1 = _mm512_insertf32x8(_mm512_castps256_ps512(r1_lo), r1_hi, 1);

        __m256bh h0 = _mm512_cvtneps_pbh(f0);
        __m256bh h1 = _mm512_cvtneps_pbh(f1);

        _mm256_storeu_si256((__m256i*)(result + i), (__m256i)h0);
        _mm256_storeu_si256((__m256i*)(result + i + 16), (__m256i)h1);
    }

    // Process 16 bfloat16 at a time
    for (; i + 15 < n; i += 16) {
        __m256i ah = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i bh = _mm256_loadu_si256((__m256i*)(b + i));

        __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        __m256i a_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), ah), perm);
        __m256i a_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), ah), perm);
        __m256i b_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), bh), perm);
        __m256i b_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), bh), perm);

        __m256 r_lo = _mm256_add_ps(_mm256_castsi256_ps(a_lo), _mm256_castsi256_ps(b_lo));
        __m256 r_hi = _mm256_add_ps(_mm256_castsi256_ps(a_hi), _mm256_castsi256_ps(b_hi));

        __m512 fv = _mm512_insertf32x8(_mm512_castps256_ps512(r_lo), r_hi, 1);
        __m256bh hv = _mm512_cvtneps_pbh(fv);

        _mm256_storeu_si256((__m256i*)(result + i), (__m256i)hv);
    }

    // Scalar remainder
    for (; i < n; i++) {
        float af = *(float*)&(unsigned int){(unsigned int)a[i] << 16};
        float bf = *(float*)&(unsigned int){(unsigned int)b[i] << 16};
        float rf = af + bf;

        unsigned int bits = *(unsigned int*)&rf;
        if ((bits & 0x7FFFFFFF) > 0x7F800000) {
            result[i] = (unsigned short)((bits >> 16) | 0x0040);
        }
        if ((bits & 0x7FFFFFFF) <= 0x7F800000) {
            unsigned int rounding = 0x7FFF + ((bits >> 16) & 1);
            bits += rounding;
            result[i] = (unsigned short)(bits >> 16);
        }
    }
}

// Vector multiplication: result[i] = a[i] * b[i]
// Uses promote -> F32 mul -> demote pattern
void mul_bf16_avx512(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 bfloat16 at a time
    for (; i + 15 < n; i += 16) {
        __m256i ah = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i bh = _mm256_loadu_si256((__m256i*)(b + i));

        __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        __m256i a_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), ah), perm);
        __m256i a_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), ah), perm);
        __m256i b_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), bh), perm);
        __m256i b_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), bh), perm);

        __m256 r_lo = _mm256_mul_ps(_mm256_castsi256_ps(a_lo), _mm256_castsi256_ps(b_lo));
        __m256 r_hi = _mm256_mul_ps(_mm256_castsi256_ps(a_hi), _mm256_castsi256_ps(b_hi));

        __m512 fv = _mm512_insertf32x8(_mm512_castps256_ps512(r_lo), r_hi, 1);
        __m256bh hv = _mm512_cvtneps_pbh(fv);

        _mm256_storeu_si256((__m256i*)(result + i), (__m256i)hv);
    }

    // Scalar remainder
    for (; i < n; i++) {
        float af = *(float*)&(unsigned int){(unsigned int)a[i] << 16};
        float bf = *(float*)&(unsigned int){(unsigned int)b[i] << 16};
        float rf = af * bf;

        unsigned int bits = *(unsigned int*)&rf;
        if ((bits & 0x7FFFFFFF) > 0x7F800000) {
            result[i] = (unsigned short)((bits >> 16) | 0x0040);
        }
        if ((bits & 0x7FFFFFFF) <= 0x7F800000) {
            unsigned int rounding = 0x7FFF + ((bits >> 16) & 1);
            bits += rounding;
            result[i] = (unsigned short)(bits >> 16);
        }
    }
}

// Fused multiply-add: result[i] = a[i] * b[i] + c[i]
// Uses promote -> F32 FMA -> demote pattern
void fma_bf16_avx512(unsigned short *a, unsigned short *b, unsigned short *c, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 bfloat16 at a time
    for (; i + 15 < n; i += 16) {
        __m256i ah = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i bh = _mm256_loadu_si256((__m256i*)(b + i));
        __m256i ch = _mm256_loadu_si256((__m256i*)(c + i));

        __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        __m256i a_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), ah), perm);
        __m256i a_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), ah), perm);
        __m256i b_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), bh), perm);
        __m256i b_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), bh), perm);
        __m256i c_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), ch), perm);
        __m256i c_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), ch), perm);

        __m256 r_lo = _mm256_fmadd_ps(_mm256_castsi256_ps(a_lo), _mm256_castsi256_ps(b_lo), _mm256_castsi256_ps(c_lo));
        __m256 r_hi = _mm256_fmadd_ps(_mm256_castsi256_ps(a_hi), _mm256_castsi256_ps(b_hi), _mm256_castsi256_ps(c_hi));

        __m512 fv = _mm512_insertf32x8(_mm512_castps256_ps512(r_lo), r_hi, 1);
        __m256bh hv = _mm512_cvtneps_pbh(fv);

        _mm256_storeu_si256((__m256i*)(result + i), (__m256i)hv);
    }

    // Scalar remainder
    for (; i < n; i++) {
        float af = *(float*)&(unsigned int){(unsigned int)a[i] << 16};
        float bf = *(float*)&(unsigned int){(unsigned int)b[i] << 16};
        float cf = *(float*)&(unsigned int){(unsigned int)c[i] << 16};
        float rf = af * bf + cf;

        unsigned int bits = *(unsigned int*)&rf;
        if ((bits & 0x7FFFFFFF) > 0x7F800000) {
            result[i] = (unsigned short)((bits >> 16) | 0x0040);
        }
        if ((bits & 0x7FFFFFFF) <= 0x7F800000) {
            unsigned int rounding = 0x7FFF + ((bits >> 16) & 1);
            bits += rounding;
            result[i] = (unsigned short)(bits >> 16);
        }
    }
}
