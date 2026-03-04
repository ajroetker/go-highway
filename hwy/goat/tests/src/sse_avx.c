// Test file for SSE and AVX SIMD operations
// Compile with: clang -O3 -mavx2 -mfma -c sse_avx.c

// During parsing, GOAT defines GOAT_PARSER and provides type definitions.
// During compilation, clang includes the real intrinsics header.
#ifndef GOAT_PARSER
#include <immintrin.h>
#endif

// ============ SSE 128-bit operations ============

// Add two 128-bit vectors of 4 floats
__m128 add_ps(__m128 a, __m128 b) {
    return _mm_add_ps(a, b);
}

// Multiply two 128-bit vectors of 4 floats
__m128 mul_ps(__m128 a, __m128 b) {
    return _mm_mul_ps(a, b);
}

// Fused multiply-add: a * b + c
__m128 fma_ps(__m128 a, __m128 b, __m128 c) {
    return _mm_fmadd_ps(a, b, c);
}

// Add two 128-bit vectors of 2 doubles
__m128d add_pd(__m128d a, __m128d b) {
    return _mm_add_pd(a, b);
}

// Multiply two 128-bit vectors of 2 doubles
__m128d mul_pd(__m128d a, __m128d b) {
    return _mm_mul_pd(a, b);
}

// Add two 128-bit vectors of 4 int32s
__m128i add_epi32(__m128i a, __m128i b) {
    return _mm_add_epi32(a, b);
}

// Horizontal sum of 4 floats
float hsum_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Dot product of two 4-float vectors
float dot_ps(__m128 a, __m128 b) {
    __m128 prod = _mm_mul_ps(a, b);
    __m128 shuf = _mm_movehdup_ps(prod);
    __m128 sums = _mm_add_ps(prod, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// ============ AVX 256-bit operations ============

// Add two 256-bit vectors of 8 floats
__m256 add256_ps(__m256 a, __m256 b) {
    return _mm256_add_ps(a, b);
}

// Multiply two 256-bit vectors of 8 floats
__m256 mul256_ps(__m256 a, __m256 b) {
    return _mm256_mul_ps(a, b);
}

// Fused multiply-add: a * b + c (256-bit)
__m256 fma256_ps(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}

// Add two 256-bit vectors of 4 doubles
__m256d add256_pd(__m256d a, __m256d b) {
    return _mm256_add_pd(a, b);
}

// Multiply two 256-bit vectors of 4 doubles
__m256d mul256_pd(__m256d a, __m256d b) {
    return _mm256_mul_pd(a, b);
}

// Add two 256-bit vectors of 8 int32s
__m256i add256_epi32(__m256i a, __m256i b) {
    return _mm256_add_epi32(a, b);
}

// Horizontal sum of 8 floats
float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Dot product of two 8-float vectors
float dot256_ps(__m256 a, __m256 b) {
    __m256 prod = _mm256_mul_ps(a, b);
    __m128 lo = _mm256_castps256_ps128(prod);
    __m128 hi = _mm256_extractf128_ps(prod, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
