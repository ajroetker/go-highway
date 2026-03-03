// Constant pool test for AMD64 AVX
// These functions use vector constants that the compiler places in constant pools.
// If GoAT's constant pool support is broken, these will produce wrong results.

#include <immintrin.h>

// Add a known constant {10, 20, 30, 40, 50, 60, 70, 80} to each input lane.
// The constant vector is loaded from a constant pool (.LCPI0_0).
__m256 add_const_ps256(__m256 v) {
    __m256 c = _mm256_set_ps(80.0f, 70.0f, 60.0f, 50.0f, 40.0f, 30.0f, 20.0f, 10.0f);
    return _mm256_add_ps(v, c);
}

// Multiply input lanes by index weights {1, 2, 3, 4, 5, 6, 7, 8}.
// Uses a separate constant pool entry.
__m256 mul_index_ps256(__m256 v) {
    __m256 weights = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
    return _mm256_mul_ps(v, weights);
}

// Horizontal sum via reduction
float hsum_ps256(__m256 v) {
    // Sum high and low 128-bit halves
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(hi, lo);
    // Horizontal add within 128-bit
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
