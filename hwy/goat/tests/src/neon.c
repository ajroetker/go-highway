// NEON vector type tests for GOAT
// These test passing NEON vectors by value (register-resident operations)

// GOAT's C parser uses GOAT_PARSER=1 with stub type definitions from prologue.
// During compilation, clang includes the real intrinsics header.
#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// Add two float32x4 vectors
float32x4_t add_f32x4(float32x4_t a, float32x4_t b) {
    return vaddq_f32(a, b);
}

// Multiply two float32x4 vectors
float32x4_t mul_f32x4(float32x4_t a, float32x4_t b) {
    return vmulq_f32(a, b);
}

// Fused multiply-add: a * b + c
float32x4_t fma_f32x4(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vfmaq_f32(c, a, b);
}

// Add two int32x4 vectors
int32x4_t add_i32x4(int32x4_t a, int32x4_t b) {
    return vaddq_s32(a, b);
}

// Horizontal sum of float32x4 (returns scalar)
float hsum_f32x4(float32x4_t v) {
    return vaddvq_f32(v);
}

// Dot product of two float32x4 vectors
float dot_f32x4(float32x4_t a, float32x4_t b) {
    float32x4_t prod = vmulq_f32(a, b);
    return vaddvq_f32(prod);
}

// ============ 64-bit NEON types ============

// Add two float32x2 vectors (64-bit)
float32x2_t add_f32x2(float32x2_t a, float32x2_t b) {
    return vadd_f32(a, b);
}

// Multiply two float32x2 vectors (64-bit)
float32x2_t mul_f32x2(float32x2_t a, float32x2_t b) {
    return vmul_f32(a, b);
}

// Add two int32x2 vectors (64-bit)
int32x2_t add_i32x2(int32x2_t a, int32x2_t b) {
    return vadd_s32(a, b);
}

// Horizontal sum of float32x2 (returns scalar)
float hsum_f32x2(float32x2_t v) {
    return vaddv_f32(v);
}

// ============ NEON array types (x2) ============

// Add corresponding vectors in two float32x4x2 arrays
float32x4x2_t add_f32x4x2(float32x4x2_t a, float32x4x2_t b) {
    float32x4x2_t result;
    result.val[0] = vaddq_f32(a.val[0], b.val[0]);
    result.val[1] = vaddq_f32(a.val[1], b.val[1]);
    return result;
}

// Create a float32x4x2 from two vectors
float32x4x2_t make_f32x4x2(float32x4_t a, float32x4_t b) {
    float32x4x2_t result;
    result.val[0] = a;
    result.val[1] = b;
    return result;
}

// Sum all elements in a float32x4x2
float sum_f32x4x2(float32x4x2_t v) {
    float sum0 = vaddvq_f32(v.val[0]);
    float sum1 = vaddvq_f32(v.val[1]);
    return sum0 + sum1;
}
