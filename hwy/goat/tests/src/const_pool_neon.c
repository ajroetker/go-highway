// Constant pool test for ARM64 NEON
// These functions use vector constants that the compiler places in constant pools.
// If GoAT's constant pool support is broken, these will produce wrong results.

#include <arm_neon.h>

// Add a known constant {10, 20, 30, 40} to each input lane.
// The constant vector is loaded from a constant pool (lCPI0_0).
float32x4_t add_const_f32x4(float32x4_t v) {
    float32x4_t c = (float32x4_t){10.0f, 20.0f, 30.0f, 40.0f};
    return vaddq_f32(v, c);
}

// Multiply input lanes by index weights {1, 2, 3, 4}.
// Uses a separate constant pool entry.
float32x4_t mul_index_f32x4(float32x4_t v) {
    float32x4_t weights = (float32x4_t){1.0f, 2.0f, 3.0f, 4.0f};
    return vmulq_f32(v, weights);
}

// Weighted sum: sum(v[i] * {0.25, 0.25, 0.25, 0.25})
// Tests that a uniform constant is handled correctly.
float weighted_sum_f32x4(float32x4_t v) {
    float32x4_t quarter = (float32x4_t){0.25f, 0.25f, 0.25f, 0.25f};
    float32x4_t scaled = vmulq_f32(v, quarter);
    return vaddvq_f32(scaled);
}
