// NEON SIMD operations for ARM64
// Used with GOAT to generate Go assembly
#include <arm_neon.h>

// ============================================================================
// Float32 Operations (4 lanes per 128-bit vector)
// ============================================================================

// Vector addition: result[i] = a[i] + b[i]
void add_f32_neon(float *a, float *b, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 floats at a time (4 vectors)
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(result + i, vaddq_f32(a0, b0));
        vst1q_f32(result + i + 4, vaddq_f32(a1, b1));
        vst1q_f32(result + i + 8, vaddq_f32(a2, b2));
        vst1q_f32(result + i + 12, vaddq_f32(a3, b3));
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_f32(result + i, vaddq_f32(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

// Vector subtraction: result[i] = a[i] - b[i]
void sub_f32_neon(float *a, float *b, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(result + i, vsubq_f32(a0, b0));
        vst1q_f32(result + i + 4, vsubq_f32(a1, b1));
        vst1q_f32(result + i + 8, vsubq_f32(a2, b2));
        vst1q_f32(result + i + 12, vsubq_f32(a3, b3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_f32(result + i, vsubq_f32(av, bv));
    }

    for (; i < n; i++) {
        result[i] = a[i] - b[i];
    }
}

// Vector multiplication: result[i] = a[i] * b[i]
void mul_f32_neon(float *a, float *b, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(result + i, vmulq_f32(a0, b0));
        vst1q_f32(result + i + 4, vmulq_f32(a1, b1));
        vst1q_f32(result + i + 8, vmulq_f32(a2, b2));
        vst1q_f32(result + i + 12, vmulq_f32(a3, b3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_f32(result + i, vmulq_f32(av, bv));
    }

    for (; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

// Vector division: result[i] = a[i] / b[i]
void div_f32_neon(float *a, float *b, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(result + i, vdivq_f32(a0, b0));
        vst1q_f32(result + i + 4, vdivq_f32(a1, b1));
        vst1q_f32(result + i + 8, vdivq_f32(a2, b2));
        vst1q_f32(result + i + 12, vdivq_f32(a3, b3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_f32(result + i, vdivq_f32(av, bv));
    }

    for (; i < n; i++) {
        result[i] = a[i] / b[i];
    }
}

// Fused multiply-add: result[i] = a[i] * b[i] + c[i]
void fma_f32_neon(float *a, float *b, float *c, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        float32x4_t c0 = vld1q_f32(c + i);
        float32x4_t c1 = vld1q_f32(c + i + 4);
        float32x4_t c2 = vld1q_f32(c + i + 8);
        float32x4_t c3 = vld1q_f32(c + i + 12);

        // vfmaq_f32(c, a, b) = a*b + c
        vst1q_f32(result + i, vfmaq_f32(c0, a0, b0));
        vst1q_f32(result + i + 4, vfmaq_f32(c1, a1, b1));
        vst1q_f32(result + i + 8, vfmaq_f32(c2, a2, b2));
        vst1q_f32(result + i + 12, vfmaq_f32(c3, a3, b3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        float32x4_t cv = vld1q_f32(c + i);
        vst1q_f32(result + i, vfmaq_f32(cv, av, bv));
    }

    for (; i < n; i++) {
        result[i] = a[i] * b[i] + c[i];
    }
}

// Vector min: result[i] = min(a[i], b[i])
void min_f32_neon(float *a, float *b, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(result + i, vminq_f32(a0, b0));
        vst1q_f32(result + i + 4, vminq_f32(a1, b1));
        vst1q_f32(result + i + 8, vminq_f32(a2, b2));
        vst1q_f32(result + i + 12, vminq_f32(a3, b3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_f32(result + i, vminq_f32(av, bv));
    }

    for (; i < n; i++) {
        result[i] = a[i] < b[i] ? a[i] : b[i];
    }
}

// Vector max: result[i] = max(a[i], b[i])
void max_f32_neon(float *a, float *b, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(result + i, vmaxq_f32(a0, b0));
        vst1q_f32(result + i + 4, vmaxq_f32(a1, b1));
        vst1q_f32(result + i + 8, vmaxq_f32(a2, b2));
        vst1q_f32(result + i + 12, vmaxq_f32(a3, b3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_f32(result + i, vmaxq_f32(av, bv));
    }

    for (; i < n; i++) {
        result[i] = a[i] > b[i] ? a[i] : b[i];
    }
}

// Horizontal sum reduction
void reduce_sum_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;
    float sum = 0.0f;

    // Process 16 floats at a time with 4 accumulators
    if (n >= 16) {
        float32x4_t sum0 = vdupq_n_f32(0);
        float32x4_t sum1 = vdupq_n_f32(0);
        float32x4_t sum2 = vdupq_n_f32(0);
        float32x4_t sum3 = vdupq_n_f32(0);

        for (; i + 15 < n; i += 16) {
            sum0 = vaddq_f32(sum0, vld1q_f32(input + i));
            sum1 = vaddq_f32(sum1, vld1q_f32(input + i + 4));
            sum2 = vaddq_f32(sum2, vld1q_f32(input + i + 8));
            sum3 = vaddq_f32(sum3, vld1q_f32(input + i + 12));
        }

        // Combine accumulators
        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);

        // Horizontal sum
        sum = vaddvq_f32(sum0);
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        sum += vaddvq_f32(v);
    }

    // Scalar remainder
    for (; i < n; i++) {
        sum += input[i];
    }

    *result = sum;
}

// Horizontal min reduction
void reduce_min_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    if (n <= 0) {
        *result = 0.0f;
        return;
    }

    long i = 0;
    float min_val = input[0];

    if (n >= 16) {
        float32x4_t min0 = vld1q_f32(input);
        float32x4_t min1 = min0;
        float32x4_t min2 = min0;
        float32x4_t min3 = min0;
        i = 4;

        for (; i + 15 < n; i += 16) {
            min0 = vminq_f32(min0, vld1q_f32(input + i));
            min1 = vminq_f32(min1, vld1q_f32(input + i + 4));
            min2 = vminq_f32(min2, vld1q_f32(input + i + 8));
            min3 = vminq_f32(min3, vld1q_f32(input + i + 12));
        }

        min0 = vminq_f32(min0, min1);
        min2 = vminq_f32(min2, min3);
        min0 = vminq_f32(min0, min2);

        min_val = vminvq_f32(min0);
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        float v_min = vminvq_f32(v);
        if (v_min < min_val) min_val = v_min;
    }

    for (; i < n; i++) {
        if (input[i] < min_val) min_val = input[i];
    }

    *result = min_val;
}

// Horizontal max reduction
void reduce_max_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    if (n <= 0) {
        *result = 0.0f;
        return;
    }

    long i = 0;
    float max_val = input[0];

    if (n >= 16) {
        float32x4_t max0 = vld1q_f32(input);
        float32x4_t max1 = max0;
        float32x4_t max2 = max0;
        float32x4_t max3 = max0;
        i = 4;

        for (; i + 15 < n; i += 16) {
            max0 = vmaxq_f32(max0, vld1q_f32(input + i));
            max1 = vmaxq_f32(max1, vld1q_f32(input + i + 4));
            max2 = vmaxq_f32(max2, vld1q_f32(input + i + 8));
            max3 = vmaxq_f32(max3, vld1q_f32(input + i + 12));
        }

        max0 = vmaxq_f32(max0, max1);
        max2 = vmaxq_f32(max2, max3);
        max0 = vmaxq_f32(max0, max2);

        max_val = vmaxvq_f32(max0);
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        float v_max = vmaxvq_f32(v);
        if (v_max > max_val) max_val = v_max;
    }

    for (; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    *result = max_val;
}

// Square root: result[i] = sqrt(a[i])
void sqrt_f32_neon(float *a, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        vst1q_f32(result + i, vsqrtq_f32(a0));
        vst1q_f32(result + i + 4, vsqrtq_f32(a1));
        vst1q_f32(result + i + 8, vsqrtq_f32(a2));
        vst1q_f32(result + i + 12, vsqrtq_f32(a3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        vst1q_f32(result + i, vsqrtq_f32(av));
    }

    for (; i < n; i++) {
        result[i] = __builtin_sqrtf(a[i]);
    }
}

// Absolute value: result[i] = abs(a[i])
void abs_f32_neon(float *a, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        vst1q_f32(result + i, vabsq_f32(a0));
        vst1q_f32(result + i + 4, vabsq_f32(a1));
        vst1q_f32(result + i + 8, vabsq_f32(a2));
        vst1q_f32(result + i + 12, vabsq_f32(a3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        vst1q_f32(result + i, vabsq_f32(av));
    }

    for (; i < n; i++) {
        result[i] = a[i] < 0 ? -a[i] : a[i];
    }
}

// Negation: result[i] = -a[i]
void neg_f32_neon(float *a, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        vst1q_f32(result + i, vnegq_f32(a0));
        vst1q_f32(result + i + 4, vnegq_f32(a1));
        vst1q_f32(result + i + 8, vnegq_f32(a2));
        vst1q_f32(result + i + 12, vnegq_f32(a3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        vst1q_f32(result + i, vnegq_f32(av));
    }

    for (; i < n; i++) {
        result[i] = -a[i];
    }
}

// ============================================================================
// Float64 Operations (2 lanes per 128-bit vector)
// ============================================================================

// Vector addition: result[i] = a[i] + b[i]
void add_f64_neon(double *a, double *b, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 doubles at a time (4 vectors)
    for (; i + 7 < n; i += 8) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t a2 = vld1q_f64(a + i + 4);
        float64x2_t a3 = vld1q_f64(a + i + 6);

        float64x2_t b0 = vld1q_f64(b + i);
        float64x2_t b1 = vld1q_f64(b + i + 2);
        float64x2_t b2 = vld1q_f64(b + i + 4);
        float64x2_t b3 = vld1q_f64(b + i + 6);

        vst1q_f64(result + i, vaddq_f64(a0, b0));
        vst1q_f64(result + i + 2, vaddq_f64(a1, b1));
        vst1q_f64(result + i + 4, vaddq_f64(a2, b2));
        vst1q_f64(result + i + 6, vaddq_f64(a3, b3));
    }

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        vst1q_f64(result + i, vaddq_f64(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

// Vector multiplication: result[i] = a[i] * b[i]
void mul_f64_neon(double *a, double *b, double *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 7 < n; i += 8) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t a2 = vld1q_f64(a + i + 4);
        float64x2_t a3 = vld1q_f64(a + i + 6);

        float64x2_t b0 = vld1q_f64(b + i);
        float64x2_t b1 = vld1q_f64(b + i + 2);
        float64x2_t b2 = vld1q_f64(b + i + 4);
        float64x2_t b3 = vld1q_f64(b + i + 6);

        vst1q_f64(result + i, vmulq_f64(a0, b0));
        vst1q_f64(result + i + 2, vmulq_f64(a1, b1));
        vst1q_f64(result + i + 4, vmulq_f64(a2, b2));
        vst1q_f64(result + i + 6, vmulq_f64(a3, b3));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        vst1q_f64(result + i, vmulq_f64(av, bv));
    }

    for (; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

// Fused multiply-add: result[i] = a[i] * b[i] + c[i]
void fma_f64_neon(double *a, double *b, double *c, double *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 7 < n; i += 8) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t a2 = vld1q_f64(a + i + 4);
        float64x2_t a3 = vld1q_f64(a + i + 6);

        float64x2_t b0 = vld1q_f64(b + i);
        float64x2_t b1 = vld1q_f64(b + i + 2);
        float64x2_t b2 = vld1q_f64(b + i + 4);
        float64x2_t b3 = vld1q_f64(b + i + 6);

        float64x2_t c0 = vld1q_f64(c + i);
        float64x2_t c1 = vld1q_f64(c + i + 2);
        float64x2_t c2 = vld1q_f64(c + i + 4);
        float64x2_t c3 = vld1q_f64(c + i + 6);

        vst1q_f64(result + i, vfmaq_f64(c0, a0, b0));
        vst1q_f64(result + i + 2, vfmaq_f64(c1, a1, b1));
        vst1q_f64(result + i + 4, vfmaq_f64(c2, a2, b2));
        vst1q_f64(result + i + 6, vfmaq_f64(c3, a3, b3));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        float64x2_t cv = vld1q_f64(c + i);
        vst1q_f64(result + i, vfmaq_f64(cv, av, bv));
    }

    for (; i < n; i++) {
        result[i] = a[i] * b[i] + c[i];
    }
}

// Horizontal sum reduction for f64
void reduce_sum_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;
    double sum = 0.0;

    if (n >= 8) {
        float64x2_t sum0 = vdupq_n_f64(0);
        float64x2_t sum1 = vdupq_n_f64(0);
        float64x2_t sum2 = vdupq_n_f64(0);
        float64x2_t sum3 = vdupq_n_f64(0);

        for (; i + 7 < n; i += 8) {
            sum0 = vaddq_f64(sum0, vld1q_f64(input + i));
            sum1 = vaddq_f64(sum1, vld1q_f64(input + i + 2));
            sum2 = vaddq_f64(sum2, vld1q_f64(input + i + 4));
            sum3 = vaddq_f64(sum3, vld1q_f64(input + i + 6));
        }

        sum0 = vaddq_f64(sum0, sum1);
        sum2 = vaddq_f64(sum2, sum3);
        sum0 = vaddq_f64(sum0, sum2);

        sum = vaddvq_f64(sum0);
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t v = vld1q_f64(input + i);
        sum += vaddvq_f64(v);
    }

    for (; i < n; i++) {
        sum += input[i];
    }

    *result = sum;
}

// ============================================================================
// Type Conversions (Phase 5)
// ============================================================================

// Promote float32 to float64: result[i] = (double)input[i]
void promote_f32_f64_neon(float *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 floats at a time (producing 8 doubles)
    for (; i + 7 < n; i += 8) {
        // Load 4 floats, convert to 2 doubles each
        float32x4_t f0 = vld1q_f32(input + i);
        float32x4_t f1 = vld1q_f32(input + i + 4);

        // vcvt_f64_f32 converts low 2 floats to 2 doubles
        // vcvt_high_f64_f32 converts high 2 floats to 2 doubles
        float64x2_t d0 = vcvt_f64_f32(vget_low_f32(f0));
        float64x2_t d1 = vcvt_high_f64_f32(f0);
        float64x2_t d2 = vcvt_f64_f32(vget_low_f32(f1));
        float64x2_t d3 = vcvt_high_f64_f32(f1);

        vst1q_f64(result + i, d0);
        vst1q_f64(result + i + 2, d1);
        vst1q_f64(result + i + 4, d2);
        vst1q_f64(result + i + 6, d3);
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t f = vld1q_f32(input + i);
        float64x2_t d0 = vcvt_f64_f32(vget_low_f32(f));
        float64x2_t d1 = vcvt_high_f64_f32(f);
        vst1q_f64(result + i, d0);
        vst1q_f64(result + i + 2, d1);
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (double)input[i];
    }
}

// Demote float64 to float32: result[i] = (float)input[i]
void demote_f64_f32_neon(double *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 doubles at a time (producing 8 floats)
    for (; i + 7 < n; i += 8) {
        float64x2_t d0 = vld1q_f64(input + i);
        float64x2_t d1 = vld1q_f64(input + i + 2);
        float64x2_t d2 = vld1q_f64(input + i + 4);
        float64x2_t d3 = vld1q_f64(input + i + 6);

        // vcvt_f32_f64 converts 2 doubles to 2 floats (low half)
        // vcvt_high_f32_f64 converts 2 doubles to high half of float32x4
        float32x4_t f0 = vcvt_high_f32_f64(vcvt_f32_f64(d0), d1);
        float32x4_t f1 = vcvt_high_f32_f64(vcvt_f32_f64(d2), d3);

        vst1q_f32(result + i, f0);
        vst1q_f32(result + i + 4, f1);
    }

    // Process 4 doubles at a time
    for (; i + 3 < n; i += 4) {
        float64x2_t d0 = vld1q_f64(input + i);
        float64x2_t d1 = vld1q_f64(input + i + 2);
        float32x4_t f = vcvt_high_f32_f64(vcvt_f32_f64(d0), d1);
        vst1q_f32(result + i, f);
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (float)input[i];
    }
}

// Convert float32 to int32 (round toward zero): result[i] = (int)input[i]
void convert_f32_i32_neon(float *input, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 floats at a time
    for (; i + 15 < n; i += 16) {
        float32x4_t f0 = vld1q_f32(input + i);
        float32x4_t f1 = vld1q_f32(input + i + 4);
        float32x4_t f2 = vld1q_f32(input + i + 8);
        float32x4_t f3 = vld1q_f32(input + i + 12);

        // vcvtq_s32_f32 converts with truncation toward zero
        vst1q_s32(result + i, vcvtq_s32_f32(f0));
        vst1q_s32(result + i + 4, vcvtq_s32_f32(f1));
        vst1q_s32(result + i + 8, vcvtq_s32_f32(f2));
        vst1q_s32(result + i + 12, vcvtq_s32_f32(f3));
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t f = vld1q_f32(input + i);
        vst1q_s32(result + i, vcvtq_s32_f32(f));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (int)input[i];
    }
}

// Convert int32 to float32: result[i] = (float)input[i]
void convert_i32_f32_neon(int *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t i0 = vld1q_s32(input + i);
        int32x4_t i1 = vld1q_s32(input + i + 4);
        int32x4_t i2 = vld1q_s32(input + i + 8);
        int32x4_t i3 = vld1q_s32(input + i + 12);

        vst1q_f32(result + i, vcvtq_f32_s32(i0));
        vst1q_f32(result + i + 4, vcvtq_f32_s32(i1));
        vst1q_f32(result + i + 8, vcvtq_f32_s32(i2));
        vst1q_f32(result + i + 12, vcvtq_f32_s32(i3));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t iv = vld1q_s32(input + i);
        vst1q_f32(result + i, vcvtq_f32_s32(iv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (float)input[i];
    }
}

// Round to nearest (ties to even): result[i] = round(input[i])
void round_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 floats at a time
    for (; i + 15 < n; i += 16) {
        float32x4_t f0 = vld1q_f32(input + i);
        float32x4_t f1 = vld1q_f32(input + i + 4);
        float32x4_t f2 = vld1q_f32(input + i + 8);
        float32x4_t f3 = vld1q_f32(input + i + 12);

        vst1q_f32(result + i, vrndnq_f32(f0));
        vst1q_f32(result + i + 4, vrndnq_f32(f1));
        vst1q_f32(result + i + 8, vrndnq_f32(f2));
        vst1q_f32(result + i + 12, vrndnq_f32(f3));
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t f = vld1q_f32(input + i);
        vst1q_f32(result + i, vrndnq_f32(f));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = __builtin_roundf(input[i]);
    }
}

// Truncate toward zero: result[i] = trunc(input[i])
void trunc_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t f0 = vld1q_f32(input + i);
        float32x4_t f1 = vld1q_f32(input + i + 4);
        float32x4_t f2 = vld1q_f32(input + i + 8);
        float32x4_t f3 = vld1q_f32(input + i + 12);

        vst1q_f32(result + i, vrndq_f32(f0));
        vst1q_f32(result + i + 4, vrndq_f32(f1));
        vst1q_f32(result + i + 8, vrndq_f32(f2));
        vst1q_f32(result + i + 12, vrndq_f32(f3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t f = vld1q_f32(input + i);
        vst1q_f32(result + i, vrndq_f32(f));
    }

    for (; i < n; i++) {
        result[i] = __builtin_truncf(input[i]);
    }
}

// Ceiling (round up): result[i] = ceil(input[i])
void ceil_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t f0 = vld1q_f32(input + i);
        float32x4_t f1 = vld1q_f32(input + i + 4);
        float32x4_t f2 = vld1q_f32(input + i + 8);
        float32x4_t f3 = vld1q_f32(input + i + 12);

        vst1q_f32(result + i, vrndpq_f32(f0));
        vst1q_f32(result + i + 4, vrndpq_f32(f1));
        vst1q_f32(result + i + 8, vrndpq_f32(f2));
        vst1q_f32(result + i + 12, vrndpq_f32(f3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t f = vld1q_f32(input + i);
        vst1q_f32(result + i, vrndpq_f32(f));
    }

    for (; i < n; i++) {
        result[i] = __builtin_ceilf(input[i]);
    }
}

// Floor (round down): result[i] = floor(input[i])
void floor_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 15 < n; i += 16) {
        float32x4_t f0 = vld1q_f32(input + i);
        float32x4_t f1 = vld1q_f32(input + i + 4);
        float32x4_t f2 = vld1q_f32(input + i + 8);
        float32x4_t f3 = vld1q_f32(input + i + 12);

        vst1q_f32(result + i, vrndmq_f32(f0));
        vst1q_f32(result + i + 4, vrndmq_f32(f1));
        vst1q_f32(result + i + 8, vrndmq_f32(f2));
        vst1q_f32(result + i + 12, vrndmq_f32(f3));
    }

    for (; i + 3 < n; i += 4) {
        float32x4_t f = vld1q_f32(input + i);
        vst1q_f32(result + i, vrndmq_f32(f));
    }

    for (; i < n; i++) {
        result[i] = __builtin_floorf(input[i]);
    }
}

// ============================================================================
// Memory Operations (Phase 4)
// ============================================================================

// Gather float32: result[i] = base[indices[i]]
// NEON doesn't have native gather, so we use scalar loop with NEON stores
void gather_f32_neon(float *base, int *indices, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 4 elements at a time using scalar gather + NEON store
    for (; i + 3 < n; i += 4) {
        float tmp[4];
        tmp[0] = base[indices[i]];
        tmp[1] = base[indices[i + 1]];
        tmp[2] = base[indices[i + 2]];
        tmp[3] = base[indices[i + 3]];
        vst1q_f32(result + i, vld1q_f32(tmp));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = base[indices[i]];
    }
}

// Gather float64: result[i] = base[indices[i]]
void gather_f64_neon(double *base, int *indices, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 2 elements at a time
    for (; i + 1 < n; i += 2) {
        double tmp[2];
        tmp[0] = base[indices[i]];
        tmp[1] = base[indices[i + 1]];
        vst1q_f64(result + i, vld1q_f64(tmp));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = base[indices[i]];
    }
}

// Gather int32: result[i] = base[indices[i]]
void gather_i32_neon(int *base, int *indices, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        int tmp[4];
        tmp[0] = base[indices[i]];
        tmp[1] = base[indices[i + 1]];
        tmp[2] = base[indices[i + 2]];
        tmp[3] = base[indices[i + 3]];
        vst1q_s32(result + i, vld1q_s32(tmp));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = base[indices[i]];
    }
}

// Scatter float32: base[indices[i]] = values[i]
void scatter_f32_neon(float *values, int *indices, float *base, long *len) {
    long n = *len;

    // Scatter is inherently serial due to potential index conflicts
    for (long i = 0; i < n; i++) {
        base[indices[i]] = values[i];
    }
}

// Scatter float64: base[indices[i]] = values[i]
void scatter_f64_neon(double *values, int *indices, double *base, long *len) {
    long n = *len;

    for (long i = 0; i < n; i++) {
        base[indices[i]] = values[i];
    }
}

// Scatter int32: base[indices[i]] = values[i]
void scatter_i32_neon(int *values, int *indices, int *base, long *len) {
    long n = *len;

    for (long i = 0; i < n; i++) {
        base[indices[i]] = values[i];
    }
}

// Masked load float32: result[i] = mask[i] ? input[i] : 0
void masked_load_f32_neon(float *input, int *mask, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        int32x4_t m = vld1q_s32(mask + i);
        // Convert mask to all 1s or 0s: compare != 0
        uint32x4_t cmp = vcgtq_s32(m, vdupq_n_s32(0));
        // Use bit select: where mask is 1, use v; where 0, use zero
        float32x4_t zero = vdupq_n_f32(0);
        float32x4_t selected = vbslq_f32(cmp, v, zero);
        vst1q_f32(result + i, selected);
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = mask[i] ? input[i] : 0.0f;
    }
}

// Masked store float32: if mask[i] then output[i] = input[i]
void masked_store_f32_neon(float *input, int *mask, float *output, long *len) {
    long n = *len;

    // Masked store needs to preserve existing values, so process element by element
    for (long i = 0; i < n; i++) {
        if (mask[i]) {
            output[i] = input[i];
        }
    }
}
