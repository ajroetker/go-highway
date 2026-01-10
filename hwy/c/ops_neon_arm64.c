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
