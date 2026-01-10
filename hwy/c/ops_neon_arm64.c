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
        // Simple Newton-Raphson sqrt for scalar remainder
        float x = a[i];
        if (x <= 0.0f) {
            result[i] = 0.0f;
        }
        if (x > 0.0f) {
            float y = x * 0.5f;  // Initial guess
            // Newton-Raphson iterations: y = (y + x/y) / 2
            y = 0.5f * (y + x / y);
            y = 0.5f * (y + x / y);
            y = 0.5f * (y + x / y);
            y = 0.5f * (y + x / y);
            result[i] = y;
        }
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
        float x = input[i];
        float adj = 0.5f;
        if (x < 0.0f) {
            adj = -0.5f;
        }
        result[i] = (float)(int)(x + adj);
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
        result[i] = (float)(int)input[i];
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
        float x = input[i];
        int ix = (int)x;
        float fi = (float)ix;
        if (x > fi) {
            result[i] = (float)(ix + 1);
        }
        if (x <= fi) {
            result[i] = fi;
        }
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
        float x = input[i];
        int ix = (int)x;
        float fi = (float)ix;
        if (x < fi) {
            result[i] = (float)(ix - 1);
        }
        if (x >= fi) {
            result[i] = fi;
        }
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

// ============================================================================
// Shuffle/Permutation Operations (Phase 6)
// ============================================================================

// Reverse float32: result[n-1-i] = input[i]
void reverse_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 4 elements at a time using NEON reverse
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(input + (n - i - 4));
        // Reverse the 4 elements: vrev64 reverses within 64-bit halves, then ext swaps halves
        float32x4_t rev = vrev64q_f32(v);
        rev = vextq_f32(rev, rev, 2);
        vst1q_f32(result + i, rev);
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = input[n - 1 - i];
    }
}

// Reverse float64: result[n-1-i] = input[i]
void reverse_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 2 elements at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t v = vld1q_f64(input + (n - i - 2));
        // Swap the two elements
        float64x2_t rev = vextq_f64(v, v, 1);
        vst1q_f64(result + i, rev);
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = input[n - 1 - i];
    }
}

// Reverse2 float32: swap adjacent pairs [0,1,2,3] -> [1,0,3,2]
void reverse2_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        // vrev64 reverses pairs within 64-bit halves: [0,1,2,3] -> [1,0,3,2]
        vst1q_f32(result + i, vrev64q_f32(v));
    }

    // Scalar remainder
    for (; i + 1 < n; i += 2) {
        result[i] = input[i + 1];
        result[i + 1] = input[i];
    }
    if (i < n) {
        result[i] = input[i];
    }
}

// Reverse4 float32: reverse within groups of 4 [0,1,2,3,4,5,6,7] -> [3,2,1,0,7,6,5,4]
void reverse4_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        // First reverse pairs, then swap halves
        float32x4_t rev = vrev64q_f32(v);
        rev = vextq_f32(rev, rev, 2);
        vst1q_f32(result + i, rev);
    }

    // Scalar remainder - reverse partial group
    if (i < n) {
        long remaining = n - i;
        for (long j = 0; j < remaining; j++) {
            result[i + j] = input[i + remaining - 1 - j];
        }
    }
}

// Broadcast float32: fill result with input[lane]
void broadcast_f32_neon(float *input, float *result, long *lane, long *len) {
    long n = *len;
    long idx = *lane;
    float value = input[idx];

    long i = 0;
    float32x4_t bcast = vdupq_n_f32(value);

    // Process 16 floats at a time
    for (; i + 15 < n; i += 16) {
        vst1q_f32(result + i, bcast);
        vst1q_f32(result + i + 4, bcast);
        vst1q_f32(result + i + 8, bcast);
        vst1q_f32(result + i + 12, bcast);
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        vst1q_f32(result + i, bcast);
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = value;
    }
}

// GetLane float32: extract a single lane value
void getlane_f32_neon(float *input, float *result, long *lane) {
    *result = input[*lane];
}

// InsertLane float32: insert value at specified lane
void insertlane_f32_neon(float *input, float *result, float *value, long *lane, long *len) {
    long n = *len;
    long idx = *lane;

    // Copy input to result
    long i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(result + i, vld1q_f32(input + i));
    }
    for (; i < n; i++) {
        result[i] = input[i];
    }

    // Insert value at lane
    result[idx] = *value;
}

// InterleaveLower float32: [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a0,b0,a1,b1]
void interleave_lo_f32_neon(float *a, float *b, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 4 elements from each input, producing 4 interleaved elements
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        // zip1 interleaves lower halves: [a0,a1], [b0,b1] -> [a0,b0,a1,b1]
        float32x4_t zipped = vzip1q_f32(av, bv);
        vst1q_f32(result + i, zipped);
    }

    // Scalar remainder
    long half = (n - i) / 2;
    for (long j = 0; j < half; j++) {
        result[i + 2*j] = a[i + j];
        result[i + 2*j + 1] = b[i + j];
    }
}

// InterleaveUpper float32: [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a2,b2,a3,b3]
void interleave_hi_f32_neon(float *a, float *b, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 4 elements from each input
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        // zip2 interleaves upper halves: [a2,a3], [b2,b3] -> [a2,b2,a3,b3]
        float32x4_t zipped = vzip2q_f32(av, bv);
        vst1q_f32(result + i, zipped);
    }

    // Scalar remainder
    long half = (n - i) / 2;
    long start = (n - i) / 2;
    for (long j = 0; j < half; j++) {
        result[i + 2*j] = a[i + start + j];
        result[i + 2*j + 1] = b[i + start + j];
    }
}

// TableLookupBytes uint8: result[i] = tbl[idx[i] & 0x0F]
// Uses NEON TBL instruction for byte-level lookup
void tbl_u8_neon(unsigned char *tbl, unsigned char *idx, unsigned char *result, long *len) {
    long n = *len;
    long i = 0;

    // Load the 16-byte table
    uint8x16_t tbl_vec = vld1q_u8(tbl);

    // Process 16 bytes at a time
    for (; i + 15 < n; i += 16) {
        uint8x16_t idx_vec = vld1q_u8(idx + i);
        uint8x16_t res = vqtbl1q_u8(tbl_vec, idx_vec);
        vst1q_u8(result + i, res);
    }

    // Scalar remainder
    for (; i < n; i++) {
        unsigned char index = idx[i];
        result[i] = (index < 16) ? tbl[index] : 0;
    }
}

// ============================================================================
// Comparison Operations (Phase 7)
// ============================================================================

// Equal float32: result[i] = (a[i] == b[i]) ? 0xFFFFFFFF : 0
void eq_f32_neon(float *a, float *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 floats at a time
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_s32(result + i, vreinterpretq_s32_u32(vceqq_f32(a0, b0)));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vceqq_f32(a1, b1)));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vceqq_f32(a2, b2)));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vceqq_f32(a3, b3)));
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vceqq_f32(av, bv)));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] == b[i]) ? -1 : 0;
    }
}

// Equal int32: result[i] = (a[i] == b[i]) ? 0xFFFFFFFF : 0
void eq_i32_neon(int *a, int *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        int32x4_t b0 = vld1q_s32(b + i);
        int32x4_t b1 = vld1q_s32(b + i + 4);
        int32x4_t b2 = vld1q_s32(b + i + 8);
        int32x4_t b3 = vld1q_s32(b + i + 12);

        vst1q_s32(result + i, vreinterpretq_s32_u32(vceqq_s32(a0, b0)));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vceqq_s32(a1, b1)));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vceqq_s32(a2, b2)));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vceqq_s32(a3, b3)));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vceqq_s32(av, bv)));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] == b[i]) ? -1 : 0;
    }
}

// NotEqual float32: result[i] = (a[i] != b[i]) ? 0xFFFFFFFF : 0
void ne_f32_neon(float *a, float *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 floats at a time
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        // NotEqual = NOT(Equal)
        vst1q_s32(result + i, vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(a0, b0))));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(a1, b1))));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(a2, b2))));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(a3, b3))));
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(av, bv))));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] != b[i]) ? -1 : 0;
    }
}

// NotEqual int32: result[i] = (a[i] != b[i]) ? 0xFFFFFFFF : 0
void ne_i32_neon(int *a, int *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        int32x4_t b0 = vld1q_s32(b + i);
        int32x4_t b1 = vld1q_s32(b + i + 4);
        int32x4_t b2 = vld1q_s32(b + i + 8);
        int32x4_t b3 = vld1q_s32(b + i + 12);

        vst1q_s32(result + i, vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a0, b0))));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a1, b1))));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a2, b2))));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a3, b3))));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(av, bv))));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] != b[i]) ? -1 : 0;
    }
}

// LessThan float32: result[i] = (a[i] < b[i]) ? 0xFFFFFFFF : 0
void lt_f32_neon(float *a, float *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 floats at a time
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_s32(result + i, vreinterpretq_s32_u32(vcltq_f32(a0, b0)));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vcltq_f32(a1, b1)));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vcltq_f32(a2, b2)));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vcltq_f32(a3, b3)));
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vcltq_f32(av, bv)));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] < b[i]) ? -1 : 0;
    }
}

// LessThan int32: result[i] = (a[i] < b[i]) ? 0xFFFFFFFF : 0
void lt_i32_neon(int *a, int *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        int32x4_t b0 = vld1q_s32(b + i);
        int32x4_t b1 = vld1q_s32(b + i + 4);
        int32x4_t b2 = vld1q_s32(b + i + 8);
        int32x4_t b3 = vld1q_s32(b + i + 12);

        vst1q_s32(result + i, vreinterpretq_s32_u32(vcltq_s32(a0, b0)));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vcltq_s32(a1, b1)));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vcltq_s32(a2, b2)));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vcltq_s32(a3, b3)));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vcltq_s32(av, bv)));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] < b[i]) ? -1 : 0;
    }
}

// LessEqual float32: result[i] = (a[i] <= b[i]) ? 0xFFFFFFFF : 0
void le_f32_neon(float *a, float *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 floats at a time
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_s32(result + i, vreinterpretq_s32_u32(vcleq_f32(a0, b0)));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vcleq_f32(a1, b1)));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vcleq_f32(a2, b2)));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vcleq_f32(a3, b3)));
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vcleq_f32(av, bv)));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] <= b[i]) ? -1 : 0;
    }
}

// LessEqual int32: result[i] = (a[i] <= b[i]) ? 0xFFFFFFFF : 0
void le_i32_neon(int *a, int *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        int32x4_t b0 = vld1q_s32(b + i);
        int32x4_t b1 = vld1q_s32(b + i + 4);
        int32x4_t b2 = vld1q_s32(b + i + 8);
        int32x4_t b3 = vld1q_s32(b + i + 12);

        vst1q_s32(result + i, vreinterpretq_s32_u32(vcleq_s32(a0, b0)));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vcleq_s32(a1, b1)));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vcleq_s32(a2, b2)));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vcleq_s32(a3, b3)));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vcleq_s32(av, bv)));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] <= b[i]) ? -1 : 0;
    }
}

// GreaterThan float32: result[i] = (a[i] > b[i]) ? 0xFFFFFFFF : 0
void gt_f32_neon(float *a, float *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 floats at a time
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_s32(result + i, vreinterpretq_s32_u32(vcgtq_f32(a0, b0)));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vcgtq_f32(a1, b1)));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vcgtq_f32(a2, b2)));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vcgtq_f32(a3, b3)));
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vcgtq_f32(av, bv)));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] > b[i]) ? -1 : 0;
    }
}

// GreaterThan int32: result[i] = (a[i] > b[i]) ? 0xFFFFFFFF : 0
void gt_i32_neon(int *a, int *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        int32x4_t b0 = vld1q_s32(b + i);
        int32x4_t b1 = vld1q_s32(b + i + 4);
        int32x4_t b2 = vld1q_s32(b + i + 8);
        int32x4_t b3 = vld1q_s32(b + i + 12);

        vst1q_s32(result + i, vreinterpretq_s32_u32(vcgtq_s32(a0, b0)));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vcgtq_s32(a1, b1)));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vcgtq_s32(a2, b2)));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vcgtq_s32(a3, b3)));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vcgtq_s32(av, bv)));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] > b[i]) ? -1 : 0;
    }
}

// GreaterEqual float32: result[i] = (a[i] >= b[i]) ? 0xFFFFFFFF : 0
void ge_f32_neon(float *a, float *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 floats at a time
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_s32(result + i, vreinterpretq_s32_u32(vcgeq_f32(a0, b0)));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vcgeq_f32(a1, b1)));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vcgeq_f32(a2, b2)));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vcgeq_f32(a3, b3)));
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vcgeq_f32(av, bv)));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] >= b[i]) ? -1 : 0;
    }
}

// GreaterEqual int32: result[i] = (a[i] >= b[i]) ? 0xFFFFFFFF : 0
void ge_i32_neon(int *a, int *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        int32x4_t b0 = vld1q_s32(b + i);
        int32x4_t b1 = vld1q_s32(b + i + 4);
        int32x4_t b2 = vld1q_s32(b + i + 8);
        int32x4_t b3 = vld1q_s32(b + i + 12);

        vst1q_s32(result + i, vreinterpretq_s32_u32(vcgeq_s32(a0, b0)));
        vst1q_s32(result + i + 4, vreinterpretq_s32_u32(vcgeq_s32(a1, b1)));
        vst1q_s32(result + i + 8, vreinterpretq_s32_u32(vcgeq_s32(a2, b2)));
        vst1q_s32(result + i + 12, vreinterpretq_s32_u32(vcgeq_s32(a3, b3)));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vreinterpretq_s32_u32(vcgeq_s32(av, bv)));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] >= b[i]) ? -1 : 0;
    }
}

// ============================================================================
// Phase 8: Bitwise Operations
// ============================================================================

// Bitwise AND int32: result[i] = a[i] & b[i]
void and_i32_neon(int *a, int *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        int32x4_t b0 = vld1q_s32(b + i);
        int32x4_t b1 = vld1q_s32(b + i + 4);
        int32x4_t b2 = vld1q_s32(b + i + 8);
        int32x4_t b3 = vld1q_s32(b + i + 12);

        vst1q_s32(result + i, vandq_s32(a0, b0));
        vst1q_s32(result + i + 4, vandq_s32(a1, b1));
        vst1q_s32(result + i + 8, vandq_s32(a2, b2));
        vst1q_s32(result + i + 12, vandq_s32(a3, b3));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vandq_s32(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] & b[i];
    }
}

// Bitwise OR int32: result[i] = a[i] | b[i]
void or_i32_neon(int *a, int *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        int32x4_t b0 = vld1q_s32(b + i);
        int32x4_t b1 = vld1q_s32(b + i + 4);
        int32x4_t b2 = vld1q_s32(b + i + 8);
        int32x4_t b3 = vld1q_s32(b + i + 12);

        vst1q_s32(result + i, vorrq_s32(a0, b0));
        vst1q_s32(result + i + 4, vorrq_s32(a1, b1));
        vst1q_s32(result + i + 8, vorrq_s32(a2, b2));
        vst1q_s32(result + i + 12, vorrq_s32(a3, b3));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vorrq_s32(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] | b[i];
    }
}

// Bitwise XOR int32: result[i] = a[i] ^ b[i]
void xor_i32_neon(int *a, int *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        int32x4_t b0 = vld1q_s32(b + i);
        int32x4_t b1 = vld1q_s32(b + i + 4);
        int32x4_t b2 = vld1q_s32(b + i + 8);
        int32x4_t b3 = vld1q_s32(b + i + 12);

        vst1q_s32(result + i, veorq_s32(a0, b0));
        vst1q_s32(result + i + 4, veorq_s32(a1, b1));
        vst1q_s32(result + i + 8, veorq_s32(a2, b2));
        vst1q_s32(result + i + 12, veorq_s32(a3, b3));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, veorq_s32(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] ^ b[i];
    }
}

// Bitwise AND-NOT int32: result[i] = a[i] & ~b[i]
void andnot_i32_neon(int *a, int *b, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        int32x4_t b0 = vld1q_s32(b + i);
        int32x4_t b1 = vld1q_s32(b + i + 4);
        int32x4_t b2 = vld1q_s32(b + i + 8);
        int32x4_t b3 = vld1q_s32(b + i + 12);

        // vbicq_s32(a, b) = a & ~b
        vst1q_s32(result + i, vbicq_s32(a0, b0));
        vst1q_s32(result + i + 4, vbicq_s32(a1, b1));
        vst1q_s32(result + i + 8, vbicq_s32(a2, b2));
        vst1q_s32(result + i + 12, vbicq_s32(a3, b3));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vbicq_s32(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] & ~b[i];
    }
}

// Bitwise NOT int32: result[i] = ~a[i]
void not_i32_neon(int *a, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        vst1q_s32(result + i, vmvnq_s32(a0));
        vst1q_s32(result + i + 4, vmvnq_s32(a1));
        vst1q_s32(result + i + 8, vmvnq_s32(a2));
        vst1q_s32(result + i + 12, vmvnq_s32(a3));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        vst1q_s32(result + i, vmvnq_s32(av));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = ~a[i];
    }
}

// Shift left int32: result[i] = a[i] << shift (uniform shift)
void shl_i32_neon(int *a, int *result, long *shift, long *len) {
    long n = *len;
    long s = *shift;
    long i = 0;

    int32x4_t shift_vec = vdupq_n_s32((int)s);

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        vst1q_s32(result + i, vshlq_s32(a0, shift_vec));
        vst1q_s32(result + i + 4, vshlq_s32(a1, shift_vec));
        vst1q_s32(result + i + 8, vshlq_s32(a2, shift_vec));
        vst1q_s32(result + i + 12, vshlq_s32(a3, shift_vec));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        vst1q_s32(result + i, vshlq_s32(av, shift_vec));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] << s;
    }
}

// Shift right int32 (arithmetic): result[i] = a[i] >> shift (uniform shift)
void shr_i32_neon(int *a, int *result, long *shift, long *len) {
    long n = *len;
    long s = *shift;
    long i = 0;

    // Negative shift means right shift in NEON vshlq
    int32x4_t shift_vec = vdupq_n_s32(-(int)s);

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t a0 = vld1q_s32(a + i);
        int32x4_t a1 = vld1q_s32(a + i + 4);
        int32x4_t a2 = vld1q_s32(a + i + 8);
        int32x4_t a3 = vld1q_s32(a + i + 12);

        vst1q_s32(result + i, vshlq_s32(a0, shift_vec));
        vst1q_s32(result + i + 4, vshlq_s32(a1, shift_vec));
        vst1q_s32(result + i + 8, vshlq_s32(a2, shift_vec));
        vst1q_s32(result + i + 12, vshlq_s32(a3, shift_vec));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        vst1q_s32(result + i, vshlq_s32(av, shift_vec));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] >> s;
    }
}

// ============================================================================
// Phase 9: Mask Operations
// ============================================================================

// IfThenElse float32: result[i] = mask[i] ? yes[i] : no[i]
void ifthenelse_f32_neon(int *mask, float *yes, float *no, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 floats at a time
    for (; i + 15 < n; i += 16) {
        uint32x4_t m0 = vld1q_u32((unsigned int *)(mask + i));
        uint32x4_t m1 = vld1q_u32((unsigned int *)(mask + i + 4));
        uint32x4_t m2 = vld1q_u32((unsigned int *)(mask + i + 8));
        uint32x4_t m3 = vld1q_u32((unsigned int *)(mask + i + 12));

        float32x4_t y0 = vld1q_f32(yes + i);
        float32x4_t y1 = vld1q_f32(yes + i + 4);
        float32x4_t y2 = vld1q_f32(yes + i + 8);
        float32x4_t y3 = vld1q_f32(yes + i + 12);

        float32x4_t n0 = vld1q_f32(no + i);
        float32x4_t n1 = vld1q_f32(no + i + 4);
        float32x4_t n2 = vld1q_f32(no + i + 8);
        float32x4_t n3 = vld1q_f32(no + i + 12);

        // vbslq_f32(mask, if_true, if_false) - selects bits
        vst1q_f32(result + i, vbslq_f32(m0, y0, n0));
        vst1q_f32(result + i + 4, vbslq_f32(m1, y1, n1));
        vst1q_f32(result + i + 8, vbslq_f32(m2, y2, n2));
        vst1q_f32(result + i + 12, vbslq_f32(m3, y3, n3));
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        uint32x4_t mv = vld1q_u32((unsigned int *)(mask + i));
        float32x4_t yv = vld1q_f32(yes + i);
        float32x4_t nv = vld1q_f32(no + i);
        vst1q_f32(result + i, vbslq_f32(mv, yv, nv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = mask[i] ? yes[i] : no[i];
    }
}

// IfThenElse int32: result[i] = mask[i] ? yes[i] : no[i]
void ifthenelse_i32_neon(int *mask, int *yes, int *no, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 ints at a time
    for (; i + 15 < n; i += 16) {
        uint32x4_t m0 = vld1q_u32((unsigned int *)(mask + i));
        uint32x4_t m1 = vld1q_u32((unsigned int *)(mask + i + 4));
        uint32x4_t m2 = vld1q_u32((unsigned int *)(mask + i + 8));
        uint32x4_t m3 = vld1q_u32((unsigned int *)(mask + i + 12));

        int32x4_t y0 = vld1q_s32(yes + i);
        int32x4_t y1 = vld1q_s32(yes + i + 4);
        int32x4_t y2 = vld1q_s32(yes + i + 8);
        int32x4_t y3 = vld1q_s32(yes + i + 12);

        int32x4_t n0 = vld1q_s32(no + i);
        int32x4_t n1 = vld1q_s32(no + i + 4);
        int32x4_t n2 = vld1q_s32(no + i + 8);
        int32x4_t n3 = vld1q_s32(no + i + 12);

        vst1q_s32(result + i, vbslq_s32(m0, y0, n0));
        vst1q_s32(result + i + 4, vbslq_s32(m1, y1, n1));
        vst1q_s32(result + i + 8, vbslq_s32(m2, y2, n2));
        vst1q_s32(result + i + 12, vbslq_s32(m3, y3, n3));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        uint32x4_t mv = vld1q_u32((unsigned int *)(mask + i));
        int32x4_t yv = vld1q_s32(yes + i);
        int32x4_t nv = vld1q_s32(no + i);
        vst1q_s32(result + i, vbslq_s32(mv, yv, nv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = mask[i] ? yes[i] : no[i];
    }
}

// CountTrue int32: counts non-zero elements in mask
void count_true_i32_neon(int *mask, long *result, long *len) {
    long n = *len;
    long i = 0;
    long count = 0;

    // Process 16 elements at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t m0 = vld1q_s32(mask + i);
        int32x4_t m1 = vld1q_s32(mask + i + 4);
        int32x4_t m2 = vld1q_s32(mask + i + 8);
        int32x4_t m3 = vld1q_s32(mask + i + 12);

        // Compare != 0 to get all 1s or 0s
        uint32x4_t c0 = vcgtq_s32(m0, vdupq_n_s32(0));
        uint32x4_t c1 = vcgtq_s32(m1, vdupq_n_s32(0));
        uint32x4_t c2 = vcgtq_s32(m2, vdupq_n_s32(0));
        uint32x4_t c3 = vcgtq_s32(m3, vdupq_n_s32(0));

        // Also check for negative values (mask could be -1)
        c0 = vorrq_u32(c0, vcltq_s32(m0, vdupq_n_s32(0)));
        c1 = vorrq_u32(c1, vcltq_s32(m1, vdupq_n_s32(0)));
        c2 = vorrq_u32(c2, vcltq_s32(m2, vdupq_n_s32(0)));
        c3 = vorrq_u32(c3, vcltq_s32(m3, vdupq_n_s32(0)));

        // Convert to 1s (right shift by 31 to get 0 or 1)
        int32x4_t ones0 = vshrq_n_u32(c0, 31);
        int32x4_t ones1 = vshrq_n_u32(c1, 31);
        int32x4_t ones2 = vshrq_n_u32(c2, 31);
        int32x4_t ones3 = vshrq_n_u32(c3, 31);

        // Sum horizontally
        count += vaddvq_s32(ones0) + vaddvq_s32(ones1) + vaddvq_s32(ones2) + vaddvq_s32(ones3);
    }

    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t mv = vld1q_s32(mask + i);
        uint32x4_t cv = vorrq_u32(vcgtq_s32(mv, vdupq_n_s32(0)), vcltq_s32(mv, vdupq_n_s32(0)));
        int32x4_t ones = vshrq_n_u32(cv, 31);
        count += vaddvq_s32(ones);
    }

    // Scalar remainder
    for (; i < n; i++) {
        if (mask[i] != 0) count++;
    }

    *result = count;
}

// AllTrue int32: returns 1 if all mask elements are non-zero, 0 otherwise
void all_true_i32_neon(int *mask, long *result, long *len) {
    long n = *len;
    long i = 0;

    if (n == 0) {
        *result = 1;
        return;
    }

    // Process 16 elements at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t m0 = vld1q_s32(mask + i);
        int32x4_t m1 = vld1q_s32(mask + i + 4);
        int32x4_t m2 = vld1q_s32(mask + i + 8);
        int32x4_t m3 = vld1q_s32(mask + i + 12);

        // Check if any element is zero
        uint32x4_t z0 = vceqq_s32(m0, vdupq_n_s32(0));
        uint32x4_t z1 = vceqq_s32(m1, vdupq_n_s32(0));
        uint32x4_t z2 = vceqq_s32(m2, vdupq_n_s32(0));
        uint32x4_t z3 = vceqq_s32(m3, vdupq_n_s32(0));

        // Combine
        uint32x4_t any_zero = vorrq_u32(vorrq_u32(z0, z1), vorrq_u32(z2, z3));

        // If any element is non-zero in any_zero, we have a false element
        if (vmaxvq_u32(any_zero) != 0) {
            *result = 0;
            return;
        }
    }

    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t mv = vld1q_s32(mask + i);
        uint32x4_t zv = vceqq_s32(mv, vdupq_n_s32(0));
        if (vmaxvq_u32(zv) != 0) {
            *result = 0;
            return;
        }
    }

    // Scalar remainder
    for (; i < n; i++) {
        if (mask[i] == 0) {
            *result = 0;
            return;
        }
    }

    *result = 1;
}

// AllFalse int32: returns 1 if all mask elements are zero, 0 otherwise
void all_false_i32_neon(int *mask, long *result, long *len) {
    long n = *len;
    long i = 0;

    if (n == 0) {
        *result = 1;
        return;
    }

    // Process 16 elements at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t m0 = vld1q_s32(mask + i);
        int32x4_t m1 = vld1q_s32(mask + i + 4);
        int32x4_t m2 = vld1q_s32(mask + i + 8);
        int32x4_t m3 = vld1q_s32(mask + i + 12);

        // Combine all masks with OR
        int32x4_t combined = vorrq_s32(vorrq_s32(m0, m1), vorrq_s32(m2, m3));

        // If any bit is set, we have a non-zero element
        uint32x4_t any_set = vreinterpretq_u32_s32(combined);
        if (vmaxvq_u32(any_set) != 0) {
            *result = 0;
            return;
        }
    }

    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t mv = vld1q_s32(mask + i);
        uint32x4_t any_set = vreinterpretq_u32_s32(mv);
        if (vmaxvq_u32(any_set) != 0) {
            *result = 0;
            return;
        }
    }

    // Scalar remainder
    for (; i < n; i++) {
        if (mask[i] != 0) {
            *result = 0;
            return;
        }
    }

    *result = 1;
}

// FirstN int32: generates mask with first n elements set to -1, rest 0
void firstn_i32_neon(int *result, long *count, long *len) {
    long n = *len;
    long c = *count;

    // Clamp to valid range
    if (c < 0) {
        c = 0;
    }
    if (c > n) {
        c = n;
    }

    int32x4_t neg_one = vdupq_n_s32(-1);
    int32x4_t zero = vdupq_n_s32(0);

    long i = 0;

    // Write -1s in groups of 4
    for (; i + 4 <= c; i += 4) {
        vst1q_s32(result + i, neg_one);
    }

    // Write remaining -1s one at a time with NEON lane store
    long rem = c - i;
    if (rem > 0) {
        vst1q_lane_s32(result + i, neg_one, 0);
        i = i + 1;
        rem = rem - 1;
        if (rem > 0) {
            vst1q_lane_s32(result + i, neg_one, 0);
            i = i + 1;
            rem = rem - 1;
            if (rem > 0) {
                vst1q_lane_s32(result + i, neg_one, 0);
                i = i + 1;
            }
        }
    }

    // Write 0s in groups of 4
    for (; i + 4 <= n; i += 4) {
        vst1q_s32(result + i, zero);
    }

    // Write remaining 0s one at a time
    rem = n - i;
    if (rem > 0) {
        vst1q_lane_s32(result + i, zero, 0);
        i = i + 1;
        rem = rem - 1;
        if (rem > 0) {
            vst1q_lane_s32(result + i, zero, 0);
            i = i + 1;
            rem = rem - 1;
            if (rem > 0) {
                vst1q_lane_s32(result + i, zero, 0);
            }
        }
    }
}

// Compress float32: packs elements where mask is non-zero
// Returns number of elements written
void compress_f32_neon(float *input, int *mask, float *result, long *count, long *len) {
    long n = *len;
    long j = 0;

    // NEON doesn't have native compress, use scalar
    for (long i = 0; i < n; i++) {
        if (mask[i]) {
            result[j++] = input[i];
        }
    }

    *count = j;
}

// Expand float32: unpacks elements to positions where mask is non-zero
void expand_f32_neon(float *input, int *mask, float *result, long *count, long *len) {
    long n = *len;
    long j = 0;

    // NEON doesn't have native expand, use scalar
    // Initialize result based on mask and place input values at masked positions
    // Combined loop to avoid memset optimization
    for (long i = 0; i < n; i++) {
        if (mask[i]) {
            result[i] = input[j++];
        }
        if (mask[i] == 0) {
            result[i] = 0.0f;
        }
    }

    *count = j;
}

// ============================================================================
// Phase 10: Transcendental Math Operations
// ============================================================================

// Exp float32: result[i] = exp(input[i])
// Uses range reduction: exp(x) = 2^k * exp(r) where x = k*ln(2) + r
void exp_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants for exp approximation
    float32x4_t v_ln2 = vdupq_n_f32(0.6931471805599453f);
    float32x4_t v_inv_ln2 = vdupq_n_f32(1.4426950408889634f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Clamp input to prevent overflow/underflow
        x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
        x = vminq_f32(x, vdupq_n_f32(88.0f));

        // k = round(x / ln(2))
        float32x4_t k = vrndnq_f32(vmulq_f32(x, v_inv_ln2));

        // r = x - k * ln(2)
        float32x4_t r = vfmsq_f32(x, k, v_ln2);

        // exp(r) using polynomial (Horner's method)
        // exp(r) ≈ 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720
        float32x4_t exp_r = vdupq_n_f32(0.001388888888888889f);  // c6
        exp_r = vfmaq_f32(vdupq_n_f32(0.008333333333333333f), exp_r, r);  // c5
        exp_r = vfmaq_f32(vdupq_n_f32(0.041666666666666664f), exp_r, r);  // c4
        exp_r = vfmaq_f32(vdupq_n_f32(0.16666666666666666f), exp_r, r);   // c3
        exp_r = vfmaq_f32(vdupq_n_f32(0.5f), exp_r, r);                    // c2
        exp_r = vfmaq_f32(vdupq_n_f32(1.0f), exp_r, r);                    // c1
        exp_r = vfmaq_f32(vdupq_n_f32(1.0f), exp_r, r);                    // c0

        // Scale by 2^k
        // Convert k to int, add to exponent bias (127), shift to exponent position
        int32x4_t ki = vcvtq_s32_f32(k);
        int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23);
        float32x4_t scale = vreinterpretq_f32_s32(exp_bits);

        vst1q_f32(result + i, vmulq_f32(exp_r, scale));
    }

    // Scalar remainder - inline polynomial exp approximation
    for (; i < n; i++) {
        float x = input[i];
        if (x < -88.0f) x = -88.0f;
        if (x > 88.0f) x = 88.0f;

        // Range reduction: k = round(x / ln(2))
        const float ln2 = 0.6931471805599453f;
        const float inv_ln2 = 1.4426950408889634f;
        float kf = x * inv_ln2;
        float adj = 0.5f;
        if (kf < 0.0f) {
            adj = -0.5f;
        }
        int ki = (int)(kf + adj);
        float k = (float)ki;
        float r = x - k * ln2;

        // Polynomial approximation for exp(r) using Horner's method
        float exp_r = 0.001388888888888889f;
        exp_r = exp_r * r + 0.008333333333333333f;
        exp_r = exp_r * r + 0.041666666666666664f;
        exp_r = exp_r * r + 0.16666666666666666f;
        exp_r = exp_r * r + 0.5f;
        exp_r = exp_r * r + 1.0f;
        exp_r = exp_r * r + 1.0f;

        // Compute 2^k by repeated multiplication
        float scale = 1.0f;
        int absK = ki;
        if (ki < 0) absK = -ki;
        long j = 0;
        if (ki >= 0) {
            for (j = 0; j < absK; j++) {
                scale = scale * 2.0f;
            }
        }
        if (ki < 0) {
            for (j = 0; j < absK; j++) {
                scale = scale * 0.5f;
            }
        }

        result[i] = exp_r * scale;
    }
}

// Log float32: result[i] = log(input[i]) (natural logarithm)
// Uses range reduction: log(x) = k*ln(2) + log(m) where x = m * 2^k, 1 <= m < 2
void log_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    const float ln2 = 0.6931471805599453f;

    // Polynomial coefficients for log(1+x) on [0, 1]
    // log(1+x) ≈ x - x^2/2 + x^3/3 - x^4/4 + ...
    const float c1 = 1.0f;
    const float c2 = -0.5f;
    const float c3 = 0.33333333333f;
    const float c4 = -0.25f;
    const float c5 = 0.2f;
    const float c6 = -0.16666666667f;

    float32x4_t v_ln2 = vdupq_n_f32(ln2);
    float32x4_t v_one = vdupq_n_f32(1.0f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Extract exponent and mantissa
        int32x4_t xi = vreinterpretq_s32_f32(x);
        int32x4_t exp_bits = vshrq_n_s32(xi, 23);
        int32x4_t k = vsubq_s32(vandq_s32(exp_bits, vdupq_n_s32(0xFF)), vdupq_n_s32(127));

        // Set exponent to 0 (bias 127) to get mantissa in [1, 2)
        int32x4_t mantissa_bits = vorrq_s32(vandq_s32(xi, vdupq_n_s32(0x007FFFFF)), vdupq_n_s32(0x3F800000));
        float32x4_t m = vreinterpretq_f32_s32(mantissa_bits);

        // f = m - 1, so we compute log(1 + f)
        float32x4_t f = vsubq_f32(m, v_one);

        // Polynomial approximation for log(1+f)
        float32x4_t f2 = vmulq_f32(f, f);
        float32x4_t f3 = vmulq_f32(f2, f);
        float32x4_t f4 = vmulq_f32(f2, f2);
        float32x4_t f5 = vmulq_f32(f4, f);
        float32x4_t f6 = vmulq_f32(f3, f3);

        float32x4_t log_m = vmulq_f32(f, vdupq_n_f32(c1));
        log_m = vfmaq_f32(log_m, f2, vdupq_n_f32(c2));
        log_m = vfmaq_f32(log_m, f3, vdupq_n_f32(c3));
        log_m = vfmaq_f32(log_m, f4, vdupq_n_f32(c4));
        log_m = vfmaq_f32(log_m, f5, vdupq_n_f32(c5));
        log_m = vfmaq_f32(log_m, f6, vdupq_n_f32(c6));

        // log(x) = k * ln(2) + log(m)
        float32x4_t kf = vcvtq_f32_s32(k);
        float32x4_t res = vfmaq_f32(log_m, kf, v_ln2);

        vst1q_f32(result + i, res);
    }

    // Scalar remainder - inline polynomial log approximation
    for (; i < n; i++) {
        float x = input[i];
        if (x <= 0.0f) {
            result[i] = -1e30f;
        }
        if (x > 0.0f) {
            const float ln2 = 0.6931471805599453f;

            // Range reduce to [1, 2)
            int k = 0;
            float m = x;
            for (; m >= 2.0f; ) {
                m = m * 0.5f;
                k = k + 1;
            }
            for (; m < 1.0f; ) {
                m = m * 2.0f;
                k = k - 1;
            }

            // f = m - 1, compute log(1 + f) using polynomial
            float f = m - 1.0f;
            float f2 = f * f;
            float f3 = f2 * f;
            float f4 = f2 * f2;
            float f5 = f4 * f;
            float f6 = f3 * f3;

            float log_m = f * 1.0f + f2 * (-0.5f) + f3 * 0.33333333333f
                        + f4 * (-0.25f) + f5 * 0.2f + f6 * (-0.16666666667f);

            result[i] = (float)k * ln2 + log_m;
        }
    }
}

// Sin float32: result[i] = sin(input[i])
// Uses range reduction to [-pi, pi], reflection to [-pi/2, pi/2], and polynomial
void sin_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants
    const float pi = 3.14159265358979323846f;
    const float inv_pi = 0.3183098861837907f;
    const float half_pi = 1.5707963267948966f;

    // Polynomial coefficients for sin(x) on [-pi/2, pi/2]
    const float s1 = 1.0f;
    const float s3 = -0.16666666666666666f;
    const float s5 = 0.008333333333333333f;
    const float s7 = -0.0001984126984126984f;

    float32x4_t v_pi = vdupq_n_f32(pi);
    float32x4_t v_neg_pi = vdupq_n_f32(-pi);
    float32x4_t v_half_pi = vdupq_n_f32(half_pi);
    float32x4_t v_neg_half_pi = vdupq_n_f32(-half_pi);
    float32x4_t v_inv_pi = vdupq_n_f32(inv_pi);
    float32x4_t v_two = vdupq_n_f32(2.0f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Range reduction: x = x - 2*pi*round(x/(2*pi)) -> x in [-pi, pi]
        float32x4_t k = vrndnq_f32(vmulq_f32(x, vmulq_f32(vdupq_n_f32(0.5f), v_inv_pi)));
        x = vfmsq_f32(x, k, vmulq_f32(v_two, v_pi));

        // Reflection to [-pi/2, pi/2]:
        // if x > pi/2:  sin(x) = sin(pi - x)
        // if x < -pi/2: sin(x) = sin(-pi - x)
        uint32x4_t need_pos_reflect = vcgtq_f32(x, v_half_pi);
        uint32x4_t need_neg_reflect = vcltq_f32(x, v_neg_half_pi);
        float32x4_t x_pos_reflected = vsubq_f32(v_pi, x);
        float32x4_t x_neg_reflected = vsubq_f32(v_neg_pi, x);
        x = vbslq_f32(need_pos_reflect, x_pos_reflected, x);
        x = vbslq_f32(need_neg_reflect, x_neg_reflected, x);

        // sin(x) using polynomial: x * (1 + x^2*(s3 + x^2*(s5 + x^2*s7)))
        float32x4_t x2 = vmulq_f32(x, x);

        float32x4_t p = vdupq_n_f32(s7);
        p = vfmaq_f32(vdupq_n_f32(s5), p, x2);
        p = vfmaq_f32(vdupq_n_f32(s3), p, x2);
        p = vfmaq_f32(vdupq_n_f32(s1), p, x2);
        p = vmulq_f32(p, x);

        vst1q_f32(result + i, p);
    }

    // Scalar remainder
    for (; i < n; i++) {
        float x = input[i];

        // Range reduction
        float kf = x * 0.5f * inv_pi;
        float adj = 0.5f;
        if (kf < 0.0f) {
            adj = -0.5f;
        }
        float kval = (float)(int)(kf + adj);
        x = x - kval * 2.0f * pi;

        // Reflection to [-pi/2, pi/2]
        if (x > half_pi) {
            x = pi - x;
        }
        if (x < -half_pi) {
            x = -pi - x;
        }

        // sin(x) using polynomial
        float x2 = x * x;
        float p = s7;
        p = p * x2 + s5;
        p = p * x2 + s3;
        p = p * x2 + s1;

        result[i] = p * x;
    }
}

// Cos float32: result[i] = cos(input[i])
// Uses range reduction to [-pi, pi], reflection to [0, pi/2], and polynomial
void cos_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants
    const float pi = 3.14159265358979323846f;
    const float inv_pi = 0.3183098861837907f;
    const float half_pi = 1.5707963267948966f;

    // Polynomial coefficients for cos(x) on [-pi/2, pi/2]
    const float c0 = 1.0f;
    const float c2 = -0.5f;
    const float c4 = 0.041666666666666664f;
    const float c6 = -0.001388888888888889f;

    float32x4_t v_pi = vdupq_n_f32(pi);
    float32x4_t v_half_pi = vdupq_n_f32(half_pi);
    float32x4_t v_inv_pi = vdupq_n_f32(inv_pi);
    float32x4_t v_two = vdupq_n_f32(2.0f);
    float32x4_t v_neg_one = vdupq_n_f32(-1.0f);
    float32x4_t v_one = vdupq_n_f32(1.0f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Range reduction: x = x - 2*pi*round(x/(2*pi)) -> x in [-pi, pi]
        float32x4_t k = vrndnq_f32(vmulq_f32(x, vmulq_f32(vdupq_n_f32(0.5f), v_inv_pi)));
        x = vfmsq_f32(x, k, vmulq_f32(v_two, v_pi));

        // cos(x) = cos(|x|) since cosine is even
        x = vabsq_f32(x);

        // Reflection: if |x| > pi/2, use cos(|x|) = -cos(pi - |x|)
        uint32x4_t need_reflect = vcgtq_f32(x, v_half_pi);
        float32x4_t x_reflected = vsubq_f32(v_pi, x);
        x = vbslq_f32(need_reflect, x_reflected, x);
        float32x4_t sign = vbslq_f32(need_reflect, v_neg_one, v_one);

        // cos(x) using polynomial: 1 + x^2*(c2 + x^2*(c4 + x^2*c6))
        float32x4_t x2 = vmulq_f32(x, x);

        float32x4_t p = vdupq_n_f32(c6);
        p = vfmaq_f32(vdupq_n_f32(c4), p, x2);
        p = vfmaq_f32(vdupq_n_f32(c2), p, x2);
        p = vfmaq_f32(vdupq_n_f32(c0), p, x2);

        // Apply sign from reflection
        p = vmulq_f32(p, sign);

        vst1q_f32(result + i, p);
    }

    // Scalar remainder
    for (; i < n; i++) {
        float x = input[i];

        // Range reduction
        float kf = x * 0.5f * inv_pi;
        float adj = 0.5f;
        if (kf < 0.0f) {
            adj = -0.5f;
        }
        float kval = (float)(int)(kf + adj);
        x = x - kval * 2.0f * pi;

        // cos(x) = cos(|x|)
        if (x < 0.0f) {
            x = -x;
        }

        // Reflection: if |x| > pi/2, use cos(|x|) = -cos(pi - |x|)
        float sign = 1.0f;
        if (x > half_pi) {
            x = pi - x;
            sign = -1.0f;
        }

        // cos(x) using polynomial
        float x2 = x * x;
        float p = c6;
        p = p * x2 + c4;
        p = p * x2 + c2;
        p = p * x2 + c0;

        result[i] = sign * p;
    }
}

// Tanh float32: result[i] = tanh(input[i])
// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
void tanh_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // For |x| > 9, tanh(x) ≈ sign(x)
    // For small x, use rational approximation

    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_neg_one = vdupq_n_f32(-1.0f);
    float32x4_t v_nine = vdupq_n_f32(9.0f);
    float32x4_t v_neg_nine = vdupq_n_f32(-9.0f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Clamp to prevent overflow
        float32x4_t x_clamped = vmaxq_f32(vminq_f32(x, v_nine), v_neg_nine);

        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        float32x4_t exp2x = x_clamped;

        // Simple exp approximation for 2x
        // This is a simplified version; for better accuracy, call exp
        float32x4_t two_x = vmulq_f32(x_clamped, vdupq_n_f32(2.0f));

        // Range reduction for exp
        const float ln2 = 0.6931471805599453f;
        const float inv_ln2 = 1.4426950408889634f;
        float32x4_t k = vrndnq_f32(vmulq_f32(two_x, vdupq_n_f32(inv_ln2)));
        float32x4_t r = vfmsq_f32(two_x, k, vdupq_n_f32(ln2));

        // exp(r) polynomial
        float32x4_t exp_r = vdupq_n_f32(1.0f);
        exp_r = vfmaq_f32(exp_r, r, vdupq_n_f32(1.0f));
        exp_r = vfmaq_f32(exp_r, vmulq_f32(r, r), vdupq_n_f32(0.5f));

        // Scale
        int32x4_t ki = vcvtq_s32_f32(k);
        int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23);
        float32x4_t scale = vreinterpretq_f32_s32(exp_bits);
        exp2x = vmulq_f32(exp_r, scale);

        // tanh = (exp2x - 1) / (exp2x + 1)
        float32x4_t num = vsubq_f32(exp2x, v_one);
        float32x4_t den = vaddq_f32(exp2x, v_one);
        float32x4_t res = vdivq_f32(num, den);

        vst1q_f32(result + i, res);
    }

    // Scalar remainder - inline tanh using exp approximation
    for (; i < n; i++) {
        float x = input[i];

        // Clamp to prevent overflow
        if (x > 9.0f) {
            result[i] = 1.0f;
        }
        if (x < -9.0f) {
            result[i] = -1.0f;
        }
        if (x >= -9.0f && x <= 9.0f) {
            // Compute exp(2x) using polynomial
            float two_x = 2.0f * x;
            const float ln2 = 0.6931471805599453f;
            const float inv_ln2 = 1.4426950408889634f;

            float kf = two_x * inv_ln2;
            float adj = 0.5f;
            if (kf < 0.0f) {
                adj = -0.5f;
            }
            int ki = (int)(kf + adj);
            float k = (float)ki;
            float r = two_x - k * ln2;

            // Polynomial for exp(r)
            float exp_r = 0.001388888888888889f;
            exp_r = exp_r * r + 0.008333333333333333f;
            exp_r = exp_r * r + 0.041666666666666664f;
            exp_r = exp_r * r + 0.16666666666666666f;
            exp_r = exp_r * r + 0.5f;
            exp_r = exp_r * r + 1.0f;
            exp_r = exp_r * r + 1.0f;

            // Compute 2^k
            float scale = 1.0f;
            int absK = ki;
            if (ki < 0) absK = -ki;
            long j = 0;
            if (ki >= 0) {
                for (j = 0; j < absK; j++) {
                    scale = scale * 2.0f;
                }
            }
            if (ki < 0) {
                for (j = 0; j < absK; j++) {
                    scale = scale * 0.5f;
                }
            }

            float exp2x = exp_r * scale;
            result[i] = (exp2x - 1.0f) / (exp2x + 1.0f);
        }
    }
}

// Sigmoid float32: result[i] = 1 / (1 + exp(-input[i]))
void sigmoid_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    float32x4_t v_one = vdupq_n_f32(1.0f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Clamp to prevent overflow
        x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
        x = vminq_f32(x, vdupq_n_f32(88.0f));

        // exp(-x)
        float32x4_t neg_x = vnegq_f32(x);

        // Range reduction for exp
        const float ln2 = 0.6931471805599453f;
        const float inv_ln2 = 1.4426950408889634f;
        float32x4_t k = vrndnq_f32(vmulq_f32(neg_x, vdupq_n_f32(inv_ln2)));
        float32x4_t r = vfmsq_f32(neg_x, k, vdupq_n_f32(ln2));

        // exp(r) polynomial
        float32x4_t exp_r = vdupq_n_f32(1.0f);
        exp_r = vfmaq_f32(exp_r, r, vdupq_n_f32(1.0f));
        float32x4_t r2 = vmulq_f32(r, r);
        exp_r = vfmaq_f32(exp_r, r2, vdupq_n_f32(0.5f));
        float32x4_t r3 = vmulq_f32(r2, r);
        exp_r = vfmaq_f32(exp_r, r3, vdupq_n_f32(0.16666667f));

        // Scale
        int32x4_t ki = vcvtq_s32_f32(k);
        int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23);
        float32x4_t scale = vreinterpretq_f32_s32(exp_bits);
        float32x4_t exp_neg_x = vmulq_f32(exp_r, scale);

        // sigmoid = 1 / (1 + exp(-x))
        float32x4_t res = vdivq_f32(v_one, vaddq_f32(v_one, exp_neg_x));

        vst1q_f32(result + i, res);
    }

    // Scalar remainder - inline sigmoid using exp approximation
    for (; i < n; i++) {
        float x = input[i];
        if (x < -88.0f) x = -88.0f;
        if (x > 88.0f) x = 88.0f;

        // Compute exp(-x)
        float neg_x = -x;
        const float ln2 = 0.6931471805599453f;
        const float inv_ln2 = 1.4426950408889634f;

        float kf = neg_x * inv_ln2;
        float adj = 0.5f;
        if (kf < 0.0f) {
            adj = -0.5f;
        }
        int ki = (int)(kf + adj);
        float k = (float)ki;
        float r = neg_x - k * ln2;

        // Polynomial for exp(r)
        float exp_r = 0.001388888888888889f;
        exp_r = exp_r * r + 0.008333333333333333f;
        exp_r = exp_r * r + 0.041666666666666664f;
        exp_r = exp_r * r + 0.16666666666666666f;
        exp_r = exp_r * r + 0.5f;
        exp_r = exp_r * r + 1.0f;
        exp_r = exp_r * r + 1.0f;

        // Compute 2^k
        float scale = 1.0f;
        int absK = ki;
        if (ki < 0) absK = -ki;
        long j = 0;
        if (ki >= 0) {
            for (j = 0; j < absK; j++) {
                scale = scale * 2.0f;
            }
        }
        if (ki < 0) {
            for (j = 0; j < absK; j++) {
                scale = scale * 0.5f;
            }
        }

        float exp_neg_x = exp_r * scale;
        result[i] = 1.0f / (1.0f + exp_neg_x);
    }
}

