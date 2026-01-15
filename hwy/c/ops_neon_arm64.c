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

// Vector subtraction: result[i] = a[i] - b[i]
void sub_f64_neon(double *a, double *b, double *result, long *len) {
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

        vst1q_f64(result + i, vsubq_f64(a0, b0));
        vst1q_f64(result + i + 2, vsubq_f64(a1, b1));
        vst1q_f64(result + i + 4, vsubq_f64(a2, b2));
        vst1q_f64(result + i + 6, vsubq_f64(a3, b3));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        vst1q_f64(result + i, vsubq_f64(av, bv));
    }

    for (; i < n; i++) {
        result[i] = a[i] - b[i];
    }
}

// Vector division: result[i] = a[i] / b[i]
void div_f64_neon(double *a, double *b, double *result, long *len) {
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

        vst1q_f64(result + i, vdivq_f64(a0, b0));
        vst1q_f64(result + i + 2, vdivq_f64(a1, b1));
        vst1q_f64(result + i + 4, vdivq_f64(a2, b2));
        vst1q_f64(result + i + 6, vdivq_f64(a3, b3));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        vst1q_f64(result + i, vdivq_f64(av, bv));
    }

    for (; i < n; i++) {
        result[i] = a[i] / b[i];
    }
}

// Element-wise minimum: result[i] = min(a[i], b[i])
void min_f64_neon(double *a, double *b, double *result, long *len) {
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

        vst1q_f64(result + i, vminq_f64(a0, b0));
        vst1q_f64(result + i + 2, vminq_f64(a1, b1));
        vst1q_f64(result + i + 4, vminq_f64(a2, b2));
        vst1q_f64(result + i + 6, vminq_f64(a3, b3));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        vst1q_f64(result + i, vminq_f64(av, bv));
    }

    for (; i < n; i++) {
        if (a[i] < b[i]) {
            result[i] = a[i];
        }
        if (a[i] >= b[i]) {
            result[i] = b[i];
        }
    }
}

// Element-wise maximum: result[i] = max(a[i], b[i])
void max_f64_neon(double *a, double *b, double *result, long *len) {
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

        vst1q_f64(result + i, vmaxq_f64(a0, b0));
        vst1q_f64(result + i + 2, vmaxq_f64(a1, b1));
        vst1q_f64(result + i + 4, vmaxq_f64(a2, b2));
        vst1q_f64(result + i + 6, vmaxq_f64(a3, b3));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        vst1q_f64(result + i, vmaxq_f64(av, bv));
    }

    for (; i < n; i++) {
        if (a[i] > b[i]) {
            result[i] = a[i];
        }
        if (a[i] <= b[i]) {
            result[i] = b[i];
        }
    }
}

// Square root: result[i] = sqrt(a[i])
void sqrt_f64_neon(double *a, double *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 7 < n; i += 8) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t a2 = vld1q_f64(a + i + 4);
        float64x2_t a3 = vld1q_f64(a + i + 6);

        vst1q_f64(result + i, vsqrtq_f64(a0));
        vst1q_f64(result + i + 2, vsqrtq_f64(a1));
        vst1q_f64(result + i + 4, vsqrtq_f64(a2));
        vst1q_f64(result + i + 6, vsqrtq_f64(a3));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        vst1q_f64(result + i, vsqrtq_f64(av));
    }

    // Scalar remainder - use NEON to compute sqrt for single element
    for (; i < n; i++) {
        float64x2_t v = vdupq_n_f64(a[i]);
        v = vsqrtq_f64(v);
        result[i] = vgetq_lane_f64(v, 0);
    }
}

// Absolute value: result[i] = |a[i]|
void abs_f64_neon(double *a, double *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 7 < n; i += 8) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t a2 = vld1q_f64(a + i + 4);
        float64x2_t a3 = vld1q_f64(a + i + 6);

        vst1q_f64(result + i, vabsq_f64(a0));
        vst1q_f64(result + i + 2, vabsq_f64(a1));
        vst1q_f64(result + i + 4, vabsq_f64(a2));
        vst1q_f64(result + i + 6, vabsq_f64(a3));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        vst1q_f64(result + i, vabsq_f64(av));
    }

    for (; i < n; i++) {
        double val = a[i];
        if (val < 0) {
            val = -val;
        }
        result[i] = val;
    }
}

// Negation: result[i] = -a[i]
void neg_f64_neon(double *a, double *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 7 < n; i += 8) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t a2 = vld1q_f64(a + i + 4);
        float64x2_t a3 = vld1q_f64(a + i + 6);

        vst1q_f64(result + i, vnegq_f64(a0));
        vst1q_f64(result + i + 2, vnegq_f64(a1));
        vst1q_f64(result + i + 4, vnegq_f64(a2));
        vst1q_f64(result + i + 6, vnegq_f64(a3));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        vst1q_f64(result + i, vnegq_f64(av));
    }

    for (; i < n; i++) {
        result[i] = -a[i];
    }
}

// Reduce minimum: result = min(input[0], input[1], ..., input[n-1])
void reduce_min_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    if (n <= 0) {
        *result = 0.0;
        return;
    }

    long i = 0;
    double min_val = input[0];

    if (n >= 8) {
        float64x2_t min0 = vld1q_f64(input);
        float64x2_t min1 = min0;
        float64x2_t min2 = min0;
        float64x2_t min3 = min0;
        i = 2;

        for (; i + 7 < n; i += 8) {
            min0 = vminq_f64(min0, vld1q_f64(input + i));
            min1 = vminq_f64(min1, vld1q_f64(input + i + 2));
            min2 = vminq_f64(min2, vld1q_f64(input + i + 4));
            min3 = vminq_f64(min3, vld1q_f64(input + i + 6));
        }

        min0 = vminq_f64(min0, min1);
        min2 = vminq_f64(min2, min3);
        min0 = vminq_f64(min0, min2);

        // Extract min from 2-lane vector (no vminvq_f64 in NEON)
        double lane0 = vgetq_lane_f64(min0, 0);
        double lane1 = vgetq_lane_f64(min0, 1);
        if (lane0 < lane1) {
            min_val = lane0;
        }
        if (lane0 >= lane1) {
            min_val = lane1;
        }
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t v = vld1q_f64(input + i);
        double lane0 = vgetq_lane_f64(v, 0);
        double lane1 = vgetq_lane_f64(v, 1);
        if (lane0 < min_val) {
            min_val = lane0;
        }
        if (lane1 < min_val) {
            min_val = lane1;
        }
    }

    for (; i < n; i++) {
        if (input[i] < min_val) {
            min_val = input[i];
        }
    }

    *result = min_val;
}

// Reduce maximum: result = max(input[0], input[1], ..., input[n-1])
void reduce_max_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    if (n <= 0) {
        *result = 0.0;
        return;
    }

    long i = 0;
    double max_val = input[0];

    if (n >= 8) {
        float64x2_t max0 = vld1q_f64(input);
        float64x2_t max1 = max0;
        float64x2_t max2 = max0;
        float64x2_t max3 = max0;
        i = 2;

        for (; i + 7 < n; i += 8) {
            max0 = vmaxq_f64(max0, vld1q_f64(input + i));
            max1 = vmaxq_f64(max1, vld1q_f64(input + i + 2));
            max2 = vmaxq_f64(max2, vld1q_f64(input + i + 4));
            max3 = vmaxq_f64(max3, vld1q_f64(input + i + 6));
        }

        max0 = vmaxq_f64(max0, max1);
        max2 = vmaxq_f64(max2, max3);
        max0 = vmaxq_f64(max0, max2);

        // Extract max from 2-lane vector (no vmaxvq_f64 in NEON)
        double lane0 = vgetq_lane_f64(max0, 0);
        double lane1 = vgetq_lane_f64(max0, 1);
        if (lane0 > lane1) {
            max_val = lane0;
        }
        if (lane0 <= lane1) {
            max_val = lane1;
        }
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t v = vld1q_f64(input + i);
        double lane0 = vgetq_lane_f64(v, 0);
        double lane1 = vgetq_lane_f64(v, 1);
        if (lane0 > max_val) {
            max_val = lane0;
        }
        if (lane1 > max_val) {
            max_val = lane1;
        }
    }

    for (; i < n; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    *result = max_val;
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

// Gather int64: result[i] = base[indices[i]]
void gather_i64_neon(long *base, int *indices, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 2 elements at a time (int64x2_t)
    for (; i + 1 < n; i += 2) {
        long tmp[2];
        tmp[0] = base[indices[i]];
        tmp[1] = base[indices[i + 1]];
        vst1q_s64(result + i, vld1q_s64(tmp));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = base[indices[i]];
    }
}

// Scatter int64: base[indices[i]] = values[i]
void scatter_i64_neon(long *values, int *indices, long *base, long *len) {
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

// Masked load float64: result[i] = mask[i] ? input[i] : 0
void masked_load_f64_neon(double *input, long *mask, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 2 elements at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t v = vld1q_f64(input + i);
        int64x2_t m = vld1q_s64(mask + i);
        // Convert mask to all 1s or 0s: compare != 0
        uint64x2_t cmp = vcgtq_s64(m, vdupq_n_s64(0));
        // Use bit select: where mask is 1, use v; where 0, use zero
        float64x2_t zero = vdupq_n_f64(0);
        float64x2_t selected = vbslq_f64(cmp, v, zero);
        vst1q_f64(result + i, selected);
    }

    // Scalar remainder
    for (; i < n; i++) {
        if (mask[i]) {
            result[i] = input[i];
        }
        if (!mask[i]) {
            result[i] = 0.0;
        }
    }
}

// Masked store float64: if mask[i] then output[i] = input[i]
void masked_store_f64_neon(double *input, long *mask, double *output, long *len) {
    long n = *len;

    for (long i = 0; i < n; i++) {
        if (mask[i]) {
            output[i] = input[i];
        }
    }
}

// Masked load int32: result[i] = mask[i] ? input[i] : 0
void masked_load_i32_neon(int *input, int *mask, int *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t v = vld1q_s32(input + i);
        int32x4_t m = vld1q_s32(mask + i);
        // Convert mask to all 1s or 0s: compare != 0
        uint32x4_t cmp = vcgtq_s32(m, vdupq_n_s32(0));
        // Use bit select: where mask is 1, use v; where 0, use zero
        int32x4_t zero = vdupq_n_s32(0);
        int32x4_t selected = vbslq_s32(cmp, v, zero);
        vst1q_s32(result + i, selected);
    }

    // Scalar remainder
    for (; i < n; i++) {
        if (mask[i]) {
            result[i] = input[i];
        }
        if (!mask[i]) {
            result[i] = 0;
        }
    }
}

// Masked store int32: if mask[i] then output[i] = input[i]
void masked_store_i32_neon(int *input, int *mask, int *output, long *len) {
    long n = *len;

    for (long i = 0; i < n; i++) {
        if (mask[i]) {
            output[i] = input[i];
        }
    }
}

// Masked load int64: result[i] = mask[i] ? input[i] : 0
void masked_load_i64_neon(long *input, long *mask, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 2 elements at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t v = vld1q_s64(input + i);
        int64x2_t m = vld1q_s64(mask + i);
        // Convert mask to all 1s or 0s: compare != 0
        uint64x2_t cmp = vcgtq_s64(m, vdupq_n_s64(0));
        // Use bit select: where mask is 1, use v; where 0, use zero
        int64x2_t zero = vdupq_n_s64(0);
        int64x2_t selected = vbslq_s64(cmp, v, zero);
        vst1q_s64(result + i, selected);
    }

    // Scalar remainder
    for (; i < n; i++) {
        if (mask[i]) {
            result[i] = input[i];
        }
        if (!mask[i]) {
            result[i] = 0;
        }
    }
}

// Masked store int64: if mask[i] then output[i] = input[i]
void masked_store_i64_neon(long *input, long *mask, long *output, long *len) {
    long n = *len;

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
// Float64 Comparison Operations (2 lanes per 128-bit vector)
// ============================================================================

// Equal float64: result[i] = (a[i] == b[i]) ? 0xFFFFFFFFFFFFFFFF : 0
void eq_f64_neon(double *a, double *b, long *result, long *len) {
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

        vst1q_s64(result + i, vreinterpretq_s64_u64(vceqq_f64(a0, b0)));
        vst1q_s64(result + i + 2, vreinterpretq_s64_u64(vceqq_f64(a1, b1)));
        vst1q_s64(result + i + 4, vreinterpretq_s64_u64(vceqq_f64(a2, b2)));
        vst1q_s64(result + i + 6, vreinterpretq_s64_u64(vceqq_f64(a3, b3)));
    }

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        vst1q_s64(result + i, vreinterpretq_s64_u64(vceqq_f64(av, bv)));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] == b[i]) ? -1L : 0L;
    }
}

// Greater than float64: result[i] = (a[i] > b[i]) ? 0xFFFFFFFFFFFFFFFF : 0
void gt_f64_neon(double *a, double *b, long *result, long *len) {
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

        vst1q_s64(result + i, vreinterpretq_s64_u64(vcgtq_f64(a0, b0)));
        vst1q_s64(result + i + 2, vreinterpretq_s64_u64(vcgtq_f64(a1, b1)));
        vst1q_s64(result + i + 4, vreinterpretq_s64_u64(vcgtq_f64(a2, b2)));
        vst1q_s64(result + i + 6, vreinterpretq_s64_u64(vcgtq_f64(a3, b3)));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        vst1q_s64(result + i, vreinterpretq_s64_u64(vcgtq_f64(av, bv)));
    }

    for (; i < n; i++) {
        result[i] = (a[i] > b[i]) ? -1L : 0L;
    }
}

// Greater than or equal float64: result[i] = (a[i] >= b[i]) ? 0xFFFFFFFFFFFFFFFF : 0
void ge_f64_neon(double *a, double *b, long *result, long *len) {
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

        vst1q_s64(result + i, vreinterpretq_s64_u64(vcgeq_f64(a0, b0)));
        vst1q_s64(result + i + 2, vreinterpretq_s64_u64(vcgeq_f64(a1, b1)));
        vst1q_s64(result + i + 4, vreinterpretq_s64_u64(vcgeq_f64(a2, b2)));
        vst1q_s64(result + i + 6, vreinterpretq_s64_u64(vcgeq_f64(a3, b3)));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        vst1q_s64(result + i, vreinterpretq_s64_u64(vcgeq_f64(av, bv)));
    }

    for (; i < n; i++) {
        result[i] = (a[i] >= b[i]) ? -1L : 0L;
    }
}

// Less than float64: result[i] = (a[i] < b[i]) ? 0xFFFFFFFFFFFFFFFF : 0
void lt_f64_neon(double *a, double *b, long *result, long *len) {
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

        vst1q_s64(result + i, vreinterpretq_s64_u64(vcltq_f64(a0, b0)));
        vst1q_s64(result + i + 2, vreinterpretq_s64_u64(vcltq_f64(a1, b1)));
        vst1q_s64(result + i + 4, vreinterpretq_s64_u64(vcltq_f64(a2, b2)));
        vst1q_s64(result + i + 6, vreinterpretq_s64_u64(vcltq_f64(a3, b3)));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        vst1q_s64(result + i, vreinterpretq_s64_u64(vcltq_f64(av, bv)));
    }

    for (; i < n; i++) {
        result[i] = (a[i] < b[i]) ? -1L : 0L;
    }
}

// Less than or equal float64: result[i] = (a[i] <= b[i]) ? 0xFFFFFFFFFFFFFFFF : 0
void le_f64_neon(double *a, double *b, long *result, long *len) {
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

        vst1q_s64(result + i, vreinterpretq_s64_u64(vcleq_f64(a0, b0)));
        vst1q_s64(result + i + 2, vreinterpretq_s64_u64(vcleq_f64(a1, b1)));
        vst1q_s64(result + i + 4, vreinterpretq_s64_u64(vcleq_f64(a2, b2)));
        vst1q_s64(result + i + 6, vreinterpretq_s64_u64(vcleq_f64(a3, b3)));
    }

    for (; i + 1 < n; i += 2) {
        float64x2_t av = vld1q_f64(a + i);
        float64x2_t bv = vld1q_f64(b + i);
        vst1q_s64(result + i, vreinterpretq_s64_u64(vcleq_f64(av, bv)));
    }

    for (; i < n; i++) {
        result[i] = (a[i] <= b[i]) ? -1L : 0L;
    }
}

// ============================================================================
// Power of 2 Operations (for exp/log implementations)
// ============================================================================

// Pow2F32: Compute 2^k for int32 k, result as float32
// Uses IEEE 754: 2^k = ((k + 127) << 23) as float32 bits
// Handles overflow (k > 127) -> +Inf, underflow (k < -126) -> 0
void pow2_f32_neon(int *k, float *result, long *len) {
    long n = *len;
    long i = 0;

    int32x4_t bias = vdupq_n_s32(127);
    int32x4_t min_exp = vdupq_n_s32(-126);
    int32x4_t max_exp = vdupq_n_s32(127);
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t inf = vdupq_n_f32(1.0f / 0.0f);

    // Process 16 elements at a time
    for (; i + 15 < n; i += 16) {
        int32x4_t k0 = vld1q_s32(k + i);
        int32x4_t k1 = vld1q_s32(k + i + 4);
        int32x4_t k2 = vld1q_s32(k + i + 8);
        int32x4_t k3 = vld1q_s32(k + i + 12);

        // Compute bits = (k + 127) << 23
        int32x4_t bits0 = vshlq_n_s32(vaddq_s32(k0, bias), 23);
        int32x4_t bits1 = vshlq_n_s32(vaddq_s32(k1, bias), 23);
        int32x4_t bits2 = vshlq_n_s32(vaddq_s32(k2, bias), 23);
        int32x4_t bits3 = vshlq_n_s32(vaddq_s32(k3, bias), 23);

        // Reinterpret as float
        float32x4_t r0 = vreinterpretq_f32_s32(bits0);
        float32x4_t r1 = vreinterpretq_f32_s32(bits1);
        float32x4_t r2 = vreinterpretq_f32_s32(bits2);
        float32x4_t r3 = vreinterpretq_f32_s32(bits3);

        // Handle underflow: k < -126 -> 0
        uint32x4_t under0 = vcltq_s32(k0, min_exp);
        uint32x4_t under1 = vcltq_s32(k1, min_exp);
        uint32x4_t under2 = vcltq_s32(k2, min_exp);
        uint32x4_t under3 = vcltq_s32(k3, min_exp);

        r0 = vbslq_f32(under0, zero, r0);
        r1 = vbslq_f32(under1, zero, r1);
        r2 = vbslq_f32(under2, zero, r2);
        r3 = vbslq_f32(under3, zero, r3);

        // Handle overflow: k > 127 -> inf
        uint32x4_t over0 = vcgtq_s32(k0, max_exp);
        uint32x4_t over1 = vcgtq_s32(k1, max_exp);
        uint32x4_t over2 = vcgtq_s32(k2, max_exp);
        uint32x4_t over3 = vcgtq_s32(k3, max_exp);

        r0 = vbslq_f32(over0, inf, r0);
        r1 = vbslq_f32(over1, inf, r1);
        r2 = vbslq_f32(over2, inf, r2);
        r3 = vbslq_f32(over3, inf, r3);

        vst1q_f32(result + i, r0);
        vst1q_f32(result + i + 4, r1);
        vst1q_f32(result + i + 8, r2);
        vst1q_f32(result + i + 12, r3);
    }

    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t kv = vld1q_s32(k + i);
        int32x4_t bits = vshlq_n_s32(vaddq_s32(kv, bias), 23);
        float32x4_t r = vreinterpretq_f32_s32(bits);

        uint32x4_t under = vcltq_s32(kv, min_exp);
        r = vbslq_f32(under, zero, r);

        uint32x4_t over = vcgtq_s32(kv, max_exp);
        r = vbslq_f32(over, inf, r);

        vst1q_f32(result + i, r);
    }

    // Scalar remainder - simplified, just handle edge cases
    // The vectorized path handles all aligned elements
    for (; i < n; i++) {
        int kv = k[i];
        if (kv < -126) {
            result[i] = 0.0f;
        }
        if (kv > 127) {
            result[i] = 1.0f / 0.0f;
        }
        // For valid range, use single-element NEON
        if (kv >= -126) {
            if (kv <= 127) {
                int32x4_t kv_vec = vdupq_n_s32(kv);
                int32x4_t bias = vdupq_n_s32(127);
                int32x4_t bits = vshlq_n_s32(vaddq_s32(kv_vec, bias), 23);
                float32x4_t r = vreinterpretq_f32_s32(bits);
                result[i] = vgetq_lane_f32(r, 0);
            }
        }
    }
}

// Pow2F64: Compute 2^k for int32 k, result as float64
// Uses IEEE 754: 2^k = ((k + 1023) << 52) as float64 bits
void pow2_f64_neon(int *k, double *result, long *len) {
    long n = *len;
    long i = 0;

    int32x2_t bias32 = vdup_n_s32(1023);
    int32x2_t min_exp = vdup_n_s32(-1022);
    int32x2_t max_exp = vdup_n_s32(1023);
    float64x2_t zero = vdupq_n_f64(0.0);
    float64x2_t inf = vdupq_n_f64(1.0 / 0.0);

    // Process 8 elements at a time
    for (; i + 7 < n; i += 8) {
        // Load 8 int32 values as 4 pairs
        int32x2_t k0 = vld1_s32(k + i);
        int32x2_t k1 = vld1_s32(k + i + 2);
        int32x2_t k2 = vld1_s32(k + i + 4);
        int32x2_t k3 = vld1_s32(k + i + 6);

        // Add bias and widen to int64
        int64x2_t biased0 = vmovl_s32(vadd_s32(k0, bias32));
        int64x2_t biased1 = vmovl_s32(vadd_s32(k1, bias32));
        int64x2_t biased2 = vmovl_s32(vadd_s32(k2, bias32));
        int64x2_t biased3 = vmovl_s32(vadd_s32(k3, bias32));

        // Shift left by 52
        int64x2_t bits0 = vshlq_n_s64(biased0, 52);
        int64x2_t bits1 = vshlq_n_s64(biased1, 52);
        int64x2_t bits2 = vshlq_n_s64(biased2, 52);
        int64x2_t bits3 = vshlq_n_s64(biased3, 52);

        // Reinterpret as float64
        float64x2_t r0 = vreinterpretq_f64_s64(bits0);
        float64x2_t r1 = vreinterpretq_f64_s64(bits1);
        float64x2_t r2 = vreinterpretq_f64_s64(bits2);
        float64x2_t r3 = vreinterpretq_f64_s64(bits3);

        // Handle underflow: k < -1022 -> 0
        uint32x2_t under0 = vclt_s32(k0, min_exp);
        uint32x2_t under1 = vclt_s32(k1, min_exp);
        uint32x2_t under2 = vclt_s32(k2, min_exp);
        uint32x2_t under3 = vclt_s32(k3, min_exp);

        // Widen mask to 64-bit
        uint64x2_t under0_64 = vmovl_u32(under0);
        uint64x2_t under1_64 = vmovl_u32(under1);
        uint64x2_t under2_64 = vmovl_u32(under2);
        uint64x2_t under3_64 = vmovl_u32(under3);

        r0 = vbslq_f64(under0_64, zero, r0);
        r1 = vbslq_f64(under1_64, zero, r1);
        r2 = vbslq_f64(under2_64, zero, r2);
        r3 = vbslq_f64(under3_64, zero, r3);

        // Handle overflow: k > 1023 -> inf
        uint32x2_t over0 = vcgt_s32(k0, max_exp);
        uint32x2_t over1 = vcgt_s32(k1, max_exp);
        uint32x2_t over2 = vcgt_s32(k2, max_exp);
        uint32x2_t over3 = vcgt_s32(k3, max_exp);

        uint64x2_t over0_64 = vmovl_u32(over0);
        uint64x2_t over1_64 = vmovl_u32(over1);
        uint64x2_t over2_64 = vmovl_u32(over2);
        uint64x2_t over3_64 = vmovl_u32(over3);

        r0 = vbslq_f64(over0_64, inf, r0);
        r1 = vbslq_f64(over1_64, inf, r1);
        r2 = vbslq_f64(over2_64, inf, r2);
        r3 = vbslq_f64(over3_64, inf, r3);

        vst1q_f64(result + i, r0);
        vst1q_f64(result + i + 2, r1);
        vst1q_f64(result + i + 4, r2);
        vst1q_f64(result + i + 6, r3);
    }

    // Process 2 elements at a time
    for (; i + 1 < n; i += 2) {
        int32x2_t kv = vld1_s32(k + i);
        int64x2_t biased = vmovl_s32(vadd_s32(kv, bias32));
        int64x2_t bits = vshlq_n_s64(biased, 52);
        float64x2_t r = vreinterpretq_f64_s64(bits);

        uint32x2_t under = vclt_s32(kv, min_exp);
        uint64x2_t under_64 = vmovl_u32(under);
        r = vbslq_f64(under_64, zero, r);

        uint32x2_t over = vcgt_s32(kv, max_exp);
        uint64x2_t over_64 = vmovl_u32(over);
        r = vbslq_f64(over_64, inf, r);

        vst1q_f64(result + i, r);
    }

    // Scalar remainder - simplified, just handle edge cases
    for (; i < n; i++) {
        int kv = k[i];
        if (kv < -1022) {
            result[i] = 0.0;
        }
        if (kv > 1023) {
            result[i] = 1.0 / 0.0;
        }
        // For valid range, use single-element NEON to avoid type punning
        if (kv >= -1022) {
            if (kv <= 1023) {
                int32x2_t kv_vec = vdup_n_s32(kv);
                int32x2_t bias = vdup_n_s32(1023);
                int64x2_t biased = vmovl_s32(vadd_s32(kv_vec, bias));
                int64x2_t bits = vshlq_n_s64(biased, 52);
                float64x2_t r = vreinterpretq_f64_s64(bits);
                result[i] = vgetq_lane_f64(r, 0);
            }
        }
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

// ============================================================================
// Int32 Arithmetic Operations
// ============================================================================

// Vector addition int32: result[i] = a[i] + b[i]
void add_i32_neon(int *a, int *b, int *result, long *len) {
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

        vst1q_s32(result + i, vaddq_s32(a0, b0));
        vst1q_s32(result + i + 4, vaddq_s32(a1, b1));
        vst1q_s32(result + i + 8, vaddq_s32(a2, b2));
        vst1q_s32(result + i + 12, vaddq_s32(a3, b3));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vaddq_s32(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

// Vector subtraction int32: result[i] = a[i] - b[i]
void sub_i32_neon(int *a, int *b, int *result, long *len) {
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

        vst1q_s32(result + i, vsubq_s32(a0, b0));
        vst1q_s32(result + i + 4, vsubq_s32(a1, b1));
        vst1q_s32(result + i + 8, vsubq_s32(a2, b2));
        vst1q_s32(result + i + 12, vsubq_s32(a3, b3));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vsubq_s32(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] - b[i];
    }
}

// Vector multiplication int32: result[i] = a[i] * b[i]
void mul_i32_neon(int *a, int *b, int *result, long *len) {
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

        vst1q_s32(result + i, vmulq_s32(a0, b0));
        vst1q_s32(result + i + 4, vmulq_s32(a1, b1));
        vst1q_s32(result + i + 8, vmulq_s32(a2, b2));
        vst1q_s32(result + i + 12, vmulq_s32(a3, b3));
    }

    // Process 4 ints at a time
    for (; i + 3 < n; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        vst1q_s32(result + i, vmulq_s32(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

// ============================================================================
// Int64 Arithmetic Operations
// ============================================================================

// Vector addition int64: result[i] = a[i] + b[i]
void add_i64_neon(long *a, long *b, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 longs at a time (4 vectors)
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        int64x2_t b0 = vld1q_s64(b + i);
        int64x2_t b1 = vld1q_s64(b + i + 2);
        int64x2_t b2 = vld1q_s64(b + i + 4);
        int64x2_t b3 = vld1q_s64(b + i + 6);

        vst1q_s64(result + i, vaddq_s64(a0, b0));
        vst1q_s64(result + i + 2, vaddq_s64(a1, b1));
        vst1q_s64(result + i + 4, vaddq_s64(a2, b2));
        vst1q_s64(result + i + 6, vaddq_s64(a3, b3));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        int64x2_t bv = vld1q_s64(b + i);
        vst1q_s64(result + i, vaddq_s64(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

// Vector subtraction int64: result[i] = a[i] - b[i]
void sub_i64_neon(long *a, long *b, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        int64x2_t b0 = vld1q_s64(b + i);
        int64x2_t b1 = vld1q_s64(b + i + 2);
        int64x2_t b2 = vld1q_s64(b + i + 4);
        int64x2_t b3 = vld1q_s64(b + i + 6);

        vst1q_s64(result + i, vsubq_s64(a0, b0));
        vst1q_s64(result + i + 2, vsubq_s64(a1, b1));
        vst1q_s64(result + i + 4, vsubq_s64(a2, b2));
        vst1q_s64(result + i + 6, vsubq_s64(a3, b3));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        int64x2_t bv = vld1q_s64(b + i);
        vst1q_s64(result + i, vsubq_s64(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] - b[i];
    }
}

// ============================================================================
// Int64 Bitwise Operations
// ============================================================================

// Bitwise AND int64: result[i] = a[i] & b[i]
void and_i64_neon(long *a, long *b, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        int64x2_t b0 = vld1q_s64(b + i);
        int64x2_t b1 = vld1q_s64(b + i + 2);
        int64x2_t b2 = vld1q_s64(b + i + 4);
        int64x2_t b3 = vld1q_s64(b + i + 6);

        vst1q_s64(result + i, vandq_s64(a0, b0));
        vst1q_s64(result + i + 2, vandq_s64(a1, b1));
        vst1q_s64(result + i + 4, vandq_s64(a2, b2));
        vst1q_s64(result + i + 6, vandq_s64(a3, b3));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        int64x2_t bv = vld1q_s64(b + i);
        vst1q_s64(result + i, vandq_s64(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] & b[i];
    }
}

// Bitwise OR int64: result[i] = a[i] | b[i]
void or_i64_neon(long *a, long *b, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        int64x2_t b0 = vld1q_s64(b + i);
        int64x2_t b1 = vld1q_s64(b + i + 2);
        int64x2_t b2 = vld1q_s64(b + i + 4);
        int64x2_t b3 = vld1q_s64(b + i + 6);

        vst1q_s64(result + i, vorrq_s64(a0, b0));
        vst1q_s64(result + i + 2, vorrq_s64(a1, b1));
        vst1q_s64(result + i + 4, vorrq_s64(a2, b2));
        vst1q_s64(result + i + 6, vorrq_s64(a3, b3));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        int64x2_t bv = vld1q_s64(b + i);
        vst1q_s64(result + i, vorrq_s64(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] | b[i];
    }
}

// Bitwise XOR int64: result[i] = a[i] ^ b[i]
void xor_i64_neon(long *a, long *b, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        int64x2_t b0 = vld1q_s64(b + i);
        int64x2_t b1 = vld1q_s64(b + i + 2);
        int64x2_t b2 = vld1q_s64(b + i + 4);
        int64x2_t b3 = vld1q_s64(b + i + 6);

        vst1q_s64(result + i, veorq_s64(a0, b0));
        vst1q_s64(result + i + 2, veorq_s64(a1, b1));
        vst1q_s64(result + i + 4, veorq_s64(a2, b2));
        vst1q_s64(result + i + 6, veorq_s64(a3, b3));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        int64x2_t bv = vld1q_s64(b + i);
        vst1q_s64(result + i, veorq_s64(av, bv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] ^ b[i];
    }
}

// ============================================================================
// Int64 Shift Operations
// ============================================================================

// Left shift int64: result[i] = a[i] << shift
void shl_i64_neon(long *a, long *result, long *shift, long *len) {
    long n = *len;
    long s = *shift;
    long i = 0;

    int64x2_t shift_vec = vdupq_n_s64(s);

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        vst1q_s64(result + i, vshlq_s64(a0, shift_vec));
        vst1q_s64(result + i + 2, vshlq_s64(a1, shift_vec));
        vst1q_s64(result + i + 4, vshlq_s64(a2, shift_vec));
        vst1q_s64(result + i + 6, vshlq_s64(a3, shift_vec));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        vst1q_s64(result + i, vshlq_s64(av, shift_vec));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] << s;
    }
}

// Arithmetic right shift int64: result[i] = a[i] >> shift
void shr_i64_neon(long *a, long *result, long *shift, long *len) {
    long n = *len;
    long s = *shift;
    long i = 0;

    // For right shift, use negative shift value
    int64x2_t shift_vec = vdupq_n_s64(-s);

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        vst1q_s64(result + i, vshlq_s64(a0, shift_vec));
        vst1q_s64(result + i + 2, vshlq_s64(a1, shift_vec));
        vst1q_s64(result + i + 4, vshlq_s64(a2, shift_vec));
        vst1q_s64(result + i + 6, vshlq_s64(a3, shift_vec));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        vst1q_s64(result + i, vshlq_s64(av, shift_vec));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = a[i] >> s;
    }
}

// ============================================================================
// Int64 Comparison Operations
// ============================================================================

// Equal int64: result[i] = (a[i] == b[i]) ? -1 : 0
void eq_i64_neon(long *a, long *b, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        int64x2_t b0 = vld1q_s64(b + i);
        int64x2_t b1 = vld1q_s64(b + i + 2);
        int64x2_t b2 = vld1q_s64(b + i + 4);
        int64x2_t b3 = vld1q_s64(b + i + 6);

        uint64x2_t cmp0 = vceqq_s64(a0, b0);
        uint64x2_t cmp1 = vceqq_s64(a1, b1);
        uint64x2_t cmp2 = vceqq_s64(a2, b2);
        uint64x2_t cmp3 = vceqq_s64(a3, b3);

        vst1q_s64(result + i, vreinterpretq_s64_u64(cmp0));
        vst1q_s64(result + i + 2, vreinterpretq_s64_u64(cmp1));
        vst1q_s64(result + i + 4, vreinterpretq_s64_u64(cmp2));
        vst1q_s64(result + i + 6, vreinterpretq_s64_u64(cmp3));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        int64x2_t bv = vld1q_s64(b + i);
        uint64x2_t cmp = vceqq_s64(av, bv);
        vst1q_s64(result + i, vreinterpretq_s64_u64(cmp));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] == b[i]) ? -1L : 0L;
    }
}

// Greater than int64: result[i] = (a[i] > b[i]) ? -1 : 0
void gt_i64_neon(long *a, long *b, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        int64x2_t b0 = vld1q_s64(b + i);
        int64x2_t b1 = vld1q_s64(b + i + 2);
        int64x2_t b2 = vld1q_s64(b + i + 4);
        int64x2_t b3 = vld1q_s64(b + i + 6);

        uint64x2_t cmp0 = vcgtq_s64(a0, b0);
        uint64x2_t cmp1 = vcgtq_s64(a1, b1);
        uint64x2_t cmp2 = vcgtq_s64(a2, b2);
        uint64x2_t cmp3 = vcgtq_s64(a3, b3);

        vst1q_s64(result + i, vreinterpretq_s64_u64(cmp0));
        vst1q_s64(result + i + 2, vreinterpretq_s64_u64(cmp1));
        vst1q_s64(result + i + 4, vreinterpretq_s64_u64(cmp2));
        vst1q_s64(result + i + 6, vreinterpretq_s64_u64(cmp3));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        int64x2_t bv = vld1q_s64(b + i);
        uint64x2_t cmp = vcgtq_s64(av, bv);
        vst1q_s64(result + i, vreinterpretq_s64_u64(cmp));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] > b[i]) ? -1L : 0L;
    }
}

// Greater or equal int64: result[i] = (a[i] >= b[i]) ? -1 : 0
void ge_i64_neon(long *a, long *b, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        int64x2_t b0 = vld1q_s64(b + i);
        int64x2_t b1 = vld1q_s64(b + i + 2);
        int64x2_t b2 = vld1q_s64(b + i + 4);
        int64x2_t b3 = vld1q_s64(b + i + 6);

        uint64x2_t cmp0 = vcgeq_s64(a0, b0);
        uint64x2_t cmp1 = vcgeq_s64(a1, b1);
        uint64x2_t cmp2 = vcgeq_s64(a2, b2);
        uint64x2_t cmp3 = vcgeq_s64(a3, b3);

        vst1q_s64(result + i, vreinterpretq_s64_u64(cmp0));
        vst1q_s64(result + i + 2, vreinterpretq_s64_u64(cmp1));
        vst1q_s64(result + i + 4, vreinterpretq_s64_u64(cmp2));
        vst1q_s64(result + i + 6, vreinterpretq_s64_u64(cmp3));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        int64x2_t bv = vld1q_s64(b + i);
        uint64x2_t cmp = vcgeq_s64(av, bv);
        vst1q_s64(result + i, vreinterpretq_s64_u64(cmp));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] >= b[i]) ? -1L : 0L;
    }
}

// Less than int64: result[i] = (a[i] < b[i]) ? -1 : 0
void lt_i64_neon(long *a, long *b, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        int64x2_t b0 = vld1q_s64(b + i);
        int64x2_t b1 = vld1q_s64(b + i + 2);
        int64x2_t b2 = vld1q_s64(b + i + 4);
        int64x2_t b3 = vld1q_s64(b + i + 6);

        uint64x2_t cmp0 = vcltq_s64(a0, b0);
        uint64x2_t cmp1 = vcltq_s64(a1, b1);
        uint64x2_t cmp2 = vcltq_s64(a2, b2);
        uint64x2_t cmp3 = vcltq_s64(a3, b3);

        vst1q_s64(result + i, vreinterpretq_s64_u64(cmp0));
        vst1q_s64(result + i + 2, vreinterpretq_s64_u64(cmp1));
        vst1q_s64(result + i + 4, vreinterpretq_s64_u64(cmp2));
        vst1q_s64(result + i + 6, vreinterpretq_s64_u64(cmp3));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        int64x2_t bv = vld1q_s64(b + i);
        uint64x2_t cmp = vcltq_s64(av, bv);
        vst1q_s64(result + i, vreinterpretq_s64_u64(cmp));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] < b[i]) ? -1L : 0L;
    }
}

// Less or equal int64: result[i] = (a[i] <= b[i]) ? -1 : 0
void le_i64_neon(long *a, long *b, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        int64x2_t a0 = vld1q_s64(a + i);
        int64x2_t a1 = vld1q_s64(a + i + 2);
        int64x2_t a2 = vld1q_s64(a + i + 4);
        int64x2_t a3 = vld1q_s64(a + i + 6);

        int64x2_t b0 = vld1q_s64(b + i);
        int64x2_t b1 = vld1q_s64(b + i + 2);
        int64x2_t b2 = vld1q_s64(b + i + 4);
        int64x2_t b3 = vld1q_s64(b + i + 6);

        uint64x2_t cmp0 = vcleq_s64(a0, b0);
        uint64x2_t cmp1 = vcleq_s64(a1, b1);
        uint64x2_t cmp2 = vcleq_s64(a2, b2);
        uint64x2_t cmp3 = vcleq_s64(a3, b3);

        vst1q_s64(result + i, vreinterpretq_s64_u64(cmp0));
        vst1q_s64(result + i + 2, vreinterpretq_s64_u64(cmp1));
        vst1q_s64(result + i + 4, vreinterpretq_s64_u64(cmp2));
        vst1q_s64(result + i + 6, vreinterpretq_s64_u64(cmp3));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        int64x2_t av = vld1q_s64(a + i);
        int64x2_t bv = vld1q_s64(b + i);
        uint64x2_t cmp = vcleq_s64(av, bv);
        vst1q_s64(result + i, vreinterpretq_s64_u64(cmp));
    }

    // Scalar remainder
    for (; i < n; i++) {
        result[i] = (a[i] <= b[i]) ? -1L : 0L;
    }
}

// ============================================================================
// Int64 If-Then-Else
// ============================================================================

// If-then-else int64: result[i] = mask[i] ? yes[i] : no[i]
void ifthenelse_i64_neon(long *mask, long *yes, long *no, long *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 longs at a time
    for (; i + 7 < n; i += 8) {
        uint64x2_t m0 = vld1q_u64((uint64_t*)(mask + i));
        uint64x2_t m1 = vld1q_u64((uint64_t*)(mask + i + 2));
        uint64x2_t m2 = vld1q_u64((uint64_t*)(mask + i + 4));
        uint64x2_t m3 = vld1q_u64((uint64_t*)(mask + i + 6));

        int64x2_t y0 = vld1q_s64(yes + i);
        int64x2_t y1 = vld1q_s64(yes + i + 2);
        int64x2_t y2 = vld1q_s64(yes + i + 4);
        int64x2_t y3 = vld1q_s64(yes + i + 6);

        int64x2_t n0 = vld1q_s64(no + i);
        int64x2_t n1 = vld1q_s64(no + i + 2);
        int64x2_t n2 = vld1q_s64(no + i + 4);
        int64x2_t n3 = vld1q_s64(no + i + 6);

        vst1q_s64(result + i, vbslq_s64(m0, y0, n0));
        vst1q_s64(result + i + 2, vbslq_s64(m1, y1, n1));
        vst1q_s64(result + i + 4, vbslq_s64(m2, y2, n2));
        vst1q_s64(result + i + 6, vbslq_s64(m3, y3, n3));
    }

    // Process 2 longs at a time
    for (; i + 1 < n; i += 2) {
        uint64x2_t mv = vld1q_u64((uint64_t*)(mask + i));
        int64x2_t yv = vld1q_s64(yes + i);
        int64x2_t nv = vld1q_s64(no + i);
        vst1q_s64(result + i, vbslq_s64(mv, yv, nv));
    }

    // Scalar remainder
    for (; i < n; i++) {
        if (mask[i]) {
            result[i] = yes[i];
        }
        if (!mask[i]) {
            result[i] = no[i];
        }
    }
}

// ============================================================================
// Single-Vector Operations (for hwygen-generated code)
// ============================================================================

// Find first true in Int32x4 mask (single 128-bit vector)
// Returns index 0-3 of first non-zero lane, or -1 if all zero
void find_first_true_i32x4_neon(int *mask, long *result) {
    int32x4_t m = vld1q_s32(mask);
    uint32x4_t nonzero = vmvnq_u32(vceqq_s32(m, vdupq_n_s32(0)));

    int lane0 = vgetq_lane_s32(vreinterpretq_s32_u32(nonzero), 0);
    int lane1 = vgetq_lane_s32(vreinterpretq_s32_u32(nonzero), 1);
    int lane2 = vgetq_lane_s32(vreinterpretq_s32_u32(nonzero), 2);
    int lane3 = vgetq_lane_s32(vreinterpretq_s32_u32(nonzero), 3);

    *result = -1;
    if (lane3) {
        *result = 3;
    }
    if (lane2) {
        *result = 2;
    }
    if (lane1) {
        *result = 1;
    }
    if (lane0) {
        *result = 0;
    }
}

// Find first true in Int64x2 mask (single 128-bit vector)
void find_first_true_i64x2_neon(long *mask, long *result) {
    int64x2_t m = vld1q_s64(mask);
    uint64x2_t zero = vceqq_s64(m, vdupq_n_s64(0));
    int64x2_t nonzero = vreinterpretq_s64_u64(vmvnq_u8(vreinterpretq_u8_u64(zero)));

    long lane0 = vgetq_lane_s64(nonzero, 0);
    long lane1 = vgetq_lane_s64(nonzero, 1);

    *result = -1;
    if (lane1) {
        *result = 1;
    }
    if (lane0) {
        *result = 0;
    }
}

// Count true in Int32x4 mask (single 128-bit vector)
void count_true_i32x4_neon(int *mask, long *result) {
    int32x4_t m = vld1q_s32(mask);
    uint32x4_t nonzero = vmvnq_u32(vceqq_s32(m, vdupq_n_s32(0)));
    uint32x4_t ones = vshrq_n_u32(nonzero, 31);
    *result = vaddvq_u32(ones);
}

// Count true in Int64x2 mask (single 128-bit vector)
void count_true_i64x2_neon(long *mask, long *result) {
    int64x2_t m = vld1q_s64(mask);
    uint64x2_t zero = vceqq_s64(m, vdupq_n_s64(0));
    uint64x2_t nonzero = vmvnq_u8(vreinterpretq_u8_u64(zero));
    uint64x2_t ones = vshrq_n_u64(vreinterpretq_u64_u8(nonzero), 63);
    *result = vaddvq_u64(ones);
}

// Single-vector equality: compare 4 float32s
void eq_f32x4_neon(float *a, float *b, int *result) {
    float32x4_t va = vld1q_f32(a);
    float32x4_t vb = vld1q_f32(b);
    uint32x4_t cmp = vceqq_f32(va, vb);
    vst1q_s32(result, vreinterpretq_s32_u32(cmp));
}

// Single-vector equality: compare 4 int32s
void eq_i32x4_neon(int *a, int *b, int *result) {
    int32x4_t va = vld1q_s32(a);
    int32x4_t vb = vld1q_s32(b);
    uint32x4_t cmp = vceqq_s32(va, vb);
    vst1q_s32(result, vreinterpretq_s32_u32(cmp));
}

// Single-vector equality: compare 2 float64s
void eq_f64x2_neon(double *a, double *b, long *result) {
    float64x2_t va = vld1q_f64(a);
    float64x2_t vb = vld1q_f64(b);
    uint64x2_t cmp = vceqq_f64(va, vb);
    vst1q_s64(result, vreinterpretq_s64_u64(cmp));
}

// Single-vector equality: compare 2 int64s
void eq_i64x2_neon(long *a, long *b, long *result) {
    int64x2_t va = vld1q_s64(a);
    int64x2_t vb = vld1q_s64(b);
    uint64x2_t cmp = vceqq_s64(va, vb);
    vst1q_s64(result, vreinterpretq_s64_u64(cmp));
}

// ============================================================================
// Single-Vector Mask Utility Operations
// ============================================================================

// AllTrue for Int32x4: returns 1 if all lanes are non-zero, 0 otherwise
void all_true_i32x4_neon(int *mask, long *result) {
    int32x4_t m = vld1q_s32(mask);
    // Compare with zero: gives 0xFFFFFFFF where mask==0, 0x0 where mask!=0
    uint32x4_t is_zero = vceqq_s32(m, vdupq_n_s32(0));
    // If any lane was zero, vmaxvq will return 0xFFFFFFFF
    // AllTrue = no lane is zero = max of is_zero is 0
    *result = 0;
    if (vmaxvq_u32(is_zero) == 0) {
        *result = 1;
    }
}

// AllTrue for Int64x2: returns 1 if all lanes are non-zero, 0 otherwise
void all_true_i64x2_neon(long *mask, long *result) {
    int64x2_t m = vld1q_s64(mask);
    uint64x2_t is_zero = vceqq_s64(m, vdupq_n_s64(0));
    // Check both lanes - need to OR them together
    uint64x2_t combined = vorrq_u64(is_zero, vextq_u64(is_zero, is_zero, 1));
    long any_zero = vgetq_lane_u64(combined, 0);
    // AllTrue = no lane was zero
    *result = 0;
    if (any_zero == 0) {
        *result = 1;
    }
}

// AllFalse for Int32x4: returns 1 if all lanes are zero, 0 otherwise
void all_false_i32x4_neon(int *mask, long *result) {
    int32x4_t m = vld1q_s32(mask);
    // vmaxvq on unsigned interpretation: if max is 0, all are zero
    uint32x4_t um = vreinterpretq_u32_s32(m);
    *result = 0;
    if (vmaxvq_u32(um) == 0) {
        *result = 1;
    }
}

// AllFalse for Int64x2: returns 1 if all lanes are zero, 0 otherwise
void all_false_i64x2_neon(long *mask, long *result) {
    int64x2_t m = vld1q_s64(mask);
    // OR both lanes - if result is 0, all were zero
    int64x2_t combined = vorrq_s64(m, vextq_s64(m, m, 1));
    long any_set = vgetq_lane_s64(combined, 0);
    *result = 0;
    if (any_set == 0) {
        *result = 1;
    }
}

// FirstN for Int32x4: creates mask with first n lanes set to -1, rest 0
void firstn_i32x4_neon(long *count, int *result) {
    long n = *count;
    int32x4_t neg_one = vdupq_n_s32(-1);
    int32x4_t zero = vdupq_n_s32(0);

    // Write based on count using conditional lane stores
    // Default all to 0
    vst1q_s32(result, zero);

    // Set lanes based on count (no else - GOAT limitation)
    if (n >= 1) {
        vst1q_lane_s32(result + 0, neg_one, 0);
    }
    if (n >= 2) {
        vst1q_lane_s32(result + 1, neg_one, 0);
    }
    if (n >= 3) {
        vst1q_lane_s32(result + 2, neg_one, 0);
    }
    if (n >= 4) {
        vst1q_lane_s32(result + 3, neg_one, 0);
    }
}

// FirstN for Int64x2: creates mask with first n lanes set to -1, rest 0
void firstn_i64x2_neon(long *count, long *result) {
    long n = *count;
    int64x2_t neg_one = vdupq_n_s64(-1);
    int64x2_t zero = vdupq_n_s64(0);

    // Default all to 0
    vst1q_s64(result, zero);

    // Set lanes based on count
    if (n >= 1) {
        vst1q_lane_s64(result + 0, neg_one, 0);
    }
    if (n >= 2) {
        vst1q_lane_s64(result + 1, neg_one, 0);
    }
}

// ============================================================================
// Fused Find/Count Operations - entire slice in one call, no intermediate copies
// ============================================================================

// Find first element equal to target in float32 slice
// Returns index of first match, or -1 if not found
void find_equal_f32_neon(float *slice, long *len, float *target, long *result) {
    long n = *len;
    float tgt = *target;
    long i = 0;

    // Broadcast target to all lanes
    float32x4_t target_vec = vdupq_n_f32(tgt);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(slice + i);
        uint32x4_t cmp = vceqq_f32(v, target_vec);

        // Check if any lane matches (horizontal max of comparison result)
        uint32_t any_match = vmaxvq_u32(cmp);
        if (any_match) {
            // Find which lane matched (check from lane 0 to 3)
            uint32_t lane0 = vgetq_lane_u32(cmp, 0);
            uint32_t lane1 = vgetq_lane_u32(cmp, 1);
            uint32_t lane2 = vgetq_lane_u32(cmp, 2);
            if (lane0) {
                *result = i;
                return;
            }
            if (lane1) {
                *result = i + 1;
                return;
            }
            if (lane2) {
                *result = i + 2;
                return;
            }
            // Must be lane 3
            *result = i + 3;
            return;
        }
    }

    // Scalar remainder
    for (; i < n; i++) {
        if (slice[i] == tgt) {
            *result = i;
            return;
        }
    }

    *result = -1;
}

// Find first element equal to target in int32 slice
void find_equal_i32_neon(int *slice, long *len, int *target, long *result) {
    long n = *len;
    int tgt = *target;
    long i = 0;

    int32x4_t target_vec = vdupq_n_s32(tgt);

    for (; i + 3 < n; i += 4) {
        int32x4_t v = vld1q_s32(slice + i);
        uint32x4_t cmp = vceqq_s32(v, target_vec);

        uint32_t any_match = vmaxvq_u32(cmp);
        if (any_match) {
            uint32_t lane0 = vgetq_lane_u32(cmp, 0);
            uint32_t lane1 = vgetq_lane_u32(cmp, 1);
            uint32_t lane2 = vgetq_lane_u32(cmp, 2);
            if (lane0) {
                *result = i;
                return;
            }
            if (lane1) {
                *result = i + 1;
                return;
            }
            if (lane2) {
                *result = i + 2;
                return;
            }
            *result = i + 3;
            return;
        }
    }

    for (; i < n; i++) {
        if (slice[i] == tgt) {
            *result = i;
            return;
        }
    }

    *result = -1;
}

// Count elements equal to target in float32 slice
void count_equal_f32_neon(float *slice, long *len, float *target, long *result) {
    long n = *len;
    float tgt = *target;
    long i = 0;
    long count = 0;

    float32x4_t target_vec = vdupq_n_f32(tgt);
    uint32x4_t count_vec = vdupq_n_u32(0);

    // Process 4 floats at a time, accumulate counts
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(slice + i);
        uint32x4_t cmp = vceqq_f32(v, target_vec);
        // cmp has 0xFFFFFFFF for match, 0 for no match
        // Shift right by 31 to get 1 or 0
        uint32x4_t ones = vshrq_n_u32(cmp, 31);
        count_vec = vaddq_u32(count_vec, ones);
    }

    // Sum the count vector
    count = vaddvq_u32(count_vec);

    // Scalar remainder
    for (; i < n; i++) {
        if (slice[i] == tgt) {
            count++;
        }
    }

    *result = count;
}

// Count elements equal to target in int32 slice
void count_equal_i32_neon(int *slice, long *len, int *target, long *result) {
    long n = *len;
    int tgt = *target;
    long i = 0;
    long count = 0;

    int32x4_t target_vec = vdupq_n_s32(tgt);
    uint32x4_t count_vec = vdupq_n_u32(0);

    for (; i + 3 < n; i += 4) {
        int32x4_t v = vld1q_s32(slice + i);
        uint32x4_t cmp = vceqq_s32(v, target_vec);
        uint32x4_t ones = vshrq_n_u32(cmp, 31);
        count_vec = vaddq_u32(count_vec, ones);
    }

    count = vaddvq_u32(count_vec);

    for (; i < n; i++) {
        if (slice[i] == tgt) {
            count++;
        }
    }

    *result = count;
}

// Find first element equal to target in float64 slice
void find_equal_f64_neon(double *slice, long *len, double *target, long *result) {
    long n = *len;
    double tgt = *target;
    long i = 0;

    float64x2_t target_vec = vdupq_n_f64(tgt);

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t v = vld1q_f64(slice + i);
        uint64x2_t cmp = vceqq_f64(v, target_vec);

        // Check if any lane matches
        uint64_t lane0 = vgetq_lane_u64(cmp, 0);
        uint64_t lane1 = vgetq_lane_u64(cmp, 1);
        if (lane0) {
            *result = i;
            return;
        }
        if (lane1) {
            *result = i + 1;
            return;
        }
    }

    // Scalar remainder
    for (; i < n; i++) {
        if (slice[i] == tgt) {
            *result = i;
            return;
        }
    }

    *result = -1;
}

// Find first element equal to target in int64 slice
void find_equal_i64_neon(long *slice, long *len, long *target, long *result) {
    long n = *len;
    long tgt = *target;
    long i = 0;

    int64x2_t target_vec = vdupq_n_s64(tgt);

    for (; i + 1 < n; i += 2) {
        int64x2_t v = vld1q_s64(slice + i);
        uint64x2_t cmp = vceqq_s64(v, target_vec);

        uint64_t lane0 = vgetq_lane_u64(cmp, 0);
        uint64_t lane1 = vgetq_lane_u64(cmp, 1);
        if (lane0) {
            *result = i;
            return;
        }
        if (lane1) {
            *result = i + 1;
            return;
        }
    }

    for (; i < n; i++) {
        if (slice[i] == tgt) {
            *result = i;
            return;
        }
    }

    *result = -1;
}

// Count elements equal to target in float64 slice
void count_equal_f64_neon(double *slice, long *len, double *target, long *result) {
    long n = *len;
    double tgt = *target;
    long i = 0;
    long count = 0;

    float64x2_t target_vec = vdupq_n_f64(tgt);
    uint64x2_t count_vec = vdupq_n_u64(0);

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t v = vld1q_f64(slice + i);
        uint64x2_t cmp = vceqq_f64(v, target_vec);
        // cmp has 0xFFFFFFFFFFFFFFFF for match, 0 for no match
        // Shift right by 63 to get 1 or 0
        uint64x2_t ones = vshrq_n_u64(cmp, 63);
        count_vec = vaddq_u64(count_vec, ones);
    }

    // Sum the count vector
    count = vgetq_lane_u64(count_vec, 0) + vgetq_lane_u64(count_vec, 1);

    // Scalar remainder
    for (; i < n; i++) {
        if (slice[i] == tgt) {
            count++;
        }
    }

    *result = count;
}

// Count elements equal to target in int64 slice
void count_equal_i64_neon(long *slice, long *len, long *target, long *result) {
    long n = *len;
    long tgt = *target;
    long i = 0;
    long count = 0;

    int64x2_t target_vec = vdupq_n_s64(tgt);
    uint64x2_t count_vec = vdupq_n_u64(0);

    for (; i + 1 < n; i += 2) {
        int64x2_t v = vld1q_s64(slice + i);
        uint64x2_t cmp = vceqq_s64(v, target_vec);
        uint64x2_t ones = vshrq_n_u64(cmp, 63);
        count_vec = vaddq_u64(count_vec, ones);
    }

    count = vgetq_lane_u64(count_vec, 0) + vgetq_lane_u64(count_vec, 1);

    for (; i < n; i++) {
        if (slice[i] == tgt) {
            count++;
        }
    }

    *result = count;
}

// CompressKeys for Float32x4 using NEON TBL instruction.
// Takes input float32x4, a 16-byte permutation table entry, and stores result.
// The permutation table maps each 4-bit mask to byte shuffle indices for
// partition-style reordering (true elements first, false elements after).
// This is the key primitive for Highway's VQSort double-store partition trick.
void compress_keys_f32x4_neon(float *input, unsigned char *perm_entry, float *output) {
    // Load input vector as bytes (16 bytes = 4 floats)
    uint8x16_t input_bytes = vld1q_u8((unsigned char*)input);

    // Load the 16-byte permutation indices
    uint8x16_t perm = vld1q_u8(perm_entry);

    // Use TBL to permute bytes - single instruction!
    uint8x16_t result_bytes = vqtbl1q_u8(input_bytes, perm);

    // Store result
    vst1q_u8((unsigned char*)output, result_bytes);
}

// CompressKeys for Int32x4 using NEON TBL instruction.
// Same as float32x4 - both are 16 bytes with 4 lanes.
void compress_keys_i32x4_neon(long *input, unsigned char *perm_entry, long *output) {
    uint8x16_t input_bytes = vld1q_u8((unsigned char*)input);
    uint8x16_t perm = vld1q_u8(perm_entry);
    uint8x16_t result_bytes = vqtbl1q_u8(input_bytes, perm);
    vst1q_u8((unsigned char*)output, result_bytes);
}

// CompressKeys for Float64x2 using NEON TBL instruction.
// 2 lanes of 8 bytes each = 16 bytes total.
void compress_keys_f64x2_neon(double *input, unsigned char *perm_entry, double *output) {
    uint8x16_t input_bytes = vld1q_u8((unsigned char*)input);
    uint8x16_t perm = vld1q_u8(perm_entry);
    uint8x16_t result_bytes = vqtbl1q_u8(input_bytes, perm);
    vst1q_u8((unsigned char*)output, result_bytes);
}

// CompressKeys for Int64x2 using NEON TBL instruction.
// Same as float64x2 - both are 16 bytes with 2 lanes.
void compress_keys_i64x2_neon(long *input, unsigned char *perm_entry, long *output) {
    uint8x16_t input_bytes = vld1q_u8((unsigned char*)input);
    uint8x16_t perm = vld1q_u8(perm_entry);
    uint8x16_t result_bytes = vqtbl1q_u8(input_bytes, perm);
    vst1q_u8((unsigned char*)output, result_bytes);
}

