// NEON Vector Operations for go-highway
// Compile with: -march=armv8-a -O3
//
// Implements common vector operations using NEON SIMD instructions.
// These are full-function implementations that keep the entire loop in assembly,
// eliminating per-operation function call overhead.
//
// GoAT generates Go assembly from this file via:
//   go tool goat vec_neon_arm64.c -O3 --target arm64

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// =============================================================================
// sum_f32: Sum all elements in a float32 slice
// =============================================================================
// func sum_f32(v unsafe.Pointer, n int64) float32
void sum_f32(float *v, long *pn, float *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0.0f;
        return;
    }

    // 4 independent accumulators for better instruction-level parallelism
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    long i = 0;

    // Process 16 elements at a time with 4x unrolling
    long limit = n - (n % 4);
    for (; i + 16 <= limit; i += 16) {
        float32x4x4_t v4 = vld1q_f32_x4(v + i);

        sum0 = vaddq_f32(sum0, v4.val[0]);
        sum1 = vaddq_f32(sum1, v4.val[1]);
        sum2 = vaddq_f32(sum2, v4.val[2]);
        sum3 = vaddq_f32(sum3, v4.val[3]);
    }

    // Process remaining 4-element chunks
    for (; i + 4 <= n; i += 4) {
        float32x4_t vec = vld1q_f32(v + i);
        sum0 = vaddq_f32(sum0, vec);
    }

    // Combine accumulators and horizontal sum
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    float scalar_sum = vaddvq_f32(sum0);

    // Handle tail elements
    for (; i < n; i++) {
        scalar_sum += v[i];
    }

    *result = scalar_sum;
}

// =============================================================================
// sum_f64: Sum all elements in a float64 slice
// =============================================================================
// func sum_f64(v unsafe.Pointer, n int64) float64
void sum_f64(double *v, long *pn, double *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0.0;
        return;
    }

    float64x2_t sum = vdupq_n_f64(0.0);
    long i = 0;

    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2) {
        float64x2_t vec = vld1q_f64(v + i);
        sum = vaddq_f64(sum, vec);
    }

    // Horizontal sum
    double scalar_sum = vaddvq_f64(sum);

    // Handle tail elements
    for (; i < n; i++) {
        scalar_sum += v[i];
    }

    *result = scalar_sum;
}

// =============================================================================
// min_f32: Find minimum element in a float32 slice
// =============================================================================
// func min_f32(v unsafe.Pointer, n int64) float32
void min_f32(float *v, long *pn, float *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0.0f;
        return;
    }

    if (n < 4) {
        float min_val = v[0];
        for (long i = 1; i < n; i++) {
            if (v[i] < min_val) {
                min_val = v[i];
            }
        }
        *result = min_val;
        return;
    }

    float32x4_t min_vec = vld1q_f32(v);
    long i = 4;

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t vec = vld1q_f32(v + i);
        min_vec = vminq_f32(min_vec, vec);
    }

    // Horizontal min
    float min_val = vminvq_f32(min_vec);

    // Handle tail elements
    for (; i < n; i++) {
        if (v[i] < min_val) {
            min_val = v[i];
        }
    }

    *result = min_val;
}

// =============================================================================
// max_f32: Find maximum element in a float32 slice
// =============================================================================
// func max_f32(v unsafe.Pointer, n int64) float32
void max_f32(float *v, long *pn, float *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0.0f;
        return;
    }

    if (n < 4) {
        float max_val = v[0];
        for (long i = 1; i < n; i++) {
            if (v[i] > max_val) {
                max_val = v[i];
            }
        }
        *result = max_val;
        return;
    }

    float32x4_t max_vec = vld1q_f32(v);
    long i = 4;

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t vec = vld1q_f32(v + i);
        max_vec = vmaxq_f32(max_vec, vec);
    }

    // Horizontal max
    float max_val = vmaxvq_f32(max_vec);

    // Handle tail elements
    for (; i < n; i++) {
        if (v[i] > max_val) {
            max_val = v[i];
        }
    }

    *result = max_val;
}

// =============================================================================
// squared_norm_f32: Compute sum of squares (dot product with self)
// =============================================================================
// func squared_norm_f32(v unsafe.Pointer, n int64) float32
void squared_norm_f32(float *v, long *pn, float *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0.0f;
        return;
    }

    // 4 independent accumulators for better instruction-level parallelism
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    long i = 0;

    // Process 16 elements at a time with 4x unrolling
    long limit = n - (n % 4);
    for (; i + 16 <= limit; i += 16) {
        float32x4x4_t v4 = vld1q_f32_x4(v + i);

        sum0 = vfmaq_f32(sum0, v4.val[0], v4.val[0]);
        sum1 = vfmaq_f32(sum1, v4.val[1], v4.val[1]);
        sum2 = vfmaq_f32(sum2, v4.val[2], v4.val[2]);
        sum3 = vfmaq_f32(sum3, v4.val[3], v4.val[3]);
    }

    // Handle remaining 4-element chunks
    for (; i + 4 <= limit; i += 4) {
        float32x4_t vec = vld1q_f32(v + i);
        sum0 = vfmaq_f32(sum0, vec, vec);
    }

    // Combine accumulators and horizontal sum
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    float scalar_sum = vaddvq_f32(sum0);

    // Handle tail elements
    for (; i < n; i++) {
        scalar_sum += v[i] * v[i];
    }

    *result = scalar_sum;
}

// =============================================================================
// squared_norm_f64: Compute sum of squares for float64
// =============================================================================
// func squared_norm_f64(v unsafe.Pointer, n int64) float64
void squared_norm_f64(double *v, long *pn, double *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0.0;
        return;
    }

    float64x2_t sum = vdupq_n_f64(0.0);
    long i = 0;

    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2) {
        float64x2_t vec = vld1q_f64(v + i);
        sum = vfmaq_f64(sum, vec, vec);  // sum += vec * vec
    }

    // Horizontal sum
    double scalar_sum = vaddvq_f64(sum);

    // Handle tail elements
    for (; i < n; i++) {
        scalar_sum += v[i] * v[i];
    }

    *result = scalar_sum;
}

// =============================================================================
// l2_squared_distance_f32: Compute squared Euclidean distance
// =============================================================================
// func l2_squared_distance_f32(a, b unsafe.Pointer, n int64) float32
void l2_squared_distance_f32(float *a, float *b, long *pn, float *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0.0f;
        return;
    }

    // 4 independent accumulators for better instruction-level parallelism
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    long i = 0;

    // Process 16 elements at a time with 4x unrolling
    long limit = n - (n % 4);
    for (; i + 16 <= limit; i += 16) {
        float32x4x4_t va4 = vld1q_f32_x4(a + i);
        float32x4x4_t vb4 = vld1q_f32_x4(b + i);

        float32x4_t diff0 = vsubq_f32(va4.val[0], vb4.val[0]);
        float32x4_t diff1 = vsubq_f32(va4.val[1], vb4.val[1]);
        float32x4_t diff2 = vsubq_f32(va4.val[2], vb4.val[2]);
        float32x4_t diff3 = vsubq_f32(va4.val[3], vb4.val[3]);

        sum0 = vfmaq_f32(sum0, diff0, diff0);
        sum1 = vfmaq_f32(sum1, diff1, diff1);
        sum2 = vfmaq_f32(sum2, diff2, diff2);
        sum3 = vfmaq_f32(sum3, diff3, diff3);
    }

    // Handle remaining 4-element chunks
    for (; i + 4 <= limit; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum0 = vfmaq_f32(sum0, diff, diff);
    }

    // Combine accumulators and horizontal sum
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    float scalar_sum = vaddvq_f32(sum0);

    // Handle tail elements
    for (; i < n; i++) {
        float d = a[i] - b[i];
        scalar_sum += d * d;
    }

    *result = scalar_sum;
}

// =============================================================================
// l2_squared_distance_f64: Compute squared Euclidean distance for float64
// =============================================================================
// func l2_squared_distance_f64(a, b unsafe.Pointer, n int64) float64
void l2_squared_distance_f64(double *a, double *b, long *pn, double *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0.0;
        return;
    }

    float64x2_t sum = vdupq_n_f64(0.0);
    long i = 0;

    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        float64x2_t diff = vsubq_f64(va, vb);
        sum = vfmaq_f64(sum, diff, diff);  // sum += diff * diff
    }

    // Horizontal sum
    double scalar_sum = vaddvq_f64(sum);

    // Handle tail elements
    for (; i < n; i++) {
        double d = a[i] - b[i];
        scalar_sum += d * d;
    }

    *result = scalar_sum;
}

// =============================================================================
// dot_f32: Compute dot product of two float32 slices
// =============================================================================
// func dot_f32(a, b unsafe.Pointer, n int64) float32
void dot_f32(float *a, float *b, long *pn, float *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0.0f;
        return;
    }

    // 4 independent accumulators for better instruction-level parallelism
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    long i = 0;

    // Process 16 elements at a time with 4x unrolling
    long limit = n - (n % 4);
    for (; i + 16 <= limit; i += 16) {
        float32x4x4_t va4 = vld1q_f32_x4(a + i);
        float32x4x4_t vb4 = vld1q_f32_x4(b + i);

        sum0 = vfmaq_f32(sum0, va4.val[0], vb4.val[0]);
        sum1 = vfmaq_f32(sum1, va4.val[1], vb4.val[1]);
        sum2 = vfmaq_f32(sum2, va4.val[2], vb4.val[2]);
        sum3 = vfmaq_f32(sum3, va4.val[3], vb4.val[3]);
    }

    // Handle remaining 4-element chunks
    for (; i + 4 <= limit; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum0 = vfmaq_f32(sum0, va, vb);
    }

    // Combine accumulators and horizontal sum
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    float scalar_sum = vaddvq_f32(sum0);

    // Handle tail elements
    for (; i < n; i++) {
        scalar_sum += a[i] * b[i];
    }

    *result = scalar_sum;
}

// =============================================================================
// dot_f64: Compute dot product of two float64 slices
// =============================================================================
// func dot_f64(a, b unsafe.Pointer, n int64) float64
void dot_f64(double *a, double *b, long *pn, double *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0.0;
        return;
    }

    float64x2_t sum = vdupq_n_f64(0.0);
    long i = 0;

    // Process 2 elements at a time
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        sum = vfmaq_f64(sum, va, vb);  // sum += va * vb
    }

    // Horizontal sum
    double scalar_sum = vaddvq_f64(sum);

    // Handle tail elements
    for (; i < n; i++) {
        scalar_sum += a[i] * b[i];
    }

    *result = scalar_sum;
}

// =============================================================================
// add_slices_f32: Element-wise addition dst[i] = a[i] + b[i]
// =============================================================================
// func add_slices_f32(dst, a, b unsafe.Pointer, n int64)
void add_slices_f32(float *dst, float *a, float *b, long *pn) {
    long n = *pn;
    long i = 0;

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t result = vaddq_f32(va, vb);
        vst1q_f32(dst + i, result);
    }

    // Handle tail elements
    for (; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
}

// =============================================================================
// sub_slices_f32: Element-wise subtraction dst[i] = a[i] - b[i]
// =============================================================================
// func sub_slices_f32(dst, a, b unsafe.Pointer, n int64)
void sub_slices_f32(float *dst, float *a, float *b, long *pn) {
    long n = *pn;
    long i = 0;

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t result = vsubq_f32(va, vb);
        vst1q_f32(dst + i, result);
    }

    // Handle tail elements
    for (; i < n; i++) {
        dst[i] = a[i] - b[i];
    }
}

// =============================================================================
// mul_slices_f32: Element-wise multiplication dst[i] = a[i] * b[i]
// =============================================================================
// func mul_slices_f32(dst, a, b unsafe.Pointer, n int64)
void mul_slices_f32(float *dst, float *a, float *b, long *pn) {
    long n = *pn;
    long i = 0;

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t result = vmulq_f32(va, vb);
        vst1q_f32(dst + i, result);
    }

    // Handle tail elements
    for (; i < n; i++) {
        dst[i] = a[i] * b[i];
    }
}

// =============================================================================
// scale_f32: Scale all elements by constant dst[i] = c * src[i]
// =============================================================================
// func scale_f32(dst, src unsafe.Pointer, c float32, n int64)
void scale_f32(float *dst, float *src, float *pc, long *pn) {
    long n = *pn;
    float c = *pc;
    float32x4_t vc = vdupq_n_f32(c);
    long i = 0;

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t vsrc = vld1q_f32(src + i);
        float32x4_t result = vmulq_f32(vc, vsrc);
        vst1q_f32(dst + i, result);
    }

    // Handle tail elements
    for (; i < n; i++) {
        dst[i] = c * src[i];
    }
}

// =============================================================================
// add_const_f32: Add constant to all elements dst[i] += c
// =============================================================================
// func add_const_f32(dst unsafe.Pointer, c float32, n int64)
void add_const_f32(float *dst, float *pc, long *pn) {
    long n = *pn;
    float c = *pc;
    float32x4_t vc = vdupq_n_f32(c);
    long i = 0;

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t vdst = vld1q_f32(dst + i);
        float32x4_t result = vaddq_f32(vdst, vc);
        vst1q_f32(dst + i, result);
    }

    // Handle tail elements
    for (; i < n; i++) {
        dst[i] += c;
    }
}

// =============================================================================
// axpy_f32: dst[i] += a * x[i] (BLAS-like axpy operation)
// =============================================================================
// func axpy_f32(dst, x unsafe.Pointer, a float32, n int64)
void axpy_f32(float *dst, float *x, float *pa, long *pn) {
    long n = *pn;
    float a = *pa;
    float32x4_t va = vdupq_n_f32(a);
    long i = 0;

    // Process 4 elements at a time using FMA
    for (; i + 4 <= n; i += 4) {
        float32x4_t vdst = vld1q_f32(dst + i);
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t result = vfmaq_f32(vdst, va, vx);  // dst + a * x
        vst1q_f32(dst + i, result);
    }

    // Handle tail elements
    for (; i < n; i++) {
        dst[i] += a * x[i];
    }
}

// =============================================================================
// reduce_max_u32_neon: Find maximum value in uint32 slice
// =============================================================================
// Uses 4x loop unrolling for better throughput
// Note: src is passed as unsigned int* which maps to uint32 in Go
void reduce_max_u32_neon(unsigned int *src, long *pn, long *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0;
        return;
    }

    long i = 0;
    unsigned int max_val = 0;

    // Process 16 uint32s at a time (4 vectors of 4)
    if (n >= 16) {
        uint32x4_t max0 = vdupq_n_u32(0);
        uint32x4_t max1 = vdupq_n_u32(0);
        uint32x4_t max2 = vdupq_n_u32(0);
        uint32x4_t max3 = vdupq_n_u32(0);

        for (; i + 15 < n; i += 16) {
            max0 = vmaxq_u32(max0, vld1q_u32(src + i));
            max1 = vmaxq_u32(max1, vld1q_u32(src + i + 4));
            max2 = vmaxq_u32(max2, vld1q_u32(src + i + 8));
            max3 = vmaxq_u32(max3, vld1q_u32(src + i + 12));
        }

        // Reduce 4 vectors to 1
        max0 = vmaxq_u32(max0, max1);
        max2 = vmaxq_u32(max2, max3);
        max0 = vmaxq_u32(max0, max2);

        // Horizontal max
        max_val = vmaxvq_u32(max0);
    }

    // Process 4 at a time
    for (; i + 3 < n; i += 4) {
        uint32x4_t v = vld1q_u32(src + i);
        unsigned int v_max = vmaxvq_u32(v);
        if (v_max > max_val) {
            max_val = v_max;
        }
    }

    // Scalar remainder
    for (; i < n; i++) {
        if (src[i] > max_val) {
            max_val = src[i];
        }
    }

    *result = max_val;
}

// =============================================================================
// reduce_max_u64_neon: Find maximum value in uint64 slice
// =============================================================================
// NEON lacks vmaxq_u64, so uses compare+select pattern
// Note: src is passed as unsigned long* which maps to uint64 in Go
void reduce_max_u64_neon(unsigned long *src, long *pn, long *result) {
    long n = *pn;

    if (n <= 0) {
        *result = 0;
        return;
    }

    long i = 0;
    unsigned long max_val = 0;

    // Process 8 uint64s at a time (4 vectors of 2)
    if (n >= 8) {
        uint64x2_t max0 = vdupq_n_u64(0);
        uint64x2_t max1 = vdupq_n_u64(0);
        uint64x2_t max2 = vdupq_n_u64(0);
        uint64x2_t max3 = vdupq_n_u64(0);

        for (; i + 7 < n; i += 8) {
            // NEON doesn't have vmaxq_u64, so use comparison and select
            uint64x2_t v0 = vld1q_u64(src + i);
            uint64x2_t v1 = vld1q_u64(src + i + 2);
            uint64x2_t v2 = vld1q_u64(src + i + 4);
            uint64x2_t v3 = vld1q_u64(src + i + 6);

            // max = (a > b) ? a : b
            uint64x2_t gt0 = vcgtq_u64(v0, max0);
            uint64x2_t gt1 = vcgtq_u64(v1, max1);
            uint64x2_t gt2 = vcgtq_u64(v2, max2);
            uint64x2_t gt3 = vcgtq_u64(v3, max3);

            max0 = vbslq_u64(gt0, v0, max0);
            max1 = vbslq_u64(gt1, v1, max1);
            max2 = vbslq_u64(gt2, v2, max2);
            max3 = vbslq_u64(gt3, v3, max3);
        }

        // Reduce 4 vectors to 1
        uint64x2_t gt01 = vcgtq_u64(max0, max1);
        max0 = vbslq_u64(gt01, max0, max1);
        uint64x2_t gt23 = vcgtq_u64(max2, max3);
        max2 = vbslq_u64(gt23, max2, max3);
        uint64x2_t gt02 = vcgtq_u64(max0, max2);
        max0 = vbslq_u64(gt02, max0, max2);

        // Horizontal max (2 lanes)
        unsigned long lane0 = vgetq_lane_u64(max0, 0);
        unsigned long lane1 = vgetq_lane_u64(max0, 1);
        if (lane0 > lane1) {
            max_val = lane0;
        }
        if (lane1 >= lane0) {
            max_val = lane1;
        }
    }

    // Process 2 at a time
    for (; i + 1 < n; i += 2) {
        if (src[i] > max_val) {
            max_val = src[i];
        }
        if (src[i + 1] > max_val) {
            max_val = src[i + 1];
        }
    }

    // Scalar remainder
    for (; i < n; i++) {
        if (src[i] > max_val) {
            max_val = src[i];
        }
    }

    *result = max_val;
}
