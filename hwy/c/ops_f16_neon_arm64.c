// Float16 SIMD operations for ARM64 with FP16 extension (ARMv8.2-A+)
// Used with GOAT to generate Go assembly
// Compile with: -march=armv8.2-a+fp16
#include <arm_neon.h>

// ============================================================================
// Float16 Conversions
// ============================================================================

// Promote float16 to float32: result[i] = (float32)a[i]
void promote_f16_to_f32_neon(unsigned short *a, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 -> 32 float32 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t h0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t h1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t h2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t h3 = vld1q_f16((float16_t*)(a + i + 24));

        // Convert lower halves
        vst1q_f32(result + i, vcvt_f32_f16(vget_low_f16(h0)));
        vst1q_f32(result + i + 4, vcvt_f32_f16(vget_high_f16(h0)));
        vst1q_f32(result + i + 8, vcvt_f32_f16(vget_low_f16(h1)));
        vst1q_f32(result + i + 12, vcvt_f32_f16(vget_high_f16(h1)));
        vst1q_f32(result + i + 16, vcvt_f32_f16(vget_low_f16(h2)));
        vst1q_f32(result + i + 20, vcvt_f32_f16(vget_high_f16(h2)));
        vst1q_f32(result + i + 24, vcvt_f32_f16(vget_low_f16(h3)));
        vst1q_f32(result + i + 28, vcvt_f32_f16(vget_high_f16(h3)));
    }

    // Process 8 float16 -> 8 float32 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t h = vld1q_f16((float16_t*)(a + i));
        vst1q_f32(result + i, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(result + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }

    // Process 4 at a time
    for (; i + 3 < n; i += 4) {
        float16x4_t h = vld1_f16((float16_t*)(a + i));
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(result + i, f);
    }

    // Scalar remainder using NEON for single element
    for (; i < n; i++) {
        float16x4_t hv = vld1_dup_f16((float16_t*)(a + i));
        float32x4_t fv = vcvt_f32_f16(hv);
        vst1q_lane_f32(result + i, fv, 0);
    }
}

// Demote float32 to float16: result[i] = (float16)a[i]
void demote_f32_to_f16_neon(float *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float32 -> 32 float16 at a time (4 output vectors)
    for (; i + 31 < n; i += 32) {
        float32x4_t f0 = vld1q_f32(a + i);
        float32x4_t f1 = vld1q_f32(a + i + 4);
        float32x4_t f2 = vld1q_f32(a + i + 8);
        float32x4_t f3 = vld1q_f32(a + i + 12);
        float32x4_t f4 = vld1q_f32(a + i + 16);
        float32x4_t f5 = vld1q_f32(a + i + 20);
        float32x4_t f6 = vld1q_f32(a + i + 24);
        float32x4_t f7 = vld1q_f32(a + i + 28);

        float16x4_t h0 = vcvt_f16_f32(f0);
        float16x4_t h1 = vcvt_f16_f32(f1);
        float16x4_t h2 = vcvt_f16_f32(f2);
        float16x4_t h3 = vcvt_f16_f32(f3);
        float16x4_t h4 = vcvt_f16_f32(f4);
        float16x4_t h5 = vcvt_f16_f32(f5);
        float16x4_t h6 = vcvt_f16_f32(f6);
        float16x4_t h7 = vcvt_f16_f32(f7);

        vst1q_f16((float16_t*)(result + i), vcombine_f16(h0, h1));
        vst1q_f16((float16_t*)(result + i + 8), vcombine_f16(h2, h3));
        vst1q_f16((float16_t*)(result + i + 16), vcombine_f16(h4, h5));
        vst1q_f16((float16_t*)(result + i + 24), vcombine_f16(h6, h7));
    }

    // Process 8 float32 -> 8 float16 at a time
    for (; i + 7 < n; i += 8) {
        float32x4_t lo = vld1q_f32(a + i);
        float32x4_t hi = vld1q_f32(a + i + 4);
        float16x4_t h_lo = vcvt_f16_f32(lo);
        float16x4_t h_hi = vcvt_f16_f32(hi);
        float16x8_t h = vcombine_f16(h_lo, h_hi);
        vst1q_f16((float16_t*)(result + i), h);
    }

    // Process 4 at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t f = vld1q_f32(a + i);
        float16x4_t h = vcvt_f16_f32(f);
        vst1_f16((float16_t*)(result + i), h);
    }

    // Scalar remainder using NEON for single element
    for (; i < n; i++) {
        float32x4_t fv = vld1q_dup_f32(a + i);
        float16x4_t hv = vcvt_f16_f32(fv);
        vst1_lane_f16((float16_t*)(result + i), hv, 0);
    }
}

// ============================================================================
// Native Float16 Arithmetic (requires ARMv8.2-A+fp16)
// ============================================================================

// Vector addition: result[i] = a[i] + b[i]
void add_f16_neon(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t a0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t a1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t a2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t a3 = vld1q_f16((float16_t*)(a + i + 24));

        float16x8_t b0 = vld1q_f16((float16_t*)(b + i));
        float16x8_t b1 = vld1q_f16((float16_t*)(b + i + 8));
        float16x8_t b2 = vld1q_f16((float16_t*)(b + i + 16));
        float16x8_t b3 = vld1q_f16((float16_t*)(b + i + 24));

        vst1q_f16((float16_t*)(result + i), vaddq_f16(a0, b0));
        vst1q_f16((float16_t*)(result + i + 8), vaddq_f16(a1, b1));
        vst1q_f16((float16_t*)(result + i + 16), vaddq_f16(a2, b2));
        vst1q_f16((float16_t*)(result + i + 24), vaddq_f16(a3, b3));
    }

    // Process 8 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        float16x8_t bv = vld1q_f16((float16_t*)(b + i));
        vst1q_f16((float16_t*)(result + i), vaddq_f16(av, bv));
    }

    // Scalar remainder using NEON single-lane
    for (; i < n; i++) {
        float16x4_t av = vld1_dup_f16((float16_t*)(a + i));
        float16x4_t bv = vld1_dup_f16((float16_t*)(b + i));
        float16x4_t rv = vadd_f16(av, bv);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}

// Vector subtraction: result[i] = a[i] - b[i]
void sub_f16_neon(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t a0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t a1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t a2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t a3 = vld1q_f16((float16_t*)(a + i + 24));

        float16x8_t b0 = vld1q_f16((float16_t*)(b + i));
        float16x8_t b1 = vld1q_f16((float16_t*)(b + i + 8));
        float16x8_t b2 = vld1q_f16((float16_t*)(b + i + 16));
        float16x8_t b3 = vld1q_f16((float16_t*)(b + i + 24));

        vst1q_f16((float16_t*)(result + i), vsubq_f16(a0, b0));
        vst1q_f16((float16_t*)(result + i + 8), vsubq_f16(a1, b1));
        vst1q_f16((float16_t*)(result + i + 16), vsubq_f16(a2, b2));
        vst1q_f16((float16_t*)(result + i + 24), vsubq_f16(a3, b3));
    }

    // Process 8 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        float16x8_t bv = vld1q_f16((float16_t*)(b + i));
        vst1q_f16((float16_t*)(result + i), vsubq_f16(av, bv));
    }

    // Scalar remainder using NEON single-lane
    for (; i < n; i++) {
        float16x4_t av = vld1_dup_f16((float16_t*)(a + i));
        float16x4_t bv = vld1_dup_f16((float16_t*)(b + i));
        float16x4_t rv = vsub_f16(av, bv);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}

// Vector multiplication: result[i] = a[i] * b[i]
void mul_f16_neon(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t a0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t a1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t a2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t a3 = vld1q_f16((float16_t*)(a + i + 24));

        float16x8_t b0 = vld1q_f16((float16_t*)(b + i));
        float16x8_t b1 = vld1q_f16((float16_t*)(b + i + 8));
        float16x8_t b2 = vld1q_f16((float16_t*)(b + i + 16));
        float16x8_t b3 = vld1q_f16((float16_t*)(b + i + 24));

        vst1q_f16((float16_t*)(result + i), vmulq_f16(a0, b0));
        vst1q_f16((float16_t*)(result + i + 8), vmulq_f16(a1, b1));
        vst1q_f16((float16_t*)(result + i + 16), vmulq_f16(a2, b2));
        vst1q_f16((float16_t*)(result + i + 24), vmulq_f16(a3, b3));
    }

    // Process 8 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        float16x8_t bv = vld1q_f16((float16_t*)(b + i));
        vst1q_f16((float16_t*)(result + i), vmulq_f16(av, bv));
    }

    // Scalar remainder using NEON single-lane
    for (; i < n; i++) {
        float16x4_t av = vld1_dup_f16((float16_t*)(a + i));
        float16x4_t bv = vld1_dup_f16((float16_t*)(b + i));
        float16x4_t rv = vmul_f16(av, bv);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}

// Vector division: result[i] = a[i] / b[i]
void div_f16_neon(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t a0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t a1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t a2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t a3 = vld1q_f16((float16_t*)(a + i + 24));

        float16x8_t b0 = vld1q_f16((float16_t*)(b + i));
        float16x8_t b1 = vld1q_f16((float16_t*)(b + i + 8));
        float16x8_t b2 = vld1q_f16((float16_t*)(b + i + 16));
        float16x8_t b3 = vld1q_f16((float16_t*)(b + i + 24));

        vst1q_f16((float16_t*)(result + i), vdivq_f16(a0, b0));
        vst1q_f16((float16_t*)(result + i + 8), vdivq_f16(a1, b1));
        vst1q_f16((float16_t*)(result + i + 16), vdivq_f16(a2, b2));
        vst1q_f16((float16_t*)(result + i + 24), vdivq_f16(a3, b3));
    }

    // Process 8 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        float16x8_t bv = vld1q_f16((float16_t*)(b + i));
        vst1q_f16((float16_t*)(result + i), vdivq_f16(av, bv));
    }

    // Scalar remainder using NEON single-lane
    for (; i < n; i++) {
        float16x4_t av = vld1_dup_f16((float16_t*)(a + i));
        float16x4_t bv = vld1_dup_f16((float16_t*)(b + i));
        float16x4_t rv = vdiv_f16(av, bv);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}

// Fused multiply-add: result[i] = a[i] * b[i] + c[i]
void fma_f16_neon(unsigned short *a, unsigned short *b, unsigned short *c, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t a0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t a1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t a2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t a3 = vld1q_f16((float16_t*)(a + i + 24));

        float16x8_t b0 = vld1q_f16((float16_t*)(b + i));
        float16x8_t b1 = vld1q_f16((float16_t*)(b + i + 8));
        float16x8_t b2 = vld1q_f16((float16_t*)(b + i + 16));
        float16x8_t b3 = vld1q_f16((float16_t*)(b + i + 24));

        float16x8_t c0 = vld1q_f16((float16_t*)(c + i));
        float16x8_t c1 = vld1q_f16((float16_t*)(c + i + 8));
        float16x8_t c2 = vld1q_f16((float16_t*)(c + i + 16));
        float16x8_t c3 = vld1q_f16((float16_t*)(c + i + 24));

        // vfmaq_f16(c, a, b) = a*b + c
        vst1q_f16((float16_t*)(result + i), vfmaq_f16(c0, a0, b0));
        vst1q_f16((float16_t*)(result + i + 8), vfmaq_f16(c1, a1, b1));
        vst1q_f16((float16_t*)(result + i + 16), vfmaq_f16(c2, a2, b2));
        vst1q_f16((float16_t*)(result + i + 24), vfmaq_f16(c3, a3, b3));
    }

    // Process 8 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        float16x8_t bv = vld1q_f16((float16_t*)(b + i));
        float16x8_t cv = vld1q_f16((float16_t*)(c + i));
        vst1q_f16((float16_t*)(result + i), vfmaq_f16(cv, av, bv));
    }

    // Scalar remainder using NEON single-lane
    for (; i < n; i++) {
        float16x4_t av = vld1_dup_f16((float16_t*)(a + i));
        float16x4_t bv = vld1_dup_f16((float16_t*)(b + i));
        float16x4_t cv = vld1_dup_f16((float16_t*)(c + i));
        float16x4_t rv = vfma_f16(cv, av, bv);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}

// Negation: result[i] = -a[i]
void neg_f16_neon(unsigned short *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t a0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t a1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t a2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t a3 = vld1q_f16((float16_t*)(a + i + 24));

        vst1q_f16((float16_t*)(result + i), vnegq_f16(a0));
        vst1q_f16((float16_t*)(result + i + 8), vnegq_f16(a1));
        vst1q_f16((float16_t*)(result + i + 16), vnegq_f16(a2));
        vst1q_f16((float16_t*)(result + i + 24), vnegq_f16(a3));
    }

    // Process 8 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        vst1q_f16((float16_t*)(result + i), vnegq_f16(av));
    }

    // Scalar remainder using NEON single-lane
    for (; i < n; i++) {
        float16x4_t av = vld1_dup_f16((float16_t*)(a + i));
        float16x4_t rv = vneg_f16(av);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}

// Absolute value: result[i] = abs(a[i])
void abs_f16_neon(unsigned short *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t a0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t a1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t a2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t a3 = vld1q_f16((float16_t*)(a + i + 24));

        vst1q_f16((float16_t*)(result + i), vabsq_f16(a0));
        vst1q_f16((float16_t*)(result + i + 8), vabsq_f16(a1));
        vst1q_f16((float16_t*)(result + i + 16), vabsq_f16(a2));
        vst1q_f16((float16_t*)(result + i + 24), vabsq_f16(a3));
    }

    // Process 8 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        vst1q_f16((float16_t*)(result + i), vabsq_f16(av));
    }

    // Scalar remainder using NEON single-lane
    for (; i < n; i++) {
        float16x4_t av = vld1_dup_f16((float16_t*)(a + i));
        float16x4_t rv = vabs_f16(av);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}

// Vector minimum: result[i] = min(a[i], b[i])
void min_f16_neon(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t a0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t a1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t a2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t a3 = vld1q_f16((float16_t*)(a + i + 24));

        float16x8_t b0 = vld1q_f16((float16_t*)(b + i));
        float16x8_t b1 = vld1q_f16((float16_t*)(b + i + 8));
        float16x8_t b2 = vld1q_f16((float16_t*)(b + i + 16));
        float16x8_t b3 = vld1q_f16((float16_t*)(b + i + 24));

        vst1q_f16((float16_t*)(result + i), vminq_f16(a0, b0));
        vst1q_f16((float16_t*)(result + i + 8), vminq_f16(a1, b1));
        vst1q_f16((float16_t*)(result + i + 16), vminq_f16(a2, b2));
        vst1q_f16((float16_t*)(result + i + 24), vminq_f16(a3, b3));
    }

    // Process 8 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        float16x8_t bv = vld1q_f16((float16_t*)(b + i));
        vst1q_f16((float16_t*)(result + i), vminq_f16(av, bv));
    }

    // Scalar remainder using NEON single-lane
    for (; i < n; i++) {
        float16x4_t av = vld1_dup_f16((float16_t*)(a + i));
        float16x4_t bv = vld1_dup_f16((float16_t*)(b + i));
        float16x4_t rv = vmin_f16(av, bv);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}

// Vector maximum: result[i] = max(a[i], b[i])
void max_f16_neon(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t a0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t a1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t a2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t a3 = vld1q_f16((float16_t*)(a + i + 24));

        float16x8_t b0 = vld1q_f16((float16_t*)(b + i));
        float16x8_t b1 = vld1q_f16((float16_t*)(b + i + 8));
        float16x8_t b2 = vld1q_f16((float16_t*)(b + i + 16));
        float16x8_t b3 = vld1q_f16((float16_t*)(b + i + 24));

        vst1q_f16((float16_t*)(result + i), vmaxq_f16(a0, b0));
        vst1q_f16((float16_t*)(result + i + 8), vmaxq_f16(a1, b1));
        vst1q_f16((float16_t*)(result + i + 16), vmaxq_f16(a2, b2));
        vst1q_f16((float16_t*)(result + i + 24), vmaxq_f16(a3, b3));
    }

    // Process 8 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        float16x8_t bv = vld1q_f16((float16_t*)(b + i));
        vst1q_f16((float16_t*)(result + i), vmaxq_f16(av, bv));
    }

    // Scalar remainder using NEON single-lane
    for (; i < n; i++) {
        float16x4_t av = vld1_dup_f16((float16_t*)(a + i));
        float16x4_t bv = vld1_dup_f16((float16_t*)(b + i));
        float16x4_t rv = vmax_f16(av, bv);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}

// Square root: result[i] = sqrt(a[i])
void sqrt_f16_neon(unsigned short *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t a0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t a1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t a2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t a3 = vld1q_f16((float16_t*)(a + i + 24));

        vst1q_f16((float16_t*)(result + i), vsqrtq_f16(a0));
        vst1q_f16((float16_t*)(result + i + 8), vsqrtq_f16(a1));
        vst1q_f16((float16_t*)(result + i + 16), vsqrtq_f16(a2));
        vst1q_f16((float16_t*)(result + i + 24), vsqrtq_f16(a3));
    }

    // Process 8 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        vst1q_f16((float16_t*)(result + i), vsqrtq_f16(av));
    }

    // Scalar remainder using promote-compute-demote
    for (; i < n; i++) {
        float16x4_t hv = vld1_dup_f16((float16_t*)(a + i));
        float32x4_t fv = vcvt_f32_f16(hv);
        float32x4_t rv32 = vsqrtq_f32(fv);
        float16x4_t rv = vcvt_f16_f32(rv32);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}
