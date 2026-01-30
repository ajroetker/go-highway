/*
 * Copyright 2025 go-highway Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// GELU NEON implementation for ARM64
//
// Provides both exact and approximate GELU activation functions:
//   - Exact:  GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//   - Approx: GELU(x) = x * sigmoid(1.702 * x)
//
// All transcendental functions (exp, erf) are computed inline using
// NEON polynomial approximations matching the Go hwy BaseExpVec precision.
//
// NOTE: Range reduction uses separate vmulq+vsubq (not fused vfmsq) to match
// the Go hwy.Sub(x, hwy.Mul(k, ln2Hi)) code path's rounding behavior.
// The Horner polynomial uses vfmaq (FMA) since Go hwy.MulAdd also uses FMA.

#include <arm_neon.h>

// =============================================================================
// gelu_approx_neon_f32: Fast GELU approximation (f32)
// =============================================================================
// GELU_approx(x) = x * sigmoid(1.702 * x)
//                 = x / (1 + exp(-1.702 * x))
//
// func gelu_approx_neon_f32(input, output, psize unsafe.Pointer)
void gelu_approx_neon_f32(float *input, float *output, long *psize) {
    long size = *psize;
    if (size <= 0) return;

    // Exp constants (matching Go hwy constants from constants.go)
    float32x4_t invLn2 = vdupq_n_f32(1.44269504088896341f);
    float32x4_t ln2Hi  = vdupq_n_f32(0.693359375f);
    float32x4_t ln2Lo  = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t overflow  = vdupq_n_f32(88.72283905206835f);
    float32x4_t underflow = vdupq_n_f32(-87.33654475055310f);
    float32x4_t c1 = vdupq_n_f32(1.0f);
    float32x4_t c2 = vdupq_n_f32(0.5f);
    float32x4_t c3 = vdupq_n_f32(0.16666666666666666f);
    float32x4_t c4 = vdupq_n_f32(0.041666666666666664f);
    float32x4_t c5 = vdupq_n_f32(0.008333333333333333f);
    float32x4_t c6 = vdupq_n_f32(0.001388888888888889f);
    int32x4_t bias = vdupq_n_s32(127);
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t inf = vdupq_n_f32(1.0f / 0.0f);

    float32x4_t coeff = vdupq_n_f32(1.702f);

    long p = 0;
    for (; p + 4 <= size; p += 4) {
        float32x4_t x = vld1q_f32(input + p);

        // neg_sx = -(1.702 * x)
        float32x4_t neg_sx = vnegq_f32(vmulq_f32(coeff, x));

        // Inline exp(neg_sx)
        uint32x4_t over = vcgtq_f32(neg_sx, overflow);
        uint32x4_t under = vcltq_f32(neg_sx, underflow);

        float32x4_t kf = vrndnq_f32(vmulq_f32(neg_sx, invLn2));
        // Range reduction: separate mul+sub (matches Go hwy.Sub/hwy.Mul)
        float32x4_t r = vsubq_f32(neg_sx, vmulq_f32(kf, ln2Hi));
        r = vsubq_f32(r, vmulq_f32(kf, ln2Lo));

        float32x4_t ep = vfmaq_f32(c5, c6, r);
        ep = vfmaq_f32(c4, ep, r);
        ep = vfmaq_f32(c3, ep, r);
        ep = vfmaq_f32(c2, ep, r);
        ep = vfmaq_f32(c1, ep, r);
        ep = vfmaq_f32(c1, ep, r);

        int32x4_t ki = vcvtnq_s32_f32(kf);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, bias), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);
        float32x4_t exp_neg_sx = vmulq_f32(ep, scale);
        exp_neg_sx = vbslq_f32(over, inf, exp_neg_sx);
        exp_neg_sx = vbslq_f32(under, zero, exp_neg_sx);

        // sigmoid = 1 / (1 + exp(-1.702*x))
        float32x4_t sigmoid = vdivq_f32(c1, vaddq_f32(c1, exp_neg_sx));

        // result = x * sigmoid
        vst1q_f32(output + p, vmulq_f32(x, sigmoid));
    }

    // Scalar tail
    for (; p < size; p++) {
        float x = input[p];
        float neg_sx = -1.702f * x;

        float32x4_t xv = vdupq_n_f32(neg_sx);
        uint32x4_t over = vcgtq_f32(xv, overflow);
        uint32x4_t under = vcltq_f32(xv, underflow);

        float32x4_t kf = vrndnq_f32(vmulq_f32(xv, invLn2));
        float32x4_t r = vsubq_f32(xv, vmulq_f32(kf, ln2Hi));
        r = vsubq_f32(r, vmulq_f32(kf, ln2Lo));

        float32x4_t ep = vfmaq_f32(c5, c6, r);
        ep = vfmaq_f32(c4, ep, r);
        ep = vfmaq_f32(c3, ep, r);
        ep = vfmaq_f32(c2, ep, r);
        ep = vfmaq_f32(c1, ep, r);
        ep = vfmaq_f32(c1, ep, r);

        int32x4_t ki = vcvtnq_s32_f32(kf);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, bias), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);
        float32x4_t exp_val = vmulq_f32(ep, scale);
        exp_val = vbslq_f32(over, inf, exp_val);
        exp_val = vbslq_f32(under, zero, exp_val);

        float ev = vgetq_lane_f32(exp_val, 0);
        float sig = 1.0f / (1.0f + ev);
        output[p] = x * sig;
    }
}

// =============================================================================
// gelu_neon_f32: Exact GELU (f32)
// =============================================================================
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//
// Inline erf uses Abramowitz & Stegun 7.1.26 approximation
// (same as math_f32_neon_arm64.c:erf_f32_neon).
//
// func gelu_neon_f32(input, output, psize unsafe.Pointer)
void gelu_neon_f32(float *input, float *output, long *psize) {
    long size = *psize;
    if (size <= 0) return;

    // Constants
    float32x4_t v_half = vdupq_n_f32(0.5f);
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_inv_sqrt2 = vdupq_n_f32(0.7071067811865476f);

    // Abramowitz and Stegun erf constants
    float32x4_t v_p  = vdupq_n_f32(0.3275911f);
    float32x4_t v_a1 = vdupq_n_f32(0.254829592f);
    float32x4_t v_a2 = vdupq_n_f32(-0.284496736f);
    float32x4_t v_a3 = vdupq_n_f32(1.421413741f);
    float32x4_t v_a4 = vdupq_n_f32(-1.453152027f);
    float32x4_t v_a5 = vdupq_n_f32(1.061405429f);

    // Exp constants for exp(-x^2) — matching Go hwy constants
    float32x4_t v_ln2Hi = vdupq_n_f32(0.693359375f);
    float32x4_t v_ln2Lo = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t v_inv_ln2 = vdupq_n_f32(1.44269504088896341f);
    float32x4_t v_min_clamp = vdupq_n_f32(-88.0f);
    float32x4_t v_max_clamp = vdupq_n_f32(88.0f);

    long p = 0;
    for (; p + 4 <= size; p += 4) {
        float32x4_t x = vld1q_f32(input + p);

        // xs = x * invSqrt2
        float32x4_t xs = vmulq_f32(x, v_inv_sqrt2);

        // --- Inline erf(xs) ---
        // Get sign and absolute value
        uint32x4_t is_negative = vcltq_f32(xs, v_zero);
        float32x4_t abs_xs = vabsq_f32(xs);

        // t = 1 / (1 + p * |xs|)
        float32x4_t t = vdivq_f32(v_one, vfmaq_f32(v_one, v_p, abs_xs));

        // Compute -xs^2 for exp
        float32x4_t neg_xs2 = vnegq_f32(vmulq_f32(xs, xs));
        neg_xs2 = vmaxq_f32(neg_xs2, v_min_clamp);
        neg_xs2 = vminq_f32(neg_xs2, v_max_clamp);

        // Inline exp(-xs^2) — separate mul+sub for range reduction (Hi/Lo split)
        float32x4_t exp_k = vrndnq_f32(vmulq_f32(neg_xs2, v_inv_ln2));
        float32x4_t r = vsubq_f32(neg_xs2, vmulq_f32(exp_k, v_ln2Hi));
        r = vsubq_f32(r, vmulq_f32(exp_k, v_ln2Lo));

        float32x4_t exp_r = vdupq_n_f32(0.001388888888888889f);
        exp_r = vfmaq_f32(vdupq_n_f32(0.008333333333333333f), exp_r, r);
        exp_r = vfmaq_f32(vdupq_n_f32(0.041666666666666664f), exp_r, r);
        exp_r = vfmaq_f32(vdupq_n_f32(0.16666666666666666f), exp_r, r);
        exp_r = vfmaq_f32(vdupq_n_f32(0.5f), exp_r, r);
        exp_r = vfmaq_f32(v_one, exp_r, r);
        exp_r = vfmaq_f32(v_one, exp_r, r);

        int32x4_t ki = vcvtnq_s32_f32(exp_k);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);
        float32x4_t exp_neg_xs2 = vmulq_f32(exp_r, scale);

        // Polynomial: t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
        float32x4_t poly = v_a5;
        poly = vfmaq_f32(v_a4, poly, t);
        poly = vfmaq_f32(v_a3, poly, t);
        poly = vfmaq_f32(v_a2, poly, t);
        poly = vfmaq_f32(v_a1, poly, t);
        poly = vmulq_f32(poly, t);

        // erf = 1 - poly * exp(-xs^2) — separate mul+sub
        float32x4_t erf_abs = vsubq_f32(v_one, vmulq_f32(poly, exp_neg_xs2));

        // Apply sign
        float32x4_t erf_val = vbslq_f32(is_negative, vnegq_f32(erf_abs), erf_abs);

        // --- GELU = x * 0.5 * (1 + erf) ---
        float32x4_t one_plus_erf = vaddq_f32(v_one, erf_val);
        float32x4_t result = vmulq_f32(x, vmulq_f32(v_half, one_plus_erf));

        vst1q_f32(output + p, result);
    }

    // Scalar tail
    for (; p < size; p++) {
        float x = input[p];
        float xs = x * 0.7071067811865476f;

        float32x4_t xv = vdupq_n_f32(xs);
        uint32x4_t is_neg = vcltq_f32(xv, v_zero);
        float32x4_t abs_xv = vabsq_f32(xv);

        float32x4_t t = vdivq_f32(v_one, vfmaq_f32(v_one, v_p, abs_xv));

        float32x4_t neg_x2 = vnegq_f32(vmulq_f32(xv, xv));
        neg_x2 = vmaxq_f32(neg_x2, v_min_clamp);
        neg_x2 = vminq_f32(neg_x2, v_max_clamp);

        float32x4_t ek = vrndnq_f32(vmulq_f32(neg_x2, v_inv_ln2));
        float32x4_t r = vsubq_f32(neg_x2, vmulq_f32(ek, v_ln2Hi));
        r = vsubq_f32(r, vmulq_f32(ek, v_ln2Lo));

        float32x4_t er = vdupq_n_f32(0.001388888888888889f);
        er = vfmaq_f32(vdupq_n_f32(0.008333333333333333f), er, r);
        er = vfmaq_f32(vdupq_n_f32(0.041666666666666664f), er, r);
        er = vfmaq_f32(vdupq_n_f32(0.16666666666666666f), er, r);
        er = vfmaq_f32(vdupq_n_f32(0.5f), er, r);
        er = vfmaq_f32(v_one, er, r);
        er = vfmaq_f32(v_one, er, r);

        int32x4_t eki = vcvtnq_s32_f32(ek);
        int32x4_t sb = vshlq_n_s32(vaddq_s32(eki, vdupq_n_s32(127)), 23);
        float32x4_t sc = vreinterpretq_f32_s32(sb);
        float32x4_t enx2 = vmulq_f32(er, sc);

        float32x4_t poly = v_a5;
        poly = vfmaq_f32(v_a4, poly, t);
        poly = vfmaq_f32(v_a3, poly, t);
        poly = vfmaq_f32(v_a2, poly, t);
        poly = vfmaq_f32(v_a1, poly, t);
        poly = vmulq_f32(poly, t);

        float32x4_t erf_a = vsubq_f32(v_one, vmulq_f32(poly, enx2));
        float32x4_t erf_v = vbslq_f32(is_neg, vnegq_f32(erf_a), erf_a);

        float erf_s = vgetq_lane_f32(erf_v, 0);
        output[p] = x * 0.5f * (1.0f + erf_s);
    }
}

// =============================================================================
// gelu_approx_neon_f64: Fast GELU approximation (f64)
// =============================================================================
//
// func gelu_approx_neon_f64(input, output, psize unsafe.Pointer)
void gelu_approx_neon_f64(double *input, double *output, long *psize) {
    long size = *psize;
    if (size <= 0) return;

    // f64 Hi/Lo ln2 split constants (matching Go expLn2Hi_f64, expLn2Lo_f64)
    float64x2_t ln2Hi_f64 = vdupq_n_f64(0.6931471803691238);
    float64x2_t ln2Lo_f64 = vdupq_n_f64(1.9082149292705877e-10);
    float64x2_t v_inv_ln2 = vdupq_n_f64(1.4426950408889634);
    float64x2_t v_one = vdupq_n_f64(1.0);
    float64x2_t coeff = vdupq_n_f64(1.702);

    long p = 0;
    for (; p + 2 <= size; p += 2) {
        float64x2_t x = vld1q_f64(input + p);

        // neg_sx = -(1.702 * x)
        float64x2_t neg_sx = vnegq_f64(vmulq_f64(coeff, x));

        // Clamp
        neg_sx = vmaxq_f64(neg_sx, vdupq_n_f64(-709.0));
        neg_sx = vminq_f64(neg_sx, vdupq_n_f64(709.0));

        // Inline exp(neg_sx) for f64
        float64x2_t k = vrndnq_f64(vmulq_f64(neg_sx, v_inv_ln2));
        // Range reduction: separate mul+sub (matches Go)
        float64x2_t r = vsubq_f64(neg_sx, vmulq_f64(k, ln2Hi_f64));
        r = vsubq_f64(r, vmulq_f64(k, ln2Lo_f64));

        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t exp_neg_sx = vmulq_f64(exp_r, scale);

        // sigmoid = 1 / (1 + exp(-1.702*x))
        float64x2_t sigmoid = vdivq_f64(v_one, vaddq_f64(v_one, exp_neg_sx));

        // result = x * sigmoid
        vst1q_f64(output + p, vmulq_f64(x, sigmoid));
    }

    // Scalar tail
    for (; p < size; p++) {
        double x = input[p];
        double neg_sx = -1.702 * x;
        if (neg_sx < -709.0) neg_sx = -709.0;
        if (neg_sx > 709.0) neg_sx = 709.0;

        float64x2_t xv = vdupq_n_f64(neg_sx);
        float64x2_t k = vrndnq_f64(vmulq_f64(xv, v_inv_ln2));
        float64x2_t r = vsubq_f64(xv, vmulq_f64(k, ln2Hi_f64));
        r = vsubq_f64(r, vmulq_f64(k, ln2Lo_f64));

        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t ev = vmulq_f64(exp_r, scale);

        double exp_val = vgetq_lane_f64(ev, 0);
        double sig = 1.0 / (1.0 + exp_val);
        output[p] = x * sig;
    }
}

// =============================================================================
// gelu_neon_f64: Exact GELU (f64)
// =============================================================================
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//
// func gelu_neon_f64(input, output, psize unsafe.Pointer)
void gelu_neon_f64(double *input, double *output, long *psize) {
    long size = *psize;
    if (size <= 0) return;

    float64x2_t v_half = vdupq_n_f64(0.5);
    float64x2_t v_one = vdupq_n_f64(1.0);
    float64x2_t v_zero = vdupq_n_f64(0.0);
    float64x2_t v_inv_sqrt2 = vdupq_n_f64(0.7071067811865476);

    // Erf constants (Abramowitz & Stegun 7.1.26)
    float64x2_t v_p  = vdupq_n_f64(0.3275911);
    float64x2_t v_a1 = vdupq_n_f64(0.254829592);
    float64x2_t v_a2 = vdupq_n_f64(-0.284496736);
    float64x2_t v_a3 = vdupq_n_f64(1.421413741);
    float64x2_t v_a4 = vdupq_n_f64(-1.453152027);
    float64x2_t v_a5 = vdupq_n_f64(1.061405429);

    // Exp constants — Hi/Lo split for f64
    float64x2_t ln2Hi_f64 = vdupq_n_f64(0.6931471803691238);
    float64x2_t ln2Lo_f64 = vdupq_n_f64(1.9082149292705877e-10);
    float64x2_t v_inv_ln2 = vdupq_n_f64(1.4426950408889634);

    long p = 0;
    for (; p + 2 <= size; p += 2) {
        float64x2_t x = vld1q_f64(input + p);

        // xs = x * invSqrt2
        float64x2_t xs = vmulq_f64(x, v_inv_sqrt2);

        // --- Inline erf(xs) ---
        uint64x2_t is_negative = vcltq_f64(xs, v_zero);
        float64x2_t abs_xs = vabsq_f64(xs);

        // t = 1 / (1 + p * |xs|)
        float64x2_t t = vdivq_f64(v_one, vfmaq_f64(v_one, abs_xs, v_p));

        // exp(-xs^2)
        float64x2_t xs2 = vmulq_f64(abs_xs, abs_xs);
        float64x2_t neg_xs2 = vnegq_f64(xs2);
        neg_xs2 = vmaxq_f64(neg_xs2, vdupq_n_f64(-709.0));

        float64x2_t k = vrndnq_f64(vmulq_f64(neg_xs2, v_inv_ln2));
        // Range reduction: separate mul+sub (matches Go)
        float64x2_t r = vsubq_f64(neg_xs2, vmulq_f64(k, ln2Hi_f64));
        r = vsubq_f64(r, vmulq_f64(k, ln2Lo_f64));

        // Full 8-term exp polynomial for double precision
        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t exp_neg_xs2 = vmulq_f64(exp_r, scale);

        // Polynomial: t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
        float64x2_t poly = v_a5;
        poly = vfmaq_f64(v_a4, poly, t);
        poly = vfmaq_f64(v_a3, poly, t);
        poly = vfmaq_f64(v_a2, poly, t);
        poly = vfmaq_f64(v_a1, poly, t);
        poly = vmulq_f64(poly, t);

        // erf = 1 - poly * exp(-xs^2) — separate mul+sub
        float64x2_t erf_abs = vsubq_f64(v_one, vmulq_f64(poly, exp_neg_xs2));

        // Apply sign
        float64x2_t erf_val = vbslq_f64(is_negative, vnegq_f64(erf_abs), erf_abs);

        // GELU = x * 0.5 * (1 + erf)
        float64x2_t one_plus_erf = vaddq_f64(v_one, erf_val);
        float64x2_t result = vmulq_f64(x, vmulq_f64(v_half, one_plus_erf));

        vst1q_f64(output + p, result);
    }

    // Scalar tail
    for (; p < size; p++) {
        double x = input[p];
        double xs = x * 0.7071067811865476;

        float64x2_t xv = vdupq_n_f64(xs);
        uint64x2_t is_neg = vcltq_f64(xv, v_zero);
        float64x2_t abs_xv = vabsq_f64(xv);

        float64x2_t t = vdivq_f64(v_one, vfmaq_f64(v_one, abs_xv, v_p));

        float64x2_t xs2 = vmulq_f64(abs_xv, abs_xv);
        float64x2_t neg_xs2 = vnegq_f64(xs2);
        neg_xs2 = vmaxq_f64(neg_xs2, vdupq_n_f64(-709.0));

        float64x2_t k = vrndnq_f64(vmulq_f64(neg_xs2, v_inv_ln2));
        float64x2_t r = vsubq_f64(neg_xs2, vmulq_f64(k, ln2Hi_f64));
        r = vsubq_f64(r, vmulq_f64(k, ln2Lo_f64));

        float64x2_t er = vdupq_n_f64(2.48015873015873015873e-5);
        er = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), er, r);
        er = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), er, r);
        er = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), er, r);
        er = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), er, r);
        er = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), er, r);
        er = vfmaq_f64(vdupq_n_f64(0.5), er, r);
        er = vfmaq_f64(vdupq_n_f64(1.0), er, r);
        er = vfmaq_f64(vdupq_n_f64(1.0), er, r);

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t eb = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t sc = vreinterpretq_f64_s64(eb);
        float64x2_t enx2 = vmulq_f64(er, sc);

        float64x2_t poly = v_a5;
        poly = vfmaq_f64(v_a4, poly, t);
        poly = vfmaq_f64(v_a3, poly, t);
        poly = vfmaq_f64(v_a2, poly, t);
        poly = vfmaq_f64(v_a1, poly, t);
        poly = vmulq_f64(poly, t);

        float64x2_t erf_a = vsubq_f64(v_one, vmulq_f64(poly, enx2));
        float64x2_t erf_v = vbslq_f64(is_neg, vnegq_f64(erf_a), erf_a);

        double erf_s = vgetq_lane_f64(erf_v, 0);
        output[p] = x * 0.5 * (1.0 + erf_s);
    }
}

// =============================================================================
// silu_neon_f32: SiLU / Swish activation (f32)
// =============================================================================
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// func silu_neon_f32(input, output, psize unsafe.Pointer)
void silu_neon_f32(float *input, float *output, long *psize) {
    long size = *psize;
    if (size <= 0) return;

    // Exp constants (matching Go hwy constants from constants.go)
    float32x4_t invLn2 = vdupq_n_f32(1.44269504088896341f);
    float32x4_t ln2Hi  = vdupq_n_f32(0.693359375f);
    float32x4_t ln2Lo  = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t overflow  = vdupq_n_f32(88.72283905206835f);
    float32x4_t underflow = vdupq_n_f32(-87.33654475055310f);
    float32x4_t c1 = vdupq_n_f32(1.0f);
    float32x4_t c2 = vdupq_n_f32(0.5f);
    float32x4_t c3 = vdupq_n_f32(0.16666666666666666f);
    float32x4_t c4 = vdupq_n_f32(0.041666666666666664f);
    float32x4_t c5 = vdupq_n_f32(0.008333333333333333f);
    float32x4_t c6 = vdupq_n_f32(0.001388888888888889f);
    int32x4_t bias = vdupq_n_s32(127);
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t inf = vdupq_n_f32(1.0f / 0.0f);

    long p = 0;
    for (; p + 4 <= size; p += 4) {
        float32x4_t x = vld1q_f32(input + p);

        // neg_x = -x
        float32x4_t neg_x = vnegq_f32(x);

        // Inline exp(-x)
        uint32x4_t over = vcgtq_f32(neg_x, overflow);
        uint32x4_t under = vcltq_f32(neg_x, underflow);

        float32x4_t kf = vrndnq_f32(vmulq_f32(neg_x, invLn2));
        // Range reduction: separate mul+sub (matches Go hwy.Sub/hwy.Mul)
        float32x4_t r = vsubq_f32(neg_x, vmulq_f32(kf, ln2Hi));
        r = vsubq_f32(r, vmulq_f32(kf, ln2Lo));

        float32x4_t ep = vfmaq_f32(c5, c6, r);
        ep = vfmaq_f32(c4, ep, r);
        ep = vfmaq_f32(c3, ep, r);
        ep = vfmaq_f32(c2, ep, r);
        ep = vfmaq_f32(c1, ep, r);
        ep = vfmaq_f32(c1, ep, r);

        int32x4_t ki = vcvtnq_s32_f32(kf);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, bias), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);
        float32x4_t exp_neg_x = vmulq_f32(ep, scale);
        exp_neg_x = vbslq_f32(over, inf, exp_neg_x);
        exp_neg_x = vbslq_f32(under, zero, exp_neg_x);

        // sigmoid = 1 / (1 + exp(-x))
        float32x4_t sigmoid = vdivq_f32(c1, vaddq_f32(c1, exp_neg_x));

        // result = x * sigmoid
        vst1q_f32(output + p, vmulq_f32(x, sigmoid));
    }

    // Scalar tail
    for (; p < size; p++) {
        float x = input[p];
        float neg_x = -x;

        float32x4_t xv = vdupq_n_f32(neg_x);
        uint32x4_t over = vcgtq_f32(xv, overflow);
        uint32x4_t under = vcltq_f32(xv, underflow);

        float32x4_t kf = vrndnq_f32(vmulq_f32(xv, invLn2));
        float32x4_t r = vsubq_f32(xv, vmulq_f32(kf, ln2Hi));
        r = vsubq_f32(r, vmulq_f32(kf, ln2Lo));

        float32x4_t ep = vfmaq_f32(c5, c6, r);
        ep = vfmaq_f32(c4, ep, r);
        ep = vfmaq_f32(c3, ep, r);
        ep = vfmaq_f32(c2, ep, r);
        ep = vfmaq_f32(c1, ep, r);
        ep = vfmaq_f32(c1, ep, r);

        int32x4_t ki = vcvtnq_s32_f32(kf);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, bias), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);
        float32x4_t exp_val = vmulq_f32(ep, scale);
        exp_val = vbslq_f32(over, inf, exp_val);
        exp_val = vbslq_f32(under, zero, exp_val);

        float ev = vgetq_lane_f32(exp_val, 0);
        float sig = 1.0f / (1.0f + ev);
        output[p] = x * sig;
    }
}

// =============================================================================
// silu_neon_f64: SiLU / Swish activation (f64)
// =============================================================================
//
// func silu_neon_f64(input, output, psize unsafe.Pointer)
void silu_neon_f64(double *input, double *output, long *psize) {
    long size = *psize;
    if (size <= 0) return;

    float64x2_t ln2Hi_f64 = vdupq_n_f64(0.6931471803691238);
    float64x2_t ln2Lo_f64 = vdupq_n_f64(1.9082149292705877e-10);
    float64x2_t v_inv_ln2 = vdupq_n_f64(1.4426950408889634);
    float64x2_t v_one = vdupq_n_f64(1.0);

    long p = 0;
    for (; p + 2 <= size; p += 2) {
        float64x2_t x = vld1q_f64(input + p);

        // neg_x = -x
        float64x2_t neg_x = vnegq_f64(x);

        // Clamp
        neg_x = vmaxq_f64(neg_x, vdupq_n_f64(-709.0));
        neg_x = vminq_f64(neg_x, vdupq_n_f64(709.0));

        // Inline exp(-x) for f64
        float64x2_t k = vrndnq_f64(vmulq_f64(neg_x, v_inv_ln2));
        float64x2_t r = vsubq_f64(neg_x, vmulq_f64(k, ln2Hi_f64));
        r = vsubq_f64(r, vmulq_f64(k, ln2Lo_f64));

        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t exp_neg_x = vmulq_f64(exp_r, scale);

        // sigmoid = 1 / (1 + exp(-x))
        float64x2_t sigmoid = vdivq_f64(v_one, vaddq_f64(v_one, exp_neg_x));

        // result = x * sigmoid
        vst1q_f64(output + p, vmulq_f64(x, sigmoid));
    }

    // Scalar tail
    for (; p < size; p++) {
        double x = input[p];
        double neg_x = -x;
        if (neg_x < -709.0) neg_x = -709.0;
        if (neg_x > 709.0) neg_x = 709.0;

        float64x2_t xv = vdupq_n_f64(neg_x);
        float64x2_t k = vrndnq_f64(vmulq_f64(xv, v_inv_ln2));
        float64x2_t r = vsubq_f64(xv, vmulq_f64(k, ln2Hi_f64));
        r = vsubq_f64(r, vmulq_f64(k, ln2Lo_f64));

        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t ev = vmulq_f64(exp_r, scale);

        double exp_val = vgetq_lane_f64(ev, 0);
        double sig = 1.0 / (1.0 + exp_val);
        output[p] = x * sig;
    }
}

// =============================================================================
// tanh_neon_f32: Hyperbolic tangent activation (f32)
// =============================================================================
// tanh(x) = 2 * sigmoid(2x) - 1
//
// func tanh_neon_f32(input, output, psize unsafe.Pointer)
void tanh_neon_f32(float *input, float *output, long *psize) {
    long size = *psize;
    if (size <= 0) return;

    // Exp constants (matching Go hwy constants from constants.go)
    float32x4_t invLn2 = vdupq_n_f32(1.44269504088896341f);
    float32x4_t ln2Hi  = vdupq_n_f32(0.693359375f);
    float32x4_t ln2Lo  = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t overflow  = vdupq_n_f32(88.72283905206835f);
    float32x4_t underflow = vdupq_n_f32(-87.33654475055310f);
    float32x4_t c1 = vdupq_n_f32(1.0f);
    float32x4_t c2 = vdupq_n_f32(0.5f);
    float32x4_t c3 = vdupq_n_f32(0.16666666666666666f);
    float32x4_t c4 = vdupq_n_f32(0.041666666666666664f);
    float32x4_t c5 = vdupq_n_f32(0.008333333333333333f);
    float32x4_t c6 = vdupq_n_f32(0.001388888888888889f);
    int32x4_t bias = vdupq_n_s32(127);
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t inf = vdupq_n_f32(1.0f / 0.0f);
    float32x4_t two = vdupq_n_f32(2.0f);
    float32x4_t neg_one = vdupq_n_f32(-1.0f);
    float32x4_t pos_one = vdupq_n_f32(1.0f);

    long p = 0;
    for (; p + 4 <= size; p += 4) {
        float32x4_t x = vld1q_f32(input + p);

        // neg_2x = -(2 * x)
        float32x4_t neg_2x = vnegq_f32(vmulq_f32(two, x));

        // Inline exp(-2x)
        uint32x4_t over = vcgtq_f32(neg_2x, overflow);
        uint32x4_t under = vcltq_f32(neg_2x, underflow);

        float32x4_t kf = vrndnq_f32(vmulq_f32(neg_2x, invLn2));
        float32x4_t r = vsubq_f32(neg_2x, vmulq_f32(kf, ln2Hi));
        r = vsubq_f32(r, vmulq_f32(kf, ln2Lo));

        float32x4_t ep = vfmaq_f32(c5, c6, r);
        ep = vfmaq_f32(c4, ep, r);
        ep = vfmaq_f32(c3, ep, r);
        ep = vfmaq_f32(c2, ep, r);
        ep = vfmaq_f32(c1, ep, r);
        ep = vfmaq_f32(c1, ep, r);

        int32x4_t ki = vcvtnq_s32_f32(kf);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, bias), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);
        float32x4_t exp_neg_2x = vmulq_f32(ep, scale);
        exp_neg_2x = vbslq_f32(over, inf, exp_neg_2x);
        exp_neg_2x = vbslq_f32(under, zero, exp_neg_2x);

        // sigmoid_2x = 1 / (1 + exp(-2x))
        float32x4_t sigmoid_2x = vdivq_f32(c1, vaddq_f32(c1, exp_neg_2x));

        // tanh = 2 * sigmoid(2x) - 1
        float32x4_t result = vsubq_f32(vmulq_f32(two, sigmoid_2x), c1);

        // Clamp to [-1, 1] for large inputs
        result = vmaxq_f32(result, neg_one);
        result = vminq_f32(result, pos_one);

        vst1q_f32(output + p, result);
    }

    // Scalar tail
    for (; p < size; p++) {
        float x = input[p];
        float neg_2x = -2.0f * x;

        float32x4_t xv = vdupq_n_f32(neg_2x);
        uint32x4_t over = vcgtq_f32(xv, overflow);
        uint32x4_t under = vcltq_f32(xv, underflow);

        float32x4_t kf = vrndnq_f32(vmulq_f32(xv, invLn2));
        float32x4_t r = vsubq_f32(xv, vmulq_f32(kf, ln2Hi));
        r = vsubq_f32(r, vmulq_f32(kf, ln2Lo));

        float32x4_t ep = vfmaq_f32(c5, c6, r);
        ep = vfmaq_f32(c4, ep, r);
        ep = vfmaq_f32(c3, ep, r);
        ep = vfmaq_f32(c2, ep, r);
        ep = vfmaq_f32(c1, ep, r);
        ep = vfmaq_f32(c1, ep, r);

        int32x4_t ki = vcvtnq_s32_f32(kf);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, bias), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);
        float32x4_t exp_val = vmulq_f32(ep, scale);
        exp_val = vbslq_f32(over, inf, exp_val);
        exp_val = vbslq_f32(under, zero, exp_val);

        float ev = vgetq_lane_f32(exp_val, 0);
        float sig = 1.0f / (1.0f + ev);
        float res = 2.0f * sig - 1.0f;
        if (res < -1.0f) res = -1.0f;
        if (res > 1.0f) res = 1.0f;
        output[p] = res;
    }
}

// =============================================================================
// tanh_neon_f64: Hyperbolic tangent activation (f64)
// =============================================================================
//
// func tanh_neon_f64(input, output, psize unsafe.Pointer)
void tanh_neon_f64(double *input, double *output, long *psize) {
    long size = *psize;
    if (size <= 0) return;

    float64x2_t ln2Hi_f64 = vdupq_n_f64(0.6931471803691238);
    float64x2_t ln2Lo_f64 = vdupq_n_f64(1.9082149292705877e-10);
    float64x2_t v_inv_ln2 = vdupq_n_f64(1.4426950408889634);
    float64x2_t v_one = vdupq_n_f64(1.0);
    float64x2_t v_two = vdupq_n_f64(2.0);
    float64x2_t v_neg_one = vdupq_n_f64(-1.0);
    float64x2_t v_pos_one = vdupq_n_f64(1.0);

    long p = 0;
    for (; p + 2 <= size; p += 2) {
        float64x2_t x = vld1q_f64(input + p);

        // neg_2x = -(2 * x)
        float64x2_t neg_2x = vnegq_f64(vmulq_f64(v_two, x));

        // Clamp
        neg_2x = vmaxq_f64(neg_2x, vdupq_n_f64(-709.0));
        neg_2x = vminq_f64(neg_2x, vdupq_n_f64(709.0));

        // Inline exp(-2x)
        float64x2_t k = vrndnq_f64(vmulq_f64(neg_2x, v_inv_ln2));
        float64x2_t r = vsubq_f64(neg_2x, vmulq_f64(k, ln2Hi_f64));
        r = vsubq_f64(r, vmulq_f64(k, ln2Lo_f64));

        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t exp_neg_2x = vmulq_f64(exp_r, scale);

        // sigmoid_2x = 1 / (1 + exp(-2x))
        float64x2_t sigmoid_2x = vdivq_f64(v_one, vaddq_f64(v_one, exp_neg_2x));

        // tanh = 2 * sigmoid(2x) - 1
        float64x2_t result = vsubq_f64(vmulq_f64(v_two, sigmoid_2x), v_one);

        // Clamp to [-1, 1]
        result = vmaxq_f64(result, v_neg_one);
        result = vminq_f64(result, v_pos_one);

        vst1q_f64(output + p, result);
    }

    // Scalar tail
    for (; p < size; p++) {
        double x = input[p];
        double neg_2x = -2.0 * x;
        if (neg_2x < -709.0) neg_2x = -709.0;
        if (neg_2x > 709.0) neg_2x = 709.0;

        float64x2_t xv = vdupq_n_f64(neg_2x);
        float64x2_t k = vrndnq_f64(vmulq_f64(xv, v_inv_ln2));
        float64x2_t r = vsubq_f64(xv, vmulq_f64(k, ln2Hi_f64));
        r = vsubq_f64(r, vmulq_f64(k, ln2Lo_f64));

        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t ev = vmulq_f64(exp_r, scale);

        double exp_val = vgetq_lane_f64(ev, 0);
        double sig = 1.0 / (1.0 + exp_val);
        double res = 2.0 * sig - 1.0;
        if (res < -1.0) res = -1.0;
        if (res > 1.0) res = 1.0;
        output[p] = res;
    }
}

// =============================================================================
// elu_neon_f32: ELU activation (f32)
// =============================================================================
// ELU(x) = x           if x > 0
//        = alpha*(exp(x)-1) if x <= 0
//
// func elu_neon_f32(input, output, psize, palpha unsafe.Pointer)
void elu_neon_f32(float *input, float *output, long *psize, float *palpha) {
    long size = *psize;
    if (size <= 0) return;
    float alpha_val = *palpha;

    // Exp constants (matching Go hwy constants from constants.go)
    float32x4_t invLn2 = vdupq_n_f32(1.44269504088896341f);
    float32x4_t ln2Hi  = vdupq_n_f32(0.693359375f);
    float32x4_t ln2Lo  = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t overflow  = vdupq_n_f32(88.72283905206835f);
    float32x4_t underflow = vdupq_n_f32(-87.33654475055310f);
    float32x4_t c1 = vdupq_n_f32(1.0f);
    float32x4_t c2 = vdupq_n_f32(0.5f);
    float32x4_t c3 = vdupq_n_f32(0.16666666666666666f);
    float32x4_t c4 = vdupq_n_f32(0.041666666666666664f);
    float32x4_t c5 = vdupq_n_f32(0.008333333333333333f);
    float32x4_t c6 = vdupq_n_f32(0.001388888888888889f);
    int32x4_t bias = vdupq_n_s32(127);
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t inf = vdupq_n_f32(1.0f / 0.0f);
    float32x4_t alpha = vdupq_n_f32(alpha_val);

    long p = 0;
    for (; p + 4 <= size; p += 4) {
        float32x4_t x = vld1q_f32(input + p);

        // isPositive = x > 0
        uint32x4_t isPositive = vcgtq_f32(x, zero);

        // Inline exp(x) for negative branch
        uint32x4_t over = vcgtq_f32(x, overflow);
        uint32x4_t under = vcltq_f32(x, underflow);

        float32x4_t kf = vrndnq_f32(vmulq_f32(x, invLn2));
        float32x4_t r = vsubq_f32(x, vmulq_f32(kf, ln2Hi));
        r = vsubq_f32(r, vmulq_f32(kf, ln2Lo));

        float32x4_t ep = vfmaq_f32(c5, c6, r);
        ep = vfmaq_f32(c4, ep, r);
        ep = vfmaq_f32(c3, ep, r);
        ep = vfmaq_f32(c2, ep, r);
        ep = vfmaq_f32(c1, ep, r);
        ep = vfmaq_f32(c1, ep, r);

        int32x4_t ki = vcvtnq_s32_f32(kf);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, bias), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);
        float32x4_t exp_x = vmulq_f32(ep, scale);
        exp_x = vbslq_f32(over, inf, exp_x);
        exp_x = vbslq_f32(under, zero, exp_x);

        // negPart = alpha * (exp(x) - 1)
        float32x4_t negPart = vmulq_f32(alpha, vsubq_f32(exp_x, c1));

        // result = x if positive, negPart otherwise
        float32x4_t result = vbslq_f32(isPositive, x, negPart);

        vst1q_f32(output + p, result);
    }

    // Scalar tail
    for (; p < size; p++) {
        float x = input[p];
        if (x > 0.0f) {
            output[p] = x;
        }
        if (!(x > 0.0f)) {
            float32x4_t xv = vdupq_n_f32(x);
            uint32x4_t over = vcgtq_f32(xv, overflow);
            uint32x4_t under = vcltq_f32(xv, underflow);

            float32x4_t kf = vrndnq_f32(vmulq_f32(xv, invLn2));
            float32x4_t r = vsubq_f32(xv, vmulq_f32(kf, ln2Hi));
            r = vsubq_f32(r, vmulq_f32(kf, ln2Lo));

            float32x4_t ep = vfmaq_f32(c5, c6, r);
            ep = vfmaq_f32(c4, ep, r);
            ep = vfmaq_f32(c3, ep, r);
            ep = vfmaq_f32(c2, ep, r);
            ep = vfmaq_f32(c1, ep, r);
            ep = vfmaq_f32(c1, ep, r);

            int32x4_t ki = vcvtnq_s32_f32(kf);
            int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, bias), 23);
            float32x4_t scale = vreinterpretq_f32_s32(scale_bits);
            float32x4_t exp_val = vmulq_f32(ep, scale);
            exp_val = vbslq_f32(over, inf, exp_val);
            exp_val = vbslq_f32(under, zero, exp_val);

            float ev = vgetq_lane_f32(exp_val, 0);
            output[p] = alpha_val * (ev - 1.0f);
        }
    }
}

// =============================================================================
// elu_neon_f64: ELU activation (f64)
// =============================================================================
//
// func elu_neon_f64(input, output, psize, palpha unsafe.Pointer)
void elu_neon_f64(double *input, double *output, long *psize, double *palpha) {
    long size = *psize;
    if (size <= 0) return;
    double alpha_val = *palpha;

    float64x2_t ln2Hi_f64 = vdupq_n_f64(0.6931471803691238);
    float64x2_t ln2Lo_f64 = vdupq_n_f64(1.9082149292705877e-10);
    float64x2_t v_inv_ln2 = vdupq_n_f64(1.4426950408889634);
    float64x2_t v_one = vdupq_n_f64(1.0);
    float64x2_t v_zero = vdupq_n_f64(0.0);
    float64x2_t v_alpha = vdupq_n_f64(alpha_val);

    long p = 0;
    for (; p + 2 <= size; p += 2) {
        float64x2_t x = vld1q_f64(input + p);

        // isPositive = x > 0
        uint64x2_t isPositive = vcgtq_f64(x, v_zero);

        // Clamp x for exp (only matters for negative values)
        float64x2_t clamped = vmaxq_f64(x, vdupq_n_f64(-709.0));
        clamped = vminq_f64(clamped, vdupq_n_f64(709.0));

        // Inline exp(x)
        float64x2_t k = vrndnq_f64(vmulq_f64(clamped, v_inv_ln2));
        float64x2_t r = vsubq_f64(clamped, vmulq_f64(k, ln2Hi_f64));
        r = vsubq_f64(r, vmulq_f64(k, ln2Lo_f64));

        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t exp_x = vmulq_f64(exp_r, scale);

        // negPart = alpha * (exp(x) - 1)
        float64x2_t negPart = vmulq_f64(v_alpha, vsubq_f64(exp_x, v_one));

        // result = x if positive, negPart otherwise
        float64x2_t result = vbslq_f64(isPositive, x, negPart);

        vst1q_f64(output + p, result);
    }

    // Scalar tail
    for (; p < size; p++) {
        double x = input[p];
        if (x > 0.0) {
            output[p] = x;
        }
        if (!(x > 0.0)) {
            double cx = x;
            if (cx < -709.0) cx = -709.0;
            if (cx > 709.0) cx = 709.0;

            float64x2_t xv = vdupq_n_f64(cx);
            float64x2_t k = vrndnq_f64(vmulq_f64(xv, v_inv_ln2));
            float64x2_t r = vsubq_f64(xv, vmulq_f64(k, ln2Hi_f64));
            r = vsubq_f64(r, vmulq_f64(k, ln2Lo_f64));

            float64x2_t er = vdupq_n_f64(2.48015873015873015873e-5);
            er = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), er, r);
            er = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), er, r);
            er = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), er, r);
            er = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), er, r);
            er = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), er, r);
            er = vfmaq_f64(vdupq_n_f64(0.5), er, r);
            er = vfmaq_f64(vdupq_n_f64(1.0), er, r);
            er = vfmaq_f64(vdupq_n_f64(1.0), er, r);

            int64x2_t ki = vcvtq_s64_f64(k);
            int64x2_t eb = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
            float64x2_t sc = vreinterpretq_f64_s64(eb);
            float64x2_t ev = vmulq_f64(er, sc);

            double exp_val = vgetq_lane_f64(ev, 0);
            output[p] = alpha_val * (exp_val - 1.0);
        }
    }
}
