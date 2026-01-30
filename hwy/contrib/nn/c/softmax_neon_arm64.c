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

// Softmax NEON implementation for ARM64
//
// Three-pass fused SIMD algorithm:
//   1. Find max (NEON vmaxq + vmaxvq horizontal reduction)
//   2. Subtract max + exp (fused into one pass, saves memory round-trip)
//   3. Normalize by 1/sum
//
// Key win over Go base: fuses subtract-max + exp into one pass,
// avoiding separate shifted[] allocation and BaseApply call.

#include <arm_neon.h>

// =============================================================================
// softmax_neon_f32: Softmax for float32
// =============================================================================
//
// func softmax_neon_f32(input, output, psize unsafe.Pointer)
void softmax_neon_f32(float *input, float *output, long *psize) {
    long size = *psize;
    if (size <= 0) return;

    // =========================================================================
    // Pass 1: Find maximum value using NEON
    // =========================================================================
    float32x4_t maxVec = vdupq_n_f32(input[0]);
    long p = 0;
    for (; p + 4 <= size; p += 4) {
        float32x4_t x = vld1q_f32(input + p);
        maxVec = vmaxq_f32(maxVec, x);
    }
    float maxVal = vmaxvq_f32(maxVec);
    for (; p < size; p++) {
        if (input[p] > maxVal) {
            maxVal = input[p];
        }
    }

    // =========================================================================
    // Pass 2: Fused subtract-max + exp + accumulate sum
    // =========================================================================
    // Inline exp polynomial (Taylor series, same as math_f32_neon_arm64.c)
    //
    // Range reduction: exp(x) = 2^k * exp(r)
    //   k = round(x * invLn2)
    //   r = x - k * ln2Hi - k * ln2Lo
    //   exp(r) via Horner: 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
    //   result = exp(r) * 2^k  (via IEEE bit manipulation)
    //
    float32x4_t invLn2 = vdupq_n_f32(1.44269504088896341f);
    float32x4_t ln2Hi  = vdupq_n_f32(0.693359375f);
    float32x4_t ln2Lo  = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t c1 = vdupq_n_f32(1.0f);
    float32x4_t c2 = vdupq_n_f32(0.5f);
    float32x4_t c3 = vdupq_n_f32(0.16666666666666666f);
    float32x4_t c4 = vdupq_n_f32(0.041666666666666664f);
    float32x4_t c5 = vdupq_n_f32(0.008333333333333333f);
    float32x4_t c6 = vdupq_n_f32(0.001388888888888889f);
    int32x4_t bias = vdupq_n_s32(127);
    float32x4_t expMin = vdupq_n_f32(-87.3365f);

    float32x4_t maxBroadcast = vdupq_n_f32(maxVal);
    float32x4_t sumVec = vdupq_n_f32(0.0f);
    float sumScalar = 0.0f;

    p = 0;
    for (; p + 4 <= size; p += 4) {
        // Subtract max
        float32x4_t x = vsubq_f32(vld1q_f32(input + p), maxBroadcast);
        x = vmaxq_f32(x, expMin);

        float32x4_t kf = vrndnq_f32(vmulq_f32(x, invLn2));
        // Range reduction using separate mul+sub (matches Go hwy.Sub/hwy.Mul)
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
        float32x4_t result = vmulq_f32(ep, scale);

        vst1q_f32(output + p, result);
        sumVec = vaddq_f32(sumVec, result);
    }
    float expSum = vaddvq_f32(sumVec) + sumScalar;

    // Scalar tail
    for (; p < size; p++) {
        float x = input[p] - maxVal;
        if (x < -87.3365f) x = -87.3365f;

        // Scalar exp using NEON for single element
        float32x4_t xv = vdupq_n_f32(x);
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
        float32x4_t result = vmulq_f32(ep, scale);

        float val = vgetq_lane_f32(result, 0);
        output[p] = val;
        expSum += val;
    }

    // =========================================================================
    // Pass 3: Normalize by 1/sum
    // =========================================================================
    float invSum = 1.0f / expSum;
    float32x4_t invSumVec = vdupq_n_f32(invSum);
    p = 0;
    for (; p + 4 <= size; p += 4) {
        float32x4_t v = vld1q_f32(output + p);
        vst1q_f32(output + p, vmulq_f32(v, invSumVec));
    }
    for (; p < size; p++) {
        output[p] *= invSum;
    }
}

// =============================================================================
// softmax_neon_f64: Softmax for float64
// =============================================================================
//
// func softmax_neon_f64(input, output, psize unsafe.Pointer)
void softmax_neon_f64(double *input, double *output, long *psize) {
    long size = *psize;
    if (size <= 0) return;

    // =========================================================================
    // Pass 1: Find maximum value
    // =========================================================================
    float64x2_t maxVec = vdupq_n_f64(input[0]);
    long p = 0;
    for (; p + 2 <= size; p += 2) {
        float64x2_t x = vld1q_f64(input + p);
        maxVec = vmaxq_f64(maxVec, x);
    }
    double maxVal = vmaxvq_f64(maxVec);
    for (; p < size; p++) {
        if (input[p] > maxVal) {
            maxVal = input[p];
        }
    }

    // =========================================================================
    // Pass 2: Fused subtract-max + exp + accumulate sum
    // =========================================================================
    // f64 Hi/Lo ln2 split constants (matching Go expLn2Hi_f64, expLn2Lo_f64)
    float64x2_t ln2Hi_f64 = vdupq_n_f64(0.6931471803691238);
    float64x2_t ln2Lo_f64 = vdupq_n_f64(1.9082149292705877e-10);
    float64x2_t v_inv_ln2 = vdupq_n_f64(1.4426950408889634);
    float64x2_t expMin_f64 = vdupq_n_f64(-708.396);

    float64x2_t maxBroadcast = vdupq_n_f64(maxVal);
    float64x2_t sumVec = vdupq_n_f64(0.0);

    p = 0;
    for (; p + 2 <= size; p += 2) {
        // Subtract max
        float64x2_t x = vsubq_f64(vld1q_f64(input + p), maxBroadcast);
        x = vmaxq_f64(x, expMin_f64);

        // Inline exp(x) for f64
        float64x2_t k = vrndnq_f64(vmulq_f64(x, v_inv_ln2));
        // Range reduction using separate mul+sub (matches Go hwy.Sub/hwy.Mul)
        float64x2_t r = vsubq_f64(x, vmulq_f64(k, ln2Hi_f64));
        r = vsubq_f64(r, vmulq_f64(k, ln2Lo_f64));

        // Horner polynomial (8 terms for double precision)
        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);   // 1/8!
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);  // 1/7!
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);  // 1/6!
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);  // 1/5!
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);  // 1/4!
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);  // 1/3!
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);                         // 1/2!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);                         // 1/1!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);                         // 1/0!

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t result = vmulq_f64(exp_r, scale);

        vst1q_f64(output + p, result);
        sumVec = vaddq_f64(sumVec, result);
    }
    double expSum = vaddvq_f64(sumVec);

    // Scalar tail
    for (; p < size; p++) {
        double x = input[p] - maxVal;
        if (x < -708.396) x = -708.396;

        // Scalar exp using NEON for single element
        float64x2_t xv = vdupq_n_f64(x);
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
        float64x2_t result = vmulq_f64(exp_r, scale);

        double val = vgetq_lane_f64(result, 0);
        output[p] = val;
        expSum += val;
    }

    // =========================================================================
    // Pass 3: Normalize by 1/sum
    // =========================================================================
    double invSum = 1.0 / expSum;
    float64x2_t invSumVec = vdupq_n_f64(invSum);
    p = 0;
    for (; p + 2 <= size; p += 2) {
        float64x2_t v = vld1q_f64(output + p);
        vst1q_f64(output + p, vmulq_f64(v, invSumVec));
    }
    for (; p < size; p++) {
        output[p] *= invSum;
    }
}
