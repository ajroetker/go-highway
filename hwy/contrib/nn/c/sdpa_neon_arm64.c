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

// Scaled Dot-Product Attention NEON implementation for ARM64
//
// Fused SDPA: Q@K^T -> scale -> mask -> softmax -> @V
// Key wins over Go base:
//   1. Fused subtract-max + exp in softmax (saves one pass)
//   2. NEON FMA for dot products and exp polynomial
//
// Algorithm:
//   1. scores[i,j] = dot(Q[i,:], K[j,:]) * scale + mask[i,j]
//   2. Per-row softmax on scores (3-pass: max, exp+sum, normalize)
//   3. output[i,:] = sum_j(scores[i,j] * V[j,:])

#include <arm_neon.h>

// =============================================================================
// sdpa_neon_f32: Scaled Dot-Product Attention for float32
// =============================================================================
//
// func sdpa_neon_f32(q, k, v, mask, scores, output, pdims, pscale unsafe.Pointer)
// pdims: [0]=seqLen, [1]=kvLen, [2]=headDim
void sdpa_neon_f32(float *q, float *k, float *v, float *mask,
                    float *scores, float *output,
                    long *pdims, float *pscale) {
    long seqLen = pdims[0];
    long kvLen = pdims[1];
    long headDim = pdims[2];
    float scale = *pscale;

    if (seqLen <= 0) return;
    if (kvLen <= 0) return;
    if (headDim <= 0) return;

    // Exp polynomial constants (same as softmax_neon_arm64.c)
    float32x4_t invLn2 = vdupq_n_f32(1.44269504088896341f);
    float32x4_t ln2Hi  = vdupq_n_f32(0.693359375f);
    float32x4_t ln2Lo  = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t c1 = vdupq_n_f32(1.0f);
    float32x4_t c2 = vdupq_n_f32(0.5f);
    float32x4_t c3 = vdupq_n_f32(0.16666666666666666f);
    float32x4_t c4 = vdupq_n_f32(0.041666666666666664f);
    float32x4_t c5 = vdupq_n_f32(0.008333333333333333f);
    float32x4_t c6 = vdupq_n_f32(0.001388888888888889f);
    int32x4_t expBias = vdupq_n_s32(127);
    // Clamp exp input to prevent 2^k overflow when k+127 < 0
    float32x4_t expMin = vdupq_n_f32(-87.3365f);

    float32x4_t vscale = vdupq_n_f32(scale);

    // Step 1: Q @ K^T -> scores, scaled + mask
    for (long i = 0; i < seqLen; i++) {
        float *qRow = q + i * headDim;
        long sOff = i * kvLen;

        for (long j = 0; j < kvLen; j++) {
            float *kRow = k + j * headDim;

            float32x4_t acc = vdupq_n_f32(0.0f);
            long p = 0;
            for (; p + 4 <= headDim; p += 4) {
                float32x4_t vq = vld1q_f32(qRow + p);
                float32x4_t vk = vld1q_f32(kRow + p);
                acc = vfmaq_f32(acc, vq, vk);
            }
            float dot = vaddvq_f32(acc);
            for (; p < headDim; p++) {
                dot += qRow[p] * kRow[p];
            }

            dot *= scale;
            if (mask) {
                dot += mask[i * kvLen + j];
            }
            scores[sOff + j] = dot;
        }

        // Step 2: Per-row softmax (fused subtract-max + exp + normalize)

        // Pass 2a: Find max
        float32x4_t maxVec = vdupq_n_f32(scores[sOff]);
        long p = 0;
        for (; p + 4 <= kvLen; p += 4) {
            float32x4_t s = vld1q_f32(scores + sOff + p);
            maxVec = vmaxq_f32(maxVec, s);
        }
        float maxVal = vmaxvq_f32(maxVec);
        for (; p < kvLen; p++) {
            if (scores[sOff + p] > maxVal) {
                maxVal = scores[sOff + p];
            }
        }

        // Pass 2b: Subtract max + exp + sum
        float32x4_t maxBroadcast = vdupq_n_f32(maxVal);
        float32x4_t sumVec = vdupq_n_f32(0.0f);
        p = 0;
        for (; p + 4 <= kvLen; p += 4) {
            float32x4_t x = vsubq_f32(vld1q_f32(scores + sOff + p), maxBroadcast);
            x = vmaxq_f32(x, expMin);

            // Inline exp
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
            int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, expBias), 23);
            float32x4_t escale = vreinterpretq_f32_s32(scale_bits);
            float32x4_t result = vmulq_f32(ep, escale);

            vst1q_f32(scores + sOff + p, result);
            sumVec = vaddq_f32(sumVec, result);
        }
        float expSum = vaddvq_f32(sumVec);
        for (; p < kvLen; p++) {
            float x = scores[sOff + p] - maxVal;
            if (x < -87.3365f) x = -87.3365f;

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
            int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, expBias), 23);
            float32x4_t escale = vreinterpretq_f32_s32(scale_bits);
            float32x4_t result = vmulq_f32(ep, escale);

            float val = vgetq_lane_f32(result, 0);
            scores[sOff + p] = val;
            expSum += val;
        }

        // Pass 2c: Normalize
        float invSum = 1.0f / expSum;
        float32x4_t invSumVec = vdupq_n_f32(invSum);
        p = 0;
        for (; p + 4 <= kvLen; p += 4) {
            float32x4_t s = vld1q_f32(scores + sOff + p);
            vst1q_f32(scores + sOff + p, vmulq_f32(s, invSumVec));
        }
        for (; p < kvLen; p++) {
            scores[sOff + p] *= invSum;
        }
    }

    // Step 3: scores @ V -> output
    for (long i = 0; i < seqLen; i++) {
        long sOff = i * kvLen;
        long oOff = i * headDim;

        // Zero output row
        long p = 0;
        for (; p + 4 <= headDim; p += 4) {
            vst1q_f32(output + oOff + p, vdupq_n_f32(0.0f));
        }
        for (; p < headDim; p++) {
            output[oOff + p] = 0.0f;
        }

        // Accumulate weighted values
        for (long j = 0; j < kvLen; j++) {
            float w = scores[sOff + j];
            if (w == 0.0f) continue;

            float *vRow = v + j * headDim;
            float32x4_t vw = vdupq_n_f32(w);

            p = 0;
            for (; p + 4 <= headDim; p += 4) {
                float32x4_t o = vld1q_f32(output + oOff + p);
                float32x4_t vv = vld1q_f32(vRow + p);
                vst1q_f32(output + oOff + p, vfmaq_f32(o, vw, vv));
            }
            for (; p < headDim; p++) {
                output[oOff + p] += w * vRow[p];
            }
        }
    }
}

// =============================================================================
// sdpa_causal_neon_f32: Causal Scaled Dot-Product Attention for float32
// =============================================================================
//
// func sdpa_causal_neon_f32(q, k, v, scores, output, pdims, pscale unsafe.Pointer)
// pdims: [0]=seqLen, [1]=kvLen, [2]=headDim
void sdpa_causal_neon_f32(float *q, float *k, float *v,
                            float *scores, float *output,
                            long *pdims, float *pscale) {
    long seqLen = pdims[0];
    long kvLen = pdims[1];
    long headDim = pdims[2];
    float scale = *pscale;
    long offset = kvLen - seqLen;

    if (seqLen <= 0) return;
    if (kvLen <= 0) return;
    if (headDim <= 0) return;

    // Exp constants
    float32x4_t invLn2 = vdupq_n_f32(1.44269504088896341f);
    float32x4_t ln2Hi  = vdupq_n_f32(0.693359375f);
    float32x4_t ln2Lo  = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t ec1 = vdupq_n_f32(1.0f);
    float32x4_t ec2 = vdupq_n_f32(0.5f);
    float32x4_t ec3 = vdupq_n_f32(0.16666666666666666f);
    float32x4_t ec4 = vdupq_n_f32(0.041666666666666664f);
    float32x4_t ec5 = vdupq_n_f32(0.008333333333333333f);
    float32x4_t ec6 = vdupq_n_f32(0.001388888888888889f);
    int32x4_t expBias = vdupq_n_s32(127);
    float32x4_t expMin = vdupq_n_f32(-87.3365f);

    float negInf = -1.0f / 0.0f;

    for (long i = 0; i < seqLen; i++) {
        float *qRow = q + i * headDim;
        long sOff = i * kvLen;
        long causalEnd = i + offset + 1;
        if (causalEnd > kvLen) {
            causalEnd = kvLen;
        }

        // Compute scores for attended positions
        for (long j = 0; j < causalEnd; j++) {
            float *kRow = k + j * headDim;

            float32x4_t acc = vdupq_n_f32(0.0f);
            long p = 0;
            for (; p + 4 <= headDim; p += 4) {
                float32x4_t vq = vld1q_f32(qRow + p);
                float32x4_t vk = vld1q_f32(kRow + p);
                acc = vfmaq_f32(acc, vq, vk);
            }
            float dot = vaddvq_f32(acc);
            for (; p < headDim; p++) {
                dot += qRow[p] * kRow[p];
            }

            scores[sOff + j] = dot * scale;
        }

        // Set masked positions to -inf
        for (long j = causalEnd; j < kvLen; j++) {
            scores[sOff + j] = negInf;
        }

        // Per-row softmax (same as non-causal)
        float32x4_t maxVec = vdupq_n_f32(scores[sOff]);
        long p = 0;
        for (; p + 4 <= kvLen; p += 4) {
            float32x4_t s = vld1q_f32(scores + sOff + p);
            maxVec = vmaxq_f32(maxVec, s);
        }
        float maxVal = vmaxvq_f32(maxVec);
        for (; p < kvLen; p++) {
            if (scores[sOff + p] > maxVal) {
                maxVal = scores[sOff + p];
            }
        }

        float32x4_t maxBroadcast = vdupq_n_f32(maxVal);
        float32x4_t sumVec = vdupq_n_f32(0.0f);
        p = 0;
        for (; p + 4 <= kvLen; p += 4) {
            float32x4_t x = vsubq_f32(vld1q_f32(scores + sOff + p), maxBroadcast);
            x = vmaxq_f32(x, expMin);

            float32x4_t kf = vrndnq_f32(vmulq_f32(x, invLn2));
            float32x4_t r = vsubq_f32(x, vmulq_f32(kf, ln2Hi));
            r = vsubq_f32(r, vmulq_f32(kf, ln2Lo));

            float32x4_t ep = vfmaq_f32(ec5, ec6, r);
            ep = vfmaq_f32(ec4, ep, r);
            ep = vfmaq_f32(ec3, ep, r);
            ep = vfmaq_f32(ec2, ep, r);
            ep = vfmaq_f32(ec1, ep, r);
            ep = vfmaq_f32(ec1, ep, r);

            int32x4_t ki = vcvtnq_s32_f32(kf);
            int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, expBias), 23);
            float32x4_t escale = vreinterpretq_f32_s32(scale_bits);
            float32x4_t result = vmulq_f32(ep, escale);

            vst1q_f32(scores + sOff + p, result);
            sumVec = vaddq_f32(sumVec, result);
        }
        float expSum = vaddvq_f32(sumVec);
        for (; p < kvLen; p++) {
            float x = scores[sOff + p] - maxVal;
            if (x < -87.3365f) x = -87.3365f;

            float32x4_t xv = vdupq_n_f32(x);
            float32x4_t kf = vrndnq_f32(vmulq_f32(xv, invLn2));
            float32x4_t r = vsubq_f32(xv, vmulq_f32(kf, ln2Hi));
            r = vsubq_f32(r, vmulq_f32(kf, ln2Lo));

            float32x4_t ep = vfmaq_f32(ec5, ec6, r);
            ep = vfmaq_f32(ec4, ep, r);
            ep = vfmaq_f32(ec3, ep, r);
            ep = vfmaq_f32(ec2, ep, r);
            ep = vfmaq_f32(ec1, ep, r);
            ep = vfmaq_f32(ec1, ep, r);

            int32x4_t ki = vcvtnq_s32_f32(kf);
            int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, expBias), 23);
            float32x4_t escale = vreinterpretq_f32_s32(scale_bits);
            float32x4_t result = vmulq_f32(ep, escale);

            float val = vgetq_lane_f32(result, 0);
            scores[sOff + p] = val;
            expSum += val;
        }

        float invSum = 1.0f / expSum;
        float32x4_t invSumVec = vdupq_n_f32(invSum);
        p = 0;
        for (; p + 4 <= kvLen; p += 4) {
            float32x4_t s = vld1q_f32(scores + sOff + p);
            vst1q_f32(scores + sOff + p, vmulq_f32(s, invSumVec));
        }
        for (; p < kvLen; p++) {
            scores[sOff + p] *= invSum;
        }

        // scores @ V -> output (only attend to causalEnd positions)
        long oOff = i * headDim;
        p = 0;
        for (; p + 4 <= headDim; p += 4) {
            vst1q_f32(output + oOff + p, vdupq_n_f32(0.0f));
        }
        for (; p < headDim; p++) {
            output[oOff + p] = 0.0f;
        }

        for (long j = 0; j < causalEnd; j++) {
            float w = scores[sOff + j];
            if (w == 0.0f) continue;

            float *vRow = v + j * headDim;
            float32x4_t vw = vdupq_n_f32(w);

            p = 0;
            for (; p + 4 <= headDim; p += 4) {
                float32x4_t o = vld1q_f32(output + oOff + p);
                float32x4_t vv = vld1q_f32(vRow + p);
                vst1q_f32(output + oOff + p, vfmaq_f32(o, vw, vv));
            }
            for (; p < headDim; p++) {
                output[oOff + p] += w * vRow[p];
            }
        }
    }
}

// =============================================================================
// sdpa_neon_f64: Scaled Dot-Product Attention for float64
// =============================================================================
//
// func sdpa_neon_f64(q, k, v, mask, scores, output, pdims, pscale unsafe.Pointer)
// pdims: [0]=seqLen, [1]=kvLen, [2]=headDim
void sdpa_neon_f64(double *q, double *k, double *v, double *mask,
                    double *scores, double *output,
                    long *pdims, double *pscale) {
    long seqLen = pdims[0];
    long kvLen = pdims[1];
    long headDim = pdims[2];
    double scale = *pscale;

    if (seqLen <= 0) return;
    if (kvLen <= 0) return;
    if (headDim <= 0) return;

    // f64 exp constants
    float64x2_t ln2Hi_f64 = vdupq_n_f64(0.6931471803691238);
    float64x2_t ln2Lo_f64 = vdupq_n_f64(1.9082149292705877e-10);
    float64x2_t v_inv_ln2 = vdupq_n_f64(1.4426950408889634);
    float64x2_t expMin_f64 = vdupq_n_f64(-708.396);

    for (long i = 0; i < seqLen; i++) {
        double *qRow = q + i * headDim;
        long sOff = i * kvLen;

        // Compute scores
        for (long j = 0; j < kvLen; j++) {
            double *kRow = k + j * headDim;

            float64x2_t acc = vdupq_n_f64(0.0);
            long p = 0;
            for (; p + 2 <= headDim; p += 2) {
                float64x2_t vq = vld1q_f64(qRow + p);
                float64x2_t vk = vld1q_f64(kRow + p);
                acc = vfmaq_f64(acc, vq, vk);
            }
            double dot = vaddvq_f64(acc);
            for (; p < headDim; p++) {
                dot += qRow[p] * kRow[p];
            }

            dot *= scale;
            if (mask) {
                dot += mask[i * kvLen + j];
            }
            scores[sOff + j] = dot;
        }

        // Per-row softmax
        float64x2_t maxVec = vdupq_n_f64(scores[sOff]);
        long p = 0;
        for (; p + 2 <= kvLen; p += 2) {
            float64x2_t s = vld1q_f64(scores + sOff + p);
            maxVec = vmaxq_f64(maxVec, s);
        }
        double maxVal = vmaxvq_f64(maxVec);
        for (; p < kvLen; p++) {
            if (scores[sOff + p] > maxVal) {
                maxVal = scores[sOff + p];
            }
        }

        float64x2_t maxBroadcast = vdupq_n_f64(maxVal);
        float64x2_t sumVec = vdupq_n_f64(0.0);
        p = 0;
        for (; p + 2 <= kvLen; p += 2) {
            float64x2_t x = vsubq_f64(vld1q_f64(scores + sOff + p), maxBroadcast);
            x = vmaxq_f64(x, expMin_f64);

            float64x2_t kk = vrndnq_f64(vmulq_f64(x, v_inv_ln2));
            float64x2_t r = vsubq_f64(x, vmulq_f64(kk, ln2Hi_f64));
            r = vsubq_f64(r, vmulq_f64(kk, ln2Lo_f64));

            float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
            exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

            int64x2_t ki = vcvtq_s64_f64(kk);
            int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
            float64x2_t escale = vreinterpretq_f64_s64(exp_bits);
            float64x2_t result = vmulq_f64(exp_r, escale);

            vst1q_f64(scores + sOff + p, result);
            sumVec = vaddq_f64(sumVec, result);
        }
        double expSum = vaddvq_f64(sumVec);
        for (; p < kvLen; p++) {
            double x = scores[sOff + p] - maxVal;
            if (x < -708.396) x = -708.396;

            float64x2_t xv = vdupq_n_f64(x);
            float64x2_t kk = vrndnq_f64(vmulq_f64(xv, v_inv_ln2));
            float64x2_t r = vsubq_f64(xv, vmulq_f64(kk, ln2Hi_f64));
            r = vsubq_f64(r, vmulq_f64(kk, ln2Lo_f64));

            float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
            exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

            int64x2_t ki = vcvtq_s64_f64(kk);
            int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
            float64x2_t escale = vreinterpretq_f64_s64(exp_bits);
            float64x2_t result = vmulq_f64(exp_r, escale);

            double val = vgetq_lane_f64(result, 0);
            scores[sOff + p] = val;
            expSum += val;
        }

        double invSum = 1.0 / expSum;
        float64x2_t invSumVec = vdupq_n_f64(invSum);
        p = 0;
        for (; p + 2 <= kvLen; p += 2) {
            float64x2_t s = vld1q_f64(scores + sOff + p);
            vst1q_f64(scores + sOff + p, vmulq_f64(s, invSumVec));
        }
        for (; p < kvLen; p++) {
            scores[sOff + p] *= invSum;
        }

        // scores @ V -> output
        long oOff = i * headDim;
        p = 0;
        for (; p + 2 <= headDim; p += 2) {
            vst1q_f64(output + oOff + p, vdupq_n_f64(0.0));
        }
        for (; p < headDim; p++) {
            output[oOff + p] = 0.0;
        }

        for (long j = 0; j < kvLen; j++) {
            double w = scores[sOff + j];
            if (w == 0.0) continue;

            double *vRow = v + j * headDim;
            float64x2_t vw = vdupq_n_f64(w);

            p = 0;
            for (; p + 2 <= headDim; p += 2) {
                float64x2_t o = vld1q_f64(output + oOff + p);
                float64x2_t vv = vld1q_f64(vRow + p);
                vst1q_f64(output + oOff + p, vfmaq_f64(o, vw, vv));
            }
            for (; p < headDim; p++) {
                output[oOff + p] += w * vRow[p];
            }
        }
    }
}

// =============================================================================
// sdpa_causal_neon_f64: Causal SDPA for float64
// =============================================================================
//
// func sdpa_causal_neon_f64(q, k, v, scores, output, pdims, pscale unsafe.Pointer)
// pdims: [0]=seqLen, [1]=kvLen, [2]=headDim
void sdpa_causal_neon_f64(double *q, double *k, double *v,
                            double *scores, double *output,
                            long *pdims, double *pscale) {
    long seqLen = pdims[0];
    long kvLen = pdims[1];
    long headDim = pdims[2];
    double scale = *pscale;
    long loffset = kvLen - seqLen;

    if (seqLen <= 0) return;
    if (kvLen <= 0) return;
    if (headDim <= 0) return;

    double negInf = -1.0 / 0.0;

    float64x2_t ln2Hi_f64 = vdupq_n_f64(0.6931471803691238);
    float64x2_t ln2Lo_f64 = vdupq_n_f64(1.9082149292705877e-10);
    float64x2_t v_inv_ln2 = vdupq_n_f64(1.4426950408889634);
    float64x2_t expMin_f64 = vdupq_n_f64(-708.396);

    for (long i = 0; i < seqLen; i++) {
        double *qRow = q + i * headDim;
        long sOff = i * kvLen;
        long causalEnd = i + loffset + 1;
        if (causalEnd > kvLen) {
            causalEnd = kvLen;
        }

        for (long j = 0; j < causalEnd; j++) {
            double *kRow = k + j * headDim;

            float64x2_t acc = vdupq_n_f64(0.0);
            long p = 0;
            for (; p + 2 <= headDim; p += 2) {
                float64x2_t vq = vld1q_f64(qRow + p);
                float64x2_t vk = vld1q_f64(kRow + p);
                acc = vfmaq_f64(acc, vq, vk);
            }
            double dot = vaddvq_f64(acc);
            for (; p < headDim; p++) {
                dot += qRow[p] * kRow[p];
            }

            scores[sOff + j] = dot * scale;
        }

        for (long j = causalEnd; j < kvLen; j++) {
            scores[sOff + j] = negInf;
        }

        // Softmax
        float64x2_t maxVec = vdupq_n_f64(scores[sOff]);
        long p = 0;
        for (; p + 2 <= kvLen; p += 2) {
            float64x2_t s = vld1q_f64(scores + sOff + p);
            maxVec = vmaxq_f64(maxVec, s);
        }
        double maxVal = vmaxvq_f64(maxVec);
        for (; p < kvLen; p++) {
            if (scores[sOff + p] > maxVal) {
                maxVal = scores[sOff + p];
            }
        }

        float64x2_t maxBroadcast = vdupq_n_f64(maxVal);
        float64x2_t sumVec = vdupq_n_f64(0.0);
        p = 0;
        for (; p + 2 <= kvLen; p += 2) {
            float64x2_t x = vsubq_f64(vld1q_f64(scores + sOff + p), maxBroadcast);
            x = vmaxq_f64(x, expMin_f64);

            float64x2_t kk = vrndnq_f64(vmulq_f64(x, v_inv_ln2));
            float64x2_t r = vsubq_f64(x, vmulq_f64(kk, ln2Hi_f64));
            r = vsubq_f64(r, vmulq_f64(kk, ln2Lo_f64));

            float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
            exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

            int64x2_t ki = vcvtq_s64_f64(kk);
            int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
            float64x2_t escale = vreinterpretq_f64_s64(exp_bits);
            float64x2_t result = vmulq_f64(exp_r, escale);

            vst1q_f64(scores + sOff + p, result);
            sumVec = vaddq_f64(sumVec, result);
        }
        double expSum = vaddvq_f64(sumVec);
        for (; p < kvLen; p++) {
            double x = scores[sOff + p] - maxVal;
            if (x < -708.396) x = -708.396;

            float64x2_t xv = vdupq_n_f64(x);
            float64x2_t kk = vrndnq_f64(vmulq_f64(xv, v_inv_ln2));
            float64x2_t r = vsubq_f64(xv, vmulq_f64(kk, ln2Hi_f64));
            r = vsubq_f64(r, vmulq_f64(kk, ln2Lo_f64));

            float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
            exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
            exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

            int64x2_t ki = vcvtq_s64_f64(kk);
            int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
            float64x2_t escale = vreinterpretq_f64_s64(exp_bits);
            float64x2_t result = vmulq_f64(exp_r, escale);

            double val = vgetq_lane_f64(result, 0);
            scores[sOff + p] = val;
            expSum += val;
        }

        double invSum = 1.0 / expSum;
        float64x2_t invSumVec = vdupq_n_f64(invSum);
        p = 0;
        for (; p + 2 <= kvLen; p += 2) {
            float64x2_t s = vld1q_f64(scores + sOff + p);
            vst1q_f64(scores + sOff + p, vmulq_f64(s, invSumVec));
        }
        for (; p < kvLen; p++) {
            scores[sOff + p] *= invSum;
        }

        // scores @ V -> output
        long oOff = i * headDim;
        p = 0;
        for (; p + 2 <= headDim; p += 2) {
            vst1q_f64(output + oOff + p, vdupq_n_f64(0.0));
        }
        for (; p < headDim; p++) {
            output[oOff + p] = 0.0;
        }

        for (long j = 0; j < causalEnd; j++) {
            double w = scores[sOff + j];
            if (w == 0.0) continue;

            double *vRow = v + j * headDim;
            float64x2_t vw = vdupq_n_f64(w);

            p = 0;
            for (; p + 2 <= headDim; p += 2) {
                float64x2_t o = vld1q_f64(output + oOff + p);
                float64x2_t vv = vld1q_f64(vRow + p);
                vst1q_f64(output + oOff + p, vfmaq_f64(o, vw, vv));
            }
            for (; p < headDim; p++) {
                output[oOff + p] += w * vRow[p];
            }
        }
    }
}
