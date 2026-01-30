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

// LayerNorm NEON implementation for ARM64
//
// Computes layer normalization over groups of normSize elements:
//   output[i] = (input[i] - mean) / sqrt(variance + epsilon) * gamma[i] + beta[i]
//
// Three-pass SIMD algorithm per group:
//   1. Sum for mean (NEON vaddq + vaddvq horizontal reduction)
//   2. Sum of squared deviations for variance (NEON vfmaq FMA)
//   3. Normalize + optional affine transform (NEON vmulq, vfmaq)
//
// Inverse sqrt computed via NEON vrsqrte + 2 Newton-Raphson iterations (f32)
// or 3 iterations (f64) for full precision.

#include <arm_neon.h>

// =============================================================================
// layernorm_neon_f32: Layer normalization with gamma and beta (f32)
// =============================================================================
//
// func layernorm_neon_f32(input, output, gamma, beta unsafe.Pointer,
//                         psize, pnormsize unsafe.Pointer, pepsilon unsafe.Pointer)
void layernorm_neon_f32(float *input, float *output, float *gamma, float *beta,
                         long *psize, long *pnormsize, float *pepsilon) {
    long size = *psize;
    long normSize = *pnormsize;
    float epsilon = *pepsilon;

    if (size == 0) return;
    if (normSize <= 0) return;

    long numGroups = size / normSize;

    for (long g = 0; g < numGroups; g++) {
        long off = g * normSize;

        // Pass 1: Compute mean using NEON accumulation
        float32x4_t sumVec = vdupq_n_f32(0.0f);
        long p = 0;
        for (; p + 4 <= normSize; p += 4) {
            float32x4_t x = vld1q_f32(input + off + p);
            sumVec = vaddq_f32(sumVec, x);
        }
        float sum = vaddvq_f32(sumVec);
        for (; p < normSize; p++) {
            sum += input[off + p];
        }
        float mean = sum / (float)normSize;

        // Pass 2: Compute variance using NEON FMA
        float32x4_t meanVec = vdupq_n_f32(mean);
        float32x4_t varVec = vdupq_n_f32(0.0f);
        p = 0;
        for (; p + 4 <= normSize; p += 4) {
            float32x4_t x = vld1q_f32(input + off + p);
            float32x4_t diff = vsubq_f32(x, meanVec);
            varVec = vfmaq_f32(varVec, diff, diff);
        }
        float variance = vaddvq_f32(varVec);
        for (; p < normSize; p++) {
            float diff = input[off + p] - mean;
            variance += diff * diff;
        }
        variance /= (float)normSize;

        // Compute invStd = 1/sqrt(variance + epsilon) via NEON rsqrt + Newton-Raphson
        float varPlusEps = variance + epsilon;
        float32x2_t vpeps = vdup_n_f32(varPlusEps);
        float32x2_t est = vrsqrte_f32(vpeps);
        est = vmul_f32(est, vrsqrts_f32(vmul_f32(vpeps, est), est));
        est = vmul_f32(est, vrsqrts_f32(vmul_f32(vpeps, est), est));
        float invStd = vget_lane_f32(est, 0);

        // Pass 3: Normalize + affine transform
        float32x4_t invStdVec = vdupq_n_f32(invStd);
        p = 0;
        for (; p + 4 <= normSize; p += 4) {
            float32x4_t x = vld1q_f32(input + off + p);
            float32x4_t diff = vsubq_f32(x, meanVec);
            float32x4_t normed = vmulq_f32(diff, invStdVec);
            float32x4_t gv = vld1q_f32(gamma + p);
            float32x4_t bv = vld1q_f32(beta + p);
            float32x4_t result = vfmaq_f32(bv, normed, gv);
            vst1q_f32(output + off + p, result);
        }
        for (; p < normSize; p++) {
            float normed = (input[off + p] - mean) * invStd;
            output[off + p] = normed * gamma[p] + beta[p];
        }
    }
}

// =============================================================================
// layernorm_neon_f32_no_affine: Layer normalization without gamma/beta (f32)
// =============================================================================
//
// func layernorm_neon_f32_no_affine(input, output unsafe.Pointer,
//                                    psize, pnormsize unsafe.Pointer, pepsilon unsafe.Pointer)
void layernorm_neon_f32_no_affine(float *input, float *output,
                                   long *psize, long *pnormsize, float *pepsilon) {
    long size = *psize;
    long normSize = *pnormsize;
    float epsilon = *pepsilon;

    if (size == 0) return;
    if (normSize <= 0) return;

    long numGroups = size / normSize;

    for (long g = 0; g < numGroups; g++) {
        long off = g * normSize;

        // Pass 1: Mean
        float32x4_t sumVec = vdupq_n_f32(0.0f);
        long p = 0;
        for (; p + 4 <= normSize; p += 4) {
            float32x4_t x = vld1q_f32(input + off + p);
            sumVec = vaddq_f32(sumVec, x);
        }
        float sum = vaddvq_f32(sumVec);
        for (; p < normSize; p++) {
            sum += input[off + p];
        }
        float mean = sum / (float)normSize;

        // Pass 2: Variance
        float32x4_t meanVec = vdupq_n_f32(mean);
        float32x4_t varVec = vdupq_n_f32(0.0f);
        p = 0;
        for (; p + 4 <= normSize; p += 4) {
            float32x4_t x = vld1q_f32(input + off + p);
            float32x4_t diff = vsubq_f32(x, meanVec);
            varVec = vfmaq_f32(varVec, diff, diff);
        }
        float variance = vaddvq_f32(varVec);
        for (; p < normSize; p++) {
            float diff = input[off + p] - mean;
            variance += diff * diff;
        }
        variance /= (float)normSize;

        // Compute invStd via NEON rsqrt + Newton-Raphson
        float varPlusEps = variance + epsilon;
        float32x2_t vpeps = vdup_n_f32(varPlusEps);
        float32x2_t est = vrsqrte_f32(vpeps);
        est = vmul_f32(est, vrsqrts_f32(vmul_f32(vpeps, est), est));
        est = vmul_f32(est, vrsqrts_f32(vmul_f32(vpeps, est), est));
        float invStd = vget_lane_f32(est, 0);

        // Pass 3: Normalize (no affine)
        float32x4_t invStdVec = vdupq_n_f32(invStd);
        p = 0;
        for (; p + 4 <= normSize; p += 4) {
            float32x4_t x = vld1q_f32(input + off + p);
            float32x4_t diff = vsubq_f32(x, meanVec);
            float32x4_t result = vmulq_f32(diff, invStdVec);
            vst1q_f32(output + off + p, result);
        }
        for (; p < normSize; p++) {
            output[off + p] = (input[off + p] - mean) * invStd;
        }
    }
}

// =============================================================================
// layernorm_neon_f64: Layer normalization with gamma and beta (f64)
// =============================================================================
// Uses 2-wide vectors (float64x2), 3 Newton-Raphson iterations for full precision
//
// func layernorm_neon_f64(input, output, gamma, beta unsafe.Pointer,
//                         psize, pnormsize unsafe.Pointer, pepsilon unsafe.Pointer)
void layernorm_neon_f64(double *input, double *output, double *gamma, double *beta,
                         long *psize, long *pnormsize, double *pepsilon) {
    long size = *psize;
    long normSize = *pnormsize;
    double epsilon = *pepsilon;

    if (size == 0) return;
    if (normSize <= 0) return;

    long numGroups = size / normSize;

    for (long g = 0; g < numGroups; g++) {
        long off = g * normSize;

        // Pass 1: Mean
        float64x2_t sumVec = vdupq_n_f64(0.0);
        long p = 0;
        for (; p + 2 <= normSize; p += 2) {
            float64x2_t x = vld1q_f64(input + off + p);
            sumVec = vaddq_f64(sumVec, x);
        }
        double sum = vaddvq_f64(sumVec);
        for (; p < normSize; p++) {
            sum += input[off + p];
        }
        double mean = sum / (double)normSize;

        // Pass 2: Variance
        float64x2_t meanVec = vdupq_n_f64(mean);
        float64x2_t varVec = vdupq_n_f64(0.0);
        p = 0;
        for (; p + 2 <= normSize; p += 2) {
            float64x2_t x = vld1q_f64(input + off + p);
            float64x2_t diff = vsubq_f64(x, meanVec);
            varVec = vfmaq_f64(varVec, diff, diff);
        }
        double variance = vaddvq_f64(varVec);
        for (; p < normSize; p++) {
            double diff = input[off + p] - mean;
            variance += diff * diff;
        }
        variance /= (double)normSize;

        // Compute invStd via NEON rsqrt + 3 Newton-Raphson iterations (f64 needs more)
        double varPlusEps = variance + epsilon;
        float64x1_t vpeps = vdup_n_f64(varPlusEps);
        float64x1_t est64 = vrsqrte_f64(vpeps);
        est64 = vmul_f64(est64, vrsqrts_f64(vmul_f64(vpeps, est64), est64));
        est64 = vmul_f64(est64, vrsqrts_f64(vmul_f64(vpeps, est64), est64));
        est64 = vmul_f64(est64, vrsqrts_f64(vmul_f64(vpeps, est64), est64));
        double invStd = vget_lane_f64(est64, 0);

        // Pass 3: Normalize + affine
        float64x2_t invStdVec = vdupq_n_f64(invStd);
        p = 0;
        for (; p + 2 <= normSize; p += 2) {
            float64x2_t x = vld1q_f64(input + off + p);
            float64x2_t diff = vsubq_f64(x, meanVec);
            float64x2_t normed = vmulq_f64(diff, invStdVec);
            float64x2_t gv = vld1q_f64(gamma + p);
            float64x2_t bv = vld1q_f64(beta + p);
            float64x2_t result = vfmaq_f64(bv, normed, gv);
            vst1q_f64(output + off + p, result);
        }
        for (; p < normSize; p++) {
            double normed = (input[off + p] - mean) * invStd;
            output[off + p] = normed * gamma[p] + beta[p];
        }
    }
}

// =============================================================================
// layernorm_neon_f64_no_affine: Layer normalization without gamma/beta (f64)
// =============================================================================
//
// func layernorm_neon_f64_no_affine(input, output unsafe.Pointer,
//                                    psize, pnormsize unsafe.Pointer, pepsilon unsafe.Pointer)
void layernorm_neon_f64_no_affine(double *input, double *output,
                                   long *psize, long *pnormsize, double *pepsilon) {
    long size = *psize;
    long normSize = *pnormsize;
    double epsilon = *pepsilon;

    if (size == 0) return;
    if (normSize <= 0) return;

    long numGroups = size / normSize;

    for (long g = 0; g < numGroups; g++) {
        long off = g * normSize;

        // Pass 1: Mean
        float64x2_t sumVec = vdupq_n_f64(0.0);
        long p = 0;
        for (; p + 2 <= normSize; p += 2) {
            float64x2_t x = vld1q_f64(input + off + p);
            sumVec = vaddq_f64(sumVec, x);
        }
        double sum = vaddvq_f64(sumVec);
        for (; p < normSize; p++) {
            sum += input[off + p];
        }
        double mean = sum / (double)normSize;

        // Pass 2: Variance
        float64x2_t meanVec = vdupq_n_f64(mean);
        float64x2_t varVec = vdupq_n_f64(0.0);
        p = 0;
        for (; p + 2 <= normSize; p += 2) {
            float64x2_t x = vld1q_f64(input + off + p);
            float64x2_t diff = vsubq_f64(x, meanVec);
            varVec = vfmaq_f64(varVec, diff, diff);
        }
        double variance = vaddvq_f64(varVec);
        for (; p < normSize; p++) {
            double diff = input[off + p] - mean;
            variance += diff * diff;
        }
        variance /= (double)normSize;

        // Compute invStd via NEON rsqrt + 3 Newton-Raphson iterations
        double varPlusEps = variance + epsilon;
        float64x1_t vpeps = vdup_n_f64(varPlusEps);
        float64x1_t est64 = vrsqrte_f64(vpeps);
        est64 = vmul_f64(est64, vrsqrts_f64(vmul_f64(vpeps, est64), est64));
        est64 = vmul_f64(est64, vrsqrts_f64(vmul_f64(vpeps, est64), est64));
        est64 = vmul_f64(est64, vrsqrts_f64(vmul_f64(vpeps, est64), est64));
        double invStd = vget_lane_f64(est64, 0);

        // Pass 3: Normalize (no affine)
        float64x2_t invStdVec = vdupq_n_f64(invStd);
        p = 0;
        for (; p + 2 <= normSize; p += 2) {
            float64x2_t x = vld1q_f64(input + off + p);
            float64x2_t diff = vsubq_f64(x, meanVec);
            float64x2_t result = vmulq_f64(diff, invStdVec);
            vst1q_f64(output + off + p, result);
        }
        for (; p < normSize; p++) {
            output[off + p] = (input[off + p] - mean) * invStd;
        }
    }
}
