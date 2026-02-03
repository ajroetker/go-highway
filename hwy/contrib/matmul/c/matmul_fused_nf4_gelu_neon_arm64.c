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

// NEON Fused NF4 Dequant + Matrix Multiplication + GELU Activation for ARM64
// Compile with: -march=armv8-a+simd
//
// Performs fused dequantization, matmul, and GELU activation in a single pass:
//   output[m,n] = GELU(sum_k(input[m,k] * dequant(packed[k,n])))
//
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// NF4 lookup table - 16 fixed values for 4-bit NormalFloat quantization
static const float nf4_table[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

// =============================================================================
// fused_nf4_gelu_matmul_neon: Fused NF4 dequant + matmul + GELU using NEON
// =============================================================================
// Computes output = GELU(input @ dequant(packed, scales))
//
// Parameters:
//   input:     [M, K] float32 input matrix (row-major)
//   packed:    [K, N/2] uint8 packed NF4 weights (2 values per byte)
//   scales:    [K, numGroups] float32 per-row, per-group scales
//   output:    [M, N] float32 output matrix (row-major)
//   M, K, N:   matrix dimensions
//   groupSize: number of columns per scale group
//
// Packing format: low nibble = even column, high nibble = odd column
//
// func fused_nf4_gelu_matmul_neon(input, packed, scales, output unsafe.Pointer,
//                                  M, K, N, groupSize, numGroups *int64)
void fused_nf4_gelu_matmul_neon(float *input, unsigned char *packed, float *scales,
                                 float *output, long *pM, long *pK, long *pN,
                                 long *pGroupSize, long *pNumGroups) {
    long M = *pM;
    long K = *pK;
    long N = *pN;
    long groupSize = *pGroupSize;
    long numGroups = *pNumGroups;

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

    // Exp constants
    float32x4_t v_ln2Hi = vdupq_n_f32(0.693359375f);
    float32x4_t v_ln2Lo = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t v_inv_ln2 = vdupq_n_f32(1.44269504088896341f);
    float32x4_t v_min_clamp = vdupq_n_f32(-88.0f);
    float32x4_t v_max_clamp = vdupq_n_f32(88.0f);

    // Process each output row
    for (long m = 0; m < M; m++) {
        float *inputRow = input + m * K;
        float *outputRow = output + m * N;

        // Process output columns in chunks of 4
        for (long n = 0; n < N; n += 4) {
            // Initialize accumulator
            float32x4_t acc = vdupq_n_f32(0.0f);

            // Accumulate over K dimension
            for (long k = 0; k < K; k++) {
                // Broadcast input[m, k]
                float32x4_t inputVal = vdupq_n_f32(inputRow[k]);

                // Dequantize 4 weights from packed[k, n:n+4]
                long weightIdx0 = k * N + n;
                long weightIdx2 = k * N + n + 2;

                long packedIdx0 = weightIdx0 / 2;
                long packedIdx1 = weightIdx2 / 2;

                unsigned char byte0 = packed[packedIdx0];
                unsigned char byte1 = packed[packedIdx1];

                // Extract nibbles
                int q0 = byte0 & 0x0F;
                int q1 = (byte0 >> 4) & 0x0F;
                int q2 = byte1 & 0x0F;
                int q3 = (byte1 >> 4) & 0x0F;

                // Table lookup for NF4 values
                float w0 = nf4_table[q0];
                float w1 = nf4_table[q1];
                float w2 = nf4_table[q2];
                float w3 = nf4_table[q3];

                // Get scales for each column's group
                long g0 = (n + 0) / groupSize;
                long g1 = (n + 1) / groupSize;
                long g2 = (n + 2) / groupSize;
                long g3 = (n + 3) / groupSize;

                float s0 = scales[k * numGroups + g0];
                float s1 = scales[k * numGroups + g1];
                float s2 = scales[k * numGroups + g2];
                float s3 = scales[k * numGroups + g3];

                // Apply scales
                w0 *= s0;
                w1 *= s1;
                w2 *= s2;
                w3 *= s3;

                // Create weight vector and accumulate
                float weights[4] = {w0, w1, w2, w3};
                float32x4_t weightVec = vld1q_f32(weights);

                acc = vfmaq_f32(acc, inputVal, weightVec);
            }

            // Apply GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
            // xs = x * invSqrt2
            float32x4_t xs = vmulq_f32(acc, v_inv_sqrt2);

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

            // Inline exp(-xs^2)
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

            // erf = 1 - poly * exp(-xs^2)
            float32x4_t erf_abs = vsubq_f32(v_one, vmulq_f32(poly, exp_neg_xs2));

            // Apply sign
            float32x4_t erf_val = vbslq_f32(is_negative, vnegq_f32(erf_abs), erf_abs);

            // GELU = x * 0.5 * (1 + erf)
            float32x4_t one_plus_erf = vaddq_f32(v_one, erf_val);
            float32x4_t result = vmulq_f32(acc, vmulq_f32(v_half, one_plus_erf));

            // Store result
            vst1q_f32(outputRow + n, result);
        }
    }
}

// =============================================================================
// fused_int4_gelu_matmul_neon: Fused Int4 dequant + matmul + GELU using NEON
// =============================================================================
// Same as NF4 but uses symmetric integer quantization:
// Values 0-15 map to -8 to +7 (subtract 8)
//
// func fused_int4_gelu_matmul_neon(input, packed, scales, output unsafe.Pointer,
//                                   M, K, N, groupSize, numGroups *int64)
void fused_int4_gelu_matmul_neon(float *input, unsigned char *packed, float *scales,
                                  float *output, long *pM, long *pK, long *pN,
                                  long *pGroupSize, long *pNumGroups) {
    long M = *pM;
    long K = *pK;
    long N = *pN;
    long groupSize = *pGroupSize;
    long numGroups = *pNumGroups;

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

    // Exp constants
    float32x4_t v_ln2Hi = vdupq_n_f32(0.693359375f);
    float32x4_t v_ln2Lo = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t v_inv_ln2 = vdupq_n_f32(1.44269504088896341f);
    float32x4_t v_min_clamp = vdupq_n_f32(-88.0f);
    float32x4_t v_max_clamp = vdupq_n_f32(88.0f);

    for (long m = 0; m < M; m++) {
        float *inputRow = input + m * K;
        float *outputRow = output + m * N;

        for (long n = 0; n < N; n += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);

            for (long k = 0; k < K; k++) {
                float32x4_t inputVal = vdupq_n_f32(inputRow[k]);

                long weightIdx0 = k * N + n;
                long weightIdx2 = k * N + n + 2;

                long packedIdx0 = weightIdx0 / 2;
                long packedIdx1 = weightIdx2 / 2;

                unsigned char byte0 = packed[packedIdx0];
                unsigned char byte1 = packed[packedIdx1];

                // Extract nibbles and convert to signed [-8, 7]
                int q0 = (byte0 & 0x0F) - 8;
                int q1 = ((byte0 >> 4) & 0x0F) - 8;
                int q2 = (byte1 & 0x0F) - 8;
                int q3 = ((byte1 >> 4) & 0x0F) - 8;

                // Get scales
                long g0 = (n + 0) / groupSize;
                long g1 = (n + 1) / groupSize;
                long g2 = (n + 2) / groupSize;
                long g3 = (n + 3) / groupSize;

                float s0 = scales[k * numGroups + g0];
                float s1 = scales[k * numGroups + g1];
                float s2 = scales[k * numGroups + g2];
                float s3 = scales[k * numGroups + g3];

                // Dequantize: int4_val * scale
                float w0 = (float)q0 * s0;
                float w1 = (float)q1 * s1;
                float w2 = (float)q2 * s2;
                float w3 = (float)q3 * s3;

                float weights[4] = {w0, w1, w2, w3};
                float32x4_t weightVec = vld1q_f32(weights);

                acc = vfmaq_f32(acc, inputVal, weightVec);
            }

            // Apply GELU
            float32x4_t xs = vmulq_f32(acc, v_inv_sqrt2);

            uint32x4_t is_negative = vcltq_f32(xs, v_zero);
            float32x4_t abs_xs = vabsq_f32(xs);

            float32x4_t t = vdivq_f32(v_one, vfmaq_f32(v_one, v_p, abs_xs));

            float32x4_t neg_xs2 = vnegq_f32(vmulq_f32(xs, xs));
            neg_xs2 = vmaxq_f32(neg_xs2, v_min_clamp);
            neg_xs2 = vminq_f32(neg_xs2, v_max_clamp);

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

            float32x4_t poly = v_a5;
            poly = vfmaq_f32(v_a4, poly, t);
            poly = vfmaq_f32(v_a3, poly, t);
            poly = vfmaq_f32(v_a2, poly, t);
            poly = vfmaq_f32(v_a1, poly, t);
            poly = vmulq_f32(poly, t);

            float32x4_t erf_abs = vsubq_f32(v_one, vmulq_f32(poly, exp_neg_xs2));
            float32x4_t erf_val = vbslq_f32(is_negative, vnegq_f32(erf_abs), erf_abs);

            float32x4_t one_plus_erf = vaddq_f32(v_one, erf_val);
            float32x4_t result = vmulq_f32(acc, vmulq_f32(v_half, one_plus_erf));

            vst1q_f32(outputRow + n, result);
        }
    }
}
