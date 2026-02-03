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

// NEON Fused Int8 Dequantization + Matrix Multiplication for ARM64
// Compile with: -march=armv8-a+simd
//
// Performs fused dequantization and matmul in a single pass:
//   output[m,n] = sum_k(input[m,k] * (weights[k,n] * scale[k,groupIdx]))
//
// Int8: 8-bit signed integer quantization with per-group scales

#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// =============================================================================
// fused_int8_matmul_neon: Fused Int8 dequant + matmul using NEON
// =============================================================================
// Computes output = input @ dequant(weights, scales)
//
// Parameters:
//   input:     [M, K] float32 input matrix (row-major)
//   weights:   [K, N] int8 quantized weights (row-major)
//   scales:    [K, numGroups] float32 per-row, per-group scales
//   output:    [M, N] float32 output matrix (row-major)
//   M, K, N:   matrix dimensions
//   groupSize: number of columns per scale group
//
// func fused_int8_matmul_neon(input, weights, scales, output unsafe.Pointer,
//                              M, K, N, groupSize, numGroups *int64)
void fused_int8_matmul_neon(float *input, signed char *weights, float *scales,
                             float *output, long *pM, long *pK, long *pN,
                             long *pGroupSize, long *pNumGroups) {
    long M = *pM;
    long K = *pK;
    long N = *pN;
    long groupSize = *pGroupSize;
    long numGroups = *pNumGroups;

    // Process each output row
    for (long m = 0; m < M; m++) {
        float *inputRow = input + m * K;
        float *outputRow = output + m * N;

        // Process output columns in chunks of 4 (NEON f32 vector width)
        for (long n = 0; n < N; n += 4) {
            // Initialize accumulator
            float32x4_t acc = vdupq_n_f32(0.0f);

            // Accumulate over K dimension
            for (long k = 0; k < K; k++) {
                // Broadcast input[m, k]
                float32x4_t inputVal = vdupq_n_f32(inputRow[k]);

                // Load and dequantize 4 int8 weights from weights[k, n:n+4]
                long weightBase = k * N + n;
                signed char w0 = weights[weightBase + 0];
                signed char w1 = weights[weightBase + 1];
                signed char w2 = weights[weightBase + 2];
                signed char w3 = weights[weightBase + 3];

                // Convert int8 to float32
                float fw0 = (float)w0;
                float fw1 = (float)w1;
                float fw2 = (float)w2;
                float fw3 = (float)w3;

                // Get scales for each column's group
                long g0 = (n + 0) / groupSize;
                long g1 = (n + 1) / groupSize;
                long g2 = (n + 2) / groupSize;
                long g3 = (n + 3) / groupSize;

                float s0 = scales[k * numGroups + g0];
                float s1 = scales[k * numGroups + g1];
                float s2 = scales[k * numGroups + g2];
                float s3 = scales[k * numGroups + g3];

                // Apply scales: dequant = int8_val * scale
                fw0 *= s0;
                fw1 *= s1;
                fw2 *= s2;
                fw3 *= s3;

                // Create weight vector and accumulate
                float weightArr[4] = {fw0, fw1, fw2, fw3};
                float32x4_t weightVec = vld1q_f32(weightArr);

                acc = vfmaq_f32(acc, inputVal, weightVec);
            }

            // Store result
            vst1q_f32(outputRow + n, acc);
        }
    }
}
