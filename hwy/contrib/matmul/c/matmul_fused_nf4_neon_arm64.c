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

// NEON Fused NF4/Int4 Dequantization + Matrix Multiplication for ARM64
// Compile with: -march=armv8-a+simd
//
// Performs fused dequantization and matmul in a single pass:
//   output[m,n] = sum_k(input[m,k] * dequant(packed[k,n]))
//
// NF4: 4-bit NormalFloat quantization with 16-entry lookup table
// Int4: 4-bit symmetric integer quantization (values 0-15 map to -8 to +7)

#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// NF4 lookup table - 16 fixed values for 4-bit NormalFloat quantization
// These are the optimal quantization points for normally distributed weights
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
// fused_nf4_matmul_neon: Fused NF4 dequant + matmul using NEON
// =============================================================================
// Computes output = input @ dequant(packed, scales)
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
// func fused_nf4_matmul_neon(input, packed, scales, output unsafe.Pointer,
//                            M, K, N, groupSize, numGroups *int64)
void fused_nf4_matmul_neon(float *input, unsigned char *packed, float *scales,
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

                // Dequantize 4 weights from packed[k, n:n+4]
                // Each pair of weights shares a byte
                long weightIdx0 = k * N + n;
                long weightIdx1 = k * N + n + 1;
                long weightIdx2 = k * N + n + 2;
                long weightIdx3 = k * N + n + 3;

                long packedIdx0 = weightIdx0 / 2;
                long packedIdx1 = weightIdx2 / 2;

                unsigned char byte0 = packed[packedIdx0];
                unsigned char byte1 = packed[packedIdx1];

                // Extract nibbles
                int q0 = byte0 & 0x0F;        // low nibble (even index)
                int q1 = (byte0 >> 4) & 0x0F; // high nibble (odd index)
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

            // Store result
            vst1q_f32(outputRow + n, acc);
        }
    }
}

// =============================================================================
// fused_int4_matmul_neon: Fused Int4 dequant + matmul using NEON
// =============================================================================
// Same as NF4 but uses symmetric integer quantization:
// Values 0-15 map to -8 to +7 (subtract 8)
//
// func fused_int4_matmul_neon(input, packed, scales, output unsafe.Pointer,
//                             M, K, N, groupSize, numGroups *int64)
void fused_int4_matmul_neon(float *input, unsigned char *packed, float *scales,
                            float *output, long *pM, long *pK, long *pN,
                            long *pGroupSize, long *pNumGroups) {
    long M = *pM;
    long K = *pK;
    long N = *pN;
    long groupSize = *pGroupSize;
    long numGroups = *pNumGroups;

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

            vst1q_f32(outputRow + n, acc);
        }
    }
}
