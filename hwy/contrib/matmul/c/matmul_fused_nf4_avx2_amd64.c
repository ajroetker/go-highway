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

// AVX2 Fused NF4/Int4 Dequantization + Matrix Multiplication for AMD64
// Compile with: -mavx2 -mfma
//
// Performs fused dequantization and matmul in a single pass:
//   output[m,n] = sum_k(input[m,k] * dequant(packed[k,n]))
//
// NF4: 4-bit NormalFloat quantization with 16-entry lookup table
// Int4: 4-bit symmetric integer quantization (values 0-15 map to -8 to +7)

#ifndef GOAT_PARSER
#include <immintrin.h>
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
// fused_nf4_matmul_avx2: Fused NF4 dequant + matmul using AVX2
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
// func fused_nf4_matmul_avx2(input, packed, scales, output unsafe.Pointer,
//                            M, K, N, groupSize, numGroups *int64)
void fused_nf4_matmul_avx2(float *input, unsigned char *packed, float *scales,
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

        // Process output columns in chunks of 8 (AVX2 f32 vector width)
        for (long n = 0; n < N; n += 8) {
            // Initialize accumulator
            __m256 acc = _mm256_setzero_ps();

            // Accumulate over K dimension
            for (long k = 0; k < K; k++) {
                // Broadcast input[m, k]
                __m256 inputVal = _mm256_set1_ps(inputRow[k]);

                // Dequantize 8 weights from packed[k, n:n+8]
                // Process pairs of bytes (each byte has 2 nibbles)
                long baseIdx = k * N + n;

                // Load 4 bytes containing 8 nibbles
                unsigned char b0 = packed[(baseIdx + 0) / 2];
                unsigned char b1 = packed[(baseIdx + 2) / 2];
                unsigned char b2 = packed[(baseIdx + 4) / 2];
                unsigned char b3 = packed[(baseIdx + 6) / 2];

                // Extract nibbles (assuming baseIdx is even)
                int q0 = b0 & 0x0F;
                int q1 = (b0 >> 4) & 0x0F;
                int q2 = b1 & 0x0F;
                int q3 = (b1 >> 4) & 0x0F;
                int q4 = b2 & 0x0F;
                int q5 = (b2 >> 4) & 0x0F;
                int q6 = b3 & 0x0F;
                int q7 = (b3 >> 4) & 0x0F;

                // Table lookup for NF4 values
                float w0 = nf4_table[q0];
                float w1 = nf4_table[q1];
                float w2 = nf4_table[q2];
                float w3 = nf4_table[q3];
                float w4 = nf4_table[q4];
                float w5 = nf4_table[q5];
                float w6 = nf4_table[q6];
                float w7 = nf4_table[q7];

                // Get scales for each column's group
                long g0 = (n + 0) / groupSize;
                long g1 = (n + 1) / groupSize;
                long g2 = (n + 2) / groupSize;
                long g3 = (n + 3) / groupSize;
                long g4 = (n + 4) / groupSize;
                long g5 = (n + 5) / groupSize;
                long g6 = (n + 6) / groupSize;
                long g7 = (n + 7) / groupSize;

                float s0 = scales[k * numGroups + g0];
                float s1 = scales[k * numGroups + g1];
                float s2 = scales[k * numGroups + g2];
                float s3 = scales[k * numGroups + g3];
                float s4 = scales[k * numGroups + g4];
                float s5 = scales[k * numGroups + g5];
                float s6 = scales[k * numGroups + g6];
                float s7 = scales[k * numGroups + g7];

                // Apply scales and create weight vector
                __m256 weightVec = _mm256_set_ps(
                    w7 * s7, w6 * s6, w5 * s5, w4 * s4,
                    w3 * s3, w2 * s2, w1 * s1, w0 * s0
                );

                // FMA: acc += input * weight
                acc = _mm256_fmadd_ps(inputVal, weightVec, acc);
            }

            // Store result
            _mm256_storeu_ps(outputRow + n, acc);
        }
    }
}

// =============================================================================
// fused_int4_matmul_avx2: Fused Int4 dequant + matmul using AVX2
// =============================================================================
// Same as NF4 but uses symmetric integer quantization:
// Values 0-15 map to -8 to +7 (subtract 8)
//
// func fused_int4_matmul_avx2(input, packed, scales, output unsafe.Pointer,
//                             M, K, N, groupSize, numGroups *int64)
void fused_int4_matmul_avx2(float *input, unsigned char *packed, float *scales,
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

        for (long n = 0; n < N; n += 8) {
            __m256 acc = _mm256_setzero_ps();

            for (long k = 0; k < K; k++) {
                __m256 inputVal = _mm256_set1_ps(inputRow[k]);

                long baseIdx = k * N + n;

                unsigned char b0 = packed[(baseIdx + 0) / 2];
                unsigned char b1 = packed[(baseIdx + 2) / 2];
                unsigned char b2 = packed[(baseIdx + 4) / 2];
                unsigned char b3 = packed[(baseIdx + 6) / 2];

                // Extract nibbles and convert to signed [-8, 7]
                int q0 = (b0 & 0x0F) - 8;
                int q1 = ((b0 >> 4) & 0x0F) - 8;
                int q2 = (b1 & 0x0F) - 8;
                int q3 = ((b1 >> 4) & 0x0F) - 8;
                int q4 = (b2 & 0x0F) - 8;
                int q5 = ((b2 >> 4) & 0x0F) - 8;
                int q6 = (b3 & 0x0F) - 8;
                int q7 = ((b3 >> 4) & 0x0F) - 8;

                // Get scales
                long g0 = (n + 0) / groupSize;
                long g1 = (n + 1) / groupSize;
                long g2 = (n + 2) / groupSize;
                long g3 = (n + 3) / groupSize;
                long g4 = (n + 4) / groupSize;
                long g5 = (n + 5) / groupSize;
                long g6 = (n + 6) / groupSize;
                long g7 = (n + 7) / groupSize;

                float s0 = scales[k * numGroups + g0];
                float s1 = scales[k * numGroups + g1];
                float s2 = scales[k * numGroups + g2];
                float s3 = scales[k * numGroups + g3];
                float s4 = scales[k * numGroups + g4];
                float s5 = scales[k * numGroups + g5];
                float s6 = scales[k * numGroups + g6];
                float s7 = scales[k * numGroups + g7];

                // Dequantize and create weight vector
                __m256 weightVec = _mm256_set_ps(
                    (float)q7 * s7, (float)q6 * s6, (float)q5 * s5, (float)q4 * s4,
                    (float)q3 * s3, (float)q2 * s2, (float)q1 * s1, (float)q0 * s0
                );

                acc = _mm256_fmadd_ps(inputVal, weightVec, acc);
            }

            _mm256_storeu_ps(outputRow + n, acc);
        }
    }
}
