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

// Fused float32 → int8 quantize + SMOPA A-panel pack kernels.
//
// Eliminates the intermediate Q8_K buffer by quantizing float32 input
// directly into A-panel layout. The Q8_K scale (absmax/127 over 256 values)
// is precomputed externally; these functions handle per-sub-block quantization
// and packing in a single pass.
//
// A-panel layout: aPanel[k4*64 + row*4 + g]
//   - k4 groups of 4 values
//   - 16 rows (M-tile, padded with zeros)
//   - 4 values per k4 group
//
// Compile with: -march=armv8.2-a+dotprod+simd+fp -O3

#ifndef GOAT_PARSER
#include <arm_neon.h>
#endif

// compute_absmax computes the maximum absolute value over n float32 values.
// n must be a multiple of 4 (always true for QK_K=256).
float compute_absmax(float *input, long n) {
    float32x4_t maxv = vdupq_n_f32(0.0f);
    for (long i = 0; i < n; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        maxv = vmaxq_f32(maxv, vabsq_f32(v));
    }
    return vmaxvq_f32(maxv);
}

// fused_quantize_pack quantizes subBlockSize float32 values per row and
// packs directly into SMOPA A-panel format.
//
// input: base pointer, input[row * inputStride + 0..subBlockSize-1]
// inputStride: row stride in float32 elements (= K, the full row width)
// invScale: invScale[row] = 127.0 / absmax, length mRows
// subBlockSize: 16 or 32 (must be multiple of 4)
// mRows: number of valid rows (1..16)
// aPanel: output buffer, kGroups*64 bytes where kGroups = subBlockSize/4
void fused_quantize_pack(
    float *input,
    long inputStride,
    float *invScale,
    long subBlockSize,
    long mRows,
    signed char *aPanel
) {
    long kGroups = subBlockSize / 4;
    float32x4_t minVec = vdupq_n_f32(-128.0f);
    float32x4_t maxVec = vdupq_n_f32(127.0f);

    for (long row = 0; row < mRows; row++) {
        float *rowPtr = input + row * inputStride;
        float32x4_t idVec = vdupq_n_f32(invScale[row]);

        for (long k4 = 0; k4 < kGroups; k4++) {
            float32x4_t v = vld1q_f32(rowPtr + k4 * 4);
            float32x4_t scaled = vmulq_f32(v, idVec);
            float32x4_t rounded = vrndnq_f32(scaled);
            float32x4_t clamped = vmaxq_f32(vminq_f32(rounded, maxVec), minVec);
            int32x4_t qi = vcvtq_s32_f32(clamped);

            // Store 4 int8 values into A-panel with stride 64.
            long off = k4 * 64 + row * 4;
            aPanel[off]     = (signed char)vgetq_lane_s32(qi, 0);
            aPanel[off + 1] = (signed char)vgetq_lane_s32(qi, 1);
            aPanel[off + 2] = (signed char)vgetq_lane_s32(qi, 2);
            aPanel[off + 3] = (signed char)vgetq_lane_s32(qi, 3);
        }
    }

    // Zero-fill unused rows (rows mRows..15).
    for (long row = mRows; row < 16; row++) {
        for (long k4 = 0; k4 < kGroups; k4++) {
            long off = k4 * 64 + row * 4;
            aPanel[off]     = 0;
            aPanel[off + 1] = 0;
            aPanel[off + 2] = 0;
            aPanel[off + 3] = 0;
        }
    }
}

// fused_quantize_pack_bsum is the same as fused_quantize_pack but also
// computes the sum of quantized int8 values per row (bsums) for unsigned
// K-quant accumulation correction.
//
// bsums: output, bsums[row] = sum of all quantized int8 values for that row's
//        sub-block. Length mRows.
void fused_quantize_pack_bsum(
    float *input,
    long inputStride,
    float *invScale,
    long subBlockSize,
    long mRows,
    signed char *aPanel,
    long *bsums
) {
    long kGroups = subBlockSize / 4;
    float32x4_t minVec = vdupq_n_f32(-128.0f);
    float32x4_t maxVec = vdupq_n_f32(127.0f);

    for (long row = 0; row < mRows; row++) {
        float *rowPtr = input + row * inputStride;
        float32x4_t idVec = vdupq_n_f32(invScale[row]);
        int32x4_t sumv = vdupq_n_s32(0);

        for (long k4 = 0; k4 < kGroups; k4++) {
            float32x4_t v = vld1q_f32(rowPtr + k4 * 4);
            float32x4_t scaled = vmulq_f32(v, idVec);
            float32x4_t rounded = vrndnq_f32(scaled);
            float32x4_t clamped = vmaxq_f32(vminq_f32(rounded, maxVec), minVec);
            int32x4_t qi = vcvtq_s32_f32(clamped);
            sumv = vaddq_s32(sumv, qi);

            long off = k4 * 64 + row * 4;
            aPanel[off]     = (signed char)vgetq_lane_s32(qi, 0);
            aPanel[off + 1] = (signed char)vgetq_lane_s32(qi, 1);
            aPanel[off + 2] = (signed char)vgetq_lane_s32(qi, 2);
            aPanel[off + 3] = (signed char)vgetq_lane_s32(qi, 3);
        }
        bsums[row] = (long)vaddvq_s32(sumv);
    }

    for (long row = mRows; row < 16; row++) {
        for (long k4 = 0; k4 < kGroups; k4++) {
            long off = k4 * 64 + row * 4;
            aPanel[off]     = 0;
            aPanel[off + 1] = 0;
            aPanel[off + 2] = 0;
            aPanel[off + 3] = 0;
        }
    }
}
