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

// QKV Linear Projection NEON implementation for ARM64
//
// Fused matmul + split + bias for QKV projection: x @ wQKV^T -> q, k, v
// Key win: writes directly to q/k/v outputs, avoiding temp buffer + scatter copy.
//
// Uses NEON FMA for dot-product accumulation with 4-wide vectorization.

#include <arm_neon.h>

// =============================================================================
// qkvlinear_neon_f32: Fused QKV projection for float32
// =============================================================================
//
// func qkvlinear_neon_f32(x, wqkv, biasq, biask, biasv, q, k, params unsafe.Pointer)
// params: [0]=v pointer (as long), [1]=batch, [2]=in, [3]=qd, [4]=kvd
void qkvlinear_neon_f32(float *x, float *wqkv, float *biasq, float *biask, float *biasv,
                          float *q, float *k, long *params) {
    float *v = (float *)params[0];
    long batch = params[1];
    long in = params[2];
    long qd = params[3];
    long kvd = params[4];

    for (long i = 0; i < batch; i++) {
        float *xRow = x + i * in;

        // Q outputs
        for (long j = 0; j < qd; j++) {
            float *wRow = wqkv + j * in;

            float32x4_t acc = vdupq_n_f32(0.0f);
            long p = 0;
            for (; p + 4 <= in; p += 4) {
                float32x4_t vx = vld1q_f32(xRow + p);
                float32x4_t vw = vld1q_f32(wRow + p);
                acc = vfmaq_f32(acc, vx, vw);
            }
            float sum = vaddvq_f32(acc);
            for (; p < in; p++) {
                sum += xRow[p] * wRow[p];
            }
            if (biasq) {
                sum += biasq[j];
            }
            q[i * qd + j] = sum;
        }

        // K outputs
        for (long j = 0; j < kvd; j++) {
            float *wRow = wqkv + (qd + j) * in;

            float32x4_t acc = vdupq_n_f32(0.0f);
            long p = 0;
            for (; p + 4 <= in; p += 4) {
                float32x4_t vx = vld1q_f32(xRow + p);
                float32x4_t vw = vld1q_f32(wRow + p);
                acc = vfmaq_f32(acc, vx, vw);
            }
            float sum = vaddvq_f32(acc);
            for (; p < in; p++) {
                sum += xRow[p] * wRow[p];
            }
            if (biask) {
                sum += biask[j];
            }
            k[i * kvd + j] = sum;
        }

        // V outputs
        for (long j = 0; j < kvd; j++) {
            float *wRow = wqkv + (qd + kvd + j) * in;

            float32x4_t acc = vdupq_n_f32(0.0f);
            long p = 0;
            for (; p + 4 <= in; p += 4) {
                float32x4_t vx = vld1q_f32(xRow + p);
                float32x4_t vw = vld1q_f32(wRow + p);
                acc = vfmaq_f32(acc, vx, vw);
            }
            float sum = vaddvq_f32(acc);
            for (; p < in; p++) {
                sum += xRow[p] * wRow[p];
            }
            if (biasv) {
                sum += biasv[j];
            }
            v[i * kvd + j] = sum;
        }
    }
}

// =============================================================================
// qkvlinear_neon_f64: Fused QKV projection for float64
// =============================================================================
//
// func qkvlinear_neon_f64(x, wqkv, biasq, biask, biasv, q, k, params unsafe.Pointer)
// params: [0]=v pointer (as long), [1]=batch, [2]=in, [3]=qd, [4]=kvd
void qkvlinear_neon_f64(double *x, double *wqkv, double *biasq, double *biask, double *biasv,
                          double *q, double *k, long *params) {
    double *v = (double *)params[0];
    long batch = params[1];
    long in = params[2];
    long qd = params[3];
    long kvd = params[4];

    for (long i = 0; i < batch; i++) {
        double *xRow = x + i * in;

        // Q outputs
        for (long j = 0; j < qd; j++) {
            double *wRow = wqkv + j * in;

            float64x2_t acc = vdupq_n_f64(0.0);
            long p = 0;
            for (; p + 2 <= in; p += 2) {
                float64x2_t vx = vld1q_f64(xRow + p);
                float64x2_t vw = vld1q_f64(wRow + p);
                acc = vfmaq_f64(acc, vx, vw);
            }
            double sum = vaddvq_f64(acc);
            for (; p < in; p++) {
                sum += xRow[p] * wRow[p];
            }
            if (biasq) {
                sum += biasq[j];
            }
            q[i * qd + j] = sum;
        }

        // K outputs
        for (long j = 0; j < kvd; j++) {
            double *wRow = wqkv + (qd + j) * in;

            float64x2_t acc = vdupq_n_f64(0.0);
            long p = 0;
            for (; p + 2 <= in; p += 2) {
                float64x2_t vx = vld1q_f64(xRow + p);
                float64x2_t vw = vld1q_f64(wRow + p);
                acc = vfmaq_f64(acc, vx, vw);
            }
            double sum = vaddvq_f64(acc);
            for (; p < in; p++) {
                sum += xRow[p] * wRow[p];
            }
            if (biask) {
                sum += biask[j];
            }
            k[i * kvd + j] = sum;
        }

        // V outputs
        for (long j = 0; j < kvd; j++) {
            double *wRow = wqkv + (qd + kvd + j) * in;

            float64x2_t acc = vdupq_n_f64(0.0);
            long p = 0;
            for (; p + 2 <= in; p += 2) {
                float64x2_t vx = vld1q_f64(xRow + p);
                float64x2_t vw = vld1q_f64(wRow + p);
                acc = vfmaq_f64(acc, vx, vw);
            }
            double sum = vaddvq_f64(acc);
            for (; p < in; p++) {
                sum += xRow[p] * wRow[p];
            }
            if (biasv) {
                sum += biasv[j];
            }
            v[i * kvd + j] = sum;
        }
    }
}
