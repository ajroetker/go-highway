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

// QKV Linear Projection SME implementation for ARM64
//
// Uses SME FMOPA outer product accumulate to compute x @ wQKV^T in 16x16 tiles,
// then stores directly to separate q, k, v output buffers with bias add.
//
// This avoids the temporary buffer entirely: the FMOPA tile store is split
// across q/k/v segments on the fly.

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

// =============================================================================
// qkvlinear_fmopa_f32: SME FMOPA-based fused QKV projection for float32
// =============================================================================
//
// Computes x @ wQKV^T and stores to q, k, v with bias add.
// x is [batch, in], wQKV is [totalOut, in] (row-major), need transposed access.
// xt is [in, batch] (pre-transposed x for contiguous column access).
//
// func qkvlinear_fmopa_f32(xt, wqkv, biasq, biask, biasv, q, k, params unsafe.Pointer)
// params: [0]=v pointer (as long), [1]=batch, [2]=in, [3]=qd, [4]=kvd
void qkvlinear_fmopa_f32(float *xt, float *wqkv, float *biasq, float *biask, float *biasv,
                            float *q, float *k, long *params)
    __arm_streaming __arm_out("za") {
    float *v = (float *)params[0];
    long batch = params[1];
    long in = params[2];
    long qd = params[3];
    long kvd = params[4];
    long totalOut = qd + 2 * kvd;

    // Process output in 16x16 tiles
    // Rows = batch dimension, Cols = totalOut dimension
    for (long ti = 0; ti < batch; ti += 16) {
        for (long tj = 0; tj < totalOut; tj += 16) {
            // Zero accumulator tile
            svzero_za();

            // Accumulate over in dimension
            for (long kk = 0; kk < in; kk++) {
                // Load x column: xt[kk, ti:ti+16] (contiguous in transposed layout)
                svfloat32_t za_col = svld1_f32(svptrue_b32(), xt + kk * batch + ti);

                // Load wQKV row: wqkv[tj:tj+16, kk] — need column access
                // Since wQKV is [totalOut, in], row kk of the "transposed" view is
                // wqkv[tj+0..15, kk] which is strided. Instead, we treat wQKV as
                // the B matrix: wqkv is [totalOut, in], and we want column kk.
                // Column kk = wqkv[0*in+kk, 1*in+kk, ...] — strided, not contiguous.
                // Better: expect wQKV transposed as wqkvt [in, totalOut] for FMOPA.
                svfloat32_t zb_row = svld1_f32(svptrue_b32(), wqkv + kk * totalOut + tj);

                // Outer product accumulate
                svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), za_col, zb_row);
            }

            // Store result tile: C[ti:ti+16, tj:tj+16]
            // Split across q/k/v based on tj offset
            for (int row = 0; row < 16; row++) {
                long batchIdx = ti + row;
                if (batchIdx >= batch) break;

                svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, row);

                // Store each element of the tile row to the correct q/k/v buffer
                // For simplicity, store to a temp row then scatter
                float tile_row[16];
                svst1_f32(svptrue_b32(), tile_row, zrow);

                for (int col = 0; col < 16; col++) {
                    long outIdx = tj + col;
                    if (outIdx >= totalOut) break;

                    float val = tile_row[col];

                    if (outIdx < qd) {
                        if (biasq) {
                            val += biasq[outIdx];
                        }
                        q[batchIdx * qd + outIdx] = val;
                    }
                    if (outIdx >= qd) {
                        if (outIdx < qd + kvd) {
                            long kIdx = outIdx - qd;
                            if (biask) {
                                val += biask[kIdx];
                            }
                            k[batchIdx * kvd + kIdx] = val;
                        }
                    }
                    if (outIdx >= qd + kvd) {
                        long vIdx = outIdx - qd - kvd;
                        if (biasv) {
                            val += biasv[vIdx];
                        }
                        v[batchIdx * kvd + vIdx] = val;
                    }
                }
            }
        }
    }
}

// =============================================================================
// qkvlinear_fmopa_f64: SME FMOPA-based fused QKV projection for float64
// =============================================================================
//
// Same algorithm with 8x8 tiles for float64.
//
// func qkvlinear_fmopa_f64(xt, wqkv, biasq, biask, biasv, q, k, params unsafe.Pointer)
// params: [0]=v pointer (as long), [1]=batch, [2]=in, [3]=qd, [4]=kvd
void qkvlinear_fmopa_f64(double *xt, double *wqkv, double *biasq, double *biask, double *biasv,
                            double *q, double *k, long *params)
    __arm_streaming __arm_out("za") {
    double *v = (double *)params[0];
    long batch = params[1];
    long in = params[2];
    long qd = params[3];
    long kvd = params[4];
    long totalOut = qd + 2 * kvd;

    for (long ti = 0; ti < batch; ti += 8) {
        for (long tj = 0; tj < totalOut; tj += 8) {
            svzero_za();

            for (long kk = 0; kk < in; kk++) {
                svfloat64_t za_col = svld1_f64(svptrue_b64(), xt + kk * batch + ti);
                svfloat64_t zb_row = svld1_f64(svptrue_b64(), wqkv + kk * totalOut + tj);
                svmopa_za64_f64_m(0, svptrue_b64(), svptrue_b64(), za_col, zb_row);
            }

            for (int row = 0; row < 8; row++) {
                long batchIdx = ti + row;
                if (batchIdx >= batch) break;

                svfloat64_t zrow = svread_hor_za64_f64_m(svundef_f64(), svptrue_b64(), 0, row);

                double tile_row[8];
                svst1_f64(svptrue_b64(), tile_row, zrow);

                for (int col = 0; col < 8; col++) {
                    long outIdx = tj + col;
                    if (outIdx >= totalOut) break;

                    double val = tile_row[col];

                    if (outIdx < qd) {
                        if (biasq) {
                            val += biasq[outIdx];
                        }
                        q[batchIdx * qd + outIdx] = val;
                    }
                    if (outIdx >= qd) {
                        if (outIdx < qd + kvd) {
                            long kIdx = outIdx - qd;
                            if (biask) {
                                val += biask[kIdx];
                            }
                            k[batchIdx * kvd + kIdx] = val;
                        }
                    }
                    if (outIdx >= qd + kvd) {
                        long vIdx = outIdx - qd - kvd;
                        if (biasv) {
                            val += biasv[vIdx];
                        }
                        v[batchIdx * kvd + vIdx] = val;
                    }
                }
            }
        }
    }
}
