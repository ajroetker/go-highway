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

// SME Flash Attention for ARM64
//
// Tiled Flash Attention using FMOPA with online softmax.
// Avoids materializing the full [seqLen, kvLen] scores matrix.
// Memory: O(seqLen * headDim) instead of O(seqLen * kvLen).
//
// Algorithm (FlashAttention-2 style):
//   For each Q tile (block of TILE_M=16 rows):
//     Initialize O = 0, l = 0, m = -inf per row
//     For each K/V tile (TILE_N=16 columns):
//       S_tile = Q_tile @ K_tile^T (via FMOPA)
//       Scale S_tile, add mask
//       m_new = max(m_prev, rowmax(S_tile))
//       alpha = exp(m_prev - m_new)
//       l_new = alpha * l_prev + rowsum(exp(S_tile - m_new))
//       O = alpha * O + exp(S_tile - m_new) @ V_tile
//     O /= l_new
//
// NEON intrinsics cannot be used inside __arm_streaming functions.
// All non-FMOPA operations use scalar C or SVE intrinsics.

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

// =============================================================================
// sdpa_fmopa_f32: SME Flash Attention for float32
// =============================================================================
//
// Q is [seqLen, headDim], K is [kvLen, headDim], V is [kvLen, headDim]
// kt is [headDim, kvLen] (pre-transposed K for FMOPA column access)
// mask is [seqLen, kvLen] or NULL
// output is [seqLen, headDim]
//
// func sdpa_fmopa_f32(q, kt, v, mask, output, pdims, pscale unsafe.Pointer)
// pdims: [0]=seqLen, [1]=kvLen, [2]=headDim
void sdpa_fmopa_f32(float *q, float *kt, float *v, float *mask,
                      float *output,
                      long *pdims, float *pscale)
    __arm_streaming __arm_out("za") {
    long seqLen = pdims[0];
    long kvLen = pdims[1];
    long headDim = pdims[2];
    float scale = *pscale;

    if (seqLen <= 0) return;
    if (kvLen <= 0) return;
    if (headDim <= 0) return;

    // Scalar exp f32 constants
    float inv_ln2 = 1.44269504088896341f;
    float ln2_hi = 0.693359375f;
    float ln2_lo = -2.12194440e-4f;

    float negInfVal = -1.0f / 0.0f;

    // Process Q in blocks of 16 rows
    for (long qi = 0; qi < seqLen; qi += 16) {
        long qRows = 16;
        if (qi + qRows > seqLen) {
            qRows = seqLen - qi;
        }

        // Per-row running max (m) and sum (l) for online softmax
        float m_arr[16];
        float l_arr[16];
        for (int r = 0; r < 16; r++) {
            m_arr[r] = negInfVal;
            l_arr[r] = 0.0f;
        }

        // Zero output accumulator for this Q block
        for (long r = 0; r < qRows; r++) {
            for (long d = 0; d < headDim; d++) {
                output[(qi + r) * headDim + d] = 0.0f;
            }
        }

        // Iterate over K/V in blocks of 16 columns
        for (long kj = 0; kj < kvLen; kj += 16) {
            long kCols = 16;
            if (kj + kCols > kvLen) {
                kCols = kvLen - kj;
            }

            // Compute S_tile = Q_block @ K_block^T using FMOPA
            svzero_za();

            for (long dd = 0; dd < headDim; dd++) {
                // Load Q column: q[qi:qi+16, dd] — strided by headDim
                float q_col[16];
                for (int r = 0; r < 16; r++) {
                    if (qi + r < seqLen) {
                        q_col[r] = q[(qi + r) * headDim + dd];
                    }
                    if (qi + r >= seqLen) {
                        q_col[r] = 0.0f;
                    }
                }
                svfloat32_t za_col = svld1_f32(svptrue_b32(), q_col);

                // Load K^T row: kt[dd, kj:kj+16] — contiguous
                svfloat32_t zb_row = svld1_f32(svptrue_b32(), kt + dd * kvLen + kj);

                svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), za_col, zb_row);
            }

            // Read S_tile from ZA, apply scale + mask, online softmax update
            for (int row = 0; row < 16; row++) {
                if (qi + row >= seqLen) break;

                svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, row);
                float s_row[16];
                svst1_f32(svptrue_b32(), s_row, zrow);

                // Scale + mask, find new max
                float row_max = m_arr[row];
                for (int col = 0; col < 16; col++) {
                    if (kj + col >= kvLen) {
                        s_row[col] = negInfVal;
                        continue;
                    }
                    s_row[col] *= scale;
                    if (mask) {
                        s_row[col] += mask[(qi + row) * kvLen + kj + col];
                    }
                    if (s_row[col] > row_max) {
                        row_max = s_row[col];
                    }
                }

                // Online softmax: correction factor
                float m_prev = m_arr[row];
                float m_new = row_max;
                m_arr[row] = m_new;

                // alpha = exp(m_prev - m_new) using scalar Horner polynomial
                float alpha_scalar = 1.0f;
                if (m_prev != negInfVal) {
                    float ax = m_prev - m_new;
                    if (ax < -87.3365f) ax = -87.3365f;
                    float akf = ax * inv_ln2;
                    int aki = (int)(akf + (akf >= 0 ? 0.5f : -0.5f));
                    float akff = (float)aki;
                    float ar = ax - akff * ln2_hi;
                    ar = ar - akff * ln2_lo;
                    float ap = 0.001388888888888889f;
                    ap = 0.008333333333333333f + ap * ar;
                    ap = 0.041666666666666664f + ap * ar;
                    ap = 0.16666666666666666f + ap * ar;
                    ap = 0.5f + ap * ar;
                    ap = 1.0f + ap * ar;
                    ap = 1.0f + ap * ar;
                    int a_scale_bits = (aki + 127) << 23;
                    float a_scale_val = *(float *)&a_scale_bits;
                    alpha_scalar = ap * a_scale_val;
                }

                // Rescale previous l and O
                l_arr[row] = alpha_scalar * l_arr[row];
                long oOff = (qi + row) * headDim;
                for (long p = 0; p < headDim; p++) {
                    output[oOff + p] *= alpha_scalar;
                }

                // Compute exp(s_row - m_new) and accumulate
                float p_row[16];
                float row_sum = 0.0f;
                for (int col = 0; col < 16; col++) {
                    if (kj + col >= kvLen) {
                        p_row[col] = 0.0f;
                        continue;
                    }
                    float sx = s_row[col] - m_new;
                    if (sx < -87.3365f) sx = -87.3365f;
                    // Scalar exp(sx) using Horner polynomial
                    float skf = sx * inv_ln2;
                    int ski = (int)(skf + (skf >= 0 ? 0.5f : -0.5f));
                    float skff = (float)ski;
                    float sr = sx - skff * ln2_hi;
                    sr = sr - skff * ln2_lo;
                    float sp = 0.001388888888888889f;
                    sp = 0.008333333333333333f + sp * sr;
                    sp = 0.041666666666666664f + sp * sr;
                    sp = 0.16666666666666666f + sp * sr;
                    sp = 0.5f + sp * sr;
                    sp = 1.0f + sp * sr;
                    sp = 1.0f + sp * sr;
                    int s_scale_bits = (ski + 127) << 23;
                    float s_scale_val = *(float *)&s_scale_bits;
                    p_row[col] = sp * s_scale_val;
                    row_sum += p_row[col];
                }
                l_arr[row] += row_sum;

                // Accumulate: O[row,:] += p_row @ V[kj:kj+kCols, :]
                for (int col = 0; col < 16; col++) {
                    if (kj + col >= kvLen) break;
                    if (p_row[col] == 0.0f) continue;

                    float w = p_row[col];
                    float *vRow = v + (kj + col) * headDim;
                    for (long p = 0; p < headDim; p++) {
                        output[oOff + p] += w * vRow[p];
                    }
                }
            }
        }

        // Final normalize: O /= l
        for (long r = 0; r < qRows; r++) {
            if (l_arr[r] == 0.0f) continue;
            float invL = 1.0f / l_arr[r];
            long oOff = (qi + r) * headDim;
            for (long p = 0; p < headDim; p++) {
                output[oOff + p] *= invL;
            }
        }
    }
}

// =============================================================================
// sdpa_fmopa_f64: SME Flash Attention for float64
// =============================================================================
//
// Same algorithm with 8x8 tiles for float64.
//
// func sdpa_fmopa_f64(q, kt, v, mask, output, pdims, pscale unsafe.Pointer)
// pdims: [0]=seqLen, [1]=kvLen, [2]=headDim
void sdpa_fmopa_f64(double *q, double *kt, double *v, double *mask,
                      double *output,
                      long *pdims, double *pscale)
    __arm_streaming __arm_out("za") {
    long seqLen = pdims[0];
    long kvLen = pdims[1];
    long headDim = pdims[2];
    double scale = *pscale;

    if (seqLen <= 0) return;
    if (kvLen <= 0) return;
    if (headDim <= 0) return;

    // Scalar exp f64 constants
    double inv_ln2_d = 1.4426950408889634;
    double ln2_hi_d = 0.6931471803691238;
    double ln2_lo_d = 1.9082149292705877e-10;

    double negInfVal = -1.0 / 0.0;

    for (long qi = 0; qi < seqLen; qi += 8) {
        long qRows = 8;
        if (qi + qRows > seqLen) {
            qRows = seqLen - qi;
        }

        double m_arr[8];
        double l_arr[8];
        for (int r = 0; r < 8; r++) {
            m_arr[r] = negInfVal;
            l_arr[r] = 0.0;
        }

        for (long r = 0; r < qRows; r++) {
            for (long d = 0; d < headDim; d++) {
                output[(qi + r) * headDim + d] = 0.0;
            }
        }

        for (long kj = 0; kj < kvLen; kj += 8) {
            long kCols = 8;
            if (kj + kCols > kvLen) {
                kCols = kvLen - kj;
            }

            svzero_za();

            for (long dd = 0; dd < headDim; dd++) {
                double q_col[8];
                for (int r = 0; r < 8; r++) {
                    if (qi + r < seqLen) {
                        q_col[r] = q[(qi + r) * headDim + dd];
                    }
                    if (qi + r >= seqLen) {
                        q_col[r] = 0.0;
                    }
                }
                svfloat64_t za_col = svld1_f64(svptrue_b64(), q_col);
                svfloat64_t zb_row = svld1_f64(svptrue_b64(), kt + dd * kvLen + kj);
                svmopa_za64_f64_m(0, svptrue_b64(), svptrue_b64(), za_col, zb_row);
            }

            for (int row = 0; row < 8; row++) {
                if (qi + row >= seqLen) break;

                svfloat64_t zrow = svread_hor_za64_f64_m(svundef_f64(), svptrue_b64(), 0, row);
                double s_row[8];
                svst1_f64(svptrue_b64(), s_row, zrow);

                double row_max = m_arr[row];
                for (int col = 0; col < 8; col++) {
                    if (kj + col >= kvLen) {
                        s_row[col] = negInfVal;
                        continue;
                    }
                    s_row[col] *= scale;
                    if (mask) {
                        s_row[col] += mask[(qi + row) * kvLen + kj + col];
                    }
                    if (s_row[col] > row_max) {
                        row_max = s_row[col];
                    }
                }

                double m_prev = m_arr[row];
                double m_new = row_max;
                m_arr[row] = m_new;

                // alpha = exp(m_prev - m_new) using scalar Horner polynomial
                double alpha_scalar = 1.0;
                if (m_prev != negInfVal) {
                    double ax = m_prev - m_new;
                    if (ax < -708.396) ax = -708.396;
                    double akf = ax * inv_ln2_d;
                    long aki = (long)(akf + (akf >= 0 ? 0.5 : -0.5));
                    double akff = (double)aki;
                    double ar = ax - akff * ln2_hi_d;
                    ar = ar - akff * ln2_lo_d;
                    double ap = 2.48015873015873015873e-5;
                    ap = 1.98412698412698412698e-4 + ap * ar;
                    ap = 1.38888888888888888889e-3 + ap * ar;
                    ap = 8.33333333333333333333e-3 + ap * ar;
                    ap = 4.16666666666666666667e-2 + ap * ar;
                    ap = 1.66666666666666666667e-1 + ap * ar;
                    ap = 0.5 + ap * ar;
                    ap = 1.0 + ap * ar;
                    ap = 1.0 + ap * ar;
                    long a_scale_bits = (aki + 1023) << 52;
                    double a_scale_val = *(double *)&a_scale_bits;
                    alpha_scalar = ap * a_scale_val;
                }

                l_arr[row] = alpha_scalar * l_arr[row];
                long oOff = (qi + row) * headDim;
                for (long p = 0; p < headDim; p++) {
                    output[oOff + p] *= alpha_scalar;
                }

                double p_row[8];
                double row_sum = 0.0;
                for (int col = 0; col < 8; col++) {
                    if (kj + col >= kvLen) {
                        p_row[col] = 0.0;
                        continue;
                    }
                    double sx = s_row[col] - m_new;
                    if (sx < -708.396) sx = -708.396;
                    double skf = sx * inv_ln2_d;
                    long ski = (long)(skf + (skf >= 0 ? 0.5 : -0.5));
                    double skff = (double)ski;
                    double sr = sx - skff * ln2_hi_d;
                    sr = sr - skff * ln2_lo_d;
                    double sp = 2.48015873015873015873e-5;
                    sp = 1.98412698412698412698e-4 + sp * sr;
                    sp = 1.38888888888888888889e-3 + sp * sr;
                    sp = 8.33333333333333333333e-3 + sp * sr;
                    sp = 4.16666666666666666667e-2 + sp * sr;
                    sp = 1.66666666666666666667e-1 + sp * sr;
                    sp = 0.5 + sp * sr;
                    sp = 1.0 + sp * sr;
                    sp = 1.0 + sp * sr;
                    long s_scale_bits = (ski + 1023) << 52;
                    double s_scale_val = *(double *)&s_scale_bits;
                    p_row[col] = sp * s_scale_val;
                    row_sum += p_row[col];
                }
                l_arr[row] += row_sum;

                for (int col = 0; col < 8; col++) {
                    if (kj + col >= kvLen) break;
                    if (p_row[col] == 0.0) continue;

                    double w = p_row[col];
                    double *vRow = v + (kj + col) * headDim;
                    for (long p = 0; p < headDim; p++) {
                        output[oOff + p] += w * vRow[p];
                    }
                }
            }
        }

        // Final normalize
        for (long r = 0; r < qRows; r++) {
            if (l_arr[r] == 0.0) continue;
            double invL = 1.0 / l_arr[r];
            long oOff = (qi + r) * headDim;
            for (long p = 0; p < headDim; p++) {
                output[oOff + p] *= invL;
            }
        }
    }
}
