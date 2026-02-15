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

// SME UMOPA Single-Tile Kernel for go-highway
// Compile with: -march=armv9-a+sme+sme-i16i64
//
// Computes a single 16×16 int32 output tile using UMOPA (uint8×uint8→int32).
//
// UMOPA groups uint8 inputs into 16 groups of 4:
//   ZA[i][j] += sum_{g=0..3} av[i*4+g] * bv[j*4+g]
//
// Input panels must be pre-packed in interleaved format:
//   panel[k4 * 64 + lane * 4 + g] = value for (tile_lane, k_index=k4*4+g)
//
// Output is a 16×16 int32 tile with stride 16.
// Assumes SVL = 512 bits (Apple M4).

#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

// tile_umopa_u8: single 16×16 UMOPA tile kernel
//
// func tile_umopa_u8(aPanel, bPanel unsafe.Pointer, c unsafe.Pointer, kGroups int64)
void tile_umopa_u8(unsigned char * restrict aPanel, unsigned char * restrict bPanel,
                   int * restrict c, long kGroups)
    __arm_streaming __arm_out("za") {

    svbool_t pg8 = svptrue_b8();
    svbool_t pg32 = svptrue_b32();

    svzero_za();

    for (long k4 = 0; k4 < kGroups; k4++) {
        svuint8_t av = svld1_u8(pg8, aPanel + k4 * 64);
        svuint8_t bv = svld1_u8(pg8, bPanel + k4 * 64);
        svmopa_za32_u8_m(0, pg8, pg8, av, bv);
    }

    // Extract ZA0 rows to output (16×16 tile, stride 16)
    int *c_ptr = c;
    for (int row = 0; row < 16; row++) {
        svint32_t za_row = svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row);
        svst1_s32(pg32, c_ptr, za_row);
        c_ptr += 16;
    }
}
