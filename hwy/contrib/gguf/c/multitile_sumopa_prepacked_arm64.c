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

// 4-Tile SUMOPA kernel for pre-packed GGUF K-quant weights (unsigned types).
// Processes one sub-block for a 16M × 64N output block using ZA0-ZA3.
//
// Used for: Q4_K (kGroups=8), Q5_K (kGroups=8), Q2_K (kGroups=4).
//
// 1×4 tile layout:
//     cols 0-15   cols 16-31  cols 32-47  cols 48-63
//       ZA0         ZA1         ZA2         ZA3
//
// B panels are pre-packed with 4 tiles interleaved per k4 group:
//   bPanels[k4 * 256 + tile * 64 + col * 4 + g]
// This enables svld1_vnum_u8 group loads (4 × 64 bytes contiguous).
//
// A panel is pre-packed in Go outside streaming mode:
//   aPanel[k4 * 64 + row * 4 + g]
//
// Output: contiguous buffer of 1024 int32s (4 tiles of 16×16).
//   Tile layout: tiles[vnum*16 + col] where vnum in [0,64).
//   ZA0: vnum 0-15, ZA1: vnum 16-31, ZA2: vnum 32-47, ZA3: vnum 48-63.
//
// Compile with: -march=armv9-a+sme+sme-i16i64

#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

// func multitile_sumopa_prepacked(aPanel, bPanels, tiles unsafe.Pointer, kGroups int64)
void multitile_sumopa_prepacked(signed char * restrict aPanel,
                                unsigned char * restrict bPanels,
                                int * restrict tiles,
                                long kGroups)
    __arm_streaming __arm_out("za") {

    svbool_t pg8 = svptrue_b8();
    svbool_t pg32 = svptrue_b32();

    svzero_za();

    for (long k4 = 0; k4 < kGroups; k4++) {
        svint8_t av = svld1_s8(pg8, aPanel + k4 * 64);

        unsigned char *bBase = bPanels + k4 * 256;
        svuint8_t b0 = svld1_vnum_u8(pg8, bBase, 0);
        svuint8_t b1 = svld1_vnum_u8(pg8, bBase, 1);
        svuint8_t b2 = svld1_vnum_u8(pg8, bBase, 2);
        svuint8_t b3 = svld1_vnum_u8(pg8, bBase, 3);

        svsumopa_za32_s8_m(0, pg8, pg8, av, b0);
        svsumopa_za32_s8_m(1, pg8, pg8, av, b1);
        svsumopa_za32_s8_m(2, pg8, pg8, av, b2);
        svsumopa_za32_s8_m(3, pg8, pg8, av, b3);
    }

    // Read out all 4 ZA tiles using svst1_vnum_s32 to avoid address arithmetic.
    // VL for int32 at SVL=512 is 16 elements (64 bytes), so vnum N stores at
    // tiles + N*16 int32s. ZA0→vnum 0-15, ZA1→16-31, ZA2→32-47, ZA3→48-63.
    for (int row = 0; row < 16; row++) {
        svst1_vnum_s32(pg32, tiles, row,
            svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row));
    }
    for (int row = 0; row < 16; row++) {
        svst1_vnum_s32(pg32, tiles, 16 + row,
            svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row));
    }
    for (int row = 0; row < 16; row++) {
        svst1_vnum_s32(pg32, tiles, 32 + row,
            svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row));
    }
    for (int row = 0; row < 16; row++) {
        svst1_vnum_s32(pg32, tiles, 48 + row,
            svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row));
    }
}
