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

// 4-Tile SMOPA kernel for pre-packed GGUF K-quant weights (signed types).
// Processes one sub-block for a 16M × 64N output block using ZA0-ZA3.
//
// Used for: Q6_K (kGroups=4), Q3_K (kGroups=4).
//
// Same layout as the unsigned kernel, but uses SMOPA (signed × signed)
// instead of SUMOPA (signed × unsigned).
//
// Compile with: -march=armv9-a+sme+sme-i16i64

#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

// func multitile_smopa_prepacked(aPanel, bPanels, tiles unsafe.Pointer, kGroups int64)
void multitile_smopa_prepacked(signed char * restrict aPanel,
                               signed char * restrict bPanels,
                               int * restrict tiles,
                               long kGroups)
    __arm_streaming __arm_out("za") {

    svbool_t pg8 = svptrue_b8();
    svbool_t pg32 = svptrue_b32();

    svzero_za();

    for (long k4 = 0; k4 < kGroups; k4++) {
        // Load A vector (shared across all 4 tiles).
        svint8_t av = svld1_s8(pg8, aPanel + k4 * 64);

        // 4-register group load of pre-packed B panels (signed).
        signed char *bBase = bPanels + k4 * 256;
        svint8_t b0 = svld1_vnum_s8(pg8, bBase, 0);
        svint8_t b1 = svld1_vnum_s8(pg8, bBase, 1);
        svint8_t b2 = svld1_vnum_s8(pg8, bBase, 2);
        svint8_t b3 = svld1_vnum_s8(pg8, bBase, 3);

        // 4 SMOPA outer products (signed × signed).
        svmopa_za32_s8_m(0, pg8, pg8, av, b0);
        svmopa_za32_s8_m(1, pg8, pg8, av, b1);
        svmopa_za32_s8_m(2, pg8, pg8, av, b2);
        svmopa_za32_s8_m(3, pg8, pg8, av, b3);
    }

    // Read out all 4 ZA tiles using svst1_vnum_s32 to avoid address arithmetic.
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
