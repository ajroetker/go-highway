// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !noasm && arm64

// Pre-packed 4-tile SMOPA/SUMOPA kernel wrappers for GGUF K-quant matmul.
package asm

import "unsafe"

// MultiTileSUMOPAPrepacked calls the 4-tile SUMOPA kernel for unsigned types
// (Q4_K, Q5_K, Q2_K). Processes one sub-block producing 4 ZA tiles (16×16 each).
//
// Parameters:
//   - aPanel: pre-packed A panel, kGroups*64 bytes (signed int8)
//   - bPanels: pre-packed B panels, kGroups*256 bytes (unsigned uint8)
//   - tiles: output buffer, 1024 int32s (4 tiles of 16×16)
//   - kGroups: number of k4 groups in this sub-block (SubBlockSize/4)
func MultiTileSUMOPAPrepacked(aPanel, bPanels, tiles unsafe.Pointer, kGroups int64) {
	multitile_sumopa_prepacked(aPanel, bPanels, tiles, kGroups)
}

// MultiTileSMOPAPrepacked calls the 4-tile SMOPA kernel for signed types
// (Q6_K, Q3_K). Same interface as unsigned but uses signed×signed outer product.
//
// Parameters:
//   - aPanel: pre-packed A panel, kGroups*64 bytes (signed int8)
//   - bPanels: pre-packed B panels, kGroups*256 bytes (signed int8)
//   - tiles: output buffer, 1024 int32s (4 tiles of 16×16)
//   - kGroups: number of k4 groups in this sub-block (SubBlockSize/4)
func MultiTileSMOPAPrepacked(aPanel, bPanels, tiles unsafe.Pointer, kGroups int64) {
	multitile_smopa_prepacked(aPanel, bPanels, tiles, kGroups)
}
