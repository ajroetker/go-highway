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

// SDOT/TBL vecdot wrappers for GGUF Tier 1 quantized formats.
package asm

import "unsafe"

// VecDotQ8_0Q8_0_SDOT computes Q8_0 × Q8_0 dot product using integer SDOT.
func VecDotQ8_0Q8_0_SDOT(wdata, adata unsafe.Pointer, nblocks int64, result unsafe.Pointer) {
	vecdot_q8_0q8_0_sdot(wdata, adata, nblocks, result)
}

// VecDotQ4_0Q8_0_SDOT computes Q4_0 × Q8_0 dot product using integer SDOT.
func VecDotQ4_0Q8_0_SDOT(wdata, adata unsafe.Pointer, nblocks int64, result unsafe.Pointer) {
	vecdot_q4_0q8_0_sdot(wdata, adata, nblocks, result)
}

// VecDotIQ4NLQ8_0_SDOT computes IQ4_NL × Q8_0 dot product using SDOT+TBL.
func VecDotIQ4NLQ8_0_SDOT(wdata, adata unsafe.Pointer, nblocks int64, result unsafe.Pointer) {
	vecdot_iq4nlq8_0_sdot(wdata, adata, nblocks, result)
}
