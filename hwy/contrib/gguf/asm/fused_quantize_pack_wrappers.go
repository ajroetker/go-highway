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

// Fused quantize+pack wrappers for SMOPA activation panel preparation.
package asm

import "unsafe"

// ComputeAbsmax returns the maximum absolute value over n float32 values.
// n must be a multiple of 4.
func ComputeAbsmax(input unsafe.Pointer, n int64) float32 {
	return compute_absmax(input, n)
}

// FusedQuantizePack quantizes subBlockSize float32 values per row using
// precomputed inverse scales, and packs directly into SMOPA A-panel layout.
//
// input points to the first row's sub-block data. Subsequent rows are at
// input + row*inputStride (in float32 elements).
// invScale[row] = 127.0 / absmax for each row.
// aPanel output: kGroups*64 bytes in A-panel layout (k4*64 + row*4 + g).
func FusedQuantizePack(input unsafe.Pointer, inputStride int64,
	invScale unsafe.Pointer, subBlockSize, mRows int64,
	aPanel unsafe.Pointer) {
	fused_quantize_pack(input, inputStride, invScale, subBlockSize, mRows, aPanel)
}

// FusedQuantizePackBsum is like FusedQuantizePack but also computes per-row
// bsums (sum of quantized int8 values) for unsigned K-quant accumulation.
// bsums[row] = sum of all quantized int8 values in that row's sub-block.
func FusedQuantizePackBsum(input unsafe.Pointer, inputStride int64,
	invScale unsafe.Pointer, subBlockSize, mRows int64,
	aPanel unsafe.Pointer, bsums unsafe.Pointer) {
	fused_quantize_pack_bsum(input, inputStride, invScale, subBlockSize, mRows, aPanel, bsums)
}
