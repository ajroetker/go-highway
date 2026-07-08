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

//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides AVX-512 SIMD implementations of gather and scatter operations.
// These work directly with archsimd vector types.
// AVX-512 has hardware support for gather and scatter operations
// (VGATHERDPS, VGATHERQPD, VSCATTERDPS, etc.)
// but archsimd may not expose these directly, so we use store/scalar/load pattern.

// GatherIndex_AVX512_F32x16 gathers float32 elements using int32 indices.
func GatherIndex_AVX512_F32x16(src []float32, indices archsimd.Int32x16) archsimd.Float32x16 {
	var idxData [16]int32
	indices.StoreArray(&idxData)

	var result [16]float32
	for i := 0; i < 16; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(src) {
			result[i] = src[idx]
		}
	}
	return archsimd.LoadFloat32x16(result[:])
}

// GatherIndex_AVX512_F64x8 gathers float64 elements using int64 indices.
func GatherIndex_AVX512_F64x8(src []float64, indices archsimd.Int64x8) archsimd.Float64x8 {
	var idxData [8]int64
	indices.StoreArray(&idxData)

	var result [8]float64
	for i := 0; i < 8; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(src) {
			result[i] = src[idx]
		}
	}
	return archsimd.LoadFloat64x8(result[:])
}

// GatherIndex_AVX512_I32x16 gathers int32 elements using int32 indices.
func GatherIndex_AVX512_I32x16(src []int32, indices archsimd.Int32x16) archsimd.Int32x16 {
	var idxData [16]int32
	indices.StoreArray(&idxData)

	var result [16]int32
	for i := 0; i < 16; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(src) {
			result[i] = src[idx]
		}
	}
	return archsimd.LoadInt32x16(result[:])
}

// GatherIndex_AVX512_I64x8 gathers int64 elements using int64 indices.
func GatherIndex_AVX512_I64x8(src []int64, indices archsimd.Int64x8) archsimd.Int64x8 {
	var idxData [8]int64
	indices.StoreArray(&idxData)

	var result [8]int64
	for i := 0; i < 8; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(src) {
			result[i] = src[idx]
		}
	}
	return archsimd.LoadInt64x8(result[:])
}

// GatherIndexMasked_AVX512_F32x16 gathers float32 elements with mask.
func GatherIndexMasked_AVX512_F32x16(src []float32, indices archsimd.Int32x16, mask archsimd.Int32x16) archsimd.Float32x16 {
	var idxData [16]int32
	indices.StoreArray(&idxData)

	var maskData [16]int32
	mask.StoreArray(&maskData)

	var result [16]float32
	for i := 0; i < 16; i++ {
		if maskData[i] != 0 {
			idx := int(idxData[i])
			if idx >= 0 && idx < len(src) {
				result[i] = src[idx]
			}
		}
	}
	return archsimd.LoadFloat32x16(result[:])
}

// GatherIndexMasked_AVX512_F64x8 gathers float64 elements with mask.
func GatherIndexMasked_AVX512_F64x8(src []float64, indices archsimd.Int64x8, mask archsimd.Int64x8) archsimd.Float64x8 {
	var idxData [8]int64
	indices.StoreArray(&idxData)

	var maskData [8]int64
	mask.StoreArray(&maskData)

	var result [8]float64
	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			idx := int(idxData[i])
			if idx >= 0 && idx < len(src) {
				result[i] = src[idx]
			}
		}
	}
	return archsimd.LoadFloat64x8(result[:])
}

// ScatterIndex_AVX512_F32x16 scatters float32 elements to indices.
func ScatterIndex_AVX512_F32x16(v archsimd.Float32x16, dst []float32, indices archsimd.Int32x16) {
	var data [16]float32
	v.StoreArray(&data)

	var idxData [16]int32
	indices.StoreArray(&idxData)

	for i := 0; i < 16; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(dst) {
			dst[idx] = data[i]
		}
	}
}

// ScatterIndex_AVX512_F64x8 scatters float64 elements to indices.
func ScatterIndex_AVX512_F64x8(v archsimd.Float64x8, dst []float64, indices archsimd.Int64x8) {
	var data [8]float64
	v.StoreArray(&data)

	var idxData [8]int64
	indices.StoreArray(&idxData)

	for i := 0; i < 8; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(dst) {
			dst[idx] = data[i]
		}
	}
}

// ScatterIndex_AVX512_I32x16 scatters int32 elements to indices.
func ScatterIndex_AVX512_I32x16(v archsimd.Int32x16, dst []int32, indices archsimd.Int32x16) {
	var data [16]int32
	v.StoreArray(&data)

	var idxData [16]int32
	indices.StoreArray(&idxData)

	for i := 0; i < 16; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(dst) {
			dst[idx] = data[i]
		}
	}
}

// ScatterIndex_AVX512_I64x8 scatters int64 elements to indices.
func ScatterIndex_AVX512_I64x8(v archsimd.Int64x8, dst []int64, indices archsimd.Int64x8) {
	var data [8]int64
	v.StoreArray(&data)

	var idxData [8]int64
	indices.StoreArray(&idxData)

	for i := 0; i < 8; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(dst) {
			dst[idx] = data[i]
		}
	}
}

// ScatterIndexMasked_AVX512_F32x16 scatters float32 elements with mask.
func ScatterIndexMasked_AVX512_F32x16(v archsimd.Float32x16, dst []float32, indices archsimd.Int32x16, mask archsimd.Int32x16) {
	var data [16]float32
	v.StoreArray(&data)

	var idxData [16]int32
	indices.StoreArray(&idxData)

	var maskData [16]int32
	mask.StoreArray(&maskData)

	for i := 0; i < 16; i++ {
		if maskData[i] != 0 {
			idx := int(idxData[i])
			if idx >= 0 && idx < len(dst) {
				dst[idx] = data[i]
			}
		}
	}
}

// ScatterIndexMasked_AVX512_F64x8 scatters float64 elements with mask.
func ScatterIndexMasked_AVX512_F64x8(v archsimd.Float64x8, dst []float64, indices archsimd.Int64x8, mask archsimd.Int64x8) {
	var data [8]float64
	v.StoreArray(&data)

	var idxData [8]int64
	indices.StoreArray(&idxData)

	var maskData [8]int64
	mask.StoreArray(&maskData)

	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			idx := int(idxData[i])
			if idx >= 0 && idx < len(dst) {
				dst[idx] = data[i]
			}
		}
	}
}

// IndicesIota_AVX512_I32x16 creates indices [0, 1, 2, ..., 15].
func IndicesIota_AVX512_I32x16() archsimd.Int32x16 {
	return archsimd.LoadInt32x16([]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
}

// IndicesIota_AVX512_I64x8 creates indices [0, 1, 2, 3, 4, 5, 6, 7].
func IndicesIota_AVX512_I64x8() archsimd.Int64x8 {
	return archsimd.LoadInt64x8([]int64{0, 1, 2, 3, 4, 5, 6, 7})
}

// IndicesStride_AVX512_I32x16 creates indices [start, start+stride, start+2*stride, ...].
func IndicesStride_AVX512_I32x16(start, stride int32) archsimd.Int32x16 {
	var result [16]int32
	for i := 0; i < 16; i++ {
		result[i] = start + int32(i)*stride
	}
	return archsimd.LoadInt32x16(result[:])
}

// IndicesStride_AVX512_I64x8 creates indices [start, start+stride, ...].
func IndicesStride_AVX512_I64x8(start, stride int64) archsimd.Int64x8 {
	var result [8]int64
	for i := 0; i < 8; i++ {
		result[i] = start + int64(i)*stride
	}
	return archsimd.LoadInt64x8(result[:])
}
