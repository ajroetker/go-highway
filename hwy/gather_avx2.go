//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides AVX2 SIMD implementations of gather and scatter operations.
// These work directly with archsimd vector types.
// AVX2 has hardware support for gather (VGATHERDPS, VGATHERDPD, etc.)
// but archsimd may not expose these directly, so we use store/scalar/load pattern.

// GatherIndex_AVX2_F32x8 gathers float32 elements using int32 indices.
func GatherIndex_AVX2_F32x8(src []float32, indices archsimd.Int32x8) archsimd.Float32x8 {
	var idxData [8]int32
	indices.StoreSlice(idxData[:])

	var result [8]float32
	for i := 0; i < 8; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(src) {
			result[i] = src[idx]
		}
	}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// GatherIndex_AVX2_F64x4 gathers float64 elements using int64 indices.
func GatherIndex_AVX2_F64x4(src []float64, indices archsimd.Int64x4) archsimd.Float64x4 {
	var idxData [4]int64
	indices.StoreSlice(idxData[:])

	var result [4]float64
	for i := 0; i < 4; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(src) {
			result[i] = src[idx]
		}
	}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// GatherIndex_AVX2_I32x8 gathers int32 elements using int32 indices.
func GatherIndex_AVX2_I32x8(src []int32, indices archsimd.Int32x8) archsimd.Int32x8 {
	var idxData [8]int32
	indices.StoreSlice(idxData[:])

	var result [8]int32
	for i := 0; i < 8; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(src) {
			result[i] = src[idx]
		}
	}
	return archsimd.LoadInt32x8Slice(result[:])
}

// GatherIndex_AVX2_I64x4 gathers int64 elements using int64 indices.
func GatherIndex_AVX2_I64x4(src []int64, indices archsimd.Int64x4) archsimd.Int64x4 {
	var idxData [4]int64
	indices.StoreSlice(idxData[:])

	var result [4]int64
	for i := 0; i < 4; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(src) {
			result[i] = src[idx]
		}
	}
	return archsimd.LoadInt64x4Slice(result[:])
}

// GatherIndexMasked_AVX2_F32x8 gathers float32 elements with mask.
func GatherIndexMasked_AVX2_F32x8(src []float32, indices archsimd.Int32x8, mask archsimd.Int32x8) archsimd.Float32x8 {
	var idxData [8]int32
	indices.StoreSlice(idxData[:])

	var maskData [8]int32
	mask.StoreSlice(maskData[:])

	var result [8]float32
	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			idx := int(idxData[i])
			if idx >= 0 && idx < len(src) {
				result[i] = src[idx]
			}
		}
	}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// GatherIndexMasked_AVX2_F64x4 gathers float64 elements with mask.
func GatherIndexMasked_AVX2_F64x4(src []float64, indices archsimd.Int64x4, mask archsimd.Int64x4) archsimd.Float64x4 {
	var idxData [4]int64
	indices.StoreSlice(idxData[:])

	var maskData [4]int64
	mask.StoreSlice(maskData[:])

	var result [4]float64
	for i := 0; i < 4; i++ {
		if maskData[i] != 0 {
			idx := int(idxData[i])
			if idx >= 0 && idx < len(src) {
				result[i] = src[idx]
			}
		}
	}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// ScatterIndex_AVX2_F32x8 scatters float32 elements to indices.
// Note: AVX2 does not have hardware scatter, this is emulated.
func ScatterIndex_AVX2_F32x8(v archsimd.Float32x8, dst []float32, indices archsimd.Int32x8) {
	var data [8]float32
	v.StoreSlice(data[:])

	var idxData [8]int32
	indices.StoreSlice(idxData[:])

	for i := 0; i < 8; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(dst) {
			dst[idx] = data[i]
		}
	}
}

// ScatterIndex_AVX2_F64x4 scatters float64 elements to indices.
func ScatterIndex_AVX2_F64x4(v archsimd.Float64x4, dst []float64, indices archsimd.Int64x4) {
	var data [4]float64
	v.StoreSlice(data[:])

	var idxData [4]int64
	indices.StoreSlice(idxData[:])

	for i := 0; i < 4; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(dst) {
			dst[idx] = data[i]
		}
	}
}

// ScatterIndex_AVX2_I32x8 scatters int32 elements to indices.
func ScatterIndex_AVX2_I32x8(v archsimd.Int32x8, dst []int32, indices archsimd.Int32x8) {
	var data [8]int32
	v.StoreSlice(data[:])

	var idxData [8]int32
	indices.StoreSlice(idxData[:])

	for i := 0; i < 8; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(dst) {
			dst[idx] = data[i]
		}
	}
}

// ScatterIndex_AVX2_I64x4 scatters int64 elements to indices.
func ScatterIndex_AVX2_I64x4(v archsimd.Int64x4, dst []int64, indices archsimd.Int64x4) {
	var data [4]int64
	v.StoreSlice(data[:])

	var idxData [4]int64
	indices.StoreSlice(idxData[:])

	for i := 0; i < 4; i++ {
		idx := int(idxData[i])
		if idx >= 0 && idx < len(dst) {
			dst[idx] = data[i]
		}
	}
}

// ScatterIndexMasked_AVX2_F32x8 scatters float32 elements with mask.
func ScatterIndexMasked_AVX2_F32x8(v archsimd.Float32x8, dst []float32, indices archsimd.Int32x8, mask archsimd.Int32x8) {
	var data [8]float32
	v.StoreSlice(data[:])

	var idxData [8]int32
	indices.StoreSlice(idxData[:])

	var maskData [8]int32
	mask.StoreSlice(maskData[:])

	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			idx := int(idxData[i])
			if idx >= 0 && idx < len(dst) {
				dst[idx] = data[i]
			}
		}
	}
}

// ScatterIndexMasked_AVX2_F64x4 scatters float64 elements with mask.
func ScatterIndexMasked_AVX2_F64x4(v archsimd.Float64x4, dst []float64, indices archsimd.Int64x4, mask archsimd.Int64x4) {
	var data [4]float64
	v.StoreSlice(data[:])

	var idxData [4]int64
	indices.StoreSlice(idxData[:])

	var maskData [4]int64
	mask.StoreSlice(maskData[:])

	for i := 0; i < 4; i++ {
		if maskData[i] != 0 {
			idx := int(idxData[i])
			if idx >= 0 && idx < len(dst) {
				dst[idx] = data[i]
			}
		}
	}
}

// IndicesIota_AVX2_I32x8 creates indices [0, 1, 2, 3, 4, 5, 6, 7].
func IndicesIota_AVX2_I32x8() archsimd.Int32x8 {
	return archsimd.LoadInt32x8Slice([]int32{0, 1, 2, 3, 4, 5, 6, 7})
}

// IndicesIota_AVX2_I64x4 creates indices [0, 1, 2, 3].
func IndicesIota_AVX2_I64x4() archsimd.Int64x4 {
	return archsimd.LoadInt64x4Slice([]int64{0, 1, 2, 3})
}

// IndicesStride_AVX2_I32x8 creates indices [start, start+stride, start+2*stride, ...].
func IndicesStride_AVX2_I32x8(start, stride int32) archsimd.Int32x8 {
	result := [8]int32{
		start,
		start + stride,
		start + 2*stride,
		start + 3*stride,
		start + 4*stride,
		start + 5*stride,
		start + 6*stride,
		start + 7*stride,
	}
	return archsimd.LoadInt32x8Slice(result[:])
}

// IndicesStride_AVX2_I64x4 creates indices [start, start+stride, start+2*stride, start+3*stride].
func IndicesStride_AVX2_I64x4(start, stride int64) archsimd.Int64x4 {
	result := [4]int64{
		start,
		start + stride,
		start + 2*stride,
		start + 3*stride,
	}
	return archsimd.LoadInt64x4Slice(result[:])
}
