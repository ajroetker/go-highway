//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides AVX2 SIMD implementations of shuffle operations.
// These work directly with archsimd vector types.
// Operations without direct archsimd support use store/scalar/load pattern.

// Reverse_AVX2_F32x8 reverses all lanes.
// [0,1,2,3,4,5,6,7] -> [7,6,5,4,3,2,1,0]
func Reverse_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])
	result := [8]float32{data[7], data[6], data[5], data[4], data[3], data[2], data[1], data[0]}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// Reverse_AVX2_F64x4 reverses all lanes.
// [0,1,2,3] -> [3,2,1,0]
func Reverse_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])
	result := [4]float64{data[3], data[2], data[1], data[0]}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// Reverse2_AVX2_F32x8 reverses pairs of lanes.
// [0,1,2,3,4,5,6,7] -> [1,0,3,2,5,4,7,6]
func Reverse2_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])
	data[0], data[1] = data[1], data[0]
	data[2], data[3] = data[3], data[2]
	data[4], data[5] = data[5], data[4]
	data[6], data[7] = data[7], data[6]
	return archsimd.LoadFloat32x8Slice(data[:])
}

// Reverse2_AVX2_F64x4 reverses pairs of lanes.
// [0,1,2,3] -> [1,0,3,2]
func Reverse2_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])
	data[0], data[1] = data[1], data[0]
	data[2], data[3] = data[3], data[2]
	return archsimd.LoadFloat64x4Slice(data[:])
}

// Reverse4_AVX2_F32x8 reverses groups of 4 lanes.
// [0,1,2,3,4,5,6,7] -> [3,2,1,0,7,6,5,4]
func Reverse4_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])
	// Reverse first group of 4
	data[0], data[1], data[2], data[3] = data[3], data[2], data[1], data[0]
	// Reverse second group of 4
	data[4], data[5], data[6], data[7] = data[7], data[6], data[5], data[4]
	return archsimd.LoadFloat32x8Slice(data[:])
}

// Reverse4_AVX2_F64x4 reverses all 4 lanes (same as Reverse for F64x4).
func Reverse4_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	return Reverse_AVX2_F64x4(v)
}

// GetLane_AVX2_F32x8 extracts a single lane value.
func GetLane_AVX2_F32x8(v archsimd.Float32x8, idx int) float32 {
	if idx < 0 || idx >= 8 {
		return 0
	}
	var data [8]float32
	v.StoreSlice(data[:])
	return data[idx]
}

// GetLane_AVX2_F64x4 extracts a single lane value.
func GetLane_AVX2_F64x4(v archsimd.Float64x4, idx int) float64 {
	if idx < 0 || idx >= 4 {
		return 0
	}
	var data [4]float64
	v.StoreSlice(data[:])
	return data[idx]
}

// InsertLane_AVX2_F32x8 inserts a value at the given lane.
func InsertLane_AVX2_F32x8(v archsimd.Float32x8, idx int, val float32) archsimd.Float32x8 {
	if idx < 0 || idx >= 8 {
		return v
	}
	var data [8]float32
	v.StoreSlice(data[:])
	data[idx] = val
	return archsimd.LoadFloat32x8Slice(data[:])
}

// InsertLane_AVX2_F64x4 inserts a value at the given lane.
func InsertLane_AVX2_F64x4(v archsimd.Float64x4, idx int, val float64) archsimd.Float64x4 {
	if idx < 0 || idx >= 4 {
		return v
	}
	var data [4]float64
	v.StoreSlice(data[:])
	data[idx] = val
	return archsimd.LoadFloat64x4Slice(data[:])
}

// InterleaveLower_AVX2_F32x8 interleaves lower halves.
// [a0,a1,a2,a3,a4,a5,a6,a7], [b0,b1,b2,b3,b4,b5,b6,b7] -> [a0,b0,a1,b1,a2,b2,a3,b3]
func InterleaveLower_AVX2_F32x8(a, b archsimd.Float32x8) archsimd.Float32x8 {
	var dataA, dataB [8]float32
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	result := [8]float32{
		dataA[0], dataB[0], dataA[1], dataB[1],
		dataA[2], dataB[2], dataA[3], dataB[3],
	}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// InterleaveLower_AVX2_F64x4 interleaves lower halves.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a0,b0,a1,b1]
func InterleaveLower_AVX2_F64x4(a, b archsimd.Float64x4) archsimd.Float64x4 {
	var dataA, dataB [4]float64
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	result := [4]float64{dataA[0], dataB[0], dataA[1], dataB[1]}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// InterleaveUpper_AVX2_F32x8 interleaves upper halves.
// [a0,a1,a2,a3,a4,a5,a6,a7], [b0,b1,b2,b3,b4,b5,b6,b7] -> [a4,b4,a5,b5,a6,b6,a7,b7]
func InterleaveUpper_AVX2_F32x8(a, b archsimd.Float32x8) archsimd.Float32x8 {
	var dataA, dataB [8]float32
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	result := [8]float32{
		dataA[4], dataB[4], dataA[5], dataB[5],
		dataA[6], dataB[6], dataA[7], dataB[7],
	}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// InterleaveUpper_AVX2_F64x4 interleaves upper halves.
// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a2,b2,a3,b3]
func InterleaveUpper_AVX2_F64x4(a, b archsimd.Float64x4) archsimd.Float64x4 {
	var dataA, dataB [4]float64
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	result := [4]float64{dataA[2], dataB[2], dataA[3], dataB[3]}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// ConcatLowerLower_AVX2_F32x8 concatenates lower halves.
func ConcatLowerLower_AVX2_F32x8(a, b archsimd.Float32x8) archsimd.Float32x8 {
	var dataA, dataB [8]float32
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	result := [8]float32{
		dataA[0], dataA[1], dataA[2], dataA[3],
		dataB[0], dataB[1], dataB[2], dataB[3],
	}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// ConcatLowerLower_AVX2_F64x4 concatenates lower halves.
func ConcatLowerLower_AVX2_F64x4(a, b archsimd.Float64x4) archsimd.Float64x4 {
	var dataA, dataB [4]float64
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	result := [4]float64{dataA[0], dataA[1], dataB[0], dataB[1]}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// OddEven_AVX2_F32x8 combines odd lanes from a with even lanes from b.
// [a0,a1,a2,a3,a4,a5,a6,a7], [b0,b1,b2,b3,b4,b5,b6,b7] -> [b0,a1,b2,a3,b4,a5,b6,a7]
func OddEven_AVX2_F32x8(a, b archsimd.Float32x8) archsimd.Float32x8 {
	var dataA, dataB [8]float32
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	result := [8]float32{
		dataB[0], dataA[1], dataB[2], dataA[3],
		dataB[4], dataA[5], dataB[6], dataA[7],
	}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// OddEven_AVX2_F64x4 combines odd lanes from a with even lanes from b.
func OddEven_AVX2_F64x4(a, b archsimd.Float64x4) archsimd.Float64x4 {
	var dataA, dataB [4]float64
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	result := [4]float64{dataB[0], dataA[1], dataB[2], dataA[3]}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// DupEven_AVX2_F32x8 duplicates even lanes.
// [a0,a1,a2,a3,a4,a5,a6,a7] -> [a0,a0,a2,a2,a4,a4,a6,a6]
func DupEven_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])
	result := [8]float32{
		data[0], data[0], data[2], data[2],
		data[4], data[4], data[6], data[6],
	}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// DupEven_AVX2_F64x4 duplicates even lanes.
// [a0,a1,a2,a3] -> [a0,a0,a2,a2]
func DupEven_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])
	result := [4]float64{data[0], data[0], data[2], data[2]}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// DupOdd_AVX2_F32x8 duplicates odd lanes.
// [a0,a1,a2,a3,a4,a5,a6,a7] -> [a1,a1,a3,a3,a5,a5,a7,a7]
func DupOdd_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])
	result := [8]float32{
		data[1], data[1], data[3], data[3],
		data[5], data[5], data[7], data[7],
	}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// DupOdd_AVX2_F64x4 duplicates odd lanes.
// [a0,a1,a2,a3] -> [a1,a1,a3,a3]
func DupOdd_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])
	result := [4]float64{data[1], data[1], data[3], data[3]}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// SwapAdjacentBlocks_AVX2_F32x8 swaps the two 128-bit halves.
// [a0,a1,a2,a3,a4,a5,a6,a7] -> [a4,a5,a6,a7,a0,a1,a2,a3]
func SwapAdjacentBlocks_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])
	result := [8]float32{
		data[4], data[5], data[6], data[7],
		data[0], data[1], data[2], data[3],
	}
	return archsimd.LoadFloat32x8Slice(result[:])
}

// SwapAdjacentBlocks_AVX2_F64x4 swaps the two 128-bit halves.
// [a0,a1,a2,a3] -> [a2,a3,a0,a1]
func SwapAdjacentBlocks_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])
	result := [4]float64{data[2], data[3], data[0], data[1]}
	return archsimd.LoadFloat64x4Slice(result[:])
}

// Slide operations

// SlideUpLanes_AVX2_F32x8 shifts lanes up by offset, filling low lanes with zeros.
// [1,2,3,4,5,6,7,8] with offset=2 -> [0,0,1,2,3,4,5,6]
func SlideUpLanes_AVX2_F32x8(v archsimd.Float32x8, offset int) archsimd.Float32x8 {
	if offset <= 0 {
		return v
	}
	if offset >= 8 {
		return archsimd.Float32x8{}
	}
	var data [8]float32
	v.StoreSlice(data[:])
	var result [8]float32
	copy(result[offset:], data[:8-offset])
	return archsimd.LoadFloat32x8Slice(result[:])
}

// SlideUpLanes_AVX2_F64x4 shifts lanes up by offset, filling low lanes with zeros.
func SlideUpLanes_AVX2_F64x4(v archsimd.Float64x4, offset int) archsimd.Float64x4 {
	if offset <= 0 {
		return v
	}
	if offset >= 4 {
		return archsimd.Float64x4{}
	}
	var data [4]float64
	v.StoreSlice(data[:])
	var result [4]float64
	copy(result[offset:], data[:4-offset])
	return archsimd.LoadFloat64x4Slice(result[:])
}

// SlideDownLanes_AVX2_F32x8 shifts lanes down by offset, filling high lanes with zeros.
// [1,2,3,4,5,6,7,8] with offset=2 -> [3,4,5,6,7,8,0,0]
func SlideDownLanes_AVX2_F32x8(v archsimd.Float32x8, offset int) archsimd.Float32x8 {
	if offset <= 0 {
		return v
	}
	if offset >= 8 {
		return archsimd.Float32x8{}
	}
	var data [8]float32
	v.StoreSlice(data[:])
	var result [8]float32
	copy(result[:8-offset], data[offset:])
	return archsimd.LoadFloat32x8Slice(result[:])
}

// SlideDownLanes_AVX2_F64x4 shifts lanes down by offset, filling high lanes with zeros.
func SlideDownLanes_AVX2_F64x4(v archsimd.Float64x4, offset int) archsimd.Float64x4 {
	if offset <= 0 {
		return v
	}
	if offset >= 4 {
		return archsimd.Float64x4{}
	}
	var data [4]float64
	v.StoreSlice(data[:])
	var result [4]float64
	copy(result[:4-offset], data[offset:])
	return archsimd.LoadFloat64x4Slice(result[:])
}

// Slide1Up_AVX2_F32x8 shifts lanes up by 1, filling first lane with zero.
// [1,2,3,4,5,6,7,8] -> [0,1,2,3,4,5,6,7]
func Slide1Up_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	return SlideUpLanes_AVX2_F32x8(v, 1)
}

// Slide1Up_AVX2_F64x4 shifts lanes up by 1, filling first lane with zero.
func Slide1Up_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	return SlideUpLanes_AVX2_F64x4(v, 1)
}

// Slide1Down_AVX2_F32x8 shifts lanes down by 1, filling last lane with zero.
// [1,2,3,4,5,6,7,8] -> [2,3,4,5,6,7,8,0]
func Slide1Down_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	return SlideDownLanes_AVX2_F32x8(v, 1)
}

// Slide1Down_AVX2_F64x4 shifts lanes down by 1, filling last lane with zero.
func Slide1Down_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	return SlideDownLanes_AVX2_F64x4(v, 1)
}
