//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides low-level AVX-512 SIMD operations that work directly with
// archsimd vector types. These are core ops (direct hardware instructions),
// not transcendental functions which belong in contrib/math.

// Sqrt_AVX512_F32x16 computes sqrt(x) for a single Float32x16 vector.
// Uses the hardware VSQRTPS instruction which provides correctly rounded results.
func Sqrt_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	return x.Sqrt()
}

// Sqrt_AVX512_F64x8 computes sqrt(x) for a single Float64x8 vector.
// Uses the hardware VSQRTPD instruction which provides correctly rounded results.
func Sqrt_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	return x.Sqrt()
}

// ReduceMax_AVX512_Uint32x16 returns the maximum element in the vector.
func ReduceMax_AVX512_Uint32x16(v archsimd.Uint32x16) uint32 {
	// Reduce 16 -> 8 -> 4 -> scalar
	lo := v.GetLo()
	hi := v.GetHi()
	max8 := lo.Max(hi)
	return ReduceMax_AVX2_Uint32x8(max8)
}

// ReduceMax_AVX512_Uint64x8 returns the maximum element in the vector.
func ReduceMax_AVX512_Uint64x8(v archsimd.Uint64x8) uint64 {
	// Reduce 8 -> 4 -> 2 -> scalar
	lo := v.GetLo()
	hi := v.GetHi()
	max4 := lo.Max(hi)
	return ReduceMax_AVX2_Uint64x4(max4)
}

// GetLane_AVX512_Uint32x16 extracts the element at the given lane index.
func GetLane_AVX512_Uint32x16(v archsimd.Uint32x16, lane int) uint32 {
	// Uint32x16 -> GetLo/GetHi -> Uint32x8 -> GetLo/GetHi -> Uint32x4 -> GetElem
	if lane < 8 {
		return GetLane_AVX2_Uint32x8(v.GetLo(), lane)
	}
	return GetLane_AVX2_Uint32x8(v.GetHi(), lane-8)
}

// GetLane_AVX512_Uint64x8 extracts the element at the given lane index.
func GetLane_AVX512_Uint64x8(v archsimd.Uint64x8, lane int) uint64 {
	// Uint64x8 -> GetLo/GetHi -> Uint64x4 -> GetLo/GetHi -> Uint64x2 -> GetElem
	if lane < 4 {
		return GetLane_AVX2_Uint64x4(v.GetLo(), lane)
	}
	return GetLane_AVX2_Uint64x4(v.GetHi(), lane-4)
}
