//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides low-level AVX2 SIMD operations that work directly with
// archsimd vector types. These are core ops (direct hardware instructions),
// not transcendental functions which belong in contrib/math.
//
// These functions are used by contrib/algo transforms and can be used directly
// by users who want to work with raw SIMD types instead of the Vec abstraction.

// Sqrt_AVX2_F32x8 computes sqrt(x) for a single Float32x8 vector.
// Uses the hardware VSQRTPS instruction which provides correctly rounded results.
func Sqrt_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	return x.Sqrt()
}

// Sqrt_AVX2_F64x4 computes sqrt(x) for a single Float64x4 vector.
// Uses the hardware VSQRTPD instruction which provides correctly rounded results.
func Sqrt_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	return x.Sqrt()
}

// ReduceMax_AVX2_Uint32x8 returns the maximum element in the vector.
func ReduceMax_AVX2_Uint32x8(v archsimd.Uint32x8) uint32 {
	// Reduce 8 -> 4 -> 2 -> 1
	lo := v.GetLo()
	hi := v.GetHi()
	max4 := lo.Max(hi)
	// Now max4 is Uint32x4, reduce further
	e0 := max4.GetElem(0)
	e1 := max4.GetElem(1)
	e2 := max4.GetElem(2)
	e3 := max4.GetElem(3)
	m := e0
	if e1 > m {
		m = e1
	}
	if e2 > m {
		m = e2
	}
	if e3 > m {
		m = e3
	}
	return m
}

// ReduceMax_AVX2_Uint64x4 returns the maximum element in the vector.
func ReduceMax_AVX2_Uint64x4(v archsimd.Uint64x4) uint64 {
	// AVX2 doesn't have VPMAXUQ (unsigned 64-bit max), so extract all elements
	// and compare scalarly
	lo := v.GetLo()
	hi := v.GetHi()
	e0 := lo.GetElem(0)
	e1 := lo.GetElem(1)
	e2 := hi.GetElem(0)
	e3 := hi.GetElem(1)
	m := e0
	if e1 > m {
		m = e1
	}
	if e2 > m {
		m = e2
	}
	if e3 > m {
		m = e3
	}
	return m
}

// Max_AVX2_Uint64x4 returns the element-wise maximum of two Uint64x4 vectors.
// AVX2 doesn't have VPMAXUQ (unsigned 64-bit max), so we use scalar comparison.
func Max_AVX2_Uint64x4(a, b archsimd.Uint64x4) archsimd.Uint64x4 {
	var result [4]uint64
	aLo, aHi := a.GetLo(), a.GetHi()
	bLo, bHi := b.GetLo(), b.GetHi()
	a0, a1 := aLo.GetElem(0), aLo.GetElem(1)
	a2, a3 := aHi.GetElem(0), aHi.GetElem(1)
	b0, b1 := bLo.GetElem(0), bLo.GetElem(1)
	b2, b3 := bHi.GetElem(0), bHi.GetElem(1)
	if a0 > b0 {
		result[0] = a0
	} else {
		result[0] = b0
	}
	if a1 > b1 {
		result[1] = a1
	} else {
		result[1] = b1
	}
	if a2 > b2 {
		result[2] = a2
	} else {
		result[2] = b2
	}
	if a3 > b3 {
		result[3] = a3
	} else {
		result[3] = b3
	}
	return archsimd.LoadUint64x4Slice(result[:])
}

// Max_AVX2_Int64x4 returns the element-wise maximum of two Int64x4 vectors.
// AVX2 doesn't have VPMAXSQ (signed 64-bit max), so we use scalar comparison.
func Max_AVX2_Int64x4(a, b archsimd.Int64x4) archsimd.Int64x4 {
	var result [4]int64
	aLo, aHi := a.GetLo(), a.GetHi()
	bLo, bHi := b.GetLo(), b.GetHi()
	a0, a1 := aLo.GetElem(0), aLo.GetElem(1)
	a2, a3 := aHi.GetElem(0), aHi.GetElem(1)
	b0, b1 := bLo.GetElem(0), bLo.GetElem(1)
	b2, b3 := bHi.GetElem(0), bHi.GetElem(1)
	if a0 > b0 {
		result[0] = a0
	} else {
		result[0] = b0
	}
	if a1 > b1 {
		result[1] = a1
	} else {
		result[1] = b1
	}
	if a2 > b2 {
		result[2] = a2
	} else {
		result[2] = b2
	}
	if a3 > b3 {
		result[3] = a3
	} else {
		result[3] = b3
	}
	return archsimd.LoadInt64x4Slice(result[:])
}

// Min_AVX2_Uint64x4 returns the element-wise minimum of two Uint64x4 vectors.
// AVX2 doesn't have VPMINUQ (unsigned 64-bit min), so we use scalar comparison.
func Min_AVX2_Uint64x4(a, b archsimd.Uint64x4) archsimd.Uint64x4 {
	var result [4]uint64
	aLo, aHi := a.GetLo(), a.GetHi()
	bLo, bHi := b.GetLo(), b.GetHi()
	a0, a1 := aLo.GetElem(0), aLo.GetElem(1)
	a2, a3 := aHi.GetElem(0), aHi.GetElem(1)
	b0, b1 := bLo.GetElem(0), bLo.GetElem(1)
	b2, b3 := bHi.GetElem(0), bHi.GetElem(1)
	if a0 < b0 {
		result[0] = a0
	} else {
		result[0] = b0
	}
	if a1 < b1 {
		result[1] = a1
	} else {
		result[1] = b1
	}
	if a2 < b2 {
		result[2] = a2
	} else {
		result[2] = b2
	}
	if a3 < b3 {
		result[3] = a3
	} else {
		result[3] = b3
	}
	return archsimd.LoadUint64x4Slice(result[:])
}

// Min_AVX2_Int64x4 returns the element-wise minimum of two Int64x4 vectors.
// AVX2 doesn't have VPMINSQ (signed 64-bit min), so we use scalar comparison.
func Min_AVX2_Int64x4(a, b archsimd.Int64x4) archsimd.Int64x4 {
	var result [4]int64
	aLo, aHi := a.GetLo(), a.GetHi()
	bLo, bHi := b.GetLo(), b.GetHi()
	a0, a1 := aLo.GetElem(0), aLo.GetElem(1)
	a2, a3 := aHi.GetElem(0), aHi.GetElem(1)
	b0, b1 := bLo.GetElem(0), bLo.GetElem(1)
	b2, b3 := bHi.GetElem(0), bHi.GetElem(1)
	if a0 < b0 {
		result[0] = a0
	} else {
		result[0] = b0
	}
	if a1 < b1 {
		result[1] = a1
	} else {
		result[1] = b1
	}
	if a2 < b2 {
		result[2] = a2
	} else {
		result[2] = b2
	}
	if a3 < b3 {
		result[3] = a3
	} else {
		result[3] = b3
	}
	return archsimd.LoadInt64x4Slice(result[:])
}

// GetLane_AVX2_Uint32x8 extracts the element at the given lane index.
func GetLane_AVX2_Uint32x8(v archsimd.Uint32x8, lane int) uint32 {
	if lane < 4 {
		return v.GetLo().GetElem(uint8(lane))
	}
	return v.GetHi().GetElem(uint8(lane - 4))
}

// GetLane_AVX2_Uint64x4 extracts the element at the given lane index.
func GetLane_AVX2_Uint64x4(v archsimd.Uint64x4, lane int) uint64 {
	if lane < 2 {
		return v.GetLo().GetElem(uint8(lane))
	}
	return v.GetHi().GetElem(uint8(lane - 2))
}
