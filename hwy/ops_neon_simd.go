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

//go:build arm64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides hwy wrapper functions for operations that Go's
// simd/archsimd package does not expose natively on arm64. They are emitted
// by hwygen for the archsimd-backed NEON variant (suffix _NEON_SIMD; see
// neonArchsimdOps in cmd/hwygen/targets.go).
//
// Where an operation needs an int32 companion vector for float64 lanes,
// archsimd's lack of sub-128-bit vectors means the companion is a full
// Int32x4 whose upper two lanes carry unspecified values; all uses are
// lane-wise, so those lanes are ignored.

// Merge_NEON_SIMD_F32x4 selects elements from a where mask is true, from b
// otherwise (hwy.Merge semantics via archsimd IfElse).
func Merge_NEON_SIMD_F32x4(a, b archsimd.Float32x4, mask archsimd.Mask32x4) archsimd.Float32x4 {
	return a.IfElse(mask, b)
}

// Merge_NEON_SIMD_F64x2 selects elements from a where mask is true, from b
// otherwise.
func Merge_NEON_SIMD_F64x2(a, b archsimd.Float64x2, mask archsimd.Mask64x2) archsimd.Float64x2 {
	return a.IfElse(mask, b)
}

// Merge_NEON_SIMD_I32x4 selects elements from a where mask is true, from b
// otherwise.
func Merge_NEON_SIMD_I32x4(a, b archsimd.Int32x4, mask archsimd.Mask32x4) archsimd.Int32x4 {
	return a.IfElse(mask, b)
}

// Merge_NEON_SIMD_I64x2 selects elements from a where mask is true, from b
// otherwise.
func Merge_NEON_SIMD_I64x2(a, b archsimd.Int64x2, mask archsimd.Mask64x2) archsimd.Int64x2 {
	return a.IfElse(mask, b)
}

// IfThenElse_NEON_SIMD_F32x4 is hwy.IfThenElse for archsimd NEON vectors.
func IfThenElse_NEON_SIMD_F32x4(mask archsimd.Mask32x4, a, b archsimd.Float32x4) archsimd.Float32x4 {
	return a.IfElse(mask, b)
}

// IfThenElse_NEON_SIMD_F64x2 is hwy.IfThenElse for archsimd NEON vectors.
func IfThenElse_NEON_SIMD_F64x2(mask archsimd.Mask64x2, a, b archsimd.Float64x2) archsimd.Float64x2 {
	return a.IfElse(mask, b)
}

// ConvertToInt32_NEON_SIMD_Float64x2 converts float64 lanes to int32
// (truncating). The result's lanes 0-1 hold the converted values; upper
// lanes are unspecified.
func ConvertToInt32_NEON_SIMD_Float64x2(v archsimd.Float64x2) archsimd.Int32x4 {
	return v.ConvertToInt64().SaturateToInt32()
}

// GetExponent_NEON_SIMD_F32x4 extracts the unbiased IEEE 754 exponent.
// Denormals and specials (Inf/NaN) produce 0, matching the asm and scalar
// implementations.
func GetExponent_NEON_SIMD_F32x4(v archsimd.Float32x4) archsimd.Int32x4 {
	bits := v.ToBits()
	rawExp := bits.ShiftAllRight(23).And(archsimd.BroadcastUint32x4(0xFF))
	unbiased := rawExp.BitsToInt32().Sub(archsimd.BroadcastInt32x4(127))
	isDenormal := rawExp.Equal(archsimd.BroadcastUint32x4(0))
	isSpecial := rawExp.Equal(archsimd.BroadcastUint32x4(0xFF))
	zero := archsimd.BroadcastInt32x4(0)
	return zero.IfElse(isDenormal.Or(isSpecial), unbiased)
}

// GetExponent_NEON_SIMD_F64x2 extracts the unbiased IEEE 754 exponent as
// int64 lanes (archsimd arm64 has no Int32x2; Int64x2 converts directly to
// Float64x2 downstream).
func GetExponent_NEON_SIMD_F64x2(v archsimd.Float64x2) archsimd.Int64x2 {
	bits := v.ToBits()
	rawExp := bits.ShiftAllRight(52).And(archsimd.BroadcastUint64x2(0x7FF))
	unbiased := rawExp.BitsToInt64().Sub(archsimd.BroadcastInt64x2(1023))
	isDenormal := rawExp.Equal(archsimd.BroadcastUint64x2(0))
	isSpecial := rawExp.Equal(archsimd.BroadcastUint64x2(0x7FF))
	zero := archsimd.BroadcastInt64x2(0)
	return zero.IfElse(isDenormal.Or(isSpecial), unbiased)
}

// GetMantissa_NEON_SIMD_F32x4 extracts the mantissa with the exponent
// normalized to [1, 2).
func GetMantissa_NEON_SIMD_F32x4(v archsimd.Float32x4) archsimd.Float32x4 {
	bits := v.ToBits()
	mantissa := bits.And(archsimd.BroadcastUint32x4(0x807FFFFF)).
		Or(archsimd.BroadcastUint32x4(0x3F800000))
	return mantissa.BitsToFloat32()
}

// GetMantissa_NEON_SIMD_F64x2 extracts the mantissa with the exponent
// normalized to [1, 2).
func GetMantissa_NEON_SIMD_F64x2(v archsimd.Float64x2) archsimd.Float64x2 {
	bits := v.ToBits()
	mantissa := bits.And(archsimd.BroadcastUint64x2(0x800FFFFFFFFFFFFF)).
		Or(archsimd.BroadcastUint64x2(0x3FF0000000000000))
	return mantissa.BitsToFloat64()
}

// Pow2_NEON_SIMD_F32x4 computes 2^k for each lane using IEEE 754 bit
// manipulation. Exponents outside the normal range produce unspecified
// values; callers clamp with overflow/underflow masks.
func Pow2_NEON_SIMD_F32x4(k archsimd.Int32x4) archsimd.Float32x4 {
	biased := k.Add(archsimd.BroadcastInt32x4(127))
	return biased.ShiftAllLeft(23).ToBits().BitsToFloat32()
}

// Pow2_NEON_SIMD_F64x2 computes 2^k for lanes 0-1 of an int32 companion
// vector (see ConvertToInt32_NEON_SIMD_Float64x2).
func Pow2_NEON_SIMD_F64x2(k archsimd.Int32x4) archsimd.Float64x2 {
	k64 := k.ExtendLo2ToInt64()
	biased := k64.Add(archsimd.BroadcastInt64x2(1023))
	return biased.ShiftAllLeft(52).ToBits().BitsToFloat64()
}

// RSqrt_NEON_SIMD_F32x4 computes an approximate reciprocal square root.
// archsimd arm64 does not expose FRSQRTE; a full-precision divide matches
// the accuracy expectations of RSqrt callers at NEON width.
func RSqrt_NEON_SIMD_F32x4(v archsimd.Float32x4) archsimd.Float32x4 {
	return archsimd.BroadcastFloat32x4(1).Div(v.Sqrt())
}

// RSqrt_NEON_SIMD_F64x2 computes the reciprocal square root.
func RSqrt_NEON_SIMD_F64x2(v archsimd.Float64x2) archsimd.Float64x2 {
	return archsimd.BroadcastFloat64x2(1).Div(v.Sqrt())
}

// And_NEON_SIMD_F32x4 performs a bitwise AND on float32 lanes.
func And_NEON_SIMD_F32x4(a, b archsimd.Float32x4) archsimd.Float32x4 {
	return a.ToBits().And(b.ToBits()).BitsToFloat32()
}

// And_NEON_SIMD_F64x2 performs a bitwise AND on float64 lanes.
func And_NEON_SIMD_F64x2(a, b archsimd.Float64x2) archsimd.Float64x2 {
	return a.ToBits().And(b.ToBits()).BitsToFloat64()
}

// Or_NEON_SIMD_F32x4 performs a bitwise OR on float32 lanes.
func Or_NEON_SIMD_F32x4(a, b archsimd.Float32x4) archsimd.Float32x4 {
	return a.ToBits().Or(b.ToBits()).BitsToFloat32()
}

// Or_NEON_SIMD_F64x2 performs a bitwise OR on float64 lanes.
func Or_NEON_SIMD_F64x2(a, b archsimd.Float64x2) archsimd.Float64x2 {
	return a.ToBits().Or(b.ToBits()).BitsToFloat64()
}

// Xor_NEON_SIMD_F32x4 performs a bitwise XOR on float32 lanes.
func Xor_NEON_SIMD_F32x4(a, b archsimd.Float32x4) archsimd.Float32x4 {
	return a.ToBits().Xor(b.ToBits()).BitsToFloat32()
}

// Xor_NEON_SIMD_F64x2 performs a bitwise XOR on float64 lanes.
func Xor_NEON_SIMD_F64x2(a, b archsimd.Float64x2) archsimd.Float64x2 {
	return a.ToBits().Xor(b.ToBits()).BitsToFloat64()
}

// Not_NEON_SIMD_F32x4 performs a bitwise NOT on float32 lanes.
func Not_NEON_SIMD_F32x4(a archsimd.Float32x4) archsimd.Float32x4 {
	return a.ToBits().Not().BitsToFloat32()
}

// Not_NEON_SIMD_F64x2 performs a bitwise NOT on float64 lanes.
func Not_NEON_SIMD_F64x2(a archsimd.Float64x2) archsimd.Float64x2 {
	return a.ToBits().Not().BitsToFloat64()
}

// AsInt32_NEON_SIMD_F32x4 reinterprets float32 bits as int32.
func AsInt32_NEON_SIMD_F32x4(v archsimd.Float32x4) archsimd.Int32x4 {
	return v.ToBits().BitsToInt32()
}

// AsFloat32_NEON_SIMD_I32x4 reinterprets int32 bits as float32.
func AsFloat32_NEON_SIMD_I32x4(v archsimd.Int32x4) archsimd.Float32x4 {
	return v.ToBits().BitsToFloat32()
}

// AsInt64_NEON_SIMD_F64x2 reinterprets float64 bits as int64.
func AsInt64_NEON_SIMD_F64x2(v archsimd.Float64x2) archsimd.Int64x2 {
	return v.ToBits().BitsToInt64()
}

// AsFloat64_NEON_SIMD_I64x2 reinterprets int64 bits as float64.
func AsFloat64_NEON_SIMD_I64x2(v archsimd.Int64x2) archsimd.Float64x2 {
	return v.ToBits().BitsToFloat64()
}

// SignBit_NEON_SIMD_F32x4 returns a vector with only the sign bit set in
// each float32 lane.
func SignBit_NEON_SIMD_F32x4() archsimd.Float32x4 {
	return archsimd.BroadcastUint32x4(0x80000000).BitsToFloat32()
}

// SignBit_NEON_SIMD_F64x2 returns a vector with only the sign bit set in
// each float64 lane.
func SignBit_NEON_SIMD_F64x2() archsimd.Float64x2 {
	return archsimd.BroadcastUint64x2(0x8000000000000000).BitsToFloat64()
}

// ReduceSum_NEON_SIMD_F32x4 sums all lanes (pairwise adds).
func ReduceSum_NEON_SIMD_F32x4(v archsimd.Float32x4) float32 {
	s := v.ConcatAddPairs(v) // [v0+v1, v2+v3, v0+v1, v2+v3]
	s = s.ConcatAddPairs(s)  // all lanes = v0+v1+v2+v3
	return s.GetElem(0)
}

// ReduceSum_NEON_SIMD_F64x2 sums both lanes.
func ReduceSum_NEON_SIMD_F64x2(v archsimd.Float64x2) float64 {
	return v.GetElem(0) + v.GetElem(1)
}

// ReduceSum_NEON_SIMD_I32x4 sums all lanes.
func ReduceSum_NEON_SIMD_I32x4(v archsimd.Int32x4) int32 {
	return v.ReduceSum()
}

// ReduceSum_NEON_SIMD_I64x2 sums both lanes (Int64x2 has no native
// ReduceSum on arm64 archsimd).
func ReduceSum_NEON_SIMD_I64x2(v archsimd.Int64x2) int64 {
	return v.GetElem(0) + v.GetElem(1)
}
