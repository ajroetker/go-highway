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

// BFloat16x8AVX2 is a promoted storage wrapper: BFloat16 data is converted to float32
// at Load boundaries and computed on as archsimd.Float32x8, then converted back at Store.
package asm

import (
	"math"
	"simd/archsimd"
	"unsafe"
)

// BFloat16x8AVX2 holds 8 bfloat16 values promoted to float32 in an AVX2 register.
type BFloat16x8AVX2 struct {
	data archsimd.Float32x8
}

// ============================================================================
// Load/Store (boundary conversion)
// ============================================================================

// LoadBFloat16x8AVX2Slice loads 8 bfloat16 values from a uint16 slice, promoting to float32.
func LoadBFloat16x8AVX2Slice(s []uint16) BFloat16x8AVX2 {
	var buf [8]float32
	for i := 0; i < 8; i++ {
		buf[i] = bfloat16BitsToFloat32(s[i])
	}
	return BFloat16x8AVX2{data: archsimd.LoadFloat32x8Slice(buf[:])}
}

// LoadBFloat16x8AVX2Ptr loads 8 bfloat16 values from an unsafe.Pointer, promoting to float32.
func LoadBFloat16x8AVX2Ptr(ptr unsafe.Pointer) BFloat16x8AVX2 {
	s := unsafe.Slice((*uint16)(ptr), 8)
	return LoadBFloat16x8AVX2Slice(s)
}

// StoreSlice demotes float32 back to bfloat16 and stores 8 values to a uint16 slice.
func (v BFloat16x8AVX2) StoreSlice(s []uint16) {
	var buf [8]float32
	v.data.StoreSlice(buf[:])
	for i := 0; i < 8; i++ {
		s[i] = float32ToBFloat16Bits(buf[i])
	}
}

// StorePtr demotes float32 back to bfloat16 and stores 8 values to an unsafe.Pointer.
func (v BFloat16x8AVX2) StorePtr(ptr unsafe.Pointer) {
	s := unsafe.Slice((*uint16)(ptr), 8)
	v.StoreSlice(s)
}

// BroadcastBFloat16x8AVX2 broadcasts a single bfloat16 (as uint16) to all 8 lanes.
func BroadcastBFloat16x8AVX2(val uint16) BFloat16x8AVX2 {
	f := bfloat16BitsToFloat32(val)
	return BFloat16x8AVX2{data: archsimd.BroadcastFloat32x8(f)}
}

// ZeroBFloat16x8AVX2 returns a zero vector.
func ZeroBFloat16x8AVX2() BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: archsimd.BroadcastFloat32x8(0)}
}

// ============================================================================
// Arithmetic (delegate to archsimd.Float32x8)
// ============================================================================

func (v BFloat16x8AVX2) Add(other BFloat16x8AVX2) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.Add(other.data)}
}

func (v BFloat16x8AVX2) Sub(other BFloat16x8AVX2) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.Sub(other.data)}
}

func (v BFloat16x8AVX2) Mul(other BFloat16x8AVX2) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.Mul(other.data)}
}

func (v BFloat16x8AVX2) Div(other BFloat16x8AVX2) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.Div(other.data)}
}

func (v BFloat16x8AVX2) Min(other BFloat16x8AVX2) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.Min(other.data)}
}

func (v BFloat16x8AVX2) Max(other BFloat16x8AVX2) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.Max(other.data)}
}

func (v BFloat16x8AVX2) Sqrt() BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.Sqrt()}
}

func (v BFloat16x8AVX2) Neg() BFloat16x8AVX2 {
	zero := archsimd.BroadcastFloat32x8(0)
	return BFloat16x8AVX2{data: zero.Sub(v.data)}
}

func (v BFloat16x8AVX2) Abs() BFloat16x8AVX2 {
	neg := v.Neg()
	return BFloat16x8AVX2{data: v.data.Max(neg.data)}
}

func (v BFloat16x8AVX2) MulAdd(b, c BFloat16x8AVX2) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.MulAdd(b.data, c.data)}
}

func (v BFloat16x8AVX2) MulSub(b, c BFloat16x8AVX2) BFloat16x8AVX2 {
	negC := BFloat16x8AVX2{data: archsimd.BroadcastFloat32x8(0).Sub(c.data)}
	return BFloat16x8AVX2{data: v.data.MulAdd(b.data, negC.data)}
}

func (v BFloat16x8AVX2) ReciprocalSqrt() BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.ReciprocalSqrt()}
}

// ============================================================================
// Comparisons (return archsimd.Mask32x8)
// ============================================================================

func (v BFloat16x8AVX2) Less(other BFloat16x8AVX2) archsimd.Mask32x8 {
	return v.data.Less(other.data)
}

func (v BFloat16x8AVX2) Greater(other BFloat16x8AVX2) archsimd.Mask32x8 {
	return v.data.Greater(other.data)
}

func (v BFloat16x8AVX2) Equal(other BFloat16x8AVX2) archsimd.Mask32x8 {
	return v.data.Equal(other.data)
}

func (v BFloat16x8AVX2) NotEqual(other BFloat16x8AVX2) archsimd.Mask32x8 {
	return v.data.NotEqual(other.data)
}

func (v BFloat16x8AVX2) LessEqual(other BFloat16x8AVX2) archsimd.Mask32x8 {
	return v.data.LessEqual(other.data)
}

func (v BFloat16x8AVX2) GreaterEqual(other BFloat16x8AVX2) archsimd.Mask32x8 {
	return v.data.GreaterEqual(other.data)
}

// ============================================================================
// Rounding/Conversion (delegate)
// ============================================================================

func (v BFloat16x8AVX2) RoundToEven() BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.RoundToEven()}
}

func (v BFloat16x8AVX2) ConvertToInt32() archsimd.Int32x8 {
	return v.data.ConvertToInt32()
}

// ============================================================================
// Bit casting (delegate)
// ============================================================================

func (v BFloat16x8AVX2) AsInt32x8() archsimd.Int32x8 {
	return v.data.AsInt32x8()
}

func (v BFloat16x8AVX2) AsFloat32x8() archsimd.Float32x8 {
	return v.data
}

// BFloat16x8AVX2FromFloat32x8 wraps a Float32x8 as a promoted BFloat16 vector.
func BFloat16x8AVX2FromFloat32x8(v archsimd.Float32x8) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v}
}

// ============================================================================
// Bitwise (via int32 cast)
// ============================================================================

func (v BFloat16x8AVX2) And(other BFloat16x8AVX2) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.AsInt32x8().And(other.data.AsInt32x8()).AsFloat32x8()}
}

func (v BFloat16x8AVX2) Or(other BFloat16x8AVX2) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.AsInt32x8().Or(other.data.AsInt32x8()).AsFloat32x8()}
}

func (v BFloat16x8AVX2) Xor(other BFloat16x8AVX2) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.AsInt32x8().Xor(other.data.AsInt32x8()).AsFloat32x8()}
}

func (v BFloat16x8AVX2) AndNot(other BFloat16x8AVX2) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.AsInt32x8().AndNot(other.data.AsInt32x8()).AsFloat32x8()}
}

func (v BFloat16x8AVX2) Not() BFloat16x8AVX2 {
	allOnes := archsimd.BroadcastInt32x8(int32(-1))
	return BFloat16x8AVX2{data: v.data.AsInt32x8().Xor(allOnes).AsFloat32x8()}
}

// ============================================================================
// Merge/Blend
// ============================================================================

func (v BFloat16x8AVX2) Merge(other BFloat16x8AVX2, mask archsimd.Mask32x8) BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: v.data.Merge(other.data, mask)}
}

// ============================================================================
// Reductions (return float32)
// ============================================================================

func (v BFloat16x8AVX2) ReduceSum() float32 {
	var buf [8]float32
	v.data.StoreSlice(buf[:])
	return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]
}

func (v BFloat16x8AVX2) ReduceMax() float32 {
	var buf [8]float32
	v.data.StoreSlice(buf[:])
	m := buf[0]
	for i := 1; i < 8; i++ {
		if buf[i] > m {
			m = buf[i]
		}
	}
	return m
}

func (v BFloat16x8AVX2) ReduceMin() float32 {
	var buf [8]float32
	v.data.StoreSlice(buf[:])
	m := buf[0]
	for i := 1; i < 8; i++ {
		if buf[i] < m {
			m = buf[i]
		}
	}
	return m
}

// ============================================================================
// Initialization
// ============================================================================

// IotaBFloat16x8AVX2 returns a vector with lane indices [0, 1, 2, ..., 7] as promoted float32.
func IotaBFloat16x8AVX2() BFloat16x8AVX2 {
	buf := [8]float32{0, 1, 2, 3, 4, 5, 6, 7}
	return BFloat16x8AVX2{data: archsimd.LoadFloat32x8Slice(buf[:])}
}

// SignBitBFloat16x8AVX2 returns a vector with the sign bit set in each float32 lane.
func SignBitBFloat16x8AVX2() BFloat16x8AVX2 {
	return BFloat16x8AVX2{data: archsimd.BroadcastFloat32x8(math.Float32frombits(0x80000000))}
}

// ============================================================================
// Data access
// ============================================================================

func (v BFloat16x8AVX2) Data() archsimd.Float32x8 {
	return v.data
}

// ============================================================================
// Interleave
// ============================================================================

func (v BFloat16x8AVX2) InterleaveLower(other BFloat16x8AVX2) BFloat16x8AVX2 {
	var a, b [8]float32
	v.data.StoreSlice(a[:])
	other.data.StoreSlice(b[:])
	result := [8]float32{a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]}
	return BFloat16x8AVX2{data: archsimd.LoadFloat32x8Slice(result[:])}
}

func (v BFloat16x8AVX2) InterleaveUpper(other BFloat16x8AVX2) BFloat16x8AVX2 {
	var a, b [8]float32
	v.data.StoreSlice(a[:])
	other.data.StoreSlice(b[:])
	result := [8]float32{a[4], b[4], a[5], b[5], a[6], b[6], a[7], b[7]}
	return BFloat16x8AVX2{data: archsimd.LoadFloat32x8Slice(result[:])}
}

// ============================================================================
// Helpers
// ============================================================================

// bfloat16BitsToFloat32 converts a bfloat16 bit pattern (uint16) to float32.
// BFloat16 is simply the upper 16 bits of a float32, so shift left by 16.
func bfloat16BitsToFloat32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}

// float32ToBFloat16Bits converts a float32 to bfloat16 with round-to-nearest-even.
func float32ToBFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	// Round to nearest even
	rounding := uint32(0x7FFF) + ((bits >> 16) & 1)
	return uint16((bits + rounding) >> 16)
}
