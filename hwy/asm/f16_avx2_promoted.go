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

// Float16x8AVX2 is a promoted storage wrapper: Float16 data is converted to float32
// at Load boundaries and computed on as archsimd.Float32x8, then converted back at Store.
// This gives real SIMD throughput (8 lanes of float32 AVX2) instead of scalar promote-compute-demote.
package asm

import (
	"math"
	"simd/archsimd"
	"unsafe"
)

// Float16x8AVX2 holds 8 float16 values promoted to float32 in an AVX2 register.
// All arithmetic is performed in float32; conversion to/from float16 happens only at Load/Store.
type Float16x8AVX2 struct {
	data archsimd.Float32x8
}

// ============================================================================
// Load/Store (boundary conversion)
// ============================================================================

// LoadFloat16x8AVX2Slice loads 8 float16 values from a uint16 slice, promoting to float32.
func LoadFloat16x8AVX2Slice(s []uint16) Float16x8AVX2 {
	var buf [8]float32
	PromoteF16ToF32F16C(s[:8], buf[:])
	return Float16x8AVX2{data: archsimd.LoadFloat32x8Slice(buf[:])}
}

// LoadFloat16x8AVX2Ptr loads 8 float16 values from an unsafe.Pointer, promoting to float32.
func LoadFloat16x8AVX2Ptr(ptr unsafe.Pointer) Float16x8AVX2 {
	s := unsafe.Slice((*uint16)(ptr), 8)
	return LoadFloat16x8AVX2Slice(s)
}

// StoreSlice demotes float32 back to float16 and stores 8 values to a uint16 slice.
func (v Float16x8AVX2) StoreSlice(s []uint16) {
	var buf [8]float32
	v.data.StoreSlice(buf[:])
	DemoteF32ToF16F16C(buf[:], s[:8])
}

// StorePtr demotes float32 back to float16 and stores 8 values to an unsafe.Pointer.
func (v Float16x8AVX2) StorePtr(ptr unsafe.Pointer) {
	s := unsafe.Slice((*uint16)(ptr), 8)
	v.StoreSlice(s)
}

// BroadcastFloat16x8AVX2 broadcasts a single float16 (as uint16) to all 8 lanes.
func BroadcastFloat16x8AVX2(val uint16) Float16x8AVX2 {
	f := float16BitsToFloat32(val)
	return Float16x8AVX2{data: archsimd.BroadcastFloat32x8(f)}
}

// ZeroFloat16x8AVX2 returns a zero vector.
func ZeroFloat16x8AVX2() Float16x8AVX2 {
	return Float16x8AVX2{data: archsimd.BroadcastFloat32x8(0)}
}

// ============================================================================
// Arithmetic (delegate to archsimd.Float32x8)
// ============================================================================

func (v Float16x8AVX2) Add(other Float16x8AVX2) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.Add(other.data)}
}

func (v Float16x8AVX2) Sub(other Float16x8AVX2) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.Sub(other.data)}
}

func (v Float16x8AVX2) Mul(other Float16x8AVX2) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.Mul(other.data)}
}

func (v Float16x8AVX2) Div(other Float16x8AVX2) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.Div(other.data)}
}

func (v Float16x8AVX2) Min(other Float16x8AVX2) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.Min(other.data)}
}

func (v Float16x8AVX2) Max(other Float16x8AVX2) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.Max(other.data)}
}

func (v Float16x8AVX2) Sqrt() Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.Sqrt()}
}

func (v Float16x8AVX2) Neg() Float16x8AVX2 {
	zero := archsimd.BroadcastFloat32x8(0)
	return Float16x8AVX2{data: zero.Sub(v.data)}
}

func (v Float16x8AVX2) Abs() Float16x8AVX2 {
	neg := v.Neg()
	return Float16x8AVX2{data: v.data.Max(neg.data)}
}

func (v Float16x8AVX2) MulAdd(b, c Float16x8AVX2) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.MulAdd(b.data, c.data)}
}

func (v Float16x8AVX2) MulSub(b, c Float16x8AVX2) Float16x8AVX2 {
	negC := Float16x8AVX2{data: archsimd.BroadcastFloat32x8(0).Sub(c.data)}
	return Float16x8AVX2{data: v.data.MulAdd(b.data, negC.data)}
}

func (v Float16x8AVX2) ReciprocalSqrt() Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.ReciprocalSqrt()}
}

// ============================================================================
// Comparisons (return archsimd.Mask32x8)
// ============================================================================

func (v Float16x8AVX2) Less(other Float16x8AVX2) archsimd.Mask32x8 {
	return v.data.Less(other.data)
}

func (v Float16x8AVX2) Greater(other Float16x8AVX2) archsimd.Mask32x8 {
	return v.data.Greater(other.data)
}

func (v Float16x8AVX2) Equal(other Float16x8AVX2) archsimd.Mask32x8 {
	return v.data.Equal(other.data)
}

func (v Float16x8AVX2) NotEqual(other Float16x8AVX2) archsimd.Mask32x8 {
	return v.data.NotEqual(other.data)
}

func (v Float16x8AVX2) LessEqual(other Float16x8AVX2) archsimd.Mask32x8 {
	return v.data.LessEqual(other.data)
}

func (v Float16x8AVX2) GreaterEqual(other Float16x8AVX2) archsimd.Mask32x8 {
	return v.data.GreaterEqual(other.data)
}

// ============================================================================
// Rounding/Conversion (delegate)
// ============================================================================

func (v Float16x8AVX2) RoundToEven() Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.RoundToEven()}
}

func (v Float16x8AVX2) ConvertToInt32() archsimd.Int32x8 {
	return v.data.ConvertToInt32()
}

// ============================================================================
// Bit casting (delegate)
// ============================================================================

func (v Float16x8AVX2) AsInt32x8() archsimd.Int32x8 {
	return v.data.AsInt32x8()
}

func (v Float16x8AVX2) AsFloat32x8() archsimd.Float32x8 {
	return v.data
}

// Float16x8AVX2FromFloat32x8 wraps a Float32x8 as a promoted Float16 vector.
func Float16x8AVX2FromFloat32x8(v archsimd.Float32x8) Float16x8AVX2 {
	return Float16x8AVX2{data: v}
}

// ============================================================================
// Bitwise (via int32 cast)
// ============================================================================

func (v Float16x8AVX2) And(other Float16x8AVX2) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.AsInt32x8().And(other.data.AsInt32x8()).AsFloat32x8()}
}

func (v Float16x8AVX2) Or(other Float16x8AVX2) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.AsInt32x8().Or(other.data.AsInt32x8()).AsFloat32x8()}
}

func (v Float16x8AVX2) Xor(other Float16x8AVX2) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.AsInt32x8().Xor(other.data.AsInt32x8()).AsFloat32x8()}
}

func (v Float16x8AVX2) AndNot(other Float16x8AVX2) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.AsInt32x8().AndNot(other.data.AsInt32x8()).AsFloat32x8()}
}

func (v Float16x8AVX2) Not() Float16x8AVX2 {
	allOnes := archsimd.BroadcastInt32x8(int32(-1))
	return Float16x8AVX2{data: v.data.AsInt32x8().Xor(allOnes).AsFloat32x8()}
}

// ============================================================================
// Merge/Blend
// ============================================================================

func (v Float16x8AVX2) Merge(other Float16x8AVX2, mask archsimd.Mask32x8) Float16x8AVX2 {
	return Float16x8AVX2{data: v.data.Merge(other.data, mask)}
}

// ============================================================================
// Reductions (return float32)
// ============================================================================

func (v Float16x8AVX2) ReduceSum() float32 {
	var buf [8]float32
	v.data.StoreSlice(buf[:])
	return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]
}

func (v Float16x8AVX2) ReduceMax() float32 {
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

func (v Float16x8AVX2) ReduceMin() float32 {
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

// IotaFloat16x8AVX2 returns a vector with lane indices [0, 1, 2, ..., 7] as promoted float32.
func IotaFloat16x8AVX2() Float16x8AVX2 {
	buf := [8]float32{0, 1, 2, 3, 4, 5, 6, 7}
	return Float16x8AVX2{data: archsimd.LoadFloat32x8Slice(buf[:])}
}

// SignBitFloat16x8AVX2 returns a vector with the sign bit set in each float32 lane.
func SignBitFloat16x8AVX2() Float16x8AVX2 {
	return Float16x8AVX2{data: archsimd.BroadcastFloat32x8(math.Float32frombits(0x80000000))}
}

// ============================================================================
// Data access
// ============================================================================

// Data returns the underlying archsimd.Float32x8 register.
func (v Float16x8AVX2) Data() archsimd.Float32x8 {
	return v.data
}

// ============================================================================
// Interleave
// ============================================================================

func (v Float16x8AVX2) InterleaveLower(other Float16x8AVX2) Float16x8AVX2 {
	var a, b [8]float32
	v.data.StoreSlice(a[:])
	other.data.StoreSlice(b[:])
	result := [8]float32{a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]}
	return Float16x8AVX2{data: archsimd.LoadFloat32x8Slice(result[:])}
}

func (v Float16x8AVX2) InterleaveUpper(other Float16x8AVX2) Float16x8AVX2 {
	var a, b [8]float32
	v.data.StoreSlice(a[:])
	other.data.StoreSlice(b[:])
	result := [8]float32{a[4], b[4], a[5], b[5], a[6], b[6], a[7], b[7]}
	return Float16x8AVX2{data: archsimd.LoadFloat32x8Slice(result[:])}
}

// ============================================================================
// Helpers
// ============================================================================

// float16BitsToFloat32 converts a float16 bit pattern (uint16) to float32.
func float16BitsToFloat32(bits uint16) float32 {
	sign := uint32(bits>>15) << 31
	exp := uint32(bits>>10) & 0x1F
	mant := uint32(bits) & 0x3FF

	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Denormalized: normalize
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
		return math.Float32frombits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
	case exp == 0x1F:
		if mant == 0 {
			return math.Float32frombits(sign | 0x7F800000)
		}
		return math.Float32frombits(sign | 0x7FC00000 | (mant << 13))
	default:
		return math.Float32frombits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
	}
}
