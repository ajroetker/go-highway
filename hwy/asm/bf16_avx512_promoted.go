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

// BFloat16x16AVX512 is a promoted storage wrapper: BFloat16 data is converted to float32
// at Load boundaries and computed on as archsimd.Float32x16, then converted back at Store.
package asm

import (
	"math"
	"simd/archsimd"
	"unsafe"
)

// BFloat16x16AVX512 holds 16 bfloat16 values promoted to float32 in an AVX-512 register.
type BFloat16x16AVX512 struct {
	data archsimd.Float32x16
}

// ============================================================================
// Load/Store (boundary conversion)
// ============================================================================

// LoadBFloat16x16AVX512Slice loads 16 bfloat16 values from a uint16 slice, promoting to float32.
func LoadBFloat16x16AVX512Slice(s []uint16) BFloat16x16AVX512 {
	var buf [16]float32
	for i := 0; i < 16; i++ {
		buf[i] = bfloat16BitsToFloat32(s[i])
	}
	return BFloat16x16AVX512{data: archsimd.LoadFloat32x16Slice(buf[:])}
}

// LoadBFloat16x16AVX512Ptr loads 16 bfloat16 values from an unsafe.Pointer, promoting to float32.
func LoadBFloat16x16AVX512Ptr(ptr unsafe.Pointer) BFloat16x16AVX512 {
	s := unsafe.Slice((*uint16)(ptr), 16)
	return LoadBFloat16x16AVX512Slice(s)
}

// StoreSlice demotes float32 back to bfloat16 and stores 16 values to a uint16 slice.
func (v BFloat16x16AVX512) StoreSlice(s []uint16) {
	var buf [16]float32
	v.data.StoreSlice(buf[:])
	for i := 0; i < 16; i++ {
		s[i] = float32ToBFloat16Bits(buf[i])
	}
}

// StorePtr demotes float32 back to bfloat16 and stores 16 values to an unsafe.Pointer.
func (v BFloat16x16AVX512) StorePtr(ptr unsafe.Pointer) {
	s := unsafe.Slice((*uint16)(ptr), 16)
	v.StoreSlice(s)
}

// BroadcastBFloat16x16AVX512 broadcasts a single bfloat16 (as uint16) to all 16 lanes.
func BroadcastBFloat16x16AVX512(val uint16) BFloat16x16AVX512 {
	f := bfloat16BitsToFloat32(val)
	return BFloat16x16AVX512{data: archsimd.BroadcastFloat32x16(f)}
}

// ZeroBFloat16x16AVX512 returns a zero vector.
func ZeroBFloat16x16AVX512() BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: archsimd.BroadcastFloat32x16(0)}
}

// ============================================================================
// Arithmetic (delegate to archsimd.Float32x16)
// ============================================================================

func (v BFloat16x16AVX512) Add(other BFloat16x16AVX512) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.Add(other.data)}
}

func (v BFloat16x16AVX512) Sub(other BFloat16x16AVX512) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.Sub(other.data)}
}

func (v BFloat16x16AVX512) Mul(other BFloat16x16AVX512) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.Mul(other.data)}
}

func (v BFloat16x16AVX512) Div(other BFloat16x16AVX512) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.Div(other.data)}
}

func (v BFloat16x16AVX512) Min(other BFloat16x16AVX512) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.Min(other.data)}
}

func (v BFloat16x16AVX512) Max(other BFloat16x16AVX512) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.Max(other.data)}
}

func (v BFloat16x16AVX512) Sqrt() BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.Sqrt()}
}

func (v BFloat16x16AVX512) Neg() BFloat16x16AVX512 {
	zero := archsimd.BroadcastFloat32x16(0)
	return BFloat16x16AVX512{data: zero.Sub(v.data)}
}

func (v BFloat16x16AVX512) Abs() BFloat16x16AVX512 {
	neg := v.Neg()
	return BFloat16x16AVX512{data: v.data.Max(neg.data)}
}

func (v BFloat16x16AVX512) MulAdd(b, c BFloat16x16AVX512) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.MulAdd(b.data, c.data)}
}

func (v BFloat16x16AVX512) MulSub(b, c BFloat16x16AVX512) BFloat16x16AVX512 {
	negC := BFloat16x16AVX512{data: archsimd.BroadcastFloat32x16(0).Sub(c.data)}
	return BFloat16x16AVX512{data: v.data.MulAdd(b.data, negC.data)}
}

func (v BFloat16x16AVX512) ReciprocalSqrt() BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.ReciprocalSqrt()}
}

// ============================================================================
// Comparisons (return archsimd.Mask32x16)
// ============================================================================

func (v BFloat16x16AVX512) Less(other BFloat16x16AVX512) archsimd.Mask32x16 {
	return v.data.Less(other.data)
}

func (v BFloat16x16AVX512) Greater(other BFloat16x16AVX512) archsimd.Mask32x16 {
	return v.data.Greater(other.data)
}

func (v BFloat16x16AVX512) Equal(other BFloat16x16AVX512) archsimd.Mask32x16 {
	return v.data.Equal(other.data)
}

func (v BFloat16x16AVX512) NotEqual(other BFloat16x16AVX512) archsimd.Mask32x16 {
	return v.data.NotEqual(other.data)
}

func (v BFloat16x16AVX512) LessEqual(other BFloat16x16AVX512) archsimd.Mask32x16 {
	return v.data.LessEqual(other.data)
}

func (v BFloat16x16AVX512) GreaterEqual(other BFloat16x16AVX512) archsimd.Mask32x16 {
	return v.data.GreaterEqual(other.data)
}

// ============================================================================
// Rounding/Conversion (delegate)
// ============================================================================

func (v BFloat16x16AVX512) RoundToEven() BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.RoundToEven()}
}

func (v BFloat16x16AVX512) ConvertToInt32() archsimd.Int32x16 {
	return v.data.ConvertToInt32()
}

// ============================================================================
// Bit casting (delegate)
// ============================================================================

func (v BFloat16x16AVX512) AsInt32x16() archsimd.Int32x16 {
	return v.data.AsInt32x16()
}

func (v BFloat16x16AVX512) AsFloat32x16() archsimd.Float32x16 {
	return v.data
}

// BFloat16x16AVX512FromFloat32x16 wraps a Float32x16 as a promoted BFloat16 vector.
func BFloat16x16AVX512FromFloat32x16(v archsimd.Float32x16) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v}
}

// ============================================================================
// Bitwise (via int32 cast)
// ============================================================================

func (v BFloat16x16AVX512) And(other BFloat16x16AVX512) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.AsInt32x16().And(other.data.AsInt32x16()).AsFloat32x16()}
}

func (v BFloat16x16AVX512) Or(other BFloat16x16AVX512) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.AsInt32x16().Or(other.data.AsInt32x16()).AsFloat32x16()}
}

func (v BFloat16x16AVX512) Xor(other BFloat16x16AVX512) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.AsInt32x16().Xor(other.data.AsInt32x16()).AsFloat32x16()}
}

func (v BFloat16x16AVX512) AndNot(other BFloat16x16AVX512) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.AsInt32x16().AndNot(other.data.AsInt32x16()).AsFloat32x16()}
}

func (v BFloat16x16AVX512) Not() BFloat16x16AVX512 {
	allOnes := archsimd.BroadcastInt32x16(int32(-1))
	return BFloat16x16AVX512{data: v.data.AsInt32x16().Xor(allOnes).AsFloat32x16()}
}

// ============================================================================
// Merge/Blend
// ============================================================================

func (v BFloat16x16AVX512) Merge(other BFloat16x16AVX512, mask archsimd.Mask32x16) BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: v.data.Merge(other.data, mask)}
}

// ============================================================================
// Reductions (return float32)
// ============================================================================

func (v BFloat16x16AVX512) ReduceSum() float32 {
	var buf [16]float32
	v.data.StoreSlice(buf[:])
	var sum float32
	for i := 0; i < 16; i++ {
		sum += buf[i]
	}
	return sum
}

func (v BFloat16x16AVX512) ReduceMax() float32 {
	var buf [16]float32
	v.data.StoreSlice(buf[:])
	m := buf[0]
	for i := 1; i < 16; i++ {
		if buf[i] > m {
			m = buf[i]
		}
	}
	return m
}

func (v BFloat16x16AVX512) ReduceMin() float32 {
	var buf [16]float32
	v.data.StoreSlice(buf[:])
	m := buf[0]
	for i := 1; i < 16; i++ {
		if buf[i] < m {
			m = buf[i]
		}
	}
	return m
}

// ============================================================================
// Initialization
// ============================================================================

func IotaBFloat16x16AVX512() BFloat16x16AVX512 {
	buf := [16]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	return BFloat16x16AVX512{data: archsimd.LoadFloat32x16Slice(buf[:])}
}

func SignBitBFloat16x16AVX512() BFloat16x16AVX512 {
	return BFloat16x16AVX512{data: archsimd.BroadcastFloat32x16(math.Float32frombits(0x80000000))}
}

// ============================================================================
// Data access
// ============================================================================

func (v BFloat16x16AVX512) Data() archsimd.Float32x16 {
	return v.data
}

// ============================================================================
// Interleave
// ============================================================================

func (v BFloat16x16AVX512) InterleaveLower(other BFloat16x16AVX512) BFloat16x16AVX512 {
	var a, b [16]float32
	v.data.StoreSlice(a[:])
	other.data.StoreSlice(b[:])
	var result [16]float32
	for i := 0; i < 8; i++ {
		result[2*i] = a[i]
		result[2*i+1] = b[i]
	}
	return BFloat16x16AVX512{data: archsimd.LoadFloat32x16Slice(result[:])}
}

func (v BFloat16x16AVX512) InterleaveUpper(other BFloat16x16AVX512) BFloat16x16AVX512 {
	var a, b [16]float32
	v.data.StoreSlice(a[:])
	other.data.StoreSlice(b[:])
	var result [16]float32
	for i := 0; i < 8; i++ {
		result[2*i] = a[8+i]
		result[2*i+1] = b[8+i]
	}
	return BFloat16x16AVX512{data: archsimd.LoadFloat32x16Slice(result[:])}
}
