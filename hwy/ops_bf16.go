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

package hwy

import "math"

// This file provides BFloat16 arithmetic operations using the promote-compute-demote pattern.
// Since Go has no native bfloat16 arithmetic, we:
// 1. Promote BFloat16 to float32 (trivial - just bit shift)
// 2. Perform the arithmetic in float32
// 3. Demote the result back to BFloat16 (with rounding)
//
// When SIMD implementations are available (e.g., AVX-512 BF16, NEON BF16),
// native hardware operations will be used instead.

// AddBF16 performs element-wise addition of two BFloat16 vectors.
func AddBF16(a, b Vec[BFloat16]) Vec[BFloat16] {
	n := min(len(b.data), len(a.data))
	result := make([]BFloat16, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		result[i] = Float32ToBFloat16(af + bf)
	}
	return Vec[BFloat16]{data: result}
}

// SubBF16 performs element-wise subtraction of two BFloat16 vectors.
func SubBF16(a, b Vec[BFloat16]) Vec[BFloat16] {
	n := min(len(b.data), len(a.data))
	result := make([]BFloat16, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		result[i] = Float32ToBFloat16(af - bf)
	}
	return Vec[BFloat16]{data: result}
}

// MulBF16 performs element-wise multiplication of two BFloat16 vectors.
func MulBF16(a, b Vec[BFloat16]) Vec[BFloat16] {
	n := min(len(b.data), len(a.data))
	result := make([]BFloat16, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		result[i] = Float32ToBFloat16(af * bf)
	}
	return Vec[BFloat16]{data: result}
}

// DivBF16 performs element-wise division of two BFloat16 vectors.
func DivBF16(a, b Vec[BFloat16]) Vec[BFloat16] {
	n := min(len(b.data), len(a.data))
	result := make([]BFloat16, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		result[i] = Float32ToBFloat16(af / bf)
	}
	return Vec[BFloat16]{data: result}
}

// FMABF16 performs fused multiply-add: a * b + c.
func FMABF16(a, b, c Vec[BFloat16]) Vec[BFloat16] {
	n := min(len(c.data), min(len(b.data), len(a.data)))
	result := make([]BFloat16, n)
	for i := range n {
		af := float64(BFloat16ToFloat32(a.data[i]))
		bf := float64(BFloat16ToFloat32(b.data[i]))
		cf := float64(BFloat16ToFloat32(c.data[i]))
		result[i] = Float32ToBFloat16(float32(math.FMA(af, bf, cf)))
	}
	return Vec[BFloat16]{data: result}
}

// NegBF16 negates all lanes.
func NegBF16(v Vec[BFloat16]) Vec[BFloat16] {
	result := make([]BFloat16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		// Negation is just flipping the sign bit
		result[i] = v.data[i] ^ 0x8000
	}
	return Vec[BFloat16]{data: result}
}

// AbsBF16 computes absolute value.
func AbsBF16(v Vec[BFloat16]) Vec[BFloat16] {
	result := make([]BFloat16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		// Absolute value is just clearing the sign bit
		result[i] = v.data[i] & 0x7FFF
	}
	return Vec[BFloat16]{data: result}
}

// MinBF16 returns element-wise minimum.
func MinBF16(a, b Vec[BFloat16]) Vec[BFloat16] {
	n := min(len(b.data), len(a.data))
	result := make([]BFloat16, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		if af < bf {
			result[i] = a.data[i]
		} else {
			result[i] = b.data[i]
		}
	}
	return Vec[BFloat16]{data: result}
}

// MaxBF16 returns element-wise maximum.
func MaxBF16(a, b Vec[BFloat16]) Vec[BFloat16] {
	n := min(len(b.data), len(a.data))
	result := make([]BFloat16, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		if af > bf {
			result[i] = a.data[i]
		} else {
			result[i] = b.data[i]
		}
	}
	return Vec[BFloat16]{data: result}
}

// SqrtBF16 computes square root.
func SqrtBF16(v Vec[BFloat16]) Vec[BFloat16] {
	result := make([]BFloat16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		f := float64(BFloat16ToFloat32(v.data[i]))
		result[i] = Float32ToBFloat16(float32(math.Sqrt(f)))
	}
	return Vec[BFloat16]{data: result}
}

// ReciprocalBF16 computes 1/x (reciprocal).
func ReciprocalBF16(v Vec[BFloat16]) Vec[BFloat16] {
	result := make([]BFloat16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		f := BFloat16ToFloat32(v.data[i])
		result[i] = Float32ToBFloat16(1.0 / f)
	}
	return Vec[BFloat16]{data: result}
}

// ReciprocalSqrtBF16 computes 1/sqrt(x).
func ReciprocalSqrtBF16(v Vec[BFloat16]) Vec[BFloat16] {
	result := make([]BFloat16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		f := float64(BFloat16ToFloat32(v.data[i]))
		result[i] = Float32ToBFloat16(float32(1.0 / math.Sqrt(f)))
	}
	return Vec[BFloat16]{data: result}
}

// ReduceSumBF16 sums all lanes and returns the result as float32.
// Accumulation is done in float32 to avoid precision loss.
func ReduceSumBF16(v Vec[BFloat16]) float32 {
	var sum float32
	for i := 0; i < len(v.data); i++ {
		sum += BFloat16ToFloat32(v.data[i])
	}
	return sum
}

// ReduceMinBF16 returns the minimum value across all lanes.
func ReduceMinBF16(v Vec[BFloat16]) BFloat16 {
	if len(v.data) == 0 {
		return BFloat16Inf
	}
	min := v.data[0]
	minF := BFloat16ToFloat32(min)
	for i := 1; i < len(v.data); i++ {
		f := BFloat16ToFloat32(v.data[i])
		if f < minF {
			min = v.data[i]
			minF = f
		}
	}
	return min
}

// ReduceMaxBF16 returns the maximum value across all lanes.
func ReduceMaxBF16(v Vec[BFloat16]) BFloat16 {
	if len(v.data) == 0 {
		return BFloat16NegInf
	}
	max := v.data[0]
	maxF := BFloat16ToFloat32(max)
	for i := 1; i < len(v.data); i++ {
		f := BFloat16ToFloat32(v.data[i])
		if f > maxF {
			max = v.data[i]
			maxF = f
		}
	}
	return max
}

// DotBF16 computes dot product of two BFloat16 vectors, returning float32.
// This is the common ML pattern: accumulate in higher precision.
// On hardware with VDPBF16PS (AVX-512 BF16), this uses dedicated instruction.
func DotBF16(a, b Vec[BFloat16]) float32 {
	n := min(len(b.data), len(a.data))
	var sum float32
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		sum += af * bf
	}
	return sum
}

// NotEqualBF16 compares two BFloat16 vectors for inequality.
func NotEqualBF16(a, b Vec[BFloat16]) Mask[BFloat16] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		bits[i] = af != bf
	}
	return Mask[BFloat16]{bits: bits}
}

// EqualBF16 compares two BFloat16 vectors for equality.
func EqualBF16(a, b Vec[BFloat16]) Mask[BFloat16] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		bits[i] = af == bf
	}
	return Mask[BFloat16]{bits: bits}
}

// LessThanBF16 compares a < b element-wise.
func LessThanBF16(a, b Vec[BFloat16]) Mask[BFloat16] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		bits[i] = af < bf
	}
	return Mask[BFloat16]{bits: bits}
}

// LessThanOrEqualBF16 compares a <= b element-wise.
func LessThanOrEqualBF16(a, b Vec[BFloat16]) Mask[BFloat16] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		bits[i] = af <= bf
	}
	return Mask[BFloat16]{bits: bits}
}

// GreaterThanBF16 compares a > b element-wise.
func GreaterThanBF16(a, b Vec[BFloat16]) Mask[BFloat16] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		bits[i] = af > bf
	}
	return Mask[BFloat16]{bits: bits}
}

// GreaterThanOrEqualBF16 compares a >= b element-wise.
func GreaterThanOrEqualBF16(a, b Vec[BFloat16]) Mask[BFloat16] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		af := BFloat16ToFloat32(a.data[i])
		bf := BFloat16ToFloat32(b.data[i])
		bits[i] = af >= bf
	}
	return Mask[BFloat16]{bits: bits}
}

// IsNaNBF16 returns a mask where true indicates NaN.
func IsNaNBF16(v Vec[BFloat16]) Mask[BFloat16] {
	bits := make([]bool, len(v.data))
	for i := 0; i < len(v.data); i++ {
		bits[i] = v.data[i].IsNaN()
	}
	return Mask[BFloat16]{bits: bits}
}

// IsInfBF16 returns a mask where true indicates infinity.
func IsInfBF16(v Vec[BFloat16], sign int) Mask[BFloat16] {
	bits := make([]bool, len(v.data))
	for i := 0; i < len(v.data); i++ {
		if !v.data[i].IsInf() {
			bits[i] = false
			continue
		}
		switch sign {
		case 0:
			bits[i] = true
		case 1:
			bits[i] = !v.data[i].IsNegative()
		case -1:
			bits[i] = v.data[i].IsNegative()
		default:
			bits[i] = true
		}
	}
	return Mask[BFloat16]{bits: bits}
}

// IsFiniteBF16 returns a mask where true indicates finite (not NaN or Inf).
func IsFiniteBF16(v Vec[BFloat16]) Mask[BFloat16] {
	bits := make([]bool, len(v.data))
	for i := 0; i < len(v.data); i++ {
		bits[i] = !v.data[i].IsNaN() && !v.data[i].IsInf()
	}
	return Mask[BFloat16]{bits: bits}
}

// IfThenElseBF16 selects elements based on mask: mask ? yes : no.
func IfThenElseBF16(mask Mask[BFloat16], yes, no Vec[BFloat16]) Vec[BFloat16] {
	n := min(len(no.data), min(len(yes.data), len(mask.bits)))
	result := make([]BFloat16, n)
	for i := range n {
		if mask.bits[i] {
			result[i] = yes.data[i]
		} else {
			result[i] = no.data[i]
		}
	}
	return Vec[BFloat16]{data: result}
}

// ClampBF16 clamps values to [lo, hi] range.
func ClampBF16(v, lo, hi Vec[BFloat16]) Vec[BFloat16] {
	return MinBF16(MaxBF16(v, lo), hi)
}

// MulAddBF16 computes a*b + c (non-fused).
func MulAddBF16(a, b, c Vec[BFloat16]) Vec[BFloat16] {
	return AddBF16(MulBF16(a, b), c)
}

// MulSubBF16 computes a*b - c.
func MulSubBF16(a, b, c Vec[BFloat16]) Vec[BFloat16] {
	return SubBF16(MulBF16(a, b), c)
}

// NegMulAddBF16 computes -a*b + c = c - a*b.
func NegMulAddBF16(a, b, c Vec[BFloat16]) Vec[BFloat16] {
	return SubBF16(c, MulBF16(a, b))
}

// NegMulSubBF16 computes -a*b - c = -(a*b + c).
func NegMulSubBF16(a, b, c Vec[BFloat16]) Vec[BFloat16] {
	return NegBF16(AddBF16(MulBF16(a, b), c))
}
