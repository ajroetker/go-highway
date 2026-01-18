package hwy

import "math"

// This file provides Float16 arithmetic operations using the promote-compute-demote pattern.
// Since Go has no native float16 arithmetic, we:
// 1. Promote Float16 to float32
// 2. Perform the arithmetic in float32
// 3. Demote the result back to Float16
//
// When SIMD implementations are available (e.g., NEON FP16, AVX-512 FP16),
// native hardware operations will be used instead.

// AddF16 performs element-wise addition of two Float16 vectors.
func AddF16(a, b Vec[Float16]) Vec[Float16] {
	n := min(len(b.data), len(a.data))
	result := make([]Float16, n)
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		result[i] = Float32ToFloat16(af + bf)
	}
	return Vec[Float16]{data: result}
}

// SubF16 performs element-wise subtraction of two Float16 vectors.
func SubF16(a, b Vec[Float16]) Vec[Float16] {
	n := min(len(b.data), len(a.data))
	result := make([]Float16, n)
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		result[i] = Float32ToFloat16(af - bf)
	}
	return Vec[Float16]{data: result}
}

// MulF16 performs element-wise multiplication of two Float16 vectors.
func MulF16(a, b Vec[Float16]) Vec[Float16] {
	n := min(len(b.data), len(a.data))
	result := make([]Float16, n)
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		result[i] = Float32ToFloat16(af * bf)
	}
	return Vec[Float16]{data: result}
}

// DivF16 performs element-wise division of two Float16 vectors.
func DivF16(a, b Vec[Float16]) Vec[Float16] {
	n := min(len(b.data), len(a.data))
	result := make([]Float16, n)
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		result[i] = Float32ToFloat16(af / bf)
	}
	return Vec[Float16]{data: result}
}

// FMAF16 performs fused multiply-add: a * b + c.
// Using FMA in float32 preserves precision better than separate mul and add.
func FMAF16(a, b, c Vec[Float16]) Vec[Float16] {
	n := min(len(c.data), min(len(b.data), len(a.data)))
	result := make([]Float16, n)
	for i := range n {
		af := float64(Float16ToFloat32(a.data[i]))
		bf := float64(Float16ToFloat32(b.data[i]))
		cf := float64(Float16ToFloat32(c.data[i]))
		result[i] = Float32ToFloat16(float32(math.FMA(af, bf, cf)))
	}
	return Vec[Float16]{data: result}
}

// NegF16 negates all lanes.
func NegF16(v Vec[Float16]) Vec[Float16] {
	result := make([]Float16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		// Negation is just flipping the sign bit
		result[i] = v.data[i] ^ 0x8000
	}
	return Vec[Float16]{data: result}
}

// AbsF16 computes absolute value.
func AbsF16(v Vec[Float16]) Vec[Float16] {
	result := make([]Float16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		// Absolute value is just clearing the sign bit
		result[i] = v.data[i] & 0x7FFF
	}
	return Vec[Float16]{data: result}
}

// MinF16 returns element-wise minimum.
func MinF16(a, b Vec[Float16]) Vec[Float16] {
	n := min(len(b.data), len(a.data))
	result := make([]Float16, n)
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		if af < bf {
			result[i] = a.data[i]
		} else {
			result[i] = b.data[i]
		}
	}
	return Vec[Float16]{data: result}
}

// MaxF16 returns element-wise maximum.
func MaxF16(a, b Vec[Float16]) Vec[Float16] {
	n := min(len(b.data), len(a.data))
	result := make([]Float16, n)
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		if af > bf {
			result[i] = a.data[i]
		} else {
			result[i] = b.data[i]
		}
	}
	return Vec[Float16]{data: result}
}

// SqrtF16 computes square root.
func SqrtF16(v Vec[Float16]) Vec[Float16] {
	result := make([]Float16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		f := float64(Float16ToFloat32(v.data[i]))
		result[i] = Float32ToFloat16(float32(math.Sqrt(f)))
	}
	return Vec[Float16]{data: result}
}

// ReciprocalF16 computes 1/x (reciprocal).
func ReciprocalF16(v Vec[Float16]) Vec[Float16] {
	result := make([]Float16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		f := Float16ToFloat32(v.data[i])
		result[i] = Float32ToFloat16(1.0 / f)
	}
	return Vec[Float16]{data: result}
}

// ReciprocalSqrtF16 computes 1/sqrt(x).
func ReciprocalSqrtF16(v Vec[Float16]) Vec[Float16] {
	result := make([]Float16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		f := float64(Float16ToFloat32(v.data[i]))
		result[i] = Float32ToFloat16(float32(1.0 / math.Sqrt(f)))
	}
	return Vec[Float16]{data: result}
}

// ReduceSumF16 sums all lanes and returns the result as float32.
// Accumulation is done in float32 to avoid precision loss.
func ReduceSumF16(v Vec[Float16]) float32 {
	var sum float32
	for i := 0; i < len(v.data); i++ {
		sum += Float16ToFloat32(v.data[i])
	}
	return sum
}

// ReduceMinF16 returns the minimum value across all lanes.
func ReduceMinF16(v Vec[Float16]) Float16 {
	if len(v.data) == 0 {
		return Float16Inf // Return +Inf for empty vector
	}
	min := v.data[0]
	minF := Float16ToFloat32(min)
	for i := 1; i < len(v.data); i++ {
		f := Float16ToFloat32(v.data[i])
		if f < minF {
			min = v.data[i]
			minF = f
		}
	}
	return min
}

// ReduceMaxF16 returns the maximum value across all lanes.
func ReduceMaxF16(v Vec[Float16]) Float16 {
	if len(v.data) == 0 {
		return Float16NegInf // Return -Inf for empty vector
	}
	max := v.data[0]
	maxF := Float16ToFloat32(max)
	for i := 1; i < len(v.data); i++ {
		f := Float16ToFloat32(v.data[i])
		if f > maxF {
			max = v.data[i]
			maxF = f
		}
	}
	return max
}

// DotF16 computes dot product of two Float16 vectors, returning float32.
// This is the common ML pattern: accumulate in higher precision.
func DotF16(a, b Vec[Float16]) float32 {
	n := min(len(b.data), len(a.data))
	var sum float32
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		sum += af * bf
	}
	return sum
}

// EqualF16 compares two Float16 vectors for equality.
func EqualF16(a, b Vec[Float16]) Mask[Float16] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		bits[i] = af == bf
	}
	return Mask[Float16]{bits: bits}
}

// LessThanF16 compares a < b element-wise.
func LessThanF16(a, b Vec[Float16]) Mask[Float16] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		bits[i] = af < bf
	}
	return Mask[Float16]{bits: bits}
}

// LessThanOrEqualF16 compares a <= b element-wise.
func LessThanOrEqualF16(a, b Vec[Float16]) Mask[Float16] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		bits[i] = af <= bf
	}
	return Mask[Float16]{bits: bits}
}

// GreaterThanF16 compares a > b element-wise.
func GreaterThanF16(a, b Vec[Float16]) Mask[Float16] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		bits[i] = af > bf
	}
	return Mask[Float16]{bits: bits}
}

// GreaterThanOrEqualF16 compares a >= b element-wise.
func GreaterThanOrEqualF16(a, b Vec[Float16]) Mask[Float16] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		af := Float16ToFloat32(a.data[i])
		bf := Float16ToFloat32(b.data[i])
		bits[i] = af >= bf
	}
	return Mask[Float16]{bits: bits}
}

// IsNaNF16 returns a mask where true indicates NaN.
func IsNaNF16(v Vec[Float16]) Mask[Float16] {
	bits := make([]bool, len(v.data))
	for i := 0; i < len(v.data); i++ {
		bits[i] = v.data[i].IsNaN()
	}
	return Mask[Float16]{bits: bits}
}

// IsInfF16 returns a mask where true indicates infinity.
// sign: 0 = any infinity, 1 = positive infinity, -1 = negative infinity
func IsInfF16(v Vec[Float16], sign int) Mask[Float16] {
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
	return Mask[Float16]{bits: bits}
}

// IsFiniteF16 returns a mask where true indicates finite (not NaN or Inf).
func IsFiniteF16(v Vec[Float16]) Mask[Float16] {
	bits := make([]bool, len(v.data))
	for i := 0; i < len(v.data); i++ {
		bits[i] = !v.data[i].IsNaN() && !v.data[i].IsInf()
	}
	return Mask[Float16]{bits: bits}
}

// IfThenElseF16 selects elements based on mask: mask ? yes : no.
func IfThenElseF16(mask Mask[Float16], yes, no Vec[Float16]) Vec[Float16] {
	n := min(len(no.data), min(len(yes.data), len(mask.bits)))
	result := make([]Float16, n)
	for i := range n {
		if mask.bits[i] {
			result[i] = yes.data[i]
		} else {
			result[i] = no.data[i]
		}
	}
	return Vec[Float16]{data: result}
}

// ClampF16 clamps values to [lo, hi] range.
func ClampF16(v, lo, hi Vec[Float16]) Vec[Float16] {
	return MinF16(MaxF16(v, lo), hi)
}

// MulAddF16 computes a*b + c (non-fused, for consistency with integer ops).
func MulAddF16(a, b, c Vec[Float16]) Vec[Float16] {
	return AddF16(MulF16(a, b), c)
}

// MulSubF16 computes a*b - c.
func MulSubF16(a, b, c Vec[Float16]) Vec[Float16] {
	return SubF16(MulF16(a, b), c)
}

// NegMulAddF16 computes -a*b + c = c - a*b.
func NegMulAddF16(a, b, c Vec[Float16]) Vec[Float16] {
	return SubF16(c, MulF16(a, b))
}

// NegMulSubF16 computes -a*b - c = -(a*b + c).
func NegMulSubF16(a, b, c Vec[Float16]) Vec[Float16] {
	return NegF16(AddF16(MulF16(a, b), c))
}
