package hwy

import "math"

// This file provides saturated arithmetic and related operations.
// Saturated operations clamp results to the type's valid range instead of wrapping.

// SaturatedAdd performs element-wise addition with saturation.
// Results are clamped to the type's valid range instead of wrapping.
// For example, uint8: 250 + 10 = 255 (not 4)
func SaturatedAdd[T Integers](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		result[i] = saturatedAdd(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

// SaturatedSub performs element-wise subtraction with saturation.
// Results are clamped to the type's valid range instead of wrapping.
// For example, uint8: 10 - 20 = 0 (not 246)
func SaturatedSub[T Integers](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		result[i] = saturatedSub(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

// Clamp clamps each element to the range [lo, hi].
// Elements less than lo become lo, elements greater than hi become hi.
func Clamp[T Lanes](v, lo, hi Vec[T]) Vec[T] {
	n := min(len(hi.data), min(len(lo.data), len(v.data)))
	result := make([]T, n)
	for i := range n {
		val := v.data[i]
		if val < lo.data[i] {
			val = lo.data[i]
		}
		if val > hi.data[i] {
			val = hi.data[i]
		}
		result[i] = val
	}
	return Vec[T]{data: result}
}

// AbsDiff computes the absolute difference |a - b| for each element.
// For unsigned types, this is max(a,b) - min(a,b).
// For signed types, this is |a - b|.
func AbsDiff[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		result[i] = absDiff(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

// Avg computes the rounded average (a + b + 1) / 2 for each element.
// This is useful for image processing and avoids overflow.
func Avg[T Integers](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		result[i] = roundedAvg(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

// MulHigh returns the high bits of the widening multiplication a * b.
// For n-bit integers, multiplying produces 2n bits; this returns the upper n bits.
func MulHigh[T Integers](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		result[i] = mulHigh(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

// Helper functions for saturated arithmetic

func saturatedAdd[T Integers](a, b T) T {
	switch any(a).(type) {
	case int8:
		sum := int16(any(a).(int8)) + int16(any(b).(int8))
		if sum > 127 {
			return T(any(int8(127)).(int8))
		}
		if sum < -128 {
			return T(any(int8(-128)).(int8))
		}
		return T(any(int8(sum)).(int8))
	case int16:
		sum := int32(any(a).(int16)) + int32(any(b).(int16))
		if sum > 32767 {
			return T(any(int16(32767)).(int16))
		}
		if sum < -32768 {
			return T(any(int16(-32768)).(int16))
		}
		return T(any(int16(sum)).(int16))
	case int32:
		sum := int64(any(a).(int32)) + int64(any(b).(int32))
		if sum > 2147483647 {
			return T(any(int32(2147483647)).(int32))
		}
		if sum < -2147483648 {
			return T(any(int32(-2147483648)).(int32))
		}
		return T(any(int32(sum)).(int32))
	case int64:
		av := any(a).(int64)
		bv := any(b).(int64)
		// Check for overflow before adding
		if bv > 0 && av > math.MaxInt64-bv {
			return T(any(int64(math.MaxInt64)).(int64))
		}
		if bv < 0 && av < math.MinInt64-bv {
			return T(any(int64(math.MinInt64)).(int64))
		}
		return T(any(av + bv).(int64))
	case uint8:
		sum := uint16(any(a).(uint8)) + uint16(any(b).(uint8))
		if sum > 255 {
			return T(any(uint8(255)).(uint8))
		}
		return T(any(uint8(sum)).(uint8))
	case uint16:
		sum := uint32(any(a).(uint16)) + uint32(any(b).(uint16))
		if sum > 65535 {
			return T(any(uint16(65535)).(uint16))
		}
		return T(any(uint16(sum)).(uint16))
	case uint32:
		sum := uint64(any(a).(uint32)) + uint64(any(b).(uint32))
		if sum > 4294967295 {
			return T(any(uint32(4294967295)).(uint32))
		}
		return T(any(uint32(sum)).(uint32))
	case uint64:
		av := any(a).(uint64)
		bv := any(b).(uint64)
		// Check for overflow before adding
		if av > math.MaxUint64-bv {
			return T(any(uint64(math.MaxUint64)).(uint64))
		}
		return T(any(av + bv).(uint64))
	default:
		return a + b
	}
}

func saturatedSub[T Integers](a, b T) T {
	switch any(a).(type) {
	case int8:
		diff := int16(any(a).(int8)) - int16(any(b).(int8))
		if diff > 127 {
			return T(any(int8(127)).(int8))
		}
		if diff < -128 {
			return T(any(int8(-128)).(int8))
		}
		return T(any(int8(diff)).(int8))
	case int16:
		diff := int32(any(a).(int16)) - int32(any(b).(int16))
		if diff > 32767 {
			return T(any(int16(32767)).(int16))
		}
		if diff < -32768 {
			return T(any(int16(-32768)).(int16))
		}
		return T(any(int16(diff)).(int16))
	case int32:
		diff := int64(any(a).(int32)) - int64(any(b).(int32))
		if diff > 2147483647 {
			return T(any(int32(2147483647)).(int32))
		}
		if diff < -2147483648 {
			return T(any(int32(-2147483648)).(int32))
		}
		return T(any(int32(diff)).(int32))
	case int64:
		av := any(a).(int64)
		bv := any(b).(int64)
		// Check for overflow before subtracting
		if bv < 0 && av > math.MaxInt64+bv {
			return T(any(int64(math.MaxInt64)).(int64))
		}
		if bv > 0 && av < math.MinInt64+bv {
			return T(any(int64(math.MinInt64)).(int64))
		}
		return T(any(av - bv).(int64))
	case uint8:
		av := any(a).(uint8)
		bv := any(b).(uint8)
		if bv > av {
			return T(any(uint8(0)).(uint8))
		}
		return T(any(av - bv).(uint8))
	case uint16:
		av := any(a).(uint16)
		bv := any(b).(uint16)
		if bv > av {
			return T(any(uint16(0)).(uint16))
		}
		return T(any(av - bv).(uint16))
	case uint32:
		av := any(a).(uint32)
		bv := any(b).(uint32)
		if bv > av {
			return T(any(uint32(0)).(uint32))
		}
		return T(any(av - bv).(uint32))
	case uint64:
		av := any(a).(uint64)
		bv := any(b).(uint64)
		if bv > av {
			return T(any(uint64(0)).(uint64))
		}
		return T(any(av - bv).(uint64))
	default:
		return a - b
	}
}

func absDiff[T Lanes](a, b T) T {
	switch any(a).(type) {
	case float32:
		av := any(a).(float32)
		bv := any(b).(float32)
		diff := av - bv
		if diff < 0 {
			diff = -diff
		}
		return T(any(diff).(float32))
	case float64:
		av := any(a).(float64)
		bv := any(b).(float64)
		diff := av - bv
		if diff < 0 {
			diff = -diff
		}
		return T(any(diff).(float64))
	case int8:
		av := int16(any(a).(int8))
		bv := int16(any(b).(int8))
		diff := av - bv
		if diff < 0 {
			diff = -diff
		}
		return T(any(int8(diff)).(int8))
	case int16:
		av := int32(any(a).(int16))
		bv := int32(any(b).(int16))
		diff := av - bv
		if diff < 0 {
			diff = -diff
		}
		return T(any(int16(diff)).(int16))
	case int32:
		av := int64(any(a).(int32))
		bv := int64(any(b).(int32))
		diff := av - bv
		if diff < 0 {
			diff = -diff
		}
		return T(any(int32(diff)).(int32))
	case int64:
		av := any(a).(int64)
		bv := any(b).(int64)
		diff := av - bv
		if diff < 0 {
			diff = -diff
		}
		return T(any(diff).(int64))
	case uint8:
		av := any(a).(uint8)
		bv := any(b).(uint8)
		if av > bv {
			return T(any(av - bv).(uint8))
		}
		return T(any(bv - av).(uint8))
	case uint16:
		av := any(a).(uint16)
		bv := any(b).(uint16)
		if av > bv {
			return T(any(av - bv).(uint16))
		}
		return T(any(bv - av).(uint16))
	case uint32:
		av := any(a).(uint32)
		bv := any(b).(uint32)
		if av > bv {
			return T(any(av - bv).(uint32))
		}
		return T(any(bv - av).(uint32))
	case uint64:
		av := any(a).(uint64)
		bv := any(b).(uint64)
		if av > bv {
			return T(any(av - bv).(uint64))
		}
		return T(any(bv - av).(uint64))
	default:
		if a > b {
			return a - b
		}
		return b - a
	}
}

func roundedAvg[T Integers](a, b T) T {
	switch any(a).(type) {
	case int8:
		// Use larger type to avoid overflow
		sum := int16(any(a).(int8)) + int16(any(b).(int8)) + 1
		return T(any(int8(sum / 2)).(int8))
	case int16:
		sum := int32(any(a).(int16)) + int32(any(b).(int16)) + 1
		return T(any(int16(sum / 2)).(int16))
	case int32:
		sum := int64(any(a).(int32)) + int64(any(b).(int32)) + 1
		return T(any(int32(sum / 2)).(int32))
	case int64:
		// For int64, need to be careful about overflow
		av := any(a).(int64)
		bv := any(b).(int64)
		// (a + b + 1) / 2 = a/2 + b/2 + (a%2 + b%2 + 1)/2
		return T(any(av/2 + bv/2 + (av%2+bv%2+1)/2).(int64))
	case uint8:
		sum := uint16(any(a).(uint8)) + uint16(any(b).(uint8)) + 1
		return T(any(uint8(sum / 2)).(uint8))
	case uint16:
		sum := uint32(any(a).(uint16)) + uint32(any(b).(uint16)) + 1
		return T(any(uint16(sum / 2)).(uint16))
	case uint32:
		sum := uint64(any(a).(uint32)) + uint64(any(b).(uint32)) + 1
		return T(any(uint32(sum / 2)).(uint32))
	case uint64:
		// For uint64, need to be careful about overflow
		av := any(a).(uint64)
		bv := any(b).(uint64)
		// (a + b + 1) / 2 = a/2 + b/2 + (a%2 + b%2 + 1)/2
		return T(any(av/2 + bv/2 + (av%2+bv%2+1)/2).(uint64))
	default:
		return (a + b + 1) / 2
	}
}

func mulHigh[T Integers](a, b T) T {
	switch any(a).(type) {
	case int8:
		product := int16(any(a).(int8)) * int16(any(b).(int8))
		return T(any(int8(product >> 8)).(int8))
	case int16:
		product := int32(any(a).(int16)) * int32(any(b).(int16))
		return T(any(int16(product >> 16)).(int16))
	case int32:
		product := int64(any(a).(int32)) * int64(any(b).(int32))
		return T(any(int32(product >> 32)).(int32))
	case int64:
		// For 64-bit, we need 128-bit multiplication
		// Use the algorithm: (a * b) >> 64
		av := any(a).(int64)
		bv := any(b).(int64)
		return T(any(mulHigh64(av, bv)).(int64))
	case uint8:
		product := uint16(any(a).(uint8)) * uint16(any(b).(uint8))
		return T(any(uint8(product >> 8)).(uint8))
	case uint16:
		product := uint32(any(a).(uint16)) * uint32(any(b).(uint16))
		return T(any(uint16(product >> 16)).(uint16))
	case uint32:
		product := uint64(any(a).(uint32)) * uint64(any(b).(uint32))
		return T(any(uint32(product >> 32)).(uint32))
	case uint64:
		// For 64-bit, we need 128-bit multiplication
		av := any(a).(uint64)
		bv := any(b).(uint64)
		return T(any(mulHighU64(av, bv)).(uint64))
	default:
		return 0
	}
}

// mulHigh64 computes the high 64 bits of a signed 64-bit multiplication.
func mulHigh64(a, b int64) int64 {
	// Split into 32-bit parts
	negative := (a < 0) != (b < 0)
	if a < 0 {
		a = -a
	}
	if b < 0 {
		b = -b
	}

	hi := mulHighU64(uint64(a), uint64(b))

	if negative {
		// For negative results, we need to adjust
		lo := uint64(a) * uint64(b)
		if lo != 0 {
			hi = ^hi
		} else {
			hi = -hi
		}
	}
	return int64(hi)
}

// mulHighU64 computes the high 64 bits of an unsigned 64-bit multiplication.
func mulHighU64(a, b uint64) uint64 {
	// Split each 64-bit number into two 32-bit halves
	aLo := a & 0xFFFFFFFF
	aHi := a >> 32
	bLo := b & 0xFFFFFFFF
	bHi := b >> 32

	// Compute partial products
	// a * b = (aHi*2^32 + aLo) * (bHi*2^32 + bLo)
	//       = aHi*bHi*2^64 + (aHi*bLo + aLo*bHi)*2^32 + aLo*bLo
	p0 := aLo * bLo // Low 64 bits of low product
	p1 := aHi * bLo
	p2 := aLo * bHi
	p3 := aHi * bHi // High 64 bits directly

	// Sum the middle products and propagate carry
	mid := p1 + p2
	carry := uint64(0)
	if mid < p1 {
		carry = 1 << 32 // Overflow in middle sum
	}

	// High result = p3 + upper 32 bits of mid + carry + carry from (p0 + mid<<32)
	midHi := mid >> 32
	midLo := mid << 32

	// Check for carry from adding midLo to p0
	if p0+midLo < p0 {
		carry++
	}

	return p3 + midHi + carry
}
