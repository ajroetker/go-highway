package hwy

import "math/bits"

// This file provides bit manipulation operations for integer vectors.
// These are pure Go (scalar) implementations that work with any integer type.
// When SIMD implementations are available, they will be used via dispatch.

// PopCount counts the number of set bits (1s) in each lane.
func PopCount[T Integers](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = popCount(v.data[i])
	}
	return Vec[T]{data: result}
}

// popCount counts set bits for a single value.
func popCount[T Integers](val T) T {
	switch any(val).(type) {
	case int8:
		return T(bits.OnesCount8(uint8(any(val).(int8))))
	case uint8:
		return T(bits.OnesCount8(any(val).(uint8)))
	case int16:
		return T(bits.OnesCount16(uint16(any(val).(int16))))
	case uint16:
		return T(bits.OnesCount16(any(val).(uint16)))
	case int32:
		return T(bits.OnesCount32(uint32(any(val).(int32))))
	case uint32:
		return T(bits.OnesCount32(any(val).(uint32)))
	case int64:
		return T(bits.OnesCount64(uint64(any(val).(int64))))
	case uint64:
		return T(bits.OnesCount64(any(val).(uint64)))
	default:
		return 0
	}
}

// LeadingZeroCount counts the number of leading zero bits in each lane.
func LeadingZeroCount[T Integers](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = leadingZeroCount(v.data[i])
	}
	return Vec[T]{data: result}
}

// leadingZeroCount counts leading zeros for a single value.
func leadingZeroCount[T Integers](val T) T {
	switch any(val).(type) {
	case int8:
		return T(bits.LeadingZeros8(uint8(any(val).(int8))))
	case uint8:
		return T(bits.LeadingZeros8(any(val).(uint8)))
	case int16:
		return T(bits.LeadingZeros16(uint16(any(val).(int16))))
	case uint16:
		return T(bits.LeadingZeros16(any(val).(uint16)))
	case int32:
		return T(bits.LeadingZeros32(uint32(any(val).(int32))))
	case uint32:
		return T(bits.LeadingZeros32(any(val).(uint32)))
	case int64:
		return T(bits.LeadingZeros64(uint64(any(val).(int64))))
	case uint64:
		return T(bits.LeadingZeros64(any(val).(uint64)))
	default:
		return 0
	}
}

// TrailingZeroCount counts the number of trailing zero bits in each lane.
func TrailingZeroCount[T Integers](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = trailingZeroCount(v.data[i])
	}
	return Vec[T]{data: result}
}

// trailingZeroCount counts trailing zeros for a single value.
func trailingZeroCount[T Integers](val T) T {
	switch any(val).(type) {
	case int8:
		v := uint8(any(val).(int8))
		if v == 0 {
			return T(8)
		}
		return T(bits.TrailingZeros8(v))
	case uint8:
		v := any(val).(uint8)
		if v == 0 {
			return T(8)
		}
		return T(bits.TrailingZeros8(v))
	case int16:
		v := uint16(any(val).(int16))
		if v == 0 {
			return T(16)
		}
		return T(bits.TrailingZeros16(v))
	case uint16:
		v := any(val).(uint16)
		if v == 0 {
			return T(16)
		}
		return T(bits.TrailingZeros16(v))
	case int32:
		v := uint32(any(val).(int32))
		if v == 0 {
			return T(32)
		}
		return T(bits.TrailingZeros32(v))
	case uint32:
		v := any(val).(uint32)
		if v == 0 {
			return T(32)
		}
		return T(bits.TrailingZeros32(v))
	case int64:
		v := uint64(any(val).(int64))
		if v == 0 {
			return T(64)
		}
		return T(bits.TrailingZeros64(v))
	case uint64:
		v := any(val).(uint64)
		if v == 0 {
			return T(64)
		}
		return T(bits.TrailingZeros64(v))
	default:
		return 0
	}
}

// RotateRight rotates the bits in each lane to the right by the specified count.
func RotateRight[T Integers](v Vec[T], count int) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = rotateRight(v.data[i], count)
	}
	return Vec[T]{data: result}
}

// rotateRight rotates bits right for a single value.
func rotateRight[T Integers](val T, count int) T {
	switch any(val).(type) {
	case int8:
		v := uint8(any(val).(int8))
		return T(int8(bits.RotateLeft8(v, -count)))
	case uint8:
		v := any(val).(uint8)
		return T(bits.RotateLeft8(v, -count))
	case int16:
		v := uint16(any(val).(int16))
		return T(int16(bits.RotateLeft16(v, -count)))
	case uint16:
		v := any(val).(uint16)
		return T(bits.RotateLeft16(v, -count))
	case int32:
		v := uint32(any(val).(int32))
		return T(int32(bits.RotateLeft32(v, -count)))
	case uint32:
		v := any(val).(uint32)
		return T(bits.RotateLeft32(v, -count))
	case int64:
		v := uint64(any(val).(int64))
		return T(int64(bits.RotateLeft64(v, -count)))
	case uint64:
		v := any(val).(uint64)
		return T(bits.RotateLeft64(v, -count))
	default:
		return val
	}
}

// ReverseBits reverses the bit order in each lane.
func ReverseBits[T Integers](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = reverseBits(v.data[i])
	}
	return Vec[T]{data: result}
}

// reverseBits reverses bit order for a single value.
func reverseBits[T Integers](val T) T {
	switch any(val).(type) {
	case int8:
		v := uint8(any(val).(int8))
		return T(int8(bits.Reverse8(v)))
	case uint8:
		v := any(val).(uint8)
		return T(bits.Reverse8(v))
	case int16:
		v := uint16(any(val).(int16))
		return T(int16(bits.Reverse16(v)))
	case uint16:
		v := any(val).(uint16)
		return T(bits.Reverse16(v))
	case int32:
		v := uint32(any(val).(int32))
		return T(int32(bits.Reverse32(v)))
	case uint32:
		v := any(val).(uint32)
		return T(bits.Reverse32(v))
	case int64:
		v := uint64(any(val).(int64))
		return T(int64(bits.Reverse64(v)))
	case uint64:
		v := any(val).(uint64)
		return T(bits.Reverse64(v))
	default:
		return val
	}
}

// HighestSetBitIndex returns the index of the highest set bit in each lane.
// Returns -1 for lanes with value 0.
// For a value with bit pattern ...001xxx, returns the position of the leftmost 1.
// This is equivalent to floor(log2(x)) for non-zero values.
func HighestSetBitIndex[T Integers](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = highestSetBitIndex(v.data[i])
	}
	return Vec[T]{data: result}
}

// highestSetBitIndex returns the index of the highest set bit for a single value.
// Returns the maximum value of the type (all bits set) if the value is 0.
// For signed types this is -1, for unsigned types this is the max value.
func highestSetBitIndex[T Integers](val T) T {
	switch v := any(val).(type) {
	case int8:
		uv := uint8(v)
		if uv == 0 {
			return any(int8(-1)).(T)
		}
		return any(int8(bits.Len8(uv) - 1)).(T)
	case uint8:
		if v == 0 {
			return any(uint8(0xFF)).(T)
		}
		return any(uint8(bits.Len8(v) - 1)).(T)
	case int16:
		uv := uint16(v)
		if uv == 0 {
			return any(int16(-1)).(T)
		}
		return any(int16(bits.Len16(uv) - 1)).(T)
	case uint16:
		if v == 0 {
			return any(uint16(0xFFFF)).(T)
		}
		return any(uint16(bits.Len16(v) - 1)).(T)
	case int32:
		uv := uint32(v)
		if uv == 0 {
			return any(int32(-1)).(T)
		}
		return any(int32(bits.Len32(uv) - 1)).(T)
	case uint32:
		if v == 0 {
			return any(uint32(0xFFFFFFFF)).(T)
		}
		return any(uint32(bits.Len32(v) - 1)).(T)
	case int64:
		uv := uint64(v)
		if uv == 0 {
			return any(int64(-1)).(T)
		}
		return any(int64(bits.Len64(uv) - 1)).(T)
	case uint64:
		if v == 0 {
			return any(uint64(0xFFFFFFFFFFFFFFFF)).(T)
		}
		return any(uint64(bits.Len64(v) - 1)).(T)
	default:
		var zero T
		return zero
	}
}
