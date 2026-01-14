//go:build !noasm && arm64

package asm

import (
	"math"
)

// Float32x4 represents a 128-bit NEON vector of 4 float32 values.
// This provides an API similar to archsimd.Float32x4 for NEON.
type Float32x4 [4]float32

// Float64x2 represents a 128-bit NEON vector of 2 float64 values.
// This provides an API similar to archsimd.Float64x2 for NEON.
type Float64x2 [2]float64

// Int32x4 represents a 128-bit NEON vector of 4 int32 values.
type Int32x4 [4]int32

// Int64x2 represents a 128-bit NEON vector of 2 int64 values.
type Int64x2 [2]int64

// Int32x2 represents a 64-bit vector of 2 int32 values.
// Used for type conversions from Float64x2.
type Int32x2 [2]int32

// ===== Float32x4 constructors =====

// BroadcastFloat32x4 creates a vector with all lanes set to the given value.
func BroadcastFloat32x4(v float32) Float32x4 {
	return Float32x4{v, v, v, v}
}

// LoadFloat32x4 loads 4 float32 values from a slice.
func LoadFloat32x4(s []float32) Float32x4 {
	var v Float32x4
	copy(v[:], s[:4])
	return v
}

// LoadFloat32x4Slice is an alias for LoadFloat32x4 (matches archsimd naming).
func LoadFloat32x4Slice(s []float32) Float32x4 {
	return LoadFloat32x4(s)
}

// ZeroFloat32x4 returns a zero vector.
func ZeroFloat32x4() Float32x4 {
	return Float32x4{}
}

// ===== Float32x4 methods =====

// StoreSlice stores the vector to a slice.
func (v Float32x4) StoreSlice(s []float32) {
	copy(s[:4], v[:])
}

// Add performs element-wise addition.
func (v Float32x4) Add(other Float32x4) Float32x4 {
	var result Float32x4
	AddF32(v[:], other[:], result[:])
	return result
}

// Sub performs element-wise subtraction.
func (v Float32x4) Sub(other Float32x4) Float32x4 {
	var result Float32x4
	SubF32(v[:], other[:], result[:])
	return result
}

// Mul performs element-wise multiplication.
func (v Float32x4) Mul(other Float32x4) Float32x4 {
	var result Float32x4
	MulF32(v[:], other[:], result[:])
	return result
}

// Div performs element-wise division.
func (v Float32x4) Div(other Float32x4) Float32x4 {
	var result Float32x4
	DivF32(v[:], other[:], result[:])
	return result
}

// Min performs element-wise minimum.
func (v Float32x4) Min(other Float32x4) Float32x4 {
	var result Float32x4
	MinF32(v[:], other[:], result[:])
	return result
}

// Max performs element-wise maximum.
func (v Float32x4) Max(other Float32x4) Float32x4 {
	var result Float32x4
	MaxF32(v[:], other[:], result[:])
	return result
}

// Sqrt performs element-wise square root.
func (v Float32x4) Sqrt() Float32x4 {
	var result Float32x4
	SqrtF32(v[:], result[:])
	return result
}

// Abs performs element-wise absolute value.
func (v Float32x4) Abs() Float32x4 {
	var result Float32x4
	AbsF32(v[:], result[:])
	return result
}

// Neg performs element-wise negation.
func (v Float32x4) Neg() Float32x4 {
	var result Float32x4
	NegF32(v[:], result[:])
	return result
}

// MulAdd performs fused multiply-add: v * a + b
func (v Float32x4) MulAdd(a, b Float32x4) Float32x4 {
	var result Float32x4
	FmaF32(v[:], a[:], b[:], result[:])
	return result
}

// ReduceSum returns the sum of all elements.
func (v Float32x4) ReduceSum() float32 {
	return ReduceSumF32(v[:])
}

// ReduceMin returns the minimum element.
func (v Float32x4) ReduceMin() float32 {
	return ReduceMinF32(v[:])
}

// ReduceMax returns the maximum element.
func (v Float32x4) ReduceMax() float32 {
	return ReduceMaxF32(v[:])
}

// Greater returns a mask where v > other.
func (v Float32x4) Greater(other Float32x4) Int32x4 {
	var result Int32x4
	GtF32(v[:], other[:], result[:])
	return result
}

// Less returns a mask where v < other.
func (v Float32x4) Less(other Float32x4) Int32x4 {
	var result Int32x4
	LtF32(v[:], other[:], result[:])
	return result
}

// GreaterEqual returns a mask where v >= other.
func (v Float32x4) GreaterEqual(other Float32x4) Int32x4 {
	var result Int32x4
	GeF32(v[:], other[:], result[:])
	return result
}

// LessEqual returns a mask where v <= other.
func (v Float32x4) LessEqual(other Float32x4) Int32x4 {
	var result Int32x4
	LeF32(v[:], other[:], result[:])
	return result
}

// Equal returns a mask where v == other.
func (v Float32x4) Equal(other Float32x4) Int32x4 {
	var result Int32x4
	EqF32(v[:], other[:], result[:])
	return result
}

// Merge selects elements: mask ? v : other
// Note: mask values should be all-ones (-1) for true, all-zeros (0) for false.
func (v Float32x4) Merge(other Float32x4, mask Int32x4) Float32x4 {
	var result Float32x4
	IfThenElseF32(mask[:], v[:], other[:], result[:])
	return result
}

// RoundToEven rounds to nearest even.
func (v Float32x4) RoundToEven() Float32x4 {
	var result Float32x4
	RoundF32(v[:], result[:])
	return result
}

// ConvertToInt32 converts to int32.
func (v Float32x4) ConvertToInt32() Int32x4 {
	var result Int32x4
	ConvertF32ToI32(v[:], result[:])
	return result
}

// AsInt32x4 reinterprets bits as int32.
func (v Float32x4) AsInt32x4() Int32x4 {
	var result Int32x4
	for i := 0; i < 4; i++ {
		result[i] = int32(math.Float32bits(v[i]))
	}
	return result
}

// GetExponent extracts the unbiased exponent from IEEE 754 floats.
func (v Float32x4) GetExponent() Int32x4 {
	var result Int32x4
	for i := 0; i < 4; i++ {
		bits := math.Float32bits(v[i])
		exp := int32((bits >> 23) & 0xFF)
		if exp == 0 || exp == 255 {
			result[i] = 0 // denormal or special
		} else {
			result[i] = exp - 127
		}
	}
	return result
}

// GetMantissa extracts the mantissa with exponent normalized to [1, 2).
func (v Float32x4) GetMantissa() Float32x4 {
	var result Float32x4
	for i := 0; i < 4; i++ {
		bits := math.Float32bits(v[i])
		// Clear exponent, set to 127 (2^0 = 1)
		mantissa := (bits & 0x807FFFFF) | 0x3F800000
		result[i] = math.Float32frombits(mantissa)
	}
	return result
}

// ===== Float64x2 constructors =====

// BroadcastFloat64x2 creates a vector with all lanes set to the given value.
func BroadcastFloat64x2(v float64) Float64x2 {
	return Float64x2{v, v}
}

// LoadFloat64x2 loads 2 float64 values from a slice.
func LoadFloat64x2(s []float64) Float64x2 {
	var v Float64x2
	copy(v[:], s[:2])
	return v
}

// LoadFloat64x2Slice is an alias for LoadFloat64x2 (matches archsimd naming).
func LoadFloat64x2Slice(s []float64) Float64x2 {
	return LoadFloat64x2(s)
}

// ZeroFloat64x2 returns a zero vector.
func ZeroFloat64x2() Float64x2 {
	return Float64x2{}
}

// ===== Float64x2 methods =====

// StoreSlice stores the vector to a slice.
func (v Float64x2) StoreSlice(s []float64) {
	copy(s[:2], v[:])
}

// Add performs element-wise addition.
func (v Float64x2) Add(other Float64x2) Float64x2 {
	var result Float64x2
	AddF64(v[:], other[:], result[:])
	return result
}

// Sub performs element-wise subtraction.
func (v Float64x2) Sub(other Float64x2) Float64x2 {
	var result Float64x2
	SubF64(v[:], other[:], result[:])
	return result
}

// Mul performs element-wise multiplication.
func (v Float64x2) Mul(other Float64x2) Float64x2 {
	var result Float64x2
	MulF64(v[:], other[:], result[:])
	return result
}

// Div performs element-wise division.
func (v Float64x2) Div(other Float64x2) Float64x2 {
	var result Float64x2
	DivF64(v[:], other[:], result[:])
	return result
}

// Min performs element-wise minimum.
func (v Float64x2) Min(other Float64x2) Float64x2 {
	var result Float64x2
	MinF64(v[:], other[:], result[:])
	return result
}

// Max performs element-wise maximum.
func (v Float64x2) Max(other Float64x2) Float64x2 {
	var result Float64x2
	MaxF64(v[:], other[:], result[:])
	return result
}

// Sqrt performs element-wise square root.
func (v Float64x2) Sqrt() Float64x2 {
	var result Float64x2
	SqrtF64(v[:], result[:])
	return result
}

// Abs performs element-wise absolute value.
func (v Float64x2) Abs() Float64x2 {
	var result Float64x2
	AbsF64(v[:], result[:])
	return result
}

// Neg performs element-wise negation.
func (v Float64x2) Neg() Float64x2 {
	var result Float64x2
	NegF64(v[:], result[:])
	return result
}

// MulAdd performs fused multiply-add: v * a + b
func (v Float64x2) MulAdd(a, b Float64x2) Float64x2 {
	var result Float64x2
	FmaF64(v[:], a[:], b[:], result[:])
	return result
}

// ReduceSum returns the sum of all elements.
func (v Float64x2) ReduceSum() float64 {
	return ReduceSumF64(v[:])
}

// ReduceMin returns the minimum element.
func (v Float64x2) ReduceMin() float64 {
	return ReduceMinF64(v[:])
}

// ReduceMax returns the maximum element.
func (v Float64x2) ReduceMax() float64 {
	return ReduceMaxF64(v[:])
}

// Greater returns a mask where v > other.
// Note: Returns Int64x2 mask for Float64x2 comparisons.
func (v Float64x2) Greater(other Float64x2) Int64x2 {
	var result Int64x2
	GtF64(v[:], other[:], result[:])
	return result
}

// Less returns a mask where v < other.
func (v Float64x2) Less(other Float64x2) Int64x2 {
	var result Int64x2
	LtF64(v[:], other[:], result[:])
	return result
}

// LessEqual returns a mask where v <= other.
func (v Float64x2) LessEqual(other Float64x2) Int64x2 {
	var result Int64x2
	LeF64(v[:], other[:], result[:])
	return result
}

// GreaterEqual returns a mask where v >= other.
func (v Float64x2) GreaterEqual(other Float64x2) Int64x2 {
	var result Int64x2
	GeF64(v[:], other[:], result[:])
	return result
}

// Merge selects elements: mask ? v : other
func (v Float64x2) Merge(other Float64x2, mask Int64x2) Float64x2 {
	var result Float64x2
	for i := 0; i < 2; i++ {
		if mask[i] != 0 {
			result[i] = v[i]
		} else {
			result[i] = other[i]
		}
	}
	return result
}

// RoundToEven rounds to nearest even.
func (v Float64x2) RoundToEven() Float64x2 {
	// Use scalar rounding for now
	var result Float64x2
	for i := 0; i < 2; i++ {
		result[i] = math.RoundToEven(v[i])
	}
	return result
}

// ConvertToInt32 converts float64 to int32 (truncate toward zero).
func (v Float64x2) ConvertToInt32() Int32x2 {
	var result Int32x2
	for i := 0; i < 2; i++ {
		result[i] = int32(v[i])
	}
	return result
}

// ConvertToInt64 converts float64 to int64 (truncate toward zero).
func (v Float64x2) ConvertToInt64() Int64x2 {
	var result Int64x2
	for i := 0; i < 2; i++ {
		result[i] = int64(v[i])
	}
	return result
}

// AsInt64x2 reinterprets bits as int64.
func (v Float64x2) AsInt64x2() Int64x2 {
	var result Int64x2
	for i := 0; i < 2; i++ {
		result[i] = int64(math.Float64bits(v[i]))
	}
	return result
}

// Equal returns a mask where v == other.
func (v Float64x2) Equal(other Float64x2) Int64x2 {
	var result Int64x2
	EqF64(v[:], other[:], result[:])
	return result
}

// GetExponent extracts the unbiased exponent from IEEE 754 floats.
func (v Float64x2) GetExponent() Int32x2 {
	var result Int32x2
	for i := 0; i < 2; i++ {
		bits := math.Float64bits(v[i])
		exp := int32((bits >> 52) & 0x7FF)
		if exp == 0 || exp == 2047 {
			result[i] = 0 // denormal or special
		} else {
			result[i] = exp - 1023
		}
	}
	return result
}

// GetMantissa extracts the mantissa with exponent normalized to [1, 2).
func (v Float64x2) GetMantissa() Float64x2 {
	var result Float64x2
	for i := 0; i < 2; i++ {
		bits := math.Float64bits(v[i])
		// Clear exponent, set to 1023 (2^0 = 1)
		mantissa := (bits & 0x800FFFFFFFFFFFFF) | 0x3FF0000000000000
		result[i] = math.Float64frombits(mantissa)
	}
	return result
}

// ===== Int32x2 constructors and methods =====

// BroadcastInt32x2 creates a vector with all lanes set to the given value.
func BroadcastInt32x2(v int32) Int32x2 {
	return Int32x2{v, v}
}

// Add performs element-wise addition.
func (v Int32x2) Add(other Int32x2) Int32x2 {
	return Int32x2{v[0] + other[0], v[1] + other[1]}
}

// Sub performs element-wise subtraction.
func (v Int32x2) Sub(other Int32x2) Int32x2 {
	return Int32x2{v[0] - other[0], v[1] - other[1]}
}

// Mul performs element-wise multiplication.
func (v Int32x2) Mul(other Int32x2) Int32x2 {
	return Int32x2{v[0] * other[0], v[1] * other[1]}
}

// ShiftAllLeft shifts all elements left by the given count.
func (v Int32x2) ShiftAllLeft(count int) Int32x2 {
	return Int32x2{v[0] << count, v[1] << count}
}

// Pow2Float64 computes 2^k for each lane and returns Float64x2.
// Uses IEEE 754 bit manipulation: 2^k = ((k + 1023) << 52) as float64 bits.
func (v Int32x2) Pow2Float64() Float64x2 {
	var result Float64x2
	Pow2F64(v[:], result[:])
	return result
}

// ===== Int32x4 constructors and methods =====

// BroadcastInt32x4 creates a vector with all lanes set to the given value.
func BroadcastInt32x4(v int32) Int32x4 {
	return Int32x4{v, v, v, v}
}

// Add performs element-wise addition.
func (v Int32x4) Add(other Int32x4) Int32x4 {
	var result Int32x4
	AddI32(v[:], other[:], result[:])
	return result
}

// Sub performs element-wise subtraction.
func (v Int32x4) Sub(other Int32x4) Int32x4 {
	var result Int32x4
	SubI32(v[:], other[:], result[:])
	return result
}

// Mul performs element-wise multiplication.
func (v Int32x4) Mul(other Int32x4) Int32x4 {
	var result Int32x4
	MulI32(v[:], other[:], result[:])
	return result
}

// ShiftAllLeft shifts all elements left by the given count.
func (v Int32x4) ShiftAllLeft(count int) Int32x4 {
	var result Int32x4
	ShiftLeftI32(v[:], count, result[:])
	return result
}

// ShiftAllRight shifts all elements right by the given count.
func (v Int32x4) ShiftAllRight(count int) Int32x4 {
	var result Int32x4
	ShiftRightI32(v[:], count, result[:])
	return result
}

// AsFloat32x4 reinterprets bits as float32.
func (v Int32x4) AsFloat32x4() Float32x4 {
	var result Float32x4
	for i := 0; i < 4; i++ {
		result[i] = math.Float32frombits(uint32(v[i]))
	}
	return result
}

// ConvertToFloat32 converts int32 to float32.
func (v Int32x4) ConvertToFloat32() Float32x4 {
	var result Float32x4
	ConvertI32ToF32(v[:], result[:])
	return result
}

// Pow2Float32 computes 2^k for each lane and returns Float32x4.
// Uses IEEE 754 bit manipulation: 2^k = ((k + 127) << 23) as float32 bits.
func (v Int32x4) Pow2Float32() Float32x4 {
	var result Float32x4
	Pow2F32(v[:], result[:])
	return result
}

// And performs element-wise bitwise AND.
func (v Int32x4) And(other Int32x4) Int32x4 {
	var result Int32x4
	AndI32(v[:], other[:], result[:])
	return result
}

// Or performs element-wise bitwise OR.
func (v Int32x4) Or(other Int32x4) Int32x4 {
	var result Int32x4
	OrI32(v[:], other[:], result[:])
	return result
}

// Xor performs element-wise bitwise XOR.
func (v Int32x4) Xor(other Int32x4) Int32x4 {
	var result Int32x4
	XorI32(v[:], other[:], result[:])
	return result
}

// Equal performs element-wise equality comparison.
// Returns -1 (all bits set) for true, 0 for false - matches archsimd convention.
func (v Int32x4) Equal(other Int32x4) Int32x4 {
	var result Int32x4
	for i := 0; i < 4; i++ {
		if v[i] == other[i] {
			result[i] = -1 // all bits set
		} else {
			result[i] = 0
		}
	}
	return result
}

// Merge selects elements: mask ? v : other
// mask should have -1 (all bits set) for true, 0 for false.
func (v Int32x4) Merge(other Int32x4, mask Int32x4) Int32x4 {
	var result Int32x4
	for i := 0; i < 4; i++ {
		if mask[i] != 0 {
			result[i] = v[i]
		} else {
			result[i] = other[i]
		}
	}
	return result
}

// StoreSlice stores the vector to a slice.
func (v Int32x4) StoreSlice(s []int32) {
	copy(s[:4], v[:])
}

// Data returns the underlying array as a slice.
func (v Int32x4) Data() []int32 {
	return v[:]
}

// ===== Int32x2 additional methods =====

// And performs element-wise bitwise AND.
func (v Int32x2) And(other Int32x2) Int32x2 {
	var result Int32x2
	for i := 0; i < 2; i++ {
		result[i] = v[i] & other[i]
	}
	return result
}

// Or performs element-wise bitwise OR.
func (v Int32x2) Or(other Int32x2) Int32x2 {
	var result Int32x2
	for i := 0; i < 2; i++ {
		result[i] = v[i] | other[i]
	}
	return result
}

// Xor performs element-wise bitwise XOR.
func (v Int32x2) Xor(other Int32x2) Int32x2 {
	var result Int32x2
	for i := 0; i < 2; i++ {
		result[i] = v[i] ^ other[i]
	}
	return result
}

// Equal performs element-wise equality comparison.
// Returns -1 (all bits set) for true, 0 for false - matches archsimd convention.
func (v Int32x2) Equal(other Int32x2) Int32x2 {
	var result Int32x2
	for i := 0; i < 2; i++ {
		if v[i] == other[i] {
			result[i] = -1 // all bits set
		} else {
			result[i] = 0
		}
	}
	return result
}

// Merge selects elements: mask ? v : other
// mask should have -1 (all bits set) for true, 0 for false.
func (v Int32x2) Merge(other Int32x2, mask Int32x2) Int32x2 {
	var result Int32x2
	for i := 0; i < 2; i++ {
		if mask[i] != 0 {
			result[i] = v[i]
		} else {
			result[i] = other[i]
		}
	}
	return result
}

// StoreSlice stores the vector to a slice.
func (v Int32x2) StoreSlice(s []int32) {
	copy(s[:2], v[:])
}

// Data returns the underlying array as a slice.
func (v Int32x2) Data() []int32 {
	return v[:]
}

// ===== Bool mask types for conditional operations =====

// BoolMask32x4 represents a 4-element boolean mask.
type BoolMask32x4 [4]bool

// GetBit returns the boolean value at the given index.
func (m BoolMask32x4) GetBit(i int) bool {
	return m[i]
}

// BoolMask32x2 represents a 2-element boolean mask.
type BoolMask32x2 [2]bool

// GetBit returns the boolean value at the given index.
func (m BoolMask32x2) GetBit(i int) bool {
	return m[i]
}

// ===== Data accessor methods =====

// Data returns the underlying array as a slice.
func (v Float32x4) Data() []float32 {
	return v[:]
}

// Data returns the underlying array as a slice.
func (v Float64x2) Data() []float64 {
	return v[:]
}

// ===== Int64x2 constructors and methods =====

// BroadcastInt64x2 creates a vector with all lanes set to the given value.
func BroadcastInt64x2(v int64) Int64x2 {
	return Int64x2{v, v}
}

// ShiftAllRight shifts all elements right by the given count.
func (v Int64x2) ShiftAllRight(count int) Int64x2 {
	var result Int64x2
	ShiftRightI64(v[:], count, result[:])
	return result
}

// ShiftAllLeft shifts all elements left by the given count.
func (v Int64x2) ShiftAllLeft(count int) Int64x2 {
	var result Int64x2
	ShiftLeftI64(v[:], count, result[:])
	return result
}

// StoreSlice stores the vector to a slice.
func (v Int64x2) StoreSlice(s []int64) {
	copy(s[:2], v[:])
}

// Add performs element-wise addition.
func (v Int64x2) Add(other Int64x2) Int64x2 {
	var result Int64x2
	AddI64(v[:], other[:], result[:])
	return result
}

// Sub performs element-wise subtraction.
func (v Int64x2) Sub(other Int64x2) Int64x2 {
	var result Int64x2
	SubI64(v[:], other[:], result[:])
	return result
}

// Mul performs element-wise multiplication.
func (v Int64x2) Mul(other Int64x2) Int64x2 {
	// Note: NEON doesn't have native 64-bit integer multiply, so keep scalar
	return Int64x2{v[0] * other[0], v[1] * other[1]}
}

// And performs element-wise bitwise AND.
func (v Int64x2) And(other Int64x2) Int64x2 {
	var result Int64x2
	AndI64(v[:], other[:], result[:])
	return result
}

// Or performs element-wise bitwise OR.
func (v Int64x2) Or(other Int64x2) Int64x2 {
	var result Int64x2
	OrI64(v[:], other[:], result[:])
	return result
}

// Xor performs element-wise bitwise XOR.
func (v Int64x2) Xor(other Int64x2) Int64x2 {
	var result Int64x2
	XorI64(v[:], other[:], result[:])
	return result
}

// Equal performs element-wise equality comparison.
// Returns -1 (all bits set) for true, 0 for false - matches archsimd convention.
func (v Int64x2) Equal(other Int64x2) Int64x2 {
	var result Int64x2
	for i := 0; i < 2; i++ {
		if v[i] == other[i] {
			result[i] = -1 // all bits set
		} else {
			result[i] = 0
		}
	}
	return result
}

// AsFloat64x2 reinterprets bits as float64.
func (v Int64x2) AsFloat64x2() Float64x2 {
	var result Float64x2
	for i := 0; i < 2; i++ {
		result[i] = math.Float64frombits(uint64(v[i]))
	}
	return result
}

// ConvertToFloat64 converts int64 to float64.
func (v Int64x2) ConvertToFloat64() Float64x2 {
	var result Float64x2
	for i := 0; i < 2; i++ {
		result[i] = float64(v[i])
	}
	return result
}

// ===== Int32x4 Load =====

// LoadInt32x4 loads 4 int32 values from a slice.
func LoadInt32x4(s []int32) Int32x4 {
	var v Int32x4
	copy(v[:], s[:4])
	return v
}

// LoadInt32x4Slice is an alias for LoadInt32x4 (matches archsimd naming).
func LoadInt32x4Slice(s []int32) Int32x4 {
	return LoadInt32x4(s)
}

// ===== Int64x2 Load =====

// LoadInt64x2 loads 2 int64 values from a slice.
func LoadInt64x2(s []int64) Int64x2 {
	var v Int64x2
	copy(v[:], s[:2])
	return v
}

// LoadInt64x2Slice is an alias for LoadInt64x2 (matches archsimd naming).
func LoadInt64x2Slice(s []int64) Int64x2 {
	return LoadInt64x2(s)
}

// ===== Mask operations =====

// FindFirstTrue returns the index of the first true lane in the mask, or -1 if none.
// For Int32x4 masks, a non-zero value (typically -1/0xFFFFFFFF) indicates true.
// Note: Pure Go is faster than assembly for single-vector operations due to inlining.
func FindFirstTrue[T Int32x4 | Int64x2](mask T) int {
	switch m := any(mask).(type) {
	case Int32x4:
		for i := 0; i < 4; i++ {
			if m[i] != 0 {
				return i
			}
		}
	case Int64x2:
		for i := 0; i < 2; i++ {
			if m[i] != 0 {
				return i
			}
		}
	}
	return -1
}

// CountTrue returns the number of true lanes in the mask.
// For integer masks, a non-zero value indicates true.
// Note: Pure Go is faster than assembly for single-vector operations due to inlining.
func CountTrue[T Int32x4 | Int64x2](mask T) int {
	switch m := any(mask).(type) {
	case Int32x4:
		count := 0
		for i := 0; i < 4; i++ {
			if m[i] != 0 {
				count++
			}
		}
		return count
	case Int64x2:
		count := 0
		for i := 0; i < 2; i++ {
			if m[i] != 0 {
				count++
			}
		}
		return count
	}
	return 0
}

// ===== CompressStore functions =====
// These compress elements where mask is true and store directly to a slice.

// CompressStoreF32x4 compresses float32 elements and stores to dst.
// Returns number of elements stored.
func CompressStoreF32x4(v Float32x4, mask Int32x4, dst []float32) int {
	count := 0
	for i := 0; i < 4; i++ {
		if mask[i] != 0 {
			if count < len(dst) {
				dst[count] = v[i]
			}
			count++
		}
	}
	return count
}

// CompressStoreF64x2 compresses float64 elements and stores to dst.
// Returns number of elements stored.
func CompressStoreF64x2(v Float64x2, mask Int64x2, dst []float64) int {
	count := 0
	for i := 0; i < 2; i++ {
		if mask[i] != 0 {
			if count < len(dst) {
				dst[count] = v[i]
			}
			count++
		}
	}
	return count
}

// CompressStoreI32x4 compresses int32 elements and stores to dst.
// Returns number of elements stored.
func CompressStoreI32x4(v Int32x4, mask Int32x4, dst []int32) int {
	count := 0
	for i := 0; i < 4; i++ {
		if mask[i] != 0 {
			if count < len(dst) {
				dst[count] = v[i]
			}
			count++
		}
	}
	return count
}

// CompressStoreI64x2 compresses int64 elements and stores to dst.
// Returns number of elements stored.
func CompressStoreI64x2(v Int64x2, mask Int64x2, dst []int64) int {
	count := 0
	for i := 0; i < 2; i++ {
		if mask[i] != 0 {
			if count < len(dst) {
				dst[count] = v[i]
			}
			count++
		}
	}
	return count
}

// ===== FirstN functions =====
// These create masks with the first n lanes set to true (-1).

// FirstNI32x4 returns a mask with the first n lanes set to true.
func FirstNI32x4(n int) Int32x4 {
	var mask Int32x4
	firstNI32x4Asm(n, (*[4]int32)(&mask))
	return mask
}

// FirstNI64x2 returns a mask with the first n lanes set to true.
func FirstNI64x2(n int) Int64x2 {
	var mask Int64x2
	firstNI64x2Asm(n, (*[2]int64)(&mask))
	return mask
}

// ===== Generic wrapper functions for hwygen =====
// These are used by generated code that calls asm.CompressStore, asm.FirstN, etc.

// CompressStore compresses and stores float32 elements.
func CompressStore(v Float32x4, mask Int32x4, dst []float32) int {
	return CompressStoreF32x4(v, mask, dst)
}

// CompressStoreFloat64 compresses and stores float64 elements.
func CompressStoreFloat64(v Float64x2, mask Int64x2, dst []float64) int {
	return CompressStoreF64x2(v, mask, dst)
}

// CompressStoreInt32 compresses and stores int32 elements.
func CompressStoreInt32(v Int32x4, mask Int32x4, dst []int32) int {
	return CompressStoreI32x4(v, mask, dst)
}

// CompressStoreInt64 compresses and stores int64 elements.
func CompressStoreInt64(v Int64x2, mask Int64x2, dst []int64) int {
	return CompressStoreI64x2(v, mask, dst)
}

// FirstN returns a mask with the first n lanes set to true (for float32).
func FirstN(n int) Int32x4 {
	return FirstNI32x4(n)
}

// FirstNFloat64 returns a mask with the first n lanes set to true (for float64).
func FirstNFloat64(n int) Int64x2 {
	return FirstNI64x2(n)
}

// FirstNInt32 returns a mask with the first n lanes set to true (for int32).
func FirstNInt32(n int) Int32x4 {
	return FirstNI32x4(n)
}

// FirstNInt64 returns a mask with the first n lanes set to true (for int64).
func FirstNInt64(n int) Int64x2 {
	return FirstNI64x2(n)
}

// MaskAnd performs bitwise AND on two masks (for float32).
func MaskAnd(a, b Int32x4) Int32x4 {
	return a.And(b)
}

// MaskAndFloat64 performs bitwise AND on two masks (for float64).
func MaskAndFloat64(a, b Int64x2) Int64x2 {
	return a.And(b)
}

// MaskAndNot performs a AND (NOT b) on masks (for float32).
func MaskAndNot(a, b Int32x4) Int32x4 {
	var result Int32x4
	for i := 0; i < 4; i++ {
		result[i] = a[i] &^ b[i]
	}
	return result
}

// MaskAndNotFloat64 performs a AND (NOT b) on masks (for float64).
func MaskAndNotFloat64(a, b Int64x2) Int64x2 {
	var result Int64x2
	for i := 0; i < 2; i++ {
		result[i] = a[i] &^ b[i]
	}
	return result
}

// AllTrueVal returns true if all lanes in the mask are true (for float32).
// Uses value receiver instead of slice.
func AllTrueVal(mask Int32x4) bool {
	return allTrueI32x4Asm((*[4]int32)(&mask))
}

// AllTrueValFloat64 returns true if all lanes in the mask are true (for float64).
func AllTrueValFloat64(mask Int64x2) bool {
	return allTrueI64x2Asm((*[2]int64)(&mask))
}

// AllFalseVal returns true if all lanes in the mask are false (for float32).
func AllFalseVal(mask Int32x4) bool {
	return allFalseI32x4Asm((*[4]int32)(&mask))
}

// AllFalseValFloat64 returns true if all lanes in the mask are false (for float64).
func AllFalseValFloat64(mask Int64x2) bool {
	return allFalseI64x2Asm((*[2]int64)(&mask))
}
