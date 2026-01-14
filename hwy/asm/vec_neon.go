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

// Pow computes v^exp element-wise using exp(exp * log(v)).
// Uses atanh-based log for improved accuracy (~0.1% relative error).
func (v Float32x4) Pow(exponent Float32x4) Float32x4 {
	one := BroadcastFloat32x4(1.0)
	two := BroadcastFloat32x4(2.0)
	ln2 := BroadcastFloat32x4(0.6931471805599453)
	invLn2 := BroadcastFloat32x4(1.4426950408889634)
	tiny := BroadcastFloat32x4(1e-30)

	// Clamp input to avoid log(0)
	x := v.Max(tiny)

	// Extract exponent k and mantissa m where x = m * 2^k, m in [1, 2)
	k := x.GetExponent()
	m := x.GetMantissa()

	// log(m) = 2*atanh((m-1)/(m+1)), where s = (m-1)/(m+1) is in [0, 1/3]
	s := m.Sub(one).Div(m.Add(one))
	s2 := s.Mul(s)

	// atanh series: s * (1 + s²/3 + s⁴/5 + s⁶/7 + s⁸/9 + s¹⁰/11 + s¹²/13)
	// Using Horner: poly = 1/13 + s²*(1/11 + s²*(1/9 + s²*(1/7 + s²*(1/5 + s²*(1/3 + s²)))))
	c13 := BroadcastFloat32x4(0.076923077) // 1/13
	c11 := BroadcastFloat32x4(0.090909091) // 1/11
	c9 := BroadcastFloat32x4(0.111111111)  // 1/9
	c7 := BroadcastFloat32x4(0.142857143)  // 1/7
	c5 := BroadcastFloat32x4(0.2)          // 1/5
	c3 := BroadcastFloat32x4(0.333333333)  // 1/3

	poly := c13.MulAdd(s2, c11)
	poly = poly.MulAdd(s2, c9)
	poly = poly.MulAdd(s2, c7)
	poly = poly.MulAdd(s2, c5)
	poly = poly.MulAdd(s2, c3)
	poly = poly.MulAdd(s2, one)
	logM := two.Mul(s).Mul(poly)

	// log(x) = log(m) + k * ln(2)
	kf := k.ConvertToFloat32()
	logX := kf.Mul(ln2).Add(logM)

	// z = exponent * log(x), clamp to [-88, 88] for exp range
	z := exponent.Mul(logX)
	z = z.Max(BroadcastFloat32x4(-88.0))
	z = z.Min(BroadcastFloat32x4(88.0))

	// exp(z) = 2^k * exp(r) where k = round(z/ln2), r = z - k*ln2
	expK := z.Mul(invLn2).RoundToEven()
	r := z.Sub(expK.Mul(ln2))

	// exp(r) Taylor: 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720 + r⁷/5040
	e7 := BroadcastFloat32x4(1.984126984e-4) // 1/5040
	e6 := BroadcastFloat32x4(1.388888889e-3) // 1/720
	e5 := BroadcastFloat32x4(8.333333333e-3) // 1/120
	e4 := BroadcastFloat32x4(4.166666667e-2) // 1/24
	e3 := BroadcastFloat32x4(1.666666667e-1) // 1/6
	e2 := BroadcastFloat32x4(0.5)            // 1/2

	expR := e7.MulAdd(r, e6)
	expR = expR.MulAdd(r, e5)
	expR = expR.MulAdd(r, e4)
	expR = expR.MulAdd(r, e3)
	expR = expR.MulAdd(r, e2)
	expR = expR.MulAdd(r, one)
	expR = expR.MulAdd(r, one)

	// Scale by 2^k: construct float with exponent k+127
	ki := expK.ConvertToInt32()
	scaleBits := ki.Add(Int32x4{127, 127, 127, 127}).ShiftAllLeft(23)
	scale := scaleBits.AsFloat32x4()

	return expR.Mul(scale)
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
// Pure Go for inlining - avoids slice overhead.
func (v Float32x4) Greater(other Float32x4) Int32x4 {
	var r Int32x4
	if v[0] > other[0] {
		r[0] = -1
	}
	if v[1] > other[1] {
		r[1] = -1
	}
	if v[2] > other[2] {
		r[2] = -1
	}
	if v[3] > other[3] {
		r[3] = -1
	}
	return r
}

// GreaterThan is an alias for Greater (matches archsimd naming).
func (v Float32x4) GreaterThan(other Float32x4) Int32x4 {
	return v.Greater(other)
}

// Less returns a mask where v < other.
// Pure Go for inlining - avoids slice overhead.
func (v Float32x4) Less(other Float32x4) Int32x4 {
	var r Int32x4
	if v[0] < other[0] {
		r[0] = -1
	}
	if v[1] < other[1] {
		r[1] = -1
	}
	if v[2] < other[2] {
		r[2] = -1
	}
	if v[3] < other[3] {
		r[3] = -1
	}
	return r
}

// LessThan is an alias for Less (matches archsimd naming).
func (v Float32x4) LessThan(other Float32x4) Int32x4 {
	return v.Less(other)
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
// Uses direct bit manipulation for efficiency on small vectors.
func (v Float32x4) Merge(other Float32x4, mask Int32x4) Float32x4 {
	// Direct implementation using bit select: (mask & v) | (~mask & other)
	// This avoids the overhead of calling the bulk slice-based assembly function.
	var result Float32x4
	for i := 0; i < 4; i++ {
		if mask[i] != 0 {
			result[i] = v[i]
		} else {
			result[i] = other[i]
		}
	}
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

// Pow computes v^exp element-wise using SIMD math.
func (v Float64x2) Pow(exp Float64x2) Float64x2 {
	var result Float64x2
	PowF64(v[:], exp[:], result[:])
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
// Pure Go for inlining - avoids slice overhead.
func (v Float64x2) Greater(other Float64x2) Int64x2 {
	var r Int64x2
	if v[0] > other[0] {
		r[0] = -1
	}
	if v[1] > other[1] {
		r[1] = -1
	}
	return r
}

// GreaterThan is an alias for Greater (matches archsimd naming).
func (v Float64x2) GreaterThan(other Float64x2) Int64x2 {
	return v.Greater(other)
}

// Less returns a mask where v < other.
// Pure Go for inlining - avoids slice overhead.
func (v Float64x2) Less(other Float64x2) Int64x2 {
	var r Int64x2
	if v[0] < other[0] {
		r[0] = -1
	}
	if v[1] < other[1] {
		r[1] = -1
	}
	return r
}

// LessThan is an alias for Less (matches archsimd naming).
func (v Float64x2) LessThan(other Float64x2) Int64x2 {
	return v.Less(other)
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

// Greater returns a mask where v > other.
// Pure Go for inlining - avoids slice overhead.
func (v Int32x4) Greater(other Int32x4) Int32x4 {
	var r Int32x4
	if v[0] > other[0] {
		r[0] = -1
	}
	if v[1] > other[1] {
		r[1] = -1
	}
	if v[2] > other[2] {
		r[2] = -1
	}
	if v[3] > other[3] {
		r[3] = -1
	}
	return r
}

// GreaterThan is an alias for Greater (matches archsimd naming).
func (v Int32x4) GreaterThan(other Int32x4) Int32x4 {
	return v.Greater(other)
}

// Less returns a mask where v < other.
// Pure Go for inlining - avoids slice overhead.
func (v Int32x4) Less(other Int32x4) Int32x4 {
	var r Int32x4
	if v[0] < other[0] {
		r[0] = -1
	}
	if v[1] < other[1] {
		r[1] = -1
	}
	if v[2] < other[2] {
		r[2] = -1
	}
	if v[3] < other[3] {
		r[3] = -1
	}
	return r
}

// LessThan is an alias for Less (matches archsimd naming).
func (v Int32x4) LessThan(other Int32x4) Int32x4 {
	return v.Less(other)
}

// LessEqual returns a mask where v <= other.
func (v Int32x4) LessEqual(other Int32x4) Int32x4 {
	var result Int32x4
	LeI32(v[:], other[:], result[:])
	return result
}

// GreaterEqual returns a mask where v >= other.
func (v Int32x4) GreaterEqual(other Int32x4) Int32x4 {
	var result Int32x4
	GeI32(v[:], other[:], result[:])
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

// GetBit returns true if the element at index i is non-zero (used for mask operations).
func (v Int32x4) GetBit(i int) bool {
	return v[i] != 0
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

// GetBit returns true if the element at index i is non-zero (used for mask operations).
func (v Int32x2) GetBit(i int) bool {
	return v[i] != 0
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

// Greater returns a mask where v > other.
// Pure Go for inlining - avoids slice overhead.
func (v Int64x2) Greater(other Int64x2) Int64x2 {
	var r Int64x2
	if v[0] > other[0] {
		r[0] = -1
	}
	if v[1] > other[1] {
		r[1] = -1
	}
	return r
}

// GreaterThan is an alias for Greater (matches archsimd naming).
func (v Int64x2) GreaterThan(other Int64x2) Int64x2 {
	return v.Greater(other)
}

// Less returns a mask where v < other.
// Pure Go for inlining - avoids slice overhead.
func (v Int64x2) Less(other Int64x2) Int64x2 {
	var r Int64x2
	if v[0] < other[0] {
		r[0] = -1
	}
	if v[1] < other[1] {
		r[1] = -1
	}
	return r
}

// LessThan is an alias for Less (matches archsimd naming).
func (v Int64x2) LessThan(other Int64x2) Int64x2 {
	return v.Less(other)
}

// LessEqual returns a mask where v <= other.
func (v Int64x2) LessEqual(other Int64x2) Int64x2 {
	var result Int64x2
	LeI64(v[:], other[:], result[:])
	return result
}

// GreaterEqual returns a mask where v >= other.
func (v Int64x2) GreaterEqual(other Int64x2) Int64x2 {
	var result Int64x2
	GeI64(v[:], other[:], result[:])
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

// AllTrue returns true if all lanes in the mask are true.
// Pure Go for inlining - checks all lanes directly.
func AllTrue[T Int32x4 | Int64x2](mask T) bool {
	switch m := any(mask).(type) {
	case Int32x4:
		return m[0] != 0 && m[1] != 0 && m[2] != 0 && m[3] != 0
	case Int64x2:
		return m[0] != 0 && m[1] != 0
	}
	return false
}

// AllFalse returns true if all lanes in the mask are false.
// Pure Go for inlining - checks all lanes directly.
func AllFalse[T Int32x4 | Int64x2](mask T) bool {
	switch m := any(mask).(type) {
	case Int32x4:
		return m[0] == 0 && m[1] == 0 && m[2] == 0 && m[3] == 0
	case Int64x2:
		return m[0] == 0 && m[1] == 0
	}
	return false
}

// ===== Compress lookup tables =====
// For 4-lane NEON, we use lookup tables to avoid branching.
// Each entry maps a 4-bit mask to the lane indices that should be gathered.

// compressTableF32 maps a 4-bit mask to byte offsets for gathering.
// Entry i corresponds to mask bits (lane3<<3 | lane2<<2 | lane1<<1 | lane0).
// Values are byte offsets within the 16-byte vector (lane * 4 bytes).
var compressTableF32 = [16][4]uint8{
	{0, 0, 0, 0},    // 0b0000: no elements
	{0, 0, 0, 0},    // 0b0001: lane 0
	{4, 0, 0, 0},    // 0b0010: lane 1
	{0, 4, 0, 0},    // 0b0011: lane 0,1
	{8, 0, 0, 0},    // 0b0100: lane 2
	{0, 8, 0, 0},    // 0b0101: lane 0,2
	{4, 8, 0, 0},    // 0b0110: lane 1,2
	{0, 4, 8, 0},    // 0b0111: lane 0,1,2
	{12, 0, 0, 0},   // 0b1000: lane 3
	{0, 12, 0, 0},   // 0b1001: lane 0,3
	{4, 12, 0, 0},   // 0b1010: lane 1,3
	{0, 4, 12, 0},   // 0b1011: lane 0,1,3
	{8, 12, 0, 0},   // 0b1100: lane 2,3
	{0, 8, 12, 0},   // 0b1101: lane 0,2,3
	{4, 8, 12, 0},   // 0b1110: lane 1,2,3
	{0, 4, 8, 12},   // 0b1111: all elements
}

// compressCountTable maps a 4-bit mask to the count of set bits.
var compressCountTable = [16]int{0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4}

// compressPartitionBytesF32 maps a 4-bit mask to 16-byte permutation indices for TBL.
// Each entry is 16 bytes representing how to shuffle bytes for partition-style compress.
// This is the byte-level version of compressPartitionTableF32 for NEON TBL instruction.
// Lane 0 = bytes 0-3, Lane 1 = bytes 4-7, Lane 2 = bytes 8-11, Lane 3 = bytes 12-15.
var compressPartitionBytesF32 = [16][16]byte{
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},   // 0b0000: [0,1,2,3]
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},   // 0b0001: [0,1,2,3]
	{4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15},   // 0b0010: [1,0,2,3]
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},   // 0b0011: [0,1,2,3]
	{8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15},   // 0b0100: [2,0,1,3]
	{0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15},   // 0b0101: [0,2,1,3]
	{4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 12, 13, 14, 15},   // 0b0110: [1,2,0,3]
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},   // 0b0111: [0,1,2,3]
	{12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},   // 0b1000: [3,0,1,2]
	{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11},   // 0b1001: [0,3,1,2]
	{4, 5, 6, 7, 12, 13, 14, 15, 0, 1, 2, 3, 8, 9, 10, 11},   // 0b1010: [1,3,0,2]
	{0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11},   // 0b1011: [0,1,3,2]
	{8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7},   // 0b1100: [2,3,0,1]
	{0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7},   // 0b1101: [0,2,3,1]
	{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3},   // 0b1110: [1,2,3,0]
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},   // 0b1111: [0,1,2,3]
}

// compressPartitionBytesF64 maps a 2-bit mask to 16-byte permutation indices for TBL.
// For Float64x2: Lane 0 = bytes 0-7, Lane 1 = bytes 8-15.
var compressPartitionBytesF64 = [4][16]byte{
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},   // 0b00: [0,1]
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},   // 0b01: [0,1]
	{8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7},   // 0b10: [1,0]
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},   // 0b11: [0,1]
}

// compressPartitionTableF32 maps a 4-bit mask to lane indices for PARTITION ordering.
// Unlike compress which only stores matching elements, partition reorders so that:
// - Lanes where mask=true come first (positions 0, 1, ...)
// - Lanes where mask=false come after (positions count, count+1, ...)
// This enables the Highway VQSort double-store trick.
// Values are lane indices (0-3), not byte offsets.
var compressPartitionTableF32 = [16][4]uint8{
	{0, 1, 2, 3}, // 0b0000: all false → keep original order for false elements
	{0, 1, 2, 3}, // 0b0001: lane 0 true → [0], [1,2,3]
	{1, 0, 2, 3}, // 0b0010: lane 1 true → [1], [0,2,3]
	{0, 1, 2, 3}, // 0b0011: lanes 0,1 true → [0,1], [2,3]
	{2, 0, 1, 3}, // 0b0100: lane 2 true → [2], [0,1,3]
	{0, 2, 1, 3}, // 0b0101: lanes 0,2 true → [0,2], [1,3]
	{1, 2, 0, 3}, // 0b0110: lanes 1,2 true → [1,2], [0,3]
	{0, 1, 2, 3}, // 0b0111: lanes 0,1,2 true → [0,1,2], [3]
	{3, 0, 1, 2}, // 0b1000: lane 3 true → [3], [0,1,2]
	{0, 3, 1, 2}, // 0b1001: lanes 0,3 true → [0,3], [1,2]
	{1, 3, 0, 2}, // 0b1010: lanes 1,3 true → [1,3], [0,2]
	{0, 1, 3, 2}, // 0b1011: lanes 0,1,3 true → [0,1,3], [2]
	{2, 3, 0, 1}, // 0b1100: lanes 2,3 true → [2,3], [0,1]
	{0, 2, 3, 1}, // 0b1101: lanes 0,2,3 true → [0,2,3], [1]
	{1, 2, 3, 0}, // 0b1110: lanes 1,2,3 true → [1,2,3], [0]
	{0, 1, 2, 3}, // 0b1111: all true → keep original order
}

// ===== CompressKeys (partition-style compress) =====
// These reorder vectors so matching elements come first, non-matching come after.
// This is the key primitive for Highway's VQSort double-store partition trick.

// maskToIndex4 converts a 4-lane Int32x4 mask to a 4-bit index.
// Inlined for performance.
func maskToIndex4(mask Int32x4) int {
	idx := 0
	if mask[0] != 0 {
		idx |= 1
	}
	if mask[1] != 0 {
		idx |= 2
	}
	if mask[2] != 0 {
		idx |= 4
	}
	if mask[3] != 0 {
		idx |= 8
	}
	return idx
}

// CompressKeysF32x4 reorders v so elements where mask=true come first.
// Returns (reordered vector, count of true elements).
// Unlike CompressStore which discards false elements, this keeps ALL elements
// in partition order: [true elements...][false elements...].
// Note: Scalar implementation is faster than TBL assembly due to Go inlining.
func CompressKeysF32x4(v Float32x4, mask Int32x4) (Float32x4, int) {
	idx := maskToIndex4(mask)
	count := compressCountTable[idx]
	perm := &compressPartitionTableF32[idx]

	var result Float32x4
	result[0] = v[perm[0]]
	result[1] = v[perm[1]]
	result[2] = v[perm[2]]
	result[3] = v[perm[3]]
	return result, count
}

// CompressKeysI32x4 reorders v so elements where mask=true come first.
// Returns (reordered vector, count of true elements).
// Note: Scalar implementation is faster than TBL assembly due to Go inlining.
func CompressKeysI32x4(v Int32x4, mask Int32x4) (Int32x4, int) {
	idx := maskToIndex4(mask)
	count := compressCountTable[idx]
	perm := &compressPartitionTableF32[idx]

	var result Int32x4
	result[0] = v[perm[0]]
	result[1] = v[perm[1]]
	result[2] = v[perm[2]]
	result[3] = v[perm[3]]
	return result, count
}

// ===== Fast CountTrue functions =====
// These use popcount to count set mask bits efficiently.

// CountTrueF32x4 returns the count of true lanes in an Int32x4 mask.
// Inlinable pure Go implementation.
func CountTrueF32x4(mask Int32x4) int {
	idx := 0
	if mask[0] != 0 {
		idx |= 1
	}
	if mask[1] != 0 {
		idx |= 2
	}
	if mask[2] != 0 {
		idx |= 4
	}
	if mask[3] != 0 {
		idx |= 8
	}
	return compressCountTable[idx]
}

// CountTrueF64x2 returns the count of true lanes in an Int64x2 mask.
// Inlinable pure Go implementation.
func CountTrueF64x2(mask Int64x2) int {
	count := 0
	if mask[0] != 0 {
		count++
	}
	if mask[1] != 0 {
		count++
	}
	return count
}

// ===== CompressStore functions =====
// These compress elements where mask is true and store directly to a slice.

// CompressStoreF32x4 compresses float32 elements and stores to dst.
// Returns number of elements stored.
// Uses lookup table for efficient gathering without branches.
func CompressStoreF32x4(v Float32x4, mask Int32x4, dst []float32) int {
	// Build mask index
	idx := 0
	if mask[0] != 0 {
		idx |= 1
	}
	if mask[1] != 0 {
		idx |= 2
	}
	if mask[2] != 0 {
		idx |= 4
	}
	if mask[3] != 0 {
		idx |= 8
	}

	count := compressCountTable[idx]
	perm := &compressTableF32[idx]

	// Gather elements using byte offsets
	// Each byte offset / 4 gives the lane index
	for i := 0; i < count; i++ {
		dst[i] = v[perm[i]>>2]
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

// IfThenElse selects elements based on mask: result = mask ? yes : no (for float32).
func IfThenElse(mask Int32x4, yes, no Float32x4) Float32x4 {
	return yes.Merge(no, mask)
}

// IfThenElseFloat64 selects elements based on mask: result = mask ? yes : no (for float64).
func IfThenElseFloat64(mask Int64x2, yes, no Float64x2) Float64x2 {
	return yes.Merge(no, mask)
}

// IfThenElseInt32 selects elements based on mask: result = mask ? yes : no (for int32).
func IfThenElseInt32(mask Int32x4, yes, no Int32x4) Int32x4 {
	var result Int32x4
	for i := 0; i < 4; i++ {
		if mask[i] != 0 {
			result[i] = yes[i]
		} else {
			result[i] = no[i]
		}
	}
	return result
}

// IfThenElseInt64 selects elements based on mask: result = mask ? yes : no (for int64).
func IfThenElseInt64(mask Int64x2, yes, no Int64x2) Int64x2 {
	var result Int64x2
	for i := 0; i < 2; i++ {
		if mask[i] != 0 {
			result[i] = yes[i]
		} else {
			result[i] = no[i]
		}
	}
	return result
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
// Pure Go for inlining - direct comparisons instead of assembly.
func AllTrueVal(mask Int32x4) bool {
	return mask[0] != 0 && mask[1] != 0 && mask[2] != 0 && mask[3] != 0
}

// AllTrueValFloat64 returns true if all lanes in the mask are true (for float64).
// Pure Go for inlining - direct comparisons instead of assembly.
func AllTrueValFloat64(mask Int64x2) bool {
	return mask[0] != 0 && mask[1] != 0
}

// AllFalseVal returns true if all lanes in the mask are false (for float32).
// Pure Go for inlining - direct comparisons instead of assembly.
func AllFalseVal(mask Int32x4) bool {
	return mask[0] == 0 && mask[1] == 0 && mask[2] == 0 && mask[3] == 0
}

// AllFalseValFloat64 returns true if all lanes in the mask are false (for float64).
// Pure Go for inlining - direct comparisons instead of assembly.
func AllFalseValFloat64(mask Int64x2) bool {
	return mask[0] == 0 && mask[1] == 0
}

// ===== SIMD Sorting Networks =====
// These use Min/Max operations for efficient small array sorting.

// Sort4F32 sorts 4 float32 elements in place using a sorting network.
// Uses 5 compare-swap operations with SIMD min/max.
func Sort4F32(data []float32) {
	if len(data) < 4 {
		// Insertion sort for smaller
		for i := 1; i < len(data); i++ {
			key := data[i]
			j := i - 1
			for j >= 0 && data[j] > key {
				data[j+1] = data[j]
				j--
			}
			data[j+1] = key
		}
		return
	}

	// Optimal 4-element sorting network (5 comparators)
	// Comparisons: (0,1), (2,3), (0,2), (1,3), (1,2)
	a, b, c, d := data[0], data[1], data[2], data[3]

	// Stage 1: (0,1), (2,3) - sort pairs
	if a > b {
		a, b = b, a
	}
	if c > d {
		c, d = d, c
	}

	// Stage 2: (0,2), (1,3) - compare across pairs
	if a > c {
		a, c = c, a
	}
	if b > d {
		b, d = d, b
	}

	// Stage 3: (1,2) - final comparison
	if b > c {
		b, c = c, b
	}

	data[0], data[1], data[2], data[3] = a, b, c, d
}

// Sort8F32 sorts 8 float32 elements in place using a bitonic sorting network.
// Uses SIMD min/max for parallel comparisons.
func Sort8F32(data []float32) {
	if len(data) < 8 {
		// Use Sort4F32 or insertion sort for smaller
		if len(data) <= 4 {
			Sort4F32(data)
		} else {
			Sort4F32(data[:4])
			Sort4F32(data[4:])
			// Merge the two sorted halves
			merge4x4F32(data)
		}
		return
	}

	// Load two vectors
	v0 := LoadFloat32x4Slice(data[0:])
	v1 := LoadFloat32x4Slice(data[4:])

	// Sort each half (sorting networks)
	v0 = sortVec4F32(v0)
	v1 = sortVec4F32(v1)

	// Bitonic merge: v0 ascending, v1 needs to be descending for bitonic
	// Reverse v1 to make it descending
	v1 = Float32x4{v1[3], v1[2], v1[1], v1[0]}

	// Bitonic merge across vectors
	// Stage 1: Compare v0[i] with v1[i]
	lo := v0.Min(v1)
	hi := v0.Max(v1)

	// Stage 2: Compare within each vector (distance 2)
	v0 = Float32x4{lo[0], lo[1], hi[0], hi[1]}
	v1 = Float32x4{lo[2], lo[3], hi[2], hi[3]}
	lo = v0.Min(v1)
	hi = v0.Max(v1)

	// Stage 3: Compare within each vector (distance 1)
	v0 = Float32x4{lo[0], hi[0], lo[2], hi[2]}
	v1 = Float32x4{lo[1], hi[1], lo[3], hi[3]}
	lo = v0.Min(v1)
	hi = v0.Max(v1)

	// Interleave results
	data[0] = lo[0]
	data[1] = hi[0]
	data[2] = lo[1]
	data[3] = hi[1]
	data[4] = lo[2]
	data[5] = hi[2]
	data[6] = lo[3]
	data[7] = hi[3]
}

// sortVec4F32 sorts 4 elements within a vector using a sorting network.
func sortVec4F32(v Float32x4) Float32x4 {
	a, b, c, d := v[0], v[1], v[2], v[3]

	// Optimal 4-element sorting network
	if a > b {
		a, b = b, a
	}
	if c > d {
		c, d = d, c
	}
	if a > c {
		a, c = c, a
	}
	if b > d {
		b, d = d, b
	}
	if b > c {
		b, c = c, b
	}

	return Float32x4{a, b, c, d}
}

// merge4x4F32 merges two sorted 4-element halves.
func merge4x4F32(data []float32) {
	// Simple merge for 8 elements
	var tmp [8]float32
	i, j, k := 0, 4, 0

	for i < 4 && j < 8 {
		if data[i] <= data[j] {
			tmp[k] = data[i]
			i++
		} else {
			tmp[k] = data[j]
			j++
		}
		k++
	}
	for i < 4 {
		tmp[k] = data[i]
		i++
		k++
	}
	for j < 8 {
		tmp[k] = data[j]
		j++
		k++
	}
	copy(data, tmp[:])
}

// SortSmallF32 sorts a small float32 slice using optimized sorting networks.
// For n <= 4: uses 4-element network
// For n <= 8: uses 8-element bitonic network
// For n <= 16: uses 16-element bitonic network
// For larger: uses insertion sort
func SortSmallF32(data []float32) {
	n := len(data)
	switch {
	case n <= 1:
		return
	case n <= 4:
		Sort4F32(data)
	case n <= 8:
		Sort8F32(data)
	default:
		// Use insertion sort for larger small arrays
		for i := 1; i < n; i++ {
			key := data[i]
			j := i - 1
			for j >= 0 && data[j] > key {
				data[j+1] = data[j]
				j--
			}
			data[j+1] = key
		}
	}
}
