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

//go:build !noasm && arm64

package asm

import (
	"math"
	"unsafe"
)

// Float32x4 represents a 128-bit NEON vector of 4 float32 values.
// Uses [16]byte backing for efficient register passing via GoAT-generated assembly.
type Float32x4 [16]byte

// Float64x2 represents a 128-bit NEON vector of 2 float64 values.
// Uses [16]byte backing for efficient register passing via GoAT-generated assembly.
type Float64x2 [16]byte

// Int32x4 represents a 128-bit NEON vector of 4 int32 values.
// Uses [16]byte backing for efficient register passing via GoAT-generated assembly.
type Int32x4 [16]byte

// Int64x2 represents a 128-bit NEON vector of 2 int64 values.
// Uses [16]byte backing for efficient register passing via GoAT-generated assembly.
type Int64x2 [16]byte

// Float32x2 represents a 64-bit NEON vector of 2 float32 values.
// Uses [8]byte backing for efficient register passing via GoAT-generated assembly.
type Float32x2 [8]byte

// Int32x2 represents a 64-bit vector of 2 int32 values.
// Used for type conversions from Float64x2.
type Int32x2 [8]byte

// ===== Float32x4 constructors =====

// BroadcastFloat32x4 creates a vector with all lanes set to the given value.
func BroadcastFloat32x4(v float32) Float32x4 {
	arr := [4]float32{v, v, v, v}
	return *(*Float32x4)(unsafe.Pointer(&arr))
}

// LoadFloat32x4 loads 4 float32 values from a slice.
func LoadFloat32x4(s []float32) Float32x4 {
	return *(*Float32x4)(unsafe.Pointer(&s[0]))
}

// LoadFloat32x4Slice is an alias for LoadFloat32x4 (matches archsimd naming).
func LoadFloat32x4Slice(s []float32) Float32x4 {
	return LoadFloat32x4(s)
}

// Load4Float32x4Slice loads 4 consecutive Float32x4 vectors (16 floats = 64 bytes)
// using a single ARM ld1 instruction with 4 registers. This is more efficient
// than 4 separate loads for 4x loop unrolling.
func Load4Float32x4Slice(s []float32) (Float32x4, Float32x4, Float32x4, Float32x4) {
	var v0, v1, v2, v3 Float32x4
	load4_f32x4(unsafe.Pointer(&s[0]), unsafe.Pointer(&v0), unsafe.Pointer(&v1), unsafe.Pointer(&v2), unsafe.Pointer(&v3))
	return v0, v1, v2, v3
}

// ZeroFloat32x4 returns a zero vector.
func ZeroFloat32x4() Float32x4 {
	return Float32x4{}
}

// SignBitFloat32x4 returns a vector with sign bit set in all lanes.
func SignBitFloat32x4() Float32x4 {
	signBit := math.Float32frombits(0x80000000)
	return BroadcastFloat32x4(signBit)
}

// ===== Float32x4 accessors =====

// asF32 returns the vector as a pointer to [4]float32 for element access.
func (v *Float32x4) asF32() *[4]float32 {
	return (*[4]float32)(unsafe.Pointer(v))
}

// Get returns the element at the given index.
func (v Float32x4) Get(i int) float32 {
	return (*[4]float32)(unsafe.Pointer(&v))[i]
}

// Set sets the element at the given index.
func (v *Float32x4) Set(i int, val float32) {
	v.asF32()[i] = val
}

// ===== Float32x4 methods =====

// StoreSlice stores the vector to a slice.
func (v Float32x4) StoreSlice(s []float32) {
	*(*Float32x4)(unsafe.Pointer(&s[0])) = v
}

// Add performs element-wise addition.
func (v Float32x4) Add(other Float32x4) Float32x4 {
	return Float32x4(add_f32x4([16]byte(v), [16]byte(other)))
}

// Sub performs element-wise subtraction.
func (v Float32x4) Sub(other Float32x4) Float32x4 {
	return Float32x4(sub_f32x4([16]byte(v), [16]byte(other)))
}

// Mul performs element-wise multiplication.
func (v Float32x4) Mul(other Float32x4) Float32x4 {
	return Float32x4(mul_f32x4([16]byte(v), [16]byte(other)))
}

// Div performs element-wise division.
func (v Float32x4) Div(other Float32x4) Float32x4 {
	return Float32x4(div_f32x4([16]byte(v), [16]byte(other)))
}

// Min performs element-wise minimum.
func (v Float32x4) Min(other Float32x4) Float32x4 {
	return Float32x4(min_f32x4([16]byte(v), [16]byte(other)))
}

// Max performs element-wise maximum.
func (v Float32x4) Max(other Float32x4) Float32x4 {
	return Float32x4(max_f32x4([16]byte(v), [16]byte(other)))
}

// Sqrt performs element-wise square root.
func (v Float32x4) Sqrt() Float32x4 {
	return Float32x4(sqrt_f32x4([16]byte(v)))
}

// Abs performs element-wise absolute value.
func (v Float32x4) Abs() Float32x4 {
	return Float32x4(abs_f32x4([16]byte(v)))
}

// Neg performs element-wise negation.
func (v Float32x4) Neg() Float32x4 {
	return Float32x4(neg_f32x4([16]byte(v)))
}

// MulAdd performs fused multiply-add: v * a + b
func (v Float32x4) MulAdd(a, b Float32x4) Float32x4 {
	return Float32x4(fma_f32x4([16]byte(v), [16]byte(a), [16]byte(b)))
}

// MulSub performs fused multiply-subtract: v * a - b
func (v Float32x4) MulSub(a, b Float32x4) Float32x4 {
	return Float32x4(fms_f32x4([16]byte(v), [16]byte(a), [16]byte(b)))
}

// Recip returns the reciprocal estimate (1/x).
func (v Float32x4) Recip() Float32x4 {
	return Float32x4(recip_f32x4([16]byte(v)))
}

// RSqrt returns the reciprocal square root estimate (1/sqrt(x)).
func (v Float32x4) RSqrt() Float32x4 {
	return Float32x4(rsqrt_f32x4([16]byte(v)))
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
	scaleBits := ki.Add(BroadcastInt32x4(127)).ShiftAllLeft(23)
	scale := scaleBits.AsFloat32x4()

	return expR.Mul(scale)
}

// ReduceSum returns the sum of all elements.
func (v Float32x4) ReduceSum() float32 {
	return hsum_f32x4([16]byte(v))
}

// ReduceMin returns the minimum element.
func (v Float32x4) ReduceMin() float32 {
	return hmin_f32x4([16]byte(v))
}

// ReduceMax returns the maximum element.
func (v Float32x4) ReduceMax() float32 {
	return hmax_f32x4([16]byte(v))
}

// Dot returns the dot product of two vectors.
func (v Float32x4) Dot(other Float32x4) float32 {
	return dot_f32x4([16]byte(v), [16]byte(other))
}

// Greater returns a mask where v > other.
func (v Float32x4) Greater(other Float32x4) Int32x4 {
	return Int32x4(gt_f32x4([16]byte(v), [16]byte(other)))
}

// GreaterThan is an alias for Greater (matches archsimd naming).
func (v Float32x4) GreaterThan(other Float32x4) Int32x4 {
	return v.Greater(other)
}

// Less returns a mask where v < other.
func (v Float32x4) Less(other Float32x4) Int32x4 {
	return Int32x4(lt_f32x4([16]byte(v), [16]byte(other)))
}

// LessThan is an alias for Less (matches archsimd naming).
func (v Float32x4) LessThan(other Float32x4) Int32x4 {
	return v.Less(other)
}

// GreaterEqual returns a mask where v >= other.
func (v Float32x4) GreaterEqual(other Float32x4) Int32x4 {
	return Int32x4(ge_f32x4([16]byte(v), [16]byte(other)))
}

// LessEqual returns a mask where v <= other.
func (v Float32x4) LessEqual(other Float32x4) Int32x4 {
	return Int32x4(le_f32x4([16]byte(v), [16]byte(other)))
}

// Equal returns a mask where v == other.
func (v Float32x4) Equal(other Float32x4) Int32x4 {
	return Int32x4(eq_f32x4([16]byte(v), [16]byte(other)))
}

// NotEqual returns a mask where v != other.
func (v Float32x4) NotEqual(other Float32x4) Int32x4 {
	return Int32x4(ne_f32x4([16]byte(v), [16]byte(other)))
}

// And performs bitwise AND.
func (v Float32x4) And(other Float32x4) Float32x4 {
	return Float32x4(and_f32x4([16]byte(v), [16]byte(other)))
}

// Or performs bitwise OR.
func (v Float32x4) Or(other Float32x4) Float32x4 {
	return Float32x4(or_f32x4([16]byte(v), [16]byte(other)))
}

// Xor performs bitwise XOR.
func (v Float32x4) Xor(other Float32x4) Float32x4 {
	return Float32x4(xor_f32x4([16]byte(v), [16]byte(other)))
}

// Not performs bitwise NOT (reuses Int32x4 assembly).
func (v Float32x4) Not() Float32x4 {
	return Float32x4(not_i32x4([16]byte(v)))
}

// Merge selects elements: mask ? v : other
// Note: mask values should be all-ones (-1) for true, all-zeros (0) for false.
func (v Float32x4) Merge(other Float32x4, mask Int32x4) Float32x4 {
	return Float32x4(sel_f32x4([16]byte(mask), [16]byte(v), [16]byte(other)))
}

// RoundToEven rounds to nearest even.
func (v Float32x4) RoundToEven() Float32x4 {
	return Float32x4(round_f32x4([16]byte(v)))
}

// Floor rounds toward negative infinity.
func (v Float32x4) Floor() Float32x4 {
	return Float32x4(floor_f32x4([16]byte(v)))
}

// Ceil rounds toward positive infinity.
func (v Float32x4) Ceil() Float32x4 {
	return Float32x4(ceil_f32x4([16]byte(v)))
}

// Trunc rounds toward zero.
func (v Float32x4) Trunc() Float32x4 {
	return Float32x4(trunc_f32x4([16]byte(v)))
}

// ConvertToInt32 converts to int32.
func (v Float32x4) ConvertToInt32() Int32x4 {
	return Int32x4(cvt_f32x4_i32x4([16]byte(v)))
}

// AsInt32x4 reinterprets bits as int32.
func (v Float32x4) AsInt32x4() Int32x4 {
	return Int32x4(v)
}

// GetExponent extracts the unbiased exponent from IEEE 754 floats.
func (v Float32x4) GetExponent() Int32x4 {
	f := (*[4]float32)(unsafe.Pointer(&v))
	var result [4]int32
	for i := range 4 {
		bits := math.Float32bits(f[i])
		exp := int32((bits >> 23) & 0xFF)
		if exp == 0 || exp == 255 {
			result[i] = 0 // denormal or special
		} else {
			result[i] = exp - 127
		}
	}
	return *(*Int32x4)(unsafe.Pointer(&result))
}

// GetMantissa extracts the mantissa with exponent normalized to [1, 2).
func (v Float32x4) GetMantissa() Float32x4 {
	f := (*[4]float32)(unsafe.Pointer(&v))
	var result [4]float32
	for i := range 4 {
		bits := math.Float32bits(f[i])
		// Clear exponent, set to 127 (2^0 = 1)
		mantissa := (bits & 0x807FFFFF) | 0x3F800000
		result[i] = math.Float32frombits(mantissa)
	}
	return *(*Float32x4)(unsafe.Pointer(&result))
}

// Data returns the underlying data as a slice.
func (v Float32x4) Data() []float32 {
	return (*[4]float32)(unsafe.Pointer(&v))[:]
}

// ===== Float64x2 constructors =====

// BroadcastFloat64x2 creates a vector with all lanes set to the given value.
func BroadcastFloat64x2(v float64) Float64x2 {
	arr := [2]float64{v, v}
	return *(*Float64x2)(unsafe.Pointer(&arr))
}

// LoadFloat64x2 loads 2 float64 values from a slice.
func LoadFloat64x2(s []float64) Float64x2 {
	return *(*Float64x2)(unsafe.Pointer(&s[0]))
}

// LoadFloat64x2Slice is an alias for LoadFloat64x2 (matches archsimd naming).
func LoadFloat64x2Slice(s []float64) Float64x2 {
	return LoadFloat64x2(s)
}

// Load4Float64x2Slice loads 4 consecutive Float64x2 vectors (8 doubles = 64 bytes)
// using a single ARM ld1 instruction with 4 registers. This is more efficient
// than 4 separate loads for 4x loop unrolling.
func Load4Float64x2Slice(s []float64) (Float64x2, Float64x2, Float64x2, Float64x2) {
	var v0, v1, v2, v3 Float64x2
	load4_f64x2(unsafe.Pointer(&s[0]), unsafe.Pointer(&v0), unsafe.Pointer(&v1), unsafe.Pointer(&v2), unsafe.Pointer(&v3))
	return v0, v1, v2, v3
}

// ZeroFloat64x2 returns a zero vector.
func ZeroFloat64x2() Float64x2 {
	return Float64x2{}
}

// SignBitFloat64x2 returns a vector with sign bit set in all lanes.
func SignBitFloat64x2() Float64x2 {
	signBit := math.Float64frombits(0x8000000000000000)
	return BroadcastFloat64x2(signBit)
}

// ===== Float64x2 accessors =====

// asF64 returns the vector as a pointer to [2]float64 for element access.
func (v *Float64x2) asF64() *[2]float64 {
	return (*[2]float64)(unsafe.Pointer(v))
}

// Get returns the element at the given index.
func (v Float64x2) Get(i int) float64 {
	return (*[2]float64)(unsafe.Pointer(&v))[i]
}

// Set sets the element at the given index.
func (v *Float64x2) Set(i int, val float64) {
	v.asF64()[i] = val
}

// ===== Float64x2 methods =====

// StoreSlice stores the vector to a slice.
func (v Float64x2) StoreSlice(s []float64) {
	*(*Float64x2)(unsafe.Pointer(&s[0])) = v
}

// Add performs element-wise addition.
func (v Float64x2) Add(other Float64x2) Float64x2 {
	return Float64x2(add_f64x2([16]byte(v), [16]byte(other)))
}

// Sub performs element-wise subtraction.
func (v Float64x2) Sub(other Float64x2) Float64x2 {
	return Float64x2(sub_f64x2([16]byte(v), [16]byte(other)))
}

// Mul performs element-wise multiplication.
func (v Float64x2) Mul(other Float64x2) Float64x2 {
	return Float64x2(mul_f64x2([16]byte(v), [16]byte(other)))
}

// Div performs element-wise division.
func (v Float64x2) Div(other Float64x2) Float64x2 {
	return Float64x2(div_f64x2([16]byte(v), [16]byte(other)))
}

// Min performs element-wise minimum.
func (v Float64x2) Min(other Float64x2) Float64x2 {
	return Float64x2(min_f64x2([16]byte(v), [16]byte(other)))
}

// Max performs element-wise maximum.
func (v Float64x2) Max(other Float64x2) Float64x2 {
	return Float64x2(max_f64x2([16]byte(v), [16]byte(other)))
}

// Sqrt performs element-wise square root.
func (v Float64x2) Sqrt() Float64x2 {
	return Float64x2(sqrt_f64x2([16]byte(v)))
}

// Abs performs element-wise absolute value.
func (v Float64x2) Abs() Float64x2 {
	return Float64x2(abs_f64x2([16]byte(v)))
}

// Neg performs element-wise negation.
func (v Float64x2) Neg() Float64x2 {
	return Float64x2(neg_f64x2([16]byte(v)))
}

// MulAdd performs fused multiply-add: v * a + b
func (v Float64x2) MulAdd(a, b Float64x2) Float64x2 {
	return Float64x2(fma_f64x2([16]byte(v), [16]byte(a), [16]byte(b)))
}

// Pow computes v^exp element-wise using SIMD math.
func (v Float64x2) Pow(exp Float64x2) Float64x2 {
	var result Float64x2
	PowF64(v.Data(), exp.Data(), result.Data())
	return result
}

// ReduceSum returns the sum of all elements.
func (v Float64x2) ReduceSum() float64 {
	return hsum_f64x2([16]byte(v))
}

// ReduceMin returns the minimum element.
func (v Float64x2) ReduceMin() float64 {
	f := (*[2]float64)(unsafe.Pointer(&v))
	if f[0] < f[1] {
		return f[0]
	}
	return f[1]
}

// ReduceMax returns the maximum element.
func (v Float64x2) ReduceMax() float64 {
	f := (*[2]float64)(unsafe.Pointer(&v))
	if f[0] > f[1] {
		return f[0]
	}
	return f[1]
}

// Dot returns the dot product of two vectors.
func (v Float64x2) Dot(other Float64x2) float64 {
	return dot_f64x2([16]byte(v), [16]byte(other))
}

// Greater returns a mask where v > other.
func (v Float64x2) Greater(other Float64x2) Int64x2 {
	f1 := (*[2]float64)(unsafe.Pointer(&v))
	f2 := (*[2]float64)(unsafe.Pointer(&other))
	var r [2]int64
	if f1[0] > f2[0] {
		r[0] = -1
	}
	if f1[1] > f2[1] {
		r[1] = -1
	}
	return *(*Int64x2)(unsafe.Pointer(&r))
}

// GreaterThan is an alias for Greater (matches archsimd naming).
func (v Float64x2) GreaterThan(other Float64x2) Int64x2 {
	return v.Greater(other)
}

// Less returns a mask where v < other.
func (v Float64x2) Less(other Float64x2) Int64x2 {
	f1 := (*[2]float64)(unsafe.Pointer(&v))
	f2 := (*[2]float64)(unsafe.Pointer(&other))
	var r [2]int64
	if f1[0] < f2[0] {
		r[0] = -1
	}
	if f1[1] < f2[1] {
		r[1] = -1
	}
	return *(*Int64x2)(unsafe.Pointer(&r))
}

// LessThan is an alias for Less (matches archsimd naming).
func (v Float64x2) LessThan(other Float64x2) Int64x2 {
	return v.Less(other)
}

// LessEqual returns a mask where v <= other.
func (v Float64x2) LessEqual(other Float64x2) Int64x2 {
	f1 := (*[2]float64)(unsafe.Pointer(&v))
	f2 := (*[2]float64)(unsafe.Pointer(&other))
	var r [2]int64
	if f1[0] <= f2[0] {
		r[0] = -1
	}
	if f1[1] <= f2[1] {
		r[1] = -1
	}
	return *(*Int64x2)(unsafe.Pointer(&r))
}

// GreaterEqual returns a mask where v >= other.
func (v Float64x2) GreaterEqual(other Float64x2) Int64x2 {
	f1 := (*[2]float64)(unsafe.Pointer(&v))
	f2 := (*[2]float64)(unsafe.Pointer(&other))
	var r [2]int64
	if f1[0] >= f2[0] {
		r[0] = -1
	}
	if f1[1] >= f2[1] {
		r[1] = -1
	}
	return *(*Int64x2)(unsafe.Pointer(&r))
}

// Equal returns a mask where v == other.
func (v Float64x2) Equal(other Float64x2) Int64x2 {
	f1 := (*[2]float64)(unsafe.Pointer(&v))
	f2 := (*[2]float64)(unsafe.Pointer(&other))
	var r [2]int64
	if f1[0] == f2[0] {
		r[0] = -1
	}
	if f1[1] == f2[1] {
		r[1] = -1
	}
	return *(*Int64x2)(unsafe.Pointer(&r))
}

// NotEqual returns a mask where v != other.
func (v Float64x2) NotEqual(other Float64x2) Int64x2 {
	f1 := (*[2]float64)(unsafe.Pointer(&v))
	f2 := (*[2]float64)(unsafe.Pointer(&other))
	var r [2]int64
	if f1[0] != f2[0] {
		r[0] = -1
	}
	if f1[1] != f2[1] {
		r[1] = -1
	}
	return *(*Int64x2)(unsafe.Pointer(&r))
}

// And performs bitwise AND (reuses Int64x2 assembly).
func (v Float64x2) And(other Float64x2) Float64x2 {
	return Float64x2(and_i64x2([16]byte(v), [16]byte(other)))
}

// Or performs bitwise OR.
func (v Float64x2) Or(other Float64x2) Float64x2 {
	return Float64x2(or_i64x2([16]byte(v), [16]byte(other)))
}

// Xor performs bitwise XOR.
func (v Float64x2) Xor(other Float64x2) Float64x2 {
	return Float64x2(xor_i64x2([16]byte(v), [16]byte(other)))
}

// Not performs bitwise NOT (XOR with all ones).
func (v Float64x2) Not() Float64x2 {
	allOnes := [16]byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}
	return Float64x2(xor_i64x2([16]byte(v), allOnes))
}

// Merge selects elements: mask ? v : other
func (v Float64x2) Merge(other Float64x2, mask Int64x2) Float64x2 {
	f1 := (*[2]float64)(unsafe.Pointer(&v))
	f2 := (*[2]float64)(unsafe.Pointer(&other))
	m := (*[2]int64)(unsafe.Pointer(&mask))
	var result [2]float64
	for i := range 2 {
		if m[i] != 0 {
			result[i] = f1[i]
		} else {
			result[i] = f2[i]
		}
	}
	return *(*Float64x2)(unsafe.Pointer(&result))
}

// RoundToEven rounds to nearest even.
func (v Float64x2) RoundToEven() Float64x2 {
	f := (*[2]float64)(unsafe.Pointer(&v))
	var result [2]float64
	for i := range 2 {
		result[i] = math.RoundToEven(f[i])
	}
	return *(*Float64x2)(unsafe.Pointer(&result))
}

// ConvertToInt32 converts float64 to int32 (truncate toward zero).
func (v Float64x2) ConvertToInt32() Int32x2 {
	f := (*[2]float64)(unsafe.Pointer(&v))
	result := [2]int32{int32(f[0]), int32(f[1])}
	return *(*Int32x2)(unsafe.Pointer(&result))
}

// ConvertToInt64 converts float64 to int64 (truncate toward zero).
func (v Float64x2) ConvertToInt64() Int64x2 {
	f := (*[2]float64)(unsafe.Pointer(&v))
	result := [2]int64{int64(f[0]), int64(f[1])}
	return *(*Int64x2)(unsafe.Pointer(&result))
}

// AsInt64x2 reinterprets bits as int64.
func (v Float64x2) AsInt64x2() Int64x2 {
	return Int64x2(v)
}

// GetExponent extracts the unbiased exponent from IEEE 754 floats.
func (v Float64x2) GetExponent() Int32x2 {
	f := (*[2]float64)(unsafe.Pointer(&v))
	var result [2]int32
	for i := range 2 {
		bits := math.Float64bits(f[i])
		exp := int32((bits >> 52) & 0x7FF)
		if exp == 0 || exp == 2047 {
			result[i] = 0 // denormal or special
		} else {
			result[i] = exp - 1023
		}
	}
	return *(*Int32x2)(unsafe.Pointer(&result))
}

// GetMantissa extracts the mantissa with exponent normalized to [1, 2).
func (v Float64x2) GetMantissa() Float64x2 {
	f := (*[2]float64)(unsafe.Pointer(&v))
	var result [2]float64
	for i := range 2 {
		bits := math.Float64bits(f[i])
		// Clear exponent, set to 1023 (2^0 = 1)
		mantissa := (bits & 0x800FFFFFFFFFFFFF) | 0x3FF0000000000000
		result[i] = math.Float64frombits(mantissa)
	}
	return *(*Float64x2)(unsafe.Pointer(&result))
}

// Data returns the underlying data as a slice.
func (v Float64x2) Data() []float64 {
	return (*[2]float64)(unsafe.Pointer(&v))[:]
}

// ===== Float32x2 constructors and methods =====

// BroadcastFloat32x2 creates a vector with all lanes set to the given value.
func BroadcastFloat32x2(v float32) Float32x2 {
	arr := [2]float32{v, v}
	return *(*Float32x2)(unsafe.Pointer(&arr))
}

// LoadFloat32x2 loads 2 float32 values from a slice.
func LoadFloat32x2(s []float32) Float32x2 {
	return *(*Float32x2)(unsafe.Pointer(&s[0]))
}

// ZeroFloat32x2 returns a zero vector.
func ZeroFloat32x2() Float32x2 {
	return Float32x2{}
}

// Get returns the element at the given index.
func (v Float32x2) Get(i int) float32 {
	return (*[2]float32)(unsafe.Pointer(&v))[i]
}

// StoreSlice stores the vector to a slice.
func (v Float32x2) StoreSlice(s []float32) {
	*(*Float32x2)(unsafe.Pointer(&s[0])) = v
}

// Add performs element-wise addition.
func (v Float32x2) Add(other Float32x2) Float32x2 {
	return Float32x2(add_f32x2([8]byte(v), [8]byte(other)))
}

// Sub performs element-wise subtraction.
func (v Float32x2) Sub(other Float32x2) Float32x2 {
	return Float32x2(sub_f32x2([8]byte(v), [8]byte(other)))
}

// Mul performs element-wise multiplication.
func (v Float32x2) Mul(other Float32x2) Float32x2 {
	return Float32x2(mul_f32x2([8]byte(v), [8]byte(other)))
}

// Div performs element-wise division.
func (v Float32x2) Div(other Float32x2) Float32x2 {
	return Float32x2(div_f32x2([8]byte(v), [8]byte(other)))
}

// Min performs element-wise minimum.
func (v Float32x2) Min(other Float32x2) Float32x2 {
	return Float32x2(min_f32x2([8]byte(v), [8]byte(other)))
}

// Max performs element-wise maximum.
func (v Float32x2) Max(other Float32x2) Float32x2 {
	return Float32x2(max_f32x2([8]byte(v), [8]byte(other)))
}

// ReduceSum returns the sum of all elements.
func (v Float32x2) ReduceSum() float32 {
	return hsum_f32x2([8]byte(v))
}

// Dot returns the dot product of two vectors.
func (v Float32x2) Dot(other Float32x2) float32 {
	return dot_f32x2([8]byte(v), [8]byte(other))
}

// ConvertToFloat64x2 converts float32x2 to float64x2.
func (v Float32x2) ConvertToFloat64x2() Float64x2 {
	return Float64x2(cvt_f32x2_f64x2([8]byte(v)))
}

// Data returns the underlying data as a slice.
func (v Float32x2) Data() []float32 {
	return (*[2]float32)(unsafe.Pointer(&v))[:]
}

// ===== Int32x2 constructors and methods =====

// BroadcastInt32x2 creates a vector with all lanes set to the given value.
func BroadcastInt32x2(v int32) Int32x2 {
	arr := [2]int32{v, v}
	return *(*Int32x2)(unsafe.Pointer(&arr))
}

// Get returns the element at the given index.
func (v Int32x2) Get(i int) int32 {
	return (*[2]int32)(unsafe.Pointer(&v))[i]
}

// Add performs element-wise addition.
func (v Int32x2) Add(other Int32x2) Int32x2 {
	a := (*[2]int32)(unsafe.Pointer(&v))
	b := (*[2]int32)(unsafe.Pointer(&other))
	result := [2]int32{a[0] + b[0], a[1] + b[1]}
	return *(*Int32x2)(unsafe.Pointer(&result))
}

// Sub performs element-wise subtraction.
func (v Int32x2) Sub(other Int32x2) Int32x2 {
	a := (*[2]int32)(unsafe.Pointer(&v))
	b := (*[2]int32)(unsafe.Pointer(&other))
	result := [2]int32{a[0] - b[0], a[1] - b[1]}
	return *(*Int32x2)(unsafe.Pointer(&result))
}

// Mul performs element-wise multiplication.
func (v Int32x2) Mul(other Int32x2) Int32x2 {
	a := (*[2]int32)(unsafe.Pointer(&v))
	b := (*[2]int32)(unsafe.Pointer(&other))
	result := [2]int32{a[0] * b[0], a[1] * b[1]}
	return *(*Int32x2)(unsafe.Pointer(&result))
}

// ShiftAllLeft shifts all elements left by the given count.
func (v Int32x2) ShiftAllLeft(count int) Int32x2 {
	a := (*[2]int32)(unsafe.Pointer(&v))
	result := [2]int32{a[0] << count, a[1] << count}
	return *(*Int32x2)(unsafe.Pointer(&result))
}

// Pow2Float64 computes 2^k for each lane and returns Float64x2.
// Uses IEEE 754 bit manipulation: 2^k = ((k + 1023) << 52) as float64 bits.
func (v Int32x2) Pow2Float64() Float64x2 {
	a := (*[2]int32)(unsafe.Pointer(&v))
	var result [2]float64
	Pow2F64(a[:], result[:])
	return *(*Float64x2)(unsafe.Pointer(&result))
}

// And performs element-wise bitwise AND.
func (v Int32x2) And(other Int32x2) Int32x2 {
	a := (*[2]int32)(unsafe.Pointer(&v))
	b := (*[2]int32)(unsafe.Pointer(&other))
	result := [2]int32{a[0] & b[0], a[1] & b[1]}
	return *(*Int32x2)(unsafe.Pointer(&result))
}

// Or performs element-wise bitwise OR.
func (v Int32x2) Or(other Int32x2) Int32x2 {
	a := (*[2]int32)(unsafe.Pointer(&v))
	b := (*[2]int32)(unsafe.Pointer(&other))
	result := [2]int32{a[0] | b[0], a[1] | b[1]}
	return *(*Int32x2)(unsafe.Pointer(&result))
}

// Xor performs element-wise bitwise XOR.
func (v Int32x2) Xor(other Int32x2) Int32x2 {
	a := (*[2]int32)(unsafe.Pointer(&v))
	b := (*[2]int32)(unsafe.Pointer(&other))
	result := [2]int32{a[0] ^ b[0], a[1] ^ b[1]}
	return *(*Int32x2)(unsafe.Pointer(&result))
}

// Equal performs element-wise equality comparison.
// Returns -1 (all bits set) for true, 0 for false - matches archsimd convention.
func (v Int32x2) Equal(other Int32x2) Int32x2 {
	a := (*[2]int32)(unsafe.Pointer(&v))
	b := (*[2]int32)(unsafe.Pointer(&other))
	var result [2]int32
	for i := range 2 {
		if a[i] == b[i] {
			result[i] = -1 // all bits set
		} else {
			result[i] = 0
		}
	}
	return *(*Int32x2)(unsafe.Pointer(&result))
}

// Merge selects elements: mask ? v : other
// mask should have -1 (all bits set) for true, 0 for false.
func (v Int32x2) Merge(other Int32x2, mask Int32x2) Int32x2 {
	a := (*[2]int32)(unsafe.Pointer(&v))
	b := (*[2]int32)(unsafe.Pointer(&other))
	m := (*[2]int32)(unsafe.Pointer(&mask))
	var result [2]int32
	for i := range 2 {
		if m[i] != 0 {
			result[i] = a[i]
		} else {
			result[i] = b[i]
		}
	}
	return *(*Int32x2)(unsafe.Pointer(&result))
}

// StoreSlice stores the vector to a slice.
func (v Int32x2) StoreSlice(s []int32) {
	a := (*[2]int32)(unsafe.Pointer(&v))
	s[0] = a[0]
	s[1] = a[1]
}

// Data returns the underlying array as a slice.
func (v Int32x2) Data() []int32 {
	return (*[2]int32)(unsafe.Pointer(&v))[:]
}

// GetBit returns true if the element at index i is non-zero (used for mask operations).
func (v Int32x2) GetBit(i int) bool {
	return (*[2]int32)(unsafe.Pointer(&v))[i] != 0
}

// ===== Int32x4 constructors and methods =====

// BroadcastInt32x4 creates a vector with all lanes set to the given value.
func BroadcastInt32x4(v int32) Int32x4 {
	arr := [4]int32{v, v, v, v}
	return *(*Int32x4)(unsafe.Pointer(&arr))
}

// LoadInt32x4 loads 4 int32 values from a slice.
func LoadInt32x4(s []int32) Int32x4 {
	return *(*Int32x4)(unsafe.Pointer(&s[0]))
}

// LoadInt32x4Slice is an alias for LoadInt32x4 (matches archsimd naming).
func LoadInt32x4Slice(s []int32) Int32x4 {
	return LoadInt32x4(s)
}

// Load4Int32x4Slice loads 4 consecutive Int32x4 vectors (16 int32s = 64 bytes)
// using a single ARM ld1 instruction with 4 registers.
func Load4Int32x4Slice(s []int32) (Int32x4, Int32x4, Int32x4, Int32x4) {
	var v0, v1, v2, v3 Int32x4
	load4_i32x4(unsafe.Pointer(&s[0]), unsafe.Pointer(&v0), unsafe.Pointer(&v1), unsafe.Pointer(&v2), unsafe.Pointer(&v3))
	return v0, v1, v2, v3
}

// Get returns the element at the given index.
func (v Int32x4) Get(i int) int32 {
	return (*[4]int32)(unsafe.Pointer(&v))[i]
}

// Set sets the element at the given index.
func (v *Int32x4) Set(i int, val int32) {
	(*[4]int32)(unsafe.Pointer(v))[i] = val
}

// Add performs element-wise addition.
func (v Int32x4) Add(other Int32x4) Int32x4 {
	return Int32x4(add_i32x4([16]byte(v), [16]byte(other)))
}

// Sub performs element-wise subtraction.
func (v Int32x4) Sub(other Int32x4) Int32x4 {
	return Int32x4(sub_i32x4([16]byte(v), [16]byte(other)))
}

// Mul performs element-wise multiplication.
func (v Int32x4) Mul(other Int32x4) Int32x4 {
	return Int32x4(mul_i32x4([16]byte(v), [16]byte(other)))
}

// Min performs element-wise minimum.
func (v Int32x4) Min(other Int32x4) Int32x4 {
	return Int32x4(min_i32x4([16]byte(v), [16]byte(other)))
}

// Max performs element-wise maximum.
func (v Int32x4) Max(other Int32x4) Int32x4 {
	return Int32x4(max_i32x4([16]byte(v), [16]byte(other)))
}

// Abs performs element-wise absolute value.
func (v Int32x4) Abs() Int32x4 {
	return Int32x4(abs_i32x4([16]byte(v)))
}

// Neg performs element-wise negation.
func (v Int32x4) Neg() Int32x4 {
	return Int32x4(neg_i32x4([16]byte(v)))
}

// ShiftAllLeft shifts all elements left by the given count.
func (v Int32x4) ShiftAllLeft(count int) Int32x4 {
	a := (*[4]int32)(unsafe.Pointer(&v))
	result := [4]int32{a[0] << count, a[1] << count, a[2] << count, a[3] << count}
	return *(*Int32x4)(unsafe.Pointer(&result))
}

// ShiftAllRight shifts all elements right by the given count.
func (v Int32x4) ShiftAllRight(count int) Int32x4 {
	a := (*[4]int32)(unsafe.Pointer(&v))
	result := [4]int32{a[0] >> count, a[1] >> count, a[2] >> count, a[3] >> count}
	return *(*Int32x4)(unsafe.Pointer(&result))
}

// AsFloat32x4 reinterprets bits as float32.
func (v Int32x4) AsFloat32x4() Float32x4 {
	return Float32x4(v)
}

// ConvertToFloat32 converts int32 to float32.
func (v Int32x4) ConvertToFloat32() Float32x4 {
	return Float32x4(cvt_i32x4_f32x4([16]byte(v)))
}

// Pow2Float32 computes 2^k for each lane and returns Float32x4.
// Uses IEEE 754 bit manipulation: 2^k = ((k + 127) << 23) as float32 bits.
func (v Int32x4) Pow2Float32() Float32x4 {
	a := (*[4]int32)(unsafe.Pointer(&v))
	var result [4]float32
	Pow2F32(a[:], result[:])
	return *(*Float32x4)(unsafe.Pointer(&result))
}

// And performs element-wise bitwise AND.
func (v Int32x4) And(other Int32x4) Int32x4 {
	return Int32x4(and_i32x4([16]byte(v), [16]byte(other)))
}

// Or performs element-wise bitwise OR.
func (v Int32x4) Or(other Int32x4) Int32x4 {
	return Int32x4(or_i32x4([16]byte(v), [16]byte(other)))
}

// Xor performs element-wise bitwise XOR.
func (v Int32x4) Xor(other Int32x4) Int32x4 {
	return Int32x4(xor_i32x4([16]byte(v), [16]byte(other)))
}

// Not performs element-wise bitwise NOT.
func (v Int32x4) Not() Int32x4 {
	return Int32x4(not_i32x4([16]byte(v)))
}

// AndNot performs element-wise a AND (NOT b).
func (v Int32x4) AndNot(other Int32x4) Int32x4 {
	return Int32x4(andnot_i32x4([16]byte(v), [16]byte(other)))
}

// Equal performs element-wise equality comparison.
// Returns -1 (all bits set) for true, 0 for false - matches archsimd convention.
func (v Int32x4) Equal(other Int32x4) Int32x4 {
	return Int32x4(eq_i32x4([16]byte(v), [16]byte(other)))
}

// Greater returns a mask where v > other.
func (v Int32x4) Greater(other Int32x4) Int32x4 {
	return Int32x4(gt_i32x4([16]byte(v), [16]byte(other)))
}

// GreaterThan is an alias for Greater (matches archsimd naming).
func (v Int32x4) GreaterThan(other Int32x4) Int32x4 {
	return v.Greater(other)
}

// Less returns a mask where v < other.
func (v Int32x4) Less(other Int32x4) Int32x4 {
	return Int32x4(lt_i32x4([16]byte(v), [16]byte(other)))
}

// LessThan is an alias for Less (matches archsimd naming).
func (v Int32x4) LessThan(other Int32x4) Int32x4 {
	return v.Less(other)
}

// LessEqual returns a mask where v <= other.
func (v Int32x4) LessEqual(other Int32x4) Int32x4 {
	// le = NOT gt
	return v.Greater(other).Not()
}

// GreaterEqual returns a mask where v >= other.
func (v Int32x4) GreaterEqual(other Int32x4) Int32x4 {
	// ge = NOT lt
	return v.Less(other).Not()
}

// Merge selects elements: mask ? v : other
// mask should have -1 (all bits set) for true, 0 for false.
func (v Int32x4) Merge(other Int32x4, mask Int32x4) Int32x4 {
	return Int32x4(sel_i32x4([16]byte(mask), [16]byte(v), [16]byte(other)))
}

// StoreSlice stores the vector to a slice.
func (v Int32x4) StoreSlice(s []int32) {
	*(*Int32x4)(unsafe.Pointer(&s[0])) = v
}

// Data returns the underlying array as a slice.
func (v Int32x4) Data() []int32 {
	return (*[4]int32)(unsafe.Pointer(&v))[:]
}

// GetBit returns true if the element at index i is non-zero (used for mask operations).
func (v Int32x4) GetBit(i int) bool {
	return (*[4]int32)(unsafe.Pointer(&v))[i] != 0
}

// ReduceSum returns the sum of all elements.
func (v Int32x4) ReduceSum() int64 {
	return hsum_i32x4([16]byte(v))
}

// ReduceMax returns the maximum of all elements.
func (v Int32x4) ReduceMax() int32 {
	a := (*[4]int32)(unsafe.Pointer(&v))
	maxVal := a[0]
	for i := 1; i < 4; i++ {
		if a[i] > maxVal {
			maxVal = a[i]
		}
	}
	return maxVal
}

// ReduceMin returns the minimum of all elements.
func (v Int32x4) ReduceMin() int32 {
	a := (*[4]int32)(unsafe.Pointer(&v))
	minVal := a[0]
	for i := 1; i < 4; i++ {
		if a[i] < minVal {
			minVal = a[i]
		}
	}
	return minVal
}

// ===== Int64x2 constructors and methods =====

// BroadcastInt64x2 creates a vector with all lanes set to the given value.
func BroadcastInt64x2(v int64) Int64x2 {
	arr := [2]int64{v, v}
	return *(*Int64x2)(unsafe.Pointer(&arr))
}

// LoadInt64x2 loads 2 int64 values from a slice.
func LoadInt64x2(s []int64) Int64x2 {
	return *(*Int64x2)(unsafe.Pointer(&s[0]))
}

// LoadInt64x2Slice is an alias for LoadInt64x2 (matches archsimd naming).
func LoadInt64x2Slice(s []int64) Int64x2 {
	return LoadInt64x2(s)
}

// Load4Int64x2Slice loads 4 consecutive Int64x2 vectors (8 int64s = 64 bytes)
// using a single ARM ld1 instruction with 4 registers.
func Load4Int64x2Slice(s []int64) (Int64x2, Int64x2, Int64x2, Int64x2) {
	var v0, v1, v2, v3 Int64x2
	load4_i64x2(unsafe.Pointer(&s[0]), unsafe.Pointer(&v0), unsafe.Pointer(&v1), unsafe.Pointer(&v2), unsafe.Pointer(&v3))
	return v0, v1, v2, v3
}

// Get returns the element at the given index.
func (v Int64x2) Get(i int) int64 {
	return (*[2]int64)(unsafe.Pointer(&v))[i]
}

// Set sets the element at the given index.
func (v *Int64x2) Set(i int, val int64) {
	(*[2]int64)(unsafe.Pointer(v))[i] = val
}

// Add performs element-wise addition.
func (v Int64x2) Add(other Int64x2) Int64x2 {
	return Int64x2(add_i64x2([16]byte(v), [16]byte(other)))
}

// Sub performs element-wise subtraction.
func (v Int64x2) Sub(other Int64x2) Int64x2 {
	return Int64x2(sub_i64x2([16]byte(v), [16]byte(other)))
}

// Mul performs element-wise multiplication.
func (v Int64x2) Mul(other Int64x2) Int64x2 {
	// Note: NEON doesn't have native 64-bit integer multiply, so use scalar
	a := (*[2]int64)(unsafe.Pointer(&v))
	b := (*[2]int64)(unsafe.Pointer(&other))
	result := [2]int64{a[0] * b[0], a[1] * b[1]}
	return *(*Int64x2)(unsafe.Pointer(&result))
}

// And performs element-wise bitwise AND.
func (v Int64x2) And(other Int64x2) Int64x2 {
	return Int64x2(and_i64x2([16]byte(v), [16]byte(other)))
}

// Or performs element-wise bitwise OR.
func (v Int64x2) Or(other Int64x2) Int64x2 {
	return Int64x2(or_i64x2([16]byte(v), [16]byte(other)))
}

// Xor performs element-wise bitwise XOR.
func (v Int64x2) Xor(other Int64x2) Int64x2 {
	return Int64x2(xor_i64x2([16]byte(v), [16]byte(other)))
}

// ShiftAllLeft shifts all elements left by the given count.
func (v Int64x2) ShiftAllLeft(count int) Int64x2 {
	a := (*[2]int64)(unsafe.Pointer(&v))
	result := [2]int64{a[0] << count, a[1] << count}
	return *(*Int64x2)(unsafe.Pointer(&result))
}

// ShiftAllRight shifts all elements right by the given count.
func (v Int64x2) ShiftAllRight(count int) Int64x2 {
	a := (*[2]int64)(unsafe.Pointer(&v))
	result := [2]int64{a[0] >> count, a[1] >> count}
	return *(*Int64x2)(unsafe.Pointer(&result))
}

// Equal performs element-wise equality comparison.
// Returns -1 (all bits set) for true, 0 for false - matches archsimd convention.
func (v Int64x2) Equal(other Int64x2) Int64x2 {
	return Int64x2(eq_i64x2([16]byte(v), [16]byte(other)))
}

// Greater returns a mask where v > other.
func (v Int64x2) Greater(other Int64x2) Int64x2 {
	a := (*[2]int64)(unsafe.Pointer(&v))
	b := (*[2]int64)(unsafe.Pointer(&other))
	var r [2]int64
	if a[0] > b[0] {
		r[0] = -1
	}
	if a[1] > b[1] {
		r[1] = -1
	}
	return *(*Int64x2)(unsafe.Pointer(&r))
}

// GreaterThan is an alias for Greater (matches archsimd naming).
func (v Int64x2) GreaterThan(other Int64x2) Int64x2 {
	return v.Greater(other)
}

// Less returns a mask where v < other.
func (v Int64x2) Less(other Int64x2) Int64x2 {
	a := (*[2]int64)(unsafe.Pointer(&v))
	b := (*[2]int64)(unsafe.Pointer(&other))
	var r [2]int64
	if a[0] < b[0] {
		r[0] = -1
	}
	if a[1] < b[1] {
		r[1] = -1
	}
	return *(*Int64x2)(unsafe.Pointer(&r))
}

// LessThan is an alias for Less (matches archsimd naming).
func (v Int64x2) LessThan(other Int64x2) Int64x2 {
	return v.Less(other)
}

// LessEqual returns a mask where v <= other.
func (v Int64x2) LessEqual(other Int64x2) Int64x2 {
	a := (*[2]int64)(unsafe.Pointer(&v))
	b := (*[2]int64)(unsafe.Pointer(&other))
	var r [2]int64
	if a[0] <= b[0] {
		r[0] = -1
	}
	if a[1] <= b[1] {
		r[1] = -1
	}
	return *(*Int64x2)(unsafe.Pointer(&r))
}

// GreaterEqual returns a mask where v >= other.
func (v Int64x2) GreaterEqual(other Int64x2) Int64x2 {
	a := (*[2]int64)(unsafe.Pointer(&v))
	b := (*[2]int64)(unsafe.Pointer(&other))
	var r [2]int64
	if a[0] >= b[0] {
		r[0] = -1
	}
	if a[1] >= b[1] {
		r[1] = -1
	}
	return *(*Int64x2)(unsafe.Pointer(&r))
}

// AsFloat64x2 reinterprets bits as float64.
func (v Int64x2) AsFloat64x2() Float64x2 {
	return Float64x2(v)
}

// ConvertToFloat64 converts int64 to float64.
func (v Int64x2) ConvertToFloat64() Float64x2 {
	a := (*[2]int64)(unsafe.Pointer(&v))
	result := [2]float64{float64(a[0]), float64(a[1])}
	return *(*Float64x2)(unsafe.Pointer(&result))
}

// StoreSlice stores the vector to a slice.
func (v Int64x2) StoreSlice(s []int64) {
	*(*Int64x2)(unsafe.Pointer(&s[0])) = v
}

// Data returns the underlying array as a slice.
func (v Int64x2) Data() []int64 {
	return (*[2]int64)(unsafe.Pointer(&v))[:]
}

// GetBit returns true if the element at index i is non-zero (used for mask operations).
func (v Int64x2) GetBit(i int) bool {
	return (*[2]int64)(unsafe.Pointer(&v))[i] != 0
}

// Max performs element-wise maximum.
func (v Int64x2) Max(other Int64x2) Int64x2 {
	a := (*[2]int64)(unsafe.Pointer(&v))
	b := (*[2]int64)(unsafe.Pointer(&other))
	result := [2]int64{a[0], a[1]}
	if b[0] > result[0] {
		result[0] = b[0]
	}
	if b[1] > result[1] {
		result[1] = b[1]
	}
	return *(*Int64x2)(unsafe.Pointer(&result))
}

// Min performs element-wise minimum.
func (v Int64x2) Min(other Int64x2) Int64x2 {
	a := (*[2]int64)(unsafe.Pointer(&v))
	b := (*[2]int64)(unsafe.Pointer(&other))
	result := [2]int64{a[0], a[1]}
	if b[0] < result[0] {
		result[0] = b[0]
	}
	if b[1] < result[1] {
		result[1] = b[1]
	}
	return *(*Int64x2)(unsafe.Pointer(&result))
}

// ReduceMax returns the maximum of all elements.
func (v Int64x2) ReduceMax() int64 {
	a := (*[2]int64)(unsafe.Pointer(&v))
	if a[0] > a[1] {
		return a[0]
	}
	return a[1]
}

// ReduceMin returns the minimum of all elements.
func (v Int64x2) ReduceMin() int64 {
	a := (*[2]int64)(unsafe.Pointer(&v))
	if a[0] < a[1] {
		return a[0]
	}
	return a[1]
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

// ===== Mask operations =====

// FindFirstTrue returns the index of the first true lane in the mask, or -1 if none.
// For Int32x4 masks, a non-zero value (typically -1/0xFFFFFFFF) indicates true.
func FindFirstTrue[T Int32x4 | Int64x2 | Uint32x4 | Uint64x2](mask T) int {
	switch m := any(mask).(type) {
	case Int32x4:
		a := (*[4]int32)(unsafe.Pointer(&m))
		for i := range 4 {
			if a[i] != 0 {
				return i
			}
		}
	case Int64x2:
		a := (*[2]int64)(unsafe.Pointer(&m))
		for i := range 2 {
			if a[i] != 0 {
				return i
			}
		}
	case Uint32x4:
		a := (*[4]uint32)(unsafe.Pointer(&m))
		for i := range 4 {
			if a[i] != 0 {
				return i
			}
		}
	case Uint64x2:
		a := (*[2]uint64)(unsafe.Pointer(&m))
		for i := range 2 {
			if a[i] != 0 {
				return i
			}
		}
	}
	return -1
}

// CountTrue returns the number of true lanes in the mask.
// For integer masks, a non-zero value indicates true.
func CountTrue[T Int32x4 | Int64x2 | Uint32x4 | Uint64x2](mask T) int {
	switch m := any(mask).(type) {
	case Int32x4:
		return int(counttrue_i32x4([16]byte(m)))
	case Int64x2:
		a := (*[2]int64)(unsafe.Pointer(&m))
		count := 0
		for i := range 2 {
			if a[i] != 0 {
				count++
			}
		}
		return count
	case Uint32x4:
		return int(counttrue_i32x4([16]byte(m)))
	case Uint64x2:
		a := (*[2]uint64)(unsafe.Pointer(&m))
		count := 0
		for i := range 2 {
			if a[i] != 0 {
				count++
			}
		}
		return count
	}
	return 0
}

// AllTrue returns true if all lanes in the mask are true.
func AllTrue[T Int32x4 | Int64x2 | Uint32x4 | Uint64x2](mask T) bool {
	switch m := any(mask).(type) {
	case Int32x4:
		return alltrue_i32x4([16]byte(m)) != 0
	case Int64x2:
		a := (*[2]int64)(unsafe.Pointer(&m))
		return a[0] != 0 && a[1] != 0
	case Uint32x4:
		return alltrue_i32x4([16]byte(m)) != 0
	case Uint64x2:
		a := (*[2]uint64)(unsafe.Pointer(&m))
		return a[0] != 0 && a[1] != 0
	}
	return false
}

// AllFalse returns true if all lanes in the mask are false.
func AllFalse[T Int32x4 | Int64x2 | Uint32x4 | Uint64x2](mask T) bool {
	switch m := any(mask).(type) {
	case Int32x4:
		return anytrue_i32x4([16]byte(m)) == 0
	case Int64x2:
		a := (*[2]int64)(unsafe.Pointer(&m))
		return a[0] == 0 && a[1] == 0
	case Uint32x4:
		return anytrue_i32x4([16]byte(m)) == 0
	case Uint64x2:
		a := (*[2]uint64)(unsafe.Pointer(&m))
		return a[0] == 0 && a[1] == 0
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
	{0, 0, 0, 0},  // 0b0000: no elements
	{0, 0, 0, 0},  // 0b0001: lane 0
	{4, 0, 0, 0},  // 0b0010: lane 1
	{0, 4, 0, 0},  // 0b0011: lane 0,1
	{8, 0, 0, 0},  // 0b0100: lane 2
	{0, 8, 0, 0},  // 0b0101: lane 0,2
	{4, 8, 0, 0},  // 0b0110: lane 1,2
	{0, 4, 8, 0},  // 0b0111: lane 0,1,2
	{12, 0, 0, 0}, // 0b1000: lane 3
	{0, 12, 0, 0}, // 0b1001: lane 0,3
	{4, 12, 0, 0}, // 0b1010: lane 1,3
	{0, 4, 12, 0}, // 0b1011: lane 0,1,3
	{8, 12, 0, 0}, // 0b1100: lane 2,3
	{0, 8, 12, 0}, // 0b1101: lane 0,2,3
	{4, 8, 12, 0}, // 0b1110: lane 1,2,3
	{0, 4, 8, 12}, // 0b1111: all elements
}

// compressCountTable maps a 4-bit mask to the count of set bits.
var compressCountTable = [16]int{0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4}

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
	m := (*[4]int32)(unsafe.Pointer(&mask))
	idx := 0
	if m[0] != 0 {
		idx |= 1
	}
	if m[1] != 0 {
		idx |= 2
	}
	if m[2] != 0 {
		idx |= 4
	}
	if m[3] != 0 {
		idx |= 8
	}
	return idx
}

// CompressKeysF32x4 reorders v so elements where mask=true come first.
// Returns (reordered vector, count of true elements).
// Unlike CompressStore which discards false elements, this keeps ALL elements
// in partition order: [true elements...][false elements...].
func CompressKeysF32x4(v Float32x4, mask Int32x4) (Float32x4, int) {
	idx := maskToIndex4(mask)
	count := compressCountTable[idx]
	perm := &compressPartitionTableF32[idx]

	f := (*[4]float32)(unsafe.Pointer(&v))
	result := [4]float32{f[perm[0]], f[perm[1]], f[perm[2]], f[perm[3]]}
	return *(*Float32x4)(unsafe.Pointer(&result)), count
}

// CompressKeysI32x4 reorders v so elements where mask=true come first.
// Returns (reordered vector, count of true elements).
func CompressKeysI32x4(v Int32x4, mask Int32x4) (Int32x4, int) {
	idx := maskToIndex4(mask)
	count := compressCountTable[idx]
	perm := &compressPartitionTableF32[idx]

	a := (*[4]int32)(unsafe.Pointer(&v))
	result := [4]int32{a[perm[0]], a[perm[1]], a[perm[2]], a[perm[3]]}
	return *(*Int32x4)(unsafe.Pointer(&result)), count
}

// CompressKeysF64x2 reorders v so elements where mask=true come first.
// Returns (reordered vector, count of true elements).
// For 2-lane vectors, only 4 permutations are possible.
func CompressKeysF64x2(v Float64x2, mask Int64x2) (Float64x2, int) {
	f := (*[2]float64)(unsafe.Pointer(&v))
	m := (*[2]int64)(unsafe.Pointer(&mask))

	// Compute mask index (0-3)
	idx := 0
	if m[0] != 0 {
		idx |= 1
	}
	if m[1] != 0 {
		idx |= 2
	}

	var result [2]float64
	switch idx {
	case 0: // 00: neither true
		result[0] = f[0]
		result[1] = f[1]
		return *(*Float64x2)(unsafe.Pointer(&result)), 0
	case 1: // 01: first true
		result[0] = f[0]
		result[1] = f[1]
		return *(*Float64x2)(unsafe.Pointer(&result)), 1
	case 2: // 10: second true
		result[0] = f[1]
		result[1] = f[0]
		return *(*Float64x2)(unsafe.Pointer(&result)), 1
	case 3: // 11: both true
		result[0] = f[0]
		result[1] = f[1]
		return *(*Float64x2)(unsafe.Pointer(&result)), 2
	}
	return *(*Float64x2)(unsafe.Pointer(&result)), 0
}

// CompressKeysI64x2 reorders v so elements where mask=true come first.
// Returns (reordered vector, count of true elements).
func CompressKeysI64x2(v Int64x2, mask Int64x2) (Int64x2, int) {
	a := (*[2]int64)(unsafe.Pointer(&v))
	m := (*[2]int64)(unsafe.Pointer(&mask))

	idx := 0
	if m[0] != 0 {
		idx |= 1
	}
	if m[1] != 0 {
		idx |= 2
	}

	var result [2]int64
	switch idx {
	case 0:
		result[0] = a[0]
		result[1] = a[1]
		return *(*Int64x2)(unsafe.Pointer(&result)), 0
	case 1:
		result[0] = a[0]
		result[1] = a[1]
		return *(*Int64x2)(unsafe.Pointer(&result)), 1
	case 2:
		result[0] = a[1]
		result[1] = a[0]
		return *(*Int64x2)(unsafe.Pointer(&result)), 1
	case 3:
		result[0] = a[0]
		result[1] = a[1]
		return *(*Int64x2)(unsafe.Pointer(&result)), 2
	}
	return *(*Int64x2)(unsafe.Pointer(&result)), 0
}

// ===== Fast CountTrue functions =====
// These use popcount to count set mask bits efficiently.

// CountTrueF32x4 returns the count of true lanes in an Int32x4 mask.
func CountTrueF32x4(mask Int32x4) int {
	return int(counttrue_i32x4([16]byte(mask)))
}

// CountTrueF64x2 returns the count of true lanes in an Int64x2 mask.
func CountTrueF64x2(mask Int64x2) int {
	m := (*[2]int64)(unsafe.Pointer(&mask))
	count := 0
	if m[0] != 0 {
		count++
	}
	if m[1] != 0 {
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
	idx := maskToIndex4(mask)
	count := compressCountTable[idx]
	perm := &compressTableF32[idx]

	f := (*[4]float32)(unsafe.Pointer(&v))
	// Gather elements using byte offsets
	// Each byte offset / 4 gives the lane index
	for i := range count {
		dst[i] = f[perm[i]>>2]
	}

	return count
}

// CompressStoreF64x2 compresses float64 elements and stores to dst.
// Returns number of elements stored.
func CompressStoreF64x2(v Float64x2, mask Int64x2, dst []float64) int {
	f := (*[2]float64)(unsafe.Pointer(&v))
	m := (*[2]int64)(unsafe.Pointer(&mask))
	count := 0
	for i := range 2 {
		if m[i] != 0 {
			if count < len(dst) {
				dst[count] = f[i]
			}
			count++
		}
	}
	return count
}

// CompressStoreI32x4 compresses int32 elements and stores to dst.
// Returns number of elements stored.
func CompressStoreI32x4(v Int32x4, mask Int32x4, dst []int32) int {
	a := (*[4]int32)(unsafe.Pointer(&v))
	m := (*[4]int32)(unsafe.Pointer(&mask))
	count := 0
	for i := range 4 {
		if m[i] != 0 {
			if count < len(dst) {
				dst[count] = a[i]
			}
			count++
		}
	}
	return count
}

// CompressStoreI64x2 compresses int64 elements and stores to dst.
// Returns number of elements stored.
func CompressStoreI64x2(v Int64x2, mask Int64x2, dst []int64) int {
	a := (*[2]int64)(unsafe.Pointer(&v))
	m := (*[2]int64)(unsafe.Pointer(&mask))
	count := 0
	for i := range 2 {
		if m[i] != 0 {
			if count < len(dst) {
				dst[count] = a[i]
			}
			count++
		}
	}
	return count
}

// CompressStoreU32x4 compresses uint32 elements and stores to dst.
// Returns number of elements stored.
// Uses lookup table for efficient gathering without branches.
func CompressStoreU32x4(v Uint32x4, mask Int32x4, dst []uint32) int {
	idx := maskToIndex4(mask)
	count := compressCountTable[idx]
	perm := &compressTableF32[idx]

	u := (*[4]uint32)(unsafe.Pointer(&v))
	// Gather elements using byte offsets
	// Each byte offset / 4 gives the lane index
	for i := range count {
		dst[i] = u[perm[i]>>2]
	}

	return count
}

// CompressStoreU64x2 compresses uint64 elements and stores to dst.
// Returns number of elements stored.
func CompressStoreU64x2(v Uint64x2, mask Int64x2, dst []uint64) int {
	a := (*[2]uint64)(unsafe.Pointer(&v))
	m := (*[2]int64)(unsafe.Pointer(&mask))
	count := 0
	for i := range 2 {
		if m[i] != 0 {
			if count < len(dst) {
				dst[count] = a[i]
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
	var mask [4]int32
	firstNI32x4Asm(n, &mask)
	return *(*Int32x4)(unsafe.Pointer(&mask))
}

// FirstNI64x2 returns a mask with the first n lanes set to true.
func FirstNI64x2(n int) Int64x2 {
	var mask [2]int64
	firstNI64x2Asm(n, &mask)
	return *(*Int64x2)(unsafe.Pointer(&mask))
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

// CompressStoreUint32 compresses and stores uint32 elements.
func CompressStoreUint32(v Uint32x4, mask Int32x4, dst []uint32) int {
	return CompressStoreU32x4(v, mask, dst)
}

// CompressStoreUint64 compresses and stores uint64 elements.
func CompressStoreUint64(v Uint64x2, mask Int64x2, dst []uint64) int {
	return CompressStoreU64x2(v, mask, dst)
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
	return yes.Merge(no, mask)
}

// IfThenElseInt64 selects elements based on mask: result = mask ? yes : no (for int64).
func IfThenElseInt64(mask Int64x2, yes, no Int64x2) Int64x2 {
	f1 := (*[2]int64)(unsafe.Pointer(&yes))
	f2 := (*[2]int64)(unsafe.Pointer(&no))
	m := (*[2]int64)(unsafe.Pointer(&mask))
	var result [2]int64
	for i := range 2 {
		if m[i] != 0 {
			result[i] = f1[i]
		} else {
			result[i] = f2[i]
		}
	}
	return *(*Int64x2)(unsafe.Pointer(&result))
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
	return a.AndNot(b)
}

// MaskAndNotFloat64 performs a AND (NOT b) on masks (for float64).
func MaskAndNotFloat64(a, b Int64x2) Int64x2 {
	a1 := (*[2]int64)(unsafe.Pointer(&a))
	b1 := (*[2]int64)(unsafe.Pointer(&b))
	result := [2]int64{a1[0] &^ b1[0], a1[1] &^ b1[1]}
	return *(*Int64x2)(unsafe.Pointer(&result))
}

// AllTrueVal returns true if all lanes in the mask are true (for float32).
func AllTrueVal(mask Int32x4) bool {
	return alltrue_i32x4([16]byte(mask)) != 0
}

// AllTrueValFloat64 returns true if all lanes in the mask are true (for float64).
func AllTrueValFloat64(mask Int64x2) bool {
	m := (*[2]int64)(unsafe.Pointer(&mask))
	return m[0] != 0 && m[1] != 0
}

// AllTrueValUint32 returns true if all lanes in the mask are true (for uint32).
func AllTrueValUint32(mask Uint32x4) bool {
	return alltrue_i32x4([16]byte(mask)) != 0
}

// AllTrueValUint64 returns true if all lanes in the mask are true (for uint64).
func AllTrueValUint64(mask Uint64x2) bool {
	m := (*[2]uint64)(unsafe.Pointer(&mask))
	return m[0] != 0 && m[1] != 0
}

// AllFalseVal returns true if all lanes in the mask are false (for float32).
func AllFalseVal(mask Int32x4) bool {
	return anytrue_i32x4([16]byte(mask)) == 0
}

// AllFalseValFloat64 returns true if all lanes in the mask are false (for float64).
func AllFalseValFloat64(mask Int64x2) bool {
	m := (*[2]int64)(unsafe.Pointer(&mask))
	return m[0] == 0 && m[1] == 0
}

// AllFalseValUint32 returns true if all lanes in the mask are false (for uint32).
func AllFalseValUint32(mask Uint32x4) bool {
	return anytrue_i32x4([16]byte(mask)) == 0
}

// AllFalseValUint64 returns true if all lanes in the mask are false (for uint64).
func AllFalseValUint64(mask Uint64x2) bool {
	m := (*[2]uint64)(unsafe.Pointer(&mask))
	return m[0] == 0 && m[1] == 0
}

// ============================================================================
// Uint8x16 - 128-bit vector of 16 uint8 values
// ============================================================================

// Uint8x16 represents a 128-bit NEON vector of 16 uint8 values.
type Uint8x16 [16]byte

// BroadcastUint8x16 creates a vector with all lanes set to the given value.
func BroadcastUint8x16(v uint8) Uint8x16 {
	var arr [16]uint8
	for i := range arr {
		arr[i] = v
	}
	return *(*Uint8x16)(unsafe.Pointer(&arr))
}

// LoadUint8x16 loads 16 uint8 values from a slice.
func LoadUint8x16(s []uint8) Uint8x16 {
	return *(*Uint8x16)(unsafe.Pointer(&s[0]))
}

// LoadUint8x16Slice is an alias for LoadUint8x16 for consistency with other types.
func LoadUint8x16Slice(s []uint8) Uint8x16 {
	return LoadUint8x16(s)
}

// Load4Uint8x16Slice loads 4 consecutive Uint8x16 vectors (64 bytes)
// using a single ARM ld1 instruction with 4 registers.
func Load4Uint8x16Slice(s []uint8) (Uint8x16, Uint8x16, Uint8x16, Uint8x16) {
	var v0, v1, v2, v3 Uint8x16
	load4_u8x16(unsafe.Pointer(&s[0]), unsafe.Pointer(&v0), unsafe.Pointer(&v1), unsafe.Pointer(&v2), unsafe.Pointer(&v3))
	return v0, v1, v2, v3
}

// ZeroUint8x16 returns a zero vector.
func ZeroUint8x16() Uint8x16 {
	return Uint8x16{}
}

// Get returns the element at the given index.
func (v Uint8x16) Get(i int) uint8 {
	return v[i]
}

// Set sets the element at the given index.
func (v *Uint8x16) Set(i int, val uint8) {
	v[i] = val
}

// StoreSlice stores the vector to a slice.
func (v Uint8x16) StoreSlice(s []uint8) {
	*(*Uint8x16)(unsafe.Pointer(&s[0])) = v
}

// Data returns the underlying data as a slice.
func (v Uint8x16) Data() []uint8 {
	return v[:]
}

// Add performs element-wise addition (wrapping).
func (v Uint8x16) Add(other Uint8x16) Uint8x16 {
	// Reuse signed int8 add - bitwise identical
	result := add_i32x4([16]byte(v), [16]byte(other))
	return Uint8x16(result)
}

// Sub performs element-wise subtraction (wrapping).
func (v Uint8x16) Sub(other Uint8x16) Uint8x16 {
	result := sub_i32x4([16]byte(v), [16]byte(other))
	return Uint8x16(result)
}

// AddSaturated performs element-wise saturating addition.
func (v Uint8x16) AddSaturated(other Uint8x16) Uint8x16 {
	return Uint8x16(adds_u8x16([16]byte(v), [16]byte(other)))
}

// SubSaturated performs element-wise saturating subtraction.
func (v Uint8x16) SubSaturated(other Uint8x16) Uint8x16 {
	return Uint8x16(subs_u8x16([16]byte(v), [16]byte(other)))
}

// Min performs element-wise unsigned minimum.
func (v Uint8x16) Min(other Uint8x16) Uint8x16 {
	return Uint8x16(min_u8x16([16]byte(v), [16]byte(other)))
}

// Max performs element-wise unsigned maximum.
func (v Uint8x16) Max(other Uint8x16) Uint8x16 {
	return Uint8x16(max_u8x16([16]byte(v), [16]byte(other)))
}

// LessThan returns a mask where v < other (unsigned comparison).
func (v Uint8x16) LessThan(other Uint8x16) Uint8x16 {
	return Uint8x16(lt_u8x16([16]byte(v), [16]byte(other)))
}

// GreaterThan returns a mask where v > other (unsigned comparison).
func (v Uint8x16) GreaterThan(other Uint8x16) Uint8x16 {
	return Uint8x16(gt_u8x16([16]byte(v), [16]byte(other)))
}

// LessEqual returns a mask where v <= other (unsigned comparison).
func (v Uint8x16) LessEqual(other Uint8x16) Uint8x16 {
	return Uint8x16(le_u8x16([16]byte(v), [16]byte(other)))
}

// GreaterEqual returns a mask where v >= other (unsigned comparison).
func (v Uint8x16) GreaterEqual(other Uint8x16) Uint8x16 {
	return Uint8x16(ge_u8x16([16]byte(v), [16]byte(other)))
}

// Equal returns a mask where v == other.
func (v Uint8x16) Equal(other Uint8x16) Uint8x16 {
	return Uint8x16(eq_u8x16([16]byte(v), [16]byte(other)))
}

// And performs bitwise AND.
func (v Uint8x16) And(other Uint8x16) Uint8x16 {
	return Uint8x16(and_u8x16([16]byte(v), [16]byte(other)))
}

// Or performs bitwise OR.
func (v Uint8x16) Or(other Uint8x16) Uint8x16 {
	return Uint8x16(or_u8x16([16]byte(v), [16]byte(other)))
}

// Xor performs bitwise XOR.
func (v Uint8x16) Xor(other Uint8x16) Uint8x16 {
	return Uint8x16(xor_u8x16([16]byte(v), [16]byte(other)))
}

// Not performs bitwise NOT.
func (v Uint8x16) Not() Uint8x16 {
	return Uint8x16(not_u8x16([16]byte(v)))
}

// GetBit returns true if the element at index i is non-zero.
func (v Uint8x16) GetBit(i int) bool {
	return v[i] != 0
}

// TableLookupBytes performs byte-level table lookup: result[i] = v[idx[i]].
// If idx[i] >= 16, the result is 0 (NEON TBL behavior with out-of-range indices).
func (v Uint8x16) TableLookupBytes(idx Uint8x16) Uint8x16 {
	var result [16]uint8
	n := int64(16)
	tbl_u8_neon(unsafe.Pointer(&v[0]), unsafe.Pointer(&idx[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
	return *(*Uint8x16)(unsafe.Pointer(&result))
}

// ============================================================================
// Uint16x8 - 128-bit vector of 8 uint16 values
// ============================================================================

// Uint16x8 represents a 128-bit NEON vector of 8 uint16 values.
type Uint16x8 [16]byte

// BroadcastUint16x8 creates a vector with all lanes set to the given value.
func BroadcastUint16x8(v uint16) Uint16x8 {
	arr := [8]uint16{v, v, v, v, v, v, v, v}
	return *(*Uint16x8)(unsafe.Pointer(&arr))
}

// LoadUint16x8 loads 8 uint16 values from a slice.
func LoadUint16x8(s []uint16) Uint16x8 {
	return *(*Uint16x8)(unsafe.Pointer(&s[0]))
}

// Load4Uint16x8Slice loads 4 consecutive Uint16x8 vectors (32 uint16s = 64 bytes)
// using a single ARM ld1 instruction with 4 registers.
func Load4Uint16x8Slice(s []uint16) (Uint16x8, Uint16x8, Uint16x8, Uint16x8) {
	var v0, v1, v2, v3 Uint16x8
	load4_u16x8(unsafe.Pointer(&s[0]), unsafe.Pointer(&v0), unsafe.Pointer(&v1), unsafe.Pointer(&v2), unsafe.Pointer(&v3))
	return v0, v1, v2, v3
}

// ZeroUint16x8 returns a zero vector.
func ZeroUint16x8() Uint16x8 {
	return Uint16x8{}
}

// Get returns the element at the given index.
func (v Uint16x8) Get(i int) uint16 {
	return (*[8]uint16)(unsafe.Pointer(&v))[i]
}

// Set sets the element at the given index.
func (v *Uint16x8) Set(i int, val uint16) {
	(*[8]uint16)(unsafe.Pointer(v))[i] = val
}

// StoreSlice stores the vector to a slice.
func (v Uint16x8) StoreSlice(s []uint16) {
	*(*Uint16x8)(unsafe.Pointer(&s[0])) = v
}

// Data returns the underlying data as a slice.
func (v Uint16x8) Data() []uint16 {
	return (*[8]uint16)(unsafe.Pointer(&v))[:]
}

// Add performs element-wise addition (wrapping).
func (v Uint16x8) Add(other Uint16x8) Uint16x8 {
	result := add_i32x4([16]byte(v), [16]byte(other))
	return Uint16x8(result)
}

// Sub performs element-wise subtraction (wrapping).
func (v Uint16x8) Sub(other Uint16x8) Uint16x8 {
	result := sub_i32x4([16]byte(v), [16]byte(other))
	return Uint16x8(result)
}

// AddSaturated performs element-wise saturating addition.
func (v Uint16x8) AddSaturated(other Uint16x8) Uint16x8 {
	return Uint16x8(adds_u16x8([16]byte(v), [16]byte(other)))
}

// SubSaturated performs element-wise saturating subtraction.
func (v Uint16x8) SubSaturated(other Uint16x8) Uint16x8 {
	return Uint16x8(subs_u16x8([16]byte(v), [16]byte(other)))
}

// Min performs element-wise unsigned minimum.
func (v Uint16x8) Min(other Uint16x8) Uint16x8 {
	return Uint16x8(min_u16x8([16]byte(v), [16]byte(other)))
}

// Max performs element-wise unsigned maximum.
func (v Uint16x8) Max(other Uint16x8) Uint16x8 {
	return Uint16x8(max_u16x8([16]byte(v), [16]byte(other)))
}

// LessThan returns a mask where v < other (unsigned comparison).
func (v Uint16x8) LessThan(other Uint16x8) Uint16x8 {
	return Uint16x8(lt_u16x8([16]byte(v), [16]byte(other)))
}

// GreaterThan returns a mask where v > other (unsigned comparison).
func (v Uint16x8) GreaterThan(other Uint16x8) Uint16x8 {
	return Uint16x8(gt_u16x8([16]byte(v), [16]byte(other)))
}

// LessEqual returns a mask where v <= other (unsigned comparison).
func (v Uint16x8) LessEqual(other Uint16x8) Uint16x8 {
	return Uint16x8(le_u16x8([16]byte(v), [16]byte(other)))
}

// GreaterEqual returns a mask where v >= other (unsigned comparison).
func (v Uint16x8) GreaterEqual(other Uint16x8) Uint16x8 {
	return Uint16x8(ge_u16x8([16]byte(v), [16]byte(other)))
}

// Equal returns a mask where v == other.
func (v Uint16x8) Equal(other Uint16x8) Uint16x8 {
	return Uint16x8(eq_u16x8([16]byte(v), [16]byte(other)))
}

// And performs bitwise AND.
func (v Uint16x8) And(other Uint16x8) Uint16x8 {
	return Uint16x8(and_u16x8([16]byte(v), [16]byte(other)))
}

// Or performs bitwise OR.
func (v Uint16x8) Or(other Uint16x8) Uint16x8 {
	return Uint16x8(or_u16x8([16]byte(v), [16]byte(other)))
}

// Xor performs bitwise XOR.
func (v Uint16x8) Xor(other Uint16x8) Uint16x8 {
	return Uint16x8(xor_u16x8([16]byte(v), [16]byte(other)))
}

// Not performs bitwise NOT.
func (v Uint16x8) Not() Uint16x8 {
	return Uint16x8(not_u16x8([16]byte(v)))
}

// GetBit returns true if the element at index i is non-zero.
func (v Uint16x8) GetBit(i int) bool {
	return (*[8]uint16)(unsafe.Pointer(&v))[i] != 0
}

// ============================================================================
// Uint32x4 - 128-bit vector of 4 uint32 values
// ============================================================================

// Uint32x4 represents a 128-bit NEON vector of 4 uint32 values.
type Uint32x4 [16]byte

// BroadcastUint32x4 creates a vector with all lanes set to the given value.
func BroadcastUint32x4(v uint32) Uint32x4 {
	arr := [4]uint32{v, v, v, v}
	return *(*Uint32x4)(unsafe.Pointer(&arr))
}

// LoadUint32x4 loads 4 uint32 values from a slice.
func LoadUint32x4(s []uint32) Uint32x4 {
	return *(*Uint32x4)(unsafe.Pointer(&s[0]))
}

// LoadUint32x4Slice is an alias for LoadUint32x4 (matches archsimd naming).
func LoadUint32x4Slice(s []uint32) Uint32x4 {
	return LoadUint32x4(s)
}

// Load4Uint32x4Slice loads 4 consecutive Uint32x4 vectors (16 uint32s = 64 bytes)
// using a single ARM ld1 instruction with 4 registers.
func Load4Uint32x4Slice(s []uint32) (Uint32x4, Uint32x4, Uint32x4, Uint32x4) {
	var v0, v1, v2, v3 Uint32x4
	load4_u32x4(unsafe.Pointer(&s[0]), unsafe.Pointer(&v0), unsafe.Pointer(&v1), unsafe.Pointer(&v2), unsafe.Pointer(&v3))
	return v0, v1, v2, v3
}

// ZeroUint32x4 returns a zero vector.
func ZeroUint32x4() Uint32x4 {
	return Uint32x4{}
}

// Get returns the element at the given index.
func (v Uint32x4) Get(i int) uint32 {
	return (*[4]uint32)(unsafe.Pointer(&v))[i]
}

// Set sets the element at the given index.
func (v *Uint32x4) Set(i int, val uint32) {
	(*[4]uint32)(unsafe.Pointer(v))[i] = val
}

// StoreSlice stores the vector to a slice.
func (v Uint32x4) StoreSlice(s []uint32) {
	*(*Uint32x4)(unsafe.Pointer(&s[0])) = v
}

// Data returns the underlying data as a slice.
func (v Uint32x4) Data() []uint32 {
	return (*[4]uint32)(unsafe.Pointer(&v))[:]
}

// Add performs element-wise addition (wrapping).
func (v Uint32x4) Add(other Uint32x4) Uint32x4 {
	return Uint32x4(add_u32x4([16]byte(v), [16]byte(other)))
}

// Sub performs element-wise subtraction (wrapping).
func (v Uint32x4) Sub(other Uint32x4) Uint32x4 {
	return Uint32x4(sub_u32x4([16]byte(v), [16]byte(other)))
}

// Mul performs element-wise multiplication (low 32 bits).
func (v Uint32x4) Mul(other Uint32x4) Uint32x4 {
	return Uint32x4(mul_u32x4([16]byte(v), [16]byte(other)))
}

// AddSaturated performs element-wise saturating addition.
func (v Uint32x4) AddSaturated(other Uint32x4) Uint32x4 {
	return Uint32x4(adds_u32x4([16]byte(v), [16]byte(other)))
}

// SubSaturated performs element-wise saturating subtraction.
func (v Uint32x4) SubSaturated(other Uint32x4) Uint32x4 {
	return Uint32x4(subs_u32x4([16]byte(v), [16]byte(other)))
}

// Min performs element-wise unsigned minimum.
func (v Uint32x4) Min(other Uint32x4) Uint32x4 {
	return Uint32x4(min_u32x4([16]byte(v), [16]byte(other)))
}

// Max performs element-wise unsigned maximum.
func (v Uint32x4) Max(other Uint32x4) Uint32x4 {
	return Uint32x4(max_u32x4([16]byte(v), [16]byte(other)))
}

// LessThan returns a mask where v < other (unsigned comparison).
func (v Uint32x4) LessThan(other Uint32x4) Uint32x4 {
	return Uint32x4(lt_u32x4([16]byte(v), [16]byte(other)))
}

// GreaterThan returns a mask where v > other (unsigned comparison).
func (v Uint32x4) GreaterThan(other Uint32x4) Uint32x4 {
	return Uint32x4(gt_u32x4([16]byte(v), [16]byte(other)))
}

// LessEqual returns a mask where v <= other (unsigned comparison).
func (v Uint32x4) LessEqual(other Uint32x4) Uint32x4 {
	return Uint32x4(le_u32x4([16]byte(v), [16]byte(other)))
}

// GreaterEqual returns a mask where v >= other (unsigned comparison).
func (v Uint32x4) GreaterEqual(other Uint32x4) Uint32x4 {
	return Uint32x4(ge_u32x4([16]byte(v), [16]byte(other)))
}

// Equal returns a mask where v == other.
func (v Uint32x4) Equal(other Uint32x4) Uint32x4 {
	return Uint32x4(eq_u32x4([16]byte(v), [16]byte(other)))
}

// And performs bitwise AND.
func (v Uint32x4) And(other Uint32x4) Uint32x4 {
	return Uint32x4(and_u32x4([16]byte(v), [16]byte(other)))
}

// Or performs bitwise OR.
func (v Uint32x4) Or(other Uint32x4) Uint32x4 {
	return Uint32x4(or_u32x4([16]byte(v), [16]byte(other)))
}

// Xor performs bitwise XOR.
func (v Uint32x4) Xor(other Uint32x4) Uint32x4 {
	return Uint32x4(xor_u32x4([16]byte(v), [16]byte(other)))
}

// Not performs bitwise NOT.
func (v Uint32x4) Not() Uint32x4 {
	return Uint32x4(not_u32x4([16]byte(v)))
}

// AndNot performs a AND (NOT b).
func (v Uint32x4) AndNot(other Uint32x4) Uint32x4 {
	return Uint32x4(andnot_u32x4([16]byte(v), [16]byte(other)))
}

// ShiftAllLeft shifts all elements left by the given count.
func (v Uint32x4) ShiftAllLeft(count int) Uint32x4 {
	a := (*[4]uint32)(unsafe.Pointer(&v))
	result := [4]uint32{a[0] << count, a[1] << count, a[2] << count, a[3] << count}
	return *(*Uint32x4)(unsafe.Pointer(&result))
}

// ShiftAllRight shifts all elements right by the given count (logical shift).
func (v Uint32x4) ShiftAllRight(count int) Uint32x4 {
	a := (*[4]uint32)(unsafe.Pointer(&v))
	result := [4]uint32{a[0] >> count, a[1] >> count, a[2] >> count, a[3] >> count}
	return *(*Uint32x4)(unsafe.Pointer(&result))
}

// ReduceSum returns the sum of all elements.
func (v Uint32x4) ReduceSum() uint64 {
	return uint64(hsum_u32x4([16]byte(v)))
}

// ReduceMax returns the maximum element.
func (v Uint32x4) ReduceMax() uint32 {
	a := (*[4]uint32)(unsafe.Pointer(&v))
	maxVal := a[0]
	for i := 1; i < 4; i++ {
		if a[i] > maxVal {
			maxVal = a[i]
		}
	}
	return maxVal
}

// GetBit returns true if the element at index i is non-zero.
func (v Uint32x4) GetBit(i int) bool {
	return (*[4]uint32)(unsafe.Pointer(&v))[i] != 0
}

// AsInt32x4 reinterprets bits as Int32x4.
func (v Uint32x4) AsInt32x4() Int32x4 {
	return Int32x4(v)
}

// ============================================================================
// Uint64x2 - 128-bit vector of 2 uint64 values
// ============================================================================

// Uint64x2 represents a 128-bit NEON vector of 2 uint64 values.
type Uint64x2 [16]byte

// BroadcastUint64x2 creates a vector with all lanes set to the given value.
func BroadcastUint64x2(v uint64) Uint64x2 {
	arr := [2]uint64{v, v}
	return *(*Uint64x2)(unsafe.Pointer(&arr))
}

// LoadUint64x2 loads 2 uint64 values from a slice.
func LoadUint64x2(s []uint64) Uint64x2 {
	return *(*Uint64x2)(unsafe.Pointer(&s[0]))
}

// LoadUint64x2Slice is an alias for LoadUint64x2 (matches archsimd naming).
func LoadUint64x2Slice(s []uint64) Uint64x2 {
	return LoadUint64x2(s)
}

// Load4Uint64x2Slice loads 4 consecutive Uint64x2 vectors (8 uint64s = 64 bytes)
// using a single ARM ld1 instruction with 4 registers.
func Load4Uint64x2Slice(s []uint64) (Uint64x2, Uint64x2, Uint64x2, Uint64x2) {
	var v0, v1, v2, v3 Uint64x2
	load4_u64x2(unsafe.Pointer(&s[0]), unsafe.Pointer(&v0), unsafe.Pointer(&v1), unsafe.Pointer(&v2), unsafe.Pointer(&v3))
	return v0, v1, v2, v3
}

// ZeroUint64x2 returns a zero vector.
func ZeroUint64x2() Uint64x2 {
	return Uint64x2{}
}

// Get returns the element at the given index.
func (v Uint64x2) Get(i int) uint64 {
	return (*[2]uint64)(unsafe.Pointer(&v))[i]
}

// Set sets the element at the given index.
func (v *Uint64x2) Set(i int, val uint64) {
	(*[2]uint64)(unsafe.Pointer(v))[i] = val
}

// StoreSlice stores the vector to a slice.
func (v Uint64x2) StoreSlice(s []uint64) {
	*(*Uint64x2)(unsafe.Pointer(&s[0])) = v
}

// Data returns the underlying data as a slice.
func (v Uint64x2) Data() []uint64 {
	return (*[2]uint64)(unsafe.Pointer(&v))[:]
}

// Add performs element-wise addition (wrapping).
func (v Uint64x2) Add(other Uint64x2) Uint64x2 {
	return Uint64x2(add_u64x2([16]byte(v), [16]byte(other)))
}

// Sub performs element-wise subtraction (wrapping).
func (v Uint64x2) Sub(other Uint64x2) Uint64x2 {
	return Uint64x2(sub_u64x2([16]byte(v), [16]byte(other)))
}

// Mul performs element-wise multiplication.
// Note: NEON doesn't have native 64-bit multiply, uses scalar fallback.
func (v Uint64x2) Mul(other Uint64x2) Uint64x2 {
	a := (*[2]uint64)(unsafe.Pointer(&v))
	b := (*[2]uint64)(unsafe.Pointer(&other))
	result := [2]uint64{a[0] * b[0], a[1] * b[1]}
	return *(*Uint64x2)(unsafe.Pointer(&result))
}

// AddSaturated performs element-wise saturating addition.
func (v Uint64x2) AddSaturated(other Uint64x2) Uint64x2 {
	return Uint64x2(adds_u64x2([16]byte(v), [16]byte(other)))
}

// SubSaturated performs element-wise saturating subtraction.
func (v Uint64x2) SubSaturated(other Uint64x2) Uint64x2 {
	return Uint64x2(subs_u64x2([16]byte(v), [16]byte(other)))
}

// Min performs element-wise unsigned minimum.
func (v Uint64x2) Min(other Uint64x2) Uint64x2 {
	return Uint64x2(min_u64x2([16]byte(v), [16]byte(other)))
}

// Max performs element-wise unsigned maximum.
func (v Uint64x2) Max(other Uint64x2) Uint64x2 {
	return Uint64x2(max_u64x2([16]byte(v), [16]byte(other)))
}

// LessThan returns a mask where v < other (unsigned comparison).
func (v Uint64x2) LessThan(other Uint64x2) Uint64x2 {
	return Uint64x2(lt_u64x2([16]byte(v), [16]byte(other)))
}

// GreaterThan returns a mask where v > other (unsigned comparison).
func (v Uint64x2) GreaterThan(other Uint64x2) Uint64x2 {
	return Uint64x2(gt_u64x2([16]byte(v), [16]byte(other)))
}

// LessEqual returns a mask where v <= other (unsigned comparison).
func (v Uint64x2) LessEqual(other Uint64x2) Uint64x2 {
	return Uint64x2(le_u64x2([16]byte(v), [16]byte(other)))
}

// GreaterEqual returns a mask where v >= other (unsigned comparison).
func (v Uint64x2) GreaterEqual(other Uint64x2) Uint64x2 {
	return Uint64x2(ge_u64x2([16]byte(v), [16]byte(other)))
}

// Equal returns a mask where v == other.
func (v Uint64x2) Equal(other Uint64x2) Uint64x2 {
	return Uint64x2(eq_u64x2([16]byte(v), [16]byte(other)))
}

// And performs bitwise AND.
func (v Uint64x2) And(other Uint64x2) Uint64x2 {
	return Uint64x2(and_u64x2([16]byte(v), [16]byte(other)))
}

// Or performs bitwise OR.
func (v Uint64x2) Or(other Uint64x2) Uint64x2 {
	return Uint64x2(or_u64x2([16]byte(v), [16]byte(other)))
}

// Xor performs bitwise XOR.
func (v Uint64x2) Xor(other Uint64x2) Uint64x2 {
	return Uint64x2(xor_u64x2([16]byte(v), [16]byte(other)))
}

// Not performs bitwise NOT.
func (v Uint64x2) Not() Uint64x2 {
	allOnes := [16]byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}
	return Uint64x2(xor_u64x2([16]byte(v), allOnes))
}

// ShiftAllLeft shifts all elements left by the given count.
func (v Uint64x2) ShiftAllLeft(count int) Uint64x2 {
	a := (*[2]uint64)(unsafe.Pointer(&v))
	result := [2]uint64{a[0] << count, a[1] << count}
	return *(*Uint64x2)(unsafe.Pointer(&result))
}

// ShiftAllRight shifts all elements right by the given count (logical shift).
func (v Uint64x2) ShiftAllRight(count int) Uint64x2 {
	a := (*[2]uint64)(unsafe.Pointer(&v))
	result := [2]uint64{a[0] >> count, a[1] >> count}
	return *(*Uint64x2)(unsafe.Pointer(&result))
}

// Merge selects elements: mask ? v : other
func (v Uint64x2) Merge(other Uint64x2, mask Uint64x2) Uint64x2 {
	return Uint64x2(sel_u64x2([16]byte(mask), [16]byte(v), [16]byte(other)))
}

// ReduceMax returns the maximum element.
func (v Uint64x2) ReduceMax() uint64 {
	a := (*[2]uint64)(unsafe.Pointer(&v))
	if a[0] > a[1] {
		return a[0]
	}
	return a[1]
}

// GetBit returns true if the element at index i is non-zero.
func (v Uint64x2) GetBit(i int) bool {
	return (*[2]uint64)(unsafe.Pointer(&v))[i] != 0
}

// AsInt64x2 reinterprets bits as Int64x2.
func (v Uint64x2) AsInt64x2() Int64x2 {
	return Int64x2(v)
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
	f1 := (*[4]float32)(unsafe.Pointer(&v1))
	v1 = *(*Float32x4)(unsafe.Pointer(&[4]float32{f1[3], f1[2], f1[1], f1[0]}))

	// Bitonic merge across vectors
	// Stage 1: Compare v0[i] with v1[i]
	lo := v0.Min(v1)
	hi := v0.Max(v1)

	// Stage 2: Compare within each vector (distance 2)
	loF := (*[4]float32)(unsafe.Pointer(&lo))
	hiF := (*[4]float32)(unsafe.Pointer(&hi))
	v0 = *(*Float32x4)(unsafe.Pointer(&[4]float32{loF[0], loF[1], hiF[0], hiF[1]}))
	v1 = *(*Float32x4)(unsafe.Pointer(&[4]float32{loF[2], loF[3], hiF[2], hiF[3]}))
	lo = v0.Min(v1)
	hi = v0.Max(v1)

	// Stage 3: Compare within each vector (distance 1)
	loF = (*[4]float32)(unsafe.Pointer(&lo))
	hiF = (*[4]float32)(unsafe.Pointer(&hi))
	v0 = *(*Float32x4)(unsafe.Pointer(&[4]float32{loF[0], hiF[0], loF[2], hiF[2]}))
	v1 = *(*Float32x4)(unsafe.Pointer(&[4]float32{loF[1], hiF[1], loF[3], hiF[3]}))
	lo = v0.Min(v1)
	hi = v0.Max(v1)

	// Interleave results
	loF = (*[4]float32)(unsafe.Pointer(&lo))
	hiF = (*[4]float32)(unsafe.Pointer(&hi))
	data[0] = loF[0]
	data[1] = hiF[0]
	data[2] = loF[1]
	data[3] = hiF[1]
	data[4] = loF[2]
	data[5] = hiF[2]
	data[6] = loF[3]
	data[7] = hiF[3]
}

// sortVec4F32 sorts 4 elements within a vector using a sorting network.
func sortVec4F32(v Float32x4) Float32x4 {
	f := (*[4]float32)(unsafe.Pointer(&v))
	a, b, c, d := f[0], f[1], f[2], f[3]

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

	result := [4]float32{a, b, c, d}
	return *(*Float32x4)(unsafe.Pointer(&result))
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

// ===== SlideUpLanes operations =====
// SlideUpLanes shifts lanes up by offset positions, filling low lanes with zeros.
// [a,b,c,d] with offset=1 -> [0,a,b,c]

// SlideUpLanesFloat32x4 shifts Float32x4 lanes up by offset.
func SlideUpLanesFloat32x4(v Float32x4, offset int) Float32x4 {
	switch offset {
	case 1:
		return Float32x4(slide_up_1_f32x4([16]byte(v)))
	case 2:
		return Float32x4(slide_up_2_f32x4([16]byte(v)))
	default:
		arr := (*[4]float32)(unsafe.Pointer(&v))
		var out [4]float32
		for i := offset; i < 4; i++ {
			out[i] = arr[i-offset]
		}
		return *(*Float32x4)(unsafe.Pointer(&out))
	}
}

// SlideUpLanesFloat64x2 shifts Float64x2 lanes up by offset.
func SlideUpLanesFloat64x2(v Float64x2, offset int) Float64x2 {
	if offset == 1 {
		return Float64x2(slide_up_1_f64x2([16]byte(v)))
	}
	arr := (*[2]float64)(unsafe.Pointer(&v))
	var out [2]float64
	for i := offset; i < 2; i++ {
		out[i] = arr[i-offset]
	}
	return *(*Float64x2)(unsafe.Pointer(&out))
}

// SlideUpLanesInt32x4 shifts Int32x4 lanes up by offset.
func SlideUpLanesInt32x4(v Int32x4, offset int) Int32x4 {
	switch offset {
	case 1:
		return Int32x4(slide_up_1_i32x4([16]byte(v)))
	case 2:
		return Int32x4(slide_up_2_i32x4([16]byte(v)))
	default:
		arr := (*[4]int32)(unsafe.Pointer(&v))
		var out [4]int32
		for i := offset; i < 4; i++ {
			out[i] = arr[i-offset]
		}
		return *(*Int32x4)(unsafe.Pointer(&out))
	}
}

// SlideUpLanesInt64x2 shifts Int64x2 lanes up by offset.
func SlideUpLanesInt64x2(v Int64x2, offset int) Int64x2 {
	if offset == 1 {
		return Int64x2(slide_up_1_i64x2([16]byte(v)))
	}
	arr := (*[2]int64)(unsafe.Pointer(&v))
	var out [2]int64
	for i := offset; i < 2; i++ {
		out[i] = arr[i-offset]
	}
	return *(*Int64x2)(unsafe.Pointer(&out))
}

// SlideUpLanesUint32x4 shifts Uint32x4 lanes up by offset.
func SlideUpLanesUint32x4(v Uint32x4, offset int) Uint32x4 {
	switch offset {
	case 1:
		return Uint32x4(slide_up_1_u32x4([16]byte(v)))
	case 2:
		return Uint32x4(slide_up_2_u32x4([16]byte(v)))
	default:
		arr := (*[4]uint32)(unsafe.Pointer(&v))
		var out [4]uint32
		for i := offset; i < 4; i++ {
			out[i] = arr[i-offset]
		}
		return *(*Uint32x4)(unsafe.Pointer(&out))
	}
}

// SlideUpLanesUint64x2 shifts Uint64x2 lanes up by offset.
func SlideUpLanesUint64x2(v Uint64x2, offset int) Uint64x2 {
	if offset == 1 {
		return Uint64x2(slide_up_1_u64x2([16]byte(v)))
	}
	arr := (*[2]uint64)(unsafe.Pointer(&v))
	var out [2]uint64
	for i := offset; i < 2; i++ {
		out[i] = arr[i-offset]
	}
	return *(*Uint64x2)(unsafe.Pointer(&out))
}

// ===== InsertLane operations =====
// InsertLane inserts a value at the specified lane index.

// InsertLaneFloat32x4 inserts a float32 value at the specified lane.
func InsertLaneFloat32x4(v Float32x4, lane int, val float32) Float32x4 {
	arr := (*[4]float32)(unsafe.Pointer(&v))
	arr[lane] = val
	return *(*Float32x4)(unsafe.Pointer(arr))
}

// InsertLaneFloat64x2 inserts a float64 value at the specified lane.
func InsertLaneFloat64x2(v Float64x2, lane int, val float64) Float64x2 {
	arr := (*[2]float64)(unsafe.Pointer(&v))
	arr[lane] = val
	return *(*Float64x2)(unsafe.Pointer(arr))
}

// InsertLaneInt32x4 inserts an int32 value at the specified lane.
func InsertLaneInt32x4(v Int32x4, lane int, val int32) Int32x4 {
	arr := (*[4]int32)(unsafe.Pointer(&v))
	arr[lane] = val
	return *(*Int32x4)(unsafe.Pointer(arr))
}

// InsertLaneInt64x2 inserts an int64 value at the specified lane.
func InsertLaneInt64x2(v Int64x2, lane int, val int64) Int64x2 {
	arr := (*[2]int64)(unsafe.Pointer(&v))
	arr[lane] = val
	return *(*Int64x2)(unsafe.Pointer(arr))
}

// InsertLaneUint32x4 inserts a uint32 value at the specified lane.
func InsertLaneUint32x4(v Uint32x4, lane int, val uint32) Uint32x4 {
	arr := (*[4]uint32)(unsafe.Pointer(&v))
	arr[lane] = val
	return *(*Uint32x4)(unsafe.Pointer(arr))
}

// InsertLaneUint64x2 inserts a uint64 value at the specified lane.
func InsertLaneUint64x2(v Uint64x2, lane int, val uint64) Uint64x2 {
	arr := (*[2]uint64)(unsafe.Pointer(&v))
	arr[lane] = val
	return *(*Uint64x2)(unsafe.Pointer(arr))
}
