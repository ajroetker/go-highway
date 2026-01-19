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

// This file provides pure Go (scalar) implementations of all Highway operations.
// When SIMD implementations are available (ops_simd_*.go), they will replace these
// implementations via build tags. The scalar implementations serve as the fallback
// and are also used when HWY_NO_SIMD is set.

// Load creates a vector by loading data from a slice.
func Load[T Lanes](src []T) Vec[T] {
	n := min(len(src), MaxLanes[T]())
	data := make([]T, n)
	copy(data, src[:n])
	return Vec[T]{data: data}
}

// Load4 loads 4 consecutive vectors from a slice for 4x loop unrolling.
// On ARM NEON, this maps to a single ld1 instruction with 4 registers,
// which is more efficient than 4 separate Load calls.
// On AVX2/AVX512, the code generator expands this to 4 separate loads
// since wider registers already provide efficient memory throughput.
func Load4[T Lanes](src []T) (Vec[T], Vec[T], Vec[T], Vec[T]) {
	lanes := MaxLanes[T]()
	v0 := Load(src)
	v1 := Load(src[lanes:])
	v2 := Load(src[lanes*2:])
	v3 := Load(src[lanes*3:])
	return v0, v1, v2, v3
}

// Load4_NEON_Vec is a wrapper for Float16/BFloat16 types on NEON.
// These types don't have native SIMD support, so we use the generic Load4.
func Load4_NEON_Vec[T Lanes](src []T) (Vec[T], Vec[T], Vec[T], Vec[T]) {
	return Load4(src)
}

// Load4_Fallback_Vec is a wrapper for Float16/BFloat16 types on Fallback.
// Uses the generic Load4 implementation.
func Load4_Fallback_Vec[T Lanes](src []T) (Vec[T], Vec[T], Vec[T], Vec[T]) {
	return Load4(src)
}

// Store writes a vector's data to a slice.
func Store[T Lanes](v Vec[T], dst []T) {
	n := min(len(dst), len(v.data))
	copy(dst[:n], v.data[:n])
}

// Set creates a vector with all lanes set to the same value.
func Set[T Lanes](value T) Vec[T] {
	n := MaxLanes[T]()
	data := make([]T, n)
	for i := range data {
		data[i] = value
	}
	return Vec[T]{data: data}
}

// Const creates a vector with all lanes set to the given float32 constant.
// This allows writing generic code without T(constant) conversions.
// Usage: hwy.Const[T](1.0) creates a Vec[T] with all lanes set to 1.0
func Const[T Lanes](val float32) Vec[T] {
	return Set(ConstValue[T](val))
}

// ConstValue converts a float32 constant to type T.
func ConstValue[T Lanes](val float32) T {
	var zero T
	switch any(zero).(type) {
	case Float16:
		return any(Float32ToFloat16(val)).(T)
	case BFloat16:
		return any(Float32ToBFloat16(val)).(T)
	}
	// Native types support direct conversion from float32
	return T(val)
}

// Zero creates a vector with all lanes set to zero.
func Zero[T Lanes]() Vec[T] {
	n := MaxLanes[T]()
	data := make([]T, n)
	return Vec[T]{data: data}
}

// Add performs element-wise addition.
func Add[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		result[i] = addHelper(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

func addHelper[T Lanes](a, b T) T {
	// Fast path: check for Float16/BFloat16 which need special handling
	if av, ok := any(a).(Float16); ok {
		bv := any(b).(Float16)
		return any(Float32ToFloat16(av.Float32() + bv.Float32())).(T)
	}
	if av, ok := any(a).(BFloat16); ok {
		bv := any(b).(BFloat16)
		return any(Float32ToBFloat16(av.Float32() + bv.Float32())).(T)
	}
	// For all other numeric types, use interface conversion which the compiler
	// can optimize better than boxing in most cases
	switch av := any(a).(type) {
	case float32:
		return any(av + any(b).(float32)).(T)
	case float64:
		return any(av + any(b).(float64)).(T)
	case int8:
		return any(av + any(b).(int8)).(T)
	case int16:
		return any(av + any(b).(int16)).(T)
	case int32:
		return any(av + any(b).(int32)).(T)
	case int64:
		return any(av + any(b).(int64)).(T)
	case uint8:
		return any(av + any(b).(uint8)).(T)
	case uint16:
		return any(av + any(b).(uint16)).(T)
	case uint32:
		return any(av + any(b).(uint32)).(T)
	case uint64:
		return any(av + any(b).(uint64)).(T)
	default:
		return a
	}
}

// Sub performs element-wise subtraction.
func Sub[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		result[i] = subHelper(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

func subHelper[T Lanes](a, b T) T {
	switch av := any(a).(type) {
	case Float16:
		bv := any(b).(Float16)
		return any(Float32ToFloat16(av.Float32() - bv.Float32())).(T)
	case BFloat16:
		bv := any(b).(BFloat16)
		return any(Float32ToBFloat16(av.Float32() - bv.Float32())).(T)
	case float32:
		return any(av - any(b).(float32)).(T)
	case float64:
		return any(av - any(b).(float64)).(T)
	case int8:
		return any(av - any(b).(int8)).(T)
	case int16:
		return any(av - any(b).(int16)).(T)
	case int32:
		return any(av - any(b).(int32)).(T)
	case int64:
		return any(av - any(b).(int64)).(T)
	case uint8:
		return any(av - any(b).(uint8)).(T)
	case uint16:
		return any(av - any(b).(uint16)).(T)
	case uint32:
		return any(av - any(b).(uint32)).(T)
	case uint64:
		return any(av - any(b).(uint64)).(T)
	default:
		return a
	}
}

// Mul performs element-wise multiplication.
func Mul[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		result[i] = mulHelper(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

func mulHelper[T Lanes](a, b T) T {
	switch av := any(a).(type) {
	case Float16:
		bv := any(b).(Float16)
		return any(Float32ToFloat16(av.Float32() * bv.Float32())).(T)
	case BFloat16:
		bv := any(b).(BFloat16)
		return any(Float32ToBFloat16(av.Float32() * bv.Float32())).(T)
	case float32:
		return any(av * any(b).(float32)).(T)
	case float64:
		return any(av * any(b).(float64)).(T)
	case int8:
		return any(av * any(b).(int8)).(T)
	case int16:
		return any(av * any(b).(int16)).(T)
	case int32:
		return any(av * any(b).(int32)).(T)
	case int64:
		return any(av * any(b).(int64)).(T)
	case uint8:
		return any(av * any(b).(uint8)).(T)
	case uint16:
		return any(av * any(b).(uint16)).(T)
	case uint32:
		return any(av * any(b).(uint32)).(T)
	case uint64:
		return any(av * any(b).(uint64)).(T)
	default:
		return a
	}
}

// Div performs element-wise division.
func Div[T Floats](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)

	// Check type once, then use optimized loop
	var zero T
	switch any(zero).(type) {
	case Float16:
		for i := range n {
			av := any(a.data[i]).(Float16)
			bv := any(b.data[i]).(Float16)
			result[i] = any(Float32ToFloat16(av.Float32() / bv.Float32())).(T)
		}
	case BFloat16:
		for i := range n {
			av := any(a.data[i]).(BFloat16)
			bv := any(b.data[i]).(BFloat16)
			result[i] = any(Float32ToBFloat16(av.Float32() / bv.Float32())).(T)
		}
	case float32:
		// Direct slice access for native floats - no boxing per element
		aData := any(a.data).([]float32)
		bData := any(b.data).([]float32)
		rData := any(result).([]float32)
		for i := range n {
			rData[i] = aData[i] / bData[i]
		}
	case float64:
		aData := any(a.data).([]float64)
		bData := any(b.data).([]float64)
		rData := any(result).([]float64)
		for i := range n {
			rData[i] = aData[i] / bData[i]
		}
	}
	return Vec[T]{data: result}
}

// Neg negates all lanes.
func Neg[T Lanes](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = negHelper(v.data[i])
	}
	return Vec[T]{data: result}
}

func negHelper[T Lanes](a T) T {
	switch av := any(a).(type) {
	case Float16:
		return any(Float32ToFloat16(-av.Float32())).(T)
	case BFloat16:
		return any(Float32ToBFloat16(-av.Float32())).(T)
	case float32:
		return any(-av).(T)
	case float64:
		return any(-av).(T)
	case int8:
		return any(-av).(T)
	case int16:
		return any(-av).(T)
	case int32:
		return any(-av).(T)
	case int64:
		return any(-av).(T)
	case uint8:
		return any(-av).(T)
	case uint16:
		return any(-av).(T)
	case uint32:
		return any(-av).(T)
	case uint64:
		return any(-av).(T)
	default:
		return a
	}
}

// Abs computes absolute value.
func Abs[T Lanes](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = absHelper(v.data[i])
	}
	return Vec[T]{data: result}
}

func absHelper[T Lanes](a T) T {
	switch av := any(a).(type) {
	case Float16:
		f := av.Float32()
		if f < 0 {
			f = -f
		}
		return any(Float32ToFloat16(f)).(T)
	case BFloat16:
		f := av.Float32()
		if f < 0 {
			f = -f
		}
		return any(Float32ToBFloat16(f)).(T)
	case float32:
		if av < 0 {
			return any(-av).(T)
		}
		return any(av).(T)
	case float64:
		if av < 0 {
			return any(-av).(T)
		}
		return any(av).(T)
	case int8:
		if av < 0 {
			return any(-av).(T)
		}
		return any(av).(T)
	case int16:
		if av < 0 {
			return any(-av).(T)
		}
		return any(av).(T)
	case int32:
		if av < 0 {
			return any(-av).(T)
		}
		return any(av).(T)
	case int64:
		if av < 0 {
			return any(-av).(T)
		}
		return any(av).(T)
	case uint8:
		return any(av).(T) // unsigned always positive
	case uint16:
		return any(av).(T)
	case uint32:
		return any(av).(T)
	case uint64:
		return any(av).(T)
	default:
		return a
	}
}

// Min returns element-wise minimum.
func Min[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		if lessHelper(a.data[i], b.data[i]) {
			result[i] = a.data[i]
		} else {
			result[i] = b.data[i]
		}
	}
	return Vec[T]{data: result}
}

func lessHelper[T Lanes](a, b T) bool {
	switch av := any(a).(type) {
	case Float16:
		return av.Float32() < any(b).(Float16).Float32()
	case BFloat16:
		return av.Float32() < any(b).(BFloat16).Float32()
	case float32:
		return av < any(b).(float32)
	case float64:
		return av < any(b).(float64)
	case int8:
		return av < any(b).(int8)
	case int16:
		return av < any(b).(int16)
	case int32:
		return av < any(b).(int32)
	case int64:
		return av < any(b).(int64)
	case uint8:
		return av < any(b).(uint8)
	case uint16:
		return av < any(b).(uint16)
	case uint32:
		return av < any(b).(uint32)
	case uint64:
		return av < any(b).(uint64)
	default:
		return false
	}
}

// Max returns element-wise maximum.
func Max[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		if greaterHelper(a.data[i], b.data[i]) {
			result[i] = a.data[i]
		} else {
			result[i] = b.data[i]
		}
	}
	return Vec[T]{data: result}
}

func greaterHelper[T Lanes](a, b T) bool {
	switch av := any(a).(type) {
	case Float16:
		return av.Float32() > any(b).(Float16).Float32()
	case BFloat16:
		return av.Float32() > any(b).(BFloat16).Float32()
	case float32:
		return av > any(b).(float32)
	case float64:
		return av > any(b).(float64)
	case int8:
		return av > any(b).(int8)
	case int16:
		return av > any(b).(int16)
	case int32:
		return av > any(b).(int32)
	case int64:
		return av > any(b).(int64)
	case uint8:
		return av > any(b).(uint8)
	case uint16:
		return av > any(b).(uint16)
	case uint32:
		return av > any(b).(uint32)
	case uint64:
		return av > any(b).(uint64)
	default:
		return false
	}
}

// Sqrt computes square root.
func Sqrt[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		switch x := any(v.data[i]).(type) {
		case Float16:
			result[i] = any(Float32ToFloat16(float32(math.Sqrt(float64(x.Float32()))))).(T)
		case BFloat16:
			result[i] = any(Float32ToBFloat16(float32(math.Sqrt(float64(x.Float32()))))).(T)
		case float32:
			result[i] = any(float32(math.Sqrt(float64(x)))).(T)
		case float64:
			result[i] = any(math.Sqrt(x)).(T)
		}
	}
	return Vec[T]{data: result}
}

// RSqrt computes reciprocal square root (1/sqrt(x)).
// This is a scalar fallback implementation.
func RSqrt[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		switch x := any(v.data[i]).(type) {
		case Float16:
			result[i] = any(Float32ToFloat16(float32(1.0 / math.Sqrt(float64(x.Float32()))))).(T)
		case BFloat16:
			result[i] = any(Float32ToBFloat16(float32(1.0 / math.Sqrt(float64(x.Float32()))))).(T)
		case float32:
			result[i] = any(float32(1.0 / math.Sqrt(float64(x)))).(T)
		case float64:
			result[i] = any(1.0 / math.Sqrt(x)).(T)
		}
	}
	return Vec[T]{data: result}
}

// RSqrtNewtonRaphson computes 1/sqrt(x) with one Newton-Raphson refinement.
// For the fallback, this is identical to RSqrt since we use precise math.Sqrt.
func RSqrtNewtonRaphson[T Floats](v Vec[T]) Vec[T] {
	return RSqrt(v)
}

// RSqrtPrecise computes precise 1/sqrt(x).
// For the fallback, this is identical to RSqrt since we use precise math.Sqrt.
func RSqrtPrecise[T Floats](v Vec[T]) Vec[T] {
	return RSqrt(v)
}

// Pow computes base^exp element-wise.
func Pow[T Floats](base, exp Vec[T]) Vec[T] {
	n := min(len(exp.data), len(base.data))
	result := make([]T, n)
	for i := range n {
		switch b := any(base.data[i]).(type) {
		case Float16:
			e := any(exp.data[i]).(Float16)
			result[i] = any(Float32ToFloat16(float32(math.Pow(float64(b.Float32()), float64(e.Float32()))))).(T)
		case BFloat16:
			e := any(exp.data[i]).(BFloat16)
			result[i] = any(Float32ToBFloat16(float32(math.Pow(float64(b.Float32()), float64(e.Float32()))))).(T)
		case float32:
			e := any(exp.data[i]).(float32)
			result[i] = any(float32(math.Pow(float64(b), float64(e)))).(T)
		case float64:
			e := any(exp.data[i]).(float64)
			result[i] = any(math.Pow(b, e)).(T)
		}
	}
	return Vec[T]{data: result}
}

// FMA performs fused multiply-add.
func FMA[T Floats](a, b, c Vec[T]) Vec[T] {
	n := min(len(c.data), min(len(b.data), len(a.data)))
	result := make([]T, n)
	for i := range n {
		switch av := any(a.data[i]).(type) {
		case Float16:
			bv := any(b.data[i]).(Float16)
			cv := any(c.data[i]).(Float16)
			result[i] = any(Float32ToFloat16(float32(math.FMA(float64(av.Float32()), float64(bv.Float32()), float64(cv.Float32()))))).(T)
		case BFloat16:
			bv := any(b.data[i]).(BFloat16)
			cv := any(c.data[i]).(BFloat16)
			result[i] = any(Float32ToBFloat16(float32(math.FMA(float64(av.Float32()), float64(bv.Float32()), float64(cv.Float32()))))).(T)
		case float32:
			bv := any(b.data[i]).(float32)
			cv := any(c.data[i]).(float32)
			result[i] = any(float32(math.FMA(float64(av), float64(bv), float64(cv)))).(T)
		case float64:
			bv := any(b.data[i]).(float64)
			cv := any(c.data[i]).(float64)
			result[i] = any(math.FMA(av, bv, cv)).(T)
		}
	}
	return Vec[T]{data: result}
}

// ReduceSum sums all lanes.
func ReduceSum[T Lanes](v Vec[T]) T {
	var sum T
	for i := 0; i < len(v.data); i++ {
		sum += v.data[i]
	}
	return sum
}

// ReduceMin returns the minimum value across all lanes.
func ReduceMin[T Lanes](v Vec[T]) T {
	if len(v.data) == 0 {
		var zero T
		return zero
	}
	min := v.data[0]
	for i := 1; i < len(v.data); i++ {
		if v.data[i] < min {
			min = v.data[i]
		}
	}
	return min
}

// ReduceMax returns the maximum value across all lanes.
func ReduceMax[T Lanes](v Vec[T]) T {
	if len(v.data) == 0 {
		var zero T
		return zero
	}
	max := v.data[0]
	for i := 1; i < len(v.data); i++ {
		if v.data[i] > max {
			max = v.data[i]
		}
	}
	return max
}

// Equal performs element-wise equality comparison.
func Equal[T Lanes](a, b Vec[T]) Mask[T] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		bits[i] = a.data[i] == b.data[i]
	}
	return Mask[T]{bits: bits}
}

// NotEqual performs element-wise inequality comparison.
func NotEqual[T Lanes](a, b Vec[T]) Mask[T] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		bits[i] = a.data[i] != b.data[i]
	}
	return Mask[T]{bits: bits}
}

// LessThan performs element-wise less-than comparison.
func LessThan[T Lanes](a, b Vec[T]) Mask[T] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		bits[i] = a.data[i] < b.data[i]
	}
	return Mask[T]{bits: bits}
}

// GreaterThan performs element-wise greater-than comparison.
func GreaterThan[T Lanes](a, b Vec[T]) Mask[T] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		bits[i] = a.data[i] > b.data[i]
	}
	return Mask[T]{bits: bits}
}

// LessEqual performs element-wise less-than-or-equal comparison.
func LessEqual[T Lanes](a, b Vec[T]) Mask[T] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		bits[i] = a.data[i] <= b.data[i]
	}
	return Mask[T]{bits: bits}
}

// GreaterEqual performs element-wise greater-than-or-equal comparison.
func GreaterEqual[T Lanes](a, b Vec[T]) Mask[T] {
	n := min(len(b.data), len(a.data))
	bits := make([]bool, n)
	for i := range n {
		bits[i] = a.data[i] >= b.data[i]
	}
	return Mask[T]{bits: bits}
}

// IsNaN returns a mask indicating which lanes contain NaN values.
// For integer types, this always returns all false.
func IsNaN[T Floats](v Vec[T]) Mask[T] {
	bits := make([]bool, len(v.data))
	for i, val := range v.data {
		switch x := any(val).(type) {
		case Float16:
			bits[i] = math.IsNaN(float64(x.Float32()))
		case BFloat16:
			bits[i] = math.IsNaN(float64(x.Float32()))
		case float32:
			bits[i] = math.IsNaN(float64(x))
		case float64:
			bits[i] = math.IsNaN(x)
		}
	}
	return Mask[T]{bits: bits}
}

// IsInf returns a mask indicating which lanes contain infinity.
// The sign parameter: 0 = either, > 0 = +Inf only, < 0 = -Inf only.
func IsInf[T Floats](v Vec[T], sign int) Mask[T] {
	bits := make([]bool, len(v.data))
	for i, val := range v.data {
		switch x := any(val).(type) {
		case Float16:
			bits[i] = math.IsInf(float64(x.Float32()), sign)
		case BFloat16:
			bits[i] = math.IsInf(float64(x.Float32()), sign)
		case float32:
			bits[i] = math.IsInf(float64(x), sign)
		case float64:
			bits[i] = math.IsInf(x, sign)
		}
	}
	return Mask[T]{bits: bits}
}

// IsFinite returns a mask indicating which lanes contain finite values.
// A value is finite if it is neither NaN nor infinity.
func IsFinite[T Floats](v Vec[T]) Mask[T] {
	bits := make([]bool, len(v.data))
	for i, val := range v.data {
		var f float64
		switch x := any(val).(type) {
		case Float16:
			f = float64(x.Float32())
		case BFloat16:
			f = float64(x.Float32())
		case float32:
			f = float64(x)
		case float64:
			f = x
		}
		bits[i] = !math.IsNaN(f) && !math.IsInf(f, 0)
	}
	return Mask[T]{bits: bits}
}

// TestBit returns a mask indicating which lanes have the specified bit set.
// Bit 0 is the least significant bit.
func TestBit[T Integers](v Vec[T], bit int) Mask[T] {
	bits := make([]bool, len(v.data))
	for i, val := range v.data {
		bits[i] = testBitHelper(val, bit)
	}
	return Mask[T]{bits: bits}
}

func testBitHelper[T Integers](val T, bit int) bool {
	switch any(val).(type) {
	case int8:
		return (any(val).(int8) & (1 << bit)) != 0
	case int16:
		return (any(val).(int16) & (1 << bit)) != 0
	case int32:
		return (any(val).(int32) & (1 << bit)) != 0
	case int64:
		return (any(val).(int64) & (1 << bit)) != 0
	case uint8:
		return (any(val).(uint8) & (1 << bit)) != 0
	case uint16:
		return (any(val).(uint16) & (1 << bit)) != 0
	case uint32:
		return (any(val).(uint32) & (1 << bit)) != 0
	case uint64:
		return (any(val).(uint64) & (1 << bit)) != 0
	default:
		return false
	}
}

// IfThenElse performs conditional selection.
func IfThenElse[T Lanes](mask Mask[T], a, b Vec[T]) Vec[T] {
	n := min(len(b.data), min(len(a.data), len(mask.bits)))
	result := make([]T, n)
	for i := range n {
		if mask.bits[i] {
			result[i] = a.data[i]
		} else {
			result[i] = b.data[i]
		}
	}
	return Vec[T]{data: result}
}

// IfThenElseZero returns a where mask is true, zero otherwise.
// Equivalent to IfThenElse(mask, a, Zero()) but more efficient.
func IfThenElseZero[T Lanes](mask Mask[T], a Vec[T]) Vec[T] {
	n := min(len(a.data), len(mask.bits))
	result := make([]T, n)
	for i := range n {
		if mask.bits[i] {
			result[i] = a.data[i]
		}
		// else: leave as zero value
	}
	return Vec[T]{data: result}
}

// IfThenZeroElse returns zero where mask is true, b otherwise.
// Equivalent to IfThenElse(mask, Zero(), b) but more efficient.
func IfThenZeroElse[T Lanes](mask Mask[T], b Vec[T]) Vec[T] {
	n := min(len(b.data), len(mask.bits))
	result := make([]T, n)
	for i := range n {
		if !mask.bits[i] {
			result[i] = b.data[i]
		}
		// else: leave as zero value
	}
	return Vec[T]{data: result}
}

// ZeroIfNegative returns zero for negative lanes, original value otherwise.
// Useful for clamping negative values to zero.
func ZeroIfNegative[T Lanes](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i, val := range v.data {
		if val >= 0 {
			result[i] = val
		}
		// else: leave as zero value
	}
	return Vec[T]{data: result}
}

// MaskLoad loads data from a slice only for lanes where the mask is true.
func MaskLoad[T Lanes](mask Mask[T], src []T) Vec[T] {
	n := min(len(src), len(mask.bits))
	result := make([]T, len(mask.bits))
	for i := range n {
		if mask.bits[i] {
			result[i] = src[i]
		}
		// else: leave as zero value
	}
	return Vec[T]{data: result}
}

// MaskStore stores vector data to a slice only for lanes where the mask is true.
func MaskStore[T Lanes](mask Mask[T], v Vec[T], dst []T) {
	n := min(len(dst), min(len(v.data), len(mask.bits)))
	for i := range n {
		if mask.bits[i] {
			dst[i] = v.data[i]
		}
	}
}

// And performs element-wise bitwise AND.
func And[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		// Perform bitwise AND by reinterpreting as integers
		result[i] = bitwiseAnd(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

// Or performs element-wise bitwise OR.
func Or[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		result[i] = bitwiseOr(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

// Xor performs element-wise bitwise XOR.
func Xor[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		result[i] = bitwiseXor(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

// Not performs element-wise bitwise NOT (ones complement).
func Not[T Lanes](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = bitwiseNot(v.data[i])
	}
	return Vec[T]{data: result}
}

// AndNot performs element-wise bitwise AND NOT (~a & b).
func AndNot[T Lanes](a, b Vec[T]) Vec[T] {
	n := min(len(b.data), len(a.data))
	result := make([]T, n)
	for i := range n {
		result[i] = bitwiseAndNot(a.data[i], b.data[i])
	}
	return Vec[T]{data: result}
}

// ShiftLeft performs element-wise left shift by a constant number of bits.
// Only valid for integer types.
func ShiftLeft[T Integers](v Vec[T], bits int) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = shiftLeft(v.data[i], bits)
	}
	return Vec[T]{data: result}
}

// ShiftRight performs element-wise right shift by a constant number of bits.
// For signed integers, this is arithmetic shift (sign-extended).
// For unsigned integers, this is logical shift (zero-filled).
func ShiftRight[T Integers](v Vec[T], bits int) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = shiftRight(v.data[i], bits)
	}
	return Vec[T]{data: result}
}

// Iota returns a vector with lanes set to [0, 1, 2, 3, ...].
func Iota[T Lanes]() Vec[T] {
	n := MaxLanes[T]()
	data := make([]T, n)
	for i := range data {
		data[i] = T(i)
	}
	return Vec[T]{data: data}
}

// SignBit returns a vector with the sign bit set in each lane.
// For floats, this is -0.0. For signed integers, this is the minimum value.
// For unsigned integers, this is the high bit set.
func SignBit[T Lanes]() Vec[T] {
	n := MaxLanes[T]()
	data := make([]T, n)
	signBit := getSignBit[T]()
	for i := range data {
		data[i] = signBit
	}
	return Vec[T]{data: data}
}

// Helper functions for bitwise operations that work with any numeric type

func bitwiseAnd[T Lanes](a, b T) T {
	// Use type switch to handle different types
	switch any(a).(type) {
	case float32:
		aU := math.Float32bits(float32(any(a).(float32)))
		bU := math.Float32bits(float32(any(b).(float32)))
		return T(any(math.Float32frombits(aU & bU)).(float32))
	case float64:
		aU := math.Float64bits(float64(any(a).(float64)))
		bU := math.Float64bits(float64(any(b).(float64)))
		return T(any(math.Float64frombits(aU & bU)).(float64))
	case int8:
		return T(any(int8(any(a).(int8)) & int8(any(b).(int8))).(int8))
	case int16:
		return T(any(int16(any(a).(int16)) & int16(any(b).(int16))).(int16))
	case int32:
		return T(any(int32(any(a).(int32)) & int32(any(b).(int32))).(int32))
	case int64:
		return T(any(int64(any(a).(int64)) & int64(any(b).(int64))).(int64))
	case uint8:
		return T(any(uint8(any(a).(uint8)) & uint8(any(b).(uint8))).(uint8))
	case uint16:
		return T(any(uint16(any(a).(uint16)) & uint16(any(b).(uint16))).(uint16))
	case uint32:
		return T(any(uint32(any(a).(uint32)) & uint32(any(b).(uint32))).(uint32))
	case uint64:
		return T(any(uint64(any(a).(uint64)) & uint64(any(b).(uint64))).(uint64))
	default:
		return a // Should never happen
	}
}

func bitwiseOr[T Lanes](a, b T) T {
	switch any(a).(type) {
	case float32:
		aU := math.Float32bits(float32(any(a).(float32)))
		bU := math.Float32bits(float32(any(b).(float32)))
		return T(any(math.Float32frombits(aU | bU)).(float32))
	case float64:
		aU := math.Float64bits(float64(any(a).(float64)))
		bU := math.Float64bits(float64(any(b).(float64)))
		return T(any(math.Float64frombits(aU | bU)).(float64))
	case int8:
		return T(any(int8(any(a).(int8)) | int8(any(b).(int8))).(int8))
	case int16:
		return T(any(int16(any(a).(int16)) | int16(any(b).(int16))).(int16))
	case int32:
		return T(any(int32(any(a).(int32)) | int32(any(b).(int32))).(int32))
	case int64:
		return T(any(int64(any(a).(int64)) | int64(any(b).(int64))).(int64))
	case uint8:
		return T(any(uint8(any(a).(uint8)) | uint8(any(b).(uint8))).(uint8))
	case uint16:
		return T(any(uint16(any(a).(uint16)) | uint16(any(b).(uint16))).(uint16))
	case uint32:
		return T(any(uint32(any(a).(uint32)) | uint32(any(b).(uint32))).(uint32))
	case uint64:
		return T(any(uint64(any(a).(uint64)) | uint64(any(b).(uint64))).(uint64))
	default:
		return a
	}
}

func bitwiseXor[T Lanes](a, b T) T {
	switch any(a).(type) {
	case float32:
		aU := math.Float32bits(float32(any(a).(float32)))
		bU := math.Float32bits(float32(any(b).(float32)))
		return T(any(math.Float32frombits(aU ^ bU)).(float32))
	case float64:
		aU := math.Float64bits(float64(any(a).(float64)))
		bU := math.Float64bits(float64(any(b).(float64)))
		return T(any(math.Float64frombits(aU ^ bU)).(float64))
	case int8:
		return T(any(int8(any(a).(int8)) ^ int8(any(b).(int8))).(int8))
	case int16:
		return T(any(int16(any(a).(int16)) ^ int16(any(b).(int16))).(int16))
	case int32:
		return T(any(int32(any(a).(int32)) ^ int32(any(b).(int32))).(int32))
	case int64:
		return T(any(int64(any(a).(int64)) ^ int64(any(b).(int64))).(int64))
	case uint8:
		return T(any(uint8(any(a).(uint8)) ^ uint8(any(b).(uint8))).(uint8))
	case uint16:
		return T(any(uint16(any(a).(uint16)) ^ uint16(any(b).(uint16))).(uint16))
	case uint32:
		return T(any(uint32(any(a).(uint32)) ^ uint32(any(b).(uint32))).(uint32))
	case uint64:
		return T(any(uint64(any(a).(uint64)) ^ uint64(any(b).(uint64))).(uint64))
	default:
		return a
	}
}

func bitwiseNot[T Lanes](a T) T {
	switch any(a).(type) {
	case float32:
		aU := math.Float32bits(float32(any(a).(float32)))
		return T(any(math.Float32frombits(^aU)).(float32))
	case float64:
		aU := math.Float64bits(float64(any(a).(float64)))
		return T(any(math.Float64frombits(^aU)).(float64))
	case int8:
		return T(any(^int8(any(a).(int8))).(int8))
	case int16:
		return T(any(^int16(any(a).(int16))).(int16))
	case int32:
		return T(any(^int32(any(a).(int32))).(int32))
	case int64:
		return T(any(^int64(any(a).(int64))).(int64))
	case uint8:
		return T(any(^uint8(any(a).(uint8))).(uint8))
	case uint16:
		return T(any(^uint16(any(a).(uint16))).(uint16))
	case uint32:
		return T(any(^uint32(any(a).(uint32))).(uint32))
	case uint64:
		return T(any(^uint64(any(a).(uint64))).(uint64))
	default:
		return a
	}
}

func bitwiseAndNot[T Lanes](a, b T) T {
	switch any(a).(type) {
	case float32:
		aU := math.Float32bits(float32(any(a).(float32)))
		bU := math.Float32bits(float32(any(b).(float32)))
		return T(any(math.Float32frombits((^aU) & bU)).(float32))
	case float64:
		aU := math.Float64bits(float64(any(a).(float64)))
		bU := math.Float64bits(float64(any(b).(float64)))
		return T(any(math.Float64frombits((^aU) & bU)).(float64))
	case int8:
		return T(any((^int8(any(a).(int8))) & int8(any(b).(int8))).(int8))
	case int16:
		return T(any((^int16(any(a).(int16))) & int16(any(b).(int16))).(int16))
	case int32:
		return T(any((^int32(any(a).(int32))) & int32(any(b).(int32))).(int32))
	case int64:
		return T(any((^int64(any(a).(int64))) & int64(any(b).(int64))).(int64))
	case uint8:
		return T(any((^uint8(any(a).(uint8))) & uint8(any(b).(uint8))).(uint8))
	case uint16:
		return T(any((^uint16(any(a).(uint16))) & uint16(any(b).(uint16))).(uint16))
	case uint32:
		return T(any((^uint32(any(a).(uint32))) & uint32(any(b).(uint32))).(uint32))
	case uint64:
		return T(any((^uint64(any(a).(uint64))) & uint64(any(b).(uint64))).(uint64))
	default:
		return a
	}
}

func shiftLeft[T Integers](a T, bits int) T {
	switch any(a).(type) {
	case int8:
		return T(any(int8(any(a).(int8)) << bits).(int8))
	case int16:
		return T(any(int16(any(a).(int16)) << bits).(int16))
	case int32:
		return T(any(int32(any(a).(int32)) << bits).(int32))
	case int64:
		return T(any(int64(any(a).(int64)) << bits).(int64))
	case uint8:
		return T(any(uint8(any(a).(uint8)) << bits).(uint8))
	case uint16:
		return T(any(uint16(any(a).(uint16)) << bits).(uint16))
	case uint32:
		return T(any(uint32(any(a).(uint32)) << bits).(uint32))
	case uint64:
		return T(any(uint64(any(a).(uint64)) << bits).(uint64))
	default:
		return a
	}
}

func shiftRight[T Integers](a T, bits int) T {
	// Right shift is arithmetic for signed, logical for unsigned
	switch any(a).(type) {
	case int8:
		return T(any(int8(any(a).(int8)) >> bits).(int8))
	case int16:
		return T(any(int16(any(a).(int16)) >> bits).(int16))
	case int32:
		return T(any(int32(any(a).(int32)) >> bits).(int32))
	case int64:
		return T(any(int64(any(a).(int64)) >> bits).(int64))
	case uint8:
		return T(any(uint8(any(a).(uint8)) >> bits).(uint8))
	case uint16:
		return T(any(uint16(any(a).(uint16)) >> bits).(uint16))
	case uint32:
		return T(any(uint32(any(a).(uint32)) >> bits).(uint32))
	case uint64:
		return T(any(uint64(any(a).(uint64)) >> bits).(uint64))
	default:
		return a
	}
}

func getSignBit[T Lanes]() T {
	switch any(T(0)).(type) {
	case float32:
		// Sign bit set: -0.0
		return T(any(math.Float32frombits(0x80000000)).(float32))
	case float64:
		// Sign bit set: -0.0
		return T(any(math.Float64frombits(0x8000000000000000)).(float64))
	case int8:
		// Minimum value has sign bit set
		return T(any(int8(-128)).(int8))
	case int16:
		return T(any(int16(-32768)).(int16))
	case int32:
		return T(any(int32(-2147483648)).(int32))
	case int64:
		return T(any(int64(-9223372036854775808)).(int64))
	case uint8:
		// High bit set
		return T(any(uint8(0x80)).(uint8))
	case uint16:
		return T(any(uint16(0x8000)).(uint16))
	case uint32:
		return T(any(uint32(0x80000000)).(uint32))
	case uint64:
		return T(any(uint64(0x8000000000000000)).(uint64))
	default:
		return T(0)
	}
}

// ============================================================================
// Math support operations (for contrib/math code generation)
// ============================================================================

// MulAdd performs fused multiply-add: a*b + c.
// This is an alias for FMA with the common a.MulAdd(b, c) semantics.
func MulAdd[T Floats](a, b, c Vec[T]) Vec[T] {
	return FMA(a, b, c)
}

// RoundToEven rounds to the nearest even integer (banker's rounding).
// This is the default IEEE 754 rounding mode.
func RoundToEven[T Floats](v Vec[T]) Vec[T] {
	n := len(v.data)
	result := make([]T, n)

	// Check type once, then use optimized loop
	var zero T
	switch any(zero).(type) {
	case Float16:
		vData := any(v.data).([]Float16)
		rData := any(result).([]Float16)
		for i := range n {
			rData[i] = Float32ToFloat16(float32(math.RoundToEven(float64(vData[i].Float32()))))
		}
	case BFloat16:
		vData := any(v.data).([]BFloat16)
		rData := any(result).([]BFloat16)
		for i := range n {
			rData[i] = Float32ToBFloat16(float32(math.RoundToEven(float64(vData[i].Float32()))))
		}
	case float32:
		vData := any(v.data).([]float32)
		rData := any(result).([]float32)
		for i := range n {
			rData[i] = float32(math.RoundToEven(float64(vData[i])))
		}
	case float64:
		vData := any(v.data).([]float64)
		rData := any(result).([]float64)
		for i := range n {
			rData[i] = math.RoundToEven(vData[i])
		}
	}
	return Vec[T]{data: result}
}

// Greater performs element-wise greater-than comparison.
// Alias for GreaterThan for compatibility with SIMD method naming.
func Greater[T Lanes](a, b Vec[T]) Mask[T] {
	return GreaterThan(a, b)
}

// Less performs element-wise less-than comparison.
// Alias for LessThan for compatibility with SIMD method naming.
func Less[T Lanes](a, b Vec[T]) Mask[T] {
	return LessThan(a, b)
}

// Merge selects elements from a where mask is true, from b otherwise.
// This is equivalent to IfThenElse(mask, a, b).
func Merge[T Lanes](a, b Vec[T], mask Mask[T]) Vec[T] {
	return IfThenElse(mask, a, b)
}

// ============================================================================
// Type reinterpretation operations (bit cast, no value conversion)
// ============================================================================

// AsInt32 reinterprets a float32 vector as int32 (bit cast).
func AsInt32(v Vec[float32]) Vec[int32] {
	result := make([]int32, len(v.data))
	for i, x := range v.data {
		result[i] = int32(math.Float32bits(x))
	}
	return Vec[int32]{data: result}
}

// AsFloat32 reinterprets an int32 vector as float32 (bit cast).
func AsFloat32(v Vec[int32]) Vec[float32] {
	result := make([]float32, len(v.data))
	for i, x := range v.data {
		result[i] = math.Float32frombits(uint32(x))
	}
	return Vec[float32]{data: result}
}

// AsInt64 reinterprets a float64 vector as int64 (bit cast).
func AsInt64(v Vec[float64]) Vec[int64] {
	result := make([]int64, len(v.data))
	for i, x := range v.data {
		result[i] = int64(math.Float64bits(x))
	}
	return Vec[int64]{data: result}
}

// AsFloat64 reinterprets an int64 vector as float64 (bit cast).
func AsFloat64(v Vec[int64]) Vec[float64] {
	result := make([]float64, len(v.data))
	for i, x := range v.data {
		result[i] = math.Float64frombits(uint64(x))
	}
	return Vec[float64]{data: result}
}
