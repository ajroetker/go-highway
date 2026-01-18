package hwy

import "math"

// This file provides pure Go (scalar) implementations of type conversion operations.
// When SIMD implementations are available (convert_avx2.go, convert_avx512.go),
// they can be used for higher performance on supported hardware.

// ConvertToInt32 converts float types to int32 (truncate toward zero).
// For values outside the int32 range, the result is undefined.
func ConvertToInt32[T Floats](v Vec[T]) Vec[int32] {
	result := make([]int32, len(v.data))
	var zero T
	switch any(zero).(type) {
	case Float16:
		for i := 0; i < len(v.data); i++ {
			result[i] = int32(any(v.data[i]).(Float16).Float32())
		}
	case BFloat16:
		for i := 0; i < len(v.data); i++ {
			result[i] = int32(any(v.data[i]).(BFloat16).Float32())
		}
	case float32:
		data := any(v.data).([]float32)
		for i := range data {
			result[i] = int32(data[i])
		}
	case float64:
		data := any(v.data).([]float64)
		for i := range data {
			result[i] = int32(data[i])
		}
	}
	return Vec[int32]{data: result}
}

// ConvertToFloat32 converts int32 or int64 to float32.
// Large int64 values may lose precision.
func ConvertToFloat32[T ~int32 | ~int64](v Vec[T]) Vec[float32] {
	result := make([]float32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = float32(v.data[i])
	}
	return Vec[float32]{data: result}
}

// ConvertToF16 converts int32 to Float16.
// Values are converted through float32 first.
func ConvertToF16(v Vec[int32]) Vec[Float16] {
	result := make([]Float16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = Float32ToFloat16(float32(v.data[i]))
	}
	return Vec[Float16]{data: result}
}

// ConvertToBF16 converts int32 to BFloat16.
// Values are converted through float32 first.
func ConvertToBF16(v Vec[int32]) Vec[BFloat16] {
	result := make([]BFloat16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = Float32ToBFloat16(float32(v.data[i]))
	}
	return Vec[BFloat16]{data: result}
}

// ConvertExponentToFloat converts integer exponents to the target float type.
// This is a generic function that gets specialized by hwygen for each float type.
// For Float16/BFloat16, values go through float32 conversion.
// For float32/float64, direct conversion is used.
func ConvertExponentToFloat[T Floats](e Vec[int32]) Vec[T] {
	// This is the fallback/scalar implementation.
	// hwygen will specialize this for each target type.
	result := make([]T, len(e.data))
	for i := 0; i < len(e.data); i++ {
		// Use float64 as intermediate to avoid type assertion issues
		f := float64(e.data[i])
		switch r := (any)(result).(type) {
		case []Float16:
			r[i] = Float32ToFloat16(float32(f))
		case []BFloat16:
			r[i] = Float32ToBFloat16(float32(f))
		case []float32:
			r[i] = float32(f)
		case []float64:
			r[i] = f
		}
	}
	return Vec[T]{data: result}
}

// ConvertToInt64 converts float64 to int64 (truncate toward zero).
// For values outside the int64 range, the result is undefined.
func ConvertToInt64[T ~float64](v Vec[T]) Vec[int64] {
	result := make([]int64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int64(v.data[i])
	}
	return Vec[int64]{data: result}
}

// ConvertToFloat64 converts int32 or int64 to float64.
// Large int64 values may lose precision.
func ConvertToFloat64[T ~int32 | ~int64](v Vec[T]) Vec[float64] {
	result := make([]float64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = float64(v.data[i])
	}
	return Vec[float64]{data: result}
}

// Round rounds each lane to the nearest integer (using banker's rounding).
func Round[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		var val T
		switch x := any(v.data[i]).(type) {
		case Float16:
			val = any(Float32ToFloat16(float32(math.Round(float64(x.Float32()))))).(T)
		case BFloat16:
			val = any(Float32ToBFloat16(float32(math.Round(float64(x.Float32()))))).(T)
		case float32:
			val = any(float32(math.Round(float64(x)))).(T)
		case float64:
			val = any(math.Round(x)).(T)
		}
		result[i] = val
	}
	return Vec[T]{data: result}
}

// Trunc truncates each lane toward zero.
func Trunc[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		var val T
		switch x := any(v.data[i]).(type) {
		case Float16:
			val = any(Float32ToFloat16(float32(math.Trunc(float64(x.Float32()))))).(T)
		case BFloat16:
			val = any(Float32ToBFloat16(float32(math.Trunc(float64(x.Float32()))))).(T)
		case float32:
			val = any(float32(math.Trunc(float64(x)))).(T)
		case float64:
			val = any(math.Trunc(x)).(T)
		}
		result[i] = val
	}
	return Vec[T]{data: result}
}

// Ceil rounds each lane up (toward positive infinity).
func Ceil[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		var val T
		switch x := any(v.data[i]).(type) {
		case Float16:
			val = any(Float32ToFloat16(float32(math.Ceil(float64(x.Float32()))))).(T)
		case BFloat16:
			val = any(Float32ToBFloat16(float32(math.Ceil(float64(x.Float32()))))).(T)
		case float32:
			val = any(float32(math.Ceil(float64(x)))).(T)
		case float64:
			val = any(math.Ceil(x)).(T)
		}
		result[i] = val
	}
	return Vec[T]{data: result}
}

// Floor rounds each lane down (toward negative infinity).
func Floor[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		var val T
		switch x := any(v.data[i]).(type) {
		case Float16:
			val = any(Float32ToFloat16(float32(math.Floor(float64(x.Float32()))))).(T)
		case BFloat16:
			val = any(Float32ToBFloat16(float32(math.Floor(float64(x.Float32()))))).(T)
		case float32:
			val = any(float32(math.Floor(float64(x)))).(T)
		case float64:
			val = any(math.Floor(x)).(T)
		}
		result[i] = val
	}
	return Vec[T]{data: result}
}

// NearestInt rounds each lane to the nearest integer and returns the same type.
// This is similar to Round but ensures the result is an integral value
// stored in the floating-point type.
func NearestInt[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		var val T
		switch x := any(v.data[i]).(type) {
		case Float16:
			val = any(Float32ToFloat16(float32(math.RoundToEven(float64(x.Float32()))))).(T)
		case BFloat16:
			val = any(Float32ToBFloat16(float32(math.RoundToEven(float64(x.Float32()))))).(T)
		case float32:
			val = any(float32(math.RoundToEven(float64(x)))).(T)
		case float64:
			val = any(math.RoundToEven(x)).(T)
		}
		result[i] = val
	}
	return Vec[T]{data: result}
}

// BitCastF32ToI32 reinterprets float32 bits as int32 without conversion.
// This is a bitwise reinterpretation, not a numeric conversion.
func BitCastF32ToI32(v Vec[float32]) Vec[int32] {
	result := make([]int32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int32(math.Float32bits(v.data[i]))
	}
	return Vec[int32]{data: result}
}

// BitCastI32ToF32 reinterprets int32 bits as float32 without conversion.
// This is a bitwise reinterpretation, not a numeric conversion.
func BitCastI32ToF32(v Vec[int32]) Vec[float32] {
	result := make([]float32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float32frombits(uint32(v.data[i]))
	}
	return Vec[float32]{data: result}
}

// BitCastF64ToI64 reinterprets float64 bits as int64 without conversion.
// This is a bitwise reinterpretation, not a numeric conversion.
func BitCastF64ToI64(v Vec[float64]) Vec[int64] {
	result := make([]int64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int64(math.Float64bits(v.data[i]))
	}
	return Vec[int64]{data: result}
}

// BitCastI64ToF64 reinterprets int64 bits as float64 without conversion.
// This is a bitwise reinterpretation, not a numeric conversion.
func BitCastI64ToF64(v Vec[int64]) Vec[float64] {
	result := make([]float64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float64frombits(uint64(v.data[i]))
	}
	return Vec[float64]{data: result}
}

// BitCastU32ToF32 reinterprets uint32 bits as float32 without conversion.
func BitCastU32ToF32(v Vec[uint32]) Vec[float32] {
	result := make([]float32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float32frombits(v.data[i])
	}
	return Vec[float32]{data: result}
}

// BitCastF32ToU32 reinterprets float32 bits as uint32 without conversion.
func BitCastF32ToU32(v Vec[float32]) Vec[uint32] {
	result := make([]uint32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float32bits(v.data[i])
	}
	return Vec[uint32]{data: result}
}

// BitCastU64ToF64 reinterprets uint64 bits as float64 without conversion.
func BitCastU64ToF64(v Vec[uint64]) Vec[float64] {
	result := make([]float64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float64frombits(v.data[i])
	}
	return Vec[float64]{data: result}
}

// BitCastF64ToU64 reinterprets float64 bits as uint64 without conversion.
func BitCastF64ToU64(v Vec[float64]) Vec[uint64] {
	result := make([]uint64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float64bits(v.data[i])
	}
	return Vec[uint64]{data: result}
}

// ============================================================================
// IEEE 754 Exponent Manipulation (for range reduction in math functions)
// ============================================================================

// Pow2[T Floats](k) computes 2^k for integer k values via IEEE 754 bit manipulation.
// This is essential for reconstructing exp(x) = 2^k * exp(r) after range reduction.
// k should be in the valid exponent range: [-126, 127] for float32, [-1022, 1023] for float64.
// For Float16/BFloat16, the computation is done in float32 and converted back.
func Pow2[T Floats](k Vec[int32]) Vec[T] {
	result := make([]T, len(k.data))
	// Determine type at runtime
	var zero T
	switch any(zero).(type) {
	case Float16:
		// Float16: compute in float32, convert back
		// Float16 exponent range is [-14, 15] but we compute via float32
		for i, ki := range k.data {
			var f float32
			if ki < -126 {
				f = 0 // underflow
			} else if ki > 127 {
				f = float32(math.Inf(1)) // overflow
			} else {
				bits := uint32(ki+127) << 23
				f = math.Float32frombits(bits)
			}
			result[i] = any(Float32ToFloat16(f)).(T)
		}
	case BFloat16:
		// BFloat16: compute in float32, convert back
		for i, ki := range k.data {
			var f float32
			if ki < -126 {
				f = 0 // underflow
			} else if ki > 127 {
				f = float32(math.Inf(1)) // overflow
			} else {
				bits := uint32(ki+127) << 23
				f = math.Float32frombits(bits)
			}
			result[i] = any(Float32ToBFloat16(f)).(T)
		}
	case float32:
		// float32: exponent in bits [23:30], bias = 127
		// 2^k = bits (k+127) << 23 interpreted as float32
		for i, ki := range k.data {
			var f float32
			if ki < -126 {
				f = 0 // underflow
			} else if ki > 127 {
				f = float32(math.Inf(1)) // overflow
			} else {
				bits := uint32(ki+127) << 23
				f = math.Float32frombits(bits)
			}
			result[i] = any(f).(T)
		}
	case float64:
		// float64: exponent in bits [52:62], bias = 1023
		// 2^k = bits (k+1023) << 52 interpreted as float64
		for i, ki := range k.data {
			var f float64
			if ki < -1022 {
				f = 0 // underflow
			} else if ki > 1023 {
				f = math.Inf(1) // overflow
			} else {
				bits := uint64(int64(ki)+1023) << 52
				f = math.Float64frombits(bits)
			}
			result[i] = any(f).(T)
		}
	}
	return Vec[T]{data: result}
}

// GetExponent[T Floats](v) extracts the unbiased exponent from IEEE 754 floats.
// Returns the integer exponent such that v ≈ 2^exp * mantissa, where 1 <= mantissa < 2.
// For denormals and special values, behavior is implementation-defined.
// For Float16/BFloat16, the value is converted to float32 first.
func GetExponent[T Floats](v Vec[T]) Vec[int32] {
	result := make([]int32, len(v.data))
	var zero T
	switch any(zero).(type) {
	case Float16:
		// Float16: exponent in bits [10:14], bias = 15
		for i, x := range v.data {
			bits := uint16(any(x).(Float16))
			exp := int32((bits >> 10) & 0x1F)
			if exp == 0 || exp == 31 {
				result[i] = 0 // denormal or special
			} else {
				result[i] = exp - 15
			}
		}
	case BFloat16:
		// BFloat16: same exponent format as float32 (8 bits), bias = 127
		for i, x := range v.data {
			bits := uint16(any(x).(BFloat16))
			exp := int32((bits >> 7) & 0xFF)
			if exp == 0 || exp == 255 {
				result[i] = 0 // denormal or special
			} else {
				result[i] = exp - 127
			}
		}
	case float32:
		for i, x := range v.data {
			bits := math.Float32bits(any(x).(float32))
			exp := int32((bits >> 23) & 0xFF)
			if exp == 0 || exp == 255 {
				result[i] = 0 // denormal or special
			} else {
				result[i] = exp - 127
			}
		}
	case float64:
		for i, x := range v.data {
			bits := math.Float64bits(any(x).(float64))
			exp := int32((bits >> 52) & 0x7FF)
			if exp == 0 || exp == 2047 {
				result[i] = 0 // denormal or special
			} else {
				result[i] = exp - 1023
			}
		}
	}
	return Vec[int32]{data: result}
}

// GetMantissa[T Floats](v) extracts the mantissa with exponent normalized to [1, 2).
// Returns a value m such that v = 2^exp * m, where 1 <= m < 2.
// For denormals and special values, behavior is implementation-defined.
func GetMantissa[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	var zero T
	switch any(zero).(type) {
	case Float16:
		// Float16: sign(1) | exp(5) | mantissa(10)
		// Clear exponent, set to 15 (2^0 = 1)
		for i, x := range v.data {
			bits := uint16(any(x).(Float16))
			// Mask: keep sign (bit 15) and mantissa (bits 0-9), set exponent to 15
			mantissa := (bits & 0x83FF) | 0x3C00 // 0x3C00 = exponent 15 << 10
			result[i] = any(Float16(mantissa)).(T)
		}
	case BFloat16:
		// BFloat16: sign(1) | exp(8) | mantissa(7)
		// Clear exponent, set to 127 (2^0 = 1)
		for i, x := range v.data {
			bits := uint16(any(x).(BFloat16))
			// Mask: keep sign (bit 15) and mantissa (bits 0-6), set exponent to 127
			mantissa := (bits & 0x807F) | 0x3F80 // 0x3F80 = exponent 127 << 7
			result[i] = any(BFloat16(mantissa)).(T)
		}
	case float32:
		for i, x := range v.data {
			bits := math.Float32bits(any(x).(float32))
			// Clear exponent, set to 127 (2^0 = 1)
			mantissa := (bits & 0x807FFFFF) | 0x3F800000
			result[i] = any(math.Float32frombits(mantissa)).(T)
		}
	case float64:
		for i, x := range v.data {
			bits := math.Float64bits(any(x).(float64))
			// Clear exponent, set to 1023 (2^0 = 1)
			mantissa := (bits & 0x800FFFFFFFFFFFFF) | 0x3FF0000000000000
			result[i] = any(math.Float64frombits(mantissa)).(T)
		}
	}
	return Vec[T]{data: result}
}
