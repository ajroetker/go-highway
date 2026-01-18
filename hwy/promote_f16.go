package hwy

// This file provides pure Go (scalar) implementations of Float16 promotion and demotion operations.
// When SIMD implementations are available (via GoAT-generated assembly), they will provide
// higher performance on supported hardware.
//
// Float16 (IEEE 754 half-precision) uses 16 bits: Sign(1) | Exp(5) | Mantissa(10)
// Float32 (IEEE 754 single-precision) uses 32 bits: Sign(1) | Exp(8) | Mantissa(23)
//
// Promotion: Float16 -> Float32 (widens, no precision loss)
// Demotion: Float32 -> Float16 (narrows, may lose precision or overflow)

// PromoteF16ToF32 widens Float16 to float32.
// Each Float16 lane is converted to float32 with no loss of precision.
func PromoteF16ToF32(v Vec[Float16]) Vec[float32] {
	result := make([]float32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = Float16ToFloat32(v.data[i])
	}
	return Vec[float32]{data: result}
}

// PromoteLowerF16ToF32 promotes only the lower half of Float16 lanes to float32.
// Input: 2N Float16 lanes -> Output: N float32 lanes (from lower N Float16).
// This is useful for SIMD where Float16 vectors have 2x the lanes of float32.
func PromoteLowerF16ToF32(v Vec[Float16]) Vec[float32] {
	n := len(v.data) / 2
	result := make([]float32, n)
	for i := range n {
		result[i] = Float16ToFloat32(v.data[i])
	}
	return Vec[float32]{data: result}
}

// PromoteUpperF16ToF32 promotes only the upper half of Float16 lanes to float32.
// Input: 2N Float16 lanes -> Output: N float32 lanes (from upper N Float16).
func PromoteUpperF16ToF32(v Vec[Float16]) Vec[float32] {
	half := len(v.data) / 2
	n := len(v.data) - half
	result := make([]float32, n)
	for i := range n {
		result[i] = Float16ToFloat32(v.data[half+i])
	}
	return Vec[float32]{data: result}
}

// DemoteF32ToF16 narrows float32 to Float16.
// Each float32 lane is converted to Float16 with round-to-nearest-even.
// Values outside Float16 range overflow to infinity.
func DemoteF32ToF16(v Vec[float32]) Vec[Float16] {
	result := make([]Float16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = Float32ToFloat16(v.data[i])
	}
	return Vec[Float16]{data: result}
}

// DemoteTwoF32ToF16 demotes two float32 vectors to a single Float16 vector.
// Input: 2 vectors of N float32 each -> Output: 1 vector of 2N Float16.
// The 'lo' vector fills the lower lanes, 'hi' vector fills the upper lanes.
func DemoteTwoF32ToF16(lo, hi Vec[float32]) Vec[Float16] {
	n := len(lo.data) + len(hi.data)
	result := make([]Float16, n)
	for i := 0; i < len(lo.data); i++ {
		result[i] = Float32ToFloat16(lo.data[i])
	}
	for i := 0; i < len(hi.data); i++ {
		result[len(lo.data)+i] = Float32ToFloat16(hi.data[i])
	}
	return Vec[Float16]{data: result}
}

// PromoteF16ToF64 widens Float16 to float64.
// This is a two-step promotion: Float16 -> float32 -> float64.
func PromoteF16ToF64(v Vec[Float16]) Vec[float64] {
	result := make([]float64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = float64(Float16ToFloat32(v.data[i]))
	}
	return Vec[float64]{data: result}
}

// DemoteF64ToF16 narrows float64 to Float16.
// This is a two-step demotion: float64 -> float32 -> Float16.
// May lose significant precision due to double narrowing.
func DemoteF64ToF16(v Vec[float64]) Vec[Float16] {
	result := make([]Float16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = Float32ToFloat16(float32(v.data[i]))
	}
	return Vec[Float16]{data: result}
}

// LoadF16 creates a Float16 vector by loading uint16 bits from a slice
// and interpreting them as Float16 values.
func LoadF16(src []uint16) Vec[Float16] {
	n := min(len(src), MaxLanes[Float16]())
	data := make([]Float16, n)
	for i := range n {
		data[i] = Float16(src[i])
	}
	return Vec[Float16]{data: data}
}

// StoreF16 writes Float16 vector data to a uint16 slice.
func StoreF16(v Vec[Float16], dst []uint16) {
	n := min(len(dst), len(v.data))
	for i := range n {
		dst[i] = uint16(v.data[i])
	}
}

// LoadF16FromF32 loads float32 values and converts them to Float16.
// This is equivalent to Load followed by Demote, but more efficient.
func LoadF16FromF32(src []float32) Vec[Float16] {
	n := min(len(src), MaxLanes[Float16]())
	data := make([]Float16, n)
	for i := range n {
		data[i] = Float32ToFloat16(src[i])
	}
	return Vec[Float16]{data: data}
}

// StoreF16ToF32 converts Float16 vector to float32 and stores.
// This is equivalent to Promote followed by Store, but more efficient.
func StoreF16ToF32(v Vec[Float16], dst []float32) {
	n := min(len(dst), len(v.data))
	for i := range n {
		dst[i] = Float16ToFloat32(v.data[i])
	}
}

// SetF16 creates a Float16 vector with all lanes set to the same value.
func SetF16(value Float16) Vec[Float16] {
	n := MaxLanes[Float16]()
	data := make([]Float16, n)
	for i := range data {
		data[i] = value
	}
	return Vec[Float16]{data: data}
}

// SetF16FromF32 creates a Float16 vector with all lanes set to a converted float32.
func SetF16FromF32(value float32) Vec[Float16] {
	return SetF16(Float32ToFloat16(value))
}

// ZeroF16 creates a Float16 vector with all lanes set to zero.
func ZeroF16() Vec[Float16] {
	n := MaxLanes[Float16]()
	data := make([]Float16, n)
	return Vec[Float16]{data: data}
}
