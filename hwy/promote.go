package hwy

// This file provides pure Go (scalar) implementations of type promotion and demotion operations.
// When SIMD implementations are available (promote_avx2.go, promote_avx512.go),
// they can be used for higher performance on supported hardware.
//
// PromoteTo operations widen types (e.g., float32 -> float64, int16 -> int32)
// DemoteTo operations narrow types (e.g., float64 -> float32, int32 -> int16)
//
// Note: Go generics don't support type relationships like "T is narrower than U",
// so we provide concrete type-specific functions.

// PromoteF32ToF64 widens float32 to float64.
// Each float32 lane is converted to float64.
func PromoteF32ToF64(v Vec[float32]) Vec[float64] {
	result := make([]float64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = float64(v.data[i])
	}
	return Vec[float64]{data: result}
}

// PromoteLowerF32ToF64 promotes only the lower half of float32 lanes to float64.
// Input: 8 float32 lanes -> Output: 4 float64 lanes (from lower 4 float32).
func PromoteLowerF32ToF64(v Vec[float32]) Vec[float64] {
	n := len(v.data) / 2
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = float64(v.data[i])
	}
	return Vec[float64]{data: result}
}

// PromoteUpperF32ToF64 promotes only the upper half of float32 lanes to float64.
// Input: 8 float32 lanes -> Output: 4 float64 lanes (from upper 4 float32).
func PromoteUpperF32ToF64(v Vec[float32]) Vec[float64] {
	half := len(v.data) / 2
	n := len(v.data) - half
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = float64(v.data[half+i])
	}
	return Vec[float64]{data: result}
}

// DemoteF64ToF32 narrows float64 to float32.
// Each float64 lane is converted to float32, potentially losing precision.
func DemoteF64ToF32(v Vec[float64]) Vec[float32] {
	result := make([]float32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = float32(v.data[i])
	}
	return Vec[float32]{data: result}
}

// DemoteTwoF64ToF32 demotes two float64 vectors to a single float32 vector.
// Input: 2 vectors of 4 float64 each -> Output: 1 vector of 8 float32.
func DemoteTwoF64ToF32(lo, hi Vec[float64]) Vec[float32] {
	n := len(lo.data) + len(hi.data)
	result := make([]float32, n)
	for i := 0; i < len(lo.data); i++ {
		result[i] = float32(lo.data[i])
	}
	for i := 0; i < len(hi.data); i++ {
		result[len(lo.data)+i] = float32(hi.data[i])
	}
	return Vec[float32]{data: result}
}

// PromoteI8ToI16 widens int8 to int16 (sign-extended).
func PromoteI8ToI16(v Vec[int8]) Vec[int16] {
	result := make([]int16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int16(v.data[i])
	}
	return Vec[int16]{data: result}
}

// PromoteLowerI8ToI16 promotes only the lower half of int8 lanes to int16.
func PromoteLowerI8ToI16(v Vec[int8]) Vec[int16] {
	n := len(v.data) / 2
	result := make([]int16, n)
	for i := 0; i < n; i++ {
		result[i] = int16(v.data[i])
	}
	return Vec[int16]{data: result}
}

// PromoteUpperI8ToI16 promotes only the upper half of int8 lanes to int16.
func PromoteUpperI8ToI16(v Vec[int8]) Vec[int16] {
	half := len(v.data) / 2
	n := len(v.data) - half
	result := make([]int16, n)
	for i := 0; i < n; i++ {
		result[i] = int16(v.data[half+i])
	}
	return Vec[int16]{data: result}
}

// PromoteI16ToI32 widens int16 to int32 (sign-extended).
func PromoteI16ToI32(v Vec[int16]) Vec[int32] {
	result := make([]int32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int32(v.data[i])
	}
	return Vec[int32]{data: result}
}

// PromoteLowerI16ToI32 promotes only the lower half of int16 lanes to int32.
func PromoteLowerI16ToI32(v Vec[int16]) Vec[int32] {
	n := len(v.data) / 2
	result := make([]int32, n)
	for i := 0; i < n; i++ {
		result[i] = int32(v.data[i])
	}
	return Vec[int32]{data: result}
}

// PromoteUpperI16ToI32 promotes only the upper half of int16 lanes to int32.
func PromoteUpperI16ToI32(v Vec[int16]) Vec[int32] {
	half := len(v.data) / 2
	n := len(v.data) - half
	result := make([]int32, n)
	for i := 0; i < n; i++ {
		result[i] = int32(v.data[half+i])
	}
	return Vec[int32]{data: result}
}

// PromoteI32ToI64 widens int32 to int64 (sign-extended).
func PromoteI32ToI64(v Vec[int32]) Vec[int64] {
	result := make([]int64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int64(v.data[i])
	}
	return Vec[int64]{data: result}
}

// PromoteLowerI32ToI64 promotes only the lower half of int32 lanes to int64.
func PromoteLowerI32ToI64(v Vec[int32]) Vec[int64] {
	n := len(v.data) / 2
	result := make([]int64, n)
	for i := 0; i < n; i++ {
		result[i] = int64(v.data[i])
	}
	return Vec[int64]{data: result}
}

// PromoteUpperI32ToI64 promotes only the upper half of int32 lanes to int64.
func PromoteUpperI32ToI64(v Vec[int32]) Vec[int64] {
	half := len(v.data) / 2
	n := len(v.data) - half
	result := make([]int64, n)
	for i := 0; i < n; i++ {
		result[i] = int64(v.data[half+i])
	}
	return Vec[int64]{data: result}
}

// PromoteU8ToU16 widens uint8 to uint16 (zero-extended).
func PromoteU8ToU16(v Vec[uint8]) Vec[uint16] {
	result := make([]uint16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = uint16(v.data[i])
	}
	return Vec[uint16]{data: result}
}

// PromoteLowerU8ToU16 promotes only the lower half of uint8 lanes to uint16.
func PromoteLowerU8ToU16(v Vec[uint8]) Vec[uint16] {
	n := len(v.data) / 2
	result := make([]uint16, n)
	for i := 0; i < n; i++ {
		result[i] = uint16(v.data[i])
	}
	return Vec[uint16]{data: result}
}

// PromoteUpperU8ToU16 promotes only the upper half of uint8 lanes to uint16.
func PromoteUpperU8ToU16(v Vec[uint8]) Vec[uint16] {
	half := len(v.data) / 2
	n := len(v.data) - half
	result := make([]uint16, n)
	for i := 0; i < n; i++ {
		result[i] = uint16(v.data[half+i])
	}
	return Vec[uint16]{data: result}
}

// PromoteU16ToU32 widens uint16 to uint32 (zero-extended).
func PromoteU16ToU32(v Vec[uint16]) Vec[uint32] {
	result := make([]uint32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = uint32(v.data[i])
	}
	return Vec[uint32]{data: result}
}

// PromoteLowerU16ToU32 promotes only the lower half of uint16 lanes to uint32.
func PromoteLowerU16ToU32(v Vec[uint16]) Vec[uint32] {
	n := len(v.data) / 2
	result := make([]uint32, n)
	for i := 0; i < n; i++ {
		result[i] = uint32(v.data[i])
	}
	return Vec[uint32]{data: result}
}

// PromoteUpperU16ToU32 promotes only the upper half of uint16 lanes to uint32.
func PromoteUpperU16ToU32(v Vec[uint16]) Vec[uint32] {
	half := len(v.data) / 2
	n := len(v.data) - half
	result := make([]uint32, n)
	for i := 0; i < n; i++ {
		result[i] = uint32(v.data[half+i])
	}
	return Vec[uint32]{data: result}
}

// PromoteU32ToU64 widens uint32 to uint64 (zero-extended).
func PromoteU32ToU64(v Vec[uint32]) Vec[uint64] {
	result := make([]uint64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = uint64(v.data[i])
	}
	return Vec[uint64]{data: result}
}

// PromoteLowerU32ToU64 promotes only the lower half of uint32 lanes to uint64.
func PromoteLowerU32ToU64(v Vec[uint32]) Vec[uint64] {
	n := len(v.data) / 2
	result := make([]uint64, n)
	for i := 0; i < n; i++ {
		result[i] = uint64(v.data[i])
	}
	return Vec[uint64]{data: result}
}

// PromoteUpperU32ToU64 promotes only the upper half of uint32 lanes to uint64.
func PromoteUpperU32ToU64(v Vec[uint32]) Vec[uint64] {
	half := len(v.data) / 2
	n := len(v.data) - half
	result := make([]uint64, n)
	for i := 0; i < n; i++ {
		result[i] = uint64(v.data[half+i])
	}
	return Vec[uint64]{data: result}
}

// DemoteI16ToI8 narrows int16 to int8 (saturating).
// Values outside int8 range are clamped to [-128, 127].
func DemoteI16ToI8(v Vec[int16]) Vec[int8] {
	result := make([]int8, len(v.data))
	for i := 0; i < len(v.data); i++ {
		val := v.data[i]
		if val > 127 {
			result[i] = 127
		} else if val < -128 {
			result[i] = -128
		} else {
			result[i] = int8(val)
		}
	}
	return Vec[int8]{data: result}
}

// DemoteTwoI16ToI8 demotes two int16 vectors to a single int8 vector (saturating).
func DemoteTwoI16ToI8(lo, hi Vec[int16]) Vec[int8] {
	n := len(lo.data) + len(hi.data)
	result := make([]int8, n)
	for i := 0; i < len(lo.data); i++ {
		val := lo.data[i]
		if val > 127 {
			result[i] = 127
		} else if val < -128 {
			result[i] = -128
		} else {
			result[i] = int8(val)
		}
	}
	for i := 0; i < len(hi.data); i++ {
		val := hi.data[i]
		if val > 127 {
			result[len(lo.data)+i] = 127
		} else if val < -128 {
			result[len(lo.data)+i] = -128
		} else {
			result[len(lo.data)+i] = int8(val)
		}
	}
	return Vec[int8]{data: result}
}

// DemoteI32ToI16 narrows int32 to int16 (saturating).
// Values outside int16 range are clamped to [-32768, 32767].
func DemoteI32ToI16(v Vec[int32]) Vec[int16] {
	result := make([]int16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		val := v.data[i]
		if val > 32767 {
			result[i] = 32767
		} else if val < -32768 {
			result[i] = -32768
		} else {
			result[i] = int16(val)
		}
	}
	return Vec[int16]{data: result}
}

// DemoteTwoI32ToI16 demotes two int32 vectors to a single int16 vector (saturating).
func DemoteTwoI32ToI16(lo, hi Vec[int32]) Vec[int16] {
	n := len(lo.data) + len(hi.data)
	result := make([]int16, n)
	for i := 0; i < len(lo.data); i++ {
		val := lo.data[i]
		if val > 32767 {
			result[i] = 32767
		} else if val < -32768 {
			result[i] = -32768
		} else {
			result[i] = int16(val)
		}
	}
	for i := 0; i < len(hi.data); i++ {
		val := hi.data[i]
		if val > 32767 {
			result[len(lo.data)+i] = 32767
		} else if val < -32768 {
			result[len(lo.data)+i] = -32768
		} else {
			result[len(lo.data)+i] = int16(val)
		}
	}
	return Vec[int16]{data: result}
}

// DemoteI64ToI32 narrows int64 to int32 (saturating).
// Values outside int32 range are clamped to [-2147483648, 2147483647].
func DemoteI64ToI32(v Vec[int64]) Vec[int32] {
	result := make([]int32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		val := v.data[i]
		if val > 2147483647 {
			result[i] = 2147483647
		} else if val < -2147483648 {
			result[i] = -2147483648
		} else {
			result[i] = int32(val)
		}
	}
	return Vec[int32]{data: result}
}

// DemoteTwoI64ToI32 demotes two int64 vectors to a single int32 vector (saturating).
func DemoteTwoI64ToI32(lo, hi Vec[int64]) Vec[int32] {
	n := len(lo.data) + len(hi.data)
	result := make([]int32, n)
	for i := 0; i < len(lo.data); i++ {
		val := lo.data[i]
		if val > 2147483647 {
			result[i] = 2147483647
		} else if val < -2147483648 {
			result[i] = -2147483648
		} else {
			result[i] = int32(val)
		}
	}
	for i := 0; i < len(hi.data); i++ {
		val := hi.data[i]
		if val > 2147483647 {
			result[len(lo.data)+i] = 2147483647
		} else if val < -2147483648 {
			result[len(lo.data)+i] = -2147483648
		} else {
			result[len(lo.data)+i] = int32(val)
		}
	}
	return Vec[int32]{data: result}
}

// DemoteU16ToU8 narrows uint16 to uint8 (saturating).
// Values > 255 are clamped to 255.
func DemoteU16ToU8(v Vec[uint16]) Vec[uint8] {
	result := make([]uint8, len(v.data))
	for i := 0; i < len(v.data); i++ {
		if v.data[i] > 255 {
			result[i] = 255
		} else {
			result[i] = uint8(v.data[i])
		}
	}
	return Vec[uint8]{data: result}
}

// DemoteTwoU16ToU8 demotes two uint16 vectors to a single uint8 vector (saturating).
func DemoteTwoU16ToU8(lo, hi Vec[uint16]) Vec[uint8] {
	n := len(lo.data) + len(hi.data)
	result := make([]uint8, n)
	for i := 0; i < len(lo.data); i++ {
		if lo.data[i] > 255 {
			result[i] = 255
		} else {
			result[i] = uint8(lo.data[i])
		}
	}
	for i := 0; i < len(hi.data); i++ {
		if hi.data[i] > 255 {
			result[len(lo.data)+i] = 255
		} else {
			result[len(lo.data)+i] = uint8(hi.data[i])
		}
	}
	return Vec[uint8]{data: result}
}

// DemoteU32ToU16 narrows uint32 to uint16 (saturating).
// Values > 65535 are clamped to 65535.
func DemoteU32ToU16(v Vec[uint32]) Vec[uint16] {
	result := make([]uint16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		if v.data[i] > 65535 {
			result[i] = 65535
		} else {
			result[i] = uint16(v.data[i])
		}
	}
	return Vec[uint16]{data: result}
}

// DemoteTwoU32ToU16 demotes two uint32 vectors to a single uint16 vector (saturating).
func DemoteTwoU32ToU16(lo, hi Vec[uint32]) Vec[uint16] {
	n := len(lo.data) + len(hi.data)
	result := make([]uint16, n)
	for i := 0; i < len(lo.data); i++ {
		if lo.data[i] > 65535 {
			result[i] = 65535
		} else {
			result[i] = uint16(lo.data[i])
		}
	}
	for i := 0; i < len(hi.data); i++ {
		if hi.data[i] > 65535 {
			result[len(lo.data)+i] = 65535
		} else {
			result[len(lo.data)+i] = uint16(hi.data[i])
		}
	}
	return Vec[uint16]{data: result}
}

// DemoteU64ToU32 narrows uint64 to uint32 (saturating).
// Values > 0xFFFFFFFF are clamped to 0xFFFFFFFF.
func DemoteU64ToU32(v Vec[uint64]) Vec[uint32] {
	result := make([]uint32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		if v.data[i] > 0xFFFFFFFF {
			result[i] = 0xFFFFFFFF
		} else {
			result[i] = uint32(v.data[i])
		}
	}
	return Vec[uint32]{data: result}
}

// DemoteTwoU64ToU32 demotes two uint64 vectors to a single uint32 vector (saturating).
func DemoteTwoU64ToU32(lo, hi Vec[uint64]) Vec[uint32] {
	n := len(lo.data) + len(hi.data)
	result := make([]uint32, n)
	for i := 0; i < len(lo.data); i++ {
		if lo.data[i] > 0xFFFFFFFF {
			result[i] = 0xFFFFFFFF
		} else {
			result[i] = uint32(lo.data[i])
		}
	}
	for i := 0; i < len(hi.data); i++ {
		if hi.data[i] > 0xFFFFFFFF {
			result[len(lo.data)+i] = 0xFFFFFFFF
		} else {
			result[len(lo.data)+i] = uint32(hi.data[i])
		}
	}
	return Vec[uint32]{data: result}
}

// TruncateI16ToI8 narrows int16 to int8 (truncating, not saturating).
// Only the lower 8 bits are kept.
func TruncateI16ToI8(v Vec[int16]) Vec[int8] {
	result := make([]int8, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int8(v.data[i])
	}
	return Vec[int8]{data: result}
}

// TruncateI32ToI16 narrows int32 to int16 (truncating, not saturating).
// Only the lower 16 bits are kept.
func TruncateI32ToI16(v Vec[int32]) Vec[int16] {
	result := make([]int16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int16(v.data[i])
	}
	return Vec[int16]{data: result}
}

// TruncateI64ToI32 narrows int64 to int32 (truncating, not saturating).
// Only the lower 32 bits are kept.
func TruncateI64ToI32(v Vec[int64]) Vec[int32] {
	result := make([]int32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int32(v.data[i])
	}
	return Vec[int32]{data: result}
}

// TruncateU16ToU8 narrows uint16 to uint8 (truncating, not saturating).
// Only the lower 8 bits are kept.
func TruncateU16ToU8(v Vec[uint16]) Vec[uint8] {
	result := make([]uint8, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = uint8(v.data[i])
	}
	return Vec[uint8]{data: result}
}

// TruncateU32ToU16 narrows uint32 to uint16 (truncating, not saturating).
// Only the lower 16 bits are kept.
func TruncateU32ToU16(v Vec[uint32]) Vec[uint16] {
	result := make([]uint16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = uint16(v.data[i])
	}
	return Vec[uint16]{data: result}
}

// TruncateU64ToU32 narrows uint64 to uint32 (truncating, not saturating).
// Only the lower 32 bits are kept.
func TruncateU64ToU32(v Vec[uint64]) Vec[uint32] {
	result := make([]uint32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = uint32(v.data[i])
	}
	return Vec[uint32]{data: result}
}
