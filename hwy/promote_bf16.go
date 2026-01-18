package hwy

// This file provides pure Go (scalar) implementations of BFloat16 promotion and demotion operations.
// When SIMD implementations are available (via GoAT-generated assembly), they will provide
// higher performance on supported hardware.
//
// BFloat16 (Brain Float) uses 16 bits: Sign(1) | Exp(8) | Mantissa(7)
// Float32 (IEEE 754) uses 32 bits: Sign(1) | Exp(8) | Mantissa(23)
//
// Key insight: BFloat16 is simply float32 with the lower 16 mantissa bits truncated.
// This makes conversions trivial (just bit shifting with rounding).
//
// BFloat16 has the same dynamic range as float32 (same exponent width),
// making it ideal for ML training where range matters more than precision.

// PromoteBF16ToF32 widens BFloat16 to float32.
// This is a simple bit shift operation - just append 16 zero bits.
func PromoteBF16ToF32(v Vec[BFloat16]) Vec[float32] {
	result := make([]float32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = BFloat16ToFloat32(v.data[i])
	}
	return Vec[float32]{data: result}
}

// PromoteLowerBF16ToF32 promotes only the lower half of BFloat16 lanes to float32.
// Input: 2N BFloat16 lanes -> Output: N float32 lanes (from lower N BFloat16).
func PromoteLowerBF16ToF32(v Vec[BFloat16]) Vec[float32] {
	n := len(v.data) / 2
	result := make([]float32, n)
	for i := range n {
		result[i] = BFloat16ToFloat32(v.data[i])
	}
	return Vec[float32]{data: result}
}

// PromoteUpperBF16ToF32 promotes only the upper half of BFloat16 lanes to float32.
// Input: 2N BFloat16 lanes -> Output: N float32 lanes (from upper N BFloat16).
func PromoteUpperBF16ToF32(v Vec[BFloat16]) Vec[float32] {
	half := len(v.data) / 2
	n := len(v.data) - half
	result := make([]float32, n)
	for i := range n {
		result[i] = BFloat16ToFloat32(v.data[half+i])
	}
	return Vec[float32]{data: result}
}

// DemoteF32ToBF16 narrows float32 to BFloat16.
// Uses round-to-nearest-even on the truncated bits.
func DemoteF32ToBF16(v Vec[float32]) Vec[BFloat16] {
	result := make([]BFloat16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = Float32ToBFloat16(v.data[i])
	}
	return Vec[BFloat16]{data: result}
}

// DemoteTwoF32ToBF16 demotes two float32 vectors to a single BFloat16 vector.
// Input: 2 vectors of N float32 each -> Output: 1 vector of 2N BFloat16.
// The 'lo' vector fills the lower lanes, 'hi' vector fills the upper lanes.
func DemoteTwoF32ToBF16(lo, hi Vec[float32]) Vec[BFloat16] {
	n := len(lo.data) + len(hi.data)
	result := make([]BFloat16, n)
	for i := 0; i < len(lo.data); i++ {
		result[i] = Float32ToBFloat16(lo.data[i])
	}
	for i := 0; i < len(hi.data); i++ {
		result[len(lo.data)+i] = Float32ToBFloat16(hi.data[i])
	}
	return Vec[BFloat16]{data: result}
}

// PromoteBF16ToF64 widens BFloat16 to float64.
// This is a two-step promotion: BFloat16 -> float32 -> float64.
func PromoteBF16ToF64(v Vec[BFloat16]) Vec[float64] {
	result := make([]float64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = float64(BFloat16ToFloat32(v.data[i]))
	}
	return Vec[float64]{data: result}
}

// DemoteF64ToBF16 narrows float64 to BFloat16.
// This is a two-step demotion: float64 -> float32 -> BFloat16.
func DemoteF64ToBF16(v Vec[float64]) Vec[BFloat16] {
	result := make([]BFloat16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = Float32ToBFloat16(float32(v.data[i]))
	}
	return Vec[BFloat16]{data: result}
}

// LoadBF16 creates a BFloat16 vector by loading uint16 bits from a slice
// and interpreting them as BFloat16 values.
func LoadBF16(src []uint16) Vec[BFloat16] {
	n := min(len(src), MaxLanes[BFloat16]())
	data := make([]BFloat16, n)
	for i := range n {
		data[i] = BFloat16(src[i])
	}
	return Vec[BFloat16]{data: data}
}

// StoreBF16 writes BFloat16 vector data to a uint16 slice.
func StoreBF16(v Vec[BFloat16], dst []uint16) {
	n := min(len(dst), len(v.data))
	for i := range n {
		dst[i] = uint16(v.data[i])
	}
}

// LoadBF16FromF32 loads float32 values and converts them to BFloat16.
func LoadBF16FromF32(src []float32) Vec[BFloat16] {
	n := min(len(src), MaxLanes[BFloat16]())
	data := make([]BFloat16, n)
	for i := range n {
		data[i] = Float32ToBFloat16(src[i])
	}
	return Vec[BFloat16]{data: data}
}

// StoreBF16ToF32 converts BFloat16 vector to float32 and stores.
func StoreBF16ToF32(v Vec[BFloat16], dst []float32) {
	n := min(len(dst), len(v.data))
	for i := range n {
		dst[i] = BFloat16ToFloat32(v.data[i])
	}
}

// SetBF16 creates a BFloat16 vector with all lanes set to the same value.
func SetBF16(value BFloat16) Vec[BFloat16] {
	n := MaxLanes[BFloat16]()
	data := make([]BFloat16, n)
	for i := range data {
		data[i] = value
	}
	return Vec[BFloat16]{data: data}
}

// SetBF16FromF32 creates a BFloat16 vector with all lanes set to a converted float32.
func SetBF16FromF32(value float32) Vec[BFloat16] {
	return SetBF16(Float32ToBFloat16(value))
}

// ZeroBF16 creates a BFloat16 vector with all lanes set to zero.
func ZeroBF16() Vec[BFloat16] {
	n := MaxLanes[BFloat16]()
	data := make([]BFloat16, n)
	return Vec[BFloat16]{data: data}
}

// ConvertF16ToBF16 converts a Float16 vector to BFloat16.
// This involves promotion to float32 and then demotion to BFloat16.
func ConvertF16ToBF16(v Vec[Float16]) Vec[BFloat16] {
	result := make([]BFloat16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = Float16ToBFloat16(v.data[i])
	}
	return Vec[BFloat16]{data: result}
}

// ConvertBF16ToF16 converts a BFloat16 vector to Float16.
// This involves promotion to float32 and then demotion to Float16.
// Note: May overflow since Float16 has smaller range than BFloat16.
func ConvertBF16ToF16(v Vec[BFloat16]) Vec[Float16] {
	result := make([]Float16, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = BFloat16ToFloat16(v.data[i])
	}
	return Vec[Float16]{data: result}
}
