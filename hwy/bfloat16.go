package hwy

import "math"

// BFloat16 represents a Brain Float 16 (bfloat16) number.
// It has the same exponent range as float32 but reduced precision,
// making it ideal for machine learning training where dynamic range
// matters more than precision.
//
// Format: Sign (1 bit) | Exponent (8 bits) | Mantissa (7 bits)
//
//	S | EEEEEEEE | MMMMMMM
//
// Key insight: BFloat16 is simply float32 with the lower 16 mantissa bits
// truncated. This makes conversions trivial.
//
// Properties:
//   - Total bits: 16
//   - Exponent bits: 8 (same as float32, bias: 127)
//   - Mantissa bits: 7
//   - Max value: ~3.4e38 (same as float32)
//   - Min positive normal: ~1.2e-38 (same as float32)
//   - Precision: ~2.4 decimal digits
type BFloat16 uint16

// BFloat16 constants for special values.
const (
	BFloat16Zero      BFloat16 = 0x0000 // Positive zero
	BFloat16NegZero   BFloat16 = 0x8000 // Negative zero
	BFloat16One       BFloat16 = 0x3F80 // 1.0
	BFloat16NegOne    BFloat16 = 0xBF80 // -1.0
	BFloat16MaxValue  BFloat16 = 0x7F7F // ~3.39e38 (max finite value)
	BFloat16MinNormal BFloat16 = 0x0080 // ~1.18e-38 (smallest normal)
	BFloat16MinValue  BFloat16 = 0x0001 // Smallest denormal
	BFloat16Inf       BFloat16 = 0x7F80 // Positive infinity
	BFloat16NegInf    BFloat16 = 0xFF80 // Negative infinity
	BFloat16NaN       BFloat16 = 0x7FC0 // Quiet NaN (canonical)

	// Internal constants (same as float32)
	bfloat16ExpBias      = 127
	bfloat16ExpMask      = 0xFF
	bfloat16MantissaBits = 7
	bfloat16MantissaMask = 0x7F
	bfloat16SignMask     = 0x8000
)

// BFloat16ToFloat32 converts a single BFloat16 to float32.
// This is a simple bit shift since bfloat16 is truncated float32.
func BFloat16ToFloat32(b BFloat16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// Float32ToBFloat16 converts a float32 to BFloat16.
// Uses round-to-nearest-even on the truncated bits.
func Float32ToBFloat16(f float32) BFloat16 {
	bits := math.Float32bits(f)

	// Handle NaN specially - preserve sign and set quiet NaN
	if bits&0x7FFFFFFF > 0x7F800000 {
		// Input is NaN - return canonical quiet NaN with same sign
		return BFloat16((bits >> 16) | 0x0040)
	}

	// Round to nearest even
	// The rounding position is bit 15 (just below the bf16 mantissa)
	// Add 0x7FFF + (bit 16 of original) for rounding
	// This implements round-to-nearest-even:
	// - If bit 15 is 0, truncate
	// - If bit 15 is 1 and bits 0-14 are non-zero, round up
	// - If bit 15 is 1 and bits 0-14 are zero, round to even (based on bit 16)
	rounding := uint32(0x7FFF) + ((bits >> 16) & 1)
	bits += rounding

	return BFloat16(bits >> 16)
}

// IsNaN returns true if b is a NaN value.
func (b BFloat16) IsNaN() bool {
	exp := (b >> 7) & 0xFF
	mant := b & 0x7F
	return exp == 0xFF && mant != 0
}

// IsInf returns true if b is positive or negative infinity.
func (b BFloat16) IsInf() bool {
	exp := (b >> 7) & 0xFF
	mant := b & 0x7F
	return exp == 0xFF && mant == 0
}

// IsZero returns true if b is positive or negative zero.
func (b BFloat16) IsZero() bool {
	return b&0x7FFF == 0
}

// IsNegative returns true if the sign bit is set.
func (b BFloat16) IsNegative() bool {
	return b&0x8000 != 0
}

// IsDenormal returns true if b is a denormalized number.
func (b BFloat16) IsDenormal() bool {
	exp := (b >> 7) & 0xFF
	mant := b & 0x7F
	return exp == 0 && mant != 0
}

// Float32 converts this BFloat16 to float32.
func (b BFloat16) Float32() float32 {
	return BFloat16ToFloat32(b)
}

// Float64 converts this BFloat16 to float64.
func (b BFloat16) Float64() float64 {
	return float64(BFloat16ToFloat32(b))
}

// NewBFloat16 creates a BFloat16 from a float32 value.
func NewBFloat16(f float32) BFloat16 {
	return Float32ToBFloat16(f)
}

// NewBFloat16FromFloat64 creates a BFloat16 from a float64 value.
func NewBFloat16FromFloat64(f float64) BFloat16 {
	return Float32ToBFloat16(float32(f))
}

// Bits returns the raw uint16 representation.
func (b BFloat16) Bits() uint16 {
	return uint16(b)
}

// BFloat16FromBits creates a BFloat16 from raw bits.
func BFloat16FromBits(bits uint16) BFloat16 {
	return BFloat16(bits)
}

// Float16ToBFloat16 converts a Float16 to BFloat16.
// This involves a full conversion through float32.
func Float16ToBFloat16(h Float16) BFloat16 {
	return Float32ToBFloat16(Float16ToFloat32(h))
}

// BFloat16ToFloat16 converts a BFloat16 to Float16.
// This involves a full conversion through float32.
// Note: May lose precision or overflow since Float16 has smaller range.
func BFloat16ToFloat16(b BFloat16) Float16 {
	return Float32ToFloat16(BFloat16ToFloat32(b))
}
