package hwy

import "math"

// Float16 represents an IEEE 754 half-precision (binary16) floating-point number.
// It wraps uint16 for storage but provides float semantics.
//
// Format: Sign (1 bit) | Exponent (5 bits) | Mantissa (10 bits)
//
//	S | EEEEE | MMMMMMMMMM
//
// Properties:
//   - Total bits: 16
//   - Exponent bits: 5 (bias: 15)
//   - Mantissa bits: 10
//   - Max value: 65504
//   - Min positive normal: ~6.10e-5
//   - Precision: ~3.3 decimal digits
type Float16 uint16

// Float16 constants for special values.
const (
	Float16Zero      Float16 = 0x0000 // Positive zero
	Float16NegZero   Float16 = 0x8000 // Negative zero
	Float16One       Float16 = 0x3C00 // 1.0
	Float16NegOne    Float16 = 0xBC00 // -1.0
	Float16MaxValue  Float16 = 0x7BFF // 65504 (max finite value)
	Float16MinNormal Float16 = 0x0400 // 2^-14 (~6.10e-5, smallest normal)
	Float16MinValue  Float16 = 0x0001 // Smallest denormal (~5.96e-8)
	Float16Inf       Float16 = 0x7C00 // Positive infinity
	Float16NegInf    Float16 = 0xFC00 // Negative infinity
	Float16NaN       Float16 = 0x7E00 // Quiet NaN (canonical)

	// Internal constants
	float16ExpBias     = 15
	float16ExpMask     = 0x1F
	float16MantissaBits = 10
	float16MantissaMask = 0x3FF
	float16SignMask    = 0x8000
)

// Float16ToFloat32 converts a single Float16 to float32.
// Handles all special cases: zero, denormals, infinity, NaN.
func Float16ToFloat32(h Float16) float32 {
	bits := uint32(h)
	sign := bits >> 15
	exp := (bits >> 10) & 0x1F
	mant := bits & 0x3FF

	if exp == 0 {
		if mant == 0 {
			// Zero (positive or negative)
			return math.Float32frombits(sign << 31)
		}
		// Denormalized: convert to normalized float32
		// Find the leading 1 bit in mantissa
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3FF                // Remove the implicit leading 1
		exp = uint32(int32(exp) + 127 - 15) // Rebias exponent
	} else if exp == 31 {
		// Inf or NaN
		if mant == 0 {
			// Infinity
			return math.Float32frombits((sign << 31) | 0x7F800000)
		}
		// NaN - preserve sign and some mantissa bits
		return math.Float32frombits((sign << 31) | 0x7FC00000 | (mant << 13))
	} else {
		// Normalized: rebias exponent (15 -> 127)
		exp = exp + 127 - 15
	}

	// Assemble float32 bits
	return math.Float32frombits((sign << 31) | (exp << 23) | (mant << 13))
}

// Float32ToFloat16 converts a float32 to Float16 with round-to-nearest-even.
// Handles overflow (to infinity), underflow (to zero), and special values.
func Float32ToFloat16(f float32) Float16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127 + 15 // Rebias: 127 -> 15
	mant := bits & 0x7FFFFF

	// Handle special cases
	if exp <= 0 {
		if exp < -10 {
			// Underflow to zero (too small even for denormal)
			return Float16(sign)
		}
		// Denormalized result
		// Add implicit leading 1 to mantissa
		mant = (mant | 0x800000) >> uint(1-exp)

		// Round to nearest even
		// Check if rounding bit is set and (round up if odd or has trailing bits)
		if mant&0x1000 != 0 && (mant&0x2FFF) != 0 {
			mant += 0x2000
		}
		return Float16(sign | uint16(mant>>13))
	} else if exp == 0xFF-127+15 {
		// Input was Inf or NaN
		if mant != 0 {
			// NaN - preserve quiet bit and some payload
			return Float16(sign | 0x7E00 | uint16(mant>>13))
		}
		// Infinity
		return Float16(sign | 0x7C00)
	} else if exp >= 31 {
		// Overflow to infinity
		return Float16(sign | 0x7C00)
	}

	// Normal case: round to nearest even
	// Bit 12 is the rounding bit, bits 0-11 are truncated
	if mant&0x1000 != 0 {
		// Rounding bit is set
		if mant&0x2FFF != 0 {
			// Either odd or has trailing bits - round up
			mant += 0x2000
			if mant&0x800000 != 0 {
				// Mantissa overflowed, increment exponent
				mant = 0
				exp++
				if exp >= 31 {
					// Overflow to infinity
					return Float16(sign | 0x7C00)
				}
			}
		}
	}

	return Float16(sign | uint16(exp<<10) | uint16(mant>>13))
}

// IsNaN returns true if h is a NaN value.
func (h Float16) IsNaN() bool {
	exp := (h >> 10) & 0x1F
	mant := h & 0x3FF
	return exp == 31 && mant != 0
}

// IsInf returns true if h is positive or negative infinity.
func (h Float16) IsInf() bool {
	exp := (h >> 10) & 0x1F
	mant := h & 0x3FF
	return exp == 31 && mant == 0
}

// IsZero returns true if h is positive or negative zero.
func (h Float16) IsZero() bool {
	return h&0x7FFF == 0
}

// IsNegative returns true if the sign bit is set.
func (h Float16) IsNegative() bool {
	return h&0x8000 != 0
}

// IsDenormal returns true if h is a denormalized number.
func (h Float16) IsDenormal() bool {
	exp := (h >> 10) & 0x1F
	mant := h & 0x3FF
	return exp == 0 && mant != 0
}

// Float32 converts this Float16 to float32.
func (h Float16) Float32() float32 {
	return Float16ToFloat32(h)
}

// Float64 converts this Float16 to float64.
func (h Float16) Float64() float64 {
	return float64(Float16ToFloat32(h))
}

// NewFloat16 creates a Float16 from a float32 value.
func NewFloat16(f float32) Float16 {
	return Float32ToFloat16(f)
}

// NewFloat16FromFloat64 creates a Float16 from a float64 value.
func NewFloat16FromFloat64(f float64) Float16 {
	return Float32ToFloat16(float32(f))
}

// Bits returns the raw uint16 representation.
func (h Float16) Bits() uint16 {
	return uint16(h)
}

// Float16FromBits creates a Float16 from raw bits.
func Float16FromBits(bits uint16) Float16 {
	return Float16(bits)
}
