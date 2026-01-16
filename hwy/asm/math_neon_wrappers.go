//go:build !noasm && arm64

package asm

// Additional transcendental math operations for ARM64 NEON
// Split into separate files to avoid GOAT parsing limits and GOAT bugs with double constants
//go:generate go tool goat ../c/math_f32_neon_arm64.c -O3 --target arm64 -e="-march=armv8-a+simd+fp" -e="-fno-builtin-memset"
//go:generate go tool goat ../c/math_f64_neon_arm64.c -O3 --target arm64 -e="-march=armv8-a+simd+fp" -e="-fno-builtin-memset"

import "unsafe"

// TanF32 computes tangent: result[i] = tan(input[i])
func TanF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	tan_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// AtanF32 computes arctangent: result[i] = atan(input[i])
func AtanF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	atan_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Atan2F32 computes 2-argument arctangent: result[i] = atan2(y[i], x[i])
func Atan2F32(y, x, result []float32) {
	if len(y) == 0 {
		return
	}
	n := int64(len(y))
	atan2_f32_neon(unsafe.Pointer(&y[0]), unsafe.Pointer(&x[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// PowF32 computes power: result[i] = base[i] ^ exp[i]
func PowF32(base, exp, result []float32) {
	if len(base) == 0 {
		return
	}
	n := int64(len(base))
	pow_f32_neon(unsafe.Pointer(&base[0]), unsafe.Pointer(&exp[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ErfF32 computes error function: result[i] = erf(input[i])
func ErfF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	erf_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Exp2F32 computes 2^x: result[i] = 2^input[i]
func Exp2F32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	exp2_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Exp2F64 computes 2^x: result[i] = 2^input[i]
func Exp2F64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	exp2_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Log2F32 computes log base 2: result[i] = log2(input[i])
func Log2F32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	log2_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Log2F64 computes log base 2: result[i] = log2(input[i])
func Log2F64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	log2_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Float64 Transcendental Operations

// ExpF64 computes exponential: result[i] = exp(input[i])
func ExpF64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	exp_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// LogF64 computes natural logarithm: result[i] = log(input[i])
func LogF64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	log_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SinF64 computes sine: result[i] = sin(input[i])
func SinF64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	sin_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// CosF64 computes cosine: result[i] = cos(input[i])
func CosF64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	cos_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// TanhF64 computes hyperbolic tangent: result[i] = tanh(input[i])
func TanhF64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	tanh_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SigmoidF64 computes sigmoid: result[i] = 1 / (1 + exp(-input[i]))
func SigmoidF64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	sigmoid_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Log10F32 computes log base 10: result[i] = log10(input[i])
func Log10F32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	log10_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Exp10F32 computes 10^x: result[i] = 10^input[i]
func Exp10F32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	exp10_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SinCosF32 computes both sin and cos: sin_result[i] = sin(input[i]), cos_result[i] = cos(input[i])
func SinCosF32(input, sinResult, cosResult []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	sincos_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&sinResult[0]), unsafe.Pointer(&cosResult[0]), unsafe.Pointer(&n))
}

// Float64 Additional Transcendental Operations

// TanF64 computes tangent: result[i] = tan(input[i])
func TanF64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	tan_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// AtanF64 computes arctangent: result[i] = atan(input[i])
func AtanF64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	atan_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Atan2F64 computes 2-argument arctangent: result[i] = atan2(y[i], x[i])
func Atan2F64(y, x, result []float64) {
	if len(y) == 0 {
		return
	}
	n := int64(len(y))
	atan2_f64_neon(unsafe.Pointer(&y[0]), unsafe.Pointer(&x[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// PowF64 computes power: result[i] = base[i] ^ exp[i]
func PowF64(base, exp, result []float64) {
	if len(base) == 0 {
		return
	}
	n := int64(len(base))
	pow_f64_neon(unsafe.Pointer(&base[0]), unsafe.Pointer(&exp[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ErfF64 computes error function: result[i] = erf(input[i])
func ErfF64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	erf_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Log10F64 computes log base 10: result[i] = log10(input[i])
func Log10F64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	log10_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// Exp10F64 computes 10^x: result[i] = 10^input[i]
func Exp10F64(input, result []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	exp10_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SinCosF64 computes both sin and cos: sin_result[i] = sin(input[i]), cos_result[i] = cos(input[i])
func SinCosF64(input, sinResult, cosResult []float64) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	sincos_f64_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&sinResult[0]), unsafe.Pointer(&cosResult[0]), unsafe.Pointer(&n))
}

// ExpBulkF32 computes e^x for entire arrays in a single assembly call.
// This is significantly faster than per-element processing for large arrays.
func ExpBulkF32(input, result []float32) {
	if len(input) == 0 {
		return
	}
	n := int64(len(input))
	exp_bulk_f32_neon(unsafe.Pointer(&input[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}
