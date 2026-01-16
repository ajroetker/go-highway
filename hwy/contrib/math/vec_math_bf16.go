package math

import "github.com/ajroetker/go-highway/hwy"

// =============================================================================
// BFloat16 Mathematical Functions (Promote-Compute-Demote Pattern)
// =============================================================================
//
// These functions provide BFloat16 math operations by:
// 1. Promoting BFloat16 inputs to float32
// 2. Computing using the optimized float32 math functions
// 3. Demoting results back to BFloat16
//
// BFloat16 has the same dynamic range as float32 (8-bit exponent) but only
// 7 bits of mantissa precision. This makes it ideal for ML training where
// range matters more than precision.
//
// The promote-compute-demote pattern ensures:
// - High numerical accuracy (float32 precision during computation)
// - Consistent results across all platforms
// - Leveraging existing optimized float32 implementations
//
// For ML workloads, the key functions are:
// - ExpBF16: Used in softmax computation
// - LogBF16: Used in cross-entropy loss
// - SigmoidBF16: Activation function
// - TanhBF16: Activation function

// =============================================================================
// Exponential and Logarithmic Functions
// =============================================================================

// ExpBF16 computes e^x for BFloat16 vectors using promote-compute-demote.
// This is critical for softmax computation in ML workloads.
func ExpBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	// Promote to float32 (trivial - just bit shift)
	xf32 := hwy.PromoteBF16ToF32(x)
	// Compute exp in float32
	result := BaseExpVec(xf32)
	// Demote back to BFloat16 (with rounding)
	return hwy.DemoteF32ToBF16(result)
}

// Exp2BF16 computes 2^x for BFloat16 vectors.
func Exp2BF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseExp2Vec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// LogBF16 computes ln(x) for BFloat16 vectors.
// This is critical for cross-entropy loss computation in ML workloads.
func LogBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseLogVec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// Log2BF16 computes log2(x) for BFloat16 vectors.
func Log2BF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseLog2Vec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// Log10BF16 computes log10(x) for BFloat16 vectors.
func Log10BF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseLog10Vec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// =============================================================================
// Activation Functions (Critical for ML)
// =============================================================================

// SigmoidBF16 computes sigmoid(x) = 1/(1+e^(-x)) for BFloat16 vectors.
// This is a critical activation function for neural networks.
func SigmoidBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseSigmoidVec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// TanhBF16 computes tanh(x) for BFloat16 vectors.
// This is a critical activation function for neural networks (RNNs, LSTMs).
func TanhBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseTanhVec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// =============================================================================
// Trigonometric Functions
// =============================================================================

// SinBF16 computes sin(x) for BFloat16 vectors.
func SinBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseSinVec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// CosBF16 computes cos(x) for BFloat16 vectors.
func CosBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseCosVec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// TanBF16 computes tan(x) for BFloat16 vectors.
// tan(x) = sin(x) / cos(x)
func TanBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	sinResult := BaseSinVec(xf32)
	cosResult := BaseCosVec(xf32)
	result := hwy.Div(sinResult, cosResult)
	return hwy.DemoteF32ToBF16(result)
}

// =============================================================================
// Hyperbolic Functions
// =============================================================================

// SinhBF16 computes sinh(x) for BFloat16 vectors.
func SinhBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseSinhVec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// CoshBF16 computes cosh(x) for BFloat16 vectors.
func CoshBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseCoshVec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// =============================================================================
// Inverse Hyperbolic Functions
// =============================================================================

// AsinhBF16 computes asinh(x) = ln(x + sqrt(x^2 + 1)) for BFloat16 vectors.
func AsinhBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseAsinhVec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// AcoshBF16 computes acosh(x) = ln(x + sqrt(x^2 - 1)) for BFloat16 vectors.
// Requires x >= 1.
func AcoshBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseAcoshVec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// AtanhBF16 computes atanh(x) = 0.5 * ln((1+x)/(1-x)) for BFloat16 vectors.
// Requires |x| < 1.
func AtanhBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseAtanhVec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// =============================================================================
// Special Functions
// =============================================================================

// ErfBF16 computes the error function erf(x) for BFloat16 vectors.
func ErfBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)
	result := BaseErfVec(xf32)
	return hwy.DemoteF32ToBF16(result)
}

// PowBF16 computes base^exp for BFloat16 vectors element-wise.
// Uses the identity: base^exp = exp(exp * log(base))
// Note: Only valid for base > 0.
func PowBF16(base, exp hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	baseF32 := hwy.PromoteBF16ToF32(base)
	expF32 := hwy.PromoteBF16ToF32(exp)
	result := BasePowVec(baseF32, expF32)
	return hwy.DemoteF32ToBF16(result)
}

// =============================================================================
// Convenience Functions
// =============================================================================

// SoftmaxBF16 computes softmax over a BFloat16 vector.
// softmax(x)_i = exp(x_i) / sum(exp(x_j))
// Uses the numerically stable form: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
func SoftmaxBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)

	// Find max for numerical stability
	maxVal := hwy.ReduceMax(xf32)
	maxVec := hwy.Const[float32](maxVal)

	// Subtract max and compute exp
	shifted := hwy.Sub(xf32, maxVec)
	expVals := BaseExpVec(shifted)

	// Sum and normalize
	sum := hwy.ReduceSum(expVals)
	sumVec := hwy.Const[float32](sum)
	result := hwy.Div(expVals, sumVec)

	return hwy.DemoteF32ToBF16(result)
}

// GeluBF16 computes GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))) for BFloat16 vectors.
// GELU is a popular activation function in transformer models.
func GeluBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)

	// Constants
	half := hwy.Const[float32](0.5)
	one := hwy.Const[float32](1.0)
	invSqrt2 := hwy.Const[float32](0.7071067811865475) // 1/sqrt(2)

	// Compute erf(x / sqrt(2))
	xScaled := hwy.Mul(xf32, invSqrt2)
	erfVal := BaseErfVec(xScaled)

	// GELU = x * 0.5 * (1 + erf(x / sqrt(2)))
	onePlusErf := hwy.Add(one, erfVal)
	halfOnePlusErf := hwy.Mul(half, onePlusErf)
	result := hwy.Mul(xf32, halfOnePlusErf)

	return hwy.DemoteF32ToBF16(result)
}

// SiluBF16 computes SiLU(x) = x * sigmoid(x) for BFloat16 vectors.
// SiLU (Sigmoid Linear Unit) is also known as Swish.
func SiluBF16(x hwy.Vec[hwy.BFloat16]) hwy.Vec[hwy.BFloat16] {
	xf32 := hwy.PromoteBF16ToF32(x)

	// Compute sigmoid(x)
	sigVal := BaseSigmoidVec(xf32)

	// SiLU = x * sigmoid(x)
	result := hwy.Mul(xf32, sigVal)

	return hwy.DemoteF32ToBF16(result)
}
