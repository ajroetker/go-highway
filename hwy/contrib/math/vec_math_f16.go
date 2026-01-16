package math

import "github.com/ajroetker/go-highway/hwy"

// =============================================================================
// Float16 Mathematical Functions (Promote-Compute-Demote Pattern)
// =============================================================================
//
// These functions provide Float16 math operations by:
// 1. Promoting Float16 inputs to float32
// 2. Computing using the optimized float32 math functions
// 3. Demoting results back to Float16
//
// This approach ensures:
// - High numerical accuracy (float32 precision during computation)
// - Consistent results across all platforms
// - Leveraging existing optimized float32 implementations
//
// For ML workloads, the key functions are:
// - ExpF16: Used in softmax computation
// - LogF16: Used in cross-entropy loss
// - SigmoidF16: Activation function
// - TanhF16: Activation function

// =============================================================================
// Exponential and Logarithmic Functions
// =============================================================================

// ExpF16 computes e^x for Float16 vectors using promote-compute-demote.
// This is critical for softmax computation in ML workloads.
func ExpF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	// Promote to float32
	xf32 := hwy.PromoteF16ToF32(x)
	// Compute exp in float32
	result := BaseExpVec(xf32)
	// Demote back to Float16
	return hwy.DemoteF32ToF16(result)
}

// Exp2F16 computes 2^x for Float16 vectors.
func Exp2F16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseExp2Vec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// LogF16 computes ln(x) for Float16 vectors.
// This is critical for cross-entropy loss computation in ML workloads.
func LogF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseLogVec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// Log2F16 computes log2(x) for Float16 vectors.
func Log2F16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseLog2Vec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// Log10F16 computes log10(x) for Float16 vectors.
func Log10F16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseLog10Vec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// =============================================================================
// Activation Functions (Critical for ML)
// =============================================================================

// SigmoidF16 computes sigmoid(x) = 1/(1+e^(-x)) for Float16 vectors.
// This is a critical activation function for neural networks.
func SigmoidF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseSigmoidVec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// TanhF16 computes tanh(x) for Float16 vectors.
// This is a critical activation function for neural networks (RNNs, LSTMs).
func TanhF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseTanhVec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// =============================================================================
// Trigonometric Functions
// =============================================================================

// SinF16 computes sin(x) for Float16 vectors.
func SinF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseSinVec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// CosF16 computes cos(x) for Float16 vectors.
func CosF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseCosVec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// TanF16 computes tan(x) for Float16 vectors.
// tan(x) = sin(x) / cos(x)
func TanF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	sinResult := BaseSinVec(xf32)
	cosResult := BaseCosVec(xf32)
	result := hwy.Div(sinResult, cosResult)
	return hwy.DemoteF32ToF16(result)
}

// =============================================================================
// Hyperbolic Functions
// =============================================================================

// SinhF16 computes sinh(x) for Float16 vectors.
func SinhF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseSinhVec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// CoshF16 computes cosh(x) for Float16 vectors.
func CoshF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseCoshVec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// =============================================================================
// Inverse Hyperbolic Functions
// =============================================================================

// AsinhF16 computes asinh(x) = ln(x + sqrt(x^2 + 1)) for Float16 vectors.
func AsinhF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseAsinhVec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// AcoshF16 computes acosh(x) = ln(x + sqrt(x^2 - 1)) for Float16 vectors.
// Requires x >= 1.
func AcoshF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseAcoshVec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// AtanhF16 computes atanh(x) = 0.5 * ln((1+x)/(1-x)) for Float16 vectors.
// Requires |x| < 1.
func AtanhF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseAtanhVec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// =============================================================================
// Special Functions
// =============================================================================

// ErfF16 computes the error function erf(x) for Float16 vectors.
func ErfF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)
	result := BaseErfVec(xf32)
	return hwy.DemoteF32ToF16(result)
}

// PowF16 computes base^exp for Float16 vectors element-wise.
// Uses the identity: base^exp = exp(exp * log(base))
// Note: Only valid for base > 0.
func PowF16(base, exp hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	baseF32 := hwy.PromoteF16ToF32(base)
	expF32 := hwy.PromoteF16ToF32(exp)
	result := BasePowVec(baseF32, expF32)
	return hwy.DemoteF32ToF16(result)
}

// =============================================================================
// Convenience Functions
// =============================================================================

// SoftmaxF16 computes softmax over a Float16 vector.
// softmax(x)_i = exp(x_i) / sum(exp(x_j))
// Uses the numerically stable form: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
func SoftmaxF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)

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

	return hwy.DemoteF32ToF16(result)
}

// GeluF16 computes GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))) for Float16 vectors.
// GELU is a popular activation function in transformer models.
func GeluF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)

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

	return hwy.DemoteF32ToF16(result)
}

// SiluF16 computes SiLU(x) = x * sigmoid(x) for Float16 vectors.
// SiLU (Sigmoid Linear Unit) is also known as Swish.
func SiluF16(x hwy.Vec[hwy.Float16]) hwy.Vec[hwy.Float16] {
	xf32 := hwy.PromoteF16ToF32(x)

	// Compute sigmoid(x)
	sigVal := BaseSigmoidVec(xf32)

	// SiLU = x * sigmoid(x)
	result := hwy.Mul(xf32, sigVal)

	return hwy.DemoteF32ToF16(result)
}
