// Package contrib provides high-performance SIMD mathematical operations.
// These functions are implemented using polynomial approximations optimized
// for vectorized execution with zero allocation overhead.
//
// The package provides three levels of API:
//
// # Named Transform API (Recommended for common operations)
//
// High-level slice-based transforms for standard math functions.
// Zero allocations, processes data in batches:
//
//   - ExpTransform(input, output []float32)
//   - LogTransform(input, output []float32)
//   - SinTransform(input, output []float32)
//   - CosTransform(input, output []float32)
//   - TanhTransform(input, output []float32)
//   - SigmoidTransform(input, output []float32)
//   - ErfTransform(input, output []float32)
//
// Float64 variants are also available (e.g., ExpTransform64).
//
// # Generic Transform API (For custom operations)
//
// Apply custom SIMD operations to slices, similar to C++ Highway's std::transform:
//
//	// Transform32 signature:
//	func Transform32(input, output []float32, simd VecFunc32, scalar ScalarFunc32)
//
//	// Example: Apply x² + x to all elements
//	Transform32(input, output,
//	    func(x archsimd.Float32x8) archsimd.Float32x8 {
//	        return x.Mul(x).Add(x)
//	    },
//	    func(x float32) float32 { return x*x + x },
//	)
//
// The simd function processes 8 float32s (or 4 float64s) at a time.
// The scalar function handles tail elements that don't fill a vector.
//
// # Low-Level SIMD API
//
// Direct SIMD vector functions for library authors building custom operations.
// These work with archsimd vector types and require GOEXPERIMENT=simd:
//
//   - Exp_AVX2_F32x8(x Float32x8) Float32x8
//   - Log_AVX2_F32x8(x Float32x8) Float32x8
//   - Sin_AVX2_F32x8(x Float32x8) Float32x8
//   - Cos_AVX2_F32x8(x Float32x8) Float32x8
//   - SinCos_AVX2_F32x8(x Float32x8) (sin, cos Float32x8)
//   - Tanh_AVX2_F32x8(x Float32x8) Float32x8
//   - Sigmoid_AVX2_F32x8(x Float32x8) Float32x8
//   - Erf_AVX2_F32x8(x Float32x8) Float32x8
//
// Float64x4 variants are also available (e.g., Exp_AVX2_F64x4).
//
// # Accuracy
//
// All functions are designed to provide reasonable accuracy for typical
// machine learning and signal processing applications:
//   - Maximum error: ~4 ULP for most functions
//   - Special value handling: ±Inf, NaN, denormals
//
// # Example Usage
//
//	import "github.com/ajroetker/go-highway/hwy/contrib"
//
//	// Using named transforms
//	func ProcessData(input []float32) []float32 {
//	    output := make([]float32, len(input))
//	    contrib.ExpTransform(input, output)
//	    return output
//	}
//
//	// Using generic transform with custom operation
//	func CustomOp(input []float32) []float32 {
//	    output := make([]float32, len(input))
//	    contrib.Transform32(input, output,
//	        func(x archsimd.Float32x8) archsimd.Float32x8 {
//	            return contrib.Exp_AVX2_F32x8(x).Mul(x)  // x * exp(x)
//	        },
//	        func(x float32) float32 {
//	            return x * float32(math.Exp(float64(x)))
//	        },
//	    )
//	    return output
//	}
//
// # Build Requirements
//
// The SIMD implementations require:
//   - GOEXPERIMENT=simd build flag
//   - AMD64 architecture with AVX2 support
package contrib
