// Package algo provides algorithm utilities for SIMD operations.
// This package corresponds to Google Highway's hwy/contrib/algo directory.
//
// # Transform API
//
// The Transform functions apply operations to entire slices in a zero-allocation,
// batched manner similar to C++ Highway's std::transform.
//
// Generic transform functions:
//   - Transform32(input, output []float32, simdFunc VecFunc32, scalarFunc ScalarFunc32)
//   - Transform64(input, output []float64, simdFunc VecFunc64, scalarFunc ScalarFunc64)
//
// Named transforms for common math functions:
//   - ExpTransform, ExpTransform64
//   - LogTransform, LogTransform64
//   - SinTransform, SinTransform64
//   - CosTransform, CosTransform64
//   - TanhTransform, TanhTransform64
//   - SigmoidTransform, SigmoidTransform64
//   - ErfTransform, ErfTransform64
//
// # Example Usage
//
//	import "github.com/ajroetker/go-highway/hwy/contrib/algo"
//
//	// Using named transforms
//	func ProcessData(input []float32) []float32 {
//	    output := make([]float32, len(input))
//	    algo.ExpTransform(input, output)
//	    return output
//	}
//
//	// Using generic transform with custom operation
//	func CustomOp(input []float32) []float32 {
//	    output := make([]float32, len(input))
//	    algo.Transform32(input, output,
//	        func(x archsimd.Float32x8) archsimd.Float32x8 {
//	            return x.Mul(x).Add(x)  // x² + x
//	        },
//	        func(x float32) float32 { return x*x + x },
//	    )
//	    return output
//	}
//
// # Build Requirements
//
// The SIMD implementations require:
//   - GOEXPERIMENT=simd build flag
//   - AMD64 architecture with AVX2 support
//
// On non-SIMD builds, the transform functions fall back to scalar implementations.
package algo
