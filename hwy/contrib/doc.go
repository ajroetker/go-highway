// Package contrib provides high-performance SIMD operations.
// This package has been restructured to align with Google Highway's C++ organization.
//
// # Subpackages
//
// The contrib package is organized into subdirectories:
//
//   - algo: Transform utilities for applying operations to slices
//   - math: Transcendental math functions (exp, log, sin, cos, sqrt, sinh, cosh, etc.) and polynomial utilities
//   - dot: Dot product operations for ML applications
//   - matvec: Matrix-vector multiplication
//
// # Algorithm Utilities (hwy/contrib/algo)
//
// The algo package provides transform functions for applying operations to entire slices:
//
//	import "github.com/ajroetker/go-highway/hwy/contrib/algo"
//
//	// Named transforms
//	algo.ExpTransform(input, output)    // Apply exp(x) to all elements
//	algo.LogTransform(input, output)    // Apply ln(x) to all elements
//	algo.SinTransform(input, output)    // Apply sin(x) to all elements
//
//	// Generic transform with custom operation
//	algo.Transform32(input, output,
//	    func(x archsimd.Float32x8) archsimd.Float32x8 {
//	        return x.Mul(x).Add(x)  // x² + x
//	    },
//	    func(x float32) float32 { return x*x + x },
//	)
//
// # Math Functions (hwy/contrib/math)
//
// The math package provides low-level SIMD transcendental functions:
//
//	import "github.com/ajroetker/go-highway/hwy/contrib/math"
//
//	// Direct SIMD vector operations
//	expX := math.Exp_AVX2_F32x8(x)
//	logX := math.Log_AVX2_F32x8(x)
//	sinX, cosX := math.SinCos_AVX2_F32x8(x)
//
//	// Polynomial evaluation
//	coeffs := []float32{1.0, 2.0, 3.0}
//	result := math.Horner(x, coeffs)  // Evaluate 1 + 2x + 3x²
//
// # Migration Guide
//
// If you were using the old flat contrib package:
//
//	Old: import "github.com/ajroetker/go-highway/hwy/contrib"
//	     contrib.ExpTransform(...)
//	     contrib.Exp_AVX2_F32x8(...)
//
//	New: import "github.com/ajroetker/go-highway/hwy/contrib/algo"
//	     import "github.com/ajroetker/go-highway/hwy/contrib/math"
//	     algo.ExpTransform(...)
//	     math.Exp_AVX2_F32x8(...)
//
// # Build Requirements
//
// The SIMD implementations require:
//   - GOEXPERIMENT=simd build flag
//   - AMD64 architecture with AVX2 or AVX-512 support
//
// See subpackage documentation for detailed API information.
package contrib
