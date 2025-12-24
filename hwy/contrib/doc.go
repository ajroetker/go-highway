// Package contrib provides extended mathematical operations for SIMD vectors.
// These functions are implemented using polynomial approximations optimized
// for vectorized execution.
//
// All functions have multiple implementations:
//   - Pure-Go fallback (always available)
//   - AVX2 vectorized (when GOEXPERIMENT=simd on amd64)
//   - AVX512 vectorized (future)
//
// The appropriate implementation is selected at runtime based on CPU features.
//
// # Mathematical Functions
//
// The package provides vectorized versions of common mathematical functions:
//
// Exponential and Logarithmic:
//   - Exp: computes e^x
//   - Log: computes natural logarithm ln(x)
//
// Trigonometric:
//   - Sin: computes sine (input in radians)
//   - Cos: computes cosine (input in radians)
//   - SinCos: computes both sin and cos efficiently
//
// Special Functions:
//   - Tanh: hyperbolic tangent
//   - Sigmoid: logistic function 1/(1+exp(-x))
//   - Erf: error function
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
//	import (
//	    "github.com/go-highway/highway/hwy"
//	    "github.com/go-highway/highway/hwy/contrib"
//	)
//
//	func ProcessData(data []float32) []float32 {
//	    result := make([]float32, len(data))
//	    for i := 0; i < len(data); i += hwy.MaxLanes[float32]() {
//	        v := hwy.Load(data[i:])
//	        v = contrib.Exp(v)
//	        hwy.Store(v, result[i:])
//	    }
//	    return result
//	}
package contrib
