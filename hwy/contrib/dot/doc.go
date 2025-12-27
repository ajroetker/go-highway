// Package dot provides high-performance SIMD dot product operations.
// This package corresponds to Google Highway's hwy/contrib/dot directory.
//
// # Dot Product Functions
//
// The package provides vectorized dot product computations for float32 and float64 slices:
//   - Dot(a, b []float32) float32 - Single dot product for float32
//   - Dot64(a, b []float64) float64 - Single dot product for float64
//   - DotBatch(queries, keys [][]float32) []float32 - Batch dot products
//
// # Algorithm
//
// The implementation uses SIMD multiply-accumulate operations followed by
// horizontal reduction:
//   1. Process elements in chunks of 8 (float32) or 4 (float64)
//   2. Use SIMD multiply and add operations for vectorized computation
//   3. Perform horizontal sum reduction to get scalar result
//   4. Handle tail elements with scalar code
//
// # Example Usage
//
//	import "github.com/ajroetker/go-highway/hwy/contrib/dot"
//
//	// Simple dot product
//	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
//	b := []float32{8, 7, 6, 5, 4, 3, 2, 1}
//	result := dot.Dot(a, b)  // 120.0
//
//	// Batch dot products (useful for ML applications)
//	queries := [][]float32{
//	    {1, 2, 3},
//	    {4, 5, 6},
//	}
//	keys := [][]float32{
//	    {7, 8, 9},
//	    {1, 2, 3},
//	}
//	results := dot.DotBatch(queries, keys)  // [50.0, 32.0]
//
// # Performance
//
// The SIMD implementation provides significant speedups:
//   - AVX2 (8x float32): ~4-8x faster than scalar
//   - AVX2 (4x float64): ~2-4x faster than scalar
//   - AVX-512 (16x float32): ~8-16x faster than scalar
//
// Performance is best when the input size is a multiple of the vector width.
//
// # Build Requirements
//
// The SIMD implementations require:
//   - GOEXPERIMENT=simd build flag
//   - AMD64 architecture with AVX2 or AVX-512 support
package dot
