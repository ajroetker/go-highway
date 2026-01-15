// Package sort provides a vectorized quicksort (VQSort) implementation.
// This package corresponds to Google Highway's hwy/contrib/sort directory.
//
// VQSort uses SIMD instructions for parallel comparisons during partitioning,
// making it significantly faster than scalar sorting algorithms for large arrays.
//
// # Algorithm
//
// VQSort is an introsort variant that combines:
//   - Sorting networks for small arrays (8-32 elements depending on SIMD width)
//   - Vectorized quicksort partitioning for larger arrays
//   - Insertion sort for small subarrays
//   - Heapsort fallback to guarantee O(n log n) worst case
//
// # Supported Types
//
// The sort functions support these numeric types:
//   - float32, float64 (floating point)
//   - int32, int64 (signed integers)
//
// # Example Usage
//
//	import "github.com/ajroetker/go-highway/hwy/contrib/sort"
//
//	func ProcessData(data []float32) {
//	    sort.Sort(data)  // In-place ascending sort
//	}
//
//	func CheckSorted(data []float32) bool {
//	    return sort.IsSorted(data)
//	}
//
// # Performance
//
// VQSort typically achieves 2-4x speedup over standard library sort for:
//   - Random data (best case for quicksort)
//   - Large arrays (>1000 elements)
//
// Performance degrades gracefully for adversarial patterns (sorted, reverse,
// all-equal) due to the heapsort fallback.
//
// # Build Requirements
//
// SIMD implementations require:
//   - GOEXPERIMENT=simd build flag
//   - AMD64 with AVX2/AVX-512, or ARM64 with NEON
//
// Without SIMD, the package falls back to optimized scalar implementations.
package sort
