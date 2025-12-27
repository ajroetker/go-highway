//go:build amd64 && goexperiment.simd

package algo

import "simd/archsimd"

// This file provides zero-allocation Transform functions using AVX-512 SIMD.
// These functions process slices directly without the hwy.Vec abstraction overhead.

// Function types for AVX-512 Transform operations.
type (
	// VecFunc32x16 is a SIMD operation on 16 float32 values.
	VecFunc32x16 func(archsimd.Float32x16) archsimd.Float32x16

	// VecFunc64x8 is a SIMD operation on 8 float64 values.
	VecFunc64x8 func(archsimd.Float64x8) archsimd.Float64x8
)

// Transform32x16 applies a SIMD operation using AVX-512 (16 float32s at a time).
//
// The simd function processes 16 elements at a time using AVX-512.
// The scalar function handles remaining elements (0-15) that don't fill a full vector.
//
// Example usage:
//
//	// Apply x² + x to all elements
//	Transform32x16(input, output,
//	    func(x archsimd.Float32x16) archsimd.Float32x16 {
//	        return x.Mul(x).Add(x)
//	    },
//	    func(x float32) float32 { return x*x + x },
//	)
func Transform32x16(input, output []float32, simd VecFunc32x16, scalar ScalarFunc32) {
	n := min(len(input), len(output))

	// Process 16 float32s at a time
	for i := 0; i+16 <= n; i += 16 {
		x := archsimd.LoadFloat32x16Slice(input[i:])
		simd(x).StoreSlice(output[i:])
	}

	// Scalar tail (0-15 elements)
	for i := (n / 16) * 16; i < n; i++ {
		output[i] = scalar(input[i])
	}
}

// Transform64x8 applies a SIMD operation using AVX-512 (8 float64s at a time).
//
// The simd function processes 8 elements at a time using AVX-512.
// The scalar function handles remaining elements (0-7) that don't fill a full vector.
func Transform64x8(input, output []float64, simd VecFunc64x8, scalar ScalarFunc64) {
	n := min(len(input), len(output))

	// Process 8 float64s at a time
	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat64x8Slice(input[i:])
		simd(x).StoreSlice(output[i:])
	}

	// Scalar tail (0-7 elements)
	for i := (n / 8) * 8; i < n; i++ {
		output[i] = scalar(input[i])
	}
}
