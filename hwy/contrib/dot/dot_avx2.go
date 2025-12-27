//go:build amd64 && goexperiment.simd

package dot

import (
	"simd/archsimd"
)

// Dot_AVX2_F32x8 computes the dot product of two float32 vectors using AVX2.
// This is the low-level SIMD primitive that processes 8 elements at a time.
//
// The caller is responsible for handling tail elements.
func Dot_AVX2_F32x8(a, b []float32) float32 {
	n := min(len(a), len(b))
	sum := archsimd.BroadcastFloat32x8(0.0)

	// Process 8 float32s at a time
	for i := 0; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		sum = sum.Add(va.Mul(vb))
	}

	// Horizontal reduction: sum all 8 lanes
	// Extract both halves (each is 128-bit with 4 float32s)
	var result float32
	for i := 0; i < 8; i++ {
		result += sum.ExtractLane(i)
	}

	// Handle tail elements with scalar code
	for i := (n / 8) * 8; i < n; i++ {
		result += a[i] * b[i]
	}

	return result
}

// Dot_AVX2_F64x4 computes the dot product of two float64 vectors using AVX2.
// This is the low-level SIMD primitive that processes 4 elements at a time.
func Dot_AVX2_F64x4(a, b []float64) float64 {
	n := min(len(a), len(b))
	sum := archsimd.BroadcastFloat64x4(0.0)

	// Process 4 float64s at a time
	for i := 0; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vb := archsimd.LoadFloat64x4Slice(b[i:])
		sum = sum.Add(va.Mul(vb))
	}

	// Horizontal reduction: sum all 4 lanes
	var result float64
	for i := 0; i < 4; i++ {
		result += sum.ExtractLane(i)
	}

	// Handle tail elements with scalar code
	for i := (n / 4) * 4; i < n; i++ {
		result += a[i] * b[i]
	}

	return result
}
