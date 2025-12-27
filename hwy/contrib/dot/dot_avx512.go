//go:build amd64 && goexperiment.simd

package dot

import (
	"simd/archsimd"
)

// Dot_AVX512_F32x16 computes the dot product of two float32 vectors using AVX-512.
// This is the low-level SIMD primitive that processes 16 elements at a time.
//
// The caller is responsible for handling tail elements.
func Dot_AVX512_F32x16(a, b []float32) float32 {
	n := min(len(a), len(b))
	sum := archsimd.BroadcastFloat32x16(0.0)

	// Process 16 float32s at a time
	for i := 0; i+16 <= n; i += 16 {
		va := archsimd.LoadFloat32x16Slice(a[i:])
		vb := archsimd.LoadFloat32x16Slice(b[i:])
		sum = sum.Add(va.Mul(vb))
	}

	// Horizontal reduction: sum all 16 lanes
	// Store vector to temp array and sum elements
	var temp [16]float32
	sum.StoreSlice(temp[:])
	result := temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7] +
		temp[8] + temp[9] + temp[10] + temp[11] + temp[12] + temp[13] + temp[14] + temp[15]

	// Handle tail elements with scalar code
	for i := (n / 16) * 16; i < n; i++ {
		result += a[i] * b[i]
	}

	return result
}

// Dot_AVX512_F64x8 computes the dot product of two float64 vectors using AVX-512.
// This is the low-level SIMD primitive that processes 8 elements at a time.
func Dot_AVX512_F64x8(a, b []float64) float64 {
	n := min(len(a), len(b))
	sum := archsimd.BroadcastFloat64x8(0.0)

	// Process 8 float64s at a time
	for i := 0; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat64x8Slice(a[i:])
		vb := archsimd.LoadFloat64x8Slice(b[i:])
		sum = sum.Add(va.Mul(vb))
	}

	// Horizontal reduction: sum all 8 lanes
	// Store vector to temp array and sum elements
	var temp [8]float64
	sum.StoreSlice(temp[:])
	result := temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7]

	// Handle tail elements with scalar code
	for i := (n / 8) * 8; i < n; i++ {
		result += a[i] * b[i]
	}

	return result
}
