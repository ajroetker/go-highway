//go:build amd64 && goexperiment.simd

package dot

import "simd/archsimd"

// dotImpl32 is the SIMD implementation for float32.
// Uses AVX-512 if available, otherwise falls back to AVX2.
func dotImpl32(a, b []float32) float32 {
	if archsimd.X86.AVX512() {
		return Dot_AVX512_F32x16(a, b)
	}
	return Dot_AVX2_F32x8(a, b)
}

// dotImpl64 is the SIMD implementation for float64.
// Uses AVX-512 if available, otherwise falls back to AVX2.
func dotImpl64(a, b []float64) float64 {
	if archsimd.X86.AVX512() {
		return Dot_AVX512_F64x8(a, b)
	}
	return Dot_AVX2_F64x4(a, b)
}
