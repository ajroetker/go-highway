//go:build amd64 && goexperiment.simd

package contrib

import (
	"simd/archsimd"

	"github.com/go-highway/highway/hwy"
)

func init() {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		Log32 = log32AVX2
		Log64 = log64AVX2
	}
}

// log32AVX2 computes ln(x) for float32 values using AVX2 SIMD.
func log32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	// Process 8 elements at a time
	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Log_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	// Handle tail with scalar fallback
	for i := (n / 8) * 8; i < n; i++ {
		result[i] = log32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Log_AVX2_F32x8 computes ln(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Log_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Scalar fallback implementation
	var values [8]float32
	x.StoreSlice(values[:])

	for i := 0; i < 8; i++ {
		values[i] = log32Scalar(values[i])
	}

	return archsimd.LoadFloat32x8Slice(values[:])
}

// log32Scalar is the scalar fallback for tail elements.
func log32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := log32Base(v)
	return result.Data()[0]
}

// log64AVX2 computes ln(x) for float64 values using AVX2 SIMD.
func log64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	// Process 4 elements at a time (Float64x4)
	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Log_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	// Handle tail with scalar fallback
	for i := (n / 4) * 4; i < n; i++ {
		result[i] = log64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Log_AVX2_F64x4 computes ln(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Log_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	// Scalar fallback implementation
	var values [4]float64
	x.StoreSlice(values[:])

	for i := 0; i < 4; i++ {
		values[i] = log64Scalar(values[i])
	}

	return archsimd.LoadFloat64x4Slice(values[:])
}

// log64Scalar is the scalar fallback for tail elements.
func log64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := log64Base(v)
	return result.Data()[0]
}
