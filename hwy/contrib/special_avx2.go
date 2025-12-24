//go:build amd64 && goexperiment.simd

package contrib

import (
	"simd/archsimd"

	"github.com/go-highway/highway/hwy"
)

func init() {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		Tanh32 = tanh32AVX2
		Tanh64 = tanh64AVX2
		Sigmoid32 = sigmoid32AVX2
		Sigmoid64 = sigmoid64AVX2
		Erf32 = erf32AVX2
		Erf64 = erf64AVX2
	}
}

// tanh32AVX2 computes tanh(x) for float32 values using AVX2 SIMD.
func tanh32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Tanh_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		result[i] = tanh32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Tanh_AVX2_F32x8 computes tanh(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Tanh_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	var values [8]float32
	x.StoreSlice(values[:])

	for i := 0; i < 8; i++ {
		values[i] = tanh32Scalar(values[i])
	}

	return archsimd.LoadFloat32x8Slice(values[:])
}

func tanh32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := tanh32Base(v)
	return result.Data()[0]
}

// tanh64AVX2 computes tanh(x) for float64 values using AVX2 SIMD.
func tanh64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Tanh_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		result[i] = tanh64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Tanh_AVX2_F64x4 computes tanh(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Tanh_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	var values [4]float64
	x.StoreSlice(values[:])

	for i := 0; i < 4; i++ {
		values[i] = tanh64Scalar(values[i])
	}

	return archsimd.LoadFloat64x4Slice(values[:])
}

func tanh64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := tanh64Base(v)
	return result.Data()[0]
}

// sigmoid32AVX2 computes sigmoid(x) for float32 values using AVX2 SIMD.
func sigmoid32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Sigmoid_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		result[i] = sigmoid32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Sigmoid_AVX2_F32x8 computes sigmoid(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Sigmoid_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	var values [8]float32
	x.StoreSlice(values[:])

	for i := 0; i < 8; i++ {
		values[i] = sigmoid32Scalar(values[i])
	}

	return archsimd.LoadFloat32x8Slice(values[:])
}

func sigmoid32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := sigmoid32Base(v)
	return result.Data()[0]
}

// sigmoid64AVX2 computes sigmoid(x) for float64 values using AVX2 SIMD.
func sigmoid64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Sigmoid_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		result[i] = sigmoid64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Sigmoid_AVX2_F64x4 computes sigmoid(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Sigmoid_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	var values [4]float64
	x.StoreSlice(values[:])

	for i := 0; i < 4; i++ {
		values[i] = sigmoid64Scalar(values[i])
	}

	return archsimd.LoadFloat64x4Slice(values[:])
}

func sigmoid64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := sigmoid64Base(v)
	return result.Data()[0]
}

// erf32AVX2 computes erf(x) for float32 values using AVX2 SIMD.
func erf32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Erf_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		result[i] = erf32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Erf_AVX2_F32x8 computes erf(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Erf_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	var values [8]float32
	x.StoreSlice(values[:])

	for i := 0; i < 8; i++ {
		values[i] = erf32Scalar(values[i])
	}

	return archsimd.LoadFloat32x8Slice(values[:])
}

func erf32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := erf32Base(v)
	return result.Data()[0]
}

// erf64AVX2 computes erf(x) for float64 values using AVX2 SIMD.
func erf64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Erf_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		result[i] = erf64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Erf_AVX2_F64x4 computes erf(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Erf_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	var values [4]float64
	x.StoreSlice(values[:])

	for i := 0; i < 4; i++ {
		values[i] = erf64Scalar(values[i])
	}

	return archsimd.LoadFloat64x4Slice(values[:])
}

func erf64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := erf64Base(v)
	return result.Data()[0]
}
