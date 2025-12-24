//go:build amd64 && goexperiment.simd

package contrib

import (
	"simd/archsimd"

	"github.com/go-highway/highway/hwy"
)

func init() {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		Sin32 = sin32AVX2
		Sin64 = sin64AVX2
		Cos32 = cos32AVX2
		Cos64 = cos64AVX2
		SinCos32 = sinCos32AVX2
		SinCos64 = sinCos64AVX2
	}
}

// sin32AVX2 computes sin(x) for float32 values using AVX2 SIMD.
func sin32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Sin_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		result[i] = sin32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Sin_AVX2_F32x8 computes sin(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Sin_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	var values [8]float32
	x.StoreSlice(values[:])

	for i := 0; i < 8; i++ {
		values[i] = sin32Scalar(values[i])
	}

	return archsimd.LoadFloat32x8Slice(values[:])
}

func sin32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := sin32Base(v)
	return result.Data()[0]
}

// sin64AVX2 computes sin(x) for float64 values using AVX2 SIMD.
func sin64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Sin_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		result[i] = sin64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Sin_AVX2_F64x4 computes sin(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Sin_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	var values [4]float64
	x.StoreSlice(values[:])

	for i := 0; i < 4; i++ {
		values[i] = sin64Scalar(values[i])
	}

	return archsimd.LoadFloat64x4Slice(values[:])
}

func sin64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := sin64Base(v)
	return result.Data()[0]
}

// cos32AVX2 computes cos(x) for float32 values using AVX2 SIMD.
func cos32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		out := Cos_AVX2_F32x8(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		result[i] = cos32Scalar(data[i])
	}

	return hwy.Load(result)
}

// Cos_AVX2_F32x8 computes cos(x) for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Cos_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	var values [8]float32
	x.StoreSlice(values[:])

	for i := 0; i < 8; i++ {
		values[i] = cos32Scalar(values[i])
	}

	return archsimd.LoadFloat32x8Slice(values[:])
}

func cos32Scalar(x float32) float32 {
	v := hwy.Load([]float32{x})
	result := cos32Base(v)
	return result.Data()[0]
}

// cos64AVX2 computes cos(x) for float64 values using AVX2 SIMD.
func cos64AVX2(v hwy.Vec[float64]) hwy.Vec[float64] {
	data := v.Data()
	n := v.NumLanes()

	result := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		out := Cos_AVX2_F64x4(x)
		out.StoreSlice(result[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		result[i] = cos64Scalar(data[i])
	}

	return hwy.Load(result)
}

// Cos_AVX2_F64x4 computes cos(x) for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func Cos_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	var values [4]float64
	x.StoreSlice(values[:])

	for i := 0; i < 4; i++ {
		values[i] = cos64Scalar(values[i])
	}

	return archsimd.LoadFloat64x4Slice(values[:])
}

func cos64Scalar(x float64) float64 {
	v := hwy.Load([]float64{x})
	result := cos64Base(v)
	return result.Data()[0]
}

// sinCos32AVX2 computes both sin and cos for float32 values.
func sinCos32AVX2(v hwy.Vec[float32]) (sin, cos hwy.Vec[float32]) {
	data := v.Data()
	n := v.NumLanes()

	sinResult := make([]float32, n)
	cosResult := make([]float32, n)

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(data[i:])
		s, c := SinCos_AVX2_F32x8(x)
		s.StoreSlice(sinResult[i:])
		c.StoreSlice(cosResult[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		sinResult[i] = sin32Scalar(data[i])
		cosResult[i] = cos32Scalar(data[i])
	}

	return hwy.Load(sinResult), hwy.Load(cosResult)
}

// SinCos_AVX2_F32x8 computes both sin and cos for a single Float32x8 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func SinCos_AVX2_F32x8(x archsimd.Float32x8) (sin, cos archsimd.Float32x8) {
	return Sin_AVX2_F32x8(x), Cos_AVX2_F32x8(x)
}

// sinCos64AVX2 computes both sin and cos for float64 values.
func sinCos64AVX2(v hwy.Vec[float64]) (sin, cos hwy.Vec[float64]) {
	data := v.Data()
	n := v.NumLanes()

	sinResult := make([]float64, n)
	cosResult := make([]float64, n)

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(data[i:])
		s, c := SinCos_AVX2_F64x4(x)
		s.StoreSlice(sinResult[i:])
		c.StoreSlice(cosResult[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		sinResult[i] = sin64Scalar(data[i])
		cosResult[i] = cos64Scalar(data[i])
	}

	return hwy.Load(sinResult), hwy.Load(cosResult)
}

// SinCos_AVX2_F64x4 computes both sin and cos for a single Float64x4 vector.
// This is the exported function that hwygen-generated code can call directly.
// TODO: Implement optimized AVX2 version - currently uses scalar fallback.
func SinCos_AVX2_F64x4(x archsimd.Float64x4) (sin, cos archsimd.Float64x4) {
	return Sin_AVX2_F64x4(x), Cos_AVX2_F64x4(x)
}
