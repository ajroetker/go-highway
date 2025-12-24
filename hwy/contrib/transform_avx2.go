//go:build amd64 && goexperiment.simd

package contrib

import (
	"math"
	"simd/archsimd"

	"github.com/ajroetker/go-highway/hwy"
)

// This file provides zero-allocation Transform functions using AVX2 SIMD.
// These functions process slices directly without the hwy.Vec abstraction overhead.

// ExpTransform applies exp(x) to each element with zero allocations.
// Caller must ensure len(output) >= len(input).
func ExpTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		expTransformAVX2(input, output)
	} else {
		expTransformScalar(input, output)
	}
}

// ExpTransform64 applies exp(x) to each float64 element with zero allocations.
func ExpTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		expTransformAVX2_64(input, output)
	} else {
		expTransformScalar64(input, output)
	}
}

// LogTransform applies ln(x) to each element with zero allocations.
func LogTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		logTransformAVX2(input, output)
	} else {
		logTransformScalar(input, output)
	}
}

// LogTransform64 applies ln(x) to each float64 element with zero allocations.
func LogTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		logTransformAVX2_64(input, output)
	} else {
		logTransformScalar64(input, output)
	}
}

// SinTransform applies sin(x) to each element with zero allocations.
func SinTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		sinTransformAVX2(input, output)
	} else {
		sinTransformScalar(input, output)
	}
}

// SinTransform64 applies sin(x) to each float64 element with zero allocations.
func SinTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		sinTransformAVX2_64(input, output)
	} else {
		sinTransformScalar64(input, output)
	}
}

// CosTransform applies cos(x) to each element with zero allocations.
func CosTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		cosTransformAVX2(input, output)
	} else {
		cosTransformScalar(input, output)
	}
}

// CosTransform64 applies cos(x) to each float64 element with zero allocations.
func CosTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		cosTransformAVX2_64(input, output)
	} else {
		cosTransformScalar64(input, output)
	}
}

// TanhTransform applies tanh(x) to each element with zero allocations.
func TanhTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		tanhTransformAVX2(input, output)
	} else {
		tanhTransformScalar(input, output)
	}
}

// TanhTransform64 applies tanh(x) to each float64 element with zero allocations.
func TanhTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		tanhTransformAVX2_64(input, output)
	} else {
		tanhTransformScalar64(input, output)
	}
}

// SigmoidTransform applies sigmoid(x) = 1/(1+exp(-x)) to each element with zero allocations.
func SigmoidTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		sigmoidTransformAVX2(input, output)
	} else {
		sigmoidTransformScalar(input, output)
	}
}

// SigmoidTransform64 applies sigmoid(x) to each float64 element with zero allocations.
func SigmoidTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		sigmoidTransformAVX2_64(input, output)
	} else {
		sigmoidTransformScalar64(input, output)
	}
}

// ErfTransform applies erf(x) to each element with zero allocations.
func ErfTransform(input, output []float32) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		erfTransformAVX2(input, output)
	} else {
		erfTransformScalar(input, output)
	}
}

// ErfTransform64 applies erf(x) to each float64 element with zero allocations.
func ErfTransform64(input, output []float64) {
	if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
		erfTransformAVX2_64(input, output)
	} else {
		erfTransformScalar64(input, output)
	}
}

// AVX2 implementations - Float32

func expTransformAVX2(input, output []float32) {
	n := min(len(input), len(output))

	// Process 8 float32s at a time
	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(input[i:])
		Exp_AVX2_F32x8(x).StoreSlice(output[i:])
	}

	// Scalar tail
	for i := (n / 8) * 8; i < n; i++ {
		output[i] = float32(math.Exp(float64(input[i])))
	}
}

func logTransformAVX2(input, output []float32) {
	n := min(len(input), len(output))

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(input[i:])
		Log_AVX2_F32x8(x).StoreSlice(output[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		output[i] = float32(math.Log(float64(input[i])))
	}
}

func sinTransformAVX2(input, output []float32) {
	n := min(len(input), len(output))

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(input[i:])
		Sin_AVX2_F32x8(x).StoreSlice(output[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		output[i] = float32(math.Sin(float64(input[i])))
	}
}

func cosTransformAVX2(input, output []float32) {
	n := min(len(input), len(output))

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(input[i:])
		Cos_AVX2_F32x8(x).StoreSlice(output[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		output[i] = float32(math.Cos(float64(input[i])))
	}
}

func tanhTransformAVX2(input, output []float32) {
	n := min(len(input), len(output))

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(input[i:])
		Tanh_AVX2_F32x8(x).StoreSlice(output[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		output[i] = float32(math.Tanh(float64(input[i])))
	}
}

func sigmoidTransformAVX2(input, output []float32) {
	n := min(len(input), len(output))

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(input[i:])
		Sigmoid_AVX2_F32x8(x).StoreSlice(output[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		output[i] = float32(1.0 / (1.0 + math.Exp(-float64(input[i]))))
	}
}

func erfTransformAVX2(input, output []float32) {
	n := min(len(input), len(output))

	for i := 0; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8Slice(input[i:])
		Erf_AVX2_F32x8(x).StoreSlice(output[i:])
	}

	for i := (n / 8) * 8; i < n; i++ {
		output[i] = float32(math.Erf(float64(input[i])))
	}
}

// AVX2 implementations - Float64

func expTransformAVX2_64(input, output []float64) {
	n := min(len(input), len(output))

	// Process 4 float64s at a time
	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(input[i:])
		Exp_AVX2_F64x4(x).StoreSlice(output[i:])
	}

	// Scalar tail
	for i := (n / 4) * 4; i < n; i++ {
		output[i] = math.Exp(input[i])
	}
}

func logTransformAVX2_64(input, output []float64) {
	n := min(len(input), len(output))

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(input[i:])
		Log_AVX2_F64x4(x).StoreSlice(output[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		output[i] = math.Log(input[i])
	}
}

func sinTransformAVX2_64(input, output []float64) {
	n := min(len(input), len(output))

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(input[i:])
		Sin_AVX2_F64x4(x).StoreSlice(output[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		output[i] = math.Sin(input[i])
	}
}

func cosTransformAVX2_64(input, output []float64) {
	n := min(len(input), len(output))

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(input[i:])
		Cos_AVX2_F64x4(x).StoreSlice(output[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		output[i] = math.Cos(input[i])
	}
}

func tanhTransformAVX2_64(input, output []float64) {
	n := min(len(input), len(output))

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(input[i:])
		Tanh_AVX2_F64x4(x).StoreSlice(output[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		output[i] = math.Tanh(input[i])
	}
}

func sigmoidTransformAVX2_64(input, output []float64) {
	n := min(len(input), len(output))

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(input[i:])
		Sigmoid_AVX2_F64x4(x).StoreSlice(output[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		output[i] = 1.0 / (1.0 + math.Exp(-input[i]))
	}
}

func erfTransformAVX2_64(input, output []float64) {
	n := min(len(input), len(output))

	for i := 0; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(input[i:])
		Erf_AVX2_F64x4(x).StoreSlice(output[i:])
	}

	for i := (n / 4) * 4; i < n; i++ {
		output[i] = math.Erf(input[i])
	}
}

// Scalar fallbacks (shared with transform.go for non-SIMD builds)

func expTransformScalar(input, output []float32) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = float32(math.Exp(float64(input[i])))
	}
}

func expTransformScalar64(input, output []float64) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = math.Exp(input[i])
	}
}

func logTransformScalar(input, output []float32) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = float32(math.Log(float64(input[i])))
	}
}

func logTransformScalar64(input, output []float64) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = math.Log(input[i])
	}
}

func sinTransformScalar(input, output []float32) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = float32(math.Sin(float64(input[i])))
	}
}

func sinTransformScalar64(input, output []float64) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = math.Sin(input[i])
	}
}

func cosTransformScalar(input, output []float32) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = float32(math.Cos(float64(input[i])))
	}
}

func cosTransformScalar64(input, output []float64) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = math.Cos(input[i])
	}
}

func tanhTransformScalar(input, output []float32) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = float32(math.Tanh(float64(input[i])))
	}
}

func tanhTransformScalar64(input, output []float64) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = math.Tanh(input[i])
	}
}

func sigmoidTransformScalar(input, output []float32) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = float32(1.0 / (1.0 + math.Exp(-float64(input[i]))))
	}
}

func sigmoidTransformScalar64(input, output []float64) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = 1.0 / (1.0 + math.Exp(-input[i]))
	}
}

func erfTransformScalar(input, output []float32) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = float32(math.Erf(float64(input[i])))
	}
}

func erfTransformScalar64(input, output []float64) {
	n := min(len(input), len(output))
	for i := 0; i < n; i++ {
		output[i] = math.Erf(input[i])
	}
}
