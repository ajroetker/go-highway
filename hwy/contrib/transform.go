//go:build !amd64 || !goexperiment.simd

package contrib

import "math"

// This file provides scalar fallback implementations for non-SIMD builds.
// The AVX2 implementations in transform_avx2.go override these on amd64.

// ExpTransform applies exp(x) to each element.
// Caller must ensure len(output) >= len(input).
func ExpTransform(input, output []float32) {
	expTransformScalar(input, output)
}

// ExpTransform64 applies exp(x) to each float64 element.
func ExpTransform64(input, output []float64) {
	expTransformScalar64(input, output)
}

// LogTransform applies ln(x) to each element.
func LogTransform(input, output []float32) {
	logTransformScalar(input, output)
}

// LogTransform64 applies ln(x) to each float64 element.
func LogTransform64(input, output []float64) {
	logTransformScalar64(input, output)
}

// SinTransform applies sin(x) to each element.
func SinTransform(input, output []float32) {
	sinTransformScalar(input, output)
}

// SinTransform64 applies sin(x) to each float64 element.
func SinTransform64(input, output []float64) {
	sinTransformScalar64(input, output)
}

// CosTransform applies cos(x) to each element.
func CosTransform(input, output []float32) {
	cosTransformScalar(input, output)
}

// CosTransform64 applies cos(x) to each float64 element.
func CosTransform64(input, output []float64) {
	cosTransformScalar64(input, output)
}

// TanhTransform applies tanh(x) to each element.
func TanhTransform(input, output []float32) {
	tanhTransformScalar(input, output)
}

// TanhTransform64 applies tanh(x) to each float64 element.
func TanhTransform64(input, output []float64) {
	tanhTransformScalar64(input, output)
}

// SigmoidTransform applies sigmoid(x) = 1/(1+exp(-x)) to each element.
func SigmoidTransform(input, output []float32) {
	sigmoidTransformScalar(input, output)
}

// SigmoidTransform64 applies sigmoid(x) to each float64 element.
func SigmoidTransform64(input, output []float64) {
	sigmoidTransformScalar64(input, output)
}

// ErfTransform applies erf(x) to each element.
func ErfTransform(input, output []float32) {
	erfTransformScalar(input, output)
}

// ErfTransform64 applies erf(x) to each float64 element.
func ErfTransform64(input, output []float64) {
	erfTransformScalar64(input, output)
}

// Scalar implementations

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
