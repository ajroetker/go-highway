package contrib

import "github.com/ajroetker/go-highway/hwy"

// Tanh32 computes hyperbolic tangent tanh(x) = (e^x - e^-x) / (e^x + e^-x)
// for each lane of a float32 vector.
//
// Special cases:
//   - Tanh(±0) = ±0
//   - Tanh(±Inf) = ±1
//   - Tanh(NaN) = NaN
//
// The result is always in the range [-1, 1].
var Tanh32 func(v hwy.Vec[float32]) hwy.Vec[float32]

// Tanh64 computes hyperbolic tangent for each lane of a float64 vector.
var Tanh64 func(v hwy.Vec[float64]) hwy.Vec[float64]

// Tanh computes hyperbolic tangent for each lane of the input vector.
//
// Example:
//
//	data := []float32{-2, -1, 0, 1, 2}
//	v := hwy.Load(data)
//	result := contrib.Tanh(v)  // computes tanh for each value
func Tanh[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	var dummy T
	switch any(dummy).(type) {
	case float32:
		return any(Tanh32(any(v).(hwy.Vec[float32]))).(hwy.Vec[T])
	case float64:
		return any(Tanh64(any(v).(hwy.Vec[float64]))).(hwy.Vec[T])
	default:
		panic("unsupported float type")
	}
}

// Sigmoid32 computes the logistic sigmoid function 1/(1+exp(-x))
// for each lane of a float32 vector.
//
// Special cases:
//   - Sigmoid(+Inf) = 1
//   - Sigmoid(-Inf) = 0
//   - Sigmoid(NaN) = NaN
//
// The result is always in the range (0, 1).
// This function is commonly used as an activation function in neural networks.
var Sigmoid32 func(v hwy.Vec[float32]) hwy.Vec[float32]

// Sigmoid64 computes the logistic sigmoid function for each lane of a float64 vector.
var Sigmoid64 func(v hwy.Vec[float64]) hwy.Vec[float64]

// Sigmoid computes the logistic sigmoid function for each lane of the input vector.
//
// Example:
//
//	data := []float32{-2, -1, 0, 1, 2}
//	v := hwy.Load(data)
//	result := contrib.Sigmoid(v)  // computes 1/(1+exp(-x)) for each value
func Sigmoid[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	var dummy T
	switch any(dummy).(type) {
	case float32:
		return any(Sigmoid32(any(v).(hwy.Vec[float32]))).(hwy.Vec[T])
	case float64:
		return any(Sigmoid64(any(v).(hwy.Vec[float64]))).(hwy.Vec[T])
	default:
		panic("unsupported float type")
	}
}

// Erf32 computes the error function erf(x) for each lane of a float32 vector.
//
// The error function is defined as:
//   erf(x) = (2/√π) * ∫[0,x] e^(-t²) dt
//
// Special cases:
//   - Erf(±0) = ±0
//   - Erf(±Inf) = ±1
//   - Erf(NaN) = NaN
//
// The result is always in the range [-1, 1].
var Erf32 func(v hwy.Vec[float32]) hwy.Vec[float32]

// Erf64 computes the error function for each lane of a float64 vector.
var Erf64 func(v hwy.Vec[float64]) hwy.Vec[float64]

// Erf computes the error function for each lane of the input vector.
//
// Example:
//
//	data := []float32{-2, -1, 0, 1, 2}
//	v := hwy.Load(data)
//	result := contrib.Erf(v)
func Erf[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	var dummy T
	switch any(dummy).(type) {
	case float32:
		return any(Erf32(any(v).(hwy.Vec[float32]))).(hwy.Vec[T])
	case float64:
		return any(Erf64(any(v).(hwy.Vec[float64]))).(hwy.Vec[T])
	default:
		panic("unsupported float type")
	}
}

// Tanh_AVX2 is the AVX2-specific implementation selector for hwygen.
func Tanh_AVX2[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Tanh(v)
}

// Tanh_Fallback is the scalar fallback implementation selector for hwygen.
func Tanh_Fallback[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Tanh(v)
}

// Sigmoid_AVX2 is the AVX2-specific implementation selector for hwygen.
func Sigmoid_AVX2[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Sigmoid(v)
}

// Sigmoid_Fallback is the scalar fallback implementation selector for hwygen.
func Sigmoid_Fallback[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Sigmoid(v)
}

// Erf_AVX2 is the AVX2-specific implementation selector for hwygen.
func Erf_AVX2[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Erf(v)
}

// Erf_Fallback is the scalar fallback implementation selector for hwygen.
func Erf_Fallback[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Erf(v)
}
