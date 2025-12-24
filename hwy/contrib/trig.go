package contrib

import "github.com/go-highway/highway/hwy"

// Sin32 computes sine for each lane of a float32 vector (input in radians).
// Maximum error: ~2 ULP
//
// Special cases:
//   - Sin(±0) = ±0
//   - Sin(±Inf) = NaN
//   - Sin(NaN) = NaN
var Sin32 func(v hwy.Vec[float32]) hwy.Vec[float32]

// Sin64 computes sine for each lane of a float64 vector (input in radians).
var Sin64 func(v hwy.Vec[float64]) hwy.Vec[float64]

// Sin computes sine for each lane of the input vector.
//
// Example:
//
//	data := []float32{0, math.Pi/2, math.Pi, 3*math.Pi/2}
//	v := hwy.Load(data)
//	result := contrib.Sin(v)  // computes [0, 1, 0, -1]
func Sin[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	var dummy T
	switch any(dummy).(type) {
	case float32:
		return any(Sin32(any(v).(hwy.Vec[float32]))).(hwy.Vec[T])
	case float64:
		return any(Sin64(any(v).(hwy.Vec[float64]))).(hwy.Vec[T])
	default:
		panic("unsupported float type")
	}
}

// Cos32 computes cosine for each lane of a float32 vector (input in radians).
// Maximum error: ~2 ULP
//
// Special cases:
//   - Cos(±Inf) = NaN
//   - Cos(NaN) = NaN
var Cos32 func(v hwy.Vec[float32]) hwy.Vec[float32]

// Cos64 computes cosine for each lane of a float64 vector (input in radians).
var Cos64 func(v hwy.Vec[float64]) hwy.Vec[float64]

// Cos computes cosine for each lane of the input vector.
//
// Example:
//
//	data := []float32{0, math.Pi/2, math.Pi, 3*math.Pi/2}
//	v := hwy.Load(data)
//	result := contrib.Cos(v)  // computes [1, 0, -1, 0]
func Cos[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	var dummy T
	switch any(dummy).(type) {
	case float32:
		return any(Cos32(any(v).(hwy.Vec[float32]))).(hwy.Vec[T])
	case float64:
		return any(Cos64(any(v).(hwy.Vec[float64]))).(hwy.Vec[T])
	default:
		panic("unsupported float type")
	}
}

// SinCos32 computes both sine and cosine for each lane of a float32 vector.
// This is more efficient than calling Sin32 and Cos32 separately since
// they share the same range reduction step.
var SinCos32 func(v hwy.Vec[float32]) (sin, cos hwy.Vec[float32])

// SinCos64 computes both sine and cosine for each lane of a float64 vector.
var SinCos64 func(v hwy.Vec[float64]) (sin, cos hwy.Vec[float64])

// SinCos computes both sine and cosine for each lane of the input vector.
//
// Example:
//
//	data := []float32{0, math.Pi/4, math.Pi/2}
//	v := hwy.Load(data)
//	sin, cos := contrib.SinCos(v)
func SinCos[T hwy.Floats](v hwy.Vec[T]) (sin, cos hwy.Vec[T]) {
	var dummy T
	switch any(dummy).(type) {
	case float32:
		s, c := SinCos32(any(v).(hwy.Vec[float32]))
		return any(s).(hwy.Vec[T]), any(c).(hwy.Vec[T])
	case float64:
		s, c := SinCos64(any(v).(hwy.Vec[float64]))
		return any(s).(hwy.Vec[T]), any(c).(hwy.Vec[T])
	default:
		panic("unsupported float type")
	}
}

// Sin_AVX2 is the AVX2-specific implementation selector for hwygen.
func Sin_AVX2[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Sin(v)
}

// Sin_Fallback is the scalar fallback implementation selector for hwygen.
func Sin_Fallback[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Sin(v)
}

// Cos_AVX2 is the AVX2-specific implementation selector for hwygen.
func Cos_AVX2[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Cos(v)
}

// Cos_Fallback is the scalar fallback implementation selector for hwygen.
func Cos_Fallback[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Cos(v)
}
