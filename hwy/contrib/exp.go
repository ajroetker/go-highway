package contrib

import "github.com/ajroetker/go-highway/hwy"

// Exp32 computes e^x for each lane of a float32 vector.
// Maximum error: ~1 ULP for inputs in [-87, 88] (float32)
//
// Special cases:
//   - Exp(+Inf) = +Inf
//   - Exp(-Inf) = 0
//   - Exp(NaN) = NaN
//   - Exp(x) = +Inf for x > 88.72 (overflow)
//   - Exp(x) = 0 for x < -87.33 (underflow)
//
// This function variable is initialized by the base implementation
// and may be replaced by optimized SIMD versions at init time.
var Exp32 func(v hwy.Vec[float32]) hwy.Vec[float32]

// Exp64 computes e^x for each lane of a float64 vector.
// Maximum error: ~1 ULP for inputs in [-708, 709] (float64)
//
// Special cases are similar to Exp32 but with float64 overflow/underflow thresholds.
var Exp64 func(v hwy.Vec[float64]) hwy.Vec[float64]

// Exp computes e^x for each lane of the input vector.
// This is a generic wrapper that dispatches to the appropriate
// type-specific implementation (Exp32 or Exp64).
//
// Example:
//
//	data := []float32{0, 1, 2, -1}
//	v := hwy.Load(data)
//	result := contrib.Exp(v)  // computes [1, e, e², 1/e]
func Exp[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	var dummy T
	switch any(dummy).(type) {
	case float32:
		return any(Exp32(any(v).(hwy.Vec[float32]))).(hwy.Vec[T])
	case float64:
		return any(Exp64(any(v).(hwy.Vec[float64]))).(hwy.Vec[T])
	default:
		panic("unsupported float type")
	}
}

// Exp_AVX2 is the AVX2-specific implementation selector for hwygen.
// When hwygen transforms code for AVX2 targets, it will call this function.
func Exp_AVX2[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Exp(v) // Currently dispatches to runtime-selected implementation
}

// Exp_Fallback is the scalar fallback implementation selector for hwygen.
// When hwygen transforms code for scalar targets, it will call this function.
func Exp_Fallback[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Exp(v) // Currently dispatches to runtime-selected implementation
}
