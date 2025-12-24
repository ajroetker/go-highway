package contrib

import "github.com/go-highway/highway/hwy"

// Log32 computes natural logarithm ln(x) for each lane of a float32 vector.
// Maximum error: ~1 ULP for positive inputs
//
// Special cases:
//   - Log(+Inf) = +Inf
//   - Log(0) = -Inf
//   - Log(x) = NaN for x < 0
//   - Log(NaN) = NaN
//
// This function variable is initialized by the base implementation
// and may be replaced by optimized SIMD versions at init time.
var Log32 func(v hwy.Vec[float32]) hwy.Vec[float32]

// Log64 computes natural logarithm ln(x) for each lane of a float64 vector.
// Maximum error: ~1 ULP for positive inputs
var Log64 func(v hwy.Vec[float64]) hwy.Vec[float64]

// Log computes natural logarithm ln(x) for each lane of the input vector.
// This is a generic wrapper that dispatches to the appropriate
// type-specific implementation (Log32 or Log64).
//
// Example:
//
//	data := []float32{1, math.E, 10, 100}
//	v := hwy.Load(data)
//	result := contrib.Log(v)  // computes [0, 1, ln(10), ln(100)]
func Log[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	var dummy T
	switch any(dummy).(type) {
	case float32:
		return any(Log32(any(v).(hwy.Vec[float32]))).(hwy.Vec[T])
	case float64:
		return any(Log64(any(v).(hwy.Vec[float64]))).(hwy.Vec[T])
	default:
		panic("unsupported float type")
	}
}

// Log_AVX2 is the AVX2-specific implementation selector for hwygen.
func Log_AVX2[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Log(v)
}

// Log_Fallback is the scalar fallback implementation selector for hwygen.
func Log_Fallback[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	return Log(v)
}
