package vec

//go:generate go run ../../../cmd/hwygen -input reduce_base.go -output . -targets avx2,avx512,neon,fallback -dispatch reduce

import "github.com/ajroetker/go-highway/hwy"

// BaseSum computes the sum of all elements in a slice using hwy primitives.
//
// Returns 0 if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Example:
//
//	data := []float32{1, 2, 3, 4}
//	result := Sum(data)  // 1 + 2 + 3 + 4 = 10
func BaseSum[T hwy.Floats](v []T) T {
	if len(v) == 0 {
		return 0
	}

	sum := hwy.Zero[T]()
	lanes := sum.NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= len(v); i += lanes {
		va := hwy.Load(v[i:])
		sum = hwy.Add(sum, va)
	}

	// Reduce vector sum to scalar
	result := hwy.ReduceSum(sum)

	// Handle tail elements with scalar code
	for ; i < len(v); i++ {
		result += v[i]
	}

	return result
}

// BaseMin returns the minimum value in a slice using hwy primitives.
//
// Panics if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Note: For slices containing NaN values, behavior follows standard Go
// comparison semantics where NaN comparisons return false.
//
// Example:
//
//	data := []float32{3, 1, 4, 1, 5}
//	result := Min(data)  // 1
func BaseMin[T hwy.Floats](v []T) T {
	if len(v) == 0 {
		panic("vec: Min called on empty slice")
	}

	minVec := hwy.Load(v)
	lanes := minVec.NumLanes()

	// If slice is shorter than one vector, reduce and handle tail
	if len(v) < lanes {
		result := v[0]
		for i := 1; i < len(v); i++ {
			if v[i] < result {
				result = v[i]
			}
		}
		return result
	}

	// Process full vectors
	var i int
	for i = lanes; i+lanes <= len(v); i += lanes {
		va := hwy.Load(v[i:])
		minVec = hwy.Min(minVec, va)
	}

	// Reduce vector min to scalar
	result := hwy.ReduceMin(minVec)

	// Handle tail elements with scalar code
	for ; i < len(v); i++ {
		if v[i] < result {
			result = v[i]
		}
	}

	return result
}

// BaseMax returns the maximum value in a slice using hwy primitives.
//
// Panics if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with all numeric types: float16, bfloat16, float32, float64,
// int8, int16, int32, int64, uint8, uint16, uint32, uint64.
//
// Note: For slices containing NaN values (floats only), the behavior follows
// IEEE 754 SIMD semantics. NaN comparisons may propagate NaN.
//
// Example:
//
//	data := []float32{3, 1, 4, 1, 5}
//	result := Max(data)  // 5
func BaseMax[T hwy.Lanes](v []T) T {
	if len(v) == 0 {
		panic("vec: Max called on empty slice")
	}

	maxVec := hwy.Load(v)
	lanes := maxVec.NumLanes()

	// If slice is shorter than one vector, reduce and handle tail
	if len(v) < lanes {
		result := v[0]
		for i := 1; i < len(v); i++ {
			if v[i] > result {
				result = v[i]
			}
		}
		return result
	}

	// Process full vectors
	var i int
	for i = lanes; i+lanes <= len(v); i += lanes {
		va := hwy.Load(v[i:])
		maxVec = hwy.Max(maxVec, va)
	}

	// Reduce vector max to scalar
	result := hwy.ReduceMax(maxVec)

	// Handle tail elements with scalar code
	for ; i < len(v); i++ {
		if v[i] > result {
			result = v[i]
		}
	}

	return result
}

// BaseMinMax returns both the minimum and maximum values in a slice using hwy primitives.
//
// This is more efficient than calling Min and Max separately as it only
// makes a single pass through the data.
//
// Panics if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
// Works with float32 and float64 slices.
//
// Note: For slices containing NaN values, behavior follows standard Go
// comparison semantics where NaN comparisons return false.
//
// Example:
//
//	data := []float32{3, 1, 4, 1, 5}
//	min, max := MinMax(data)  // min=1, max=5
func BaseMinMax[T hwy.Floats](v []T) (min, max T) {
	if len(v) == 0 {
		panic("vec: MinMax called on empty slice")
	}

	minVec := hwy.Load(v)
	maxVec := minVec
	lanes := minVec.NumLanes()

	// If slice is shorter than one vector, handle with scalar code
	if len(v) < lanes {
		min = v[0]
		max = v[0]
		for i := 1; i < len(v); i++ {
			if v[i] < min {
				min = v[i]
			}
			if v[i] > max {
				max = v[i]
			}
		}
		return min, max
	}

	// Process full vectors
	var i int
	for i = lanes; i+lanes <= len(v); i += lanes {
		va := hwy.Load(v[i:])
		minVec = hwy.Min(minVec, va)
		maxVec = hwy.Max(maxVec, va)
	}

	// Reduce vector min/max to scalar
	min = hwy.ReduceMin(minVec)
	max = hwy.ReduceMax(maxVec)

	// Handle tail elements with scalar code
	for ; i < len(v); i++ {
		if v[i] < min {
			min = v[i]
		}
		if v[i] > max {
			max = v[i]
		}
	}

	return min, max
}
