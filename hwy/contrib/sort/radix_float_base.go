package sort

import (
	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go run ../../../cmd/hwygen -input radix_float_base.go -output . -targets avx2,avx512,neon,fallback -dispatch radix_float

// BaseFloatToSortable transforms float values to sortable order in-place.
// Positive floats: flip sign bit. Negative floats: flip all bits.
func BaseFloatToSortable[T hwy.Floats](data []T) {
	n := len(data)
	lanes := hwy.MaxLanes[T]()

	signBitVec := hwy.SignBit[T]()
	zeroVec := hwy.Zero[T]()
	allOnesVec := hwy.Not(zeroVec)

	i := 0
	for i+lanes <= n {
		v := hwy.Load(data[i:])

		// Check sign: negative if v < 0
		isNeg := hwy.LessThan(v, zeroVec)

		// XOR with allOnes for negative, signBit for positive
		negResult := hwy.Xor(v, allOnesVec)
		posResult := hwy.Xor(v, signBitVec)
		result := hwy.IfThenElse(isNeg, negResult, posResult)

		hwy.Store(result, data[i:])
		i += lanes
	}

	// Scalar tail handled by caller
}

// BaseSortableToFloat transforms sortable values back to float in-place.
func BaseSortableToFloat[T hwy.Floats](data []T) {
	n := len(data)
	lanes := hwy.MaxLanes[T]()

	signBitVec := hwy.SignBit[T]()
	zeroVec := hwy.Zero[T]()
	allOnesVec := hwy.Not(zeroVec)

	i := 0
	for i+lanes <= n {
		v := hwy.Load(data[i:])

		// After sorting, values with sign bit set were originally positive
		masked := hwy.And(v, signBitVec)
		wasPositive := hwy.NotEqual(masked, zeroVec)

		// XOR back
		posResult := hwy.Xor(v, signBitVec)
		negResult := hwy.Xor(v, allOnesVec)
		result := hwy.IfThenElse(wasPositive, posResult, negResult)

		hwy.Store(result, data[i:])
		i += lanes
	}

	// Scalar tail handled by caller
}
