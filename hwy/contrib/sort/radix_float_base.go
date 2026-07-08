// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package sort

import (
	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go run ../../../cmd/hwygen -input radix_float_base.go -output . -targets avx2,avx512,neon:goat,fallback -dispatch radix_float

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
		v := hwy.LoadSlice(data[i:])

		// Check sign: negative if v < 0
		isNeg := hwy.LessThan(v, zeroVec)

		// XOR with allOnes for negative, signBit for positive
		negResult := hwy.Xor(v, allOnesVec)
		posResult := hwy.Xor(v, signBitVec)
		result := hwy.IfThenElse(isNeg, negResult, posResult)

		hwy.StoreSlice(result, data[i:])
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
		v := hwy.LoadSlice(data[i:])

		// After sorting, values with sign bit set were originally positive
		masked := hwy.And(v, signBitVec)
		wasPositive := hwy.NotEqual(masked, zeroVec)

		// XOR back
		posResult := hwy.Xor(v, signBitVec)
		negResult := hwy.Xor(v, allOnesVec)
		result := hwy.IfThenElse(wasPositive, posResult, negResult)

		hwy.StoreSlice(result, data[i:])
		i += lanes
	}

	// Scalar tail handled by caller
}
