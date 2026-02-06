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

package rabitq

//go:generate go run ../../../cmd/hwygen -input rabitq_base.go -output . -targets avx2,avx512,neon,fallback -dispatch rabitq

import (
	"math"
	"math/bits"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseBitProduct computes the weighted bit product used in RaBitQ distance estimation.
// It takes a data vector code and 4 query sub-codes, computing:
//
//	1*popcount(code & q1) + 2*popcount(code & q2) +
//	4*popcount(code & q3) + 8*popcount(code & q4)
//
// All slices must have the same length.
// This is the hot path operation in RaBitQ search, called for every candidate vector.
func BaseBitProduct(code, q1, q2, q3, q4 []uint64) uint32 {
	if len(code) == 0 {
		return 0
	}

	var sum1, sum2, sum4, sum8 uint64

	lanes := hwy.Zero[uint64]().NumLanes()
	n := len(code)

	// Process 4 SIMD vectors at a time using Load4.
	// On NEON (lanes=2), this processes 8 uint64s per iteration.
	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		codeVec0, codeVec1, codeVec2, codeVec3 := hwy.Load4(code[i:])
		q1Vec0, q1Vec1, q1Vec2, q1Vec3 := hwy.Load4(q1[i:])
		q2Vec0, q2Vec1, q2Vec2, q2Vec3 := hwy.Load4(q2[i:])
		q3Vec0, q3Vec1, q3Vec2, q3Vec3 := hwy.Load4(q3[i:])
		q4Vec0, q4Vec1, q4Vec2, q4Vec3 := hwy.Load4(q4[i:])

		sum1 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec0, q1Vec0))))
		sum1 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec1, q1Vec1))))
		sum1 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec2, q1Vec2))))
		sum1 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec3, q1Vec3))))

		sum2 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec0, q2Vec0))))
		sum2 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec1, q2Vec1))))
		sum2 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec2, q2Vec2))))
		sum2 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec3, q2Vec3))))

		sum4 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec0, q3Vec0))))
		sum4 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec1, q3Vec1))))
		sum4 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec2, q3Vec2))))
		sum4 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec3, q3Vec3))))

		sum8 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec0, q4Vec0))))
		sum8 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec1, q4Vec1))))
		sum8 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec2, q4Vec2))))
		sum8 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec3, q4Vec3))))
	}

	// Process remaining full vectors one at a time.
	for i+lanes <= n {
		codeVec := hwy.LoadSlice(code[i:])
		q1Vec := hwy.LoadSlice(q1[i:])
		q2Vec := hwy.LoadSlice(q2[i:])
		q3Vec := hwy.LoadSlice(q3[i:])
		q4Vec := hwy.LoadSlice(q4[i:])

		sum1 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec, q1Vec))))
		sum2 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec, q2Vec))))
		sum4 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec, q3Vec))))
		sum8 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(codeVec, q4Vec))))

		i += lanes
	}

	// Process tail elements with scalar code
	for ; i < n; i++ {
		sum1 += uint64(bits.OnesCount64(code[i] & q1[i]))
		sum2 += uint64(bits.OnesCount64(code[i] & q2[i]))
		sum4 += uint64(bits.OnesCount64(code[i] & q3[i]))
		sum8 += uint64(bits.OnesCount64(code[i] & q4[i]))
	}

	// Compute weighted sum: 1*sum1 + 2*sum2 + 4*sum4 + 8*sum8
	return uint32(sum1 + (sum2 << 1) + (sum4 << 2) + (sum8 << 3))
}

// BaseQuantizeVectors quantizes unit vectors into 1-bit codes.
//
// For each input unit vector, this function:
//  1. Extracts sign bits (1 for positive/zero, 0 for negative)
//  2. Packs bits into uint64 codes (MSB-first within each uint64)
//  3. Computes the dot product between the unit vector and its quantized form
//  4. Counts the number of 1-bits in the code
//
// Parameters:
//   - unitVectors: flattened array of unit vectors (count × dims float32s)
//   - codes: output buffer for quantization codes (count × width uint64s)
//   - dotProducts: output buffer for inverted dot products (count float32s)
//   - codeCounts: output buffer for bit counts (count uint32s)
//   - sqrtDimsInv: precomputed 1/√dims
//   - count: number of vectors to process
//   - dims: dimensions per vector
//   - width: number of uint64s per code (typically ⌈dims/64⌉)
//
// The dotProducts output contains 1/<o̅,o> (inverted) for use in distance estimation.
// If the dot product is zero (vector equals centroid), dotProducts[i] is set to 0.
func BaseQuantizeVectors(
	unitVectors []float32,
	codes []uint64,
	dotProducts []float32,
	codeCounts []uint32,
	sqrtDimsInv float32,
	count, dims, width int,
) {
	negSqrtDimsInv := -sqrtDimsInv

	for i := range count {
		vec := unitVectors[i*dims : (i+1)*dims]
		code := codes[i*width : (i+1)*width]

		var dotProduct float64
		var codeBits uint64
		var codeCount uint32
		codeIdx := 0
		bitPos := 0

		// Process using SIMD where beneficial
		lanes := hwy.Zero[float32]().NumLanes()
		dim := 0

		// Process full SIMD vectors
		for dim+lanes <= dims {
			vecData := hwy.LoadSlice(vec[dim:])

			// Get sign bits: 1 if negative, 0 otherwise
			// We'll compute this by checking if value < 0
			zeroVec := hwy.Zero[float32]()
			negMask := hwy.LessThan(vecData, zeroVec)

			// Compute dot product contribution
			// If positive: element * sqrtDimsInv
			// If negative: element * (-sqrtDimsInv)
			posMultVec := hwy.Set[float32](sqrtDimsInv)
			negMultVec := hwy.Set[float32](negSqrtDimsInv)
			multVec := hwy.IfThenElse(negMask, negMultVec, posMultVec)
			prodVec := hwy.Mul(vecData, multVec)
			dotProduct += float64(hwy.ReduceSum(prodVec))

			// Extract sign bits and pack into code
			// Sign bit in IEEE 754: bit 31 is 1 for negative
			// We want: 1 for positive/zero, 0 for negative
			for j := range lanes {
				element := hwy.GetLane(vecData, j)
				signBit := getSignBit(element)
				// Invert: 1 for positive, 0 for negative
				codeBits = (codeBits << 1) | uint64(1-signBit)
				bitPos++

				if bitPos == 64 {
					code[codeIdx] = codeBits
					codeCount += uint32(bits.OnesCount64(codeBits))
					codeIdx++
					codeBits = 0
					bitPos = 0
				}
			}

			dim += lanes
		}

		// Process remaining elements
		for ; dim < dims; dim++ {
			element := vec[dim]
			signBit := getSignBit(element)

			// Compute dot product contribution
			var mult float32
			if signBit == 1 {
				mult = negSqrtDimsInv
			} else {
				mult = sqrtDimsInv
			}
			dotProduct += float64(element) * float64(mult)

			// Pack bit (inverted: 1 for positive, 0 for negative)
			codeBits = (codeBits << 1) | uint64(1-signBit)
			bitPos++

			if bitPos == 64 {
				code[codeIdx] = codeBits
				codeCount += uint32(bits.OnesCount64(codeBits))
				codeIdx++
				codeBits = 0
				bitPos = 0
			}
		}

		// Handle remaining bits - shift to MSB positions
		if bitPos > 0 {
			codeBits = codeBits << (64 - bitPos)
			code[codeIdx] = codeBits
			codeCount += uint32(bits.OnesCount64(codeBits))
		}

		// Store results
		codeCounts[i] = codeCount
		if dotProduct != 0 {
			dotProducts[i] = 1.0 / float32(dotProduct)
		} else {
			dotProducts[i] = 0
		}
	}
}

// getSignBit returns 1 if the float is negative (including -0), 0 otherwise.
func getSignBit(f float32) uint32 {
	return math.Float32bits(f) >> 31
}

// MultiplySigns returns a float32 with the magnitude of x and a sign
// that is the product of the signs of x and y.
// This is equivalent to: sign(x) * sign(y) * |x|
func MultiplySigns(x, y float32) float32 {
	const signMask = 1 << 31
	xBits := math.Float32bits(x)
	yBits := math.Float32bits(y)
	// XOR the sign bits: if y is negative, flip x's sign
	resultBits := xBits ^ (yBits & signMask)
	return math.Float32frombits(resultBits)
}

// CodeWidth returns the number of uint64s needed to store a code for the given dimensions.
// This is ⌈dims/64⌉.
func CodeWidth(dims int) int {
	return (dims + 63) / 64
}
