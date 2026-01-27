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

	// Use 4 accumulators for each weight to maximize ILP
	var sum1_0, sum1_1, sum2_0, sum2_1 uint64
	var sum4_0, sum4_1, sum8_0, sum8_1 uint64

	lanes := hwy.Zero[uint64]().NumLanes()
	n := len(code)

	// Process 2 SIMD vectors at a time for better instruction-level parallelism
	stride := lanes * 2
	var i int
	for i = 0; i+stride <= n; i += stride {
		// Load vectors for first block
		codeVec0 := hwy.Load(code[i:])
		q1Vec0 := hwy.Load(q1[i:])
		q2Vec0 := hwy.Load(q2[i:])
		q3Vec0 := hwy.Load(q3[i:])
		q4Vec0 := hwy.Load(q4[i:])

		// Load vectors for second block
		codeVec1 := hwy.Load(code[i+lanes:])
		q1Vec1 := hwy.Load(q1[i+lanes:])
		q2Vec1 := hwy.Load(q2[i+lanes:])
		q3Vec1 := hwy.Load(q3[i+lanes:])
		q4Vec1 := hwy.Load(q4[i+lanes:])

		// AND and popcount for weight 1
		and1_0 := hwy.And(codeVec0, q1Vec0)
		and1_1 := hwy.And(codeVec1, q1Vec1)
		pop1_0 := hwy.PopCount(and1_0)
		pop1_1 := hwy.PopCount(and1_1)
		sum1_0 += uint64(hwy.ReduceSum(pop1_0))
		sum1_1 += uint64(hwy.ReduceSum(pop1_1))

		// AND and popcount for weight 2
		and2_0 := hwy.And(codeVec0, q2Vec0)
		and2_1 := hwy.And(codeVec1, q2Vec1)
		pop2_0 := hwy.PopCount(and2_0)
		pop2_1 := hwy.PopCount(and2_1)
		sum2_0 += uint64(hwy.ReduceSum(pop2_0))
		sum2_1 += uint64(hwy.ReduceSum(pop2_1))

		// AND and popcount for weight 4
		and4_0 := hwy.And(codeVec0, q3Vec0)
		and4_1 := hwy.And(codeVec1, q3Vec1)
		pop4_0 := hwy.PopCount(and4_0)
		pop4_1 := hwy.PopCount(and4_1)
		sum4_0 += uint64(hwy.ReduceSum(pop4_0))
		sum4_1 += uint64(hwy.ReduceSum(pop4_1))

		// AND and popcount for weight 8
		and8_0 := hwy.And(codeVec0, q4Vec0)
		and8_1 := hwy.And(codeVec1, q4Vec1)
		pop8_0 := hwy.PopCount(and8_0)
		pop8_1 := hwy.PopCount(and8_1)
		sum8_0 += uint64(hwy.ReduceSum(pop8_0))
		sum8_1 += uint64(hwy.ReduceSum(pop8_1))
	}

	// Process remaining full vectors
	for i+lanes <= n {
		codeVec := hwy.Load(code[i:])
		q1Vec := hwy.Load(q1[i:])
		q2Vec := hwy.Load(q2[i:])
		q3Vec := hwy.Load(q3[i:])
		q4Vec := hwy.Load(q4[i:])

		pop1 := hwy.PopCount(hwy.And(codeVec, q1Vec))
		pop2 := hwy.PopCount(hwy.And(codeVec, q2Vec))
		pop4 := hwy.PopCount(hwy.And(codeVec, q3Vec))
		pop8 := hwy.PopCount(hwy.And(codeVec, q4Vec))

		sum1_0 += uint64(hwy.ReduceSum(pop1))
		sum2_0 += uint64(hwy.ReduceSum(pop2))
		sum4_0 += uint64(hwy.ReduceSum(pop4))
		sum8_0 += uint64(hwy.ReduceSum(pop8))

		i += lanes
	}

	// Combine accumulators
	sum1 := sum1_0 + sum1_1
	sum2 := sum2_0 + sum2_1
	sum4 := sum4_0 + sum4_1
	sum8 := sum8_0 + sum8_1

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
			vecData := hwy.Load(vec[dim:])

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
