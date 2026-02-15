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

package matmul

// This file contains helper functions and internal implementations for
// fused matmul+activation. These are separated from the main file so
// hwygen doesn't try to translate them to C (they use Go-specific
// constructs like sync.Pool and intra-package function calls).

import (
	stdmath "math"
	"sync"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

// Buffer pools for fused matmul+activation kernels (used by SME internal helpers).
var (
	fusedActDequantBufPool = sync.Pool{
		New: func() any { return make([]float32, 16) },
	}
	fusedActGateBufPool = sync.Pool{
		New: func() any { return make([]float32, 16) },
	}
	fusedActUpBufPool = sync.Pool{
		New: func() any { return make([]float32, 16) },
	}
)

// applyActivationVec applies the specified activation function to a SIMD vector.
// Used by SME code paths.
func applyActivationVec(v hwy.Vec[float32], act ActivationType) hwy.Vec[float32] {
	switch act {
	case ActSiLU:
		sig := math.BaseSigmoidVec[float32](v)
		return hwy.Mul(v, sig)
	case ActGELU:
		invSqrt2 := hwy.Set(float32(0.7071067811865476))
		half := hwy.Set(float32(0.5))
		one := hwy.Set(float32(1.0))
		scaled := hwy.Mul(v, invSqrt2)
		erfVal := math.BaseErfVec[float32](scaled)
		return hwy.Mul(v, hwy.Mul(half, hwy.Add(one, erfVal)))
	case ActGELUApprox:
		coeff := hwy.Set(float32(1.702))
		scaled := hwy.Mul(v, coeff)
		sig := math.BaseSigmoidVec[float32](scaled)
		return hwy.Mul(v, sig)
	case ActReLU:
		return hwy.Max(v, hwy.Zero[float32]())
	default:
		return v
	}
}

// sigmoidf32 computes sigmoid(x) = 1 / (1 + exp(-x)) for float32.
func sigmoidf32(x float32) float32 {
	return 1.0 / (1.0 + expf32(-x))
}

// applyActivationScalar applies the specified activation function to a scalar value.
// Used by SME code paths.
func applyActivationScalar(x float32, act ActivationType) float32 {
	switch act {
	case ActSiLU:
		return x * sigmoidf32(x)
	case ActGELU:
		return x * 0.5 * (1.0 + erff32(x*0.7071067811865476))
	case ActGELUApprox:
		return x * sigmoidf32(1.702*x)
	case ActReLU:
		if x > 0 {
			return x
		}
		return 0
	default:
		return x
	}
}

// expf32 computes exp(x) for float32 using a fast approximation.
func expf32(x float32) float32 {
	if x > 88.0 {
		return float32(3.4028235e+38)
	}
	if x < -88.0 {
		return 0
	}

	k := int(x*1.4426950408889634 + 0.5)
	if x < 0 {
		k = int(x*1.4426950408889634 - 0.5)
	}
	r := x - float32(k)*0.6931471805599453

	p := float32(1.0/720.0)*r + float32(1.0/120.0)
	p = p*r + float32(1.0/24.0)
	p = p*r + float32(1.0/6.0)
	p = p*r + float32(1.0/2.0)
	p = p*r + 1.0
	p = p*r + 1.0

	return float32(stdmath.Ldexp(float64(p), k))
}

// erff32 computes a fast approximation of erf(x) for float32.
func erff32(x float32) float32 {
	sign := float32(1.0)
	if x < 0 {
		sign = -1.0
		x = -x
	}

	t := 1.0 / (1.0 + 0.3275911*x)
	t2 := t * t
	t3 := t2 * t
	t4 := t3 * t
	t5 := t4 * t

	y := 1.0 - (0.254829592*t-0.284496736*t2+1.421413741*t3-1.453152027*t4+1.061405429*t5)*expf32(-x*x)
	return sign * y
}

// baseFusedNF4MatMulAct is the internal implementation for SME code paths.
func baseFusedNF4MatMulAct(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()

	dequantBuf := fusedActDequantBufPool.Get().([]float32)[:lanes]
	defer fusedActDequantBufPool.Put(dequantBuf[:cap(dequantBuf)])

	// Accumulator buffer for one output row — fits L1 for typical N
	accBuf := make([]float32, N)

	for m := range M {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		// Zero accumulators for this row
		for i := range N {
			accBuf[i] = 0
		}

		// K-outer, N-inner: sequential weight access, single input broadcast per k
		for k := range K {
			inputVal := hwy.Set(inputRow[k])
			baseIdx := k * N
			scaleBase := k * numGroups

			// Vectorized N sweep
			var n int
			for n = 0; n+lanes <= N; n += lanes {
				for lane := range lanes {
					colIdx := n + lane
					weightIdx := baseIdx + colIdx
					packedIdx := weightIdx / 2

					var quantIdx int
					if weightIdx%2 == 0 {
						quantIdx = int(packed[packedIdx] & 0x0F)
					} else {
						quantIdx = int((packed[packedIdx] >> 4) & 0x0F)
					}

					groupIdx := colIdx / groupSize
					scale := scales[scaleBase+groupIdx]
					dequantBuf[lane] = nf4LookupTable[quantIdx] * scale
				}

				weights := hwy.Load(dequantBuf)
				acc := hwy.Load(accBuf[n:])
				acc = hwy.MulAdd(inputVal, weights, acc)
				hwy.Store(acc, accBuf[n:])
			}

			// Scalar tail
			for ; n < N; n++ {
				weightIdx := baseIdx + n
				packedIdx := weightIdx / 2

				var quantIdx int
				if weightIdx%2 == 0 {
					quantIdx = int(packed[packedIdx] & 0x0F)
				} else {
					quantIdx = int((packed[packedIdx] >> 4) & 0x0F)
				}

				groupIdx := n / groupSize
				scale := scales[scaleBase+groupIdx]
				weight := nf4LookupTable[quantIdx] * scale
				accBuf[n] += inputRow[k] * weight
			}
		}

		// Apply bias + activation and store to output
		var n int
		for n = 0; n+lanes <= N; n += lanes {
			acc := hwy.Load(accBuf[n:])
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}
			acc = applyActivationVec(acc, act)
			hwy.Store(acc, outputRow[n:])
		}
		for ; n < N; n++ {
			val := accBuf[n]
			if bias != nil {
				val += bias[n]
			}
			outputRow[n] = applyActivationScalar(val, act)
		}
	}
}

// baseFusedInt8MatMulAct is the internal implementation for SME code paths.
func baseFusedInt8MatMulAct(input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()

	dequantBuf := fusedActDequantBufPool.Get().([]float32)[:lanes]
	defer fusedActDequantBufPool.Put(dequantBuf[:cap(dequantBuf)])

	// Accumulator buffer for one output row — fits L1 for typical N
	accBuf := make([]float32, N)

	for m := range M {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		// Zero accumulators for this row
		for i := range N {
			accBuf[i] = 0
		}

		// K-outer, N-inner: sequential weight access, single input broadcast per k
		for k := range K {
			inputVal := hwy.Set(inputRow[k])
			baseIdx := k * N
			scaleBase := k * numGroups

			// Vectorized N sweep
			var n int
			for n = 0; n+lanes <= N; n += lanes {
				for lane := range lanes {
					colIdx := n + lane
					weightIdx := baseIdx + colIdx

					val := float32(weights[weightIdx])

					groupIdx := colIdx / groupSize
					scale := scales[scaleBase+groupIdx]
					dequantBuf[lane] = val * scale
				}

				w := hwy.Load(dequantBuf)
				acc := hwy.Load(accBuf[n:])
				acc = hwy.MulAdd(inputVal, w, acc)
				hwy.Store(acc, accBuf[n:])
			}

			// Scalar tail
			for ; n < N; n++ {
				weightIdx := baseIdx + n

				val := float32(weights[weightIdx])

				groupIdx := n / groupSize
				scale := scales[scaleBase+groupIdx]
				weight := val * scale
				accBuf[n] += inputRow[k] * weight
			}
		}

		// Apply bias + activation and store to output
		var n int
		for n = 0; n+lanes <= N; n += lanes {
			acc := hwy.Load(accBuf[n:])
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}
			acc = applyActivationVec(acc, act)
			hwy.Store(acc, outputRow[n:])
		}
		for ; n < N; n++ {
			val := accBuf[n]
			if bias != nil {
				val += bias[n]
			}
			outputRow[n] = applyActivationScalar(val, act)
		}
	}
}

// baseFusedInt4MatMulAct is the internal implementation for SME code paths.
func baseFusedInt4MatMulAct(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int, act ActivationType) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()

	dequantBuf := fusedActDequantBufPool.Get().([]float32)[:lanes]
	defer fusedActDequantBufPool.Put(dequantBuf[:cap(dequantBuf)])

	// Accumulator buffer for one output row — fits L1 for typical N
	accBuf := make([]float32, N)

	for m := range M {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		// Zero accumulators for this row
		for i := range N {
			accBuf[i] = 0
		}

		// K-outer, N-inner: sequential weight access, single input broadcast per k
		for k := range K {
			inputVal := hwy.Set(inputRow[k])
			baseIdx := k * N
			scaleBase := k * numGroups

			// Vectorized N sweep
			var n int
			for n = 0; n+lanes <= N; n += lanes {
				for lane := range lanes {
					colIdx := n + lane
					weightIdx := baseIdx + colIdx
					packedIdx := weightIdx / 2

					var unsignedVal int
					if weightIdx%2 == 0 {
						unsignedVal = int(packed[packedIdx] & 0x0F)
					} else {
						unsignedVal = int((packed[packedIdx] >> 4) & 0x0F)
					}

					groupIdx := colIdx / groupSize
					scale := scales[scaleBase+groupIdx]
					dequantBuf[lane] = float32(unsignedVal-8) * scale
				}

				weights := hwy.Load(dequantBuf)
				acc := hwy.Load(accBuf[n:])
				acc = hwy.MulAdd(inputVal, weights, acc)
				hwy.Store(acc, accBuf[n:])
			}

			// Scalar tail
			for ; n < N; n++ {
				weightIdx := baseIdx + n
				packedIdx := weightIdx / 2

				var unsignedVal int
				if weightIdx%2 == 0 {
					unsignedVal = int(packed[packedIdx] & 0x0F)
				} else {
					unsignedVal = int((packed[packedIdx] >> 4) & 0x0F)
				}

				groupIdx := n / groupSize
				scale := scales[scaleBase+groupIdx]
				weight := float32(unsignedVal-8) * scale
				accBuf[n] += inputRow[k] * weight
			}
		}

		// Apply bias + activation and store to output
		var n int
		for n = 0; n+lanes <= N; n += lanes {
			acc := hwy.Load(accBuf[n:])
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}
			acc = applyActivationVec(acc, act)
			hwy.Store(acc, outputRow[n:])
		}
		for ; n < N; n++ {
			val := accBuf[n]
			if bias != nil {
				val += bias[n]
			}
			outputRow[n] = applyActivationScalar(val, act)
		}
	}
}
