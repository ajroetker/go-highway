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

// NOTE: Code generation is disabled for this file because hwygen cannot properly
// transform generic function calls like math.BaseSigmoidVec[float32]() to target-specific
// versions. The base implementation uses portable hwy.* operations which already
// provide SIMD acceleration across all platforms.
// TODO: Re-enable once hwygen supports generic cross-package function calls.
// go:generate go run ../../../cmd/hwygen -input matmul_fused_nf4_act.go -dispatch fusednf4actmatmul -output . -targets avx2,avx512,neon,fallback

import (
	stdmath "math"
	"sync"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

// Buffer pools for fused matmul+activation kernels to reduce GC pressure.
// Max lane width is 16 (AVX-512), so buffers of size 16 suffice for all platforms.
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

// ActivationType specifies which activation function to apply after matmul.
type ActivationType int

const (
	// ActNone applies no activation (identity).
	ActNone ActivationType = iota
	// ActSiLU applies SiLU/Swish: x * sigmoid(x). Used in LLaMA, Mistral.
	ActSiLU
	// ActGELU applies exact GELU: x * 0.5 * (1 + erf(x/sqrt(2))). Used in BERT, GPT.
	ActGELU
	// ActGELUApprox applies approximate GELU: x * sigmoid(1.702 * x).
	ActGELUApprox
	// ActReLU applies ReLU: max(0, x).
	ActReLU
)

// BaseFusedNF4MatMulSiLU performs fused NF4 dequantization + matmul + SiLU activation.
// output[m,n] = SiLU(sum_k(input[m,k] * dequant(packed[k,n])))
//
// By fusing the activation into the matmul, we avoid a separate memory read/write
// pass over the output tensor, saving O(M*N) memory traffic.
//
// Parameters:
//   - input: [M, K] float32 input matrix (row-major)
//   - packed: [K, N/2] uint8 packed NF4 weights (2 values per byte, low nibble first)
//   - scales: [K, numGroups] float32 per-group scales
//   - output: [M, N] float32 output matrix (row-major, pre-allocated)
//   - M, K, N: matrix dimensions
//   - groupSize: number of columns per scale group
func BaseFusedNF4MatMulSiLU(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
	baseFusedNF4MatMulAct(input, packed, scales, output, M, K, N, groupSize, ActSiLU)
}

// BaseFusedNF4MatMulGELU performs fused NF4 dequantization + matmul + GELU activation.
// output[m,n] = GELU(sum_k(input[m,k] * dequant(packed[k,n])))
func BaseFusedNF4MatMulGELU(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
	baseFusedNF4MatMulAct(input, packed, scales, output, M, K, N, groupSize, ActGELU)
}

// BaseFusedNF4MatMulGELUApprox performs fused NF4 dequantization + matmul + approximate GELU.
// output[m,n] = GELUApprox(sum_k(input[m,k] * dequant(packed[k,n])))
func BaseFusedNF4MatMulGELUApprox(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
	baseFusedNF4MatMulAct(input, packed, scales, output, M, K, N, groupSize, ActGELUApprox)
}

// BaseFusedNF4MatMulReLU performs fused NF4 dequantization + matmul + ReLU activation.
// output[m,n] = ReLU(sum_k(input[m,k] * dequant(packed[k,n])))
func BaseFusedNF4MatMulReLU(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
	baseFusedNF4MatMulAct(input, packed, scales, output, M, K, N, groupSize, ActReLU)
}

// BaseFusedInt4MatMulSiLU performs fused Int4 dequantization + matmul + SiLU activation.
// output[m,n] = SiLU(sum_k(input[m,k] * dequant(packed[k,n])))
func BaseFusedInt4MatMulSiLU(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
	baseFusedInt4MatMulAct(input, packed, scales, output, M, K, N, groupSize, ActSiLU)
}

// BaseFusedInt4MatMulGELU performs fused Int4 dequantization + matmul + GELU activation.
// output[m,n] = GELU(sum_k(input[m,k] * dequant(packed[k,n])))
func BaseFusedInt4MatMulGELU(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
	baseFusedInt4MatMulAct(input, packed, scales, output, M, K, N, groupSize, ActGELU)
}

// BaseFusedInt4MatMulGELUApprox performs fused Int4 dequantization + matmul + approximate GELU.
func BaseFusedInt4MatMulGELUApprox(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
	baseFusedInt4MatMulAct(input, packed, scales, output, M, K, N, groupSize, ActGELUApprox)
}

// BaseFusedInt4MatMulReLU performs fused Int4 dequantization + matmul + ReLU activation.
func BaseFusedInt4MatMulReLU(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
	baseFusedInt4MatMulAct(input, packed, scales, output, M, K, N, groupSize, ActReLU)
}

// applyActivationVec applies the specified activation function to a SIMD vector.
func applyActivationVec(v hwy.Vec[float32], act ActivationType) hwy.Vec[float32] {
	switch act {
	case ActSiLU:
		// SiLU(x) = x * sigmoid(x)
		sig := math.BaseSigmoidVec[float32](v)
		return hwy.Mul(v, sig)
	case ActGELU:
		// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
		invSqrt2 := hwy.Const[float32](0.7071067811865476)
		half := hwy.Const[float32](0.5)
		one := hwy.Const[float32](1.0)
		scaled := hwy.Mul(v, invSqrt2)
		erfVal := math.BaseErfVec[float32](scaled)
		return hwy.Mul(v, hwy.Mul(half, hwy.Add(one, erfVal)))
	case ActGELUApprox:
		// GELU(x) ≈ x * sigmoid(1.702 * x)
		coeff := hwy.Const[float32](1.702)
		scaled := hwy.Mul(v, coeff)
		sig := math.BaseSigmoidVec[float32](scaled)
		return hwy.Mul(v, sig)
	case ActReLU:
		// ReLU(x) = max(0, x)
		zero := hwy.Const[float32](0.0)
		return hwy.Max(v, zero)
	default:
		return v
	}
}

// sigmoidf32 computes sigmoid(x) = 1 / (1 + exp(-x)) for float32.
func sigmoidf32(x float32) float32 {
	return 1.0 / (1.0 + expf32(-x))
}

// applyActivationScalar applies the specified activation function to a scalar value.
func applyActivationScalar(x float32, act ActivationType) float32 {
	switch act {
	case ActSiLU:
		// SiLU(x) = x * sigmoid(x), matching the vector implementation
		return x * sigmoidf32(x)
	case ActGELU:
		return x * 0.5 * (1.0 + erff32(x*0.7071067811865476))
	case ActGELUApprox:
		// GELUApprox(x) = x * sigmoid(1.702 * x)
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
	// Clamp to avoid overflow/underflow
	if x > 88.0 {
		return float32(3.4028235e+38)
	}
	if x < -88.0 {
		return 0
	}

	// Range reduction: x = k*ln(2) + r
	k := int(x*1.4426950408889634 + 0.5)
	if x < 0 {
		k = int(x*1.4426950408889634 - 0.5)
	}
	r := x - float32(k)*0.6931471805599453

	// Horner's method for exp(r) polynomial
	p := float32(1.0/720.0)*r + float32(1.0/120.0)
	p = p*r + float32(1.0/24.0)
	p = p*r + float32(1.0/6.0)
	p = p*r + float32(1.0/2.0)
	p = p*r + 1.0
	p = p*r + 1.0

	// Scale by 2^k using math.Ldexp for efficiency (single operation vs k iterations)
	return float32(stdmath.Ldexp(float64(p), k))
}

// erff32 computes a fast approximation of erf(x) for float32.
func erff32(x float32) float32 {
	// Abramowitz and Stegun approximation
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

	y := 1.0 - (0.254829592*t - 0.284496736*t2 + 1.421413741*t3 - 1.453152027*t4 + 1.061405429*t5) * expf32(-x*x)
	return sign * y
}

// baseFusedNF4MatMulAct is the core implementation for fused NF4 dequant + matmul + activation.
// It vectorizes over the N dimension and applies the activation in-register before storing.
func baseFusedNF4MatMulAct(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int, act ActivationType) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()

	dequantBuf := fusedActDequantBufPool.Get().([]float32)[:lanes]
	defer fusedActDequantBufPool.Put(dequantBuf[:cap(dequantBuf)])

	for m := 0; m < M; m++ {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		var n int
		for n = 0; n+lanes <= N; n += lanes {
			acc := hwy.Zero[float32]()

			for k := 0; k < K; k++ {
				inputVal := hwy.Set(inputRow[k])
				baseIdx := k * N
				scaleBase := k * numGroups

				for lane := 0; lane < lanes; lane++ {
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
				acc = hwy.MulAdd(inputVal, weights, acc)
			}

			// Apply activation in-register before storing (avoids extra memory round-trip)
			acc = applyActivationVec(acc, act)
			hwy.Store(acc, outputRow[n:])
		}

		// Scalar tail
		for ; n < N; n++ {
			groupIdx := n / groupSize
			sum := float32(0)
			for k := 0; k < K; k++ {
				weightIdx := k*N + n
				packedIdx := weightIdx / 2

				var quantIdx int
				if weightIdx%2 == 0 {
					quantIdx = int(packed[packedIdx] & 0x0F)
				} else {
					quantIdx = int((packed[packedIdx] >> 4) & 0x0F)
				}

				scale := scales[k*numGroups+groupIdx]
				weight := nf4LookupTable[quantIdx] * scale
				sum += inputRow[k] * weight
			}
			outputRow[n] = applyActivationScalar(sum, act)
		}
	}
}

// baseFusedInt4MatMulAct is the core implementation for fused Int4 dequant + matmul + activation.
func baseFusedInt4MatMulAct(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int, act ActivationType) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()

	dequantBuf := fusedActDequantBufPool.Get().([]float32)[:lanes]
	defer fusedActDequantBufPool.Put(dequantBuf[:cap(dequantBuf)])

	for m := 0; m < M; m++ {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		var n int
		for n = 0; n+lanes <= N; n += lanes {
			acc := hwy.Zero[float32]()

			for k := 0; k < K; k++ {
				inputVal := hwy.Set(inputRow[k])
				baseIdx := k * N
				scaleBase := k * numGroups

				for lane := 0; lane < lanes; lane++ {
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
				acc = hwy.MulAdd(inputVal, weights, acc)
			}

			acc = applyActivationVec(acc, act)
			hwy.Store(acc, outputRow[n:])
		}

		// Scalar tail
		for ; n < N; n++ {
			groupIdx := n / groupSize
			sum := float32(0)
			for k := 0; k < K; k++ {
				weightIdx := k*N + n
				packedIdx := weightIdx / 2

				var unsignedVal int
				if weightIdx%2 == 0 {
					unsignedVal = int(packed[packedIdx] & 0x0F)
				} else {
					unsignedVal = int((packed[packedIdx] >> 4) & 0x0F)
				}

				scale := scales[k*numGroups+groupIdx]
				weight := float32(unsignedVal-8) * scale
				sum += inputRow[k] * weight
			}
			outputRow[n] = applyActivationScalar(sum, act)
		}
	}
}

// BaseFusedNF4MatMulSwiGLU performs fused NF4 dequantization + matmul + SwiGLU activation.
//
// SwiGLU is a gated activation: SwiGLU(x, gate) = SiLU(gate) * x
// Used in LLaMA, Mistral, and other modern architectures.
//
// This fuses the entire MLP gate+up projection into one pass:
//   output[m,n] = SiLU(sum_k(input[m,k] * gate_weights[k,n])) * sum_k(input[m,k] * up_weights[k,n])
//
// By computing both projections and the gating in one pass over the input,
// we avoid materializing intermediate gate/up tensors (saving 2*M*N memory).
//
// Parameters:
//   - input: [M, K] float32 input matrix (row-major)
//   - gatePacked: [K, N/2] uint8 packed NF4 gate projection weights
//   - gateScales: [K, numGroups] float32 per-group scales for gate
//   - upPacked: [K, N/2] uint8 packed NF4 up projection weights
//   - upScales: [K, numGroups] float32 per-group scales for up
//   - output: [M, N] float32 output matrix (row-major, pre-allocated)
//   - M, K, N: matrix dimensions
//   - groupSize: number of columns per scale group
func BaseFusedNF4MatMulSwiGLU(
	input []float32,
	gatePacked []uint8, gateScales []float32,
	upPacked []uint8, upScales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()

	gateBuf := fusedActGateBufPool.Get().([]float32)[:lanes]
	defer fusedActGateBufPool.Put(gateBuf[:cap(gateBuf)])
	upBuf := fusedActUpBufPool.Get().([]float32)[:lanes]
	defer fusedActUpBufPool.Put(upBuf[:cap(upBuf)])

	for m := 0; m < M; m++ {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		var n int
		for n = 0; n+lanes <= N; n += lanes {
			gateAcc := hwy.Zero[float32]()
			upAcc := hwy.Zero[float32]()

			for k := 0; k < K; k++ {
				inputVal := hwy.Set(inputRow[k])
				baseIdx := k * N
				scaleBase := k * numGroups

				// Dequantize gate and up weights for this K row, lanes columns
				for lane := 0; lane < lanes; lane++ {
					colIdx := n + lane
					weightIdx := baseIdx + colIdx
					packedIdx := weightIdx / 2

					groupIdx := colIdx / groupSize

					// Gate weight
					var gateQuantIdx int
					if weightIdx%2 == 0 {
						gateQuantIdx = int(gatePacked[packedIdx] & 0x0F)
					} else {
						gateQuantIdx = int((gatePacked[packedIdx] >> 4) & 0x0F)
					}
					gateScale := gateScales[scaleBase+groupIdx]
					gateBuf[lane] = nf4LookupTable[gateQuantIdx] * gateScale

					// Up weight
					var upQuantIdx int
					if weightIdx%2 == 0 {
						upQuantIdx = int(upPacked[packedIdx] & 0x0F)
					} else {
						upQuantIdx = int((upPacked[packedIdx] >> 4) & 0x0F)
					}
					upScale := upScales[scaleBase+groupIdx]
					upBuf[lane] = nf4LookupTable[upQuantIdx] * upScale
				}

				gateWeights := hwy.Load(gateBuf)
				upWeights := hwy.Load(upBuf)
				gateAcc = hwy.MulAdd(inputVal, gateWeights, gateAcc)
				upAcc = hwy.MulAdd(inputVal, upWeights, upAcc)
			}

			// SwiGLU: output = SiLU(gate) * up
			gateSilu := hwy.Mul(gateAcc, math.BaseSigmoidVec[float32](gateAcc))
			result := hwy.Mul(gateSilu, upAcc)
			hwy.Store(result, outputRow[n:])
		}

		// Scalar tail
		for ; n < N; n++ {
			groupIdx := n / groupSize
			gateSum := float32(0)
			upSum := float32(0)
			for k := 0; k < K; k++ {
				weightIdx := k*N + n
				packedIdx := weightIdx / 2

				var gateQuantIdx int
				if weightIdx%2 == 0 {
					gateQuantIdx = int(gatePacked[packedIdx] & 0x0F)
				} else {
					gateQuantIdx = int((gatePacked[packedIdx] >> 4) & 0x0F)
				}
				gateScale := gateScales[k*numGroups+groupIdx]
				gateSum += inputRow[k] * nf4LookupTable[gateQuantIdx] * gateScale

				var upQuantIdx int
				if weightIdx%2 == 0 {
					upQuantIdx = int(upPacked[packedIdx] & 0x0F)
				} else {
					upQuantIdx = int((upPacked[packedIdx] >> 4) & 0x0F)
				}
				upScale := upScales[k*numGroups+groupIdx]
				upSum += inputRow[k] * nf4LookupTable[upQuantIdx] * upScale
			}
			// SwiGLU: SiLU(gate) * up
			gateSilu := gateSum / (1.0 + expf32(-gateSum))
			outputRow[n] = gateSilu * upSum
		}
	}
}

// BaseFusedInt4MatMulSwiGLU performs fused Int4 dequantization + matmul + SwiGLU activation.
// Same as BaseFusedNF4MatMulSwiGLU but for Int4 quantized weights.
func BaseFusedInt4MatMulSwiGLU(
	input []float32,
	gatePacked []uint8, gateScales []float32,
	upPacked []uint8, upScales []float32,
	output []float32,
	M, K, N, groupSize int,
) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()

	gateBuf := fusedActGateBufPool.Get().([]float32)[:lanes]
	defer fusedActGateBufPool.Put(gateBuf[:cap(gateBuf)])
	upBuf := fusedActUpBufPool.Get().([]float32)[:lanes]
	defer fusedActUpBufPool.Put(upBuf[:cap(upBuf)])

	for m := 0; m < M; m++ {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		var n int
		for n = 0; n+lanes <= N; n += lanes {
			gateAcc := hwy.Zero[float32]()
			upAcc := hwy.Zero[float32]()

			for k := 0; k < K; k++ {
				inputVal := hwy.Set(inputRow[k])
				baseIdx := k * N
				scaleBase := k * numGroups

				for lane := 0; lane < lanes; lane++ {
					colIdx := n + lane
					weightIdx := baseIdx + colIdx
					packedIdx := weightIdx / 2

					groupIdx := colIdx / groupSize

					var gateUnsigned int
					if weightIdx%2 == 0 {
						gateUnsigned = int(gatePacked[packedIdx] & 0x0F)
					} else {
						gateUnsigned = int((gatePacked[packedIdx] >> 4) & 0x0F)
					}
					gateScale := gateScales[scaleBase+groupIdx]
					gateBuf[lane] = float32(gateUnsigned-8) * gateScale

					var upUnsigned int
					if weightIdx%2 == 0 {
						upUnsigned = int(upPacked[packedIdx] & 0x0F)
					} else {
						upUnsigned = int((upPacked[packedIdx] >> 4) & 0x0F)
					}
					upScale := upScales[scaleBase+groupIdx]
					upBuf[lane] = float32(upUnsigned-8) * upScale
				}

				gateWeights := hwy.Load(gateBuf)
				upWeights := hwy.Load(upBuf)
				gateAcc = hwy.MulAdd(inputVal, gateWeights, gateAcc)
				upAcc = hwy.MulAdd(inputVal, upWeights, upAcc)
			}

			gateSilu := hwy.Mul(gateAcc, math.BaseSigmoidVec[float32](gateAcc))
			result := hwy.Mul(gateSilu, upAcc)
			hwy.Store(result, outputRow[n:])
		}

		// Scalar tail
		for ; n < N; n++ {
			groupIdx := n / groupSize
			gateSum := float32(0)
			upSum := float32(0)
			for k := 0; k < K; k++ {
				weightIdx := k*N + n
				packedIdx := weightIdx / 2

				var gateUnsigned int
				if weightIdx%2 == 0 {
					gateUnsigned = int(gatePacked[packedIdx] & 0x0F)
				} else {
					gateUnsigned = int((gatePacked[packedIdx] >> 4) & 0x0F)
				}
				gateScale := gateScales[k*numGroups+groupIdx]
				gateSum += inputRow[k] * float32(gateUnsigned-8) * gateScale

				var upUnsigned int
				if weightIdx%2 == 0 {
					upUnsigned = int(upPacked[packedIdx] & 0x0F)
				} else {
					upUnsigned = int((upPacked[packedIdx] >> 4) & 0x0F)
				}
				upScale := upScales[k*numGroups+groupIdx]
				upSum += inputRow[k] * float32(upUnsigned-8) * upScale
			}
			gateSilu := gateSum / (1.0 + expf32(-gateSum))
			outputRow[n] = gateSilu * upSum
		}
	}
}
