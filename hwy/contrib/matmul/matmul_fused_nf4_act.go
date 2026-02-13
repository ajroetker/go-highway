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

//go:generate go run ../../../cmd/hwygen -input matmul_fused_nf4_act.go -dispatch fusednf4actmatmul -output . -targets avx2,avx512,neon:asm,fallback

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
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

// BaseFusedNF4MatMulSiLU performs fused NF4 dequantization + matmul + bias + SiLU activation.
// output[m,n] = SiLU(sum_k(input[m,k] * dequant(packed[k,n])) + bias[n])
func BaseFusedNF4MatMulSiLU(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)

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

			// Add bias before activation
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}

			// SiLU(x) = x * sigmoid(x)
			sig := math.BaseSigmoidVec[float32](acc)
			acc = hwy.Mul(acc, sig)
			hwy.Store(acc, outputRow[n:])
		}

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
			if bias != nil {
				sum += bias[n]
			}
			// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
			outputRow[n] = sum / (1.0 + float32(stdmath.Exp(float64(-sum))))
		}
	}
}

// BaseFusedNF4MatMulGELU performs fused NF4 dequantization + matmul + bias + GELU activation.
// output[m,n] = GELU(sum_k(input[m,k] * dequant(packed[k,n])) + bias[n])
func BaseFusedNF4MatMulGELU(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)

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

			// Add bias before activation
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}

			// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
			invSqrt2 := hwy.Set(float32(0.7071067811865476))
			half := hwy.Set(float32(0.5))
			one := hwy.Set(float32(1.0))
			scaled := hwy.Mul(acc, invSqrt2)
			erfVal := math.BaseErfVec[float32](scaled)
			acc = hwy.Mul(acc, hwy.Mul(half, hwy.Add(one, erfVal)))
			hwy.Store(acc, outputRow[n:])
		}

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
			if bias != nil {
				sum += bias[n]
			}
			outputRow[n] = sum * 0.5 * (1.0 + float32(stdmath.Erf(float64(sum)*0.7071067811865476)))
		}
	}
}

// BaseFusedNF4MatMulGELUApprox performs fused NF4 dequantization + matmul + bias + approximate GELU.
// output[m,n] = GELUApprox(sum_k(input[m,k] * dequant(packed[k,n])) + bias[n])
func BaseFusedNF4MatMulGELUApprox(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)

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

			// Add bias before activation
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}

			// GELUApprox(x) ≈ x * sigmoid(1.702 * x)
			coeff := hwy.Set(float32(1.702))
			scaled := hwy.Mul(acc, coeff)
			sig := math.BaseSigmoidVec[float32](scaled)
			acc = hwy.Mul(acc, sig)
			hwy.Store(acc, outputRow[n:])
		}

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
			if bias != nil {
				sum += bias[n]
			}
			// GELUApprox(x) = x * sigmoid(1.702*x) = x / (1 + exp(-1.702*x))
			outputRow[n] = sum / (1.0 + float32(stdmath.Exp(float64(-1.702*sum))))
		}
	}
}

// BaseFusedNF4MatMulReLU performs fused NF4 dequantization + matmul + bias + ReLU activation.
// output[m,n] = ReLU(sum_k(input[m,k] * dequant(packed[k,n])) + bias[n])
func BaseFusedNF4MatMulReLU(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)

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

			// Add bias before activation
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}

			// ReLU(x) = max(0, x)
			acc = hwy.Max(acc, hwy.Zero[float32]())
			hwy.Store(acc, outputRow[n:])
		}

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
			if bias != nil {
				sum += bias[n]
			}
			outputRow[n] = float32(stdmath.Max(0, float64(sum)))
		}
	}
}

// BaseFusedInt4MatMulSiLU performs fused Int4 dequantization + matmul + bias + SiLU activation.
// output[m,n] = SiLU(sum_k(input[m,k] * dequant(packed[k,n])) + bias[n])
func BaseFusedInt4MatMulSiLU(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)

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

			// Add bias before activation
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}

			sig := math.BaseSigmoidVec[float32](acc)
			acc = hwy.Mul(acc, sig)
			hwy.Store(acc, outputRow[n:])
		}

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
			if bias != nil {
				sum += bias[n]
			}
			outputRow[n] = sum / (1.0 + float32(stdmath.Exp(float64(-sum))))
		}
	}
}

// BaseFusedInt4MatMulGELU performs fused Int4 dequantization + matmul + bias + GELU activation.
// output[m,n] = GELU(sum_k(input[m,k] * dequant(packed[k,n])) + bias[n])
func BaseFusedInt4MatMulGELU(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)

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

			// Add bias before activation
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}

			invSqrt2 := hwy.Set(float32(0.7071067811865476))
			half := hwy.Set(float32(0.5))
			one := hwy.Set(float32(1.0))
			scaled := hwy.Mul(acc, invSqrt2)
			erfVal := math.BaseErfVec[float32](scaled)
			acc = hwy.Mul(acc, hwy.Mul(half, hwy.Add(one, erfVal)))
			hwy.Store(acc, outputRow[n:])
		}

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
			if bias != nil {
				sum += bias[n]
			}
			outputRow[n] = sum * 0.5 * (1.0 + float32(stdmath.Erf(float64(sum)*0.7071067811865476)))
		}
	}
}

// BaseFusedInt4MatMulGELUApprox performs fused Int4 dequantization + matmul + bias + approximate GELU.
func BaseFusedInt4MatMulGELUApprox(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)

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

			// Add bias before activation
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}

			coeff := hwy.Set(float32(1.702))
			scaled := hwy.Mul(acc, coeff)
			sig := math.BaseSigmoidVec[float32](scaled)
			acc = hwy.Mul(acc, sig)
			hwy.Store(acc, outputRow[n:])
		}

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
			if bias != nil {
				sum += bias[n]
			}
			outputRow[n] = sum / (1.0 + float32(stdmath.Exp(float64(-1.702*sum))))
		}
	}
}

// BaseFusedInt4MatMulReLU performs fused Int4 dequantization + matmul + bias + ReLU activation.
func BaseFusedInt4MatMulReLU(input []float32, packed []uint8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)

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

			// Add bias before activation
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}

			acc = hwy.Max(acc, hwy.Zero[float32]())
			hwy.Store(acc, outputRow[n:])
		}

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
			if bias != nil {
				sum += bias[n]
			}
			outputRow[n] = float32(stdmath.Max(0, float64(sum)))
		}
	}
}

// BaseFusedNF4MatMulSwiGLU performs fused NF4 dequantization + matmul + SwiGLU activation.
//
// SwiGLU is a gated activation: SwiGLU(x, gate) = SiLU(gate) * x
//
//	output[m,n] = SiLU(sum_k(input[m,k] * gate_weights[k,n])) * sum_k(input[m,k] * up_weights[k,n])
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
	gateBuf := make([]float32, lanes)
	upBuf := make([]float32, lanes)

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

					var gateQuantIdx int
					if weightIdx%2 == 0 {
						gateQuantIdx = int(gatePacked[packedIdx] & 0x0F)
					} else {
						gateQuantIdx = int((gatePacked[packedIdx] >> 4) & 0x0F)
					}
					gateScale := gateScales[scaleBase+groupIdx]
					gateBuf[lane] = nf4LookupTable[gateQuantIdx] * gateScale

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

			gateSilu := hwy.Mul(gateAcc, math.BaseSigmoidVec[float32](gateAcc))
			result := hwy.Mul(gateSilu, upAcc)
			hwy.Store(result, outputRow[n:])
		}

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
			gateSilu := gateSum / (1.0 + float32(stdmath.Exp(float64(-gateSum))))
			outputRow[n] = gateSilu * upSum
		}
	}
}

// BaseFusedInt4MatMulSwiGLU performs fused Int4 dequantization + matmul + SwiGLU activation.
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
	gateBuf := make([]float32, lanes)
	upBuf := make([]float32, lanes)

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
			gateSilu := gateSum / (1.0 + float32(stdmath.Exp(float64(-gateSum))))
			outputRow[n] = gateSilu * upSum
		}
	}
}
