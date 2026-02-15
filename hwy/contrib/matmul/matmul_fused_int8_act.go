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

//go:generate go run ../../../cmd/hwygen -input matmul_fused_int8_act.go -dispatch fusedint8actmatmul -output . -targets avx2,avx512,neon:asm,fallback

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

// BaseFusedInt8MatMulSiLU performs fused Int8 dequantization + matmul + bias + SiLU activation.
// output[m,n] = SiLU(sum_k(input[m,k] * (weights[k,n] * scale[k,groupIdx])) + bias[n])
func BaseFusedInt8MatMulSiLU(input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)
	accBuf := make([]float32, N)

	for m := range M {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		for i := range N {
			accBuf[i] = 0
		}

		for k := range K {
			inputVal := hwy.Set(inputRow[k])
			baseIdx := k * N
			scaleBase := k * numGroups

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

			for ; n < N; n++ {
				weightIdx := baseIdx + n
				val := float32(weights[weightIdx])
				groupIdx := n / groupSize
				scale := scales[scaleBase+groupIdx]
				accBuf[n] += inputRow[k] * val * scale
			}
		}

		// Apply bias + SiLU activation + store
		var n int
		for n = 0; n+lanes <= N; n += lanes {
			acc := hwy.Load(accBuf[n:])
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}
			sig := math.BaseSigmoidVec[float32](acc)
			acc = hwy.Mul(acc, sig)
			hwy.Store(acc, outputRow[n:])
		}
		for ; n < N; n++ {
			sum := accBuf[n]
			if bias != nil {
				sum += bias[n]
			}
			// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
			outputRow[n] = sum / (1.0 + float32(stdmath.Exp(float64(-sum))))
		}
	}
}

// BaseFusedInt8MatMulGELU performs fused Int8 dequantization + matmul + bias + GELU activation.
// output[m,n] = GELU(sum_k(input[m,k] * (weights[k,n] * scale[k,groupIdx])) + bias[n])
func BaseFusedInt8MatMulGELU(input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)
	accBuf := make([]float32, N)

	for m := range M {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		for i := range N {
			accBuf[i] = 0
		}

		for k := range K {
			inputVal := hwy.Set(inputRow[k])
			baseIdx := k * N
			scaleBase := k * numGroups

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

			for ; n < N; n++ {
				weightIdx := baseIdx + n
				val := float32(weights[weightIdx])
				groupIdx := n / groupSize
				scale := scales[scaleBase+groupIdx]
				accBuf[n] += inputRow[k] * val * scale
			}
		}

		// Apply bias + GELU activation + store
		var n int
		for n = 0; n+lanes <= N; n += lanes {
			acc := hwy.Load(accBuf[n:])
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
			sum := accBuf[n]
			if bias != nil {
				sum += bias[n]
			}
			outputRow[n] = sum * 0.5 * (1.0 + float32(stdmath.Erf(float64(sum)*0.7071067811865476)))
		}
	}
}

// BaseFusedInt8MatMulGELUApprox performs fused Int8 dequantization + matmul + bias + approximate GELU.
// output[m,n] = GELUApprox(sum_k(input[m,k] * (weights[k,n] * scale[k,groupIdx])) + bias[n])
func BaseFusedInt8MatMulGELUApprox(input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)
	accBuf := make([]float32, N)

	for m := range M {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		for i := range N {
			accBuf[i] = 0
		}

		for k := range K {
			inputVal := hwy.Set(inputRow[k])
			baseIdx := k * N
			scaleBase := k * numGroups

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

			for ; n < N; n++ {
				weightIdx := baseIdx + n
				val := float32(weights[weightIdx])
				groupIdx := n / groupSize
				scale := scales[scaleBase+groupIdx]
				accBuf[n] += inputRow[k] * val * scale
			}
		}

		// Apply bias + GELUApprox activation + store
		var n int
		for n = 0; n+lanes <= N; n += lanes {
			acc := hwy.Load(accBuf[n:])
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
			sum := accBuf[n]
			if bias != nil {
				sum += bias[n]
			}
			// GELUApprox(x) = x * sigmoid(1.702*x) = x / (1 + exp(-1.702*x))
			outputRow[n] = sum / (1.0 + float32(stdmath.Exp(float64(-1.702*sum))))
		}
	}
}

// BaseFusedInt8MatMulReLU performs fused Int8 dequantization + matmul + bias + ReLU activation.
// output[m,n] = ReLU(sum_k(input[m,k] * (weights[k,n] * scale[k,groupIdx])) + bias[n])
func BaseFusedInt8MatMulReLU(input []float32, weights []int8, scales []float32, bias []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()
	dequantBuf := make([]float32, lanes)
	accBuf := make([]float32, N)

	for m := range M {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		for i := range N {
			accBuf[i] = 0
		}

		for k := range K {
			inputVal := hwy.Set(inputRow[k])
			baseIdx := k * N
			scaleBase := k * numGroups

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

			for ; n < N; n++ {
				weightIdx := baseIdx + n
				val := float32(weights[weightIdx])
				groupIdx := n / groupSize
				scale := scales[scaleBase+groupIdx]
				accBuf[n] += inputRow[k] * val * scale
			}
		}

		// Apply bias + ReLU activation + store
		var n int
		for n = 0; n+lanes <= N; n += lanes {
			acc := hwy.Load(accBuf[n:])
			if bias != nil {
				biasVec := hwy.Load(bias[n:])
				acc = hwy.Add(acc, biasVec)
			}
			// ReLU(x) = max(0, x)
			acc = hwy.Max(acc, hwy.Zero[float32]())
			hwy.Store(acc, outputRow[n:])
		}
		for ; n < N; n++ {
			sum := accBuf[n]
			if bias != nil {
				sum += bias[n]
			}
			outputRow[n] = float32(stdmath.Max(0, float64(sum)))
		}
	}
}
