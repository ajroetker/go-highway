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

//go:generate go run ../../../cmd/hwygen -input matmul_fused_nf4.go -dispatch fusednf4matmul -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

// nf4LookupTable contains the 16 fixed values for 4-bit NormalFloat quantization.
// These values are from the QLoRA paper and represent optimal quantization
// points for normally distributed weights.
var nf4LookupTable = [16]float32{
	-1.0,
	-0.6961928009986877,
	-0.5250730514526367,
	-0.39491748809814453,
	-0.28444138169288635,
	-0.18477343022823334,
	-0.09105003625154495,
	0.0,
	0.07958029955625534,
	0.16093020141124725,
	0.24611230194568634,
	0.33791524171829224,
	0.44070982933044434,
	0.5626170039176941,
	0.7229568362236023,
	1.0,
}

// BaseFusedNF4MatMul performs fused NF4 dequantization + matrix multiplication.
// output[m,n] = sum_k(input[m,k] * dequant(packed[k,n]))
//
// This implementation vectorizes over the N dimension, processing multiple
// output columns simultaneously using SIMD operations.
//
// Parameters:
//   - input: [M, K] float32 input matrix (row-major)
//   - packed: [K, N/2] uint8 packed NF4 weights (2 values per byte, low nibble first)
//   - scales: [K, numGroups] float32 per-group scales
//   - output: [M, N] float32 output matrix (row-major, pre-allocated)
//   - M, K, N: matrix dimensions
//   - groupSize: number of columns per scale group
func BaseFusedNF4MatMul(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()

	// Temporary buffer for dequantized weights (one vector width)
	dequantBuf := make([]float32, lanes)

	// Process each output row
	for m := 0; m < M; m++ {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		// Process output columns in groups of lanes
		var n int
		for n = 0; n+lanes <= N; n += lanes {
			// Initialize accumulator
			acc := hwy.Zero[float32]()

			// Accumulate over K dimension
			for k := 0; k < K; k++ {
				// Broadcast input[m, k]
				inputVal := hwy.Set(inputRow[k])

				// Dequantize 'lanes' weights from packed[k, n:n+lanes]
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

				// Load dequantized weights into vector
				weights := hwy.Load(dequantBuf)

				// FMA: acc += input * weight
				acc = hwy.MulAdd(inputVal, weights, acc)
			}

			// Store result
			hwy.Store(acc, outputRow[n:])
		}

		// Handle remaining columns (scalar tail)
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
			outputRow[n] = sum
		}
	}
}

// BaseFusedInt4MatMul performs fused Int4 dequantization + matrix multiplication.
// output[m,n] = sum_k(input[m,k] * dequant(packed[k,n]))
//
// Int4 uses symmetric quantization: values in [0,15] map to [-8,7].
//
// Parameters:
//   - input: [M, K] float32 input matrix (row-major)
//   - packed: [K, N/2] uint8 packed Int4 weights (2 values per byte, low nibble first)
//   - scales: [K, numGroups] float32 per-group scales
//   - output: [M, N] float32 output matrix (row-major, pre-allocated)
//   - M, K, N: matrix dimensions
//   - groupSize: number of columns per scale group
func BaseFusedInt4MatMul(input []float32, packed []uint8, scales []float32, output []float32, M, K, N, groupSize int) {
	if M == 0 || K == 0 || N == 0 {
		return
	}

	numGroups := (N + groupSize - 1) / groupSize
	lanes := hwy.Zero[float32]().NumLanes()

	// Temporary buffer for dequantized weights (one vector width)
	dequantBuf := make([]float32, lanes)

	// Process each output row
	for m := 0; m < M; m++ {
		inputRow := input[m*K : (m+1)*K]
		outputRow := output[m*N : (m+1)*N]

		// Process output columns in groups of lanes
		var n int
		for n = 0; n+lanes <= N; n += lanes {
			// Initialize accumulator
			acc := hwy.Zero[float32]()

			// Accumulate over K dimension
			for k := 0; k < K; k++ {
				// Broadcast input[m, k]
				inputVal := hwy.Set(inputRow[k])

				// Dequantize 'lanes' weights from packed[k, n:n+lanes]
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
					// Convert from [0,15] to [-8,7]
					dequantBuf[lane] = float32(unsignedVal-8) * scale
				}

				// Load dequantized weights into vector
				weights := hwy.Load(dequantBuf)

				// FMA: acc += input * weight
				acc = hwy.MulAdd(inputVal, weights, acc)
			}

			// Store result
			hwy.Store(acc, outputRow[n:])
		}

		// Handle remaining columns (scalar tail)
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
			outputRow[n] = sum
		}
	}
}
