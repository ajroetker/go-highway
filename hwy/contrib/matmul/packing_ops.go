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

//go:generate go run ../../../cmd/hwygen -input packing_ops.go -dispatch packing_ops -output . -targets avx2,avx512,neon:asm,fallback

import "github.com/ajroetker/go-highway/hwy"

// BaseApplyPackedOutput applies the computed packed output to the final output matrix.
//
// This function transfers results from a temporary packed output buffer to the
// actual output matrix, applying alpha and beta scaling:
//
//	output = alpha * packedOutput + beta * output
//
// Using a packed output buffer allows the micro-kernel to write contiguously
// without bounds checking, improving performance. The alpha/beta application
// is then done efficiently with SIMD in this separate pass.
//
// Parameters:
//   - packedOutput: Temporary buffer with computed results [height, packedStride]
//   - output: Final output matrix in row-major order
//   - alpha, beta: Scaling factors (output = alpha*packed + beta*output)
//   - packedStride: Row stride in packedOutput (typically params.Nc)
//   - outputRowOffset: Starting row in output matrix
//   - outputColOffset: Starting column in output matrix
//   - outputStride: Row stride in output matrix (N dimension)
//   - height: Number of rows to apply
//   - width: Number of columns to apply
func BaseApplyPackedOutput[T hwy.Floats](
	packedOutput, output []T,
	alpha, beta T,
	packedStride int,
	outputRowOffset, outputColOffset int,
	outputStride int,
	height, width int,
) {
	lanes := hwy.Zero[T]().NumLanes()

	// Create vectors with alpha and beta values
	alphaVec := hwy.Set(alpha)
	betaVec := hwy.Set(beta)

	for r := range height {
		packedIdx := r * packedStride
		outputIdx := (outputRowOffset+r)*outputStride + outputColOffset

		c := 0
		// Vectorized loop: process lanes elements at a time
		for ; c+lanes <= width; c += lanes {
			packedVal := hwy.Load(packedOutput[packedIdx+c:])
			outputVal := hwy.Load(output[outputIdx+c:])

			// output = alpha * packed + beta * output
			// Using MulAdd: result = packedVal * alphaVec + (outputVal * betaVec)
			scaledOutput := hwy.Mul(outputVal, betaVec)
			newVal := hwy.MulAdd(packedVal, alphaVec, scaledOutput)

			hwy.Store(newVal, output[outputIdx+c:])
		}

		// Scalar tail
		for ; c < width; c++ {
			val := packedOutput[packedIdx+c]
			output[outputIdx+c] = beta*output[outputIdx+c] + alpha*val
		}
	}
}

// BaseApplyPackedOutputSimple is a simplified version for alpha=1, beta=0.
//
// When no scaling is needed, this directly copies from packed to output,
// which is faster than the general case.
func BaseApplyPackedOutputSimple[T hwy.Floats](
	packedOutput, output []T,
	packedStride int,
	outputRowOffset, outputColOffset int,
	outputStride int,
	height, width int,
) {
	lanes := hwy.Zero[T]().NumLanes()

	for r := range height {
		packedIdx := r * packedStride
		outputIdx := (outputRowOffset+r)*outputStride + outputColOffset

		c := 0
		// Vectorized copy
		for ; c+lanes <= width; c += lanes {
			v := hwy.Load(packedOutput[packedIdx+c:])
			hwy.Store(v, output[outputIdx+c:])
		}

		// Scalar tail
		for ; c < width; c++ {
			output[outputIdx+c] = packedOutput[packedIdx+c]
		}
	}
}

// BaseApplyPackedOutputAccum is for accumulation (alpha=1, beta=1).
//
// This is the common case when accumulating K-dimension blocks:
// output += packedOutput
func BaseApplyPackedOutputAccum[T hwy.Floats](
	packedOutput, output []T,
	packedStride int,
	outputRowOffset, outputColOffset int,
	outputStride int,
	height, width int,
) {
	lanes := hwy.Zero[T]().NumLanes()

	for r := range height {
		packedIdx := r * packedStride
		outputIdx := (outputRowOffset+r)*outputStride + outputColOffset

		c := 0
		// Vectorized accumulation
		for ; c+lanes <= width; c += lanes {
			packedVal := hwy.Load(packedOutput[packedIdx+c:])
			outputVal := hwy.Load(output[outputIdx+c:])
			newVal := hwy.Add(outputVal, packedVal)
			hwy.Store(newVal, output[outputIdx+c:])
		}

		// Scalar tail
		for ; c < width; c++ {
			output[outputIdx+c] += packedOutput[packedIdx+c]
		}
	}
}
