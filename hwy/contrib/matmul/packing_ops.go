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

//go:generate go run ../../../cmd/hwygen -input packing_ops.go -dispatch packingops -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

// BasePackRHSFast packs a panel of the RHS matrix (B) using SIMD when possible.
//
// This is an optimized version of BasePackRHS that uses vector loads/stores
// for full micro-panels where nr matches common SIMD widths.
//
// For AVX-512 with float32 (nr=32), this uses 2x ZMM loads/stores per row.
// For AVX2 with float32 (nr=16), this uses 2x YMM loads/stores per row.
// For NEON with float32 (nr=8), this uses 2x vector loads/stores per row.
//
// Parameters:
//   - b: Input matrix B in row-major order (K x N)
//   - packed: Output buffer for packed data
//   - n: Number of columns in B (row stride)
//   - rowStart: Starting row index in B (K-dimension offset)
//   - colStart: Starting column index in B
//   - panelK: Number of rows to pack (K dimension)
//   - panelCols: Number of columns to pack
//   - nr: Micro-tile column dimension (should match vector width * 2)
func BasePackRHSFast[T hwy.Floats](b, packed []T, n, rowStart, colStart, panelK, panelCols, nr int) {
	lanes := hwy.Zero[T]().NumLanes()
	dstIdx := 0

	// Iterate over strips of width nr
	for stripColIdx := 0; stripColIdx < panelCols; stripColIdx += nr {
		validCols := min(nr, panelCols-stripColIdx)
		baseCol := colStart + stripColIdx

		// Fast path: full strip with SIMD (nr must be multiple of lanes)
		if validCols == nr && nr >= lanes && nr%lanes == 0 {
			for kk := 0; kk < panelK; kk++ {
				srcIdx := (rowStart+kk)*n + baseCol

				// SIMD copy: process nr elements using nr/lanes vectors
				for c := 0; c < nr; c += lanes {
					v := hwy.Load(b[srcIdx+c:])
					hwy.Store(v, packed[dstIdx+c:])
				}
				dstIdx += nr
			}
			continue
		}

		// Fallback: partial strip with scalar copy + zero padding
		for kk := 0; kk < panelK; kk++ {
			srcIdx := (rowStart+kk)*n + baseCol

			// Copy valid columns
			for c := 0; c < validCols; c++ {
				packed[dstIdx] = b[srcIdx+c]
				dstIdx++
			}
			// Zero-pad remaining columns
			for c := validCols; c < nr; c++ {
				packed[dstIdx] = 0
				dstIdx++
			}
		}
	}
}

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

	for r := 0; r < height; r++ {
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

	for r := 0; r < height; r++ {
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

	for r := 0; r < height; r++ {
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
