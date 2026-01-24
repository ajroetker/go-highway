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

//go:generate go run ../../../cmd/hwygen -input packed_kernel_v2.go -dispatch packedkernelv2 -output . -targets avx2,avx512,neon,fallback

import "github.com/ajroetker/go-highway/hwy"

// BasePackedMicroKernel4x2 computes a 4-row × 2-vector micro-tile for the V2 GEBP.
//
// This is the optimized inner kernel for V2, targeting mr=4 and nr=2*lanes.
// It uses 8 accumulator vectors (4 rows × 2 column vectors) that stay in
// registers across the entire K loop.
//
// The V2 kernel writes to a packed output buffer rather than directly to C,
// which eliminates bounds checking in the hot path.
//
// Includes 4x K-loop unrolling for better instruction-level parallelism.
//
// Parameters:
//   - packedA: Packed A micro-panel, size panelK * mr (K-first layout)
//   - packedB: Packed B micro-panel, size panelK * nr (K-first layout)
//   - output: Packed output buffer (not final C matrix)
//   - outputStride: Row stride in output buffer
//   - outRowStart: Starting row in output buffer
//   - outColStart: Starting column in output buffer
//   - panelK: K-dimension of the packed panels
//   - lanes: Vector width in elements (e.g., 8 for AVX2 float32)
func BasePackedMicroKernel4x2[T hwy.Floats](
	packedA, packedB []T,
	output []T,
	outputStride int,
	outRowStart, outColStart int,
	panelK, lanes int,
) {
	// 4 rows × 2 B vectors = 8 accumulator vectors
	acc00 := hwy.Zero[T]()
	acc01 := hwy.Zero[T]()
	acc10 := hwy.Zero[T]()
	acc11 := hwy.Zero[T]()
	acc20 := hwy.Zero[T]()
	acc21 := hwy.Zero[T]()
	acc30 := hwy.Zero[T]()
	acc31 := hwy.Zero[T]()

	nr := 2 * lanes
	mr := 4
	aIdx := 0
	bIdx := 0

	// BCE hints for bounds check elimination
	_ = packedA[panelK*mr-1]
	_ = packedB[panelK*nr-1]

	// Main loop: 4x unrolled for better ILP
	p := 0
	for ; p+3 < panelK; p += 4 {
		// --- Step 0 ---
		bVec0_0 := hwy.Load(packedB[bIdx:])
		bVec1_0 := hwy.Load(packedB[bIdx+lanes:])
		a0_0 := hwy.Set(packedA[aIdx])
		a1_0 := hwy.Set(packedA[aIdx+1])
		a2_0 := hwy.Set(packedA[aIdx+2])
		a3_0 := hwy.Set(packedA[aIdx+3])

		acc00 = hwy.MulAdd(a0_0, bVec0_0, acc00)
		acc01 = hwy.MulAdd(a0_0, bVec1_0, acc01)
		acc10 = hwy.MulAdd(a1_0, bVec0_0, acc10)
		acc11 = hwy.MulAdd(a1_0, bVec1_0, acc11)
		acc20 = hwy.MulAdd(a2_0, bVec0_0, acc20)
		acc21 = hwy.MulAdd(a2_0, bVec1_0, acc21)
		acc30 = hwy.MulAdd(a3_0, bVec0_0, acc30)
		acc31 = hwy.MulAdd(a3_0, bVec1_0, acc31)

		// --- Step 1 ---
		bVec0_1 := hwy.Load(packedB[bIdx+nr:])
		bVec1_1 := hwy.Load(packedB[bIdx+nr+lanes:])
		a0_1 := hwy.Set(packedA[aIdx+mr])
		a1_1 := hwy.Set(packedA[aIdx+mr+1])
		a2_1 := hwy.Set(packedA[aIdx+mr+2])
		a3_1 := hwy.Set(packedA[aIdx+mr+3])

		acc00 = hwy.MulAdd(a0_1, bVec0_1, acc00)
		acc01 = hwy.MulAdd(a0_1, bVec1_1, acc01)
		acc10 = hwy.MulAdd(a1_1, bVec0_1, acc10)
		acc11 = hwy.MulAdd(a1_1, bVec1_1, acc11)
		acc20 = hwy.MulAdd(a2_1, bVec0_1, acc20)
		acc21 = hwy.MulAdd(a2_1, bVec1_1, acc21)
		acc30 = hwy.MulAdd(a3_1, bVec0_1, acc30)
		acc31 = hwy.MulAdd(a3_1, bVec1_1, acc31)

		// --- Step 2 ---
		bVec0_2 := hwy.Load(packedB[bIdx+2*nr:])
		bVec1_2 := hwy.Load(packedB[bIdx+2*nr+lanes:])
		a0_2 := hwy.Set(packedA[aIdx+2*mr])
		a1_2 := hwy.Set(packedA[aIdx+2*mr+1])
		a2_2 := hwy.Set(packedA[aIdx+2*mr+2])
		a3_2 := hwy.Set(packedA[aIdx+2*mr+3])

		acc00 = hwy.MulAdd(a0_2, bVec0_2, acc00)
		acc01 = hwy.MulAdd(a0_2, bVec1_2, acc01)
		acc10 = hwy.MulAdd(a1_2, bVec0_2, acc10)
		acc11 = hwy.MulAdd(a1_2, bVec1_2, acc11)
		acc20 = hwy.MulAdd(a2_2, bVec0_2, acc20)
		acc21 = hwy.MulAdd(a2_2, bVec1_2, acc21)
		acc30 = hwy.MulAdd(a3_2, bVec0_2, acc30)
		acc31 = hwy.MulAdd(a3_2, bVec1_2, acc31)

		// --- Step 3 ---
		bVec0_3 := hwy.Load(packedB[bIdx+3*nr:])
		bVec1_3 := hwy.Load(packedB[bIdx+3*nr+lanes:])
		a0_3 := hwy.Set(packedA[aIdx+3*mr])
		a1_3 := hwy.Set(packedA[aIdx+3*mr+1])
		a2_3 := hwy.Set(packedA[aIdx+3*mr+2])
		a3_3 := hwy.Set(packedA[aIdx+3*mr+3])

		acc00 = hwy.MulAdd(a0_3, bVec0_3, acc00)
		acc01 = hwy.MulAdd(a0_3, bVec1_3, acc01)
		acc10 = hwy.MulAdd(a1_3, bVec0_3, acc10)
		acc11 = hwy.MulAdd(a1_3, bVec1_3, acc11)
		acc20 = hwy.MulAdd(a2_3, bVec0_3, acc20)
		acc21 = hwy.MulAdd(a2_3, bVec1_3, acc21)
		acc30 = hwy.MulAdd(a3_3, bVec0_3, acc30)
		acc31 = hwy.MulAdd(a3_3, bVec1_3, acc31)

		aIdx += 4 * mr
		bIdx += 4 * nr
	}

	// Handle remaining iterations (0-3)
	for ; p < panelK; p++ {
		bVec0 := hwy.Load(packedB[bIdx:])
		bVec1 := hwy.Load(packedB[bIdx+lanes:])
		bIdx += nr

		a0 := hwy.Set(packedA[aIdx])
		a1 := hwy.Set(packedA[aIdx+1])
		a2 := hwy.Set(packedA[aIdx+2])
		a3 := hwy.Set(packedA[aIdx+3])
		aIdx += mr

		acc00 = hwy.MulAdd(a0, bVec0, acc00)
		acc01 = hwy.MulAdd(a0, bVec1, acc01)
		acc10 = hwy.MulAdd(a1, bVec0, acc10)
		acc11 = hwy.MulAdd(a1, bVec1, acc11)
		acc20 = hwy.MulAdd(a2, bVec0, acc20)
		acc21 = hwy.MulAdd(a2, bVec1, acc21)
		acc30 = hwy.MulAdd(a3, bVec0, acc30)
		acc31 = hwy.MulAdd(a3, bVec1, acc31)
	}

	// Write accumulators to output
	outIdx0 := outRowStart*outputStride + outColStart
	outIdx1 := outIdx0 + outputStride
	outIdx2 := outIdx1 + outputStride
	outIdx3 := outIdx2 + outputStride

	hwy.Store(acc00, output[outIdx0:])
	hwy.Store(acc01, output[outIdx0+lanes:])
	hwy.Store(acc10, output[outIdx1:])
	hwy.Store(acc11, output[outIdx1+lanes:])
	hwy.Store(acc20, output[outIdx2:])
	hwy.Store(acc21, output[outIdx2+lanes:])
	hwy.Store(acc30, output[outIdx3:])
	hwy.Store(acc31, output[outIdx3+lanes:])
}

// BaseZeroSlice zeros a slice using SIMD.
//
// This is used to clear the packed output buffer before accumulating
// micro-kernel results.
func BaseZeroSlice[T hwy.Floats](s []T, n int) {
	vZero := hwy.Zero[T]()
	lanes := vZero.NumLanes()

	var idx int
	for idx = 0; idx+lanes <= n; idx += lanes {
		hwy.Store(vZero, s[idx:])
	}
	for ; idx < n; idx++ {
		s[idx] = 0
	}
}
