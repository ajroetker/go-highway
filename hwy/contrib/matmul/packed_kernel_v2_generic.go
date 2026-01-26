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

// This file contains the generic micro-kernel that handles arbitrary mr/nr configurations.
// It is NOT processed by hwygen because it uses dynamic slice allocation of vector types,
// which hwygen cannot transform properly.
//
// This is a fallback path that's rarely used (only for non-4x2 configurations),
// so the allocation overhead is acceptable.

import "github.com/ajroetker/go-highway/hwy"

// packedMicroKernelGenericImpl is a generic fallback for non-4x2 configs.
//
// This handles arbitrary mr and nr values, but is slower than the
// specialized 4x2 kernel. Used when the micro-tile dimensions don't
// match the common 4x(2*lanes) pattern.
//
// Note: This function uses hwy.* calls which may allocate on some platforms.
// For the hot path (4x2), use the hwygen-generated PackedMicroKernel4x2 instead.
func packedMicroKernelGenericImpl[T hwy.Floats](
	packedA, packedB []T,
	output []T,
	outputStride int,
	outRowStart, outColStart int,
	panelK, mr, nr, lanes int,
) {
	numBVecs := nr / lanes

	// Allocate accumulators (this path is rarely taken)
	acc := make([]hwy.Vec[T], mr*numBVecs)
	for i := range acc {
		acc[i] = hwy.Zero[T]()
	}

	aIdx := 0
	bIdx := 0

	for p := 0; p < panelK; p++ {
		// Load B vectors
		for v := 0; v < numBVecs; v++ {
			bVec := hwy.Load(packedB[bIdx+v*lanes:])

			// FMA with each A row
			for row := 0; row < mr; row++ {
				aVec := hwy.Set(packedA[aIdx+row])
				accIdx := row*numBVecs + v
				acc[accIdx] = hwy.MulAdd(aVec, bVec, acc[accIdx])
			}
		}
		aIdx += mr
		bIdx += nr
	}

	// Write accumulators
	for row := 0; row < mr; row++ {
		outIdx := (outRowStart+row)*outputStride + outColStart
		for v := 0; v < numBVecs; v++ {
			hwy.Store(acc[row*numBVecs+v], output[outIdx+v*lanes:])
		}
	}
}
