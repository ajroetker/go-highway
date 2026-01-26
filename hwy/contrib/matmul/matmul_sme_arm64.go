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

//go:build !noasm && darwin && arm64

package matmul

import (
	"sync"
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// Minimum dimensions to use SME FMOPA MatMul
// With pre-transposed A, FMOPA uses contiguous loads and is competitive with NEON.
// Below this threshold, NEON is used (streaming mode has fixed overhead).
const minDimForSME = 64


// Transpose buffer pools to avoid allocations
var transposePool32 = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var transposePool64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var transposePoolF16 = sync.Pool{
	New: func() any {
		return make([]hwy.Float16, 0, 256*256)
	},
}

var transposePoolBF16 = sync.Pool{
	New: func() any {
		return make([]hwy.BFloat16, 0, 256*256)
	},
}

// transposeMatrix transposes M×K matrix A into K×M matrix AT (row-major to column-major)
// AT[k,i] = A[i,k]
// Uses generics to support all float types.
func transposeMatrix[T hwy.Floats](a []T, m, k int, at []T) {
	for i := range m {
		for j := range k {
			at[j*m+i] = a[i*k+j]
		}
	}
}

// matmulFMOPA uses ARM SME FMOPA instruction for matrix multiplication.
// Uses outer product accumulate with ZA tiles - confirmed working on Apple M4!
// Processes matrices in 16x16 tiles using the ZA accumulator.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPA(a, b, c []float32, m, n, k int) {
	// For non-aligned sizes, fall back to generated NEON
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		matmulNEON(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEON(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool32.Get().([]float32)
	if cap(atBuf) < atSize {
		atBuf = make([]float32, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Call FMOPA with transposed A
	asm.MatMulFMOPAF32(atBuf, b, c, m, n, k)

	// Return buffer to pool
	transposePool32.Put(atBuf)
}

// matmulFMOPA64 uses ARM SME FMOPA instruction for float64 matrix multiplication.
// Uses outer product accumulate with ZA tiles - 8×8 tiles for float64.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPA64(a, b, c []float64, m, n, k int) {
	// For non-aligned sizes (8×8 tiles for float64), fall back to scalar
	if m%8 != 0 || n%8 != 0 || k%8 != 0 {
		matmulScalar64(a, b, c, m, n, k)
		return
	}

	// For small matrices, scalar is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulScalar64(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool64.Get().([]float64)
	if cap(atBuf) < atSize {
		atBuf = make([]float64, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Call FMOPA with transposed A
	asm.MatMulFMOPAF64(atBuf, b, c, m, n, k)

	// Return buffer to pool
	transposePool64.Put(atBuf)
}

// matmulFMOPAF16 uses ARM SME FMOPA instruction for float16 matrix multiplication.
// Uses widening: f16 -> f32 FMOPA -> f16, with 16×16 tiles (f32 accumulator).
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPAF16(a, b, c []hwy.Float16, m, n, k int) {
	// For non-aligned sizes (16×16 tiles), fall back to NEON
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		matmulNEONF16(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEONF16(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePoolF16.Get().([]hwy.Float16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.Float16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Scratch buffer for f32->f16 conversion
	var scratch [16]float32
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	// Call FMOPA with transposed A (from asm package)
	asm.MatmulFmopaAtF16(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)

	// Return buffer to pool
	transposePoolF16.Put(atBuf)
}

// matmulFMOPABF16 uses ARM SME BFMOPA instruction for bfloat16 matrix multiplication.
// Uses widening: bf16 -> f32 FMOPA -> bf16, with 16×16 tiles (f32 accumulator).
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPABF16(a, b, c []hwy.BFloat16, m, n, k int) {
	// For non-aligned sizes (16×16 tiles), fall back to NEON
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		matmulNEONBF16(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEONBF16(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePoolBF16.Get().([]hwy.BFloat16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.BFloat16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Scratch buffer for f32->bf16 conversion
	var scratch [16]float32
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	// Call BFMOPA with transposed A (from asm package)
	asm.MatmulBfmopaAtBF16(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)

	// Return buffer to pool
	transposePoolBF16.Put(atBuf)
}

// blockedMatMulFMOPAF16 uses SME FMOPA for blocked float16 matmul.
// This is used by ParallelMatMul for large matrices.
func blockedMatMulFMOPAF16(a, b, c []hwy.Float16, m, n, k int) {
	// The FMOPA implementation handles blocking internally with 16x16 tiles
	matmulFMOPAF16(a, b, c, m, n, k)
}

// blockedMatMulFMOPABF16 uses SME BFMOPA for blocked bfloat16 matmul.
// This is used by ParallelMatMul for large matrices.
func blockedMatMulFMOPABF16(a, b, c []hwy.BFloat16, m, n, k int) {
	// The BFMOPA implementation handles blocking internally with 16x16 tiles
	matmulFMOPABF16(a, b, c, m, n, k)
}

func init() {
	if hwy.HasSME() {
		// Use FMOPA implementation which works on Apple M4
		// This overrides the generated dispatch for large aligned matrices
		MatMulFloat32 = matmulFMOPA
		MatMulFloat64 = matmulFMOPA64
		MatMulFloat16 = matmulFMOPAF16
		MatMulBFloat16 = matmulFMOPABF16

		// Also override blocked matmul for ParallelMatMul usage
		BlockedMatMulFloat16 = blockedMatMulFMOPAF16
		BlockedMatMulBFloat16 = blockedMatMulFMOPABF16
	}
}
