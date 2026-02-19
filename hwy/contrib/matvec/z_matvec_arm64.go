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

package matvec

import (
	"sync"
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// NOTE: SME implementation for matrix-vector multiplication.
//
// For matvec (result = M * v), we use FMOPA to process 16 rows at a time:
//   - FMOPA computes: za[i][j] += z1[i] * z2[j]
//   - Load column M[row:row+16, k] into z1
//   - Broadcast v[k] into z2
//   - After all k: za[i][0] = dot(M_row[row+i], v)
//
// To enable contiguous column access, we pre-transpose M:
//   - MT[k][i] = M[i][k] (cols×rows, row-major)
//   - Column M[row:row+16, k] becomes row MT[k, row:row+16]
//
// Apple M4 SVL = 512 bits, meaning:
//   - Z registers hold 16 × float32 or 8 × float64
//   - ZA tiles are 16×16 = 256 float32 or 8×8 = 64 float64 elements
//   - Single FMOPA does 16×16×2 = 512 FP32 ops

// Dimension thresholds for SME MatVec
// SME is faster for smaller matrices (64-192) due to 512-bit FMOPA,
// but slower for larger matrices due to transpose overhead.
// NEON's direct dot product approach is more efficient for large matrices.
const (
	minDimForSMEMatVec = 64  // Minimum dimension to use SME
	maxDimForSMEMatVec = 192 // Maximum dimension - above this NEON is faster
)

//go:noescape
func matvec_sme_f32(mt, v, result unsafe.Pointer, rows, cols int64)

//go:noescape
func matvec_sme_f64(mt, v, result unsafe.Pointer, rows, cols int64)

// alignUp rounds m up to the next multiple of tileSize.
func alignUp(m, tileSize int) int {
	return (m + tileSize - 1) / tileSize * tileSize
}

// Buffer pools to avoid allocations
var matvecTransposePool = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var matvecTransposePool64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var matvecPadPool = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var matvecPadPool64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var matvecResultPool = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256)
	},
}

var matvecResultPool64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256)
	},
}

// transposeForMatVec transposes rows×cols matrix M into cols×rows matrix MT
// MT[k,i] = M[i,k]
func transposeForMatVec(m []float32, rows, cols int, mt []float32) {
	for i := range rows {
		for k := range cols {
			mt[k*rows+i] = m[i*cols+k]
		}
	}
}

// transposeForMatVec64 transposes rows×cols matrix M into cols×rows matrix MT for float64
func transposeForMatVec64(m []float64, rows, cols int, mt []float64) {
	for i := range rows {
		for k := range cols {
			mt[k*rows+i] = m[i*cols+k]
		}
	}
}

// matvecSME uses ARM SME FMOPA instruction for matrix-vector multiplication.
// Uses outer product accumulate with ZA tiles.
// Pre-transposes M for contiguous column access, enabling fast vector loads.
// Non-aligned row counts are handled by padding up to the tile size.
func matvecSME(m []float32, rows, cols int, v, result []float32) {
	// Bounds check: ensure slices are large enough
	_ = m[rows*cols-1]
	_ = v[cols-1]
	_ = result[rows-1]

	const tileSize = 16
	paddedRows := alignUp(rows, tileSize)

	// SME is faster for medium-sized matrices (64-192)
	// For smaller matrices, streaming mode overhead dominates
	// For larger matrices, transpose cost makes NEON faster
	if paddedRows < minDimForSMEMatVec || cols < minDimForSMEMatVec ||
		paddedRows > maxDimForSMEMatVec || cols > maxDimForSMEMatVec {
		matVecAsmF32(m, rows, cols, v, result)
		return
	}

	needsPad := paddedRows != rows

	// Prepare M: [rows, cols] → [paddedRows, cols] if needed
	smeM := m
	smeRows := rows
	if needsPad {
		padSize := paddedRows * cols
		padBuf := matvecPadPool.Get().([]float32)
		if cap(padBuf) < padSize {
			padBuf = make([]float32, padSize)
		} else {
			padBuf = padBuf[:padSize]
		}
		// Copy original rows, zero-pad extra rows
		copy(padBuf[:rows*cols], m[:rows*cols])
		clear(padBuf[rows*cols : paddedRows*cols])
		smeM = padBuf
		smeRows = paddedRows
		defer matvecPadPool.Put(padBuf)
	}

	// Get transpose buffer from pool
	mtSize := smeRows * cols
	mtBuf := matvecTransposePool.Get().([]float32)
	if cap(mtBuf) < mtSize {
		mtBuf = make([]float32, mtSize)
	} else {
		mtBuf = mtBuf[:mtSize]
	}

	// Transpose M (smeRows×cols) to MT (cols×smeRows) for contiguous column access
	transposeForMatVec(smeM, smeRows, cols, mtBuf)

	if needsPad {
		// Use padded result buffer for SME, then copy back
		resBuf := matvecResultPool.Get().([]float32)
		if cap(resBuf) < smeRows {
			resBuf = make([]float32, smeRows)
		} else {
			resBuf = resBuf[:smeRows]
		}
		matvec_sme_f32(
			unsafe.Pointer(unsafe.SliceData(mtBuf)),
			unsafe.Pointer(unsafe.SliceData(v)),
			unsafe.Pointer(unsafe.SliceData(resBuf)),
			int64(smeRows),
			int64(cols),
		)
		copy(result[:rows], resBuf[:rows])
		matvecResultPool.Put(resBuf)
	} else {
		// Call SME FMOPA with transposed M directly
		matvec_sme_f32(
			unsafe.Pointer(unsafe.SliceData(mtBuf)),
			unsafe.Pointer(unsafe.SliceData(v)),
			unsafe.Pointer(unsafe.SliceData(result)),
			int64(smeRows),
			int64(cols),
		)
	}

	// Return buffer to pool
	matvecTransposePool.Put(mtBuf)
}

// matvecSME64 uses ARM SME FMOPA instruction for float64 matrix-vector multiplication.
// Uses 8×8 tiles for float64.
// Non-aligned row counts are handled by padding up to the tile size.
func matvecSME64(m []float64, rows, cols int, v, result []float64) {
	// Bounds check: ensure slices are large enough
	_ = m[rows*cols-1]
	_ = v[cols-1]
	_ = result[rows-1]

	const tileSize = 8
	paddedRows := alignUp(rows, tileSize)

	// SME is faster for medium-sized matrices (64-192)
	// For smaller matrices, streaming mode overhead dominates
	// For larger matrices, transpose cost makes NEON faster
	if paddedRows < minDimForSMEMatVec || cols < minDimForSMEMatVec ||
		paddedRows > maxDimForSMEMatVec || cols > maxDimForSMEMatVec {
		matVecAsmF64(m, rows, cols, v, result)
		return
	}

	needsPad := paddedRows != rows

	// Prepare M: [rows, cols] → [paddedRows, cols] if needed
	smeM := m
	smeRows := rows
	if needsPad {
		padSize := paddedRows * cols
		padBuf := matvecPadPool64.Get().([]float64)
		if cap(padBuf) < padSize {
			padBuf = make([]float64, padSize)
		} else {
			padBuf = padBuf[:padSize]
		}
		copy(padBuf[:rows*cols], m[:rows*cols])
		clear(padBuf[rows*cols : paddedRows*cols])
		smeM = padBuf
		smeRows = paddedRows
		defer matvecPadPool64.Put(padBuf)
	}

	// Get transpose buffer from pool
	mtSize := smeRows * cols
	mtBuf := matvecTransposePool64.Get().([]float64)
	if cap(mtBuf) < mtSize {
		mtBuf = make([]float64, mtSize)
	} else {
		mtBuf = mtBuf[:mtSize]
	}

	// Transpose M (smeRows×cols) to MT (cols×smeRows) for contiguous column access
	transposeForMatVec64(smeM, smeRows, cols, mtBuf)

	if needsPad {
		// Use padded result buffer for SME, then copy back
		resBuf := matvecResultPool64.Get().([]float64)
		if cap(resBuf) < smeRows {
			resBuf = make([]float64, smeRows)
		} else {
			resBuf = resBuf[:smeRows]
		}
		matvec_sme_f64(
			unsafe.Pointer(unsafe.SliceData(mtBuf)),
			unsafe.Pointer(unsafe.SliceData(v)),
			unsafe.Pointer(unsafe.SliceData(resBuf)),
			int64(smeRows),
			int64(cols),
		)
		copy(result[:rows], resBuf[:rows])
		matvecResultPool64.Put(resBuf)
	} else {
		// Call SME FMOPA with transposed M directly
		matvec_sme_f64(
			unsafe.Pointer(unsafe.SliceData(mtBuf)),
			unsafe.Pointer(unsafe.SliceData(v)),
			unsafe.Pointer(unsafe.SliceData(result)),
			int64(smeRows),
			int64(cols),
		)
	}

	// Return buffer to pool
	matvecTransposePool64.Put(mtBuf)
}

func init() {
	if hwy.HasSME() {
		// Use SME FMOPA implementation for large aligned matrices
		// This overrides the generated dispatch
		MatVecFloat32 = matvecSME
		MatVecFloat64 = matvecSME64
	}
}
