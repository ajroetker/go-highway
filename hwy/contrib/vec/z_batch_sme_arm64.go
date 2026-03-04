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

package vec

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matvec"
)

// Dimension thresholds for SME batch operations.
// Based on Apple M4 Max benchmarking:
//
// MatVec FMOPA (for BatchDot only):
//   - dims 128-192 with count >= 256: ~15% faster than GoAT loop
//   - Below 128d or above 192d: transpose overhead exceeds tile benefits
//   - Below count 256: not enough work to amortize transpose cost
//
// SME Streaming mode: Not used. Benchmarks show GoAT loop is 20-60% faster
// at all dimensions due to smstart/smstop overhead.
const (
	// MatVec thresholds (uses transpose + FMOPA tiles)
	minDimForMatVecBatch   = 128 // Minimum dimension for MatVec SME
	maxDimForMatVecBatch   = 192 // Maximum - above this transpose hurts
	minCountForMatVecBatch = 256 // Minimum count to amortize transpose overhead
)

// batchDotSME dispatches to the most efficient implementation based on dimensions.
// - Dims 128-192 with count >= 256 and 16-aligned: MatVec FMOPA (~15% faster)
// - Otherwise: Loop over optimized GoAT single-vector dot products
func batchDotSME(query []float32, data []float32, dots []float32, count int, dims int) {
	// Validate inputs
	if count <= 0 || dims <= 0 {
		return
	}
	if len(data) < count*dims || len(dots) < count || len(query) < dims {
		return
	}

	// MatVec FMOPA: Only beneficial in narrow range of dims with large counts
	// MatVec computes: result = M * v where M is (count × dims) and v is (dims,)
	// This is exactly BatchDot: dots[i] = dot(data[i*dims:], query)
	if dims >= minDimForMatVecBatch && dims <= maxDimForMatVecBatch &&
		count >= minCountForMatVecBatch && count%16 == 0 {
		matvec.MatVecFloat32(data, count, dims, query, dots)
		return
	}

	// Default: Loop over optimized GoAT single-vector dot products
	// This uses the register-based GoAT assembly with 16 lanes (SVL=512)
	for i := range count {
		offset := i * dims
		dots[i] = DotFloat32(query, data[offset:offset+dims])
	}
}

// batchDotSME64 dispatches to the most efficient float64 implementation.
func batchDotSME64(query []float64, data []float64, dots []float64, count int, dims int) {
	// Validate inputs
	if count <= 0 || dims <= 0 {
		return
	}
	if len(data) < count*dims || len(dots) < count || len(query) < dims {
		return
	}

	// MatVec FMOPA: Only beneficial in narrow range of dims with large counts
	// float64 uses 8-aligned count (8 doubles per ZA tile row)
	if dims >= minDimForMatVecBatch && dims <= maxDimForMatVecBatch &&
		count >= minCountForMatVecBatch && count%8 == 0 {
		matvec.MatVecFloat64(data, count, dims, query, dots)
		return
	}

	// Default: Loop over optimized GoAT single-vector dot products
	for i := range count {
		offset := i * dims
		dots[i] = DotFloat64(query, data[offset:offset+dims])
	}
}

// batchL2SquaredSME dispatches to the most efficient L2 squared implementation.
//
// Note: We evaluated using the identity L2²(a,b) = ||a||² + ||b||² - 2(a·b) to leverage
// MatVec FMOPA, but benchmarks show the GoAT loop is 1.5-2x faster. The FMOPA approach
// requires computing ||data[i]||² for each vector separately, which negates the FMOPA
// benefit. The GoAT L2SquaredDistanceF32 computes everything in a single pass with
// better cache utilization.
func batchL2SquaredSME(query []float32, data []float32, distances []float32, count int, dims int) {
	// Validate inputs
	if count <= 0 || dims <= 0 {
		return
	}
	if len(data) < count*dims || len(distances) < count || len(query) < dims {
		return
	}

	// Loop over optimized GoAT single-vector L2 squared distance
	// Benchmarks show this is faster than FMOPA for all dimension/count combinations
	for i := range count {
		offset := i * dims
		distances[i] = L2SquaredDistanceFloat32(query, data[offset:offset+dims])
	}
}

// batchL2SquaredSME64 dispatches to the most efficient float64 L2 squared implementation.
// See batchL2SquaredSME for why FMOPA is not used.
func batchL2SquaredSME64(query []float64, data []float64, distances []float64, count int, dims int) {
	// Validate inputs
	if count <= 0 || dims <= 0 {
		return
	}
	if len(data) < count*dims || len(distances) < count || len(query) < dims {
		return
	}

	// Loop over optimized GoAT single-vector L2 squared distance
	for i := range count {
		offset := i * dims
		distances[i] = L2SquaredDistanceFloat64(query, data[offset:offset+dims])
	}
}

func init() {
	if hwy.HasSME() {
		// Override generated dispatch with SME-aware versions
		BatchDotFloat32 = batchDotSME
		BatchDotFloat64 = batchDotSME64
		BatchL2SquaredDistanceFloat32 = batchL2SquaredSME
		BatchL2SquaredDistanceFloat64 = batchL2SquaredSME64
	}
}
