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

package wavelet

import (
	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go run ../../../cmd/hwygen -input lifting_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch lifting

// BaseLiftUpdate53 applies the 5/3 update step: target[i] -= (neighbor[i+off1] + neighbor[i+off2] + 2) >> 2
// This is used for the update step in 5/3 synthesis and the predict step in analysis.
// The phase parameter determines the offset pattern for neighbor access.
func BaseLiftUpdate53[T hwy.SignedInts](target []T, tLen int, neighbor []T, nLen int, phase int) {
	if tLen == 0 || nLen == 0 {
		return
	}

	twoVec := hwy.Set(T(2))
	lanes := hwy.MaxLanes[T]()

	// phase=0: target[i] -= (neighbor[i-1] + neighbor[i] + 2) >> 2
	// phase=1: target[i] -= (neighbor[i] + neighbor[i+1] + 2) >> 2
	//
	// For phase=0: n1 = neighbor[i-1] (left-shifted), n2 = neighbor[i] (aligned)
	//   boundary: i=0 clamps n1 to neighbor[0]
	//   bulk: i=1..min(tLen,nLen)-1, both indices in [0, nLen-1]
	//   boundary: last element if i >= nLen, clamp n2 to neighbor[nLen-1]
	//
	// For phase=1: n1 = neighbor[i] (aligned), n2 = neighbor[i+1] (right-shifted)
	//   bulk: i=0..min(tLen,nLen-1)-1, both indices in [0, nLen-1]
	//   boundary: last element(s) if i+1 >= nLen, clamp n2 to neighbor[nLen-1]

	// Handle first boundary element for phase=0
	start := 0
	if phase == 0 {
		// i=0: n1=neighbor[0], n2=neighbor[0] (clamped)
		target[0] -= (neighbor[0] + neighbor[0] + 2) >> 2
		start = 1
	}

	// Determine the safe range where both neighbor accesses are in bounds
	// phase=0: need i-1 >= 0 (start=1) and i < nLen
	// phase=1: need i >= 0 and i+1 < nLen, so i < nLen-1
	safeEnd := tLen
	if phase == 0 {
		if nLen < safeEnd {
			safeEnd = nLen
		}
	} else {
		if nLen-1 < safeEnd {
			safeEnd = nLen - 1
		}
	}

	// Bulk SIMD loop for safe range
	i := start
	for ; i+lanes <= safeEnd; i += lanes {
		var n1, n2 hwy.Vec[T]
		if phase == 0 {
			n1 = hwy.Load(neighbor[i-1:])
			n2 = hwy.Load(neighbor[i:])
		} else {
			n1 = hwy.Load(neighbor[i:])
			n2 = hwy.Load(neighbor[i+1:])
		}
		sum := hwy.Add(hwy.Add(n1, n2), twoVec)
		update := hwy.ShiftRight(sum, 2)
		t := hwy.Load(target[i:])
		hwy.Store(hwy.Sub(t, update), target[i:])
	}

	// Scalar remainder within safe range
	for ; i < safeEnd; i++ {
		var n1Idx, n2Idx int
		if phase == 0 {
			n1Idx = i - 1
			n2Idx = i
		} else {
			n1Idx = i
			n2Idx = i + 1
		}
		target[i] -= (neighbor[n1Idx] + neighbor[n2Idx] + 2) >> 2
	}

	// Scalar tail with boundary clamping
	for ; i < tLen; i++ {
		var n1Idx, n2Idx int
		if phase == 0 {
			n1Idx = i - 1
			n2Idx = i
		} else {
			n1Idx = i
			n2Idx = i + 1
		}
		if n1Idx >= nLen {
			n1Idx = nLen - 1
		}
		if n2Idx >= nLen {
			n2Idx = nLen - 1
		}
		target[i] -= (neighbor[n1Idx] + neighbor[n2Idx] + 2) >> 2
	}
	_ = lanes
}

// BaseLiftPredict53 applies the 5/3 predict step: target[i] += (neighbor[i+off1] + neighbor[i+off2]) >> 1
// This is used for the predict step in 5/3 synthesis and the update step in analysis.
func BaseLiftPredict53[T hwy.SignedInts](target []T, tLen int, neighbor []T, nLen int, phase int) {
	if tLen == 0 || nLen == 0 {
		return
	}

	lanes := hwy.MaxLanes[T]()

	// phase=0: target[i] += (neighbor[i] + neighbor[i+1]) >> 1
	// phase=1: target[i] += (neighbor[i-1] + neighbor[i]) >> 1

	// Handle first boundary element for phase=1
	start := 0
	if phase == 1 {
		// i=0: n1=neighbor[-1] clamped to neighbor[0], n2=neighbor[0]
		target[0] += (neighbor[0] + neighbor[0]) >> 1
		start = 1
	}

	// Safe range where both accesses are in bounds
	safeEnd := tLen
	if phase == 0 {
		if nLen-1 < safeEnd {
			safeEnd = nLen - 1
		}
	} else {
		if nLen < safeEnd {
			safeEnd = nLen
		}
	}

	// Bulk SIMD loop
	i := start
	for ; i+lanes <= safeEnd; i += lanes {
		var n1, n2 hwy.Vec[T]
		if phase == 0 {
			n1 = hwy.Load(neighbor[i:])
			n2 = hwy.Load(neighbor[i+1:])
		} else {
			n1 = hwy.Load(neighbor[i-1:])
			n2 = hwy.Load(neighbor[i:])
		}
		update := hwy.ShiftRight(hwy.Add(n1, n2), 1)
		t := hwy.Load(target[i:])
		hwy.Store(hwy.Add(t, update), target[i:])
	}

	// Scalar remainder within safe range
	for ; i < safeEnd; i++ {
		var n1Idx, n2Idx int
		if phase == 0 {
			n1Idx = i
			n2Idx = i + 1
		} else {
			n1Idx = i - 1
			n2Idx = i
		}
		target[i] += (neighbor[n1Idx] + neighbor[n2Idx]) >> 1
	}

	// Scalar tail with boundary clamping
	for ; i < tLen; i++ {
		var n1Idx, n2Idx int
		if phase == 0 {
			n1Idx = i
			n2Idx = i + 1
		} else {
			n1Idx = i - 1
			n2Idx = i
		}
		if n1Idx < 0 {
			n1Idx = 0
		}
		if n1Idx >= nLen {
			n1Idx = nLen - 1
		}
		if n2Idx >= nLen {
			n2Idx = nLen - 1
		}
		target[i] += (neighbor[n1Idx] + neighbor[n2Idx]) >> 1
	}
	_ = lanes
}

// BaseLiftStep97 applies a generic 9/7 lifting step: target[i] -= coeff * (neighbor[i+off1] + neighbor[i+off2])
// This is used for all four lifting steps in 9/7 transforms.
func BaseLiftStep97[T hwy.Floats](target []T, tLen int, neighbor []T, nLen int, coeff T, phase int) {
	if tLen == 0 || nLen == 0 {
		return
	}

	coeffVec := hwy.Set(coeff)
	lanes := hwy.MaxLanes[T]()

	// phase=0: target[i] -= coeff * (neighbor[i] + neighbor[i+1])
	// phase=1: target[i] -= coeff * (neighbor[i-1] + neighbor[i])

	// Handle first boundary element for phase=1
	start := 0
	if phase == 1 {
		// i=0: neighbor[-1] clamped to neighbor[0]
		target[0] -= coeff * (neighbor[0] + neighbor[0])
		start = 1
	}

	// Safe range where both accesses are in bounds
	safeEnd := tLen
	if phase == 0 {
		if nLen-1 < safeEnd {
			safeEnd = nLen - 1
		}
	} else {
		if nLen < safeEnd {
			safeEnd = nLen
		}
	}

	// Bulk SIMD loop for safe range
	i := start
	for ; i+lanes <= safeEnd; i += lanes {
		var n1, n2 hwy.Vec[T]
		if phase == 0 {
			n1 = hwy.Load(neighbor[i:])
			n2 = hwy.Load(neighbor[i+1:])
		} else {
			n1 = hwy.Load(neighbor[i-1:])
			n2 = hwy.Load(neighbor[i:])
		}
		sum := hwy.Add(n1, n2)
		update := hwy.Mul(coeffVec, sum)
		t := hwy.Load(target[i:])
		hwy.Store(hwy.Sub(t, update), target[i:])
	}

	// Scalar remainder within safe range
	for ; i < safeEnd; i++ {
		var n1Idx, n2Idx int
		if phase == 0 {
			n1Idx = i
			n2Idx = i + 1
		} else {
			n1Idx = i - 1
			n2Idx = i
		}
		target[i] -= coeff * (neighbor[n1Idx] + neighbor[n2Idx])
	}

	// Scalar tail with boundary clamping
	for ; i < tLen; i++ {
		var n1Idx, n2Idx int
		if phase == 0 {
			n1Idx = i
			n2Idx = i + 1
		} else {
			n1Idx = i - 1
			n2Idx = i
		}
		if n1Idx < 0 {
			n1Idx = 0
		}
		if n1Idx >= nLen {
			n1Idx = nLen - 1
		}
		if n2Idx >= nLen {
			n2Idx = nLen - 1
		}
		target[i] -= coeff * (neighbor[n1Idx] + neighbor[n2Idx])
	}
	_ = coeffVec
	_ = lanes
}

// BaseScaleSlice multiplies all elements by a scale factor: data[i] *= scale
func BaseScaleSlice[T hwy.Floats](data []T, n int, scale T) {
	if n == 0 || data == nil {
		return
	}

	scaleVec := hwy.Set(scale)
	lanes := hwy.MaxLanes[T]()
	i := 0

	// Process full vectors
	for ; i+lanes <= n; i += lanes {
		v := hwy.Load(data[i:])
		result := hwy.Mul(v, scaleVec)
		hwy.Store(result, data[i:])
	}

	// Scalar tail — no buffer needed, avoids OOB writes on bare slices.
	for ; i < n; i++ {
		data[i] *= scale
	}
}

// BaseInterleave interleaves low and high-pass coefficients into dst.
// phase=0: dst[2i]=low[i], dst[2i+1]=high[i]
// phase=1: dst[2i]=high[i], dst[2i+1]=low[i]
func BaseInterleave[T hwy.Lanes](dst []T, low []T, sn int, high []T, dn int, phase int) {
	if phase == 0 {
		// Even-first: low at even indices, high at odd indices
		for i := 0; i < sn && i < dn; i++ {
			dst[2*i] = low[i]
			dst[2*i+1] = high[i]
		}
		// Handle remaining low (when sn > dn)
		for i := dn; i < sn; i++ {
			dst[2*i] = low[i]
		}
		// Handle remaining high (when dn > sn)
		for i := sn; i < dn; i++ {
			dst[2*i+1] = high[i]
		}
	} else {
		// Odd-first: high at even indices, low at odd indices
		for i := 0; i < sn && i < dn; i++ {
			dst[2*i] = high[i]
			dst[2*i+1] = low[i]
		}
		for i := dn; i < sn; i++ {
			dst[2*i+1] = low[i]
		}
		for i := sn; i < dn; i++ {
			dst[2*i] = high[i]
		}
	}
}

// BaseSynthesize53Core fuses copy + LiftUpdate53 + LiftPredict53 + Interleave
// into a single dispatch target, eliminating 2 indirect calls per 1D transform.
// data contains [low|high] on entry and interleaved samples on exit.
// low and high are scratch buffers with capacity >= sn and >= dn respectively.
func BaseSynthesize53Core[T hwy.SignedInts](data []T, n int, low []T, sn int, high []T, dn int, phase int) {
	// 1. Copy subbands from data into scratch buffers.
	// Use explicit loops instead of copy() so the C emitter emits
	// length-dependent loops rather than fixed-lane-count loops.
	for ci := 0; ci < sn; ci++ {
		low[ci] = data[ci]
	}
	for ci := 0; ci < dn; ci++ {
		high[ci] = data[sn+ci]
	}

	// 2. LiftUpdate53: low[i] -= (high[off1] + high[off2] + 2) >> 2
	{
		twoVec := hwy.Set(T(2))
		lanes := hwy.MaxLanes[T]()

		start := 0
		if phase == 0 {
			low[0] -= (high[0] + high[0] + 2) >> 2
			start = 1
		}

		safeEnd := sn
		if phase == 0 {
			if dn < safeEnd {
				safeEnd = dn
			}
		} else {
			if dn-1 < safeEnd {
				safeEnd = dn - 1
			}
		}

		i := start
		for ; i+lanes <= safeEnd; i += lanes {
			var n1, n2 hwy.Vec[T]
			if phase == 0 {
				n1 = hwy.Load(high[i-1:])
				n2 = hwy.Load(high[i:])
			} else {
				n1 = hwy.Load(high[i:])
				n2 = hwy.Load(high[i+1:])
			}
			sum := hwy.Add(hwy.Add(n1, n2), twoVec)
			update := hwy.ShiftRight(sum, 2)
			t := hwy.Load(low[i:])
			hwy.Store(hwy.Sub(t, update), low[i:])
		}

		for ; i < safeEnd; i++ {
			var n1Idx, n2Idx int
			if phase == 0 {
				n1Idx = i - 1
				n2Idx = i
			} else {
				n1Idx = i
				n2Idx = i + 1
			}
			low[i] -= (high[n1Idx] + high[n2Idx] + 2) >> 2
		}

		for ; i < sn; i++ {
			var n1Idx, n2Idx int
			if phase == 0 {
				n1Idx = i - 1
				n2Idx = i
			} else {
				n1Idx = i
				n2Idx = i + 1
			}
			if n1Idx >= dn {
				n1Idx = dn - 1
			}
			if n2Idx >= dn {
				n2Idx = dn - 1
			}
			low[i] -= (high[n1Idx] + high[n2Idx] + 2) >> 2
		}
		_ = twoVec
		_ = lanes
	}

	// 3. LiftPredict53: high[i] += (low[off1] + low[off2]) >> 1
	{
		lanes := hwy.MaxLanes[T]()

		start := 0
		if phase == 1 {
			high[0] += (low[0] + low[0]) >> 1
			start = 1
		}

		safeEnd := dn
		if phase == 0 {
			if sn-1 < safeEnd {
				safeEnd = sn - 1
			}
		} else {
			if sn < safeEnd {
				safeEnd = sn
			}
		}

		i := start
		for ; i+lanes <= safeEnd; i += lanes {
			var n1, n2 hwy.Vec[T]
			if phase == 0 {
				n1 = hwy.Load(low[i:])
				n2 = hwy.Load(low[i+1:])
			} else {
				n1 = hwy.Load(low[i-1:])
				n2 = hwy.Load(low[i:])
			}
			update := hwy.ShiftRight(hwy.Add(n1, n2), 1)
			t := hwy.Load(high[i:])
			hwy.Store(hwy.Add(t, update), high[i:])
		}

		for ; i < safeEnd; i++ {
			var n1Idx, n2Idx int
			if phase == 0 {
				n1Idx = i
				n2Idx = i + 1
			} else {
				n1Idx = i - 1
				n2Idx = i
			}
			high[i] += (low[n1Idx] + low[n2Idx]) >> 1
		}

		for ; i < dn; i++ {
			var n1Idx, n2Idx int
			if phase == 0 {
				n1Idx = i
				n2Idx = i + 1
			} else {
				n1Idx = i - 1
				n2Idx = i
			}
			if n1Idx < 0 {
				n1Idx = 0
			}
			if n1Idx >= sn {
				n1Idx = sn - 1
			}
			if n2Idx >= sn {
				n2Idx = sn - 1
			}
			high[i] += (low[n1Idx] + low[n2Idx]) >> 1
		}
		_ = lanes
	}

	// 4. Interleave with SIMD for phase=0
	if phase == 0 {
		lanes := hwy.MaxLanes[T]()
		minN := min(sn, dn)

		// SIMD bulk: process lanes elements at a time, producing 2*lanes outputs
		i := 0
		for ; i+lanes <= minN; i += lanes {
			lo := hwy.Load(low[i:])
			hi := hwy.Load(high[i:])
			z0 := hwy.InterleaveLower(lo, hi)
			z1 := hwy.InterleaveUpper(lo, hi)
			hwy.Store(z0, data[2*i:])
			hwy.Store(z1, data[2*i+lanes:])
		}

		// Scalar tail for paired elements
		for ; i < minN; i++ {
			data[2*i] = low[i]
			data[2*i+1] = high[i]
		}
		// Remaining low (sn > dn)
		for i := dn; i < sn; i++ {
			data[2*i] = low[i]
		}
		_ = lanes
	} else {
		// phase=1: scalar interleave (rare path, odd-phase tiles)
		minN := min(sn, dn)
		for i := range minN {
			data[2*i] = high[i]
			data[2*i+1] = low[i]
		}
		for i := dn; i < sn; i++ {
			data[2*i+1] = low[i]
		}
		for i := sn; i < dn; i++ {
			data[2*i] = high[i]
		}
	}
}

// BaseSynthesize53CoreCols fuses copy + LiftUpdate53 + LiftPredict53 + Interleave
// for multiple columns processed in parallel via SIMD. The data layout is
// column-interleaved: colBuf[y*lanes + c] holds the value for row y, column c
// (where lanes = hwy.MaxLanes). Each SIMD load/store at colBuf[y*lanes:]
// touches all `lanes` columns for row y simultaneously.
//
// colBuf has height*lanes elements on entry ([low rows | high rows] interleaved
// by column) and contains the synthesized output on exit (interleaved rows).
// lowBuf and highBuf are scratch buffers with capacity >= sn*lanes and dn*lanes.
func BaseSynthesize53CoreCols[T hwy.SignedInts](colBuf []T, height int, lowBuf []T, sn int, highBuf []T, dn int, phase int) {
	lanes := hwy.MaxLanes[T]()

	// 1. Copy subbands from colBuf into scratch buffers.
	// colBuf layout: rows [0..sn-1] are low-pass, rows [sn..sn+dn-1] are high-pass.
	for y := 0; y < sn; y++ {
		copy(lowBuf[y*lanes:y*lanes+lanes], colBuf[y*lanes:y*lanes+lanes])
	}
	for y := 0; y < dn; y++ {
		copy(highBuf[y*lanes:y*lanes+lanes], colBuf[(sn+y)*lanes:(sn+y)*lanes+lanes])
	}

	// 2. LiftUpdate53 across columns: low[y] -= (high[off1] + high[off2] + 2) >> 2
	{
		twoVec := hwy.Set(T(2))

		start := 0
		if phase == 0 {
			// Boundary: y=0, clamp high[-1] to high[0]
			h0 := hwy.Load(highBuf[0:])
			l0 := hwy.Load(lowBuf[0:])
			hwy.Store(hwy.Sub(l0, hwy.ShiftRight(hwy.Add(hwy.Add(h0, h0), twoVec), 2)), lowBuf[0:])
			start = 1
		}

		safeEnd := sn
		if phase == 0 {
			if dn < safeEnd {
				safeEnd = dn
			}
		} else {
			if dn-1 < safeEnd {
				safeEnd = dn - 1
			}
		}

		for y := start; y < safeEnd; y++ {
			var n1, n2 hwy.Vec[T]
			if phase == 0 {
				n1 = hwy.Load(highBuf[(y-1)*lanes:])
				n2 = hwy.Load(highBuf[y*lanes:])
			} else {
				n1 = hwy.Load(highBuf[y*lanes:])
				n2 = hwy.Load(highBuf[(y+1)*lanes:])
			}
			sum := hwy.Add(hwy.Add(n1, n2), twoVec)
			update := hwy.ShiftRight(sum, 2)
			t := hwy.Load(lowBuf[y*lanes:])
			hwy.Store(hwy.Sub(t, update), lowBuf[y*lanes:])
		}

		// Tail with boundary clamping
		for y := safeEnd; y < sn; y++ {
			var n1Idx, n2Idx int
			if phase == 0 {
				n1Idx = y - 1
				n2Idx = y
			} else {
				n1Idx = y
				n2Idx = y + 1
			}
			if n1Idx >= dn {
				n1Idx = dn - 1
			}
			if n2Idx >= dn {
				n2Idx = dn - 1
			}
			n1 := hwy.Load(highBuf[n1Idx*lanes:])
			n2 := hwy.Load(highBuf[n2Idx*lanes:])
			sum := hwy.Add(hwy.Add(n1, n2), twoVec)
			update := hwy.ShiftRight(sum, 2)
			t := hwy.Load(lowBuf[y*lanes:])
			hwy.Store(hwy.Sub(t, update), lowBuf[y*lanes:])
		}
		_ = twoVec
	}

	// 3. LiftPredict53 across columns: high[y] += (low[off1] + low[off2]) >> 1
	{
		start := 0
		if phase == 1 {
			// Boundary: y=0, clamp low[-1] to low[0]
			l0 := hwy.Load(lowBuf[0:])
			h0 := hwy.Load(highBuf[0:])
			hwy.Store(hwy.Add(h0, hwy.ShiftRight(hwy.Add(l0, l0), 1)), highBuf[0:])
			start = 1
		}

		safeEnd := dn
		if phase == 0 {
			if sn-1 < safeEnd {
				safeEnd = sn - 1
			}
		} else {
			if sn < safeEnd {
				safeEnd = sn
			}
		}

		for y := start; y < safeEnd; y++ {
			var n1, n2 hwy.Vec[T]
			if phase == 0 {
				n1 = hwy.Load(lowBuf[y*lanes:])
				n2 = hwy.Load(lowBuf[(y+1)*lanes:])
			} else {
				n1 = hwy.Load(lowBuf[(y-1)*lanes:])
				n2 = hwy.Load(lowBuf[y*lanes:])
			}
			update := hwy.ShiftRight(hwy.Add(n1, n2), 1)
			t := hwy.Load(highBuf[y*lanes:])
			hwy.Store(hwy.Add(t, update), highBuf[y*lanes:])
		}

		// Tail with boundary clamping
		for y := safeEnd; y < dn; y++ {
			var n1Idx, n2Idx int
			if phase == 0 {
				n1Idx = y
				n2Idx = y + 1
			} else {
				n1Idx = y - 1
				n2Idx = y
			}
			if n1Idx < 0 {
				n1Idx = 0
			}
			if n1Idx >= sn {
				n1Idx = sn - 1
			}
			if n2Idx >= sn {
				n2Idx = sn - 1
			}
			n1 := hwy.Load(lowBuf[n1Idx*lanes:])
			n2 := hwy.Load(lowBuf[n2Idx*lanes:])
			update := hwy.ShiftRight(hwy.Add(n1, n2), 1)
			t := hwy.Load(highBuf[y*lanes:])
			hwy.Store(hwy.Add(t, update), highBuf[y*lanes:])
		}
	}

	// 4. Interleave rows back into colBuf.
	// Output row order: low[0], high[0], low[1], high[1], ...
	if phase == 0 {
		minN := min(sn, dn)
		for y := 0; y < minN; y++ {
			copy(colBuf[2*y*lanes:2*y*lanes+lanes], lowBuf[y*lanes:y*lanes+lanes])
			copy(colBuf[(2*y+1)*lanes:(2*y+1)*lanes+lanes], highBuf[y*lanes:y*lanes+lanes])
		}
		// Remaining low (sn > dn)
		for y := dn; y < sn; y++ {
			copy(colBuf[2*y*lanes:2*y*lanes+lanes], lowBuf[y*lanes:y*lanes+lanes])
		}
	} else {
		minN := min(sn, dn)
		for y := 0; y < minN; y++ {
			copy(colBuf[2*y*lanes:2*y*lanes+lanes], highBuf[y*lanes:y*lanes+lanes])
			copy(colBuf[(2*y+1)*lanes:(2*y+1)*lanes+lanes], lowBuf[y*lanes:y*lanes+lanes])
		}
		for y := dn; y < sn; y++ {
			copy(colBuf[(2*y+1)*lanes:(2*y+1)*lanes+lanes], lowBuf[y*lanes:y*lanes+lanes])
		}
		for y := sn; y < dn; y++ {
			copy(colBuf[2*y*lanes:2*y*lanes+lanes], highBuf[y*lanes:y*lanes+lanes])
		}
	}
	_ = lanes
}

// BaseDeinterleave extracts low and high-pass coefficients from src.
// phase=0: low[i]=src[2i], high[i]=src[2i+1]
// phase=1: high[i]=src[2i], low[i]=src[2i+1]
func BaseDeinterleave[T hwy.Lanes](src []T, low []T, sn int, high []T, dn int, phase int) {
	if phase == 0 {
		// Even-first: even indices to low, odd indices to high
		for i := range sn {
			low[i] = src[2*i]
		}
		for i := range dn {
			high[i] = src[2*i+1]
		}
	} else {
		// Odd-first: even indices to high, odd indices to low
		for i := range dn {
			high[i] = src[2*i]
		}
		for i := range sn {
			low[i] = src[2*i+1]
		}
	}
}
