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

// Synthesize53 applies the inverse 5/3 wavelet transform using pre-allocated buffers.
// low and high must each have capacity >= ceil(n/2). This avoids per-call allocations
// and uses SIMD-dispatched lifting and interleaving.
func Synthesize53[T hwy.SignedInts](data []T, phase int, low, high []T) {
	n := len(data)
	if n <= 1 {
		if n == 1 && phase == 1 {
			data[0] /= 2
		}
		return
	}

	var sn, dn int
	if phase == 0 {
		sn = (n + 1) / 2
		dn = n / 2
	} else {
		dn = (n + 1) / 2
		sn = n / 2
	}

	if sn == 0 || dn == 0 {
		if phase == 1 && sn == 0 && dn == 1 {
			data[0] /= 2
		}
		return
	}

	low = low[:sn]
	high = high[:dn]

	// Fused kernel: copy + update + predict + interleave in a single dispatch
	Synthesize53Core(data, n, low, sn, high, dn, phase)
}

// Synthesize53Cols applies the inverse 5/3 wavelet transform to multiple columns
// simultaneously using column-interleaved SIMD. colBuf has height*lanes elements laid
// out as colBuf[y*lanes + c] for row y, column c. lowBuf and highBuf are scratch
// buffers each with capacity >= ceil(height/2)*lanes.
//
// This is designed for the vertical pass of a 2D DWT: the caller gathers `lanes`
// adjacent columns from each row into colBuf, calls this function, then scatters
// the results back. Each row contributes one contiguous SIMD load/store.
func Synthesize53Cols[T hwy.SignedInts](colBuf []T, height int, phase int, lowBuf, highBuf []T) {
	if height <= 1 {
		if height == 1 && phase == 1 {
			lanes := hwy.MaxLanes[T]()
			for c := range lanes {
				colBuf[c] /= 2
			}
		}
		return
	}

	var sn, dn int
	if phase == 0 {
		sn = (height + 1) / 2
		dn = height / 2
	} else {
		dn = (height + 1) / 2
		sn = height / 2
	}

	if sn == 0 || dn == 0 {
		if phase == 1 && sn == 0 && dn == 1 {
			lanes := hwy.MaxLanes[T]()
			for c := range lanes {
				colBuf[c] /= 2
			}
		}
		return
	}

	Synthesize53CoreCols(colBuf, height, lowBuf, sn, highBuf, dn, phase)
}

// Analyze53 applies the forward 5/3 wavelet transform using pre-allocated buffers.
// low and high must each have capacity >= ceil(n/2). This avoids per-call allocations.
func Analyze53[T hwy.SignedInts](data []T, phase int, low, high []T) {
	n := len(data)
	if n <= 1 {
		if n == 1 && phase == 1 {
			data[0] *= 2
		}
		return
	}

	var sn, dn int
	if phase == 0 {
		sn = (n + 1) / 2
		dn = n / 2
	} else {
		dn = (n + 1) / 2
		sn = n / 2
	}

	if sn == 0 || dn == 0 {
		if phase == 1 && sn == 0 && dn == 1 {
			data[0] *= 2
		}
		return
	}

	low = low[:sn]
	high = high[:dn]

	// Deinterleave
	Deinterleave(data, low, sn, high, dn, phase)

	// Forward lifting uses opposite signs from the SIMD primitives.
	// LiftPredict53 does +=, but analysis step 1 needs -=.
	// LiftUpdate53 does -=, but analysis step 2 needs +=.
	// Use scalar loops with boundary-safe access.

	getHigh := func(i int) T {
		if i < 0 {
			return high[0]
		}
		if i >= dn {
			return high[dn-1]
		}
		return high[i]
	}

	getLow := func(i int) T {
		if i < 0 {
			return low[0]
		}
		if i >= sn {
			return low[sn-1]
		}
		return low[i]
	}

	if phase == 0 {
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i + 1)
			high[i] -= (l1 + l2) >> 1
		}
		for i := 0; i < sn; i++ {
			h1 := getHigh(i - 1)
			h2 := getHigh(i)
			low[i] += (h1 + h2 + 2) >> 2
		}
	} else {
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i - 1)
			high[i] -= (l1 + l2) >> 1
		}
		for i := 0; i < sn; i++ {
			h1 := getHigh(i)
			h2 := getHigh(i + 1)
			low[i] += (h1 + h2 + 2) >> 2
		}
	}

	copy(data[:sn], low)
	copy(data[sn:], high)
}
