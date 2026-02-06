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

// Synthesize53 applies the inverse 5/3 wavelet transform (synthesis) in-place.
// The input data is in [low | high] format, and output is interleaved samples.
// The phase parameter controls boundary handling (0 or 1).
func Synthesize53[T hwy.SignedInts](data []T, phase int) {
	n := len(data)
	if n <= 1 {
		if n == 1 && phase == 1 {
			data[0] /= 2
		}
		return
	}

	// Calculate subband sizes
	// phase=0: low has ceil(n/2), high has floor(n/2)
	// phase=1: high has ceil(n/2), low has floor(n/2)
	var sn, dn int // sn = number of low-pass, dn = number of high-pass
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

	// Working buffers for low and high subbands
	low := make([]T, sn)
	high := make([]T, dn)
	copy(low, data[:sn])
	copy(high, data[sn:sn+dn])

	// Helper functions for boundary-safe access
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
		// Phase 0: low at even indices, high at odd indices
		// Inverse step 1: low[i] -= (high[i-1] + high[i] + 2) >> 2
		for i := 0; i < sn; i++ {
			h1 := getHigh(i - 1)
			h2 := getHigh(i)
			low[i] -= (h1 + h2 + 2) >> 2
		}
		// Inverse step 2: high[i] += (low[i] + low[i+1]) >> 1
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i + 1)
			high[i] += (l1 + l2) >> 1
		}
		// Interleave: even=low, odd=high
		for i := 0; i < dn; i++ {
			data[2*i] = low[i]
			data[2*i+1] = high[i]
		}
		if sn > dn {
			data[n-1] = low[sn-1]
		}
	} else {
		// Phase 1: high at even indices, low at odd indices
		// Inverse step 1: low[i] -= (high[i] + high[i+1] + 2) >> 2
		for i := 0; i < sn; i++ {
			h1 := getHigh(i)
			h2 := getHigh(i + 1)
			low[i] -= (h1 + h2 + 2) >> 2
		}
		// Inverse step 2: high[i] += (low[i] + low[i-1]) >> 1
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i - 1)
			high[i] += (l1 + l2) >> 1
		}
		// Interleave: even=high, odd=low
		for i := 0; i < dn; i++ {
			data[2*i] = high[i]
		}
		for i := 0; i < sn; i++ {
			data[2*i+1] = low[i]
		}
	}
}

// Analyze53 applies the forward 5/3 wavelet transform (analysis) in-place.
// The input data is interleaved samples, and output is in [low | high] format.
// The phase parameter controls boundary handling (0 or 1).
func Analyze53[T hwy.SignedInts](data []T, phase int) {
	n := len(data)
	if n <= 1 {
		if n == 1 && phase == 1 {
			data[0] *= 2
		}
		return
	}

	// Calculate subband sizes
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

	// Working buffers
	low := make([]T, sn)
	high := make([]T, dn)

	// Deinterleave
	if phase == 0 {
		for i := 0; i < sn; i++ {
			low[i] = data[2*i]
		}
		for i := 0; i < dn; i++ {
			high[i] = data[2*i+1]
		}
	} else {
		for i := 0; i < dn; i++ {
			high[i] = data[2*i]
		}
		for i := 0; i < sn; i++ {
			low[i] = data[2*i+1]
		}
	}

	// Helper functions for boundary-safe access
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
		// Analysis steps (reverse of synthesis):
		// Step 1: high[i] -= (low[i] + low[i+1]) >> 1
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i + 1)
			high[i] -= (l1 + l2) >> 1
		}
		// Step 2: low[i] += (high[i-1] + high[i] + 2) >> 2
		for i := 0; i < sn; i++ {
			h1 := getHigh(i - 1)
			h2 := getHigh(i)
			low[i] += (h1 + h2 + 2) >> 2
		}
	} else {
		// Analysis steps for phase=1:
		// Step 1: high[i] -= (low[i] + low[i-1]) >> 1
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i - 1)
			high[i] -= (l1 + l2) >> 1
		}
		// Step 2: low[i] += (high[i] + high[i+1] + 2) >> 2
		for i := 0; i < sn; i++ {
			h1 := getHigh(i)
			h2 := getHigh(i + 1)
			low[i] += (h1 + h2 + 2) >> 2
		}
	}

	// Copy back in [low | high] format
	copy(data[:sn], low)
	copy(data[sn:], high)
}

// Synthesize97 applies the inverse 9/7 wavelet transform (synthesis) in-place.
// The input data is in [low | high] format, and output is interleaved samples.
// Uses standard K normalization (not JPEG 2000's 2/K convention).
func Synthesize97[T hwy.Floats](data []T, phase int) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Calculate subband sizes
	var sn, dn int
	if phase == 0 {
		sn = (n + 1) / 2
		dn = n / 2
	} else {
		dn = (n + 1) / 2
		sn = n / 2
	}

	if sn == 0 || dn == 0 {
		return
	}

	// Get coefficients for this type
	alpha, beta, gamma, delta, k, invK := lift97Coeffs[T]()

	// Working buffers
	low := make([]T, sn)
	high := make([]T, dn)
	copy(low, data[:sn])
	copy(high, data[sn:sn+dn])

	// Inverse scaling
	ScaleSlice(low, sn, invK)
	ScaleSlice(high, dn, k)

	// Helper functions for boundary-safe access
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
		// Phase 0: low at even indices, high at odd indices
		// Inverse step 4: low[i] -= delta * (high[i-1] + high[i])
		for i := 0; i < sn; i++ {
			h1 := getHigh(i - 1)
			h2 := getHigh(i)
			low[i] -= delta * (h1 + h2)
		}
		// Inverse step 3: high[i] -= gamma * (low[i] + low[i+1])
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i + 1)
			high[i] -= gamma * (l1 + l2)
		}
		// Inverse step 2: low[i] -= beta * (high[i-1] + high[i])
		for i := 0; i < sn; i++ {
			h1 := getHigh(i - 1)
			h2 := getHigh(i)
			low[i] -= beta * (h1 + h2)
		}
		// Inverse step 1: high[i] -= alpha * (low[i] + low[i+1])
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i + 1)
			high[i] -= alpha * (l1 + l2)
		}
		// Interleave: even=low, odd=high
		for i := 0; i < dn; i++ {
			data[2*i] = low[i]
			data[2*i+1] = high[i]
		}
		if sn > dn {
			data[n-1] = low[sn-1]
		}
	} else {
		// Phase 1: high at even indices, low at odd indices
		// Inverse step 4: low[i] -= delta * (high[i] + high[i+1])
		for i := 0; i < sn; i++ {
			h1 := getHigh(i)
			h2 := getHigh(i + 1)
			low[i] -= delta * (h1 + h2)
		}
		// Inverse step 3: high[i] -= gamma * (low[i] + low[i-1])
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i - 1)
			high[i] -= gamma * (l1 + l2)
		}
		// Inverse step 2: low[i] -= beta * (high[i] + high[i+1])
		for i := 0; i < sn; i++ {
			h1 := getHigh(i)
			h2 := getHigh(i + 1)
			low[i] -= beta * (h1 + h2)
		}
		// Inverse step 1: high[i] -= alpha * (low[i] + low[i-1])
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i - 1)
			high[i] -= alpha * (l1 + l2)
		}
		// Interleave: even=high, odd=low
		for i := 0; i < dn; i++ {
			data[2*i] = high[i]
		}
		for i := 0; i < sn; i++ {
			data[2*i+1] = low[i]
		}
	}
}

// Analyze97 applies the forward 9/7 wavelet transform (analysis) in-place.
// The input data is interleaved samples, and output is in [low | high] format.
// Uses standard K normalization (not JPEG 2000's 2/K convention).
func Analyze97[T hwy.Floats](data []T, phase int) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Calculate subband sizes
	var sn, dn int
	if phase == 0 {
		sn = (n + 1) / 2
		dn = n / 2
	} else {
		dn = (n + 1) / 2
		sn = n / 2
	}

	if sn == 0 || dn == 0 {
		return
	}

	// Get coefficients for this type
	alpha, beta, gamma, delta, k, invK := lift97Coeffs[T]()

	// Working buffers
	low := make([]T, sn)
	high := make([]T, dn)

	// Deinterleave
	if phase == 0 {
		for i := 0; i < sn; i++ {
			low[i] = data[2*i]
		}
		for i := 0; i < dn; i++ {
			high[i] = data[2*i+1]
		}
	} else {
		for i := 0; i < dn; i++ {
			high[i] = data[2*i]
		}
		for i := 0; i < sn; i++ {
			low[i] = data[2*i+1]
		}
	}

	// Helper functions for boundary-safe access
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
		// Analysis steps (reverse of synthesis):
		// Step 1: high[i] += alpha * (low[i] + low[i+1])
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i + 1)
			high[i] += alpha * (l1 + l2)
		}
		// Step 2: low[i] += beta * (high[i-1] + high[i])
		for i := 0; i < sn; i++ {
			h1 := getHigh(i - 1)
			h2 := getHigh(i)
			low[i] += beta * (h1 + h2)
		}
		// Step 3: high[i] += gamma * (low[i] + low[i+1])
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i + 1)
			high[i] += gamma * (l1 + l2)
		}
		// Step 4: low[i] += delta * (high[i-1] + high[i])
		for i := 0; i < sn; i++ {
			h1 := getHigh(i - 1)
			h2 := getHigh(i)
			low[i] += delta * (h1 + h2)
		}
	} else {
		// Analysis steps for phase=1:
		// Step 1: high[i] += alpha * (low[i] + low[i-1])
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i - 1)
			high[i] += alpha * (l1 + l2)
		}
		// Step 2: low[i] += beta * (high[i] + high[i+1])
		for i := 0; i < sn; i++ {
			h1 := getHigh(i)
			h2 := getHigh(i + 1)
			low[i] += beta * (h1 + h2)
		}
		// Step 3: high[i] += gamma * (low[i] + low[i-1])
		for i := 0; i < dn; i++ {
			l1 := getLow(i)
			l2 := getLow(i - 1)
			high[i] += gamma * (l1 + l2)
		}
		// Step 4: low[i] += delta * (high[i] + high[i+1])
		for i := 0; i < sn; i++ {
			h1 := getHigh(i)
			h2 := getHigh(i + 1)
			low[i] += delta * (h1 + h2)
		}
	}

	// Apply scaling
	ScaleSlice(low, sn, k)
	ScaleSlice(high, dn, invK)

	// Copy back in [low | high] format
	copy(data[:sn], low)
	copy(data[sn:], high)
}
