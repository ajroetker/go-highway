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

//go:generate go run ../../../cmd/hwygen -input lifting_base.go -output . -targets avx2,avx512,neon,fallback -dispatch lifting

// BaseLiftUpdate53 applies the 5/3 update step: target[i] -= (neighbor[i+off1] + neighbor[i+off2] + 2) >> 2
// This is used for the update step in 5/3 synthesis and the predict step in analysis.
// The phase parameter determines the offset pattern for neighbor access.
func BaseLiftUpdate53[T hwy.SignedInts](target []T, tLen int, neighbor []T, nLen int, phase int) {
	if tLen == 0 || nLen == 0 {
		return
	}

	twoVec := hwy.Set(T(2))
	lanes := hwy.MaxLanes[T]()

	// The offset depends on phase:
	// phase=0: target[i] -= (neighbor[i-1] + neighbor[i] + 2) >> 2
	// phase=1: target[i] -= (neighbor[i] + neighbor[i+1] + 2) >> 2
	off1, off2 := -1, 0
	if phase == 1 {
		off1, off2 = 0, 1
	}

	// Process elements
	for i := 0; i < tLen; i++ {
		// Get neighbor indices with boundary clamping
		n1Idx := i + off1
		n2Idx := i + off2
		if n1Idx < 0 {
			n1Idx = 0
		}
		if n1Idx >= nLen {
			n1Idx = nLen - 1
		}
		if n2Idx < 0 {
			n2Idx = 0
		}
		if n2Idx >= nLen {
			n2Idx = nLen - 1
		}

		// Scalar processing for boundaries, vector for bulk
		target[i] -= (neighbor[n1Idx] + neighbor[n2Idx] + 2) >> 2
	}

	// Note: For SIMD optimization in generated code, we would process the
	// bulk middle section with shifted loads. The boundary handling above
	// is for correctness in the scalar fallback.
	_ = twoVec // Used in SIMD version
	_ = lanes
}

// BaseLiftPredict53 applies the 5/3 predict step: target[i] += (neighbor[i+off1] + neighbor[i+off2]) >> 1
// This is used for the predict step in 5/3 synthesis and the update step in analysis.
func BaseLiftPredict53[T hwy.SignedInts](target []T, tLen int, neighbor []T, nLen int, phase int) {
	if tLen == 0 || nLen == 0 {
		return
	}

	lanes := hwy.MaxLanes[T]()

	// The offset depends on phase:
	// phase=0: target[i] += (neighbor[i] + neighbor[i+1]) >> 1
	// phase=1: target[i] += (neighbor[i-1] + neighbor[i]) >> 1
	off1, off2 := 0, 1
	if phase == 1 {
		off1, off2 = -1, 0
	}

	for i := 0; i < tLen; i++ {
		n1Idx := i + off1
		n2Idx := i + off2
		if n1Idx < 0 {
			n1Idx = 0
		}
		if n1Idx >= nLen {
			n1Idx = nLen - 1
		}
		if n2Idx < 0 {
			n2Idx = 0
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

	// Determine offsets based on phase
	// For predict steps: target odd, neighbor even
	// For update steps: target even, neighbor odd
	// phase controls which direction
	off1, off2 := 0, 1
	if phase == 1 {
		off1, off2 = -1, 0
	}

	for i := 0; i < tLen; i++ {
		n1Idx := i + off1
		n2Idx := i + off2
		if n1Idx < 0 {
			n1Idx = 0
		}
		if n1Idx >= nLen {
			n1Idx = nLen - 1
		}
		if n2Idx < 0 {
			n2Idx = 0
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

	// Handle tail elements
	if remaining := n - i; remaining > 0 {
		buf := make([]T, lanes)
		copy(buf, data[i:i+remaining])
		v := hwy.Load(buf)
		result := hwy.Mul(v, scaleVec)
		hwy.Store(result, buf)
		copy(data[i:i+remaining], buf[:remaining])
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

// BaseDeinterleave extracts low and high-pass coefficients from src.
// phase=0: low[i]=src[2i], high[i]=src[2i+1]
// phase=1: high[i]=src[2i], low[i]=src[2i+1]
func BaseDeinterleave[T hwy.Lanes](src []T, low []T, sn int, high []T, dn int, phase int) {
	if phase == 0 {
		// Even-first: even indices to low, odd indices to high
		for i := 0; i < sn; i++ {
			low[i] = src[2*i]
		}
		for i := 0; i < dn; i++ {
			high[i] = src[2*i+1]
		}
	} else {
		// Odd-first: even indices to high, odd indices to low
		for i := 0; i < dn; i++ {
			high[i] = src[2*i]
		}
		for i := 0; i < sn; i++ {
			low[i] = src[2*i+1]
		}
	}
}
