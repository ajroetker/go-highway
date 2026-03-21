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

package roaring

//go:generate go run ../../../cmd/hwygen -input bitwise_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch bitwise

import "github.com/ajroetker/go-highway/hwy"

// BaseAndSlice computes dst[i] = a[i] & b[i] for all i up to min(len(dst), len(a), len(b)).
// This is the core operation for bitmap container AND in roaring bitmaps.
func BaseAndSlice(dst, a, b []uint64) {
	n := min(len(dst), min(len(a), len(b)))
	if n == 0 {
		return
	}

	lanes := hwy.Zero[uint64]().NumLanes()

	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		a0, a1, a2, a3 := hwy.Load4(a[i:])
		b0, b1, b2, b3 := hwy.Load4(b[i:])
		hwy.Store(hwy.And(a0, b0), dst[i:])
		hwy.Store(hwy.And(a1, b1), dst[i+lanes:])
		hwy.Store(hwy.And(a2, b2), dst[i+lanes*2:])
		hwy.Store(hwy.And(a3, b3), dst[i+lanes*3:])
	}

	for ; i+lanes <= n; i += lanes {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		hwy.Store(hwy.And(va, vb), dst[i:])
	}

	for ; i < n; i++ {
		dst[i] = a[i] & b[i]
	}
}

// BaseOrSlice computes dst[i] = a[i] | b[i] for all i up to min(len(dst), len(a), len(b)).
// This is the core operation for bitmap container OR in roaring bitmaps.
func BaseOrSlice(dst, a, b []uint64) {
	n := min(len(dst), min(len(a), len(b)))
	if n == 0 {
		return
	}

	lanes := hwy.Zero[uint64]().NumLanes()

	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		a0, a1, a2, a3 := hwy.Load4(a[i:])
		b0, b1, b2, b3 := hwy.Load4(b[i:])
		hwy.Store(hwy.Or(a0, b0), dst[i:])
		hwy.Store(hwy.Or(a1, b1), dst[i+lanes:])
		hwy.Store(hwy.Or(a2, b2), dst[i+lanes*2:])
		hwy.Store(hwy.Or(a3, b3), dst[i+lanes*3:])
	}

	for ; i+lanes <= n; i += lanes {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		hwy.Store(hwy.Or(va, vb), dst[i:])
	}

	for ; i < n; i++ {
		dst[i] = a[i] | b[i]
	}
}

// BaseXorSlice computes dst[i] = a[i] ^ b[i] for all i up to min(len(dst), len(a), len(b)).
// This is the core operation for bitmap container XOR in roaring bitmaps.
func BaseXorSlice(dst, a, b []uint64) {
	n := min(len(dst), min(len(a), len(b)))
	if n == 0 {
		return
	}

	lanes := hwy.Zero[uint64]().NumLanes()

	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		a0, a1, a2, a3 := hwy.Load4(a[i:])
		b0, b1, b2, b3 := hwy.Load4(b[i:])
		hwy.Store(hwy.Xor(a0, b0), dst[i:])
		hwy.Store(hwy.Xor(a1, b1), dst[i+lanes:])
		hwy.Store(hwy.Xor(a2, b2), dst[i+lanes*2:])
		hwy.Store(hwy.Xor(a3, b3), dst[i+lanes*3:])
	}

	for ; i+lanes <= n; i += lanes {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		hwy.Store(hwy.Xor(va, vb), dst[i:])
	}

	for ; i < n; i++ {
		dst[i] = a[i] ^ b[i]
	}
}

// BaseAndNotSlice computes dst[i] = a[i] &^ b[i] for all i up to min(len(dst), len(a), len(b)).
// This is the core operation for bitmap container ANDNOT in roaring bitmaps.
// Note: Go's &^ means "a AND (NOT b)".
func BaseAndNotSlice(dst, a, b []uint64) {
	n := min(len(dst), min(len(a), len(b)))
	if n == 0 {
		return
	}

	lanes := hwy.Zero[uint64]().NumLanes()

	// hwy.AndNot(x, y) = ~x & y, so to get a &^ b = a & ~b we use AndNot(b, a).
	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		a0, a1, a2, a3 := hwy.Load4(a[i:])
		b0, b1, b2, b3 := hwy.Load4(b[i:])
		hwy.Store(hwy.AndNot(b0, a0), dst[i:])
		hwy.Store(hwy.AndNot(b1, a1), dst[i+lanes:])
		hwy.Store(hwy.AndNot(b2, a2), dst[i+lanes*2:])
		hwy.Store(hwy.AndNot(b3, a3), dst[i+lanes*3:])
	}

	for ; i+lanes <= n; i += lanes {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		hwy.Store(hwy.AndNot(vb, va), dst[i:])
	}

	for ; i < n; i++ {
		dst[i] = a[i] &^ b[i]
	}
}
