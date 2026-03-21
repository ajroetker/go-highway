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

//go:generate go run ../../../cmd/hwygen -input fused_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch fused

import (
	"math/bits"

	"github.com/ajroetker/go-highway/hwy"
)

// BaseAndPopcntSlice computes dst[i] = a[i] & b[i] and returns the total
// popcount of the result in a single pass. This eliminates the two-pass
// pattern used in roaring's andBitmap (popcntAndSlice then andBitmap loop).
func BaseAndPopcntSlice(dst, a, b []uint64) uint64 {
	n := min(len(dst), min(len(a), len(b)))
	if n == 0 {
		return 0
	}

	lanes := hwy.Zero[uint64]().NumLanes()
	acc := hwy.Zero[uint64]()

	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		a0, a1, a2, a3 := hwy.Load4(a[i:])
		b0, b1, b2, b3 := hwy.Load4(b[i:])
		r0 := hwy.And(a0, b0)
		r1 := hwy.And(a1, b1)
		r2 := hwy.And(a2, b2)
		r3 := hwy.And(a3, b3)
		hwy.Store(r0, dst[i:])
		hwy.Store(r1, dst[i+lanes:])
		hwy.Store(r2, dst[i+lanes*2:])
		hwy.Store(r3, dst[i+lanes*3:])
		acc = hwy.Add(acc, hwy.PopCount(r0))
		acc = hwy.Add(acc, hwy.PopCount(r1))
		acc = hwy.Add(acc, hwy.PopCount(r2))
		acc = hwy.Add(acc, hwy.PopCount(r3))
	}

	for ; i+lanes <= n; i += lanes {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		r := hwy.And(va, vb)
		hwy.Store(r, dst[i:])
		acc = hwy.Add(acc, hwy.PopCount(r))
	}

	result := uint64(hwy.ReduceSum(acc))

	for ; i < n; i++ {
		v := a[i] & b[i]
		dst[i] = v
		result += uint64(bits.OnesCount64(v))
	}
	return result
}

// BaseOrPopcntSlice computes dst[i] = a[i] | b[i] and returns the total
// popcount of the result in a single pass.
func BaseOrPopcntSlice(dst, a, b []uint64) uint64 {
	n := min(len(dst), min(len(a), len(b)))
	if n == 0 {
		return 0
	}

	lanes := hwy.Zero[uint64]().NumLanes()
	acc := hwy.Zero[uint64]()

	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		a0, a1, a2, a3 := hwy.Load4(a[i:])
		b0, b1, b2, b3 := hwy.Load4(b[i:])
		r0 := hwy.Or(a0, b0)
		r1 := hwy.Or(a1, b1)
		r2 := hwy.Or(a2, b2)
		r3 := hwy.Or(a3, b3)
		hwy.Store(r0, dst[i:])
		hwy.Store(r1, dst[i+lanes:])
		hwy.Store(r2, dst[i+lanes*2:])
		hwy.Store(r3, dst[i+lanes*3:])
		acc = hwy.Add(acc, hwy.PopCount(r0))
		acc = hwy.Add(acc, hwy.PopCount(r1))
		acc = hwy.Add(acc, hwy.PopCount(r2))
		acc = hwy.Add(acc, hwy.PopCount(r3))
	}

	for ; i+lanes <= n; i += lanes {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		r := hwy.Or(va, vb)
		hwy.Store(r, dst[i:])
		acc = hwy.Add(acc, hwy.PopCount(r))
	}

	result := uint64(hwy.ReduceSum(acc))

	for ; i < n; i++ {
		v := a[i] | b[i]
		dst[i] = v
		result += uint64(bits.OnesCount64(v))
	}
	return result
}

// BaseXorPopcntSlice computes dst[i] = a[i] ^ b[i] and returns the total
// popcount of the result in a single pass.
func BaseXorPopcntSlice(dst, a, b []uint64) uint64 {
	n := min(len(dst), min(len(a), len(b)))
	if n == 0 {
		return 0
	}

	lanes := hwy.Zero[uint64]().NumLanes()
	acc := hwy.Zero[uint64]()

	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		a0, a1, a2, a3 := hwy.Load4(a[i:])
		b0, b1, b2, b3 := hwy.Load4(b[i:])
		r0 := hwy.Xor(a0, b0)
		r1 := hwy.Xor(a1, b1)
		r2 := hwy.Xor(a2, b2)
		r3 := hwy.Xor(a3, b3)
		hwy.Store(r0, dst[i:])
		hwy.Store(r1, dst[i+lanes:])
		hwy.Store(r2, dst[i+lanes*2:])
		hwy.Store(r3, dst[i+lanes*3:])
		acc = hwy.Add(acc, hwy.PopCount(r0))
		acc = hwy.Add(acc, hwy.PopCount(r1))
		acc = hwy.Add(acc, hwy.PopCount(r2))
		acc = hwy.Add(acc, hwy.PopCount(r3))
	}

	for ; i+lanes <= n; i += lanes {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		r := hwy.Xor(va, vb)
		hwy.Store(r, dst[i:])
		acc = hwy.Add(acc, hwy.PopCount(r))
	}

	result := uint64(hwy.ReduceSum(acc))

	for ; i < n; i++ {
		v := a[i] ^ b[i]
		dst[i] = v
		result += uint64(bits.OnesCount64(v))
	}
	return result
}

// BaseAndNotPopcntSlice computes dst[i] = a[i] &^ b[i] and returns the total
// popcount of the result in a single pass.
func BaseAndNotPopcntSlice(dst, a, b []uint64) uint64 {
	n := min(len(dst), min(len(a), len(b)))
	if n == 0 {
		return 0
	}

	lanes := hwy.Zero[uint64]().NumLanes()
	acc := hwy.Zero[uint64]()

	// hwy.AndNot(x, y) = ~x & y, so to get a &^ b = a & ~b we use AndNot(b, a).
	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		a0, a1, a2, a3 := hwy.Load4(a[i:])
		b0, b1, b2, b3 := hwy.Load4(b[i:])
		r0 := hwy.AndNot(b0, a0)
		r1 := hwy.AndNot(b1, a1)
		r2 := hwy.AndNot(b2, a2)
		r3 := hwy.AndNot(b3, a3)
		hwy.Store(r0, dst[i:])
		hwy.Store(r1, dst[i+lanes:])
		hwy.Store(r2, dst[i+lanes*2:])
		hwy.Store(r3, dst[i+lanes*3:])
		acc = hwy.Add(acc, hwy.PopCount(r0))
		acc = hwy.Add(acc, hwy.PopCount(r1))
		acc = hwy.Add(acc, hwy.PopCount(r2))
		acc = hwy.Add(acc, hwy.PopCount(r3))
	}

	for ; i+lanes <= n; i += lanes {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		r := hwy.AndNot(vb, va)
		hwy.Store(r, dst[i:])
		acc = hwy.Add(acc, hwy.PopCount(r))
	}

	result := uint64(hwy.ReduceSum(acc))

	for ; i < n; i++ {
		v := a[i] &^ b[i]
		dst[i] = v
		result += uint64(bits.OnesCount64(v))
	}
	return result
}

