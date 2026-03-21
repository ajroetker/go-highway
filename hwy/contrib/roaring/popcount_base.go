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

// Package roaring provides SIMD-accelerated bitmap operations compatible with
// the RoaringBitmap/roaring Go library. It accelerates the hot-path operations
// on bitmap containers: popcount, fused bitwise+popcount, and bulk bitwise ops.
//
// These functions operate on []uint64 slices and are direct drop-in replacements
// for the scalar implementations in the roaring library.
package roaring

//go:generate go run ../../../cmd/hwygen -input popcount_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch roaring

import (
	"math/bits"

	"github.com/ajroetker/go-highway/hwy"
)

// BasePopcntSlice returns the total number of set bits across all elements.
// This is equivalent to sum(popcount(s[i])) for all i.
func BasePopcntSlice(s []uint64) uint64 {
	if len(s) == 0 {
		return 0
	}

	var result uint64
	lanes := hwy.Zero[uint64]().NumLanes()
	acc := hwy.Zero[uint64]()

	stride := lanes * 4
	var i int
	for i = 0; i+stride <= len(s); i += stride {
		v0, v1, v2, v3 := hwy.Load4(s[i:])
		acc = hwy.Add(acc, hwy.PopCount(v0))
		acc = hwy.Add(acc, hwy.PopCount(v1))
		acc = hwy.Add(acc, hwy.PopCount(v2))
		acc = hwy.Add(acc, hwy.PopCount(v3))
	}

	for ; i+lanes <= len(s); i += lanes {
		v := hwy.Load(s[i:])
		acc = hwy.Add(acc, hwy.PopCount(v))
	}

	result = uint64(hwy.ReduceSum(acc))

	for ; i < len(s); i++ {
		result += uint64(bits.OnesCount64(s[i]))
	}
	return result
}

// BasePopcntAndSlice returns the total popcount of the bitwise AND of two slices.
// This is equivalent to sum(popcount(s[i] & m[i])) for all i.
func BasePopcntAndSlice(s, m []uint64) uint64 {
	n := min(len(s), len(m))
	if n == 0 {
		return 0
	}

	var result uint64
	lanes := hwy.Zero[uint64]().NumLanes()
	acc := hwy.Zero[uint64]()

	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		s0, s1, s2, s3 := hwy.Load4(s[i:])
		m0, m1, m2, m3 := hwy.Load4(m[i:])
		acc = hwy.Add(acc, hwy.PopCount(hwy.And(s0, m0)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.And(s1, m1)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.And(s2, m2)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.And(s3, m3)))
	}

	for ; i+lanes <= n; i += lanes {
		vs := hwy.Load(s[i:])
		vm := hwy.Load(m[i:])
		acc = hwy.Add(acc, hwy.PopCount(hwy.And(vs, vm)))
	}

	result = uint64(hwy.ReduceSum(acc))

	for ; i < n; i++ {
		result += uint64(bits.OnesCount64(s[i] & m[i]))
	}
	return result
}

// BasePopcntOrSlice returns the total popcount of the bitwise OR of two slices.
// This is equivalent to sum(popcount(s[i] | m[i])) for all i.
func BasePopcntOrSlice(s, m []uint64) uint64 {
	n := min(len(s), len(m))
	if n == 0 {
		return 0
	}

	var result uint64
	lanes := hwy.Zero[uint64]().NumLanes()
	acc := hwy.Zero[uint64]()

	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		s0, s1, s2, s3 := hwy.Load4(s[i:])
		m0, m1, m2, m3 := hwy.Load4(m[i:])
		acc = hwy.Add(acc, hwy.PopCount(hwy.Or(s0, m0)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.Or(s1, m1)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.Or(s2, m2)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.Or(s3, m3)))
	}

	for ; i+lanes <= n; i += lanes {
		vs := hwy.Load(s[i:])
		vm := hwy.Load(m[i:])
		acc = hwy.Add(acc, hwy.PopCount(hwy.Or(vs, vm)))
	}

	result = uint64(hwy.ReduceSum(acc))

	for ; i < n; i++ {
		result += uint64(bits.OnesCount64(s[i] | m[i]))
	}
	return result
}

// BasePopcntXorSlice returns the total popcount of the bitwise XOR of two slices.
// This is equivalent to sum(popcount(s[i] ^ m[i])) for all i.
func BasePopcntXorSlice(s, m []uint64) uint64 {
	n := min(len(s), len(m))
	if n == 0 {
		return 0
	}

	var result uint64
	lanes := hwy.Zero[uint64]().NumLanes()
	acc := hwy.Zero[uint64]()

	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		s0, s1, s2, s3 := hwy.Load4(s[i:])
		m0, m1, m2, m3 := hwy.Load4(m[i:])
		acc = hwy.Add(acc, hwy.PopCount(hwy.Xor(s0, m0)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.Xor(s1, m1)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.Xor(s2, m2)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.Xor(s3, m3)))
	}

	for ; i+lanes <= n; i += lanes {
		vs := hwy.Load(s[i:])
		vm := hwy.Load(m[i:])
		acc = hwy.Add(acc, hwy.PopCount(hwy.Xor(vs, vm)))
	}

	result = uint64(hwy.ReduceSum(acc))

	for ; i < n; i++ {
		result += uint64(bits.OnesCount64(s[i] ^ m[i]))
	}
	return result
}

// BasePopcntAndNotSlice returns the total popcount of the bitwise ANDNOT of two slices.
// This is equivalent to sum(popcount(s[i] &^ m[i])) for all i.
// Note: Go's &^ operator means "s AND (NOT m)".
func BasePopcntAndNotSlice(s, m []uint64) uint64 {
	n := min(len(s), len(m))
	if n == 0 {
		return 0
	}

	var result uint64
	lanes := hwy.Zero[uint64]().NumLanes()
	acc := hwy.Zero[uint64]()

	// hwy.AndNot(a, b) = ~a & b, so to get s &^ m = s & ~m we use AndNot(m, s).
	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		s0, s1, s2, s3 := hwy.Load4(s[i:])
		m0, m1, m2, m3 := hwy.Load4(m[i:])
		acc = hwy.Add(acc, hwy.PopCount(hwy.AndNot(m0, s0)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.AndNot(m1, s1)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.AndNot(m2, s2)))
		acc = hwy.Add(acc, hwy.PopCount(hwy.AndNot(m3, s3)))
	}

	for ; i+lanes <= n; i += lanes {
		vs := hwy.Load(s[i:])
		vm := hwy.Load(m[i:])
		acc = hwy.Add(acc, hwy.PopCount(hwy.AndNot(vm, vs)))
	}

	result = uint64(hwy.ReduceSum(acc))

	for ; i < n; i++ {
		result += uint64(bits.OnesCount64(s[i] &^ m[i]))
	}
	return result
}
