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

package sort

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input radix_base.go -output . -targets avx2,avx512,neon,fallback -dispatch radix

// BaseRadixPass performs one pass of LSD radix sort.
// shift specifies which byte to use for bucketing (0, 8, 16, 24 for int32).
// This is the core SIMD-accelerated function for histogram counting.
func BaseRadixPass[T hwy.SignedInts](src, dst []T, shift int) {
	n := len(src)
	if n == 0 {
		return
	}

	lanes := hwy.MaxLanes[T]()
	mask := T(radixMask8_i32)
	maskVec := hwy.Set(mask)

	// Count histogram for each bucket (256 buckets for 8 bits)
	var count [256]int

	// SIMD accelerated histogram counting
	i := 0
	for i+lanes <= n {
		v := hwy.LoadFull(src[i:])
		shifted := hwy.ShiftRight(v, shift)
		digits := hwy.And(shifted, maskVec)

		// Extract digits and count
		var buf [16]T // Max AVX-512 lanes
		hwy.Store(digits, buf[:])
		for j := range lanes {
			digit := int(buf[j]) & 0xFF
			count[digit]++
		}
		i += lanes
	}

	// Handle tail
	for ; i < n; i++ {
		digit := int(src[i]>>shift) & 0xFF
		count[digit]++
	}

	// Compute prefix sum to get bucket offsets
	offset := 0
	for b := range 256 {
		c := count[b]
		count[b] = offset
		offset += c
	}

	// Scatter elements to destination
	for i := range n {
		digit := int(src[i]>>shift) & 0xFF
		dst[count[digit]] = src[i]
		count[digit]++
	}
}

// BaseRadixPass16 performs one pass of 16-bit radix sort.
// shift specifies which 16-bit chunk to use (0, 16, 32, 48 for int64).
// Uses 65536 buckets for faster sorting of int64 (4 passes vs 8 with 8-bit radix).
func BaseRadixPass16[T hwy.SignedInts](src, dst []T, shift int) {
	n := len(src)
	if n == 0 {
		return
	}

	mask := T(radixMask16_i64)

	// Count histogram for each bucket (65536 buckets for 16 bits)
	var count [65536]int

	// Histogram counting
	for i := range n {
		digit := int((src[i] >> shift) & mask)
		count[digit]++
	}

	// Compute prefix sum to get bucket offsets
	offset := 0
	for b := range 65536 {
		c := count[b]
		count[b] = offset
		offset += c
	}

	// Scatter elements to destination
	for i := range n {
		digit := int((src[i] >> shift) & mask)
		dst[count[digit]] = src[i]
		count[digit]++
	}
}

// BaseRadixPass16Signed performs the final 16-bit pass for signed int64.
// The MSB chunk contains the sign bit, so negative numbers (32768-65535) come before positive (0-32767).
func BaseRadixPass16Signed[T hwy.SignedInts](src, dst []T, shift int) {
	n := len(src)
	if n == 0 {
		return
	}

	mask := T(radixMask16_i64)

	var count [65536]int

	// Histogram counting
	for i := range n {
		digit := int((src[i] >> shift) & mask)
		count[digit]++
	}

	// For signed MSB: buckets 32768-65535 (negative) come before 0-32767 (positive)
	offset := 0
	for b := 32768; b < 65536; b++ {
		c := count[b]
		count[b] = offset
		offset += c
	}
	for b := range 32768 {
		c := count[b]
		count[b] = offset
		offset += c
	}

	// Scatter
	for i := range n {
		digit := int((src[i] >> shift) & mask)
		dst[count[digit]] = src[i]
		count[digit]++
	}
}

// BaseRadixPassSigned performs the final pass for signed integers.
// The MSB byte contains the sign bit, so negative numbers (128-255) come before positive (0-127).
func BaseRadixPassSigned[T hwy.SignedInts](src, dst []T, shift int) {
	n := len(src)
	if n == 0 {
		return
	}

	lanes := hwy.MaxLanes[T]()
	mask := T(radixMask8_i32)
	maskVec := hwy.Set(mask)

	var count [256]int

	// SIMD histogram counting
	i := 0
	for i+lanes <= n {
		v := hwy.LoadFull(src[i:])
		shifted := hwy.ShiftRight(v, shift)
		digits := hwy.And(shifted, maskVec)

		var buf [16]T
		hwy.Store(digits, buf[:])
		for j := range lanes {
			digit := int(buf[j]) & 0xFF
			count[digit]++
		}
		i += lanes
	}

	for ; i < n; i++ {
		digit := int(src[i]>>shift) & 0xFF
		count[digit]++
	}

	// For signed MSB: buckets 128-255 (negative) come before 0-127 (positive)
	offset := 0
	for b := 128; b < 256; b++ {
		c := count[b]
		count[b] = offset
		offset += c
	}
	for b := range 128 {
		c := count[b]
		count[b] = offset
		offset += c
	}

	// Scatter
	for i := range n {
		digit := int(src[i]>>shift) & 0xFF
		dst[count[digit]] = src[i]
		count[digit]++
	}
}
