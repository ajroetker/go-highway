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

// Package vec provides element-wise vector arithmetic operations using SIMD.
//
// This package contains optimized implementations of common vector operations
// such as addition, subtraction, multiplication, division, and scaling.
// Operations come in two variants:
//   - In-place: modify the destination slice directly (e.g., BaseAdd)
//   - To: write results to a separate destination slice (e.g., BaseAddTo)
//
// All functions use SIMD acceleration when available via the hwy package.
package vec

//go:generate go run ../../../cmd/hwygen -input arithmetic_base.go -output . -targets avx2,avx512,neon,fallback -dispatch arithmetic

import "github.com/ajroetker/go-highway/hwy"

// BaseAdd performs in-place element-wise addition: dst[i] += s[i].
//
// If the slices have different lengths, the operation uses the minimum length.
// Returns early if either slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	dst := []float32{1, 2, 3, 4}
//	s := []float32{5, 6, 7, 8}
//	BaseAdd(dst, s)  // dst is now {6, 8, 10, 12}
func BaseAdd[T hwy.Floats](dst, s []T) {
	if len(dst) == 0 || len(s) == 0 {
		return
	}

	n := min(len(dst), len(s))
	lanes := hwy.Zero[T]().NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		vd := hwy.LoadFull(dst[i:])
		vs := hwy.LoadFull(s[i:])
		result := hwy.Add(vd, vs)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] += s[i]
	}
}

// BaseAddTo performs element-wise addition: dst[i] = a[i] + b[i].
//
// If the slices have different lengths, the operation uses the minimum length.
// Returns early if any slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	a := []float32{1, 2, 3, 4}
//	b := []float32{5, 6, 7, 8}
//	dst := make([]float32, 4)
//	BaseAddTo(dst, a, b)  // dst is now {6, 8, 10, 12}
func BaseAddTo[T hwy.Floats](dst, a, b []T) {
	if len(dst) == 0 || len(a) == 0 || len(b) == 0 {
		return
	}

	n := min(len(dst), min(len(a), len(b)))
	lanes := hwy.Zero[T]().NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		va := hwy.LoadFull(a[i:])
		vb := hwy.LoadFull(b[i:])
		result := hwy.Add(va, vb)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

// BaseSub performs in-place element-wise subtraction: dst[i] -= s[i].
//
// If the slices have different lengths, the operation uses the minimum length.
// Returns early if either slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	dst := []float32{10, 20, 30, 40}
//	s := []float32{1, 2, 3, 4}
//	BaseSub(dst, s)  // dst is now {9, 18, 27, 36}
func BaseSub[T hwy.Floats](dst, s []T) {
	if len(dst) == 0 || len(s) == 0 {
		return
	}

	n := min(len(dst), len(s))
	lanes := hwy.Zero[T]().NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		vd := hwy.LoadFull(dst[i:])
		vs := hwy.LoadFull(s[i:])
		result := hwy.Sub(vd, vs)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] -= s[i]
	}
}

// BaseSubTo performs element-wise subtraction: dst[i] = a[i] - b[i].
//
// If the slices have different lengths, the operation uses the minimum length.
// Returns early if any slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	a := []float32{10, 20, 30, 40}
//	b := []float32{1, 2, 3, 4}
//	dst := make([]float32, 4)
//	BaseSubTo(dst, a, b)  // dst is now {9, 18, 27, 36}
func BaseSubTo[T hwy.Floats](dst, a, b []T) {
	if len(dst) == 0 || len(a) == 0 || len(b) == 0 {
		return
	}

	n := min(len(dst), min(len(a), len(b)))
	lanes := hwy.Zero[T]().NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		va := hwy.LoadFull(a[i:])
		vb := hwy.LoadFull(b[i:])
		result := hwy.Sub(va, vb)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}

// BaseMul performs in-place element-wise multiplication: dst[i] *= s[i].
//
// If the slices have different lengths, the operation uses the minimum length.
// Returns early if either slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	dst := []float32{1, 2, 3, 4}
//	s := []float32{2, 3, 4, 5}
//	BaseMul(dst, s)  // dst is now {2, 6, 12, 20}
func BaseMul[T hwy.Floats](dst, s []T) {
	if len(dst) == 0 || len(s) == 0 {
		return
	}

	n := min(len(dst), len(s))
	lanes := hwy.Zero[T]().NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		vd := hwy.LoadFull(dst[i:])
		vs := hwy.LoadFull(s[i:])
		result := hwy.Mul(vd, vs)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] *= s[i]
	}
}

// BaseMulTo performs element-wise multiplication: dst[i] = a[i] * b[i].
//
// If the slices have different lengths, the operation uses the minimum length.
// Returns early if any slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	a := []float32{1, 2, 3, 4}
//	b := []float32{2, 3, 4, 5}
//	dst := make([]float32, 4)
//	BaseMulTo(dst, a, b)  // dst is now {2, 6, 12, 20}
func BaseMulTo[T hwy.Floats](dst, a, b []T) {
	if len(dst) == 0 || len(a) == 0 || len(b) == 0 {
		return
	}

	n := min(len(dst), min(len(a), len(b)))
	lanes := hwy.Zero[T]().NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		va := hwy.LoadFull(a[i:])
		vb := hwy.LoadFull(b[i:])
		result := hwy.Mul(va, vb)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

// BaseDiv performs in-place element-wise division: dst[i] /= s[i].
//
// If the slices have different lengths, the operation uses the minimum length.
// Returns early if either slice is empty.
//
// Note: Division by zero will result in +Inf, -Inf, or NaN according to
// IEEE 754 floating-point semantics.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	dst := []float32{10, 20, 30, 40}
//	s := []float32{2, 4, 5, 8}
//	BaseDiv(dst, s)  // dst is now {5, 5, 6, 5}
func BaseDiv[T hwy.Floats](dst, s []T) {
	if len(dst) == 0 || len(s) == 0 {
		return
	}

	n := min(len(dst), len(s))
	lanes := hwy.Zero[T]().NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		vd := hwy.LoadFull(dst[i:])
		vs := hwy.LoadFull(s[i:])
		result := hwy.Div(vd, vs)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] /= s[i]
	}
}

// BaseDivTo performs element-wise division: dst[i] = a[i] / b[i].
//
// If the slices have different lengths, the operation uses the minimum length.
// Returns early if any slice is empty.
//
// Note: Division by zero will result in +Inf, -Inf, or NaN according to
// IEEE 754 floating-point semantics.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	a := []float32{10, 20, 30, 40}
//	b := []float32{2, 4, 5, 8}
//	dst := make([]float32, 4)
//	BaseDivTo(dst, a, b)  // dst is now {5, 5, 6, 5}
func BaseDivTo[T hwy.Floats](dst, a, b []T) {
	if len(dst) == 0 || len(a) == 0 || len(b) == 0 {
		return
	}

	n := min(len(dst), min(len(a), len(b)))
	lanes := hwy.Zero[T]().NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		va := hwy.LoadFull(a[i:])
		vb := hwy.LoadFull(b[i:])
		result := hwy.Div(va, vb)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] = a[i] / b[i]
	}
}

// BaseScale performs in-place scalar multiplication: dst[i] *= c.
//
// Returns early if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	dst := []float32{1, 2, 3, 4}
//	BaseScale(2.5, dst)  // dst is now {2.5, 5, 7.5, 10}
func BaseScale[T hwy.Floats](c T, dst []T) {
	if len(dst) == 0 {
		return
	}

	n := len(dst)
	vc := hwy.Set(c)
	lanes := vc.NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		vd := hwy.LoadFull(dst[i:])
		result := hwy.Mul(vd, vc)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] *= c
	}
}

// BaseScaleTo performs scalar multiplication: dst[i] = c * s[i].
//
// If the slices have different lengths, the operation uses the minimum length.
// Returns early if either slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	s := []float32{1, 2, 3, 4}
//	dst := make([]float32, 4)
//	BaseScaleTo(dst, 2.5, s)  // dst is now {2.5, 5, 7.5, 10}
func BaseScaleTo[T hwy.Floats](dst []T, c T, s []T) {
	if len(dst) == 0 || len(s) == 0 {
		return
	}

	n := min(len(dst), len(s))
	vc := hwy.Set(c)
	lanes := vc.NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		vs := hwy.LoadFull(s[i:])
		result := hwy.Mul(vc, vs)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] = c * s[i]
	}
}

// BaseAddConst performs in-place scalar addition: dst[i] += c.
//
// Returns early if the slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	dst := []float32{1, 2, 3, 4}
//	BaseAddConst(10, dst)  // dst is now {11, 12, 13, 14}
func BaseAddConst[T hwy.Floats](c T, dst []T) {
	if len(dst) == 0 {
		return
	}

	n := len(dst)
	vc := hwy.Set(c)
	lanes := vc.NumLanes()

	// Process full vectors
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		vd := hwy.LoadFull(dst[i:])
		result := hwy.Add(vd, vc)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] += c
	}
}

// BaseMulConstAddTo performs fused multiply-add: dst[i] += a * x[i].
//
// This operation is also known as AXPY (a*x plus y) in BLAS terminology.
// It uses fused multiply-add (FMA) instructions when available for better
// performance and precision.
//
// If the slices have different lengths, the operation uses the minimum length.
// Returns early if either slice is empty.
//
// Uses SIMD acceleration when available via the hwy package primitives.
//
// Example:
//
//	dst := []float32{1, 2, 3, 4}
//	x := []float32{1, 1, 1, 1}
//	BaseMulConstAddTo(dst, 10, x)  // dst is now {11, 12, 13, 14}
func BaseMulConstAddTo[T hwy.Floats](dst []T, a T, x []T) {
	if len(dst) == 0 || len(x) == 0 {
		return
	}

	n := min(len(dst), len(x))
	va := hwy.Set(a)
	lanes := va.NumLanes()

	// Process full vectors using FMA
	var i int
	for i = 0; i+lanes <= n; i += lanes {
		vd := hwy.LoadFull(dst[i:])
		vx := hwy.LoadFull(x[i:])
		// MulAdd computes a*b + c, so we compute va*vx + vd
		result := hwy.MulAdd(va, vx, vd)
		hwy.StoreFull(result, dst[i:])
	}

	// Handle tail elements with scalar code
	for ; i < n; i++ {
		dst[i] += a * x[i]
	}
}
