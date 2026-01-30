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

//go:build !noasm && arm64

// GELU NEON implementations for ARM64.
// Uses GOAT-transpiled NEON assembly for inline exp/erf computation.
package asm

import "unsafe"

// Generate NEON assembly from C source
//go:generate go tool goat ../c/gelu_neon_arm64.c -O3 --target arm64

// ============================================================================
// GELU Exact NEON - Float32
// ============================================================================

// GELUNeonF32 computes exact GELU using NEON with inline erf.
//
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
func GELUNeonF32(input, output []float32, size int) {
	if size <= 0 {
		return
	}
	sizeVal := int64(size)
	gelu_neon_f32(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&sizeVal),
	)
}

// ============================================================================
// GELU Exact NEON - Float64
// ============================================================================

// GELUNeonF64 computes exact GELU using NEON with inline erf (f64).
func GELUNeonF64(input, output []float64, size int) {
	if size <= 0 {
		return
	}
	sizeVal := int64(size)
	gelu_neon_f64(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&sizeVal),
	)
}

// ============================================================================
// GELU Approx NEON - Float32
// ============================================================================

// GELUApproxNeonF32 computes approximate GELU using NEON with inline sigmoid.
//
// GELU_approx(x) = x * sigmoid(1.702 * x)
func GELUApproxNeonF32(input, output []float32, size int) {
	if size <= 0 {
		return
	}
	sizeVal := int64(size)
	gelu_approx_neon_f32(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&sizeVal),
	)
}

// ============================================================================
// GELU Approx NEON - Float64
// ============================================================================

// GELUApproxNeonF64 computes approximate GELU using NEON with inline sigmoid (f64).
func GELUApproxNeonF64(input, output []float64, size int) {
	if size <= 0 {
		return
	}
	sizeVal := int64(size)
	gelu_approx_neon_f64(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&sizeVal),
	)
}

// ============================================================================
// SiLU NEON - Float32
// ============================================================================

// SiLUNeonF32 computes SiLU (Swish) using NEON with inline sigmoid.
//
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
func SiLUNeonF32(input, output []float32, size int) {
	if size <= 0 {
		return
	}
	sizeVal := int64(size)
	silu_neon_f32(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&sizeVal),
	)
}

// ============================================================================
// SiLU NEON - Float64
// ============================================================================

// SiLUNeonF64 computes SiLU (Swish) using NEON with inline sigmoid (f64).
func SiLUNeonF64(input, output []float64, size int) {
	if size <= 0 {
		return
	}
	sizeVal := int64(size)
	silu_neon_f64(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&sizeVal),
	)
}

// ============================================================================
// Tanh NEON - Float32
// ============================================================================

// TanhNeonF32 computes tanh using NEON with inline exp.
//
// tanh(x) = 2 * sigmoid(2x) - 1
func TanhNeonF32(input, output []float32, size int) {
	if size <= 0 {
		return
	}
	sizeVal := int64(size)
	tanh_neon_f32(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&sizeVal),
	)
}

// ============================================================================
// Tanh NEON - Float64
// ============================================================================

// TanhNeonF64 computes tanh using NEON with inline exp (f64).
func TanhNeonF64(input, output []float64, size int) {
	if size <= 0 {
		return
	}
	sizeVal := int64(size)
	tanh_neon_f64(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&sizeVal),
	)
}

// ============================================================================
// ELU NEON - Float32
// ============================================================================

// ELUNeonF32 computes ELU using NEON with inline exp.
//
// ELU(x) = x if x > 0, alpha*(exp(x)-1) if x <= 0
func ELUNeonF32(input, output []float32, size int, alpha float32) {
	if size <= 0 {
		return
	}
	sizeVal := int64(size)
	elu_neon_f32(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&sizeVal),
		unsafe.Pointer(&alpha),
	)
}

// ============================================================================
// ELU NEON - Float64
// ============================================================================

// ELUNeonF64 computes ELU using NEON with inline exp (f64).
func ELUNeonF64(input, output []float64, size int, alpha float64) {
	if size <= 0 {
		return
	}
	sizeVal := int64(size)
	elu_neon_f64(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&sizeVal),
		unsafe.Pointer(&alpha),
	)
}

// Assembly function declarations (generated by GoAT from gelu_neon_arm64.c)
