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

package asm

import (
	"math"
	"testing"
)

// matmulReferenceF32 is a simple reference implementation for testing
func matmulReferenceF32(a, b, c []float32, m, n, k int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// TestBlockedMatMulNEONF32_SmallM tests the blocked NEON F32 with small M values.
// This reproduces the segfault seen in BenchmarkSmallMParallel/BlockedMatMul on CI.
func TestBlockedMatMulNEONF32_SmallM(t *testing.T) {
	testCases := []struct {
		name    string
		m, n, k int
	}{
		{"11x1024x1024", 11, 1024, 1024}, // This is the failing case from CI
		{"1x64x64", 1, 64, 64},
		{"3x128x128", 3, 128, 128},
		{"7x256x256", 7, 256, 256},
		{"15x512x512", 15, 512, 512},
		{"31x64x64", 31, 64, 64},
		{"47x64x64", 47, 64, 64},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m, n, k := tc.m, tc.n, tc.k
			t.Logf("Testing BlockedMatMulNEONF32 with m=%d, n=%d, k=%d", m, n, k)

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			cRef := make([]float32, m*n)

			// Initialize with small values
			for i := range a {
				a[i] = float32(i%7 - 3)
			}
			for i := range b {
				b[i] = float32(i%5 - 2)
			}

			// Compute reference
			matmulReferenceF32(a, b, cRef, m, n, k)

			// Test blocked NEON
			BlockedMatMulNEONF32(a, b, c, m, n, k)

			// Verify
			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - cRef[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			const tolerance = 1e-4
			if maxErr > tolerance {
				t.Errorf("max error %v exceeds tolerance %v", maxErr, tolerance)
			}
		})
	}
}

// TestBlockedMatMulNEONF32_SquareSizes tests the blocked NEON F32 with square matrices.
func TestBlockedMatMulNEONF32_SquareSizes(t *testing.T) {
	sizes := []int{16, 32, 48, 64, 96, 128}

	for _, size := range sizes {
		t.Run("size_"+string(rune('0'+size/10))+string(rune('0'+size%10)), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			cRef := make([]float32, m*n)

			// Initialize
			for i := range a {
				a[i] = float32(i%7 - 3)
			}
			for i := range b {
				b[i] = float32(i%5 - 2)
			}

			// Compute reference
			matmulReferenceF32(a, b, cRef, m, n, k)

			// Test blocked NEON
			BlockedMatMulNEONF32(a, b, c, m, n, k)

			// Verify
			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - cRef[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			const tolerance = 1e-4
			if maxErr > tolerance {
				t.Errorf("max error %v exceeds tolerance %v", maxErr, tolerance)
			}
		})
	}
}
