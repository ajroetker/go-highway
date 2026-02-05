//go:build arm64

package asm

import "testing"

// TestPackedMicroKernelNEONF32_BoundsCheck tests that the bounds check is correct.
// The bug: len(c) < mr*n is too strict - it should be len(c) < (mr-1)*n + nr
func TestPackedMicroKernelNEONF32_BoundsCheck(t *testing.T) {
	mr, nr := 4, 8
	kc := 16
	n := 16

	packedA := make([]float32, kc*mr)
	for i := range packedA {
		packedA[i] = float32(i + 1)
	}
	packedB := make([]float32, kc*nr)
	for i := range packedB {
		packedB[i] = float32(i + 1)
	}

	// Test 1: Full slice (should work)
	t.Run("FullSlice", func(t *testing.T) {
		c := make([]float32, mr*n) // 64 elements, exactly mr*n
		PackedMicroKernelNEONF32(packedA, packedB, c, kc, n, mr, nr)
		if c[0] == 0 {
			t.Errorf("Full slice: expected non-zero result, got zeros")
		}
		t.Logf("c[0:8] = %v", c[0:8])
	})

	// Test 2: Minimal required slice (should work but currently fails)
	// Actual writes: rows 0-3, columns 0-7
	// Row 0: c[0:8], Row 1: c[16:24], Row 2: c[32:40], Row 3: c[48:56]
	// So we need c[0:56] minimum, which is (mr-1)*n + nr = 3*16 + 8 = 56 elements
	t.Run("MinimalSlice", func(t *testing.T) {
		minRequired := (mr-1)*n + nr // 3*16 + 8 = 56
		c := make([]float32, minRequired)
		t.Logf("Minimal required: %d, mr*n would be: %d", minRequired, mr*n)

		PackedMicroKernelNEONF32(packedA, packedB, c, kc, n, mr, nr)
		if c[0] == 0 {
			t.Errorf("Minimal slice: expected non-zero result, got zeros (bounds check too strict?)")
		} else {
			t.Logf("c[0:8] = %v", c[0:8])
		}
	})

	// Test 3: Simulate the dispatch scenario - passing a sub-slice
	t.Run("SubSliceScenario", func(t *testing.T) {
		// Full C matrix
		fullC := make([]float32, 16*16) // 256 elements

		// Scenario: ir=12, jr=8 -> cOffset = 12*16+8 = 200
		// Passed slice: fullC[200:], length = 256-200 = 56
		ir, jr := 12, 8
		cOffset := ir*n + jr
		subSlice := fullC[cOffset:]

		t.Logf("cOffset=%d, len(subSlice)=%d, mr*n=%d", cOffset, len(subSlice), mr*n)

		PackedMicroKernelNEONF32(packedA, packedB, subSlice, kc, n, mr, nr)

		// Check if anything was written
		if subSlice[0] == 0 {
			t.Errorf("SubSlice scenario: got zeros - bounds check is too strict!")
			t.Logf("len(subSlice)=%d < mr*n=%d ? %v", len(subSlice), mr*n, len(subSlice) < mr*n)
		} else {
			t.Logf("subSlice[0:8] = %v", subSlice[0:8])
		}
	})
}

// TestPackedMicroKernelNEONF32_Correctness verifies the kernel produces correct results.
func TestPackedMicroKernelNEONF32_Correctness(t *testing.T) {
	mr, nr := 4, 8
	kc := 16
	n := 16

	packedA := make([]float32, kc*mr)
	for i := range packedA {
		packedA[i] = float32(i + 1)
	}
	packedB := make([]float32, kc*nr)
	for i := range packedB {
		packedB[i] = float32(i + 1)
	}

	// Use full-size C to avoid bounds check issue
	c := make([]float32, mr*n)
	PackedMicroKernelNEONF32(packedA, packedB, c, kc, n, mr, nr)

	// Compute expected result manually
	// PackedA layout: [kc][mr] - packedA[k*mr + row]
	// PackedB layout: [kc][nr] - packedB[k*nr + col]
	// C[row][col] += sum over k of packedA[k*mr + row] * packedB[k*nr + col]
	expected := make([]float32, mr*n)
	for row := range mr {
		for col := range nr {
			var sum float32
			for k := range kc {
				a := packedA[k*mr+row]
				b := packedB[k*nr+col]
				sum += a * b
			}
			expected[row*n+col] = sum
		}
	}

	t.Logf("Expected c[0:8] = %v", expected[0:8])
	t.Logf("Got      c[0:8] = %v", c[0:8])

	// Compare
	for row := range mr {
		for col := range nr {
			idx := row*n + col
			if c[idx] != expected[idx] {
				t.Errorf("Mismatch at [%d][%d]: got %f, want %f", row, col, c[idx], expected[idx])
			}
		}
	}
}
