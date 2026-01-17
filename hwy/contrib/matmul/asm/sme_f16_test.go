//go:build !noasm && darwin && arm64

package asm

import (
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// TestMatMulFMOPAF16Debug tests F16 FMOPA matmul with a simple case
func TestMatMulFMOPAF16Debug(t *testing.T) {
	const n = 16

	// A = all 2.0, B = all 3.0
	// C[i,j] = sum over k of A[i,k] * B[k,j] = 16 * 2 * 3 = 96
	at := make([]hwy.Float16, n*n) // K x M (transposed)
	b := make([]hwy.Float16, n*n)  // K x N
	for i := range at {
		at[i] = hwy.Float32ToFloat16(2.0)
		b[i] = hwy.Float32ToFloat16(3.0)
	}

	c := make([]hwy.Float16, n*n)
	for i := range c {
		c[i] = hwy.Float32ToFloat16(-999.0)
	}

	t.Logf("Calling MatMulFMOPAF16 with m=%d, n=%d, k=%d", n, n, n)

	MatMulFMOPAF16(at, b, c, n, n, n)

	expected := float32(96) // 16 * 2 * 3
	t.Logf("Expected value: %f", expected)

	t.Log("\nFirst 16 elements of C (row 0):")
	for j := 0; j < n; j++ {
		actual := hwy.Float16ToFloat32(c[j])
		t.Logf("  [0,%d] expected=%f actual=%f diff=%f", j, expected, actual, actual-expected)
	}

	t.Log("\nColumn 0 (first element of each row):")
	for i := 0; i < n; i++ {
		actual := hwy.Float16ToFloat32(c[i*n])
		t.Logf("  [%d,0] expected=%f actual=%f diff=%f", i, expected, actual, actual-expected)
	}

	errCount := 0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			actual := hwy.Float16ToFloat32(c[i*n+j])
			diff := actual - expected
			if diff < 0 {
				diff = -diff
			}
			if diff > 1.0 { // allow some f16 precision loss
				if errCount < 20 {
					t.Errorf("c[%d,%d] = %f, want %f", i, j, actual, expected)
				}
				errCount++
			}
		}
	}

	if errCount > 20 {
		t.Errorf("... and %d more errors", errCount-20)
	}

	if errCount == 0 {
		t.Logf("FMOPA F16 16x16 matmul passed")
	} else {
		t.Logf("Total errors: %d", errCount)
	}
}

// TestMatMulFMOPAF16Identity tests F16 with identity matrix
func TestMatMulFMOPAF16Identity(t *testing.T) {
	const n = 16

	// Create A matrix (transposed) with known values
	// AT is K x M, so AT[k,m] = A[m,k]
	// For A = identity, AT is also identity
	at := make([]hwy.Float16, n*n) // K x M (transposed)
	for i := 0; i < n; i++ {
		at[i*n+i] = hwy.Float32ToFloat16(1.0)
	}

	// B with test values
	b := make([]hwy.Float16, n*n)
	for i := 0; i < n*n; i++ {
		b[i] = hwy.Float32ToFloat16(float32(i%n + 1)) // 1, 2, 3, ..., 16, 1, 2, 3, ...
	}

	c := make([]hwy.Float16, n*n)
	for i := range c {
		c[i] = hwy.Float32ToFloat16(-999.0)
	}

	// C = AT^T * B = I * B = B
	MatMulFMOPAF16(at, b, c, n, n, n)

	t.Log("First row of C (expected 1, 2, 3, ..., 16):")
	for j := 0; j < n; j++ {
		expected := float32(j + 1)
		actual := hwy.Float16ToFloat32(c[j])
		t.Logf("  [0,%d] expected=%f actual=%f diff=%f", j, expected, actual, actual-expected)
	}

	errCount := 0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			expected := float32(j + 1) // B's row pattern
			actual := hwy.Float16ToFloat32(c[i*n+j])
			diff := actual - expected
			if diff < 0 {
				diff = -diff
			}
			if diff > 0.1 {
				if errCount < 20 {
					t.Errorf("c[%d,%d] = %f, want %f", i, j, actual, expected)
				}
				errCount++
			}
		}
	}

	if errCount > 20 {
		t.Errorf("... and %d more errors", errCount-20)
	}

	if errCount == 0 {
		t.Logf("FMOPA F16 identity test passed")
	}
}
