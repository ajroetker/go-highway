//go:build arm64

package tests

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

// TestSmeDot16 tests the simple SME outer product function
func TestSmeDot16(t *testing.T) {
	// Input vectors (16 floats each for 512-bit SME)
	a := make([]float32, 16)
	b := make([]float32, 16)
	c := make([]float32, 16) // Output: first row of outer product

	// Initialize: a[i] = i+1, b[i] = 1
	for i := range a {
		a[i] = float32(i + 1)
		b[i] = 1.0
	}

	// Call SME function
	sme_dot16(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
	)

	// Expected: c[j] = a[0] * b[j] = 1.0 * 1.0 = 1.0 for all j
	// (first row of outer product where a[0]=1, b[j]=1)
	for j := 0; j < 16; j++ {
		assert.Equal(t, float32(1.0), c[j], "c[%d] should be 1.0", j)
	}
}

// TestSmeFmopaTile tests the 16x16 tile outer product
func TestSmeFmopaTile(t *testing.T) {
	// Input vectors
	a := make([]float32, 16) // Column vector
	b := make([]float32, 16) // Row vector
	c := make([]float32, 256) // 16x16 output matrix

	// Initialize: a[i] = i+1, b[j] = j+1
	for i := range a {
		a[i] = float32(i + 1)
		b[i] = float32(i + 1)
	}

	// Call SME function
	sme_fmopa_tile(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
	)

	// Expected: c[i][j] = a[i] * b[j] = (i+1) * (j+1)
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			expected := float32((i + 1) * (j + 1))
			actual := c[i*16+j]
			assert.Equal(t, expected, actual, "c[%d][%d] should be %v, got %v", i, j, expected, actual)
		}
	}
}

// TestSmeMatmulF32 tests the full matrix multiplication
func TestSmeMatmulF32(t *testing.T) {
	// Small 16x16 matrices for testing
	m := int64(16)
	n := int64(16)
	k := int64(1)

	// AT is transposed A: K x M = 1 x 16
	at := make([]float32, k*m)
	// B: K x N = 1 x 16
	b := make([]float32, k*n)
	// C: M x N = 16 x 16
	c := make([]float32, m*n)

	// Initialize: at[0][i] = i+1, b[0][j] = j+1
	// This gives C[i][j] = sum_k(A[i][k] * B[k][j]) = at[0][i] * b[0][j] = (i+1)*(j+1)
	for i := int64(0); i < m; i++ {
		at[i] = float32(i + 1)
	}
	for j := int64(0); j < n; j++ {
		b[j] = float32(j + 1)
	}

	// Call SME matmul
	sme_matmul_f32(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&m),
		unsafe.Pointer(&n),
		unsafe.Pointer(&k),
	)

	// Verify: C[i][j] = (i+1) * (j+1)
	for i := int64(0); i < m; i++ {
		for j := int64(0); j < n; j++ {
			expected := float32((i + 1) * (j + 1))
			actual := c[i*n+j]
			assert.Equal(t, expected, actual, "C[%d][%d] should be %v, got %v", i, j, expected, actual)
		}
	}
}
