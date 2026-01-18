//go:build !noasm && darwin && arm64

package asm

import (
	"fmt"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

func TestF16Debug(t *testing.T) {
	m, n, k := 16, 16, 16

	a := make([]hwy.Float16, m*k)
	b := make([]hwy.Float16, k*n)
	at := make([]hwy.Float16, k*m)
	c := make([]hwy.Float16, m*n)
	expected := make([]hwy.Float16, m*n)

	// Fill with simple test values
	for i := range a {
		a[i] = hwy.NewFloat16(float32(i%7) + 0.5)
	}
	for i := range b {
		b[i] = hwy.NewFloat16(float32(i%11) + 0.25)
	}

	// Transpose A
	transposeMatrixF16(a, m, k, at)

	// Reference
	matmulReferenceF16(a, b, expected, m, n, k)

	// SME
	MatMulFMOPAF16(at, b, c, m, n, k)

	// Print first few elements
	fmt.Println("First 16 elements of C (row 0):")
	for j := range 16 {
		fmt.Printf("  [0,%d] expected=%.4f actual=%.4f diff=%.4f\n",
			j, expected[j].Float32(), c[j].Float32(),
			c[j].Float32()-expected[j].Float32())
	}

	fmt.Println("\nFirst column of C:")
	for i := range 16 {
		fmt.Printf("  [%d,0] expected=%.4f actual=%.4f diff=%.4f\n",
			i, expected[i*n].Float32(), c[i*n].Float32(),
			c[i*n].Float32()-expected[i*n].Float32())
	}
}
