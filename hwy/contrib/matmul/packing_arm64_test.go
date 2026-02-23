//go:build arm64

package matmul

import (
	"testing"
)

// TestNeonVsFallbackKernel verifies that NEON and fallback kernels produce
// the same results, specifically for the edge position (ir=12, jr=8) that
// triggered the bounds check bug.
func TestNeonVsFallbackKernel(t *testing.T) {
	mr, nr := 4, 8
	m, n, k := 16, 16, 16

	packedA := make([]float32, k*mr)
	for i := range packedA {
		packedA[i] = float32(i + 1)
	}
	packedB := make([]float32, k*nr)
	for i := range packedB {
		packedB[i] = float32(i + 1)
	}

	// Call fallback directly
	cFallback := make([]float32, m*n)
	BasePackedMicroKernel_fallback(packedA, packedB, cFallback, n, 12, 8, k, mr, nr)

	// Call C asm directly
	cNeon := make([]float32, m*n)
	packedMicroKernelAsmF32(packedA, packedB, cNeon, n, 12, 8, k, mr, nr)

	// Verify non-zero results
	if cFallback[200] == 0 {
		t.Errorf("Fallback kernel produced 0 at c[200]")
	}
	if cNeon[200] == 0 {
		t.Errorf("NEON kernel produced 0 at c[200]")
	}

	// Compare results
	for i := 200; i < 208; i++ {
		if cNeon[i] != cFallback[i] {
			t.Errorf("Mismatch at c[%d]: NEON=%f, fallback=%f", i, cNeon[i], cFallback[i])
		}
	}
}
