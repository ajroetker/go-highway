//go:build darwin && arm64

package matmul

import (
	"math"
	"math/rand"
	"testing"
)

// TestBlockMulAddFMOPA tests the SME FMOPA assembly version.
func TestBlockMulAddFMOPA(t *testing.T) {
	// FMOPA works on 16x16 tiles, so blockDim must be multiple of 16
	blockSizes := []int{16, 32, 48, 64}

	for _, blockDim := range blockSizes {
		t.Run(sizeStr(blockDim), func(t *testing.T) {
			size := blockDim * blockDim

			a := make([]float32, size)
			b := make([]float32, size)
			c := make([]float32, size)
			expected := make([]float32, size)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}
			for i := range c {
				c[i] = rand.Float32() * 0.1
				expected[i] = c[i]
			}

			aT := transposeBlock(a, blockDim)
			referenceBlockMulAdd(aT, b, expected, blockDim)
			BlockMulAddFMOPA(aT, b, c, blockDim)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := float32(1e-4) * float32(blockDim)
			if maxErr > tolerance {
				t.Errorf("BlockMulAddFMOPA: max error %e exceeds tolerance %e", maxErr, tolerance)
			} else {
				t.Logf("blockDim=%d: max error %e", blockDim, maxErr)
			}
		})
	}
}

// TestBlockMulAddFMOPAFloat64 tests the float64 SME FMOPA assembly version.
func TestBlockMulAddFMOPAFloat64(t *testing.T) {
	// FMOPA f64 works on 8x8 tiles, so blockDim must be multiple of 8
	blockSizes := []int{8, 16, 32, 48, 64}

	for _, blockDim := range blockSizes {
		t.Run(sizeStr(blockDim), func(t *testing.T) {
			size := blockDim * blockDim

			a := make([]float64, size)
			b := make([]float64, size)
			c := make([]float64, size)
			expected := make([]float64, size)

			for i := range a {
				a[i] = rand.Float64()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float64()*2 - 1
			}
			for i := range c {
				c[i] = rand.Float64() * 0.1
				expected[i] = c[i]
			}

			aT := transposeBlockFloat64(a, blockDim)
			referenceBlockMulAddFloat64(aT, b, expected, blockDim)
			BlockMulAddFMOPAFloat64(aT, b, c, blockDim)

			var maxErr float64
			for i := range c {
				err := math.Abs(c[i] - expected[i])
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := 1e-10 * float64(blockDim)
			if maxErr > tolerance {
				t.Errorf("BlockMulAddFMOPAFloat64: max error %e exceeds tolerance %e", maxErr, tolerance)
			} else {
				t.Logf("blockDim=%d: max error %e", blockDim, maxErr)
			}
		})
	}
}

// TestBlockMulAddFMOPADebug tests with simple known values.
func TestBlockMulAddFMOPADebug(t *testing.T) {
	blockDim := 16
	size := blockDim * blockDim

	// Use all 1s for A and B
	a := make([]float32, size)
	b := make([]float32, size)
	c := make([]float32, size)
	expected := make([]float32, size)

	for i := range a {
		a[i] = 1.0
	}
	for i := range b {
		b[i] = 1.0
	}
	// C starts at 0
	for i := range c {
		c[i] = 0.0
		expected[i] = 0.0
	}

	aT := transposeBlock(a, blockDim)

	// Debug: print input arrays
	t.Logf("blockDim = %d, size = %d", blockDim, size)
	t.Logf("a[0:4] = %v", a[0:4])
	t.Logf("aT[0:4] = %v", aT[0:4])
	t.Logf("b[0:4] = %v", b[0:4])
	t.Logf("expected before reference: %v", expected[0:4])

	referenceBlockMulAdd(aT, b, expected, blockDim)

	t.Logf("expected after reference: %v", expected[0:4])

	// Run FMOPA version
	BlockMulAddFMOPA(aT, b, c, blockDim)

	// Print first few values
	t.Logf("Got C after FMOPA: %v", c[0:4])

	// Manual calculation: C[0,0] = sum_k A[0,k] * B[k,0] = sum_k 1*1 = blockDim
	t.Logf("Expected C[0] = %d (sum of %d ones)", blockDim, blockDim)

	// Check
	if c[0] != float32(blockDim) {
		t.Errorf("C[0] = %v, expected %v", c[0], blockDim)
	}
}

// TestBlockMulAddFMOPADebugIdentity tests with identity matrix to verify row selection.
func TestBlockMulAddFMOPADebugIdentity(t *testing.T) {
	blockDim := 16
	size := blockDim * blockDim

	// Use identity matrix for A, all 1s for B
	// C = I * B = B, so C should equal B
	a := make([]float32, size)
	b := make([]float32, size)
	c := make([]float32, size)
	expected := make([]float32, size)

	// A = identity
	for i := range blockDim {
		a[i*blockDim+i] = 1.0
	}
	// B = increasing values
	for i := range b {
		b[i] = float32(i)
	}
	// C starts at 0
	for i := range c {
		c[i] = 0.0
		expected[i] = 0.0
	}

	aT := transposeBlock(a, blockDim)
	referenceBlockMulAdd(aT, b, expected, blockDim)

	// With A=I, C = A*B = B, so expected = b
	t.Logf("expected[0:4] = %v", expected[0:4])
	t.Logf("expected[16:20] = %v (row 1)", expected[16:20])

	BlockMulAddFMOPA(aT, b, c, blockDim)

	t.Logf("got C[0:4] = %v", c[0:4])
	t.Logf("got C[16:20] = %v (row 1)", c[16:20])

	// Check first few rows
	var maxErr float32
	var maxErrIdx int
	for i := range c {
		err := float32(math.Abs(float64(c[i] - expected[i])))
		if err > maxErr {
			maxErr = err
			maxErrIdx = i
		}
	}

	if maxErr > 1e-5 {
		row := maxErrIdx / blockDim
		col := maxErrIdx % blockDim
		t.Errorf("max error %e at [%d,%d] (idx %d): got %v, expected %v",
			maxErr, row, col, maxErrIdx, c[maxErrIdx], expected[maxErrIdx])
		// Print the row where error occurred
		rowStart := row * blockDim
		t.Logf("Row %d expected: %v", row, expected[rowStart:rowStart+4])
		t.Logf("Row %d got:      %v", row, c[rowStart:rowStart+4])
	} else {
		t.Logf("PASS: max error %e", maxErr)
	}
}

// BenchmarkBlockMulAddFMOPA benchmarks the SME FMOPA assembly.
func BenchmarkBlockMulAddFMOPA(b *testing.B) {
	blockSizes := []int{32, 48, 64}

	for _, blockDim := range blockSizes {
		size := blockDim * blockDim

		aT := make([]float32, size)
		bMat := make([]float32, size)
		c := make([]float32, size)

		for i := range aT {
			aT[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*blockDim*blockDim*blockDim) / 1e9

		// Only benchmark FMOPA for sizes that are multiples of 16
		if blockDim%16 == 0 {
			b.Run(sizeStr(blockDim)+"/FMOPA", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					BlockMulAddFMOPA(aT, bMat, c, blockDim)
				}
				b.StopTimer()
				elapsed := b.Elapsed().Seconds()
				gflops := flops * float64(b.N) / elapsed
				b.ReportMetric(gflops, "GFLOPS")
			})
		}
	}
}
