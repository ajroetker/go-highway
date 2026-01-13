package matmul

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// referenceBlockMulAdd computes C += A * B using naive triple loop.
// aT is the transposed A (rows are original A columns).
// b is normal B (rows are B rows).
// This computes C += (aT)^T * b = A * B
func referenceBlockMulAdd(aT, b, c []float32, blockDim int) {
	for i := 0; i < blockDim; i++ {
		for j := 0; j < blockDim; j++ {
			var sum float32
			for k := 0; k < blockDim; k++ {
				// A[i,k] = aT[k,i]
				// B[k,j] = b[k*blockDim+j]
				aik := aT[k*blockDim+i]
				bkj := b[k*blockDim+j]
				sum += aik * bkj
			}
			c[i*blockDim+j] += sum
		}
	}
}

// transposeBlock transposes a blockDim x blockDim matrix.
// result[j*blockDim+i] = m[i*blockDim+j]
func transposeBlock(m []float32, blockDim int) []float32 {
	result := make([]float32, blockDim*blockDim)
	for i := 0; i < blockDim; i++ {
		for j := 0; j < blockDim; j++ {
			result[j*blockDim+i] = m[i*blockDim+j]
		}
	}
	return result
}

func TestBlockMulAdd(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	blockSizes := []int{8, 16, 32, 48, 64}

	for _, blockDim := range blockSizes {
		t.Run(sizeStr(blockDim), func(t *testing.T) {
			size := blockDim * blockDim

			// Create test matrices
			a := make([]float32, size)     // Original A
			b := make([]float32, size)     // Original B (NOT transposed)
			c := make([]float32, size)
			expected := make([]float32, size)

			// Fill with random values
			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			// Initialize C with some values (to test accumulation)
			for i := range c {
				c[i] = rand.Float32() * 0.1
				expected[i] = c[i]
			}

			// Transpose A for the optimized kernel
			aT := transposeBlock(a, blockDim)

			// Compute reference: C += A * B (using transposed A format)
			referenceBlockMulAdd(aT, b, expected, blockDim)

			// Compute using BlockMulAdd
			BlockMulAdd(aT, b, c, blockDim)

			// Check results
			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := float32(1e-4) * float32(blockDim)
			if maxErr > tolerance {
				t.Errorf("BlockMulAdd: max error %e exceeds tolerance %e", maxErr, tolerance)
			} else {
				t.Logf("blockDim=%d: max error %e", blockDim, maxErr)
			}
		})
	}
}

func TestBlockMulAdd2(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	blockSizes := []int{8, 16, 32, 48, 64}

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
			BlockMulAdd2(aT, b, c, blockDim)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := float32(1e-4) * float32(blockDim)
			if maxErr > tolerance {
				t.Errorf("BlockMulAdd2: max error %e exceeds tolerance %e", maxErr, tolerance)
			} else {
				t.Logf("blockDim=%d: max error %e", blockDim, maxErr)
			}
		})
	}
}

func TestBlockMulAdd4(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	blockSizes := []int{8, 16, 32, 48, 64}

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
			BlockMulAdd4(aT, b, c, blockDim)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := float32(1e-4) * float32(blockDim)
			if maxErr > tolerance {
				t.Errorf("BlockMulAdd4: max error %e exceeds tolerance %e", maxErr, tolerance)
			} else {
				t.Logf("blockDim=%d: max error %e", blockDim, maxErr)
			}
		})
	}
}

func BenchmarkBlockMulAdd(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

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

		b.Run(sizeStr(blockDim)+"/BlockMulAdd", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				BlockMulAdd(aT, bMat, c, blockDim)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(blockDim)+"/BlockMulAdd2", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				BlockMulAdd2(aT, bMat, c, blockDim)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(blockDim)+"/BlockMulAdd4", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				BlockMulAdd4(aT, bMat, c, blockDim)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// TestBlockMulAddNEON tests the hand-written NEON assembly version.
func TestBlockMulAddNEON(t *testing.T) {
	blockSizes := []int{8, 16, 32, 48, 64}

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
			BlockMulAddNEON(aT, b, c, blockDim)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := float32(1e-4) * float32(blockDim)
			if maxErr > tolerance {
				t.Errorf("BlockMulAddNEON: max error %e exceeds tolerance %e", maxErr, tolerance)
			} else {
				t.Logf("blockDim=%d: max error %e", blockDim, maxErr)
			}
		})
	}
}

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

// TestBlockMulAddNEONFloat64 tests the float64 NEON assembly version.
func TestBlockMulAddNEONFloat64(t *testing.T) {
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
			BlockMulAddNEONFloat64(aT, b, c, blockDim)

			var maxErr float64
			for i := range c {
				err := math.Abs(c[i] - expected[i])
				if err > maxErr {
					maxErr = err
				}
			}

			tolerance := 1e-10 * float64(blockDim)
			if maxErr > tolerance {
				t.Errorf("BlockMulAddNEONFloat64: max error %e exceeds tolerance %e", maxErr, tolerance)
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

// transposeBlockFloat64 transposes a blockDim x blockDim matrix.
func transposeBlockFloat64(m []float64, blockDim int) []float64 {
	result := make([]float64, blockDim*blockDim)
	for i := 0; i < blockDim; i++ {
		for j := 0; j < blockDim; j++ {
			result[j*blockDim+i] = m[i*blockDim+j]
		}
	}
	return result
}

// referenceBlockMulAddFloat64 computes C += A * B using naive triple loop for float64.
// aT is the transposed A, b is normal B.
func referenceBlockMulAddFloat64(aT, b, c []float64, blockDim int) {
	for i := 0; i < blockDim; i++ {
		for j := 0; j < blockDim; j++ {
			var sum float64
			for k := 0; k < blockDim; k++ {
				// A[i,k] = aT[k,i]
				aik := aT[k*blockDim+i]
				bkj := b[k*blockDim+j]
				sum += aik * bkj
			}
			c[i*blockDim+j] += sum
		}
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
	for i := 0; i < blockDim; i++ {
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

// TestParallelBlockMulAdd tests the parallel generic version.
func TestParallelBlockMulAdd(t *testing.T) {
	blockDim := 64
	numBlocks := 8

	size := blockDim * blockDim

	// Create test blocks
	aTs := make([][]float32, numBlocks)
	bs := make([][]float32, numBlocks)
	cs := make([][]float32, numBlocks)
	expected := make([][]float32, numBlocks)

	for blk := 0; blk < numBlocks; blk++ {
		aTs[blk] = make([]float32, size)
		bs[blk] = make([]float32, size)
		cs[blk] = make([]float32, size)
		expected[blk] = make([]float32, size)

		// Fill with block-specific values
		for i := range aTs[blk] {
			aTs[blk][i] = rand.Float32()*2 - 1 + float32(blk)*0.01
		}
		for i := range bs[blk] {
			bs[blk][i] = rand.Float32()*2 - 1
		}
		for i := range cs[blk] {
			cs[blk][i] = rand.Float32() * 0.1
			expected[blk][i] = cs[blk][i]
		}

		// Compute reference
		referenceBlockMulAdd(aTs[blk], bs[blk], expected[blk], blockDim)
	}

	// Run parallel version (uses generic dispatch)
	ParallelBlockMulAdd(aTs, bs, cs, blockDim)

	// Verify all blocks
	for blk := 0; blk < numBlocks; blk++ {
		var maxErr float32
		for i := range cs[blk] {
			err := float32(math.Abs(float64(cs[blk][i] - expected[blk][i])))
			if err > maxErr {
				maxErr = err
			}
		}
		tolerance := float32(1e-4) * float32(blockDim)
		if maxErr > tolerance {
			t.Errorf("Block %d: max error %e exceeds tolerance %e", blk, maxErr, tolerance)
		}
	}
	t.Logf("ParallelBlockMulAdd: %d blocks of %dx%d processed successfully", numBlocks, blockDim, blockDim)
}

// BenchmarkParallelBlockMulAdd benchmarks the parallel generic version.
func BenchmarkParallelBlockMulAdd(b *testing.B) {
	blockDim := 64
	size := blockDim * blockDim
	flopsPerBlock := float64(2 * blockDim * blockDim * blockDim)

	for _, numBlocks := range []int{4, 8, 16, 32} {
		// Create test blocks
		aTs := make([][]float32, numBlocks)
		bs := make([][]float32, numBlocks)
		cs := make([][]float32, numBlocks)

		for blk := 0; blk < numBlocks; blk++ {
			aTs[blk] = make([]float32, size)
			bs[blk] = make([]float32, size)
			cs[blk] = make([]float32, size)

			for i := range aTs[blk] {
				aTs[blk][i] = rand.Float32()
			}
			for i := range bs[blk] {
				bs[blk][i] = rand.Float32()
			}
		}

		totalFlops := flopsPerBlock * float64(numBlocks) / 1e9

		b.Run(sizeStr(numBlocks)+"blocks/Sequential", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for blk := 0; blk < numBlocks; blk++ {
					BlockMulAdd(aTs[blk], bs[blk], cs[blk], blockDim)
				}
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := totalFlops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(numBlocks)+"blocks/Parallel", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ParallelBlockMulAdd(aTs, bs, cs, blockDim)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := totalFlops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkBlockMulAddASM benchmarks the hand-written assembly versions.
func BenchmarkBlockMulAddASM(b *testing.B) {
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

		b.Run(sizeStr(blockDim)+"/NEON", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				BlockMulAddNEON(aT, bMat, c, blockDim)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

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
