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
	for i := range blockDim {
		for j := range blockDim {
			var sum float32
			for k := range blockDim {
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
	for i := range blockDim {
		for j := range blockDim {
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
			a := make([]float32, size) // Original A
			b := make([]float32, size) // Original B (NOT transposed)
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

	for blk := range numBlocks {
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
	for blk := range numBlocks {
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

		for blk := range numBlocks {
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
				for blk := range numBlocks {
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
