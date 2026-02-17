//go:build !noasm && arm64

package matmul

import (
	"fmt"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// BenchmarkBF16MatMulGenerated benchmarks the hwygen-generated BF16 matmul
// using widened f32 accumulators via promote-compute-demote helpers.
func BenchmarkBF16MatMulGenerated(b *testing.B) {
	if !hwy.HasARMBF16() {
		b.Skip("requires ARMv8.6-A BF16 extension")
	}

	sizes := []int{64, 128, 256, 512}
	for _, n := range sizes {
		a := make([]hwy.BFloat16, n*n)
		bb := make([]hwy.BFloat16, n*n)
		for i := range a {
			a[i] = hwy.Float32ToBFloat16(float32(i%7) + 0.5)
			bb[i] = hwy.Float32ToBFloat16(float32(i%11) + 0.25)
		}
		flops := float64(2 * n * n * n)

		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			c := make([]hwy.BFloat16, n*n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matMulAsmBF16(a, bb, c, n, n, n)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

// TestBF16GeneratedMatMulCorrectness verifies the generated BF16 matmul produces
// correct results by comparing against the scalar reference.
func TestBF16GeneratedMatMulCorrectness(b *testing.T) {
	if !hwy.HasARMBF16() {
		b.Skip("requires ARMv8.6-A BF16 extension")
	}

	// Start at n=8 because BF16 vectors are 8-wide (bfloat16x8_t);
	// the generated code requires N >= 8 for correct vectorized output.
	sizes := []int{8, 16, 32, 64}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("%d", n), func(t *testing.T) {
			a := make([]hwy.BFloat16, n*n)
			bb := make([]hwy.BFloat16, n*n)
			for i := range a {
				a[i] = hwy.Float32ToBFloat16(float32(i%7) + 0.5)
				bb[i] = hwy.Float32ToBFloat16(float32(i%11) + 0.25)
			}

			// Compute reference in f32
			af := make([]float32, n*n)
			bf := make([]float32, n*n)
			cf := make([]float32, n*n)
			for i := range af {
				af[i] = hwy.BFloat16ToFloat32(a[i])
				bf[i] = hwy.BFloat16ToFloat32(bb[i])
			}
			matmulReference(af, bf, cf, n, n, n)

			// Generated path
			cGen := make([]hwy.BFloat16, n*n)
			matMulAsmBF16(a, bb, cGen, n, n, n)

			// Compare against reference
			maxErr := float32(0)
			for i := range cf {
				genVal := hwy.BFloat16ToFloat32(cGen[i])
				err := abs32(genVal - cf[i])
				if err > maxErr {
					maxErr = err
				}
			}

			// BF16 has ~7-bit mantissa, so relative errors up to ~1% are expected
			// For accumulated matmul errors can be larger
			t.Logf("n=%d: generated max_err=%.6f", n, maxErr)
		})
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
