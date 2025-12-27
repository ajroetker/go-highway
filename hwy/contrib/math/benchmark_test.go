//go:build amd64 && goexperiment.simd

package math_test

import (
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/algo"
	hwymath "github.com/ajroetker/go-highway/hwy/contrib/math"
)

// ============================================================================
// AVX2 vs AVX512 Direct Comparison Benchmarks
// ============================================================================

func BenchmarkExp_AVX2_vs_AVX512(b *testing.B) {
	if hwy.CurrentLevel() < hwy.DispatchAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100) * 0.1
	}

	b.Run("AVX2", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			algo.Transform32(input, output, hwymath.Exp_AVX2_F32x8, hwymath.Exp32Scalar)
		}
	})

	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		b.Run("AVX512", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				algo.Transform32x16(input, output, hwymath.Exp_AVX512_F32x16, hwymath.Exp32Scalar)
			}
		})
	}
}

func BenchmarkLog_AVX2_vs_AVX512(b *testing.B) {
	if hwy.CurrentLevel() < hwy.DispatchAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100+1) * 0.1
	}

	b.Run("AVX2", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			algo.Transform32(input, output, hwymath.Log_AVX2_F32x8, hwymath.Log32Scalar)
		}
	})

	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		b.Run("AVX512", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				algo.Transform32x16(input, output, hwymath.Log_AVX512_F32x16, hwymath.Log32Scalar)
			}
		})
	}
}

func BenchmarkSin_AVX2_vs_AVX512(b *testing.B) {
	if hwy.CurrentLevel() < hwy.DispatchAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100) * 0.1
	}

	b.Run("AVX2", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			algo.Transform32(input, output, hwymath.Sin_AVX2_F32x8, hwymath.Sin32Scalar)
		}
	})

	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		b.Run("AVX512", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				algo.Transform32x16(input, output, hwymath.Sin_AVX512_F32x16, hwymath.Sin32Scalar)
			}
		})
	}
}

func BenchmarkCos_AVX2_vs_AVX512(b *testing.B) {
	if hwy.CurrentLevel() < hwy.DispatchAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100) * 0.1
	}

	b.Run("AVX2", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			algo.Transform32(input, output, hwymath.Cos_AVX2_F32x8, hwymath.Cos32Scalar)
		}
	})

	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		b.Run("AVX512", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				algo.Transform32x16(input, output, hwymath.Cos_AVX512_F32x16, hwymath.Cos32Scalar)
			}
		})
	}
}

func BenchmarkTanh_AVX2_vs_AVX512(b *testing.B) {
	if hwy.CurrentLevel() < hwy.DispatchAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100-50) * 0.1
	}

	b.Run("AVX2", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			algo.Transform32(input, output, hwymath.Tanh_AVX2_F32x8, hwymath.Tanh32Scalar)
		}
	})

	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		b.Run("AVX512", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				algo.Transform32x16(input, output, hwymath.Tanh_AVX512_F32x16, hwymath.Tanh32Scalar)
			}
		})
	}
}

func BenchmarkSigmoid_AVX2_vs_AVX512(b *testing.B) {
	if hwy.CurrentLevel() < hwy.DispatchAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100-50) * 0.1
	}

	b.Run("AVX2", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			algo.Transform32(input, output, hwymath.Sigmoid_AVX2_F32x8, hwymath.Sigmoid32Scalar)
		}
	})

	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		b.Run("AVX512", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				algo.Transform32x16(input, output, hwymath.Sigmoid_AVX512_F32x16, hwymath.Sigmoid32Scalar)
			}
		})
	}
}

func BenchmarkErf_AVX2_vs_AVX512(b *testing.B) {
	if hwy.CurrentLevel() < hwy.DispatchAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100-50) * 0.1
	}

	b.Run("AVX2", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			algo.Transform32(input, output, hwymath.Erf_AVX2_F32x8, hwymath.Erf32Scalar)
		}
	})

	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		b.Run("AVX512", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				algo.Transform32x16(input, output, hwymath.Erf_AVX512_F32x16, hwymath.Erf32Scalar)
			}
		})
	}
}

// ============================================================================
// Float64 AVX2 vs AVX512 Benchmarks
// ============================================================================

func BenchmarkExp64_AVX2_vs_AVX512(b *testing.B) {
	if hwy.CurrentLevel() < hwy.DispatchAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 4096
	input := make([]float64, size)
	output := make([]float64, size)
	for i := range input {
		input[i] = float64(i%100) * 0.1
	}

	b.Run("AVX2", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			algo.Transform64(input, output, hwymath.Exp_AVX2_F64x4, hwymath.Exp64Scalar)
		}
	})

	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		b.Run("AVX512", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				algo.Transform64x8(input, output, hwymath.Exp_AVX512_F64x8, hwymath.Exp64Scalar)
			}
		})
	}
}

func BenchmarkSin64_AVX2_vs_AVX512(b *testing.B) {
	if hwy.CurrentLevel() < hwy.DispatchAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 4096
	input := make([]float64, size)
	output := make([]float64, size)
	for i := range input {
		input[i] = float64(i%100) * 0.1
	}

	b.Run("AVX2", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			algo.Transform64(input, output, hwymath.Sin_AVX2_F64x4, hwymath.Sin64Scalar)
		}
	})

	if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
		b.Run("AVX512", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				algo.Transform64x8(input, output, hwymath.Sin_AVX512_F64x8, hwymath.Sin64Scalar)
			}
		})
	}
}

// ============================================================================
// Scaling Benchmarks (different sizes)
// ============================================================================

func BenchmarkExpTransform_Scaling(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i%100) * 0.1
		}

		name := sizeLabel(size)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				algo.ExpTransform(input, output)
			}
		})
	}
}

func BenchmarkSigmoidTransform_Scaling(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i%100-50) * 0.1
		}

		name := sizeLabel(size)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				algo.SigmoidTransform(input, output)
			}
		})
	}
}

// ============================================================================
// Throughput Benchmarks (bytes/sec)
// ============================================================================

func BenchmarkThroughput_Exp(b *testing.B) {
	size := 16384
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100) * 0.1
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 4)) // 4 bytes per float32
		for i := 0; i < b.N; i++ {
			algo.ExpTransform(input, output)
		}
	})

	b.Run("Stdlib", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			for j := range input {
				output[j] = float32(stdmath.Exp(float64(input[j])))
			}
		}
	})
}

func BenchmarkThroughput_Sigmoid(b *testing.B) {
	size := 16384
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100-50) * 0.1
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			algo.SigmoidTransform(input, output)
		}
	})

	b.Run("Stdlib", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			for j := range input {
				output[j] = float32(1.0 / (1.0 + stdmath.Exp(-float64(input[j]))))
			}
		}
	})
}

// ============================================================================
// Helper Functions
// ============================================================================

func sizeLabel(size int) string {
	switch {
	case size >= 16384:
		return "16K"
	case size >= 4096:
		return "4K"
	case size >= 1024:
		return "1K"
	case size >= 256:
		return "256"
	default:
		return "64"
	}
}

// ============================================================================
// SIMD Level Info
// ============================================================================

func TestPrintSIMDLevel(t *testing.T) {
	t.Logf("SIMD Level: %s", hwy.CurrentLevel())
	t.Logf("SIMD Width: %d bytes", hwy.CurrentWidth())
	t.Logf("AVX2 Available: %v", hwy.CurrentLevel() >= hwy.DispatchAVX2)
	t.Logf("AVX512 Available: %v", hwy.CurrentLevel() >= hwy.DispatchAVX512)
}
