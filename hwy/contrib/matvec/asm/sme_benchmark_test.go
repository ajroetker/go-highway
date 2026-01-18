//go:build !noasm && darwin && arm64

package asm

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/ajroetker/go-highway/hwy/contrib/matvec"
)

// transposeForMatVec transposes rows×cols matrix M into cols×rows matrix MT
func transposeForMatVec(m []float32, rows, cols int, mt []float32) {
	for i := range rows {
		for k := range cols {
			mt[k*rows+i] = m[i*cols+k]
		}
	}
}

// transposeForMatVec64 transposes rows×cols matrix M into cols×rows matrix MT for float64
func transposeForMatVec64(m []float64, rows, cols int, mt []float64) {
	for i := range rows {
		for k := range cols {
			mt[k*rows+i] = m[i*cols+k]
		}
	}
}

// matvecReference computes result = M * v using naive loop
func matvecReference(m []float32, rows, cols int, v, result []float32) {
	for i := range rows {
		var sum float32
		for j := range cols {
			sum += m[i*cols+j] * v[j]
		}
		result[i] = sum
	}
}

// matvecReference64 computes result = M * v for float64
func matvecReference64(m []float64, rows, cols int, v, result []float64) {
	for i := range rows {
		var sum float64
		for j := range cols {
			sum += m[i*cols+j] * v[j]
		}
		result[i] = sum
	}
}

func sizeStr(n int) string {
	return string(rune('0'+n/100)) + string(rune('0'+(n/10)%10)) + string(rune('0'+n%10))
}

// TestGoatGeneratedMatVecF32 tests correctness of the goat-generated SME f32 implementation
func TestGoatGeneratedMatVecF32(t *testing.T) {
	sizes := []int{16, 32, 64, 128}

	for _, size := range sizes {
		rows, cols := size, size
		t.Run(sizeStr(size), func(t *testing.T) {
			m := make([]float32, rows*cols)
			mt := make([]float32, cols*rows)
			v := make([]float32, cols)
			result := make([]float32, rows)
			expected := make([]float32, rows)

			// Fill with test values
			for i := range m {
				m[i] = float32(i%7) + 0.5
			}
			for i := range v {
				v[i] = float32(i%11) + 0.25
			}

			// Transpose M for the MT-based function
			transposeForMatVec(m, rows, cols, mt)

			// Reference implementation
			matvecReference(m, rows, cols, v, expected)

			// Goat-generated SME implementation
			MatVecSMEF32(mt, v, result, rows, cols)

			var maxErr float32
			for i := range result {
				err := float32(math.Abs(float64(result[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}
			t.Logf("size %dx%d: max error %e", size, size, maxErr)

			tolerance := float32(1e-4) * float32(cols)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// TestGoatGeneratedMatVecF64 tests correctness of the goat-generated SME f64 implementation
func TestGoatGeneratedMatVecF64(t *testing.T) {
	sizes := []int{8, 16, 32, 64}

	for _, size := range sizes {
		rows, cols := size, size
		t.Run(sizeStr(size), func(t *testing.T) {
			m := make([]float64, rows*cols)
			mt := make([]float64, cols*rows)
			v := make([]float64, cols)
			result := make([]float64, rows)
			expected := make([]float64, rows)

			// Fill with test values
			for i := range m {
				m[i] = float64(i%7) + 0.5
			}
			for i := range v {
				v[i] = float64(i%11) + 0.25
			}

			// Transpose M for the MT-based function
			transposeForMatVec64(m, rows, cols, mt)

			// Reference implementation
			matvecReference64(m, rows, cols, v, expected)

			// Goat-generated SME implementation
			MatVecSMEF64(mt, v, result, rows, cols)

			var maxErr float64
			for i := range result {
				err := math.Abs(result[i] - expected[i])
				if err > maxErr {
					maxErr = err
				}
			}
			t.Logf("size %dx%d: max error %e", size, size, maxErr)

			if maxErr > 1e-9 {
				t.Errorf("max error %e exceeds threshold 1e-9", maxErr)
			}
		})
	}
}

// BenchmarkGoatVsHandwritten_MatVec_F32 compares goat-generated vs handwritten SME assembly
func BenchmarkGoatVsHandwritten_MatVec_F32(b *testing.B) {
	sizes := []int{64, 128, 192}

	for _, size := range sizes {
		rows, cols := size, size

		m := make([]float32, rows*cols)
		mt := make([]float32, cols*rows)
		v := make([]float32, cols)
		result := make([]float32, rows)

		for i := range m {
			m[i] = rand.Float32()
		}
		for i := range v {
			v[i] = rand.Float32()
		}
		transposeForMatVec(m, rows, cols, mt)

		flops := float64(2*rows*cols) / 1e9

		// Benchmark goat-generated assembly
		b.Run(sizeStr(size)+"/GoatGenerated", func(b *testing.B) {
			b.SetBytes(int64((rows*cols + cols + rows) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatVecSMEF32(mt, v, result, rows, cols)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		// Benchmark handwritten assembly (through public API)
		b.Run(sizeStr(size)+"/Handwritten", func(b *testing.B) {
			b.SetBytes(int64((rows*cols + cols + rows) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// This calls matvecSME which uses the handwritten assembly
				matvec.MatVecFloat32(m, rows, cols, v, result)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		// Handwritten direct (calling internal function)
		b.Run(sizeStr(size)+"/HandwrittenDirect", func(b *testing.B) {
			b.SetBytes(int64((rows*cols + cols + rows) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				matvec_sme_f32_handwritten(
					unsafe.Pointer(&mt[0]),
					unsafe.Pointer(&v[0]),
					unsafe.Pointer(&result[0]),
					int64(rows),
					int64(cols),
				)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkGoatVsHandwritten_MatVec_F64 compares goat-generated vs handwritten SME assembly for float64
func BenchmarkGoatVsHandwritten_MatVec_F64(b *testing.B) {
	sizes := []int{64, 128, 192}

	for _, size := range sizes {
		rows, cols := size, size

		m := make([]float64, rows*cols)
		mt := make([]float64, cols*rows)
		v := make([]float64, cols)
		result := make([]float64, rows)

		for i := range m {
			m[i] = rand.Float64()
		}
		for i := range v {
			v[i] = rand.Float64()
		}
		transposeForMatVec64(m, rows, cols, mt)

		flops := float64(2*rows*cols) / 1e9

		// Benchmark goat-generated assembly
		b.Run(sizeStr(size)+"/GoatGenerated", func(b *testing.B) {
			b.SetBytes(int64((rows*cols + cols + rows) * 8))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatVecSMEF64(mt, v, result, rows, cols)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		// Benchmark handwritten assembly (through public API)
		b.Run(sizeStr(size)+"/Handwritten", func(b *testing.B) {
			b.SetBytes(int64((rows*cols + cols + rows) * 8))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				matvec.MatVecFloat64(m, rows, cols, v, result)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		// Handwritten direct
		b.Run(sizeStr(size)+"/HandwrittenDirect", func(b *testing.B) {
			b.SetBytes(int64((rows*cols + cols + rows) * 8))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				matvec_sme_f64_handwritten(
					unsafe.Pointer(&mt[0]),
					unsafe.Pointer(&v[0]),
					unsafe.Pointer(&result[0]),
					int64(rows),
					int64(cols),
				)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// Link to handwritten assembly functions via go:linkname
// This allows direct comparison without the wrapper overhead

//go:linkname matvec_sme_f32_handwritten github.com/ajroetker/go-highway/hwy/contrib/matvec.matvec_sme_f32
func matvec_sme_f32_handwritten(mt, v, result unsafe.Pointer, rows, cols int64)

//go:linkname matvec_sme_f64_handwritten github.com/ajroetker/go-highway/hwy/contrib/matvec.matvec_sme_f64
func matvec_sme_f64_handwritten(mt, v, result unsafe.Pointer, rows, cols int64)
