//go:build darwin && arm64

package matvec

import (
	"math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// scalarMatVec computes matrix-vector product using scalar operations for reference.
func scalarMatVec(m []float32, rows, cols int, v, result []float32) {
	for i := range rows {
		var sum float32
		for j := range cols {
			sum += m[i*cols+j] * v[j]
		}
		result[i] = sum
	}
}

func scalarMatVec64(m []float64, rows, cols int, v, result []float64) {
	for i := range rows {
		var sum float64
		for j := range cols {
			sum += m[i*cols+j] * v[j]
		}
		result[i] = sum
	}
}

// TestMatVecSME tests matrix-vector multiplication with SME-aligned dimensions.
// These tests use sizes that will trigger the SME path when available.
func TestMatVecSME(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available on this hardware")
	}

	sizes := []struct {
		rows int
		cols int
	}{
		{64, 64},   // Minimum SME threshold
		{128, 128}, // 2x minimum
		{256, 256}, // Larger
		{64, 128},  // Non-square
		{128, 64},  // Non-square (transposed)
		{256, 512}, // Wider
		{512, 256}, // Taller
	}

	for _, size := range sizes {
		t.Run(formatSize(size.rows, size.cols), func(t *testing.T) {
			// Create test data
			m := make([]float32, size.rows*size.cols)
			for i := range m {
				m[i] = float32(i%100) / 100.0
			}

			v := make([]float32, size.cols)
			for i := range v {
				v[i] = float32(i%50) / 50.0
			}

			result := make([]float32, size.rows)
			expected := make([]float32, size.rows)

			// Compute with MatVec (uses SME when available)
			MatVec(m, size.rows, size.cols, v, result)

			// Compute reference with scalar
			scalarMatVec(m, size.rows, size.cols, v, expected)

			// Compare results
			for i := range result {
				diff := math.Abs(float64(result[i] - expected[i]))
				relErr := diff / math.Max(math.Abs(float64(expected[i])), 1e-7)
				if relErr > 1e-4 {
					t.Errorf("row %d: got %v, want %v (diff=%v, relErr=%v)",
						i, result[i], expected[i], diff, relErr)
				}
			}
		})
	}
}

// TestMatVecSME64 tests float64 matrix-vector multiplication with SME.
func TestMatVecSME64(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available on this hardware")
	}

	sizes := []struct {
		rows int
		cols int
	}{
		{64, 64}, // Minimum SME threshold (8-aligned for float64)
		{128, 128},
		{256, 256},
		{64, 128},
		{128, 64},
	}

	for _, size := range sizes {
		t.Run(formatSize(size.rows, size.cols)+"_f64", func(t *testing.T) {
			// Create test data
			m := make([]float64, size.rows*size.cols)
			for i := range m {
				m[i] = float64(i%100) / 100.0
			}

			v := make([]float64, size.cols)
			for i := range v {
				v[i] = float64(i%50) / 50.0
			}

			result := make([]float64, size.rows)
			expected := make([]float64, size.rows)

			// Compute with MatVec (uses SME when available)
			MatVecFloat64(m, size.rows, size.cols, v, result)

			// Compute reference with scalar
			scalarMatVec64(m, size.rows, size.cols, v, expected)

			// Compare results
			for i := range result {
				diff := math.Abs(result[i] - expected[i])
				relErr := diff / math.Max(math.Abs(expected[i]), 1e-15)
				if relErr > 1e-10 {
					t.Errorf("row %d: got %v, want %v (diff=%v, relErr=%v)",
						i, result[i], expected[i], diff, relErr)
				}
			}
		})
	}
}

// TestMatVecSMEFallback tests that non-aligned sizes fall back to NEON.
func TestMatVecSMEFallback(t *testing.T) {
	// These sizes are not aligned to SME tile size (16 for float32)
	// and should fall back to NEON
	sizes := []struct {
		rows int
		cols int
	}{
		{63, 64},   // rows not aligned
		{65, 128},  // rows not aligned
		{100, 100}, // rows not aligned
		{32, 32},   // below minimum threshold
	}

	for _, size := range sizes {
		t.Run(formatSize(size.rows, size.cols)+"_fallback", func(t *testing.T) {
			m := make([]float32, size.rows*size.cols)
			for i := range m {
				m[i] = float32(i%100) / 100.0
			}

			v := make([]float32, size.cols)
			for i := range v {
				v[i] = float32(i%50) / 50.0
			}

			result := make([]float32, size.rows)
			expected := make([]float32, size.rows)

			// Compute with MatVec
			MatVec(m, size.rows, size.cols, v, result)

			// Compute reference
			scalarMatVec(m, size.rows, size.cols, v, expected)

			// Compare
			for i := range result {
				diff := math.Abs(float64(result[i] - expected[i]))
				if diff > 1e-4 {
					t.Errorf("row %d: got %v, want %v", i, result[i], expected[i])
				}
			}
		})
	}
}

// TestMatVecSMEIdentity tests with identity-like patterns.
func TestMatVecSMEIdentity(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available on this hardware")
	}

	// 64x64 identity matrix
	rows, cols := 64, 64
	m := make([]float32, rows*cols)
	for i := range rows {
		m[i*cols+i] = 1.0
	}

	v := make([]float32, cols)
	for i := range v {
		v[i] = float32(i + 1)
	}

	result := make([]float32, rows)
	MatVec(m, rows, cols, v, result)

	// Identity * v = v
	for i := range result {
		if math.Abs(float64(result[i]-v[i])) > 1e-5 {
			t.Errorf("row %d: got %v, want %v", i, result[i], v[i])
		}
	}
}

// TestMatVecSMEOnes tests with all-ones patterns.
func TestMatVecSMEOnes(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available on this hardware")
	}

	rows, cols := 128, 64
	m := make([]float32, rows*cols)
	for i := range m {
		m[i] = 1.0
	}

	v := make([]float32, cols)
	for i := range v {
		v[i] = 1.0
	}

	result := make([]float32, rows)
	MatVec(m, rows, cols, v, result)

	// All-ones matrix * all-ones vector = [cols, cols, ...]
	expected := float32(cols)
	for i := range result {
		if math.Abs(float64(result[i]-expected)) > 1e-4 {
			t.Errorf("row %d: got %v, want %v", i, result[i], expected)
		}
	}
}

func formatSize(rows, cols int) string {
	return string(rune('0'+rows/100)) + string(rune('0'+(rows/10)%10)) + string(rune('0'+rows%10)) +
		"x" + string(rune('0'+cols/100)) + string(rune('0'+(cols/10)%10)) + string(rune('0'+cols%10))
}

// Benchmarks comparing NEON vs SME

func BenchmarkMatVecSME(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available on this hardware")
	}

	sizes := []struct {
		rows int
		cols int
	}{
		{64, 64},
		{128, 128},
		{256, 256},
		{512, 512},
		{1024, 1024},
	}

	for _, size := range sizes {
		m := make([]float32, size.rows*size.cols)
		for i := range m {
			m[i] = float32(i % 100)
		}
		v := make([]float32, size.cols)
		for i := range v {
			v[i] = float32(i)
		}
		result := make([]float32, size.rows)

		b.Run(formatSize(size.rows, size.cols), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size.rows*size.cols*4 + size.cols*4 + size.rows*4))
			for i := 0; i < b.N; i++ {
				MatVec(m, size.rows, size.cols, v, result)
			}
		})
	}
}

func BenchmarkMatVecNEON(b *testing.B) {
	sizes := []struct {
		rows int
		cols int
	}{
		{64, 64},
		{128, 128},
		{256, 256},
		{512, 512},
		{1024, 1024},
	}

	for _, size := range sizes {
		m := make([]float32, size.rows*size.cols)
		for i := range m {
			m[i] = float32(i % 100)
		}
		v := make([]float32, size.cols)
		for i := range v {
			v[i] = float32(i)
		}
		result := make([]float32, size.rows)

		b.Run(formatSize(size.rows, size.cols), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size.rows*size.cols*4 + size.cols*4 + size.rows*4))
			for i := 0; i < b.N; i++ {
				// Directly call NEON to compare
				BaseMatVec_neon(m, size.rows, size.cols, v, result)
			}
		})
	}
}
