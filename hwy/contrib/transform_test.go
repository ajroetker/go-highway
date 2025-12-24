//go:build amd64 && goexperiment.simd

package contrib

import (
	"math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

const benchSize = 1024

// Test correctness

func TestExpTransform(t *testing.T) {
	input := make([]float32, 100)
	output := make([]float32, 100)
	for i := range input {
		input[i] = float32(i) * 0.1
	}

	ExpTransform(input, output)

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-5) {
			t.Errorf("ExpTransform[%d]: got %v, want %v", i, output[i], expected)
		}
	}
}

func TestLogTransform(t *testing.T) {
	input := make([]float32, 100)
	output := make([]float32, 100)
	for i := range input {
		input[i] = float32(i+1) * 0.1 // avoid log(0)
	}

	LogTransform(input, output)

	for i := range input {
		expected := float32(math.Log(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-5) {
			t.Errorf("LogTransform[%d]: got %v, want %v", i, output[i], expected)
		}
	}
}

func TestSinTransform(t *testing.T) {
	input := make([]float32, 100)
	output := make([]float32, 100)
	for i := range input {
		input[i] = float32(i) * 0.1
	}

	SinTransform(input, output)

	for i := range input {
		expected := float32(math.Sin(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-5) {
			t.Errorf("SinTransform[%d]: got %v, want %v", i, output[i], expected)
		}
	}
}

func TestCosTransform(t *testing.T) {
	input := make([]float32, 100)
	output := make([]float32, 100)
	for i := range input {
		input[i] = float32(i) * 0.1
	}

	CosTransform(input, output)

	for i := range input {
		expected := float32(math.Cos(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-5) {
			t.Errorf("CosTransform[%d]: got %v, want %v", i, output[i], expected)
		}
	}
}

func TestTanhTransform(t *testing.T) {
	input := make([]float32, 100)
	output := make([]float32, 100)
	for i := range input {
		input[i] = float32(i-50) * 0.1
	}

	TanhTransform(input, output)

	for i := range input {
		expected := float32(math.Tanh(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-5) {
			t.Errorf("TanhTransform[%d]: got %v, want %v", i, output[i], expected)
		}
	}
}

func TestSigmoidTransform(t *testing.T) {
	input := make([]float32, 100)
	output := make([]float32, 100)
	for i := range input {
		input[i] = float32(i-50) * 0.1
	}

	SigmoidTransform(input, output)

	for i := range input {
		expected := float32(1.0 / (1.0 + math.Exp(-float64(input[i]))))
		if !closeEnough32(output[i], expected, 1e-5) {
			t.Errorf("SigmoidTransform[%d]: got %v, want %v", i, output[i], expected)
		}
	}
}

func TestErfTransform(t *testing.T) {
	input := make([]float32, 100)
	output := make([]float32, 100)
	for i := range input {
		input[i] = float32(i-50) * 0.1
	}

	ErfTransform(input, output)

	for i := range input {
		expected := float32(math.Erf(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-5) {
			t.Errorf("ErfTransform[%d]: got %v, want %v", i, output[i], expected)
		}
	}
}

func closeEnough32(a, b, tol float32) bool {
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return true
	}
	if math.IsInf(float64(a), 0) && math.IsInf(float64(b), 0) {
		return (a > 0) == (b > 0)
	}
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= tol
}

// Benchmarks - Transform API (zero allocation)

func BenchmarkExpTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ExpTransform(input, output)
	}
}

func BenchmarkLogTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i+1) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		LogTransform(input, output)
	}
}

func BenchmarkSinTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		SinTransform(input, output)
	}
}

func BenchmarkCosTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		CosTransform(input, output)
	}
}

func BenchmarkTanhTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		TanhTransform(input, output)
	}
}

func BenchmarkSigmoidTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		SigmoidTransform(input, output)
	}
}

func BenchmarkErfTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ErfTransform(input, output)
	}
}

// Benchmarks - Stdlib comparison

func BenchmarkExpTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Exp(float64(input[j])))
		}
	}
}

func BenchmarkLogTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i+1) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Log(float64(input[j])))
		}
	}
}

func BenchmarkSinTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Sin(float64(input[j])))
		}
	}
}

func BenchmarkCosTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Cos(float64(input[j])))
		}
	}
}

func BenchmarkTanhTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Tanh(float64(input[j])))
		}
	}
}

func BenchmarkSigmoidTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(1.0 / (1.0 + math.Exp(-float64(input[j]))))
		}
	}
}

func BenchmarkErfTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Erf(float64(input[j])))
		}
	}
}

// Benchmarks - Old Vec API comparison (to show allocation overhead)

func BenchmarkExpVec(b *testing.B) {
	data := make([]float32, benchSize)
	for i := range data {
		data[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(data); j += hwy.MaxLanes[float32]() {
			end := j + hwy.MaxLanes[float32]()
			if end > len(data) {
				end = len(data)
			}
			v := hwy.Load(data[j:end])
			result := Exp(v)
			_ = result
		}
	}
}

func BenchmarkLogVec(b *testing.B) {
	data := make([]float32, benchSize)
	for i := range data {
		data[i] = float32(i+1) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(data); j += hwy.MaxLanes[float32]() {
			end := j + hwy.MaxLanes[float32]()
			if end > len(data) {
				end = len(data)
			}
			v := hwy.Load(data[j:end])
			result := Log(v)
			_ = result
		}
	}
}
