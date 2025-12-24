//go:build amd64 && goexperiment.simd

package contrib

import (
	"math"
	"simd/archsimd"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

const benchSize = 1024

// Test correctness using input ranges that match the existing accuracy tests.
// The SIMD implementations have been validated for these ranges.

func TestExpTransform(t *testing.T) {
	// Use values from TestExp32_Accuracy that are known to work
	input := []float32{-10, -1, 0, 0.5, 1, 2, 5, 10, -5, -2, 0.1, 0.9, 3, 4, 6, 7}
	output := make([]float32, len(input))

	ExpTransform(input, output)

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("ExpTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestLogTransform(t *testing.T) {
	// Use positive values from reasonable range
	input := []float32{0.1, 0.5, 1.0, 2.0, 2.718, 5.0, 10.0, 100.0, 0.01, 0.25, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0}
	output := make([]float32, len(input))

	LogTransform(input, output)

	for i := range input {
		expected := float32(math.Log(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("LogTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestSinTransform(t *testing.T) {
	// Use values in [-2π, 2π] range
	input := []float32{0, 0.5, 1.0, 1.57, 2.0, 3.14, 4.0, 5.0, -0.5, -1.0, -1.57, -2.0, -3.14, -4.0, 6.0, 6.28}
	output := make([]float32, len(input))

	SinTransform(input, output)

	for i := range input {
		expected := float32(math.Sin(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("SinTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestCosTransform(t *testing.T) {
	// Use values in [-2π, 2π] range
	input := []float32{0, 0.5, 1.0, 1.57, 2.0, 3.14, 4.0, 5.0, -0.5, -1.0, -1.57, -2.0, -3.14, -4.0, 6.0, 6.28}
	output := make([]float32, len(input))

	CosTransform(input, output)

	for i := range input {
		expected := float32(math.Cos(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("CosTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestTanhTransform(t *testing.T) {
	// Use values in [-5, 5] range (tanh saturates beyond this)
	input := []float32{-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, -3, -0.1, 0.1, 3, -4, 4, 1.5}
	output := make([]float32, len(input))

	TanhTransform(input, output)

	for i := range input {
		expected := float32(math.Tanh(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("TanhTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestSigmoidTransform(t *testing.T) {
	// Use values in [-10, 10] range
	input := []float32{-10, -5, -2, -1, 0, 1, 2, 5, 10, -3, -0.5, 0.5, 3, -4, 4, 6}
	output := make([]float32, len(input))

	SigmoidTransform(input, output)

	for i := range input {
		expected := float32(1.0 / (1.0 + math.Exp(-float64(input[i]))))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("SigmoidTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestErfTransform(t *testing.T) {
	// Use values in [-3, 3] range (erf saturates beyond this)
	input := []float32{-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, -1.5, -0.25, 0.25, 1.5, -2.5, 2.5, 0.75}
	output := make([]float32, len(input))

	ErfTransform(input, output)

	for i := range input {
		expected := float32(math.Erf(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("ErfTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

// Debug test to isolate where the SIMD issue is
func TestLoadStoreRoundtrip(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	output := make([]float32, 8)

	// Test that LoadFloat32x8Slice and StoreSlice work correctly
	x := archsimd.LoadFloat32x8Slice(input)
	x.StoreSlice(output)

	for i := range input {
		if output[i] != input[i] {
			t.Errorf("LoadStoreRoundtrip[%d]: got %v, want %v", i, output[i], input[i])
		}
	}
}

func TestExpAVX2Direct(t *testing.T) {
	// Test calling Exp_AVX2_F32x8 directly with known values
	input := []float32{0, 1, 2, -1, 0.5, -0.5, 0.1, -0.1}
	output := make([]float32, 8)

	x := archsimd.LoadFloat32x8Slice(input)
	result := Exp_AVX2_F32x8(x)
	result.StoreSlice(output)

	t.Logf("Input:  %v", input)
	t.Logf("Output: %v", output)

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		t.Logf("[%d] input=%v output=%v expected=%v", i, input[i], output[i], expected)
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("ExpAVX2Direct[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

// Test that the hwy.Vec wrapper with 8 elements actually uses SIMD
func TestExpVec8Elements(t *testing.T) {
	// Use exactly 8 elements to ensure SIMD path runs (not scalar fallback)
	input := []float32{0, 1, 2, -1, 0.5, -0.5, 0.1, -0.1}
	v := hwy.Load(input)
	result := Exp(v)
	output := result.Data()

	t.Logf("Input:  %v", input)
	t.Logf("Output: %v", output)
	t.Logf("NumLanes: %d, MaxLanes: %d", v.NumLanes(), hwy.MaxLanes[float32]())

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		t.Logf("[%d] input=%v output=%v expected=%v", i, input[i], output[i], expected)
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("ExpVec8Elements[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
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
