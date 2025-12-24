package contrib

import (
	"math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// ULP (Units in the Last Place) comparison for floating-point accuracy testing
func ulpDistance32(a, b float32) float32 {
	if a == b {
		return 0
	}
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return 0
	}
	if math.IsInf(float64(a), 0) || math.IsInf(float64(b), 0) {
		if (math.IsInf(float64(a), 1) && math.IsInf(float64(b), 1)) ||
			(math.IsInf(float64(a), -1) && math.IsInf(float64(b), -1)) {
			return 0
		}
		return float32(math.Inf(1))
	}
	diff := math.Abs(float64(a - b))
	ulp := math.Abs(float64(math.Nextafter32(a, float32(math.Inf(1))) - a))
	if ulp == 0 {
		ulp = 1e-45 // Smallest positive float32
	}
	return float32(diff / ulp)
}

func ulpDistance64(a, b float64) float64 {
	if a == b {
		return 0
	}
	if math.IsNaN(a) && math.IsNaN(b) {
		return 0
	}
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		if (math.IsInf(a, 1) && math.IsInf(b, 1)) ||
			(math.IsInf(a, -1) && math.IsInf(b, -1)) {
			return 0
		}
		return math.Inf(1)
	}
	diff := math.Abs(a - b)
	ulp := math.Abs(math.Nextafter(a, math.Inf(1)) - a)
	if ulp == 0 {
		ulp = 5e-324 // Smallest positive float64
	}
	return diff / ulp
}

// Test Exp function accuracy
func TestExp32_Accuracy(t *testing.T) {
	testCases := []float32{
		0, 1, 2, -1, -2,
		0.5, -0.5,
		10, -10,
		20, -20,
		math.E, -math.E,
		math.Ln2, -math.Ln2,
	}

	for _, x := range testCases {
		data := []float32{x}
		v := hwy.Load(data)
		result := Exp(v)
		got := result.Data()[0]
		want := float32(math.Exp(float64(x)))

		ulp := ulpDistance32(got, want)
		if ulp > 4 && !math.IsInf(float64(want), 0) {
			t.Errorf("Exp32(%v) = %v, want %v (ULP error: %v)", x, got, want, ulp)
		}
	}
}

func TestExp32_SpecialValues(t *testing.T) {
	testCases := []struct {
		input float32
		want  float32
	}{
		{0, 1},
		{float32(math.Inf(1)), float32(math.Inf(1))},
		{float32(math.Inf(-1)), 0},
		{float32(math.NaN()), float32(math.NaN())},
		{100, float32(math.Inf(1))}, // Overflow
		{-100, 0},                    // Underflow
	}

	for _, tc := range testCases {
		data := []float32{tc.input}
		v := hwy.Load(data)
		result := Exp(v)
		got := result.Data()[0]

		if math.IsNaN(float64(tc.want)) {
			if !math.IsNaN(float64(got)) {
				t.Errorf("Exp32(%v) = %v, want NaN", tc.input, got)
			}
		} else if math.IsInf(float64(tc.want), 0) {
			if !math.IsInf(float64(got), int(math.Copysign(1, float64(tc.want)))) {
				t.Errorf("Exp32(%v) = %v, want %v", tc.input, got, tc.want)
			}
		} else if got != tc.want {
			t.Errorf("Exp32(%v) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

func TestExp64_Accuracy(t *testing.T) {
	testCases := []float64{
		0, 1, 2, -1, -2,
		0.5, -0.5,
		10, -10,
		20, -20,
		math.E, -math.E,
		math.Ln2, -math.Ln2,
	}

	for _, x := range testCases {
		data := []float64{x}
		v := hwy.Load(data)
		result := Exp(v)
		got := result.Data()[0]
		want := math.Exp(x)

		ulp := ulpDistance64(got, want)
		if ulp > 4 && !math.IsInf(want, 0) {
			t.Errorf("Exp64(%v) = %v, want %v (ULP error: %v)", x, got, want, ulp)
		}
	}
}

// Test Log function accuracy
func TestLog32_Accuracy(t *testing.T) {
	testCases := []float32{
		1, 2, 10, 100,
		math.E, math.Pi,
		0.1, 0.5, 0.9,
		1.5, 2.5,
	}

	for _, x := range testCases {
		data := []float32{x}
		v := hwy.Load(data)
		result := Log(v)
		got := result.Data()[0]
		want := float32(math.Log(float64(x)))

		ulp := ulpDistance32(got, want)
		if ulp > 32 { // Base implementation has relaxed tolerance
			t.Errorf("Log32(%v) = %v, want %v (ULP error: %v)", x, got, want, ulp)
		}
	}
}

func TestLog32_SpecialValues(t *testing.T) {
	testCases := []struct {
		input float32
		want  float32
	}{
		{1, 0},
		{float32(math.Inf(1)), float32(math.Inf(1))},
		{0, float32(math.Inf(-1))},
		{-1, float32(math.NaN())},
		{float32(math.NaN()), float32(math.NaN())},
	}

	for _, tc := range testCases {
		data := []float32{tc.input}
		v := hwy.Load(data)
		result := Log(v)
		got := result.Data()[0]

		if math.IsNaN(float64(tc.want)) {
			if !math.IsNaN(float64(got)) {
				t.Errorf("Log32(%v) = %v, want NaN", tc.input, got)
			}
		} else if math.IsInf(float64(tc.want), 0) {
			if !math.IsInf(float64(got), int(math.Copysign(1, float64(tc.want)))) {
				t.Errorf("Log32(%v) = %v, want %v", tc.input, got, tc.want)
			}
		} else if math.Abs(float64(got-tc.want)) > 1e-6 {
			t.Errorf("Log32(%v) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

// Test Sin function accuracy
func TestSin32_Accuracy(t *testing.T) {
	testCases := []float32{
		0,
		math.Pi / 6, math.Pi / 4, math.Pi / 3, math.Pi / 2,
		math.Pi, 3 * math.Pi / 2, 2 * math.Pi,
		-math.Pi / 4, -math.Pi / 2, -math.Pi,
	}

	for _, x := range testCases {
		data := []float32{x}
		v := hwy.Load(data)
		result := Sin(v)
		got := result.Data()[0]
		want := float32(math.Sin(float64(x)))

		diff := math.Abs(float64(got - want))
		if diff > 1e-5 { // Reasonable tolerance for float32 trig functions
			t.Errorf("Sin32(%v) = %v, want %v (diff: %v)", x, got, want, diff)
		}
	}
}

// Test Cos function accuracy
func TestCos32_Accuracy(t *testing.T) {
	testCases := []float32{
		0,
		math.Pi / 6, math.Pi / 4, math.Pi / 3, math.Pi / 2,
		math.Pi, 3 * math.Pi / 2, 2 * math.Pi,
		-math.Pi / 4, -math.Pi / 2, -math.Pi,
	}

	for _, x := range testCases {
		data := []float32{x}
		v := hwy.Load(data)
		result := Cos(v)
		got := result.Data()[0]
		want := float32(math.Cos(float64(x)))

		diff := math.Abs(float64(got - want))
		if diff > 1e-5 {
			t.Errorf("Cos32(%v) = %v, want %v (diff: %v)", x, got, want, diff)
		}
	}
}

// Test SinCos function
func TestSinCos32(t *testing.T) {
	testCases := []float32{
		0, math.Pi / 4, math.Pi / 2, math.Pi,
	}

	for _, x := range testCases {
		data := []float32{x}
		v := hwy.Load(data)
		sinResult, cosResult := SinCos(v)
		gotSin := sinResult.Data()[0]
		gotCos := cosResult.Data()[0]
		wantSin := float32(math.Sin(float64(x)))
		wantCos := float32(math.Cos(float64(x)))

		diffSin := math.Abs(float64(gotSin - wantSin))
		diffCos := math.Abs(float64(gotCos - wantCos))

		if diffSin > 1e-5 {
			t.Errorf("SinCos32(%v) sin = %v, want %v", x, gotSin, wantSin)
		}
		if diffCos > 1e-5 {
			t.Errorf("SinCos32(%v) cos = %v, want %v", x, gotCos, wantCos)
		}
	}
}

// Test Tanh function
func TestTanh32_Accuracy(t *testing.T) {
	testCases := []float32{
		-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3,
	}

	for _, x := range testCases {
		data := []float32{x}
		v := hwy.Load(data)
		result := Tanh(v)
		got := result.Data()[0]
		want := float32(math.Tanh(float64(x)))

		diff := math.Abs(float64(got - want))
		if diff > 1e-5 {
			t.Errorf("Tanh32(%v) = %v, want %v (diff: %v)", x, got, want, diff)
		}
	}
}

func TestTanh32_SpecialValues(t *testing.T) {
	testCases := []struct {
		input float32
		want  float32
	}{
		{0, 0},
		{float32(math.Inf(1)), 1},
		{float32(math.Inf(-1)), -1},
		{float32(math.NaN()), float32(math.NaN())},
	}

	for _, tc := range testCases {
		data := []float32{tc.input}
		v := hwy.Load(data)
		result := Tanh(v)
		got := result.Data()[0]

		if math.IsNaN(float64(tc.want)) {
			if !math.IsNaN(float64(got)) {
				t.Errorf("Tanh32(%v) = %v, want NaN", tc.input, got)
			}
		} else if got != tc.want {
			t.Errorf("Tanh32(%v) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

// Test Sigmoid function
func TestSigmoid32_Accuracy(t *testing.T) {
	testCases := []float32{
		-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5,
	}

	for _, x := range testCases {
		data := []float32{x}
		v := hwy.Load(data)
		result := Sigmoid(v)
		got := result.Data()[0]
		want := float32(1.0 / (1.0 + math.Exp(float64(-x))))

		diff := math.Abs(float64(got - want))
		if diff > 1e-5 {
			t.Errorf("Sigmoid32(%v) = %v, want %v (diff: %v)", x, got, want, diff)
		}
	}
}

func TestSigmoid32_Range(t *testing.T) {
	// Sigmoid should return values in [0, 1]
	// For extreme finite values, floating-point can return exactly 0 or 1
	testCases := []float32{
		-100, -10, -1, 0, 1, 10, 100,
	}

	for _, x := range testCases {
		data := []float32{x}
		v := hwy.Load(data)
		result := Sigmoid(v)
		got := result.Data()[0]

		// Check value is in [0, 1] (closed interval due to FP precision limits)
		if got < 0 || got > 1 {
			t.Errorf("Sigmoid32(%v) = %v, should be in [0, 1]", x, got)
		}

		// For moderate values, should be strictly in (0, 1)
		if math.Abs(float64(x)) < 20 && (got <= 0 || got >= 1) {
			t.Errorf("Sigmoid32(%v) = %v, should be in (0, 1) for moderate inputs", x, got)
		}
	}
}

// Test Erf function
func TestErf32_Accuracy(t *testing.T) {
	testCases := []float32{
		-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3,
	}

	for _, x := range testCases {
		data := []float32{x}
		v := hwy.Load(data)
		result := Erf(v)
		got := result.Data()[0]
		want := float32(math.Erf(float64(x)))

		diff := math.Abs(float64(got - want))
		if diff > 1e-4 { // Slightly relaxed tolerance for erf
			t.Errorf("Erf32(%v) = %v, want %v (diff: %v)", x, got, want, diff)
		}
	}
}

// Test vectorized operations (multiple lanes)
func TestExp32_Vectorized(t *testing.T) {
	data := []float32{0, 1, 2, -1}
	v := hwy.Load(data)
	result := Exp(v)
	output := result.Data()

	for i, x := range data {
		want := float32(math.Exp(float64(x)))
		ulp := ulpDistance32(output[i], want)
		if ulp > 4 {
			t.Errorf("Exp32[%d](%v) = %v, want %v (ULP: %v)", i, x, output[i], want, ulp)
		}
	}
}

// Benchmarks
func BenchmarkExp32(b *testing.B) {
	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i%10 - 5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(data); j += hwy.MaxLanes[float32]() {
			v := hwy.Load(data[j:])
			result := Exp(v)
			_ = result
		}
	}
}

func BenchmarkExp32_Stdlib(b *testing.B) {
	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i%10 - 5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range data {
			_ = math.Exp(float64(data[j]))
		}
	}
}

func BenchmarkSigmoid32(b *testing.B) {
	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i%10 - 5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(data); j += hwy.MaxLanes[float32]() {
			v := hwy.Load(data[j:])
			result := Sigmoid(v)
			_ = result
		}
	}
}

func BenchmarkLog32(b *testing.B) {
	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(data); j += hwy.MaxLanes[float32]() {
			v := hwy.Load(data[j:])
			result := Log(v)
			_ = result
		}
	}
}

func BenchmarkTanh32(b *testing.B) {
	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i%10 - 5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(data); j += hwy.MaxLanes[float32]() {
			v := hwy.Load(data[j:])
			result := Tanh(v)
			_ = result
		}
	}
}
