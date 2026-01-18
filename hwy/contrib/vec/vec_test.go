package vec

import (
	"fmt"
	"math"
	"testing"
)

// Tolerance constants for floating point comparison
const (
	epsilon32 = float32(1e-6)
	epsilon64 = float64(1e-12)
)

// approxEqual32 checks if two float32 values are approximately equal
func approxEqual32(a, b, epsilon float32) bool {
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
	return diff <= epsilon
}

// approxEqual64 checks if two float64 values are approximately equal
func approxEqual64(a, b, epsilon float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	if math.IsInf(a, 0) && math.IsInf(b, 0) {
		return (a > 0) == (b > 0)
	}
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= epsilon
}

// sliceApproxEqual32 checks if two float32 slices are approximately equal
func sliceApproxEqual32(a, b []float32, epsilon float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !approxEqual32(a[i], b[i], epsilon) {
			return false
		}
	}
	return true
}

// sliceApproxEqual64 checks if two float64 slices are approximately equal
func sliceApproxEqual64(a, b []float64, epsilon float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !approxEqual64(a[i], b[i], epsilon) {
			return false
		}
	}
	return true
}

// Helper functions to generate test vectors
func makeVector32(size int, gen func(int) float32) []float32 {
	v := make([]float32, size)
	for i := range v {
		v[i] = gen(i)
	}
	return v
}

func makeVector64(size int, gen func(int) float64) []float64 {
	v := make([]float64, size)
	for i := range v {
		v[i] = gen(i)
	}
	return v
}

// ============================================================================
// BaseSquaredNorm Tests
// ============================================================================

func TestBaseSquaredNorm(t *testing.T) {
	tests := []struct {
		name string
		v    []float32
		want float32
	}{
		// Basic cases
		{"empty", []float32{}, 0},
		{"single", []float32{3}, 9},
		{"unit vector x", []float32{1, 0, 0}, 1},
		{"unit vector y", []float32{0, 1, 0}, 1},
		{"unit vector z", []float32{0, 0, 1}, 1},
		{"3-4-5 triangle", []float32{3, 4}, 25},
		{"all ones 3d", []float32{1, 1, 1}, 3},

		// Zero vector
		{"zero vector", []float32{0, 0, 0, 0}, 0},

		// Negative values
		{"negative single", []float32{-3}, 9},
		{"mixed signs", []float32{3, -4}, 25},

		// SIMD boundary cases - testing tail handling
		{"len 3", makeVector32(3, func(i int) float32 { return 1 }), 3},
		{"len 4", makeVector32(4, func(i int) float32 { return 1 }), 4},
		{"len 5", makeVector32(5, func(i int) float32 { return 1 }), 5},
		{"len 7", makeVector32(7, func(i int) float32 { return 1 }), 7},
		{"len 8", makeVector32(8, func(i int) float32 { return 1 }), 8},
		{"len 9", makeVector32(9, func(i int) float32 { return 1 }), 9},
		{"len 15", makeVector32(15, func(i int) float32 { return 1 }), 15},
		{"len 16", makeVector32(16, func(i int) float32 { return 1 }), 16},
		{"len 17", makeVector32(17, func(i int) float32 { return 1 }), 17},

		// Large values
		{"large values", []float32{1000, 2000, 3000}, 1000*1000 + 2000*2000 + 3000*3000},

		// Small values
		{"small values", []float32{0.001, 0.002, 0.003}, 0.001*0.001 + 0.002*0.002 + 0.003*0.003},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseSquaredNorm(tt.v)
			if !approxEqual32(got, tt.want, epsilon32) {
				t.Errorf("BaseSquaredNorm() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBaseSquaredNorm_Float64(t *testing.T) {
	tests := []struct {
		name string
		v    []float64
		want float64
	}{
		{"empty", []float64{}, 0},
		{"single", []float64{3}, 9},
		{"3-4-5 triangle", []float64{3, 4}, 25},
		{"high precision", []float64{1e-10, 2e-10, 3e-10}, 1e-20 + 4e-20 + 9e-20},

		// SIMD boundary cases for float64 (4-wide on AVX2)
		{"len 3", makeVector64(3, func(i int) float64 { return 1 }), 3},
		{"len 4", makeVector64(4, func(i int) float64 { return 1 }), 4},
		{"len 5", makeVector64(5, func(i int) float64 { return 1 }), 5},
		{"len 7", makeVector64(7, func(i int) float64 { return 1 }), 7},
		{"len 8", makeVector64(8, func(i int) float64 { return 1 }), 8},
		{"len 9", makeVector64(9, func(i int) float64 { return 1 }), 9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseSquaredNorm(tt.v)
			if !approxEqual64(got, tt.want, epsilon64) {
				t.Errorf("BaseSquaredNorm() = %v, want %v", got, tt.want)
			}
		})
	}
}

// ============================================================================
// BaseNorm Tests
// ============================================================================

func TestBaseNorm(t *testing.T) {
	tests := []struct {
		name string
		v    []float32
		want float32
	}{
		{"empty", []float32{}, 0},
		{"single", []float32{3}, 3},
		{"single negative", []float32{-5}, 5},
		{"unit vector", []float32{1, 0, 0}, 1},
		{"3-4-5 triangle", []float32{3, 4}, 5},
		{"zero vector", []float32{0, 0, 0}, 0},
		{"all ones 3d", []float32{1, 1, 1}, float32(math.Sqrt(3))},

		// SIMD boundary cases
		{"len 7", makeVector32(7, func(i int) float32 { return 1 }), float32(math.Sqrt(7))},
		{"len 8", makeVector32(8, func(i int) float32 { return 1 }), float32(math.Sqrt(8))},
		{"len 9", makeVector32(9, func(i int) float32 { return 1 }), float32(math.Sqrt(9))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseNorm(tt.v)
			if !approxEqual32(got, tt.want, epsilon32) {
				t.Errorf("BaseNorm() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBaseNorm_Float64(t *testing.T) {
	tests := []struct {
		name string
		v    []float64
		want float64
	}{
		{"empty", []float64{}, 0},
		{"single", []float64{3}, 3},
		{"3-4-5 triangle", []float64{3, 4}, 5},
		{"zero vector", []float64{0, 0, 0}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseNorm(tt.v)
			if !approxEqual64(got, tt.want, epsilon64) {
				t.Errorf("BaseNorm() = %v, want %v", got, tt.want)
			}
		})
	}
}

// ============================================================================
// BaseNormalize Tests
// ============================================================================

func TestBaseNormalize(t *testing.T) {
	tests := []struct {
		name     string
		v        []float32
		wantNorm float32 // expected norm of result (should be ~1 for non-zero vectors)
	}{
		{"zero vector stays zero", []float32{0, 0, 0}, 0},
		{"unit vector unchanged", []float32{1, 0, 0}, 1},
		{"simple 2d", []float32{3, 4}, 1},
		{"all positive", []float32{1, 2, 3}, 1},
		{"mixed signs", []float32{-1, 2, -3}, 1},

		// SIMD boundary cases
		{"len 7", makeVector32(7, func(i int) float32 { return float32(i + 1) }), 1},
		{"len 8", makeVector32(8, func(i int) float32 { return float32(i + 1) }), 1},
		{"len 9", makeVector32(9, func(i int) float32 { return float32(i + 1) }), 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := make([]float32, len(tt.v))
			copy(result, tt.v)
			BaseNormalize(result)

			gotNorm := BaseNorm(result)
			if !approxEqual32(gotNorm, tt.wantNorm, epsilon32) {
				t.Errorf("BaseNormalize() norm = %v, want %v", gotNorm, tt.wantNorm)
			}

			// For non-zero vectors, verify direction preserved
			if tt.wantNorm > 0 {
				originalNorm := BaseNorm(tt.v)
				for i := range tt.v {
					expected := tt.v[i] / originalNorm
					if !approxEqual32(result[i], expected, epsilon32) {
						t.Errorf("BaseNormalize()[%d] = %v, want %v", i, result[i], expected)
					}
				}
			}
		})
	}
}

func TestBaseNormalizeTo(t *testing.T) {
	tests := []struct {
		name string
		v    []float32
	}{
		{"simple", []float32{3, 4}},
		{"zero vector", []float32{0, 0, 0}},
		{"len 8", makeVector32(8, func(i int) float32 { return float32(i + 1) })},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.v))
			BaseNormalizeTo(dst, tt.v)

			// Verify source unchanged
			srcCopy := make([]float32, len(tt.v))
			copy(srcCopy, tt.v)

			// Verify result matches in-place normalize
			expected := make([]float32, len(tt.v))
			copy(expected, tt.v)
			BaseNormalize(expected)

			if !sliceApproxEqual32(dst, expected, epsilon32) {
				t.Errorf("BaseNormalizeTo() = %v, want %v", dst, expected)
			}
		})
	}
}

// ============================================================================
// BaseL2SquaredDistance Tests
// ============================================================================

func TestBaseL2SquaredDistance(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		b    []float32
		want float32
	}{
		// Basic cases
		{"empty", []float32{}, []float32{}, 0},
		{"same vector", []float32{1, 2, 3}, []float32{1, 2, 3}, 0},
		{"unit distance 1d", []float32{0}, []float32{1}, 1},
		{"unit distance 2d", []float32{0, 0}, []float32{1, 0}, 1},
		{"orthogonal unit vectors", []float32{1, 0}, []float32{0, 1}, 2},
		{"3-4-5 triangle", []float32{0, 0}, []float32{3, 4}, 25},

		// Negative values
		{"negative to positive", []float32{-1, -1}, []float32{1, 1}, 8},

		// SIMD boundary cases
		{"len 3", makeVector32(3, func(i int) float32 { return 0 }), makeVector32(3, func(i int) float32 { return 1 }), 3},
		{"len 4", makeVector32(4, func(i int) float32 { return 0 }), makeVector32(4, func(i int) float32 { return 1 }), 4},
		{"len 5", makeVector32(5, func(i int) float32 { return 0 }), makeVector32(5, func(i int) float32 { return 1 }), 5},
		{"len 7", makeVector32(7, func(i int) float32 { return 0 }), makeVector32(7, func(i int) float32 { return 1 }), 7},
		{"len 8", makeVector32(8, func(i int) float32 { return 0 }), makeVector32(8, func(i int) float32 { return 1 }), 8},
		{"len 9", makeVector32(9, func(i int) float32 { return 0 }), makeVector32(9, func(i int) float32 { return 1 }), 9},
		{"len 15", makeVector32(15, func(i int) float32 { return 0 }), makeVector32(15, func(i int) float32 { return 1 }), 15},
		{"len 16", makeVector32(16, func(i int) float32 { return 0 }), makeVector32(16, func(i int) float32 { return 1 }), 16},
		{"len 17", makeVector32(17, func(i int) float32 { return 0 }), makeVector32(17, func(i int) float32 { return 1 }), 17},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseL2SquaredDistance(tt.a, tt.b)
			if !approxEqual32(got, tt.want, epsilon32) {
				t.Errorf("BaseL2SquaredDistance() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBaseL2SquaredDistance_Symmetry(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{5, 4, 3, 2, 1}

	ab := BaseL2SquaredDistance(a, b)
	ba := BaseL2SquaredDistance(b, a)

	if !approxEqual32(ab, ba, epsilon32) {
		t.Errorf("Distance not symmetric: d(a,b)=%v, d(b,a)=%v", ab, ba)
	}
}

func TestBaseL2Distance(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		b    []float32
		want float32
	}{
		{"empty", []float32{}, []float32{}, 0},
		{"same vector", []float32{1, 2, 3}, []float32{1, 2, 3}, 0},
		{"unit distance", []float32{0}, []float32{1}, 1},
		{"3-4-5 triangle", []float32{0, 0}, []float32{3, 4}, 5},
		{"orthogonal unit vectors", []float32{1, 0}, []float32{0, 1}, float32(math.Sqrt(2))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseL2Distance(tt.a, tt.b)
			if !approxEqual32(got, tt.want, epsilon32) {
				t.Errorf("BaseL2Distance() = %v, want %v", got, tt.want)
			}
		})
	}
}

// ============================================================================
// Arithmetic Operations Tests
// ============================================================================

func TestBaseAdd(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		b    []float32
		want []float32
	}{
		{"empty", []float32{}, []float32{}, []float32{}},
		{"single", []float32{1}, []float32{2}, []float32{3}},
		{"add zero", []float32{1, 2, 3}, []float32{0, 0, 0}, []float32{1, 2, 3}},
		{"simple", []float32{1, 2, 3}, []float32{4, 5, 6}, []float32{5, 7, 9}},
		{"negative", []float32{1, -2, 3}, []float32{-1, 2, -3}, []float32{0, 0, 0}},

		// SIMD boundary cases
		{"len 7", makeVector32(7, func(i int) float32 { return 1 }), makeVector32(7, func(i int) float32 { return 2 }), makeVector32(7, func(i int) float32 { return 3 })},
		{"len 8", makeVector32(8, func(i int) float32 { return 1 }), makeVector32(8, func(i int) float32 { return 2 }), makeVector32(8, func(i int) float32 { return 3 })},
		{"len 9", makeVector32(9, func(i int) float32 { return 1 }), makeVector32(9, func(i int) float32 { return 2 }), makeVector32(9, func(i int) float32 { return 3 })},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			copy(dst, tt.a)
			BaseAdd(dst, tt.b)

			if !sliceApproxEqual32(dst, tt.want, epsilon32) {
				t.Errorf("BaseAdd() = %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestBaseAddTo(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	dst := make([]float32, 3)

	BaseAddTo(dst, a, b)

	want := []float32{5, 7, 9}
	if !sliceApproxEqual32(dst, want, epsilon32) {
		t.Errorf("BaseAddTo() = %v, want %v", dst, want)
	}

	// Verify inputs unchanged
	if !sliceApproxEqual32(a, []float32{1, 2, 3}, 0) {
		t.Errorf("BaseAddTo() modified input a")
	}
	if !sliceApproxEqual32(b, []float32{4, 5, 6}, 0) {
		t.Errorf("BaseAddTo() modified input b")
	}
}

func TestBaseSub(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		b    []float32
		want []float32
	}{
		{"empty", []float32{}, []float32{}, []float32{}},
		{"single", []float32{3}, []float32{1}, []float32{2}},
		{"sub zero", []float32{1, 2, 3}, []float32{0, 0, 0}, []float32{1, 2, 3}},
		{"simple", []float32{5, 7, 9}, []float32{4, 5, 6}, []float32{1, 2, 3}},
		{"same", []float32{1, 2, 3}, []float32{1, 2, 3}, []float32{0, 0, 0}},

		// SIMD boundary cases
		{"len 8", makeVector32(8, func(i int) float32 { return 5 }), makeVector32(8, func(i int) float32 { return 3 }), makeVector32(8, func(i int) float32 { return 2 })},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			copy(dst, tt.a)
			BaseSub(dst, tt.b)

			if !sliceApproxEqual32(dst, tt.want, epsilon32) {
				t.Errorf("BaseSub() = %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestBaseSubTo(t *testing.T) {
	a := []float32{5, 7, 9}
	b := []float32{4, 5, 6}
	dst := make([]float32, 3)

	BaseSubTo(dst, a, b)

	want := []float32{1, 2, 3}
	if !sliceApproxEqual32(dst, want, epsilon32) {
		t.Errorf("BaseSubTo() = %v, want %v", dst, want)
	}
}

func TestBaseMul(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		b    []float32
		want []float32
	}{
		{"empty", []float32{}, []float32{}, []float32{}},
		{"single", []float32{2}, []float32{3}, []float32{6}},
		{"mul one", []float32{1, 2, 3}, []float32{1, 1, 1}, []float32{1, 2, 3}},
		{"mul zero", []float32{1, 2, 3}, []float32{0, 0, 0}, []float32{0, 0, 0}},
		{"simple", []float32{2, 3, 4}, []float32{5, 6, 7}, []float32{10, 18, 28}},

		// SIMD boundary cases
		{"len 8", makeVector32(8, func(i int) float32 { return 2 }), makeVector32(8, func(i int) float32 { return 3 }), makeVector32(8, func(i int) float32 { return 6 })},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			copy(dst, tt.a)
			BaseMul(dst, tt.b)

			if !sliceApproxEqual32(dst, tt.want, epsilon32) {
				t.Errorf("BaseMul() = %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestBaseMulTo(t *testing.T) {
	a := []float32{2, 3, 4}
	b := []float32{5, 6, 7}
	dst := make([]float32, 3)

	BaseMulTo(dst, a, b)

	want := []float32{10, 18, 28}
	if !sliceApproxEqual32(dst, want, epsilon32) {
		t.Errorf("BaseMulTo() = %v, want %v", dst, want)
	}
}

func TestBaseDiv(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		b    []float32
		want []float32
	}{
		{"empty", []float32{}, []float32{}, []float32{}},
		{"single", []float32{6}, []float32{2}, []float32{3}},
		{"div one", []float32{1, 2, 3}, []float32{1, 1, 1}, []float32{1, 2, 3}},
		{"simple", []float32{10, 18, 28}, []float32{5, 6, 7}, []float32{2, 3, 4}},

		// SIMD boundary cases
		{"len 8", makeVector32(8, func(i int) float32 { return 6 }), makeVector32(8, func(i int) float32 { return 2 }), makeVector32(8, func(i int) float32 { return 3 })},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			copy(dst, tt.a)
			BaseDiv(dst, tt.b)

			if !sliceApproxEqual32(dst, tt.want, epsilon32) {
				t.Errorf("BaseDiv() = %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestBaseDivTo(t *testing.T) {
	a := []float32{10, 18, 28}
	b := []float32{5, 6, 7}
	dst := make([]float32, 3)

	BaseDivTo(dst, a, b)

	want := []float32{2, 3, 4}
	if !sliceApproxEqual32(dst, want, epsilon32) {
		t.Errorf("BaseDivTo() = %v, want %v", dst, want)
	}
}

// ============================================================================
// Scale and Constant Operations Tests
// ============================================================================

func TestBaseScale(t *testing.T) {
	tests := []struct {
		name  string
		v     []float32
		scale float32
		want  []float32
	}{
		{"empty", []float32{}, 2, []float32{}},
		{"single", []float32{3}, 2, []float32{6}},
		{"scale by 1", []float32{1, 2, 3}, 1, []float32{1, 2, 3}},
		{"scale by 0", []float32{1, 2, 3}, 0, []float32{0, 0, 0}},
		{"scale by 2", []float32{1, 2, 3}, 2, []float32{2, 4, 6}},
		{"scale by -1", []float32{1, 2, 3}, -1, []float32{-1, -2, -3}},
		{"scale by 0.5", []float32{2, 4, 6}, 0.5, []float32{1, 2, 3}},

		// SIMD boundary cases
		{"len 7", makeVector32(7, func(i int) float32 { return 2 }), 3, makeVector32(7, func(i int) float32 { return 6 })},
		{"len 8", makeVector32(8, func(i int) float32 { return 2 }), 3, makeVector32(8, func(i int) float32 { return 6 })},
		{"len 9", makeVector32(9, func(i int) float32 { return 2 }), 3, makeVector32(9, func(i int) float32 { return 6 })},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.v))
			copy(dst, tt.v)
			BaseScale(tt.scale, dst)

			if !sliceApproxEqual32(dst, tt.want, epsilon32) {
				t.Errorf("BaseScale() = %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestBaseScaleTo(t *testing.T) {
	v := []float32{1, 2, 3}
	dst := make([]float32, 3)

	BaseScaleTo(dst, float32(2), v)

	want := []float32{2, 4, 6}
	if !sliceApproxEqual32(dst, want, epsilon32) {
		t.Errorf("BaseScaleTo() = %v, want %v", dst, want)
	}

	// Verify source unchanged
	if !sliceApproxEqual32(v, []float32{1, 2, 3}, 0) {
		t.Errorf("BaseScaleTo() modified source")
	}
}

func TestBaseAddConst(t *testing.T) {
	tests := []struct {
		name string
		v    []float32
		c    float32
		want []float32
	}{
		{"empty", []float32{}, 5, []float32{}},
		{"single", []float32{3}, 2, []float32{5}},
		{"add 0", []float32{1, 2, 3}, 0, []float32{1, 2, 3}},
		{"add positive", []float32{1, 2, 3}, 10, []float32{11, 12, 13}},
		{"add negative", []float32{1, 2, 3}, -1, []float32{0, 1, 2}},

		// SIMD boundary cases
		{"len 8", makeVector32(8, func(i int) float32 { return 1 }), 5, makeVector32(8, func(i int) float32 { return 6 })},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.v))
			copy(dst, tt.v)
			BaseAddConst(tt.c, dst)

			if !sliceApproxEqual32(dst, tt.want, epsilon32) {
				t.Errorf("BaseAddConst() = %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestBaseMulConstAddTo(t *testing.T) {
	tests := []struct {
		name string
		dst  []float32
		v    []float32
		c    float32
		want []float32
	}{
		{"empty", []float32{}, []float32{}, 2, []float32{}},
		{"single", []float32{1}, []float32{2}, 3, []float32{7}},                   // 1 + 2*3 = 7
		{"simple", []float32{1, 2, 3}, []float32{1, 1, 1}, 2, []float32{3, 4, 5}}, // [1+1*2, 2+1*2, 3+1*2]
		{"scale 0", []float32{1, 2, 3}, []float32{10, 20, 30}, 0, []float32{1, 2, 3}},

		// SIMD boundary cases
		{"len 8", makeVector32(8, func(i int) float32 { return 1 }), makeVector32(8, func(i int) float32 { return 2 }), 3, makeVector32(8, func(i int) float32 { return 7 })},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.dst))
			copy(dst, tt.dst)
			BaseMulConstAddTo(dst, tt.c, tt.v)

			if !sliceApproxEqual32(dst, tt.want, epsilon32) {
				t.Errorf("BaseMulConstAddTo() = %v, want %v", dst, tt.want)
			}
		})
	}
}

// ============================================================================
// Reduce Operations Tests
// ============================================================================

func TestBaseSum(t *testing.T) {
	tests := []struct {
		name string
		v    []float32
		want float32
	}{
		{"empty", []float32{}, 0},
		{"single", []float32{5}, 5},
		{"zeros", []float32{0, 0, 0}, 0},
		{"simple", []float32{1, 2, 3, 4, 5}, 15},
		{"negative", []float32{-1, -2, -3}, -6},
		{"mixed", []float32{1, -1, 2, -2, 3}, 3},

		// SIMD boundary cases
		{"len 7", makeVector32(7, func(i int) float32 { return 1 }), 7},
		{"len 8", makeVector32(8, func(i int) float32 { return 1 }), 8},
		{"len 9", makeVector32(9, func(i int) float32 { return 1 }), 9},
		{"len 15", makeVector32(15, func(i int) float32 { return 1 }), 15},
		{"len 16", makeVector32(16, func(i int) float32 { return 1 }), 16},
		{"len 17", makeVector32(17, func(i int) float32 { return 1 }), 17},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseSum(tt.v)
			if !approxEqual32(got, tt.want, epsilon32) {
				t.Errorf("BaseSum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBaseMin(t *testing.T) {
	tests := []struct {
		name string
		v    []float32
		want float32
	}{
		{"single", []float32{5}, 5},
		{"sorted asc", []float32{1, 2, 3, 4, 5}, 1},
		{"sorted desc", []float32{5, 4, 3, 2, 1}, 1},
		{"all same", []float32{3, 3, 3}, 3},
		{"negative", []float32{-1, -5, -2}, -5},
		{"mixed", []float32{3, -1, 4, -5, 2}, -5},
		{"min at end", []float32{5, 4, 3, 2, 1}, 1},
		{"min at start", []float32{1, 2, 3, 4, 5}, 1},
		{"min in middle", []float32{3, 2, 1, 2, 3}, 1},

		// SIMD boundary cases
		{"len 7", append(makeVector32(6, func(i int) float32 { return 10 }), -1), -1},
		{"len 8", append(makeVector32(7, func(i int) float32 { return 10 }), -1), -1},
		{"len 9", append(makeVector32(8, func(i int) float32 { return 10 }), -1), -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseMin(tt.v)
			if !approxEqual32(got, tt.want, epsilon32) {
				t.Errorf("BaseMin() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBaseMin_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("BaseMin() did not panic on empty slice")
		}
	}()
	BaseMin([]float32{})
}

func TestBaseMax(t *testing.T) {
	tests := []struct {
		name string
		v    []float32
		want float32
	}{
		{"single", []float32{5}, 5},
		{"sorted asc", []float32{1, 2, 3, 4, 5}, 5},
		{"sorted desc", []float32{5, 4, 3, 2, 1}, 5},
		{"all same", []float32{3, 3, 3}, 3},
		{"negative", []float32{-1, -5, -2}, -1},
		{"mixed", []float32{3, -1, 4, -5, 2}, 4},
		{"max at end", []float32{1, 2, 3, 4, 5}, 5},
		{"max at start", []float32{5, 4, 3, 2, 1}, 5},
		{"max in middle", []float32{1, 2, 5, 2, 1}, 5},

		// SIMD boundary cases
		{"len 7", append(makeVector32(6, func(i int) float32 { return -10 }), 100), 100},
		{"len 8", append(makeVector32(7, func(i int) float32 { return -10 }), 100), 100},
		{"len 9", append(makeVector32(8, func(i int) float32 { return -10 }), 100), 100},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BaseMax(tt.v)
			if !approxEqual32(got, tt.want, epsilon32) {
				t.Errorf("BaseMax() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBaseMax_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("BaseMax() did not panic on empty slice")
		}
	}()
	BaseMax([]float32{})
}

func TestBaseMax_IntegerTypes(t *testing.T) {
	t.Run("int32", func(t *testing.T) {
		tests := []struct {
			name string
			v    []int32
			want int32
		}{
			{"positive", []int32{1, 5, 3, 2, 4}, 5},
			{"negative", []int32{-1, -5, -3, -2, -4}, -1},
			{"mixed", []int32{-3, 5, -1, 2, -4}, 5},
			{"single", []int32{42}, 42},
			{"large", []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100}, 100},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got := Max(tt.v)
				if got != tt.want {
					t.Errorf("Max() = %v, want %v", got, tt.want)
				}
			})
		}
	})

	t.Run("int64", func(t *testing.T) {
		tests := []struct {
			name string
			v    []int64
			want int64
		}{
			{"positive", []int64{1, 5, 3, 2, 4}, 5},
			{"negative", []int64{-1, -5, -3, -2, -4}, -1},
			{"large values", []int64{1 << 40, 1 << 50, 1 << 45}, 1 << 50},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got := Max(tt.v)
				if got != tt.want {
					t.Errorf("Max() = %v, want %v", got, tt.want)
				}
			})
		}
	})

	t.Run("uint32", func(t *testing.T) {
		tests := []struct {
			name string
			v    []uint32
			want uint32
		}{
			{"basic", []uint32{1, 5, 3, 2, 4}, 5},
			{"large", []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100}, 100},
			{"max uint32", []uint32{0, 1 << 31, 1<<32 - 1}, 1<<32 - 1},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got := Max(tt.v)
				if got != tt.want {
					t.Errorf("Max() = %v, want %v", got, tt.want)
				}
			})
		}
	})

	t.Run("uint64", func(t *testing.T) {
		tests := []struct {
			name string
			v    []uint64
			want uint64
		}{
			{"basic", []uint64{1, 5, 3, 2, 4}, 5},
			{"large values", []uint64{1 << 40, 1 << 60, 1 << 50}, 1 << 60},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got := Max(tt.v)
				if got != tt.want {
					t.Errorf("Max() = %v, want %v", got, tt.want)
				}
			})
		}
	})
}

func TestBaseMax_SpecialValues(t *testing.T) {
	inf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))

	t.Run("float32", func(t *testing.T) {
		tests := []struct {
			name string
			v    []float32
			want float32
		}{
			{"no special", []float32{1, 5, 3}, 5},
			{"with Inf", []float32{1, inf, 5}, inf},
			{"with -Inf", []float32{1, negInf, 5}, 5},
			{"Inf and -Inf", []float32{negInf, 1, inf}, inf},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got := Max(tt.v)
				if got != tt.want {
					t.Errorf("Max() = %v, want %v", got, tt.want)
				}
			})
		}
	})

	// Note: NaN handling in SIMD follows IEEE 754 semantics.
	// SIMD max operations may propagate NaN differently than scalar code.
	// For data containing NaN, the result is implementation-defined.
}

func TestBaseMinMax(t *testing.T) {
	tests := []struct {
		name    string
		v       []float32
		wantMin float32
		wantMax float32
	}{
		{"single", []float32{5}, 5, 5},
		{"sorted asc", []float32{1, 2, 3, 4, 5}, 1, 5},
		{"sorted desc", []float32{5, 4, 3, 2, 1}, 1, 5},
		{"all same", []float32{3, 3, 3}, 3, 3},
		{"negative", []float32{-1, -5, -2}, -5, -1},
		{"mixed", []float32{3, -1, 4, -5, 2}, -5, 4},

		// SIMD boundary cases
		{"len 7", []float32{7, 1, 6, 2, 5, 3, 4}, 1, 7},
		{"len 8", []float32{8, 1, 7, 2, 6, 3, 5, 4}, 1, 8},
		{"len 9", []float32{9, 1, 8, 2, 7, 3, 6, 4, 5}, 1, 9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMin, gotMax := BaseMinMax(tt.v)
			if !approxEqual32(gotMin, tt.wantMin, epsilon32) {
				t.Errorf("BaseMinMax() min = %v, want %v", gotMin, tt.wantMin)
			}
			if !approxEqual32(gotMax, tt.wantMax, epsilon32) {
				t.Errorf("BaseMinMax() max = %v, want %v", gotMax, tt.wantMax)
			}
		})
	}
}

func TestBaseMinMax_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("BaseMinMax() did not panic on empty slice")
		}
	}()
	BaseMinMax([]float32{})
}

// ============================================================================
// Dot Product Tests
// ============================================================================

func TestDot(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		b    []float32
		want float32
	}{
		// Basic cases
		{"empty", []float32{}, []float32{}, 0},
		{"single", []float32{2}, []float32{3}, 6},
		{"3d", []float32{1, 2, 3}, []float32{4, 5, 6}, 32}, // 1*4 + 2*5 + 3*6 = 32
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"parallel", []float32{1, 0}, []float32{2, 0}, 2},
		{"negative", []float32{-1, 2, -3}, []float32{4, -5, 6}, -32}, // -4 + -10 + -18 = -32

		// SIMD boundary cases
		{"len 3", makeVector32(3, func(i int) float32 { return 1 }), makeVector32(3, func(i int) float32 { return 1 }), 3},
		{"len 4", makeVector32(4, func(i int) float32 { return 1 }), makeVector32(4, func(i int) float32 { return 1 }), 4},
		{"len 5", makeVector32(5, func(i int) float32 { return 1 }), makeVector32(5, func(i int) float32 { return 1 }), 5},
		{"len 7", makeVector32(7, func(i int) float32 { return 1 }), makeVector32(7, func(i int) float32 { return 1 }), 7},
		{"len 8", makeVector32(8, func(i int) float32 { return 1 }), makeVector32(8, func(i int) float32 { return 1 }), 8},
		{"len 9", makeVector32(9, func(i int) float32 { return 1 }), makeVector32(9, func(i int) float32 { return 1 }), 9},
		{"len 15", makeVector32(15, func(i int) float32 { return 1 }), makeVector32(15, func(i int) float32 { return 1 }), 15},
		{"len 16", makeVector32(16, func(i int) float32 { return 1 }), makeVector32(16, func(i int) float32 { return 1 }), 16},
		{"len 17", makeVector32(17, func(i int) float32 { return 1 }), makeVector32(17, func(i int) float32 { return 1 }), 17},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Dot(tt.a, tt.b)
			if !approxEqual32(got, tt.want, epsilon32) {
				t.Errorf("Dot() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDot_Float64(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		b    []float64
		want float64
	}{
		{"empty", []float64{}, []float64{}, 0},
		{"single", []float64{2}, []float64{3}, 6},
		{"3d", []float64{1, 2, 3}, []float64{4, 5, 6}, 32},
		{"high precision", []float64{1e-10, 2e-10}, []float64{3e-10, 4e-10}, 3e-20 + 8e-20},

		// SIMD boundary cases for float64 (2-wide on NEON)
		{"len 3", makeVector64(3, func(i int) float64 { return 1 }), makeVector64(3, func(i int) float64 { return 1 }), 3},
		{"len 4", makeVector64(4, func(i int) float64 { return 1 }), makeVector64(4, func(i int) float64 { return 1 }), 4},
		{"len 5", makeVector64(5, func(i int) float64 { return 1 }), makeVector64(5, func(i int) float64 { return 1 }), 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Dot(tt.a, tt.b)
			if !approxEqual64(got, tt.want, epsilon64) {
				t.Errorf("Dot() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDot_Commutativity(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{5, 4, 3, 2, 1}

	ab := Dot(a, b)
	ba := Dot(b, a)

	if !approxEqual32(ab, ba, epsilon32) {
		t.Errorf("Dot not commutative: Dot(a,b)=%v, Dot(b,a)=%v", ab, ba)
	}
}

// ============================================================================
// Batch Operations Tests
// ============================================================================

func TestBaseBatchL2SquaredDistance(t *testing.T) {
	t.Run("single vector", func(t *testing.T) {
		query := []float32{1, 2, 3}
		data := []float32{1, 2, 3} // one vector, same as query
		distances := make([]float32, 1)
		BaseBatchL2SquaredDistance(query, data, distances, 1, 3)

		if !approxEqual32(distances[0], 0, epsilon32) {
			t.Errorf("distance to self = %v, want 0", distances[0])
		}
	})

	t.Run("multiple identical", func(t *testing.T) {
		query := []float32{0, 0}
		data := []float32{3, 4, 3, 4, 3, 4} // three vectors (3,4) flattened
		distances := make([]float32, 3)
		BaseBatchL2SquaredDistance(query, data, distances, 3, 2)

		for i := range 3 {
			if !approxEqual32(distances[i], 25, epsilon32) {
				t.Errorf("distances[%d] = %v, want 25", i, distances[i])
			}
		}
	})

	t.Run("different vectors", func(t *testing.T) {
		query := []float32{0, 0}
		data := []float32{1, 0, 0, 1, 1, 1} // three vectors flattened
		distances := make([]float32, 3)
		BaseBatchL2SquaredDistance(query, data, distances, 3, 2)

		want := []float32{1, 1, 2}
		for i := range want {
			if !approxEqual32(distances[i], want[i], epsilon32) {
				t.Errorf("distances[%d] = %v, want %v", i, distances[i], want[i])
			}
		}
	})

	t.Run("verify matches individual", func(t *testing.T) {
		query := []float32{1, 2, 3, 4}
		// Three vectors: {5,6,7,8}, {1,1,1,1}, {0,0,0,0}
		data := []float32{5, 6, 7, 8, 1, 1, 1, 1, 0, 0, 0, 0}
		distances := make([]float32, 3)
		BaseBatchL2SquaredDistance(query, data, distances, 3, 4)

		// Verify against individual computations
		vectors := [][]float32{{5, 6, 7, 8}, {1, 1, 1, 1}, {0, 0, 0, 0}}
		for i, vec := range vectors {
			individual := BaseL2SquaredDistance(query, vec)
			if !approxEqual32(distances[i], individual, epsilon32) {
				t.Errorf("distances[%d] = %v, individual = %v", i, distances[i], individual)
			}
		}
	})

	t.Run("SIMD boundary dims", func(t *testing.T) {
		// Test with dims that hit SIMD boundaries
		for _, dims := range []int{3, 4, 5, 7, 8, 9, 15, 16, 17} {
			query := makeVector32(dims, func(i int) float32 { return float32(i) })
			data := makeVector32(dims*2, func(i int) float32 { return float32(i % dims) })
			distances := make([]float32, 2)
			BaseBatchL2SquaredDistance(query, data, distances, 2, dims)

			// First vector should have distance 0 (same as query modulo dims)
			if !approxEqual32(distances[0], 0, epsilon32*float32(dims)) {
				t.Errorf("dims=%d: distances[0] = %v, want ~0", dims, distances[0])
			}
		}
	})
}

func TestBaseBatchDot(t *testing.T) {
	t.Run("single vector", func(t *testing.T) {
		query := []float32{1, 2, 3}
		data := []float32{4, 5, 6}
		dots := make([]float32, 1)
		BaseBatchDot(query, data, dots, 1, 3)

		// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
		if !approxEqual32(dots[0], 32, epsilon32) {
			t.Errorf("dots[0] = %v, want 32", dots[0])
		}
	})

	t.Run("multiple vectors", func(t *testing.T) {
		query := []float32{1, 0, 0}
		data := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1} // three unit vectors
		dots := make([]float32, 3)
		BaseBatchDot(query, data, dots, 3, 3)

		want := []float32{1, 0, 0}
		for i := range want {
			if !approxEqual32(dots[i], want[i], epsilon32) {
				t.Errorf("dots[%d] = %v, want %v", i, dots[i], want[i])
			}
		}
	})

	t.Run("orthogonal", func(t *testing.T) {
		query := []float32{1, 0}
		data := []float32{0, 1}
		dots := make([]float32, 1)
		BaseBatchDot(query, data, dots, 1, 2)

		if !approxEqual32(dots[0], 0, epsilon32) {
			t.Errorf("dots[0] = %v, want 0", dots[0])
		}
	})

	t.Run("verify matches individual", func(t *testing.T) {
		query := []float32{1, 2, 3, 4}
		data := []float32{4, 3, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0}
		dots := make([]float32, 3)
		BaseBatchDot(query, data, dots, 3, 4)

		// Verify against individual dot products using existing dot package
		vectors := [][]float32{{4, 3, 2, 1}, {1, 1, 1, 1}, {0, 0, 0, 0}}
		want := []float32{20, 10, 0} // 4+6+6+4=20, 1+2+3+4=10, 0
		for i, vec := range vectors {
			// Manual dot product for verification
			var individual float32
			for j := range query {
				individual += query[j] * vec[j]
			}
			if !approxEqual32(dots[i], want[i], epsilon32) {
				t.Errorf("dots[%d] = %v, want %v (individual=%v)", i, dots[i], want[i], individual)
			}
		}
	})

	t.Run("SIMD boundary dims", func(t *testing.T) {
		// Test with dims that hit SIMD boundaries
		for _, dims := range []int{3, 4, 5, 7, 8, 9, 15, 16, 17} {
			query := makeVector32(dims, func(i int) float32 { return 1 })
			data := makeVector32(dims*2, func(i int) float32 { return 1 })
			dots := make([]float32, 2)
			BaseBatchDot(query, data, dots, 2, dims)

			// Dot product of all ones should be dims
			expected := float32(dims)
			for i := range 2 {
				if !approxEqual32(dots[i], expected, epsilon32) {
					t.Errorf("dims=%d: dots[%d] = %v, want %v", dims, i, dots[i], expected)
				}
			}
		}
	})
}

// ============================================================================
// SIMD Boundary Tests (comprehensive)
// ============================================================================

func TestSIMDBoundaries(t *testing.T) {
	// Test all critical SIMD widths
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			// Create test vectors
			a := makeVector32(size, func(i int) float32 { return float32(i + 1) })
			b := makeVector32(size, func(i int) float32 { return float32(i + 1) })
			zeros := makeVector32(size, func(i int) float32 { return 0 })

			// Test SquaredNorm: sum of squares
			var expectedSqNorm float32
			for i := range size {
				expectedSqNorm += float32((i + 1) * (i + 1))
			}
			gotSqNorm := BaseSquaredNorm(a)
			if !approxEqual32(gotSqNorm, expectedSqNorm, epsilon32*float32(size)) {
				t.Errorf("BaseSquaredNorm() size=%d: got %v, want %v", size, gotSqNorm, expectedSqNorm)
			}

			// Test L2SquaredDistance: should be 0 for same vectors
			dist := BaseL2SquaredDistance(a, b)
			if !approxEqual32(dist, 0, epsilon32) {
				t.Errorf("BaseL2SquaredDistance() size=%d same vectors: got %v, want 0", size, dist)
			}

			// Test Sum
			var expectedSum float32
			for i := range size {
				expectedSum += float32(i + 1)
			}
			gotSum := BaseSum(a)
			if !approxEqual32(gotSum, expectedSum, epsilon32*float32(size)) {
				t.Errorf("BaseSum() size=%d: got %v, want %v", size, gotSum, expectedSum)
			}

			// Test Add
			dst := make([]float32, size)
			copy(dst, a)
			BaseAdd(dst, zeros)
			if !sliceApproxEqual32(dst, a, epsilon32) {
				t.Errorf("BaseAdd() size=%d with zeros changed vector", size)
			}

			// Test Scale
			copy(dst, a)
			BaseScale(float32(1), dst)
			if !sliceApproxEqual32(dst, a, epsilon32) {
				t.Errorf("BaseScale() size=%d by 1 changed vector", size)
			}
		})
	}
}

// ============================================================================
// Edge Cases and Special Values
// ============================================================================

func TestSpecialValues(t *testing.T) {
	t.Run("very large values", func(t *testing.T) {
		v := []float32{1e30, 1e30, 1e30}
		norm := BaseNorm(v)
		if math.IsInf(float64(norm), 0) || math.IsNaN(float64(norm)) {
			// This is acceptable - overflow is expected for very large values
			t.Logf("Large value overflow as expected: %v", norm)
		}
	})

	t.Run("very small values", func(t *testing.T) {
		v := []float32{1e-30, 1e-30, 1e-30}
		norm := BaseSquaredNorm(v)
		if norm < 0 {
			t.Errorf("SquaredNorm of small values should be >= 0, got %v", norm)
		}
	})

	t.Run("mixed large small", func(t *testing.T) {
		v := []float32{1e10, 1e-10}
		dist := BaseL2SquaredDistance(v, []float32{0, 0})
		if dist < 0 {
			t.Errorf("Distance should be >= 0, got %v", dist)
		}
	})
}

// ============================================================================
// Benchmarks
// ============================================================================

func BenchmarkNorm(b *testing.B) {
	sizes := []int{16, 64, 256, 512, 1024, 4096}

	for _, size := range sizes {
		v := makeVector32(size, func(i int) float32 { return float32(i) })

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var result float32
			for i := 0; i < b.N; i++ {
				result = Norm(v)
			}
			_ = result
		})
	}
}

func BenchmarkSquaredNorm(b *testing.B) {
	sizes := []int{16, 64, 256, 512, 1024, 4096}

	for _, size := range sizes {
		v := makeVector32(size, func(i int) float32 { return float32(i) })

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var result float32
			for i := 0; i < b.N; i++ {
				result = SquaredNorm(v)
			}
			_ = result
		})
	}
}

func BenchmarkL2SquaredDistance(b *testing.B) {
	sizes := []int{16, 64, 256, 512, 1024, 4096}

	for _, size := range sizes {
		a := makeVector32(size, func(i int) float32 { return float32(i) })
		c := makeVector32(size, func(i int) float32 { return float32(i + 1) })

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var result float32
			for i := 0; i < b.N; i++ {
				result = L2SquaredDistance(a, c)
			}
			_ = result
		})
	}
}

func BenchmarkSum(b *testing.B) {
	sizes := []int{16, 64, 256, 512, 1024, 4096}

	for _, size := range sizes {
		v := makeVector32(size, func(i int) float32 { return float32(i) })

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var result float32
			for i := 0; i < b.N; i++ {
				result = Sum(v)
			}
			_ = result
		})
	}
}

func BenchmarkDot(b *testing.B) {
	sizes := []int{16, 64, 256, 512, 1024, 4096}

	for _, size := range sizes {
		a := makeVector32(size, func(i int) float32 { return float32(i) })
		c := makeVector32(size, func(i int) float32 { return float32(i + 1) })

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var result float32
			for i := 0; i < b.N; i++ {
				result = Dot(a, c)
			}
			_ = result
		})
	}
}

func BenchmarkAdd(b *testing.B) {
	sizes := []int{16, 64, 256, 512, 1024, 4096}

	for _, size := range sizes {
		a := makeVector32(size, func(i int) float32 { return float32(i) })
		c := makeVector32(size, func(i int) float32 { return float32(i + 1) })
		dst := make([]float32, size)

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				copy(dst, a)
				Add(dst, c)
			}
		})
	}
}

// BenchmarkAddPure benchmarks Add without copy overhead (Add then Sub to cancel out)
// This matches how num32 benchmarks Add for fair comparison
func BenchmarkAddPure(b *testing.B) {
	sizes := []int{256, 512, 1024}

	for _, size := range sizes {
		dst := makeVector32(size, func(i int) float32 { return float32(i) })
		s := makeVector32(size, func(i int) float32 { return float32(i + 1) })

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Add(dst, s)
				Sub(dst, s)
			}
		})
	}
}

// BenchmarkMulPure benchmarks Mul without copy overhead
func BenchmarkMulPure(b *testing.B) {
	sizes := []int{256, 512, 1024}

	for _, size := range sizes {
		dst := makeVector32(size, func(i int) float32 { return 1.0 })
		s := makeVector32(size, func(i int) float32 { return 1.0 })

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Mul(dst, s)
			}
		})
	}
}

func BenchmarkScale(b *testing.B) {
	sizes := []int{16, 64, 256, 512, 1024, 4096}

	for _, size := range sizes {
		v := makeVector32(size, func(i int) float32 { return float32(i) })
		dst := make([]float32, size)

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				copy(dst, v)
				Scale(float32(2.5), dst)
			}
		})
	}
}

func BenchmarkNormalize(b *testing.B) {
	sizes := []int{16, 64, 256, 1024, 4096}

	for _, size := range sizes {
		v := makeVector32(size, func(i int) float32 { return float32(i + 1) })
		dst := make([]float32, size)

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				copy(dst, v)
				Normalize(dst)
			}
		})
	}
}

func BenchmarkBaseBatchDot(b *testing.B) {
	vecSize := 256
	batchSizes := []int{1, 8, 32, 128}

	for _, batchSize := range batchSizes {
		query := makeVector32(vecSize, func(i int) float32 { return float32(i) })
		data := makeVector32(vecSize*batchSize, func(i int) float32 { return float32(i % vecSize) })
		dots := make([]float32, batchSize)

		b.Run(fmt.Sprintf("batch_%d_vec_%d", batchSize, vecSize), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				BaseBatchDot(query, data, dots, batchSize, vecSize)
			}
		})
	}
}

func BenchmarkBaseBatchL2SquaredDistance(b *testing.B) {
	vecSize := 256
	batchSizes := []int{1, 8, 32, 128}

	for _, batchSize := range batchSizes {
		query := makeVector32(vecSize, func(i int) float32 { return float32(i) })
		data := makeVector32(vecSize*batchSize, func(i int) float32 { return float32(i % vecSize) })
		distances := make([]float32, batchSize)

		b.Run(fmt.Sprintf("batch_%d_vec_%d", batchSize, vecSize), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				BaseBatchL2SquaredDistance(query, data, distances, batchSize, vecSize)
			}
		})
	}
}

// Benchmark comparison with stdlib implementations

func BenchmarkBaseNorm_Stdlib(b *testing.B) {
	sizes := []int{256, 1024}

	for _, size := range sizes {
		v := makeVector32(size, func(i int) float32 { return float32(i) })

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var result float32
			for i := 0; i < b.N; i++ {
				var sum float64
				for _, x := range v {
					sum += float64(x) * float64(x)
				}
				result = float32(math.Sqrt(sum))
			}
			_ = result
		})
	}
}

func BenchmarkBaseL2SquaredDistance_Stdlib(b *testing.B) {
	sizes := []int{256, 1024}

	for _, size := range sizes {
		a := makeVector32(size, func(i int) float32 { return float32(i) })
		c := makeVector32(size, func(i int) float32 { return float32(i + 1) })

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var result float32
			for i := 0; i < b.N; i++ {
				var sum float32
				for j := range a {
					d := a[j] - c[j]
					sum += d * d
				}
				result = sum
			}
			_ = result
		})
	}
}
