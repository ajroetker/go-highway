package matvec

import (
	"math"
	"testing"
)

func TestMatVec(t *testing.T) {
	tests := []struct {
		name string
		m    []float32
		rows int
		cols int
		v    []float32
		want []float32
	}{
		{
			name: "2x3 matrix",
			m: []float32{
				1, 2, 3,
				4, 5, 6,
			},
			rows: 2,
			cols: 3,
			v:    []float32{1, 0, 1},
			want: []float32{4, 10}, // [1*1+2*0+3*1, 4*1+5*0+6*1]
		},
		{
			name: "3x4 matrix",
			m: []float32{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 0, 1, 2,
			},
			rows: 3,
			cols: 4,
			v:    []float32{1, 2, 3, 4},
			want: []float32{30, 70, 20},
		},
		{
			name: "identity matrix 3x3",
			m: []float32{
				1, 0, 0,
				0, 1, 0,
				0, 0, 1,
			},
			rows: 3,
			cols: 3,
			v:    []float32{5, 7, 9},
			want: []float32{5, 7, 9},
		},
		{
			name: "single row",
			m:    []float32{1, 2, 3, 4},
			rows: 1,
			cols: 4,
			v:    []float32{1, 1, 1, 1},
			want: []float32{10},
		},
		{
			name: "single column",
			m:    []float32{1, 2, 3, 4},
			rows: 4,
			cols: 1,
			v:    []float32{2},
			want: []float32{2, 4, 6, 8},
		},
		{
			name: "zeros",
			m: []float32{
				0, 0,
				0, 0,
			},
			rows: 2,
			cols: 2,
			v:    []float32{5, 10},
			want: []float32{0, 0},
		},
		{
			name: "large matrix (16x16)",
			m:    make16x16Identity(),
			rows: 16,
			cols: 16,
			v:    makeRange(16),
			want: makeRange(16),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := make([]float32, tt.rows)
			MatVec(tt.m, tt.rows, tt.cols, tt.v, result)

			for i := range result {
				if math.Abs(float64(result[i]-tt.want[i])) > 1e-5 {
					t.Errorf("MatVec()[%d] = %v, want %v", i, result[i], tt.want[i])
				}
			}
		})
	}
}

func TestMatVecFloat64(t *testing.T) {
	tests := []struct {
		name string
		m    []float64
		rows int
		cols int
		v    []float64
		want []float64
	}{
		{
			name: "2x3 matrix",
			m: []float64{
				1, 2, 3,
				4, 5, 6,
			},
			rows: 2,
			cols: 3,
			v:    []float64{1, 0, 1},
			want: []float64{4, 10},
		},
		{
			name: "high precision",
			m: []float64{
				0.1, 0.2,
				0.3, 0.4,
			},
			rows: 2,
			cols: 2,
			v:    []float64{0.5, 0.6},
			want: []float64{0.17, 0.39}, // [0.1*0.5+0.2*0.6, 0.3*0.5+0.4*0.6]
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := make([]float64, tt.rows)
			MatVecFloat64(tt.m, tt.rows, tt.cols, tt.v, result)

			for i := range result {
				if math.Abs(result[i]-tt.want[i]) > 1e-10 {
					t.Errorf("MatVecFloat64()[%d] = %v, want %v", i, result[i], tt.want[i])
				}
			}
		})
	}
}

func TestMatVecPanics(t *testing.T) {
	t.Run("matrix too small", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for small matrix")
			}
		}()
		m := []float32{1, 2}
		v := []float32{1, 2, 3}
		result := make([]float32, 2)
		MatVec(m, 2, 3, v, result)
	})

	t.Run("vector too small", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for small vector")
			}
		}()
		m := []float32{1, 2, 3, 4}
		v := []float32{1}
		result := make([]float32, 2)
		MatVec(m, 2, 2, v, result)
	})

	t.Run("result too small", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for small result")
			}
		}()
		m := []float32{1, 2, 3, 4}
		v := []float32{1, 2}
		result := make([]float32, 1)
		MatVec(m, 2, 2, v, result)
	})
}

// Helper functions

func make16x16Identity() []float32 {
	m := make([]float32, 16*16)
	for i := range 16 {
		m[i*16+i] = 1
	}
	return m
}

func makeRange(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(i)
	}
	return v
}

// Benchmarks

func BenchmarkMatVec(b *testing.B) {
	sizes := []struct {
		rows int
		cols int
	}{
		{10, 10},
		{100, 100},
		{256, 256},
		{1000, 1000},
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

		b.Run(string(rune(size.rows))+"x"+string(rune(size.cols)), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				MatVec(m, size.rows, size.cols, v, result)
			}
		})
	}
}

func BenchmarkMatVecFloat64(b *testing.B) {
	rows, cols := 256, 256
	m := make([]float64, rows*cols)
	for i := range m {
		m[i] = float64(i % 100)
	}
	v := make([]float64, cols)
	for i := range v {
		v[i] = float64(i)
	}
	result := make([]float64, rows)

	b.ReportAllocs()

	for b.Loop() {
		MatVecFloat64(m, rows, cols, v, result)
	}
}
