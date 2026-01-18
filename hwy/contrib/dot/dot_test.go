package dot

import (
	"math"
	"testing"
)

func TestDot(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		b    []float32
		want float32
	}{
		{
			name: "simple case",
			a:    []float32{1, 2, 3},
			b:    []float32{4, 5, 6},
			want: 32, // 1*4 + 2*5 + 3*6 = 32
		},
		{
			name: "exact AVX2 width (8 elements)",
			a:    []float32{1, 2, 3, 4, 5, 6, 7, 8},
			b:    []float32{8, 7, 6, 5, 4, 3, 2, 1},
			want: 120, // 8+14+18+20+20+18+14+8 = 120
		},
		{
			name: "larger than AVX2 width",
			a:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			b:    []float32{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
			want: 220, // 10+18+24+28+30+30+28+24+18+10 = 220
		},
		{
			name: "empty slices",
			a:    []float32{},
			b:    []float32{},
			want: 0,
		},
		{
			name: "different lengths",
			a:    []float32{1, 2, 3, 4, 5},
			b:    []float32{1, 2, 3},
			want: 14, // 1+4+9 = 14
		},
		{
			name: "zeros",
			a:    []float32{0, 0, 0, 0},
			b:    []float32{1, 2, 3, 4},
			want: 0,
		},
		{
			name: "negative values",
			a:    []float32{-1, -2, -3},
			b:    []float32{4, 5, 6},
			want: -32,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Dot(tt.a, tt.b)
			if math.Abs(float64(got-tt.want)) > 1e-5 {
				t.Errorf("Dot() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDotFloat64(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		b    []float64
		want float64
	}{
		{
			name: "simple case",
			a:    []float64{1, 2, 3},
			b:    []float64{4, 5, 6},
			want: 32,
		},
		{
			name: "exact AVX2 width (4 elements)",
			a:    []float64{1, 2, 3, 4},
			b:    []float64{4, 3, 2, 1},
			want: 20, // 4+6+6+4 = 20
		},
		{
			name: "larger than AVX2 width",
			a:    []float64{1, 2, 3, 4, 5, 6},
			b:    []float64{6, 5, 4, 3, 2, 1},
			want: 56, // 6+10+12+12+10+6 = 56
		},
		{
			name: "empty slices",
			a:    []float64{},
			b:    []float64{},
			want: 0,
		},
		{
			name: "high precision",
			a:    []float64{0.1, 0.2, 0.3},
			b:    []float64{0.3, 0.2, 0.1},
			want: 0.1*0.3 + 0.2*0.2 + 0.3*0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DotFloat64(tt.a, tt.b)
			if math.Abs(got-tt.want) > 1e-10 {
				t.Errorf("DotFloat64() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDotBatch(t *testing.T) {
	tests := []struct {
		name    string
		queries [][]float32
		keys    [][]float32
		want    []float32
	}{
		{
			name: "simple batch",
			queries: [][]float32{
				{1, 2, 3},
				{4, 5, 6},
			},
			keys: [][]float32{
				{7, 8, 9},
				{1, 2, 3},
			},
			want: []float32{50, 32}, // [1*7+2*8+3*9, 4*1+5*2+6*3]
		},
		{
			name: "different lengths",
			queries: [][]float32{
				{1, 2},
				{3, 4},
				{5, 6},
			},
			keys: [][]float32{
				{1, 1},
				{1, 1},
			},
			want: []float32{3, 7}, // Uses min length
		},
		{
			name:    "empty batch",
			queries: [][]float32{},
			keys:    [][]float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DotBatch(tt.queries, tt.keys)
			if len(got) != len(tt.want) {
				t.Fatalf("DotBatch() length = %v, want %v", len(got), len(tt.want))
			}
			for i := range got {
				if math.Abs(float64(got[i]-tt.want[i])) > 1e-5 {
					t.Errorf("DotBatch()[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

// Benchmarks

func BenchmarkDot(b *testing.B) {
	sizes := []int{16, 64, 256, 1024, 4096}

	for _, size := range sizes {
		a := make([]float32, size)
		c := make([]float32, size)
		for i := range a {
			a[i] = float32(i)
			c[i] = float32(i + 1)
		}

		b.Run(string(rune(size)), func(b *testing.B) {
			b.ReportAllocs()
			var result float32
			for i := 0; i < b.N; i++ {
				result = Dot(a, c)
			}
			_ = result
		})
	}
}

func BenchmarkDotFloat64(b *testing.B) {
	sizes := []int{16, 64, 256, 1024, 4096}

	for _, size := range sizes {
		a := make([]float64, size)
		c := make([]float64, size)
		for i := range a {
			a[i] = float64(i)
			c[i] = float64(i + 1)
		}

		b.Run(string(rune(size)), func(b *testing.B) {
			b.ReportAllocs()
			var result float64
			for i := 0; i < b.N; i++ {
				result = DotFloat64(a, c)
			}
			_ = result
		})
	}
}

func BenchmarkDotBatch(b *testing.B) {
	batchSize := 32
	vecSize := 256

	queries := make([][]float32, batchSize)
	keys := make([][]float32, batchSize)
	for i := range queries {
		queries[i] = make([]float32, vecSize)
		keys[i] = make([]float32, vecSize)
		for j := range queries[i] {
			queries[i][j] = float32(j)
			keys[i][j] = float32(j + 1)
		}
	}

	b.ReportAllocs()

	for b.Loop() {
		_ = DotBatch(queries, keys)
	}
}
