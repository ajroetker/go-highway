package hwy

import (
	"testing"
)

func TestGatherIndex(t *testing.T) {
	src := []float32{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

	tests := []struct {
		name    string
		indices []int32
		want    []float32
	}{
		{
			name:    "sequential",
			indices: []int32{0, 1, 2, 3},
			want:    []float32{10, 20, 30, 40},
		},
		{
			name:    "reverse",
			indices: []int32{3, 2, 1, 0},
			want:    []float32{40, 30, 20, 10},
		},
		{
			name:    "scattered",
			indices: []int32{0, 4, 2, 8},
			want:    []float32{10, 50, 30, 90},
		},
		{
			name:    "repeated",
			indices: []int32{0, 0, 0, 0},
			want:    []float32{10, 10, 10, 10},
		},
		{
			name:    "out of bounds negative",
			indices: []int32{-1, 0, 1, 2},
			want:    []float32{0, 10, 20, 30},
		},
		{
			name:    "out of bounds positive",
			indices: []int32{0, 1, 100, 3},
			want:    []float32{10, 20, 0, 40},
		},
		{
			name:    "strided",
			indices: []int32{0, 2, 4, 6},
			want:    []float32{10, 30, 50, 70},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indices := Vec[int32]{data: tt.indices}
			result := GatherIndex(src, indices)

			for i := 0; i < len(tt.want) && i < len(result.data); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("GatherIndex lane %d: got %v, want %v", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestGatherIndexInt64(t *testing.T) {
	src := []float64{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8}

	indices := Vec[int64]{data: []int64{0, 2, 4, 6}}
	result := GatherIndex(src, indices)

	want := []float64{1.1, 3.3, 5.5, 7.7}
	for i := range want {
		if result.data[i] != want[i] {
			t.Errorf("GatherIndex lane %d: got %v, want %v", i, result.data[i], want[i])
		}
	}
}

func TestGatherIndexMasked(t *testing.T) {
	src := []float32{10, 20, 30, 40, 50, 60, 70, 80}

	tests := []struct {
		name    string
		indices []int32
		mask    []bool
		want    []float32
	}{
		{
			name:    "all true",
			indices: []int32{0, 1, 2, 3},
			mask:    []bool{true, true, true, true},
			want:    []float32{10, 20, 30, 40},
		},
		{
			name:    "all false",
			indices: []int32{0, 1, 2, 3},
			mask:    []bool{false, false, false, false},
			want:    []float32{0, 0, 0, 0},
		},
		{
			name:    "alternating",
			indices: []int32{0, 2, 4, 6},
			mask:    []bool{true, false, true, false},
			want:    []float32{10, 0, 50, 0},
		},
		{
			name:    "masked out of bounds",
			indices: []int32{-1, 100, 2, 3},
			mask:    []bool{true, true, true, true},
			want:    []float32{0, 0, 30, 40},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indices := Vec[int32]{data: tt.indices}
			mask := Mask[float32]{bits: tt.mask}
			result := GatherIndexMasked(src, indices, mask)

			for i := 0; i < len(tt.want) && i < len(result.data); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("GatherIndexMasked lane %d: got %v, want %v", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestScatterIndex(t *testing.T) {
	tests := []struct {
		name    string
		data    []float32
		indices []int32
		dstSize int
		want    []float32
	}{
		{
			name:    "sequential",
			data:    []float32{1, 2, 3, 4},
			indices: []int32{0, 1, 2, 3},
			dstSize: 8,
			want:    []float32{1, 2, 3, 4, 0, 0, 0, 0},
		},
		{
			name:    "reverse",
			data:    []float32{1, 2, 3, 4},
			indices: []int32{3, 2, 1, 0},
			dstSize: 8,
			want:    []float32{4, 3, 2, 1, 0, 0, 0, 0},
		},
		{
			name:    "scattered",
			data:    []float32{1, 2, 3, 4},
			indices: []int32{0, 2, 4, 6},
			dstSize: 8,
			want:    []float32{1, 0, 2, 0, 3, 0, 4, 0},
		},
		{
			name:    "with out of bounds",
			data:    []float32{1, 2, 3, 4},
			indices: []int32{0, 100, 2, -1},
			dstSize: 8,
			want:    []float32{1, 0, 3, 0, 0, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.dstSize)
			v := Vec[float32]{data: tt.data}
			indices := Vec[int32]{data: tt.indices}

			ScatterIndex(v, dst, indices)

			for i := 0; i < len(tt.want); i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ScatterIndex dst[%d]: got %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestScatterIndexMasked(t *testing.T) {
	tests := []struct {
		name    string
		data    []float32
		indices []int32
		mask    []bool
		dstSize int
		want    []float32
	}{
		{
			name:    "all true",
			data:    []float32{1, 2, 3, 4},
			indices: []int32{0, 1, 2, 3},
			mask:    []bool{true, true, true, true},
			dstSize: 8,
			want:    []float32{1, 2, 3, 4, 0, 0, 0, 0},
		},
		{
			name:    "all false",
			data:    []float32{1, 2, 3, 4},
			indices: []int32{0, 1, 2, 3},
			mask:    []bool{false, false, false, false},
			dstSize: 8,
			want:    []float32{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:    "alternating",
			data:    []float32{1, 2, 3, 4},
			indices: []int32{0, 2, 4, 6},
			mask:    []bool{true, false, true, false},
			dstSize: 8,
			want:    []float32{1, 0, 0, 0, 3, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.dstSize)
			v := Vec[float32]{data: tt.data}
			indices := Vec[int32]{data: tt.indices}
			mask := Mask[float32]{bits: tt.mask}

			ScatterIndexMasked(v, dst, indices, mask)

			for i := 0; i < len(tt.want); i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ScatterIndexMasked dst[%d]: got %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestGatherScatterRoundTrip(t *testing.T) {
	// Gather then scatter with same indices should reconstruct original (sparse) data
	src := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	indices := Vec[int32]{data: []int32{0, 2, 4, 6}}

	// Gather
	gathered := GatherIndex(src, indices)

	// Scatter to new array
	dst := make([]float32, 8)
	ScatterIndex(gathered, dst, indices)

	// Check that scattered values match original at indexed positions
	want := []float32{10, 0, 30, 0, 50, 0, 70, 0}
	for i := range want {
		if dst[i] != want[i] {
			t.Errorf("GatherScatter round trip dst[%d]: got %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestGatherIndexOffset(t *testing.T) {
	src := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

	tests := []struct {
		name    string
		base    int
		indices []int32
		scale   int
		want    []float32
	}{
		{
			name:    "base offset",
			base:    4,
			indices: []int32{0, 1, 2, 3},
			scale:   1,
			want:    []float32{4, 5, 6, 7},
		},
		{
			name:    "scaled access",
			base:    0,
			indices: []int32{0, 1, 2, 3},
			scale:   2,
			want:    []float32{0, 2, 4, 6},
		},
		{
			name:    "base and scale",
			base:    1,
			indices: []int32{0, 1, 2, 3},
			scale:   3,
			want:    []float32{1, 4, 7, 10},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indices := Vec[int32]{data: tt.indices}
			result := GatherIndexOffset(src, tt.base, indices, tt.scale)

			for i := 0; i < len(tt.want) && i < len(result.data); i++ {
				if result.data[i] != tt.want[i] {
					t.Errorf("GatherIndexOffset lane %d: got %v, want %v", i, result.data[i], tt.want[i])
				}
			}
		})
	}
}

func TestIndicesIota(t *testing.T) {
	result := IndicesIota[int32](8)

	for i := range 8 {
		if result.data[i] != int32(i) {
			t.Errorf("IndicesIota lane %d: got %v, want %v", i, result.data[i], int32(i))
		}
	}
}

func TestIndicesStride(t *testing.T) {
	result := IndicesStride[int32](8, 5, 3)

	want := []int32{5, 8, 11, 14, 17, 20, 23, 26}
	for i := 0; i < len(want) && i < len(result.data); i++ {
		if result.data[i] != want[i] {
			t.Errorf("IndicesStride lane %d: got %v, want %v", i, result.data[i], want[i])
		}
	}
}

func TestIndicesFromFunc(t *testing.T) {
	// Create indices that are squares
	result := IndicesFromFunc[int32](8, func(lane int) int32 {
		return int32(lane * lane)
	})

	want := []int32{0, 1, 4, 9, 16, 25, 36, 49}
	for i := 0; i < len(want) && i < len(result.data); i++ {
		if result.data[i] != want[i] {
			t.Errorf("IndicesFromFunc lane %d: got %v, want %v", i, result.data[i], want[i])
		}
	}
}

func TestGatherWithDifferentTypes(t *testing.T) {
	// Test with int32
	t.Run("int32", func(t *testing.T) {
		src := []int32{100, 200, 300, 400, 500}
		indices := Vec[int32]{data: []int32{4, 2, 0}}
		result := GatherIndex(src, indices)

		want := []int32{500, 300, 100}
		for i := range want {
			if result.data[i] != want[i] {
				t.Errorf("GatherIndex int32 lane %d: got %v, want %v", i, result.data[i], want[i])
			}
		}
	})

	// Test with float64
	t.Run("float64", func(t *testing.T) {
		src := []float64{1.5, 2.5, 3.5, 4.5, 5.5}
		indices := Vec[int64]{data: []int64{0, 2, 4}}
		result := GatherIndex(src, indices)

		want := []float64{1.5, 3.5, 5.5}
		for i := range want {
			if result.data[i] != want[i] {
				t.Errorf("GatherIndex float64 lane %d: got %v, want %v", i, result.data[i], want[i])
			}
		}
	})
}

// Benchmark tests
func BenchmarkGatherIndex(b *testing.B) {
	src := make([]float32, 1024)
	for i := range src {
		src[i] = float32(i)
	}
	indices := Vec[int32]{data: []int32{0, 64, 128, 192, 256, 320, 384, 448}}

	for b.Loop() {
		_ = GatherIndex(src, indices)
	}
}

func BenchmarkScatterIndex(b *testing.B) {
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	v := Vec[float32]{data: data}
	indices := Vec[int32]{data: []int32{0, 64, 128, 192, 256, 320, 384, 448}}
	dst := make([]float32, 512)

	for b.Loop() {
		ScatterIndex(v, dst, indices)
	}
}
