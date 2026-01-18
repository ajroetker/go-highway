package hwy

import (
	"testing"
)

func TestBlendedStore(t *testing.T) {
	t.Run("all true mask", func(t *testing.T) {
		v := Vec[float32]{data: []float32{1, 2, 3, 4}}
		mask := Mask[float32]{bits: []bool{true, true, true, true}}
		dst := []float32{10, 20, 30, 40}

		BlendedStore(v, mask, dst)

		want := []float32{1, 2, 3, 4}
		for i, got := range dst {
			if got != want[i] {
				t.Errorf("lane %d: got %v, want %v", i, got, want[i])
			}
		}
	})

	t.Run("all false mask", func(t *testing.T) {
		v := Vec[float32]{data: []float32{1, 2, 3, 4}}
		mask := Mask[float32]{bits: []bool{false, false, false, false}}
		dst := []float32{10, 20, 30, 40}

		BlendedStore(v, mask, dst)

		// dst should be unchanged
		want := []float32{10, 20, 30, 40}
		for i, got := range dst {
			if got != want[i] {
				t.Errorf("lane %d: got %v, want %v (should be unchanged)", i, got, want[i])
			}
		}
	})

	t.Run("mixed mask", func(t *testing.T) {
		v := Vec[float32]{data: []float32{1, 2, 3, 4}}
		mask := Mask[float32]{bits: []bool{true, false, true, false}}
		dst := []float32{10, 20, 30, 40}

		BlendedStore(v, mask, dst)

		want := []float32{1, 20, 3, 40} // lanes 0,2 changed; lanes 1,3 unchanged
		for i, got := range dst {
			if got != want[i] {
				t.Errorf("lane %d: got %v, want %v", i, got, want[i])
			}
		}
	})

	t.Run("int32", func(t *testing.T) {
		v := Vec[int32]{data: []int32{100, 200, 300, 400}}
		mask := Mask[int32]{bits: []bool{false, true, false, true}}
		dst := []int32{1, 2, 3, 4}

		BlendedStore(v, mask, dst)

		want := []int32{1, 200, 3, 400}
		for i, got := range dst {
			if got != want[i] {
				t.Errorf("lane %d: got %v, want %v", i, got, want[i])
			}
		}
	})

	t.Run("empty dst", func(t *testing.T) {
		v := Vec[float32]{data: []float32{1, 2, 3, 4}}
		mask := Mask[float32]{bits: []bool{true, true, true, true}}
		dst := []float32{}

		// Should not panic
		BlendedStore(v, mask, dst)
	})

	t.Run("partial dst", func(t *testing.T) {
		v := Vec[float32]{data: []float32{1, 2, 3, 4}}
		mask := Mask[float32]{bits: []bool{true, true, true, true}}
		dst := []float32{10, 20} // only 2 elements

		BlendedStore(v, mask, dst)

		want := []float32{1, 2}
		for i, got := range dst {
			if got != want[i] {
				t.Errorf("lane %d: got %v, want %v", i, got, want[i])
			}
		}
	})
}

func TestUndefined(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		v := Undefined[float32]()
		expectedLanes := MaxLanes[float32]()
		if v.NumLanes() != expectedLanes {
			t.Errorf("got %d lanes, want %d", v.NumLanes(), expectedLanes)
		}
	})

	t.Run("float64", func(t *testing.T) {
		v := Undefined[float64]()
		expectedLanes := MaxLanes[float64]()
		if v.NumLanes() != expectedLanes {
			t.Errorf("got %d lanes, want %d", v.NumLanes(), expectedLanes)
		}
	})

	t.Run("int32", func(t *testing.T) {
		v := Undefined[int32]()
		expectedLanes := MaxLanes[int32]()
		if v.NumLanes() != expectedLanes {
			t.Errorf("got %d lanes, want %d", v.NumLanes(), expectedLanes)
		}
	})

	t.Run("int64", func(t *testing.T) {
		v := Undefined[int64]()
		expectedLanes := MaxLanes[int64]()
		if v.NumLanes() != expectedLanes {
			t.Errorf("got %d lanes, want %d", v.NumLanes(), expectedLanes)
		}
	})

	t.Run("uint8", func(t *testing.T) {
		v := Undefined[uint8]()
		expectedLanes := MaxLanes[uint8]()
		if v.NumLanes() != expectedLanes {
			t.Errorf("got %d lanes, want %d", v.NumLanes(), expectedLanes)
		}
	})
}

func TestLoadDup128(t *testing.T) {
	t.Run("float32 basic", func(t *testing.T) {
		// 128 bits = 4 float32 values
		src := []float32{1, 2, 3, 4}
		v := LoadDup128(src)

		totalLanes := MaxLanes[float32]()
		if v.NumLanes() != totalLanes {
			t.Errorf("got %d lanes, want %d", v.NumLanes(), totalLanes)
		}

		// Check that the 128-bit block is duplicated
		blockSize := 4 // 16 bytes / 4 bytes per float32
		for block := 0; block < totalLanes; block += blockSize {
			for i := 0; i < blockSize && block+i < totalLanes; i++ {
				want := src[i]
				got := v.data[block+i]
				if got != want {
					t.Errorf("lane %d: got %v, want %v", block+i, got, want)
				}
			}
		}
	})

	t.Run("float64 basic", func(t *testing.T) {
		// 128 bits = 2 float64 values
		src := []float64{1.5, 2.5}
		v := LoadDup128(src)

		totalLanes := MaxLanes[float64]()
		if v.NumLanes() != totalLanes {
			t.Errorf("got %d lanes, want %d", v.NumLanes(), totalLanes)
		}

		// Check duplication
		blockSize := 2 // 16 bytes / 8 bytes per float64
		for block := 0; block < totalLanes; block += blockSize {
			for i := 0; i < blockSize && block+i < totalLanes; i++ {
				want := src[i]
				got := v.data[block+i]
				if got != want {
					t.Errorf("lane %d: got %v, want %v", block+i, got, want)
				}
			}
		}
	})

	t.Run("int32 basic", func(t *testing.T) {
		src := []int32{10, 20, 30, 40}
		v := LoadDup128(src)

		totalLanes := MaxLanes[int32]()
		blockSize := 4

		for block := 0; block < totalLanes; block += blockSize {
			for i := 0; i < blockSize && block+i < totalLanes; i++ {
				want := src[i]
				got := v.data[block+i]
				if got != want {
					t.Errorf("lane %d: got %v, want %v", block+i, got, want)
				}
			}
		}
	})

	t.Run("partial source", func(t *testing.T) {
		// Only 2 values provided, should fill rest with zeros in each block
		src := []float32{1, 2}
		v := LoadDup128(src)

		totalLanes := MaxLanes[float32]()
		blockSize := 4

		for block := 0; block < totalLanes; block += blockSize {
			// First two lanes of each block should have values
			if v.data[block] != 1 {
				t.Errorf("lane %d: got %v, want 1", block, v.data[block])
			}
			if block+1 < totalLanes && v.data[block+1] != 2 {
				t.Errorf("lane %d: got %v, want 2", block+1, v.data[block+1])
			}
			// Rest should be zero
			if block+2 < totalLanes && v.data[block+2] != 0 {
				t.Errorf("lane %d: got %v, want 0", block+2, v.data[block+2])
			}
			if block+3 < totalLanes && v.data[block+3] != 0 {
				t.Errorf("lane %d: got %v, want 0", block+3, v.data[block+3])
			}
		}
	})

	t.Run("empty source", func(t *testing.T) {
		src := []float32{}
		v := LoadDup128(src)

		// Should return vector of all zeros
		for i, val := range v.data {
			if val != 0 {
				t.Errorf("lane %d: got %v, want 0", i, val)
			}
		}
	})
}

func TestLoadInterleaved2(t *testing.T) {
	t.Run("float32 basic", func(t *testing.T) {
		// Interleaved pairs: [x0, y0, x1, y1, x2, y2, x3, y3, ...]
		n := MaxLanes[float32]()
		src := make([]float32, n*2)
		for i := range n {
			src[i*2] = float32(i)        // x values: 0, 1, 2, 3, ...
			src[i*2+1] = float32(i + 10) // y values: 10, 11, 12, 13, ...
		}

		x, y := LoadInterleaved2(src)

		for i := range n {
			wantX := float32(i)
			wantY := float32(i + 10)
			if x.data[i] != wantX {
				t.Errorf("x lane %d: got %v, want %v", i, x.data[i], wantX)
			}
			if y.data[i] != wantY {
				t.Errorf("y lane %d: got %v, want %v", i, y.data[i], wantY)
			}
		}
	})

	t.Run("int32", func(t *testing.T) {
		src := []int32{1, 100, 2, 200, 3, 300, 4, 400}
		x, y := LoadInterleaved2(src)

		wantX := []int32{1, 2, 3, 4}
		wantY := []int32{100, 200, 300, 400}

		for i := 0; i < 4 && i < len(x.data); i++ {
			if x.data[i] != wantX[i] {
				t.Errorf("x lane %d: got %v, want %v", i, x.data[i], wantX[i])
			}
			if y.data[i] != wantY[i] {
				t.Errorf("y lane %d: got %v, want %v", i, y.data[i], wantY[i])
			}
		}
	})
}

func TestLoadInterleaved3(t *testing.T) {
	t.Run("float32 RGB", func(t *testing.T) {
		// Simulate RGB data: [r0, g0, b0, r1, g1, b1, ...]
		n := MaxLanes[float32]()
		src := make([]float32, n*3)
		for i := range n {
			src[i*3] = float32(i)         // R: 0, 1, 2, ...
			src[i*3+1] = float32(i + 100) // G: 100, 101, ...
			src[i*3+2] = float32(i + 200) // B: 200, 201, ...
		}

		r, g, b := LoadInterleaved3(src)

		for i := range n {
			wantR := float32(i)
			wantG := float32(i + 100)
			wantB := float32(i + 200)
			if r.data[i] != wantR {
				t.Errorf("R lane %d: got %v, want %v", i, r.data[i], wantR)
			}
			if g.data[i] != wantG {
				t.Errorf("G lane %d: got %v, want %v", i, g.data[i], wantG)
			}
			if b.data[i] != wantB {
				t.Errorf("B lane %d: got %v, want %v", i, b.data[i], wantB)
			}
		}
	})
}

func TestLoadInterleaved4(t *testing.T) {
	t.Run("float32 RGBA", func(t *testing.T) {
		// Simulate RGBA data: [r0, g0, b0, a0, r1, g1, b1, a1, ...]
		n := MaxLanes[float32]()
		src := make([]float32, n*4)
		for i := range n {
			src[i*4] = float32(i)         // R
			src[i*4+1] = float32(i + 100) // G
			src[i*4+2] = float32(i + 200) // B
			src[i*4+3] = float32(i + 300) // A
		}

		r, g, b, a := LoadInterleaved4(src)

		for i := range n {
			if r.data[i] != float32(i) {
				t.Errorf("R lane %d: got %v, want %v", i, r.data[i], float32(i))
			}
			if g.data[i] != float32(i+100) {
				t.Errorf("G lane %d: got %v, want %v", i, g.data[i], float32(i+100))
			}
			if b.data[i] != float32(i+200) {
				t.Errorf("B lane %d: got %v, want %v", i, b.data[i], float32(i+200))
			}
			if a.data[i] != float32(i+300) {
				t.Errorf("A lane %d: got %v, want %v", i, a.data[i], float32(i+300))
			}
		}
	})
}

func TestStoreInterleaved2(t *testing.T) {
	t.Run("float32 basic", func(t *testing.T) {
		x := Vec[float32]{data: []float32{1, 2, 3, 4}}
		y := Vec[float32]{data: []float32{10, 20, 30, 40}}
		dst := make([]float32, 8)

		StoreInterleaved2(x, y, dst)

		want := []float32{1, 10, 2, 20, 3, 30, 4, 40}
		for i, got := range dst {
			if got != want[i] {
				t.Errorf("lane %d: got %v, want %v", i, got, want[i])
			}
		}
	})

	t.Run("int32", func(t *testing.T) {
		x := Vec[int32]{data: []int32{1, 2, 3, 4}}
		y := Vec[int32]{data: []int32{100, 200, 300, 400}}
		dst := make([]int32, 8)

		StoreInterleaved2(x, y, dst)

		want := []int32{1, 100, 2, 200, 3, 300, 4, 400}
		for i, got := range dst {
			if got != want[i] {
				t.Errorf("lane %d: got %v, want %v", i, got, want[i])
			}
		}
	})
}

func TestStoreInterleaved3(t *testing.T) {
	t.Run("float32 RGB", func(t *testing.T) {
		r := Vec[float32]{data: []float32{1, 2, 3, 4}}
		g := Vec[float32]{data: []float32{10, 20, 30, 40}}
		b := Vec[float32]{data: []float32{100, 200, 300, 400}}
		dst := make([]float32, 12)

		StoreInterleaved3(r, g, b, dst)

		want := []float32{1, 10, 100, 2, 20, 200, 3, 30, 300, 4, 40, 400}
		for i, got := range dst {
			if got != want[i] {
				t.Errorf("lane %d: got %v, want %v", i, got, want[i])
			}
		}
	})
}

func TestStoreInterleaved4(t *testing.T) {
	t.Run("float32 RGBA", func(t *testing.T) {
		r := Vec[float32]{data: []float32{1, 2, 3, 4}}
		g := Vec[float32]{data: []float32{10, 20, 30, 40}}
		b := Vec[float32]{data: []float32{100, 200, 300, 400}}
		a := Vec[float32]{data: []float32{1000, 2000, 3000, 4000}}
		dst := make([]float32, 16)

		StoreInterleaved4(r, g, b, a, dst)

		want := []float32{1, 10, 100, 1000, 2, 20, 200, 2000, 3, 30, 300, 3000, 4, 40, 400, 4000}
		for i, got := range dst {
			if got != want[i] {
				t.Errorf("lane %d: got %v, want %v", i, got, want[i])
			}
		}
	})
}

func TestLoadStoreInterleaved2_RoundTrip(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		n := MaxLanes[float32]()
		original := make([]float32, n*2)
		for i := range original {
			original[i] = float32(i)
		}

		// Load interleaved
		x, y := LoadInterleaved2(original)

		// Store interleaved
		reconstructed := make([]float32, n*2)
		StoreInterleaved2(x, y, reconstructed)

		// Should match original
		for i := 0; i < len(original) && i < len(reconstructed); i++ {
			if original[i] != reconstructed[i] {
				t.Errorf("index %d: got %v, want %v", i, reconstructed[i], original[i])
			}
		}
	})
}

func TestLoadStoreInterleaved3_RoundTrip(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		n := MaxLanes[float32]()
		original := make([]float32, n*3)
		for i := range original {
			original[i] = float32(i)
		}

		a, b, c := LoadInterleaved3(original)
		reconstructed := make([]float32, n*3)
		StoreInterleaved3(a, b, c, reconstructed)

		for i := 0; i < len(original) && i < len(reconstructed); i++ {
			if original[i] != reconstructed[i] {
				t.Errorf("index %d: got %v, want %v", i, reconstructed[i], original[i])
			}
		}
	})
}

func TestLoadStoreInterleaved4_RoundTrip(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		n := MaxLanes[float32]()
		original := make([]float32, n*4)
		for i := range original {
			original[i] = float32(i)
		}

		a, b, c, d := LoadInterleaved4(original)
		reconstructed := make([]float32, n*4)
		StoreInterleaved4(a, b, c, d, reconstructed)

		for i := 0; i < len(original) && i < len(reconstructed); i++ {
			if original[i] != reconstructed[i] {
				t.Errorf("index %d: got %v, want %v", i, reconstructed[i], original[i])
			}
		}
	})
}

// Benchmarks

func BenchmarkBlendedStore(b *testing.B) {
	v := Vec[float32]{data: make([]float32, 8)}
	mask := Mask[float32]{bits: []bool{true, false, true, false, true, false, true, false}}
	dst := make([]float32, 8)

	for i := range v.data {
		v.data[i] = float32(i)
	}

	for b.Loop() {
		BlendedStore(v, mask, dst)
	}
}

func BenchmarkLoadInterleaved2(b *testing.B) {
	n := MaxLanes[float32]()
	src := make([]float32, n*2)
	for i := range src {
		src[i] = float32(i)
	}

	for b.Loop() {
		_, _ = LoadInterleaved2(src)
	}
}

func BenchmarkStoreInterleaved2(b *testing.B) {
	n := 8
	x := Vec[float32]{data: make([]float32, n)}
	y := Vec[float32]{data: make([]float32, n)}
	dst := make([]float32, n*2)

	for i := range n {
		x.data[i] = float32(i)
		y.data[i] = float32(i + 10)
	}

	for b.Loop() {
		StoreInterleaved2(x, y, dst)
	}
}

func BenchmarkLoadDup128(b *testing.B) {
	src := []float32{1, 2, 3, 4}

	for b.Loop() {
		_ = LoadDup128(src)
	}
}
