package hwy

import (
	"testing"
)

func TestCompress(t *testing.T) {
	tests := []struct {
		name     string
		data     []float32
		mask     []bool
		wantData []float32
		wantCnt  int
	}{
		{
			name:     "all true",
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			mask:     []bool{true, true, true, true, true, true, true, true},
			wantData: []float32{1, 2, 3, 4, 5, 6, 7, 8},
			wantCnt:  8,
		},
		{
			name:     "all false",
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			mask:     []bool{false, false, false, false, false, false, false, false},
			wantData: []float32{0, 0, 0, 0, 0, 0, 0, 0},
			wantCnt:  0,
		},
		{
			name:     "alternating true first",
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			mask:     []bool{true, false, true, false, true, false, true, false},
			wantData: []float32{1, 3, 5, 7, 0, 0, 0, 0},
			wantCnt:  4,
		},
		{
			name:     "alternating false first",
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			mask:     []bool{false, true, false, true, false, true, false, true},
			wantData: []float32{2, 4, 6, 8, 0, 0, 0, 0},
			wantCnt:  4,
		},
		{
			name:     "first half true",
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			mask:     []bool{true, true, true, true, false, false, false, false},
			wantData: []float32{1, 2, 3, 4, 0, 0, 0, 0},
			wantCnt:  4,
		},
		{
			name:     "last half true",
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			mask:     []bool{false, false, false, false, true, true, true, true},
			wantData: []float32{5, 6, 7, 8, 0, 0, 0, 0},
			wantCnt:  4,
		},
		{
			name:     "single true",
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			mask:     []bool{false, false, false, true, false, false, false, false},
			wantData: []float32{4, 0, 0, 0, 0, 0, 0, 0},
			wantCnt:  1,
		},
		{
			name:     "random pattern",
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			mask:     []bool{true, false, true, true, false, false, true, false},
			wantData: []float32{1, 3, 4, 7, 0, 0, 0, 0},
			wantCnt:  4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Vec[float32]{data: tt.data}
			mask := Mask[float32]{bits: tt.mask}

			result, count := Compress(v, mask)

			if count != tt.wantCnt {
				t.Errorf("Compress count: got %d, want %d", count, tt.wantCnt)
			}

			for i := 0; i < len(tt.wantData) && i < len(result.data); i++ {
				if result.data[i] != tt.wantData[i] {
					t.Errorf("Compress lane %d: got %v, want %v", i, result.data[i], tt.wantData[i])
				}
			}
		})
	}
}

func TestExpand(t *testing.T) {
	tests := []struct {
		name     string
		data     []float32
		mask     []bool
		wantData []float32
	}{
		{
			name:     "all true",
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			mask:     []bool{true, true, true, true, true, true, true, true},
			wantData: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:     "all false",
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			mask:     []bool{false, false, false, false, false, false, false, false},
			wantData: []float32{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:     "alternating true first",
			data:     []float32{1, 2, 3, 4, 0, 0, 0, 0},
			mask:     []bool{true, false, true, false, true, false, true, false},
			wantData: []float32{1, 0, 2, 0, 3, 0, 4, 0},
		},
		{
			name:     "alternating false first",
			data:     []float32{1, 2, 3, 4, 0, 0, 0, 0},
			mask:     []bool{false, true, false, true, false, true, false, true},
			wantData: []float32{0, 1, 0, 2, 0, 3, 0, 4},
		},
		{
			name:     "first half true",
			data:     []float32{1, 2, 3, 4, 0, 0, 0, 0},
			mask:     []bool{true, true, true, true, false, false, false, false},
			wantData: []float32{1, 2, 3, 4, 0, 0, 0, 0},
		},
		{
			name:     "last half true",
			data:     []float32{5, 6, 7, 8, 0, 0, 0, 0},
			mask:     []bool{false, false, false, false, true, true, true, true},
			wantData: []float32{0, 0, 0, 0, 5, 6, 7, 8},
		},
		{
			name:     "random pattern",
			data:     []float32{1, 3, 4, 7, 0, 0, 0, 0},
			mask:     []bool{true, false, true, true, false, false, true, false},
			wantData: []float32{1, 0, 3, 4, 0, 0, 7, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Vec[float32]{data: tt.data}
			mask := Mask[float32]{bits: tt.mask}

			result := Expand(v, mask)

			for i := 0; i < len(tt.wantData) && i < len(result.data); i++ {
				if result.data[i] != tt.wantData[i] {
					t.Errorf("Expand lane %d: got %v, want %v", i, result.data[i], tt.wantData[i])
				}
			}
		})
	}
}

func TestCompressExpandRoundTrip(t *testing.T) {
	// Compress followed by Expand with the same mask should give back the original values
	// in the positions where the mask was true

	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	masks := [][]bool{
		{true, false, true, false, true, false, true, false},
		{false, true, false, true, false, true, false, true},
		{true, true, false, false, true, true, false, false},
		{true, true, true, true, true, true, true, true},
	}

	for _, maskBits := range masks {
		v := Vec[float32]{data: data}
		mask := Mask[float32]{bits: maskBits}

		// Compress then expand
		compressed, count := Compress(v, mask)
		_ = count
		expanded := Expand(compressed, mask)

		// Check that positions where mask is true have the original values
		for i := range data {
			if maskBits[i] {
				if expanded.data[i] != data[i] {
					t.Errorf("Round trip lane %d: got %v, want %v (mask pattern: %v)",
						i, expanded.data[i], data[i], maskBits)
				}
			} else {
				// Positions where mask is false should be zero
				if expanded.data[i] != 0 {
					t.Errorf("Round trip lane %d: got %v, want 0 (mask pattern: %v)",
						i, expanded.data[i], maskBits)
				}
			}
		}
	}
}

func TestCompressStore(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	mask := Mask[float32]{bits: []bool{true, false, true, true, false, false, true, false}}
	v := Vec[float32]{data: data}

	dst := make([]float32, 8)
	count := CompressStore(v, mask, dst)

	if count != 4 {
		t.Errorf("CompressStore count: got %d, want 4", count)
	}

	expected := []float32{1, 3, 4, 7}
	for i := range 4 {
		if dst[i] != expected[i] {
			t.Errorf("CompressStore dst[%d]: got %v, want %v", i, dst[i], expected[i])
		}
	}
}

func TestCountTrue(t *testing.T) {
	tests := []struct {
		name string
		mask []bool
		want int
	}{
		{"all true", []bool{true, true, true, true, true, true, true, true}, 8},
		{"all false", []bool{false, false, false, false, false, false, false, false}, 0},
		{"half true", []bool{true, true, true, true, false, false, false, false}, 4},
		{"alternating", []bool{true, false, true, false, true, false, true, false}, 4},
		{"single true", []bool{false, false, true, false, false, false, false, false}, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := Mask[float32]{bits: tt.mask}
			got := CountTrue(mask)
			if got != tt.want {
				t.Errorf("CountTrue: got %d, want %d", got, tt.want)
			}
		})
	}
}

func TestAllTrue(t *testing.T) {
	tests := []struct {
		name string
		mask []bool
		want bool
	}{
		{"all true", []bool{true, true, true, true}, true},
		{"all false", []bool{false, false, false, false}, false},
		{"one false", []bool{true, true, false, true}, false},
		{"one true", []bool{false, false, true, false}, false},
		{"empty", []bool{}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := Mask[float32]{bits: tt.mask}
			got := AllTrue(mask)
			if got != tt.want {
				t.Errorf("AllTrue: got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAllFalse(t *testing.T) {
	tests := []struct {
		name string
		mask []bool
		want bool
	}{
		{"all true", []bool{true, true, true, true}, false},
		{"all false", []bool{false, false, false, false}, true},
		{"one false", []bool{true, true, false, true}, false},
		{"one true", []bool{false, false, true, false}, false},
		{"empty", []bool{}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := Mask[float32]{bits: tt.mask}
			got := AllFalse(mask)
			if got != tt.want {
				t.Errorf("AllFalse: got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFindFirstTrue(t *testing.T) {
	tests := []struct {
		name string
		mask []bool
		want int
	}{
		{"first is true", []bool{true, false, false, false}, 0},
		{"middle is true", []bool{false, false, true, false}, 2},
		{"last is true", []bool{false, false, false, true}, 3},
		{"multiple true", []bool{false, true, true, true}, 1},
		{"all false", []bool{false, false, false, false}, -1},
		{"all true", []bool{true, true, true, true}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := Mask[float32]{bits: tt.mask}
			got := FindFirstTrue(mask)
			if got != tt.want {
				t.Errorf("FindFirstTrue: got %d, want %d", got, tt.want)
			}
		})
	}
}

func TestFindLastTrue(t *testing.T) {
	tests := []struct {
		name string
		mask []bool
		want int
	}{
		{"first is true", []bool{true, false, false, false}, 0},
		{"middle is true", []bool{false, false, true, false}, 2},
		{"last is true", []bool{false, false, false, true}, 3},
		{"multiple true", []bool{false, true, true, true}, 3},
		{"all false", []bool{false, false, false, false}, -1},
		{"all true", []bool{true, true, true, true}, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := Mask[float32]{bits: tt.mask}
			got := FindLastTrue(mask)
			if got != tt.want {
				t.Errorf("FindLastTrue: got %d, want %d", got, tt.want)
			}
		})
	}
}

func TestFirstN(t *testing.T) {
	maxLanes := MaxLanes[float32]()

	tests := []struct {
		name string
		n    int
	}{
		{"zero", 0},
		{"one", 1},
		{"half", maxLanes / 2},
		{"full", maxLanes},
		{"negative", -1},
		{"overflow", maxLanes + 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := FirstN[float32](tt.n)

			// Calculate expected count
			expectedN := min(max(tt.n, 0), maxLanes)

			// Check first expectedN lanes are true
			for i := range expectedN {
				if !mask.GetBit(i) {
					t.Errorf("FirstN(%d): lane %d should be true", tt.n, i)
				}
			}

			// Check remaining lanes are false
			for i := expectedN; i < maxLanes; i++ {
				if mask.GetBit(i) {
					t.Errorf("FirstN(%d): lane %d should be false", tt.n, i)
				}
			}
		})
	}
}

func TestLastN(t *testing.T) {
	maxLanes := MaxLanes[float32]()

	tests := []struct {
		name string
		n    int
	}{
		{"zero", 0},
		{"one", 1},
		{"half", maxLanes / 2},
		{"full", maxLanes},
		{"negative", -1},
		{"overflow", maxLanes + 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := LastN[float32](tt.n)

			// Calculate expected count
			expectedN := min(max(tt.n, 0), maxLanes)

			start := maxLanes - expectedN

			// Check first lanes are false
			for i := range start {
				if mask.GetBit(i) {
					t.Errorf("LastN(%d): lane %d should be false", tt.n, i)
				}
			}

			// Check last expectedN lanes are true
			for i := start; i < maxLanes; i++ {
				if !mask.GetBit(i) {
					t.Errorf("LastN(%d): lane %d should be true", tt.n, i)
				}
			}
		})
	}
}

func TestMaskFromBits(t *testing.T) {
	maxLanes := MaxLanes[float32]()

	tests := []struct {
		name string
		bits uint64
		want []bool
	}{
		{"zero", 0x00, make([]bool, maxLanes)},
		{"first bit", 0x01, append([]bool{true}, make([]bool, maxLanes-1)...)},
		{"second bit", 0x02, append([]bool{false, true}, make([]bool, maxLanes-2)...)},
		{"alternating", 0x55, []bool{true, false, true, false, true, false, true, false}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := MaskFromBits[float32](tt.bits)

			for i := 0; i < len(tt.want) && i < maxLanes; i++ {
				if mask.GetBit(i) != tt.want[i] {
					t.Errorf("MaskFromBits(0x%X): lane %d: got %v, want %v",
						tt.bits, i, mask.GetBit(i), tt.want[i])
				}
			}
		})
	}
}

func TestBitsFromMask(t *testing.T) {
	tests := []struct {
		name string
		mask []bool
		want uint64
	}{
		{"all false", []bool{false, false, false, false}, 0x00},
		{"first bit", []bool{true, false, false, false}, 0x01},
		{"second bit", []bool{false, true, false, false}, 0x02},
		{"all true 4", []bool{true, true, true, true}, 0x0F},
		{"alternating", []bool{true, false, true, false, true, false, true, false}, 0x55},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := Mask[float32]{bits: tt.mask}
			got := BitsFromMask(mask)
			if got != tt.want {
				t.Errorf("BitsFromMask: got 0x%X, want 0x%X", got, tt.want)
			}
		})
	}
}

func TestMaskFromBitsRoundTrip(t *testing.T) {
	maxLanes := MaxLanes[float32]()

	// Create a full mask that fits within maxLanes
	fullMask := uint64((1 << maxLanes) - 1)

	testBits := []uint64{0x00, 0x01, 0x55, 0xAA, 0x0F, 0xF0, fullMask}

	for _, bits := range testBits {
		// Mask to valid range
		validBits := bits & fullMask

		mask := MaskFromBits[float32](validBits)
		gotBits := BitsFromMask(mask)

		if gotBits != validBits {
			t.Errorf("MaskFromBits round trip: input 0x%X, got 0x%X", validBits, gotBits)
		}
	}
}

func TestMaskAnd(t *testing.T) {
	a := Mask[float32]{bits: []bool{true, true, false, false}}
	b := Mask[float32]{bits: []bool{true, false, true, false}}

	result := MaskAnd(a, b)
	expected := []bool{true, false, false, false}

	for i := range expected {
		if result.bits[i] != expected[i] {
			t.Errorf("MaskAnd lane %d: got %v, want %v", i, result.bits[i], expected[i])
		}
	}
}

func TestMaskOr(t *testing.T) {
	a := Mask[float32]{bits: []bool{true, true, false, false}}
	b := Mask[float32]{bits: []bool{true, false, true, false}}

	result := MaskOr(a, b)
	expected := []bool{true, true, true, false}

	for i := range expected {
		if result.bits[i] != expected[i] {
			t.Errorf("MaskOr lane %d: got %v, want %v", i, result.bits[i], expected[i])
		}
	}
}

func TestMaskXor(t *testing.T) {
	a := Mask[float32]{bits: []bool{true, true, false, false}}
	b := Mask[float32]{bits: []bool{true, false, true, false}}

	result := MaskXor(a, b)
	expected := []bool{false, true, true, false}

	for i := range expected {
		if result.bits[i] != expected[i] {
			t.Errorf("MaskXor lane %d: got %v, want %v", i, result.bits[i], expected[i])
		}
	}
}

func TestMaskNot(t *testing.T) {
	mask := Mask[float32]{bits: []bool{true, false, true, false}}

	result := MaskNot(mask)
	expected := []bool{false, true, false, true}

	for i := range expected {
		if result.bits[i] != expected[i] {
			t.Errorf("MaskNot lane %d: got %v, want %v", i, result.bits[i], expected[i])
		}
	}
}

func TestMaskAndNot(t *testing.T) {
	a := Mask[float32]{bits: []bool{true, true, false, false}}
	b := Mask[float32]{bits: []bool{true, false, true, false}}

	// (~a) & b
	result := MaskAndNot(a, b)
	expected := []bool{false, false, true, false}

	for i := range expected {
		if result.bits[i] != expected[i] {
			t.Errorf("MaskAndNot lane %d: got %v, want %v", i, result.bits[i], expected[i])
		}
	}
}

func TestCompressWithDifferentTypes(t *testing.T) {
	// Test with float64
	t.Run("float64", func(t *testing.T) {
		data := []float64{1, 2, 3, 4}
		mask := Mask[float64]{bits: []bool{true, false, true, false}}
		v := Vec[float64]{data: data}

		result, count := Compress(v, mask)

		if count != 2 {
			t.Errorf("Compress float64 count: got %d, want 2", count)
		}
		if result.data[0] != 1 || result.data[1] != 3 {
			t.Errorf("Compress float64: got %v, want [1, 3, ...]", result.data[:2])
		}
	})

	// Test with int32
	t.Run("int32", func(t *testing.T) {
		data := []int32{10, 20, 30, 40}
		mask := Mask[int32]{bits: []bool{false, true, false, true}}
		v := Vec[int32]{data: data}

		result, count := Compress(v, mask)

		if count != 2 {
			t.Errorf("Compress int32 count: got %d, want 2", count)
		}
		if result.data[0] != 20 || result.data[1] != 40 {
			t.Errorf("Compress int32: got %v, want [20, 40, ...]", result.data[:2])
		}
	})

	// Test with uint64
	t.Run("uint64", func(t *testing.T) {
		data := []uint64{100, 200, 300, 400}
		mask := Mask[uint64]{bits: []bool{true, true, false, false}}
		v := Vec[uint64]{data: data}

		result, count := Compress(v, mask)

		if count != 2 {
			t.Errorf("Compress uint64 count: got %d, want 2", count)
		}
		if result.data[0] != 100 || result.data[1] != 200 {
			t.Errorf("Compress uint64: got %v, want [100, 200, ...]", result.data[:2])
		}
	})
}

func TestCompressBlendedStore(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	mask := Mask[float32]{bits: []bool{true, false, true, true, false, false, true, false}}
	v := Vec[float32]{data: data}

	dst := []float32{99, 99, 99, 99, 99, 99, 99, 99}
	count := CompressBlendedStore(v, mask, dst)

	if count != 4 {
		t.Errorf("CompressBlendedStore count: got %d, want 4", count)
	}

	// First 4 should be replaced, rest should remain 99
	expected := []float32{1, 3, 4, 7, 99, 99, 99, 99}
	for i := range expected {
		if dst[i] != expected[i] {
			t.Errorf("CompressBlendedStore dst[%d]: got %v, want %v", i, dst[i], expected[i])
		}
	}
}

// Benchmark tests
func BenchmarkCompress(b *testing.B) {
	maxLanes := MaxLanes[float32]()
	data := make([]float32, maxLanes)
	for i := range data {
		data[i] = float32(i)
	}
	v := Vec[float32]{data: data}

	// Alternating mask (50% true)
	bits := make([]bool, maxLanes)
	for i := range maxLanes {
		bits[i] = i%2 == 0
	}
	mask := Mask[float32]{bits: bits}

	for b.Loop() {
		_, _ = Compress(v, mask)
	}
}

func BenchmarkExpand(b *testing.B) {
	maxLanes := MaxLanes[float32]()
	data := make([]float32, maxLanes)
	for i := range data {
		data[i] = float32(i)
	}
	v := Vec[float32]{data: data}

	// Alternating mask (50% true)
	bits := make([]bool, maxLanes)
	for i := range maxLanes {
		bits[i] = i%2 == 0
	}
	mask := Mask[float32]{bits: bits}

	for b.Loop() {
		_ = Expand(v, mask)
	}
}

func BenchmarkCompressStore(b *testing.B) {
	maxLanes := MaxLanes[float32]()
	data := make([]float32, maxLanes)
	for i := range data {
		data[i] = float32(i)
	}
	v := Vec[float32]{data: data}

	bits := make([]bool, maxLanes)
	for i := range maxLanes {
		bits[i] = i%2 == 0
	}
	mask := Mask[float32]{bits: bits}
	dst := make([]float32, maxLanes)

	for b.Loop() {
		_ = CompressStore(v, mask, dst)
	}
}
