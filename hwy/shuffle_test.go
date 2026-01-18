package hwy

import (
	"reflect"
	"testing"
)

func TestReverse2_F32(t *testing.T) {
	tests := []struct {
		name   string
		input  []float32
		expect []float32
	}{
		{
			name:   "8 lanes",
			input:  []float32{0, 1, 2, 3, 4, 5, 6, 7},
			expect: []float32{1, 0, 3, 2, 5, 4, 7, 6},
		},
		{
			name:   "4 lanes",
			input:  []float32{0, 1, 2, 3},
			expect: []float32{1, 0, 3, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Vec[float32]{data: tt.input}
			result := Reverse2(v)
			if !reflect.DeepEqual(result.data, tt.expect) {
				t.Errorf("Reverse2() = %v, want %v", result.data, tt.expect)
			}
		})
	}
}

func TestReverse4_F32(t *testing.T) {
	tests := []struct {
		name   string
		input  []float32
		expect []float32
	}{
		{
			name:   "8 lanes",
			input:  []float32{0, 1, 2, 3, 4, 5, 6, 7},
			expect: []float32{3, 2, 1, 0, 7, 6, 5, 4},
		},
		{
			name:   "4 lanes",
			input:  []float32{0, 1, 2, 3},
			expect: []float32{3, 2, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Vec[float32]{data: tt.input}
			result := Reverse4(v)
			if !reflect.DeepEqual(result.data, tt.expect) {
				t.Errorf("Reverse4() = %v, want %v", result.data, tt.expect)
			}
		})
	}
}

func TestReverse8_F32(t *testing.T) {
	tests := []struct {
		name   string
		input  []float32
		expect []float32
	}{
		{
			name:   "8 lanes",
			input:  []float32{0, 1, 2, 3, 4, 5, 6, 7},
			expect: []float32{7, 6, 5, 4, 3, 2, 1, 0},
		},
		{
			name:   "16 lanes",
			input:  []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			expect: []float32{7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Vec[float32]{data: tt.input}
			result := Reverse8(v)
			if !reflect.DeepEqual(result.data, tt.expect) {
				t.Errorf("Reverse8() = %v, want %v", result.data, tt.expect)
			}
		})
	}
}

func TestGetLane(t *testing.T) {
	v := Vec[float32]{data: []float32{1, 2, 3, 4, 5, 6, 7, 8}}

	tests := []struct {
		idx    int
		expect float32
	}{
		{0, 1},
		{3, 4},
		{7, 8},
		{-1, 0}, // out of bounds
		{8, 0},  // out of bounds
	}

	for _, tt := range tests {
		result := GetLane(v, tt.idx)
		if result != tt.expect {
			t.Errorf("GetLane(%d) = %v, want %v", tt.idx, result, tt.expect)
		}
	}
}

func TestInsertLane(t *testing.T) {
	v := Vec[float32]{data: []float32{1, 2, 3, 4, 5, 6, 7, 8}}

	result := InsertLane(v, 3, 100)
	expect := []float32{1, 2, 3, 100, 5, 6, 7, 8}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("InsertLane() = %v, want %v", result.data, expect)
	}

	// Out of bounds should return original
	result2 := InsertLane(v, 10, 100)
	if !reflect.DeepEqual(result2.data, v.data) {
		t.Errorf("InsertLane out of bounds should return original")
	}
}

func TestInterleaveLower(t *testing.T) {
	a := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}
	b := Vec[float32]{data: []float32{10, 11, 12, 13, 14, 15, 16, 17}}

	result := InterleaveLower(a, b)
	expect := []float32{0, 10, 1, 11, 2, 12, 3, 13}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("InterleaveLower() = %v, want %v", result.data, expect)
	}
}

func TestInterleaveUpper(t *testing.T) {
	a := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}
	b := Vec[float32]{data: []float32{10, 11, 12, 13, 14, 15, 16, 17}}

	result := InterleaveUpper(a, b)
	expect := []float32{4, 14, 5, 15, 6, 16, 7, 17}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("InterleaveUpper() = %v, want %v", result.data, expect)
	}
}

func TestConcatLowerLower(t *testing.T) {
	a := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}
	b := Vec[float32]{data: []float32{10, 11, 12, 13, 14, 15, 16, 17}}

	result := ConcatLowerLower(a, b)
	expect := []float32{0, 1, 2, 3, 10, 11, 12, 13}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("ConcatLowerLower() = %v, want %v", result.data, expect)
	}
}

func TestConcatUpperUpper(t *testing.T) {
	a := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}
	b := Vec[float32]{data: []float32{10, 11, 12, 13, 14, 15, 16, 17}}

	result := ConcatUpperUpper(a, b)
	expect := []float32{4, 5, 6, 7, 14, 15, 16, 17}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("ConcatUpperUpper() = %v, want %v", result.data, expect)
	}
}

func TestConcatLowerUpper(t *testing.T) {
	a := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}
	b := Vec[float32]{data: []float32{10, 11, 12, 13, 14, 15, 16, 17}}

	result := ConcatLowerUpper(a, b)
	expect := []float32{0, 1, 2, 3, 14, 15, 16, 17}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("ConcatLowerUpper() = %v, want %v", result.data, expect)
	}
}

func TestConcatUpperLower(t *testing.T) {
	a := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}
	b := Vec[float32]{data: []float32{10, 11, 12, 13, 14, 15, 16, 17}}

	result := ConcatUpperLower(a, b)
	expect := []float32{4, 5, 6, 7, 10, 11, 12, 13}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("ConcatUpperLower() = %v, want %v", result.data, expect)
	}
}

func TestOddEven(t *testing.T) {
	a := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}
	b := Vec[float32]{data: []float32{10, 11, 12, 13, 14, 15, 16, 17}}

	result := OddEven(a, b)
	// even indices from b, odd indices from a
	expect := []float32{10, 1, 12, 3, 14, 5, 16, 7}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("OddEven() = %v, want %v", result.data, expect)
	}
}

func TestDupEven(t *testing.T) {
	v := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}

	result := DupEven(v)
	expect := []float32{0, 0, 2, 2, 4, 4, 6, 6}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("DupEven() = %v, want %v", result.data, expect)
	}
}

func TestDupOdd(t *testing.T) {
	v := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}

	result := DupOdd(v)
	expect := []float32{1, 1, 3, 3, 5, 5, 7, 7}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("DupOdd() = %v, want %v", result.data, expect)
	}
}

func TestSwapAdjacentBlocks_F32(t *testing.T) {
	// For float32, 128-bit block = 4 lanes
	v := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}

	result := SwapAdjacentBlocks(v)
	expect := []float32{4, 5, 6, 7, 0, 1, 2, 3}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("SwapAdjacentBlocks() = %v, want %v", result.data, expect)
	}
}

func TestSwapAdjacentBlocks_F64(t *testing.T) {
	// For float64, 128-bit block = 2 lanes
	v := Vec[float64]{data: []float64{0, 1, 2, 3}}

	result := SwapAdjacentBlocks(v)
	expect := []float64{2, 3, 0, 1}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("SwapAdjacentBlocks() = %v, want %v", result.data, expect)
	}
}

func TestTableLookupBytes(t *testing.T) {
	tbl := Vec[int32]{data: []int32{10, 20, 30, 40, 50, 60, 70, 80}}
	idx := Vec[int32]{data: []int32{0, 2, 4, 6, 1, 3, 5, 7}}

	result := TableLookupBytes(tbl, idx)
	expect := []int32{10, 30, 50, 70, 20, 40, 60, 80}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("TableLookupBytes() = %v, want %v", result.data, expect)
	}
}

// Benchmark tests
func BenchmarkReverse2_F32(b *testing.B) {
	v := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}

	for b.Loop() {
		_ = Reverse2(v)
	}
}

func BenchmarkReverse4_F32(b *testing.B) {
	v := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}

	for b.Loop() {
		_ = Reverse4(v)
	}
}

func BenchmarkInterleaveLower_F32(b *testing.B) {
	a := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}
	bVec := Vec[float32]{data: []float32{10, 11, 12, 13, 14, 15, 16, 17}}

	for b.Loop() {
		_ = InterleaveLower(a, bVec)
	}
}

func BenchmarkOddEven_F32(b *testing.B) {
	a := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}
	bVec := Vec[float32]{data: []float32{10, 11, 12, 13, 14, 15, 16, 17}}

	for b.Loop() {
		_ = OddEven(a, bVec)
	}
}

// Slide operation tests

func TestSlideUpLanes(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		tests := []struct {
			name   string
			input  []float32
			offset int
			expect []float32
		}{
			{
				name:   "slide_by_0",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 0,
				expect: []float32{1, 2, 3, 4, 5, 6, 7, 8},
			},
			{
				name:   "slide_by_1",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 1,
				expect: []float32{0, 1, 2, 3, 4, 5, 6, 7},
			},
			{
				name:   "slide_by_2",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 2,
				expect: []float32{0, 0, 1, 2, 3, 4, 5, 6},
			},
			{
				name:   "slide_by_4",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 4,
				expect: []float32{0, 0, 0, 0, 1, 2, 3, 4},
			},
			{
				name:   "slide_by_7",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 7,
				expect: []float32{0, 0, 0, 0, 0, 0, 0, 1},
			},
			{
				name:   "slide_by_8_all_zeros",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 8,
				expect: []float32{0, 0, 0, 0, 0, 0, 0, 0},
			},
			{
				name:   "slide_by_negative",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: -1,
				expect: []float32{1, 2, 3, 4, 5, 6, 7, 8},
			},
			{
				name:   "slide_beyond_length",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 100,
				expect: []float32{0, 0, 0, 0, 0, 0, 0, 0},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[float32]{data: tt.input}
				result := SlideUpLanes(v, tt.offset)
				if !reflect.DeepEqual(result.data, tt.expect) {
					t.Errorf("SlideUpLanes() = %v, want %v", result.data, tt.expect)
				}
			})
		}
	})

	t.Run("int32", func(t *testing.T) {
		v := Vec[int32]{data: []int32{10, 20, 30, 40}}
		result := SlideUpLanes(v, 2)
		expect := []int32{0, 0, 10, 20}
		if !reflect.DeepEqual(result.data, expect) {
			t.Errorf("SlideUpLanes() = %v, want %v", result.data, expect)
		}
	})

	t.Run("float64", func(t *testing.T) {
		v := Vec[float64]{data: []float64{1, 2, 3, 4}}
		result := SlideUpLanes(v, 1)
		expect := []float64{0, 1, 2, 3}
		if !reflect.DeepEqual(result.data, expect) {
			t.Errorf("SlideUpLanes() = %v, want %v", result.data, expect)
		}
	})
}

func TestSlideDownLanes(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		tests := []struct {
			name   string
			input  []float32
			offset int
			expect []float32
		}{
			{
				name:   "slide_by_0",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 0,
				expect: []float32{1, 2, 3, 4, 5, 6, 7, 8},
			},
			{
				name:   "slide_by_1",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 1,
				expect: []float32{2, 3, 4, 5, 6, 7, 8, 0},
			},
			{
				name:   "slide_by_2",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 2,
				expect: []float32{3, 4, 5, 6, 7, 8, 0, 0},
			},
			{
				name:   "slide_by_4",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 4,
				expect: []float32{5, 6, 7, 8, 0, 0, 0, 0},
			},
			{
				name:   "slide_by_7",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 7,
				expect: []float32{8, 0, 0, 0, 0, 0, 0, 0},
			},
			{
				name:   "slide_by_8_all_zeros",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 8,
				expect: []float32{0, 0, 0, 0, 0, 0, 0, 0},
			},
			{
				name:   "slide_by_negative",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: -1,
				expect: []float32{1, 2, 3, 4, 5, 6, 7, 8},
			},
			{
				name:   "slide_beyond_length",
				input:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				offset: 100,
				expect: []float32{0, 0, 0, 0, 0, 0, 0, 0},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[float32]{data: tt.input}
				result := SlideDownLanes(v, tt.offset)
				if !reflect.DeepEqual(result.data, tt.expect) {
					t.Errorf("SlideDownLanes() = %v, want %v", result.data, tt.expect)
				}
			})
		}
	})

	t.Run("int32", func(t *testing.T) {
		v := Vec[int32]{data: []int32{10, 20, 30, 40}}
		result := SlideDownLanes(v, 2)
		expect := []int32{30, 40, 0, 0}
		if !reflect.DeepEqual(result.data, expect) {
			t.Errorf("SlideDownLanes() = %v, want %v", result.data, expect)
		}
	})

	t.Run("float64", func(t *testing.T) {
		v := Vec[float64]{data: []float64{1, 2, 3, 4}}
		result := SlideDownLanes(v, 1)
		expect := []float64{2, 3, 4, 0}
		if !reflect.DeepEqual(result.data, expect) {
			t.Errorf("SlideDownLanes() = %v, want %v", result.data, expect)
		}
	})
}

func TestSlide1Up(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		v := Vec[float32]{data: []float32{1, 2, 3, 4, 5, 6, 7, 8}}
		result := Slide1Up(v)
		expect := []float32{0, 1, 2, 3, 4, 5, 6, 7}
		if !reflect.DeepEqual(result.data, expect) {
			t.Errorf("Slide1Up() = %v, want %v", result.data, expect)
		}
	})

	t.Run("int64", func(t *testing.T) {
		v := Vec[int64]{data: []int64{10, 20, 30, 40}}
		result := Slide1Up(v)
		expect := []int64{0, 10, 20, 30}
		if !reflect.DeepEqual(result.data, expect) {
			t.Errorf("Slide1Up() = %v, want %v", result.data, expect)
		}
	})

	t.Run("uint8", func(t *testing.T) {
		v := Vec[uint8]{data: []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}
		result := Slide1Up(v)
		expect := []uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
		if !reflect.DeepEqual(result.data, expect) {
			t.Errorf("Slide1Up() = %v, want %v", result.data, expect)
		}
	})
}

func TestSlide1Down(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		v := Vec[float32]{data: []float32{1, 2, 3, 4, 5, 6, 7, 8}}
		result := Slide1Down(v)
		expect := []float32{2, 3, 4, 5, 6, 7, 8, 0}
		if !reflect.DeepEqual(result.data, expect) {
			t.Errorf("Slide1Down() = %v, want %v", result.data, expect)
		}
	})

	t.Run("int64", func(t *testing.T) {
		v := Vec[int64]{data: []int64{10, 20, 30, 40}}
		result := Slide1Down(v)
		expect := []int64{20, 30, 40, 0}
		if !reflect.DeepEqual(result.data, expect) {
			t.Errorf("Slide1Down() = %v, want %v", result.data, expect)
		}
	})

	t.Run("uint8", func(t *testing.T) {
		v := Vec[uint8]{data: []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}
		result := Slide1Down(v)
		expect := []uint8{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0}
		if !reflect.DeepEqual(result.data, expect) {
			t.Errorf("Slide1Down() = %v, want %v", result.data, expect)
		}
	})
}

// Slide operation benchmarks

func BenchmarkSlideUpLanes_F32(b *testing.B) {
	v := Vec[float32]{data: []float32{1, 2, 3, 4, 5, 6, 7, 8}}

	for b.Loop() {
		_ = SlideUpLanes(v, 3)
	}
}

func BenchmarkSlideDownLanes_F32(b *testing.B) {
	v := Vec[float32]{data: []float32{1, 2, 3, 4, 5, 6, 7, 8}}

	for b.Loop() {
		_ = SlideDownLanes(v, 3)
	}
}

func BenchmarkSlide1Up_F32(b *testing.B) {
	v := Vec[float32]{data: []float32{1, 2, 3, 4, 5, 6, 7, 8}}

	for b.Loop() {
		_ = Slide1Up(v)
	}
}

func BenchmarkSlide1Down_F32(b *testing.B) {
	v := Vec[float32]{data: []float32{1, 2, 3, 4, 5, 6, 7, 8}}

	for b.Loop() {
		_ = Slide1Down(v)
	}
}

// New shuffle operation tests

func TestTableLookupLanes(t *testing.T) {
	tbl := Vec[float32]{data: []float32{10, 20, 30, 40, 50, 60, 70, 80}}
	idx := Vec[int32]{data: []int32{7, 5, 3, 1, 6, 4, 2, 0}}

	result := TableLookupLanes(tbl, idx)
	expect := []float32{80, 60, 40, 20, 70, 50, 30, 10}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("TableLookupLanes() = %v, want %v", result.data, expect)
	}
}

func TestTableLookupLanesOutOfBounds(t *testing.T) {
	tbl := Vec[float32]{data: []float32{10, 20, 30, 40}}
	idx := Vec[int32]{data: []int32{0, 10, 2, -1}} // 10 and -1 are out of bounds

	result := TableLookupLanes(tbl, idx)
	expect := []float32{10, 0, 30, 0} // out of bounds get zero
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("TableLookupLanes() = %v, want %v", result.data, expect)
	}
}

func TestTableLookupLanesOr(t *testing.T) {
	tbl := Vec[float32]{data: []float32{10, 20, 30, 40}}
	idx := Vec[int32]{data: []int32{0, 10, 2, -1}} // 10 and -1 are out of bounds
	fallback := Vec[float32]{data: []float32{100, 200, 300, 400}}

	result := TableLookupLanesOr(tbl, idx, fallback)
	expect := []float32{10, 200, 30, 400} // out of bounds get fallback
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("TableLookupLanesOr() = %v, want %v", result.data, expect)
	}
}

func TestZipLower(t *testing.T) {
	a := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}
	b := Vec[float32]{data: []float32{10, 11, 12, 13, 14, 15, 16, 17}}

	result := ZipLower(a, b)
	// Same as InterleaveLower
	expect := []float32{0, 10, 1, 11, 2, 12, 3, 13}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("ZipLower() = %v, want %v", result.data, expect)
	}
}

func TestZipUpper(t *testing.T) {
	a := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}
	b := Vec[float32]{data: []float32{10, 11, 12, 13, 14, 15, 16, 17}}

	result := ZipUpper(a, b)
	// Same as InterleaveUpper
	expect := []float32{4, 14, 5, 15, 6, 16, 7, 17}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("ZipUpper() = %v, want %v", result.data, expect)
	}
}

func TestShuffle0123(t *testing.T) {
	v := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}

	// Reverse each 4-lane group
	result := Shuffle0123(v, 3, 2, 1, 0)
	expect := []float32{3, 2, 1, 0, 7, 6, 5, 4}
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("Shuffle0123() = %v, want %v", result.data, expect)
	}

	// Broadcast lane 0 to all lanes in each group
	result2 := Shuffle0123(v, 0, 0, 0, 0)
	expect2 := []float32{0, 0, 0, 0, 4, 4, 4, 4}
	if !reflect.DeepEqual(result2.data, expect2) {
		t.Errorf("Shuffle0123(broadcast) = %v, want %v", result2.data, expect2)
	}

	// Interleave pattern
	result3 := Shuffle0123(v, 0, 2, 1, 3)
	expect3 := []float32{0, 2, 1, 3, 4, 6, 5, 7}
	if !reflect.DeepEqual(result3.data, expect3) {
		t.Errorf("Shuffle0123(interleave) = %v, want %v", result3.data, expect3)
	}
}

func TestPer4LaneBlockShuffle(t *testing.T) {
	v := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}

	// Pattern: 0b11_10_01_00 = reverse order (3,2,1,0 -> 0,1,2,3 positions)
	result := Per4LaneBlockShuffle(v, 0b11_10_01_00)
	expect := []float32{0, 1, 2, 3, 4, 5, 6, 7} // Identity pattern
	if !reflect.DeepEqual(result.data, expect) {
		t.Errorf("Per4LaneBlockShuffle(identity) = %v, want %v", result.data, expect)
	}

	// Pattern: 0b00_01_10_11 = reverse (0,1,2,3 -> 3,2,1,0)
	result2 := Per4LaneBlockShuffle(v, 0b00_01_10_11)
	expect2 := []float32{3, 2, 1, 0, 7, 6, 5, 4}
	if !reflect.DeepEqual(result2.data, expect2) {
		t.Errorf("Per4LaneBlockShuffle(reverse) = %v, want %v", result2.data, expect2)
	}

	// Pattern: 0b00_00_00_00 = broadcast lane 0
	result3 := Per4LaneBlockShuffle(v, 0b00_00_00_00)
	expect3 := []float32{0, 0, 0, 0, 4, 4, 4, 4}
	if !reflect.DeepEqual(result3.data, expect3) {
		t.Errorf("Per4LaneBlockShuffle(broadcast) = %v, want %v", result3.data, expect3)
	}
}

func BenchmarkTableLookupLanes_F32(b *testing.B) {
	tbl := Vec[float32]{data: []float32{10, 20, 30, 40, 50, 60, 70, 80}}
	idx := Vec[int32]{data: []int32{7, 5, 3, 1, 6, 4, 2, 0}}

	for b.Loop() {
		_ = TableLookupLanes(tbl, idx)
	}
}

func BenchmarkShuffle0123_F32(b *testing.B) {
	v := Vec[float32]{data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}

	for b.Loop() {
		_ = Shuffle0123(v, 3, 2, 1, 0)
	}
}
