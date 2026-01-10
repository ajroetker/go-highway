package hwy

import (
	"testing"
)

func TestPopCount(t *testing.T) {
	t.Run("uint8", func(t *testing.T) {
		tests := []struct {
			name  string
			input []uint8
			want  []uint8
		}{
			{
				name:  "zeros",
				input: []uint8{0, 0, 0, 0},
				want:  []uint8{0, 0, 0, 0},
			},
			{
				name:  "ones",
				input: []uint8{0xFF, 0xFF, 0xFF, 0xFF},
				want:  []uint8{8, 8, 8, 8},
			},
			{
				name:  "mixed",
				input: []uint8{0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF},
				want:  []uint8{1, 2, 3, 4, 5, 6, 7, 8},
			},
			{
				name:  "powers_of_two",
				input: []uint8{1, 2, 4, 8, 16, 32, 64, 128},
				want:  []uint8{1, 1, 1, 1, 1, 1, 1, 1},
			},
			{
				name:  "alternating",
				input: []uint8{0xAA, 0x55},
				want:  []uint8{4, 4},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[uint8]{data: tt.input}
				result := PopCount(v)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got %d, want %d", i, result.data[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("int32", func(t *testing.T) {
		tests := []struct {
			name  string
			input []int32
			want  []int32
		}{
			{
				name:  "zeros",
				input: []int32{0, 0, 0, 0},
				want:  []int32{0, 0, 0, 0},
			},
			{
				name:  "negative_one",
				input: []int32{-1, -1, -1, -1},
				want:  []int32{32, 32, 32, 32},
			},
			{
				name:  "mixed",
				input: []int32{1, 3, 7, 15},
				want:  []int32{1, 2, 3, 4},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[int32]{data: tt.input}
				result := PopCount(v)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got %d, want %d", i, result.data[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("uint64", func(t *testing.T) {
		v := Vec[uint64]{data: []uint64{0, 1, 0xFFFFFFFFFFFFFFFF, 0x8000000000000001}}
		result := PopCount(v)
		want := []uint64{0, 1, 64, 2}
		for i := 0; i < len(want); i++ {
			if result.data[i] != want[i] {
				t.Errorf("lane %d: got %d, want %d", i, result.data[i], want[i])
			}
		}
	})
}

func TestLeadingZeroCount(t *testing.T) {
	t.Run("uint8", func(t *testing.T) {
		tests := []struct {
			name  string
			input []uint8
			want  []uint8
		}{
			{
				name:  "zeros",
				input: []uint8{0, 0, 0, 0},
				want:  []uint8{8, 8, 8, 8},
			},
			{
				name:  "ones",
				input: []uint8{0xFF, 0x80, 0x40, 0x01},
				want:  []uint8{0, 0, 1, 7},
			},
			{
				name:  "powers_of_two",
				input: []uint8{1, 2, 4, 8, 16, 32, 64, 128},
				want:  []uint8{7, 6, 5, 4, 3, 2, 1, 0},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[uint8]{data: tt.input}
				result := LeadingZeroCount(v)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got %d, want %d", i, result.data[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("int32", func(t *testing.T) {
		tests := []struct {
			name  string
			input []int32
			want  []int32
		}{
			{
				name:  "zeros",
				input: []int32{0, 0, 0, 0},
				want:  []int32{32, 32, 32, 32},
			},
			{
				name:  "negative_one",
				input: []int32{-1, -1, -1, -1},
				want:  []int32{0, 0, 0, 0}, // -1 is all 1s, so no leading zeros
			},
			{
				name:  "positive",
				input: []int32{1, 256, 65536, 0x7FFFFFFF},
				want:  []int32{31, 23, 15, 1},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[int32]{data: tt.input}
				result := LeadingZeroCount(v)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got %d, want %d", i, result.data[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("uint64", func(t *testing.T) {
		v := Vec[uint64]{data: []uint64{0, 1, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF}}
		result := LeadingZeroCount(v)
		want := []uint64{64, 63, 0, 0}
		for i := 0; i < len(want); i++ {
			if result.data[i] != want[i] {
				t.Errorf("lane %d: got %d, want %d", i, result.data[i], want[i])
			}
		}
	})
}

func TestTrailingZeroCount(t *testing.T) {
	t.Run("uint8", func(t *testing.T) {
		tests := []struct {
			name  string
			input []uint8
			want  []uint8
		}{
			{
				name:  "zeros",
				input: []uint8{0, 0, 0, 0},
				want:  []uint8{8, 8, 8, 8},
			},
			{
				name:  "ones",
				input: []uint8{0x01, 0x02, 0x04, 0x80},
				want:  []uint8{0, 1, 2, 7},
			},
			{
				name:  "powers_of_two",
				input: []uint8{1, 2, 4, 8, 16, 32, 64, 128},
				want:  []uint8{0, 1, 2, 3, 4, 5, 6, 7},
			},
			{
				name:  "odd_values",
				input: []uint8{1, 3, 5, 7},
				want:  []uint8{0, 0, 0, 0}, // odd numbers have no trailing zeros
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[uint8]{data: tt.input}
				result := TrailingZeroCount(v)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got %d, want %d", i, result.data[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("int32", func(t *testing.T) {
		tests := []struct {
			name  string
			input []int32
			want  []int32
		}{
			{
				name:  "zeros",
				input: []int32{0, 0, 0, 0},
				want:  []int32{32, 32, 32, 32},
			},
			{
				name:  "powers_of_two",
				input: []int32{1, 2, 4, 8},
				want:  []int32{0, 1, 2, 3},
			},
			{
				name:  "negative_powers_of_two",
				input: []int32{-2, -4, -8, -16},
				want:  []int32{1, 2, 3, 4},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[int32]{data: tt.input}
				result := TrailingZeroCount(v)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got %d, want %d", i, result.data[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("uint64", func(t *testing.T) {
		v := Vec[uint64]{data: []uint64{0, 1, 0x8000000000000000, 0x0000000100000000}}
		result := TrailingZeroCount(v)
		want := []uint64{64, 0, 63, 32}
		for i := 0; i < len(want); i++ {
			if result.data[i] != want[i] {
				t.Errorf("lane %d: got %d, want %d", i, result.data[i], want[i])
			}
		}
	})
}

func TestRotateRight(t *testing.T) {
	t.Run("uint8", func(t *testing.T) {
		tests := []struct {
			name  string
			input []uint8
			count int
			want  []uint8
		}{
			{
				name:  "rotate_by_0",
				input: []uint8{0xAB, 0xCD},
				count: 0,
				want:  []uint8{0xAB, 0xCD},
			},
			{
				name:  "rotate_by_1",
				input: []uint8{0x01}, // 00000001
				count: 1,
				want:  []uint8{0x80}, // 10000000
			},
			{
				name:  "rotate_by_4",
				input: []uint8{0xAB}, // 10101011
				count: 4,
				want:  []uint8{0xBA}, // 10111010
			},
			{
				name:  "rotate_by_8",
				input: []uint8{0x12},
				count: 8,
				want:  []uint8{0x12}, // full rotation returns same value
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[uint8]{data: tt.input}
				result := RotateRight(v, tt.count)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got 0x%02X, want 0x%02X", i, result.data[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("uint32", func(t *testing.T) {
		tests := []struct {
			name  string
			input []uint32
			count int
			want  []uint32
		}{
			{
				name:  "rotate_by_0",
				input: []uint32{0x12345678},
				count: 0,
				want:  []uint32{0x12345678},
			},
			{
				name:  "rotate_by_8",
				input: []uint32{0x12345678},
				count: 8,
				want:  []uint32{0x78123456},
			},
			{
				name:  "rotate_by_16",
				input: []uint32{0x12345678},
				count: 16,
				want:  []uint32{0x56781234},
			},
			{
				name:  "rotate_by_32",
				input: []uint32{0x12345678},
				count: 32,
				want:  []uint32{0x12345678}, // full rotation
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[uint32]{data: tt.input}
				result := RotateRight(v, tt.count)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got 0x%08X, want 0x%08X", i, result.data[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("int32", func(t *testing.T) {
		// Test with negative values
		v := Vec[int32]{data: []int32{-1}}
		result := RotateRight(v, 5)
		// -1 is all 1s, so rotating returns the same value
		if result.data[0] != -1 {
			t.Errorf("got %d, want -1", result.data[0])
		}
	})
}

func TestReverseBits(t *testing.T) {
	t.Run("uint8", func(t *testing.T) {
		tests := []struct {
			name  string
			input []uint8
			want  []uint8
		}{
			{
				name:  "zeros",
				input: []uint8{0, 0},
				want:  []uint8{0, 0},
			},
			{
				name:  "ones",
				input: []uint8{0xFF, 0xFF},
				want:  []uint8{0xFF, 0xFF},
			},
			{
				name:  "single_bit_lsb",
				input: []uint8{0x01}, // 00000001
				want:  []uint8{0x80}, // 10000000
			},
			{
				name:  "single_bit_msb",
				input: []uint8{0x80}, // 10000000
				want:  []uint8{0x01}, // 00000001
			},
			{
				name:  "alternating",
				input: []uint8{0xAA}, // 10101010
				want:  []uint8{0x55}, // 01010101
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[uint8]{data: tt.input}
				result := ReverseBits(v)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got 0x%02X, want 0x%02X", i, result.data[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("uint32", func(t *testing.T) {
		tests := []struct {
			name  string
			input []uint32
			want  []uint32
		}{
			{
				name:  "zeros",
				input: []uint32{0},
				want:  []uint32{0},
			},
			{
				name:  "ones",
				input: []uint32{0xFFFFFFFF},
				want:  []uint32{0xFFFFFFFF},
			},
			{
				name:  "single_bit",
				input: []uint32{0x00000001},
				want:  []uint32{0x80000000},
			},
			{
				name:  "pattern",
				input: []uint32{0x12345678},
				want:  []uint32{0x1E6A2C48}, // manually verified
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[uint32]{data: tt.input}
				result := ReverseBits(v)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got 0x%08X, want 0x%08X", i, result.data[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("double_reverse", func(t *testing.T) {
		// Reversing twice should return original
		input := []uint32{0x12345678, 0xABCDEF01}
		v := Vec[uint32]{data: input}
		result := ReverseBits(ReverseBits(v))
		for i := 0; i < len(input); i++ {
			if result.data[i] != input[i] {
				t.Errorf("lane %d: double reverse got 0x%08X, want 0x%08X", i, result.data[i], input[i])
			}
		}
	})
}

func TestHighestSetBitIndex(t *testing.T) {
	t.Run("uint8", func(t *testing.T) {
		tests := []struct {
			name  string
			input []uint8
			want  []uint8
		}{
			{
				name:  "zeros",
				input: []uint8{0, 0},
				want:  []uint8{0xFF, 0xFF}, // -1 as uint8
			},
			{
				name:  "ones",
				input: []uint8{1, 2, 4, 8, 16, 32, 64, 128},
				want:  []uint8{0, 1, 2, 3, 4, 5, 6, 7},
			},
			{
				name:  "all_ones",
				input: []uint8{0xFF},
				want:  []uint8{7},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[uint8]{data: tt.input}
				result := HighestSetBitIndex(v)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got %d (0x%02X), want %d (0x%02X)",
							i, result.data[i], result.data[i], tt.want[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("int32", func(t *testing.T) {
		tests := []struct {
			name  string
			input []int32
			want  []int32
		}{
			{
				name:  "zeros",
				input: []int32{0, 0},
				want:  []int32{-1, -1},
			},
			{
				name:  "powers_of_two",
				input: []int32{1, 2, 4, 8},
				want:  []int32{0, 1, 2, 3},
			},
			{
				name:  "negative_one",
				input: []int32{-1}, // all bits set
				want:  []int32{31}, // highest bit is bit 31
			},
			{
				name:  "large_values",
				input: []int32{0x7FFFFFFF, 0x40000000},
				want:  []int32{30, 30},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				v := Vec[int32]{data: tt.input}
				result := HighestSetBitIndex(v)
				for i := 0; i < len(tt.want); i++ {
					if result.data[i] != tt.want[i] {
						t.Errorf("lane %d: got %d, want %d", i, result.data[i], tt.want[i])
					}
				}
			})
		}
	})

	t.Run("uint64", func(t *testing.T) {
		v := Vec[uint64]{data: []uint64{0, 1, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF}}
		result := HighestSetBitIndex(v)
		want := []uint64{0xFFFFFFFFFFFFFFFF, 0, 63, 63} // -1 as uint64, 0, 63, 63
		for i := 0; i < len(want); i++ {
			if result.data[i] != want[i] {
				t.Errorf("lane %d: got %d, want %d", i, result.data[i], want[i])
			}
		}
	})
}

// Benchmarks

func BenchmarkPopCount_U32(b *testing.B) {
	v := Vec[uint32]{data: []uint32{0xAAAAAAAA, 0x55555555, 0xFFFF0000, 0x00FFFF00, 0x12345678, 0x87654321, 0xDEADBEEF, 0xCAFEBABE}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = PopCount(v)
	}
}

func BenchmarkLeadingZeroCount_U32(b *testing.B) {
	v := Vec[uint32]{data: []uint32{0xAAAAAAAA, 0x55555555, 0xFFFF0000, 0x00FFFF00, 0x12345678, 0x87654321, 0xDEADBEEF, 0xCAFEBABE}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = LeadingZeroCount(v)
	}
}

func BenchmarkTrailingZeroCount_U32(b *testing.B) {
	v := Vec[uint32]{data: []uint32{0xAAAAAAAA, 0x55555555, 0xFFFF0000, 0x00FFFF00, 0x12345678, 0x87654321, 0xDEADBEEF, 0xCAFEBABE}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TrailingZeroCount(v)
	}
}

func BenchmarkRotateRight_U32(b *testing.B) {
	v := Vec[uint32]{data: []uint32{0x12345678, 0x87654321, 0xDEADBEEF, 0xCAFEBABE, 0xAAAAAAAA, 0x55555555, 0xFFFF0000, 0x00FFFF00}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = RotateRight(v, 7)
	}
}

func BenchmarkReverseBits_U32(b *testing.B) {
	v := Vec[uint32]{data: []uint32{0x12345678, 0x87654321, 0xDEADBEEF, 0xCAFEBABE, 0xAAAAAAAA, 0x55555555, 0xFFFF0000, 0x00FFFF00}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ReverseBits(v)
	}
}

func BenchmarkHighestSetBitIndex_U32(b *testing.B) {
	v := Vec[uint32]{data: []uint32{0x12345678, 0x87654321, 0xDEADBEEF, 0xCAFEBABE, 0xAAAAAAAA, 0x55555555, 0xFFFF0000, 0x00FFFF00}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = HighestSetBitIndex(v)
	}
}
