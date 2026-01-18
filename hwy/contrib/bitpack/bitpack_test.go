package bitpack

import (
	"math/rand"
	"testing"
)

func TestMaxBits32(t *testing.T) {
	tests := []struct {
		name string
		src  []uint32
		want int
	}{
		{
			name: "empty slice",
			src:  []uint32{},
			want: 0,
		},
		{
			name: "all zeros",
			src:  []uint32{0, 0, 0, 0},
			want: 0,
		},
		{
			name: "max 1 (1 bit)",
			src:  []uint32{0, 1, 0, 1},
			want: 1,
		},
		{
			name: "max 3 (2 bits)",
			src:  []uint32{1, 2, 3, 0},
			want: 2,
		},
		{
			name: "max 15 (4 bits)",
			src:  []uint32{5, 12, 3, 15, 7, 2, 9, 11},
			want: 4,
		},
		{
			name: "max 255 (8 bits)",
			src:  []uint32{100, 200, 255, 50},
			want: 8,
		},
		{
			name: "max 1000 (10 bits)",
			src:  []uint32{500, 1000, 750, 250},
			want: 10,
		},
		{
			name: "single element",
			src:  []uint32{42},
			want: 6, // 42 = 0b101010
		},
		{
			name: "large values (32 bits)",
			src:  []uint32{1 << 31, 100, 200},
			want: 32,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MaxBits(tt.src)
			if got != tt.want {
				t.Errorf("MaxBits() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestMaxBits64(t *testing.T) {
	tests := []struct {
		name string
		src  []uint64
		want int
	}{
		{
			name: "empty slice",
			src:  []uint64{},
			want: 0,
		},
		{
			name: "max 15 (4 bits)",
			src:  []uint64{5, 12, 3, 15},
			want: 4,
		},
		{
			name: "large values (40 bits)",
			src:  []uint64{1 << 39, 100, 200},
			want: 40,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MaxBits64(tt.src)
			if got != tt.want {
				t.Errorf("MaxBits64() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestPackedSize(t *testing.T) {
	tests := []struct {
		n        int
		bitWidth int
		want     int
	}{
		{0, 4, 0},
		{8, 0, 0},
		{8, 4, 4},   // 8 * 4 = 32 bits = 4 bytes
		{8, 5, 5},   // 8 * 5 = 40 bits = 5 bytes
		{10, 3, 4},  // 10 * 3 = 30 bits = 4 bytes (rounded up)
		{16, 8, 16}, // 16 * 8 = 128 bits = 16 bytes
		{1, 1, 1},   // 1 * 1 = 1 bit = 1 byte
	}

	for _, tt := range tests {
		got := PackedSize(tt.n, tt.bitWidth)
		if got != tt.want {
			t.Errorf("PackedSize(%d, %d) = %d, want %d", tt.n, tt.bitWidth, got, tt.want)
		}
	}
}

func TestPack32Unpack32(t *testing.T) {
	tests := []struct {
		name     string
		src      []uint32
		bitWidth int
	}{
		{
			name:     "4-bit values",
			src:      []uint32{5, 12, 3, 15, 7, 2, 9, 11},
			bitWidth: 4,
		},
		{
			name:     "1-bit values",
			src:      []uint32{0, 1, 1, 0, 1, 0, 0, 1},
			bitWidth: 1,
		},
		{
			name:     "8-bit values",
			src:      []uint32{100, 200, 255, 50, 128, 64, 32, 16},
			bitWidth: 8,
		},
		{
			name:     "5-bit values",
			src:      []uint32{0, 1, 15, 31, 20, 10, 5, 25},
			bitWidth: 5,
		},
		{
			name:     "12-bit values",
			src:      []uint32{0, 1000, 2000, 3000, 4095, 100, 500, 750},
			bitWidth: 12,
		},
		{
			name:     "non-aligned count",
			src:      []uint32{1, 2, 3, 4, 5},
			bitWidth: 4,
		},
		{
			name:     "single element",
			src:      []uint32{42},
			bitWidth: 6,
		},
		{
			name:     "32-bit values",
			src:      []uint32{1 << 31, 0xFFFFFFFF, 0x12345678, 0xDEADBEEF},
			bitWidth: 32,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Pack
			packed := make([]byte, PackedSize(len(tt.src), tt.bitWidth))
			bytesWritten := Pack32(tt.src, tt.bitWidth, packed)
			if bytesWritten == 0 && len(tt.src) > 0 {
				t.Fatal("Pack32 wrote 0 bytes")
			}

			// Unpack
			unpacked := make([]uint32, len(tt.src))
			count := Unpack32(packed, tt.bitWidth, unpacked)
			if count != len(tt.src) {
				t.Fatalf("Unpack32 returned %d, want %d", count, len(tt.src))
			}

			// Verify
			mask := uint32((1 << tt.bitWidth) - 1)
			if tt.bitWidth == 32 {
				mask = 0xFFFFFFFF
			}
			for i := range tt.src {
				expected := tt.src[i] & mask
				if unpacked[i] != expected {
					t.Errorf("unpacked[%d] = %d, want %d", i, unpacked[i], expected)
				}
			}
		})
	}
}

func TestPack64Unpack64(t *testing.T) {
	tests := []struct {
		name     string
		src      []uint64
		bitWidth int
	}{
		{
			name:     "4-bit values",
			src:      []uint64{5, 12, 3, 15},
			bitWidth: 4,
		},
		{
			name:     "16-bit values",
			src:      []uint64{1000, 2000, 30000, 65535},
			bitWidth: 16,
		},
		{
			name:     "40-bit values",
			src:      []uint64{1 << 39, 1 << 38, 1 << 37, 1 << 36},
			bitWidth: 40,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			packed := make([]byte, PackedSize(len(tt.src), tt.bitWidth))
			bytesWritten := Pack64(tt.src, tt.bitWidth, packed)
			if bytesWritten == 0 && len(tt.src) > 0 {
				t.Fatal("Pack64 wrote 0 bytes")
			}

			unpacked := make([]uint64, len(tt.src))
			count := Unpack64(packed, tt.bitWidth, unpacked)
			if count != len(tt.src) {
				t.Fatalf("Unpack64 returned %d, want %d", count, len(tt.src))
			}

			var mask uint64
			if tt.bitWidth == 64 {
				mask = ^uint64(0)
			} else {
				mask = (1 << tt.bitWidth) - 1
			}
			for i := range tt.src {
				expected := tt.src[i] & mask
				if unpacked[i] != expected {
					t.Errorf("unpacked[%d] = %d, want %d", i, unpacked[i], expected)
				}
			}
		})
	}
}

func TestDeltaEncode32Decode32(t *testing.T) {
	tests := []struct {
		name string
		src  []uint32
		base uint32
	}{
		{
			name: "simple sorted",
			src:  []uint32{100, 102, 105, 106, 110},
			base: 100,
		},
		{
			name: "monotonically increasing by 1",
			src:  []uint32{0, 1, 2, 3, 4, 5, 6, 7},
			base: 0,
		},
		{
			name: "larger gaps",
			src:  []uint32{1000, 1050, 1200, 1500, 2000},
			base: 1000,
		},
		{
			name: "single element",
			src:  []uint32{42},
			base: 40,
		},
		{
			name: "all same",
			src:  []uint32{100, 100, 100, 100},
			base: 100,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Encode
			encoded := make([]uint32, len(tt.src))
			DeltaEncode32(tt.src, tt.base, encoded)

			// Decode
			decoded := make([]uint32, len(tt.src))
			DeltaDecode(encoded, tt.base, decoded)

			// Verify
			for i := range tt.src {
				if decoded[i] != tt.src[i] {
					t.Errorf("decoded[%d] = %d, want %d", i, decoded[i], tt.src[i])
				}
			}
		})
	}
}

func TestDeltaEncode64Decode64(t *testing.T) {
	tests := []struct {
		name string
		src  []uint64
		base uint64
	}{
		{
			name: "simple sorted",
			src:  []uint64{1000, 1002, 1005, 1010, 1020},
			base: 1000,
		},
		{
			name: "large values",
			src:  []uint64{1 << 40, (1 << 40) + 100, (1 << 40) + 500},
			base: 1 << 40,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoded := make([]uint64, len(tt.src))
			DeltaEncode64(tt.src, tt.base, encoded)

			decoded := make([]uint64, len(tt.src))
			DeltaDecode(encoded, tt.base, decoded)

			for i := range tt.src {
				if decoded[i] != tt.src[i] {
					t.Errorf("decoded[%d] = %d, want %d", i, decoded[i], tt.src[i])
				}
			}
		})
	}
}

func TestDeltaEncodingCompression(t *testing.T) {
	// Test that delta encoding reduces bit width
	sorted := []uint32{100, 102, 105, 106, 110, 115, 118, 120}

	// Without delta encoding
	directBits := MaxBits(sorted)

	// With delta encoding
	deltas := make([]uint32, len(sorted))
	DeltaEncode32(sorted, sorted[0], deltas)
	deltaBits := MaxBits(deltas)

	if deltaBits >= directBits {
		t.Errorf("delta encoding should reduce bits: direct=%d, delta=%d", directBits, deltaBits)
	}

	t.Logf("Direct encoding: %d bits, Delta encoding: %d bits, Savings: %.1f%%",
		directBits, deltaBits, 100*(1-float64(deltaBits)/float64(directBits)))
}

func TestPackUnpackRoundtrip(t *testing.T) {
	// Test with random data at various bit widths
	rng := rand.New(rand.NewSource(42))

	for bitWidth := 1; bitWidth <= 32; bitWidth++ {
		t.Run("bits="+string(rune('0'+bitWidth)), func(t *testing.T) {
			n := 128
			src := make([]uint32, n)
			mask := uint32((1 << bitWidth) - 1)
			if bitWidth == 32 {
				mask = 0xFFFFFFFF
			}
			for i := range src {
				src[i] = rng.Uint32() & mask
			}

			packed := make([]byte, PackedSize(n, bitWidth))
			Pack32(src, bitWidth, packed)

			unpacked := make([]uint32, n)
			Unpack32(packed, bitWidth, unpacked)

			for i := range src {
				if unpacked[i] != src[i] {
					t.Errorf("bitWidth=%d, idx=%d: got %d, want %d", bitWidth, i, unpacked[i], src[i])
				}
			}
		})
	}
}

func TestEmptyInputs(t *testing.T) {
	// Test edge cases with empty inputs
	if got := MaxBits([]uint32{}); got != 0 {
		t.Errorf("MaxBits([]) = %d, want 0", got)
	}

	if got := Pack32([]uint32{}, 4, nil); got != 0 {
		t.Errorf("Pack32([], 4, nil) = %d, want 0", got)
	}

	if got := Unpack32([]byte{}, 4, nil); got != 0 {
		t.Errorf("Unpack32([], 4, nil) = %d, want 0", got)
	}

	// Zero bit width
	src := []uint32{1, 2, 3}
	if got := Pack32(src, 0, nil); got != 0 {
		t.Errorf("Pack32(src, 0, nil) = %d, want 0", got)
	}
}

// Benchmarks

func BenchmarkMaxBits32(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		data := make([]uint32, size)
		for i := range data {
			data[i] = uint32(i % 1000)
		}
		b.Run(string(rune(size)), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = MaxBits(data)
			}
		})
	}
}

func BenchmarkPack32(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	bitWidths := []int{4, 8, 12, 16}

	for _, size := range sizes {
		data := make([]uint32, size)
		for i := range data {
			data[i] = uint32(i % 1000)
		}

		for _, bw := range bitWidths {
			packed := make([]byte, PackedSize(size, bw))
			name := string(rune(size)) + "_" + string(rune(bw)) + "bits"
			b.Run(name, func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(size * 4))
				for i := 0; i < b.N; i++ {
					// Clear packed buffer
					for j := range packed {
						packed[j] = 0
					}
					Pack32(data, bw, packed)
				}
			})
		}
	}
}

func BenchmarkUnpack32(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	bitWidths := []int{4, 8, 12, 16}

	for _, size := range sizes {
		data := make([]uint32, size)
		for i := range data {
			data[i] = uint32(i % 1000)
		}

		for _, bw := range bitWidths {
			packed := make([]byte, PackedSize(size, bw))
			Pack32(data, bw, packed)
			unpacked := make([]uint32, size)

			name := string(rune(size)) + "_" + string(rune(bw)) + "bits"
			b.Run(name, func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(size * 4))
				for i := 0; i < b.N; i++ {
					Unpack32(packed, bw, unpacked)
				}
			})
		}
	}
}

func BenchmarkDeltaEncode32(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		data := make([]uint32, size)
		for i := range data {
			data[i] = uint32(1000 + i*2)
		}
		dst := make([]uint32, size)

		b.Run(string(rune(size)), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 4))
			for i := 0; i < b.N; i++ {
				DeltaEncode32(data, data[0], dst)
			}
		})
	}
}

func BenchmarkDeltaDecode(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		data := make([]uint32, size)
		for i := range data {
			data[i] = uint32(i * 2) // deltas
		}
		dst := make([]uint32, size)

		b.Run(string(rune(size)), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 4))
			for i := 0; i < b.N; i++ {
				DeltaDecode(data, 1000, dst)
			}
		})
	}
}

func BenchmarkFullPipeline(b *testing.B) {
	// Benchmark the full compression pipeline: delta encode + find bits + pack
	size := 1024
	sorted := make([]uint32, size)
	for i := range sorted {
		sorted[i] = uint32(100000 + i*3) // Sorted values
	}

	b.Run("encode+pack", func(b *testing.B) {
		deltas := make([]uint32, size)
		packed := make([]byte, size*4) // Overallocate

		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			DeltaEncode32(sorted, sorted[0], deltas)
			bw := MaxBits(deltas)
			for j := range packed {
				packed[j] = 0
			}
			Pack32(deltas, bw, packed[:PackedSize(size, bw)])
		}
	})

	// Pre-encode for decode benchmark
	deltas := make([]uint32, size)
	DeltaEncode32(sorted, sorted[0], deltas)
	bw := MaxBits(deltas)
	packed := make([]byte, PackedSize(size, bw))
	Pack32(deltas, bw, packed)

	b.Run("unpack+decode", func(b *testing.B) {
		unpacked := make([]uint32, size)
		decoded := make([]uint32, size)

		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			Unpack32(packed, bw, unpacked)
			DeltaDecode(unpacked, sorted[0], decoded)
		}
	})
}
