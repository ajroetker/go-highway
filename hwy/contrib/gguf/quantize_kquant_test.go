package gguf

import (
	"fmt"
	"math"
	"testing"
)

func TestQuantizeQ8_K_RoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name: "ascending",
			input: func() []float32 {
				v := make([]float32, QK_K)
				for i := range v {
					v[i] = float32(i-128) * 0.1
				}
				return v
			}(),
		},
		{
			name: "uniform positive",
			input: func() []float32 {
				v := make([]float32, QK_K)
				for i := range v {
					v[i] = 1.0
				}
				return v
			}(),
		},
		{
			name:  "zeros",
			input: make([]float32, QK_K),
		},
		{
			name: "large values",
			input: func() []float32 {
				v := make([]float32, QK_K)
				for i := range v {
					v[i] = float32(i) * 100.0
				}
				return v
			}(),
		},
		{
			name: "two blocks",
			input: func() []float32 {
				v := make([]float32, 2*QK_K)
				for i := range v {
					v[i] = float32(i) - float32(len(v)/2)
				}
				return v
			}(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nblocks := len(tt.input) / QK_K
			qdata := make([]uint8, nblocks*BlockSizeQ8K)

			QuantizeQ8_K(tt.input, qdata)

			// Dequantize manually: d * qs[i].
			for b := range nblocks {
				block := qdata[b*BlockSizeQ8K : (b+1)*BlockSizeQ8K]
				d := f32LE(block[0], block[1], block[2], block[3])
				qs := block[4 : 4+QK_K]

				// Find max absolute value in this block.
				var amax float32
				for i := range QK_K {
					av := tt.input[b*QK_K+i]
					if av < 0 {
						av = -av
					}
					if av > amax {
						amax = av
					}
				}
				tol := float64(amax/127.0*1.01) + 1e-6

				for i := range QK_K {
					roundtrip := d * float32(int8(qs[i]))
					diff := math.Abs(float64(roundtrip - tt.input[b*QK_K+i]))
					if diff > tol {
						t.Errorf("block %d, index %d: input %f, roundtrip %f, diff %f > tol %f",
							b, i, tt.input[b*QK_K+i], roundtrip, diff, tol)
					}
				}
			}
		})
	}
}

func TestQuantizeQ8_K_Bsums(t *testing.T) {
	input := make([]float32, QK_K)
	for i := range input {
		input[i] = float32(i-128) * 0.5
	}

	qdata := make([]uint8, BlockSizeQ8K)
	QuantizeQ8_K(input, qdata)

	qs := qdata[4 : 4+QK_K]
	bsumsData := qdata[4+QK_K:]

	for j := range 16 {
		// Compute expected bsum from qs.
		var expected int16
		for k := range 16 {
			expected += int16(int8(qs[j*16+k]))
		}

		got := i16LE(bsumsData[j*2], bsumsData[j*2+1])
		if got != expected {
			t.Errorf("bsums[%d]: got %d, want %d", j, got, expected)
		}
	}
}

func TestQuantizeQ8_K_Zeros(t *testing.T) {
	input := make([]float32, QK_K)
	qdata := make([]uint8, BlockSizeQ8K)

	QuantizeQ8_K(input, qdata)

	d := f32LE(qdata[0], qdata[1], qdata[2], qdata[3])
	if d != 0 {
		t.Errorf("d = %f, want 0", d)
	}

	for i := range QK_K {
		if qdata[4+i] != 0 {
			t.Errorf("qs[%d] = %d, want 0", i, qdata[4+i])
		}
	}
}

func TestQuantizeQ8_K_Empty(t *testing.T) {
	QuantizeQ8_K(nil, nil)
	QuantizeQ8_K([]float32{}, []uint8{})
}

func TestQuantizeQ8_K_DispatchVsFallback(t *testing.T) {
	input := make([]float32, 2*QK_K)
	for i := range input {
		input[i] = float32(i%256-128) * 0.5
	}

	nblocks := len(input) / QK_K
	dispatched := make([]uint8, nblocks*BlockSizeQ8K)
	fallback := make([]uint8, nblocks*BlockSizeQ8K)

	QuantizeQ8_K(input, dispatched)
	BaseQuantizeQ8_K_fallback(input, fallback)

	for i := range dispatched {
		if dispatched[i] != fallback[i] {
			t.Errorf("byte %d: dispatched %d, fallback %d", i, dispatched[i], fallback[i])
		}
	}
}

func BenchmarkQuantizeQ8_K(b *testing.B) {
	sizes := []int{1, 4, 16, 64}
	for _, nblocks := range sizes {
		input := make([]float32, nblocks*QK_K)
		for i := range input {
			input[i] = float32(i%256-128) * 0.01
		}
		output := make([]uint8, nblocks*BlockSizeQ8K)

		b.Run(fmt.Sprintf("blocks=%d", nblocks), func(b *testing.B) {
			b.SetBytes(int64(len(input) * 4))
			for range b.N {
				QuantizeQ8_K(input, output)
			}
		})
	}
}
