package gguf

import (
	"fmt"
	"math"
	"testing"
)

// makeTestWeightData creates weight data with valid structure for the given quant type.
// Returns (weight bytes, dequantized float32 values).
func makeTestWeightData(qt QuantType, nblocks int) ([]uint8, []float32) {
	blockSize := BytesPerBlock(qt)
	wdata := make([]uint8, nblocks*blockSize)

	// Fill with deterministic pattern.
	for i := range wdata {
		wdata[i] = uint8((i * 7 + 13) % 256)
	}

	// Set valid fp16 scales at block boundaries for each format.
	for b := range nblocks {
		off := b * blockSize
		switch qt {
		case TypeQ4_K:
			// d at offset 0-1, dmin at 2-3 (fp16)
			wdata[off] = fp16One[0]
			wdata[off+1] = fp16One[1]
			wdata[off+2] = 0x00
			wdata[off+3] = 0x38 // fp16 0.5
		case TypeQ6_K:
			// d at offset 208-209 (fp16)
			wdata[off+208] = fp16One[0]
			wdata[off+209] = fp16One[1]
		case TypeQ2_K:
			// d at offset 16-17, dmin at 18-19 (fp16)
			wdata[off+16] = fp16One[0]
			wdata[off+17] = fp16One[1]
			wdata[off+18] = 0x00
			wdata[off+19] = 0x38 // fp16 0.5
		case TypeQ3_K:
			// d at offset 108-109 (fp16)
			wdata[off+108] = fp16One[0]
			wdata[off+109] = fp16One[1]
		case TypeQ5_K:
			// d at offset 0-1, dmin at 2-3 (fp16)
			wdata[off] = fp16One[0]
			wdata[off+1] = fp16One[1]
			wdata[off+2] = 0x00
			wdata[off+3] = 0x38 // fp16 0.5
		}
	}

	// Dequantize weights for reference.
	nvals := nblocks * QK_K
	wFloat := make([]float32, nvals)
	switch qt {
	case TypeQ4_K:
		DequantizeQ4K(wdata, wFloat)
	case TypeQ6_K:
		DequantizeQ6K(wdata, wFloat)
	case TypeQ2_K:
		DequantizeQ2K(wdata, wFloat)
	case TypeQ3_K:
		DequantizeQ3K(wdata, wFloat)
	case TypeQ5_K:
		DequantizeQ5K(wdata, wFloat)
	}

	return wdata, wFloat
}

// makeTestActivationData creates float32 activations and quantizes to Q8_K.
// Returns (Q8_K bytes, dequantized float32 values for reference).
func makeTestActivationData(nblocks int) ([]uint8, []float32) {
	nvals := nblocks * QK_K
	aFloat := make([]float32, nvals)
	for i := range aFloat {
		aFloat[i] = float32(i%256-128) * 0.02
	}

	adata := make([]uint8, nblocks*BlockSizeQ8K)
	QuantizeQ8_K(aFloat, adata)

	// Dequantize Q8_K activations for reference: d * qs[i].
	aDeq := make([]float32, nvals)
	for b := range nblocks {
		block := adata[b*BlockSizeQ8K : (b+1)*BlockSizeQ8K]
		d := f32LE(block[0], block[1], block[2], block[3])
		qs := block[4 : 4+QK_K]
		for i := range QK_K {
			aDeq[b*QK_K+i] = d * float32(int8(qs[i]))
		}
	}

	return adata, aDeq
}

func testVecDotKQuant(t *testing.T, qt QuantType, vecDot func([]uint8, []uint8, int) float32, name string) {
	t.Helper()

	for _, nblocks := range []int{1, 4, 8} {
		t.Run(fmt.Sprintf("%s/%d_blocks", name, nblocks), func(t *testing.T) {
			wdata, wFloat := makeTestWeightData(qt, nblocks)
			adata, aDeq := makeTestActivationData(nblocks)

			// Reference: float dot product of dequantized values.
			want := referenceDot(wFloat, aDeq)

			// Compute via vecdot.
			got := vecDot(wdata, adata, nblocks)

			// Tolerance: double quantization error.
			// K-quant types have sub-block scales (6-bit) → more error.
			relErr := float64(0)
			if want != 0 {
				relErr = math.Abs(float64(got-want)) / math.Abs(float64(want))
			}
			absDiff := math.Abs(float64(got - want))
			if relErr > 0.05 && absDiff > 1.0 {
				t.Errorf("got %f, want %f (relErr=%.4f, absDiff=%.4f)", got, want, relErr, absDiff)
			}
		})
	}
}

func TestVecDotQ4_KQ8_K(t *testing.T) {
	testVecDotKQuant(t, TypeQ4_K, VecDotQ4_KQ8_K, "Q4_K")
}

func TestVecDotQ6_KQ8_K(t *testing.T) {
	testVecDotKQuant(t, TypeQ6_K, VecDotQ6_KQ8_K, "Q6_K")
}

func TestVecDotQ2_KQ8_K(t *testing.T) {
	testVecDotKQuant(t, TypeQ2_K, VecDotQ2_KQ8_K, "Q2_K")
}

func TestVecDotQ3_KQ8_K(t *testing.T) {
	testVecDotKQuant(t, TypeQ3_K, VecDotQ3_KQ8_K, "Q3_K")
}

func TestVecDotQ5_KQ8_K(t *testing.T) {
	testVecDotKQuant(t, TypeQ5_K, VecDotQ5_KQ8_K, "Q5_K")
}

func testVecDotKQuantDispatchVsFallback(t *testing.T, qt QuantType, vecDot func([]uint8, []uint8, int) float32, fallback func([]uint8, []uint8, int) float32, name string) {
	t.Helper()

	nblocks := 4
	wdata, _ := makeTestWeightData(qt, nblocks)
	adata, _ := makeTestActivationData(nblocks)

	got := vecDot(wdata, adata, nblocks)
	want := fallback(wdata, adata, nblocks)

	// SIMD and scalar paths may accumulate floats in different order,
	// producing slightly different results. Use tolerance.
	absDiff := math.Abs(float64(got - want))
	relErr := float64(0)
	if want != 0 {
		relErr = absDiff / math.Abs(float64(want))
	}
	if relErr > 1e-4 && absDiff > 0.1 {
		t.Errorf("%s: dispatch %f != fallback %f (relErr=%.6f, absDiff=%.6f)", name, got, want, relErr, absDiff)
	}
}

func TestVecDotQ4_KQ8_K_DispatchVsFallback(t *testing.T) {
	testVecDotKQuantDispatchVsFallback(t, TypeQ4_K, VecDotQ4_KQ8_K, BaseVecDotQ4_KQ8_K_fallback, "Q4_K")
}

func TestVecDotQ6_KQ8_K_DispatchVsFallback(t *testing.T) {
	testVecDotKQuantDispatchVsFallback(t, TypeQ6_K, VecDotQ6_KQ8_K, BaseVecDotQ6_KQ8_K_fallback, "Q6_K")
}

func TestVecDotQ2_KQ8_K_DispatchVsFallback(t *testing.T) {
	testVecDotKQuantDispatchVsFallback(t, TypeQ2_K, VecDotQ2_KQ8_K, BaseVecDotQ2_KQ8_K_fallback, "Q2_K")
}

func TestVecDotQ3_KQ8_K_DispatchVsFallback(t *testing.T) {
	testVecDotKQuantDispatchVsFallback(t, TypeQ3_K, VecDotQ3_KQ8_K, BaseVecDotQ3_KQ8_K_fallback, "Q3_K")
}

func TestVecDotQ5_KQ8_K_DispatchVsFallback(t *testing.T) {
	testVecDotKQuantDispatchVsFallback(t, TypeQ5_K, VecDotQ5_KQ8_K, BaseVecDotQ5_KQ8_K_fallback, "Q5_K")
}

func benchmarkVecDotKQuant(b *testing.B, qt QuantType, vecDot func([]uint8, []uint8, int) float32, name string) {
	b.Helper()

	blockSize := BytesPerBlock(qt)
	for _, nblocks := range []int{1, 4, 16, 64} {
		wdata, _ := makeTestWeightData(qt, nblocks)
		adata, _ := makeTestActivationData(nblocks)

		b.Run(fmt.Sprintf("%s/blocks=%d", name, nblocks), func(b *testing.B) {
			b.SetBytes(int64(nblocks * (blockSize + BlockSizeQ8K)))
			for range b.N {
				vecDot(wdata, adata, nblocks)
			}
		})
	}
}

func BenchmarkVecDotQ4_KQ8_K(b *testing.B) {
	benchmarkVecDotKQuant(b, TypeQ4_K, VecDotQ4_KQ8_K, "Q4_K")
}

func BenchmarkVecDotQ6_KQ8_K(b *testing.B) {
	benchmarkVecDotKQuant(b, TypeQ6_K, VecDotQ6_KQ8_K, "Q6_K")
}

func BenchmarkVecDotQ2_KQ8_K(b *testing.B) {
	benchmarkVecDotKQuant(b, TypeQ2_K, VecDotQ2_KQ8_K, "Q2_K")
}

func BenchmarkVecDotQ3_KQ8_K(b *testing.B) {
	benchmarkVecDotKQuant(b, TypeQ3_K, VecDotQ3_KQ8_K, "Q3_K")
}

func BenchmarkVecDotQ5_KQ8_K(b *testing.B) {
	benchmarkVecDotKQuant(b, TypeQ5_K, VecDotQ5_KQ8_K, "Q5_K")
}
