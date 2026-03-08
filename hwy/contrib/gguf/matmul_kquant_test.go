package gguf

import (
	"math"
	"runtime"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func testGGUFMatMulKQuant(t *testing.T, qt QuantType, name string) {
	t.Helper()

	M, K, N := 2, QK_K, 3
	nblocks := K / QK_K

	blockSize := BytesPerBlock(qt)

	// Create weight data with valid structure [N, K].
	wdata := make([]uint8, N*nblocks*blockSize)
	for i := range wdata {
		wdata[i] = uint8((i*7 + 13) % 256)
	}

	// Set valid scales at block boundaries for each weight row.
	for n := range N {
		for b := range nblocks {
			off := n*nblocks*blockSize + b*blockSize
			switch qt {
			case TypeQ4_K:
				wdata[off] = fp16One[0]
				wdata[off+1] = fp16One[1]
				wdata[off+2] = 0x00
				wdata[off+3] = 0x38
			case TypeQ6_K:
				wdata[off+208] = fp16One[0]
				wdata[off+209] = fp16One[1]
			case TypeQ2_K:
				wdata[off+16] = fp16One[0]
				wdata[off+17] = fp16One[1]
				wdata[off+18] = 0x00
				wdata[off+19] = 0x38
			case TypeQ3_K:
				wdata[off+108] = fp16One[0]
				wdata[off+109] = fp16One[1]
			case TypeQ5_K:
				wdata[off] = fp16One[0]
				wdata[off+1] = fp16One[1]
				wdata[off+2] = 0x00
				wdata[off+3] = 0x38
			}
		}
	}

	// Dequantize weights for reference [N, K].
	wFloat := make([]float32, N*K)
	for n := range N {
		wRow := wdata[n*nblocks*blockSize : (n+1)*nblocks*blockSize]
		wOut := wFloat[n*K : (n+1)*K]
		switch qt {
		case TypeQ4_K:
			DequantizeQ4K(wRow, wOut)
		case TypeQ6_K:
			DequantizeQ6K(wRow, wOut)
		case TypeQ2_K:
			DequantizeQ2K(wRow, wOut)
		case TypeQ3_K:
			DequantizeQ3K(wRow, wOut)
		case TypeQ5_K:
			DequantizeQ5K(wRow, wOut)
		}
	}

	// Create float32 input [M, K].
	input := make([]float32, M*K)
	for i := range input {
		input[i] = float32(i%64-32) * 0.01
	}

	// Reference matmul.
	want := make([]float32, M*N)
	referenceMatMul(input, wFloat, want, M, K, N)

	// GGUF matmul.
	got := make([]float32, M*N)
	GGUFMatMul(input, wdata, got, M, K, N, qt)

	// Compare with tolerance for double quantization error.
	for i := range got {
		absDiff := math.Abs(float64(got[i] - want[i]))
		relErr := float64(0)
		if want[i] != 0 {
			relErr = absDiff / math.Abs(float64(want[i]))
		}
		if relErr > 0.10 && absDiff > 1.0 {
			t.Errorf("%s output[%d]: got %f, want %f (relErr=%.4f, absDiff=%.4f)",
				name, i, got[i], want[i], relErr, absDiff)
		}
	}
}

func TestGGUFMatMul_Q4_K(t *testing.T) {
	testGGUFMatMulKQuant(t, TypeQ4_K, "Q4_K")
}

func TestGGUFMatMul_Q6_K(t *testing.T) {
	testGGUFMatMulKQuant(t, TypeQ6_K, "Q6_K")
}

func TestGGUFMatMul_Q2_K(t *testing.T) {
	testGGUFMatMulKQuant(t, TypeQ2_K, "Q2_K")
}

func TestGGUFMatMul_Q3_K(t *testing.T) {
	testGGUFMatMulKQuant(t, TypeQ3_K, "Q3_K")
}

func TestGGUFMatMul_Q5_K(t *testing.T) {
	testGGUFMatMulKQuant(t, TypeQ5_K, "Q5_K")
}

func TestParallelGGUFMatMul_Q4_K(t *testing.T) {
	pool := workerpool.New(runtime.GOMAXPROCS(0))
	defer pool.Close()

	M, K, N := 4, QK_K, 4
	nblocks := K / QK_K

	wdata := make([]uint8, N*nblocks*BlockSizeQ4K)
	for i := range wdata {
		wdata[i] = uint8((i*7 + 13) % 256)
	}
	for n := range N {
		for b := range nblocks {
			off := n*nblocks*BlockSizeQ4K + b*BlockSizeQ4K
			wdata[off] = fp16One[0]
			wdata[off+1] = fp16One[1]
			wdata[off+2] = 0x00
			wdata[off+3] = 0x38
		}
	}

	input := make([]float32, M*K)
	for i := range input {
		input[i] = float32(i%64-32) * 0.01
	}

	serial := make([]float32, M*N)
	parallel := make([]float32, M*N)

	GGUFMatMul(input, wdata, serial, M, K, N, TypeQ4_K)
	ParallelGGUFMatMul(pool, input, wdata, parallel, M, K, N, TypeQ4_K)

	for i := range serial {
		if serial[i] != parallel[i] {
			t.Errorf("output[%d]: serial %f != parallel %f", i, serial[i], parallel[i])
		}
	}
}
