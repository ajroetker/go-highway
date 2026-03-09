// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !noasm && darwin && arm64

package gguf

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// makeTestMatMulData creates consistent weight and input data for matmul tests.
func makeTestMatMulData(qt QuantType, M, K, N int) (weights []uint8, input []float32) {
	nblocks := K / QK_K
	blockSize := BytesPerBlock(qt)
	wRowBytes := nblocks * blockSize

	weights = make([]uint8, N*wRowBytes)
	for i := range weights {
		weights[i] = uint8((i*7 + 13) % 256)
	}

	// Set valid scales at block boundaries for each weight row.
	for n := range N {
		for b := range nblocks {
			off := n*wRowBytes + b*blockSize
			switch qt {
			case TypeQ4_K:
				weights[off] = fp16One[0]
				weights[off+1] = fp16One[1]
				weights[off+2] = 0x00
				weights[off+3] = 0x38
			case TypeQ6_K:
				weights[off+208] = fp16One[0]
				weights[off+209] = fp16One[1]
			case TypeQ2_K:
				weights[off+16] = fp16One[0]
				weights[off+17] = fp16One[1]
				weights[off+18] = 0x00
				weights[off+19] = 0x38
			case TypeQ3_K:
				weights[off+108] = fp16One[0]
				weights[off+109] = fp16One[1]
			case TypeQ5_K:
				weights[off] = fp16One[0]
				weights[off+1] = fp16One[1]
				weights[off+2] = 0x00
				weights[off+3] = 0x38
			}
		}
	}

	input = make([]float32, M*K)
	for i := range input {
		input[i] = float32(i%64-32) * 0.01
	}

	return weights, input
}

// testSMOPAvsVecdot compares the SMOPA matmul against the vecdot fallback.
func testSMOPAvsVecdot(t *testing.T, qt QuantType, M, K, N int, name string) {
	t.Helper()

	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	weights, input := makeTestMatMulData(qt, M, K, N)

	// Compute via vecdot fallback.
	want := make([]float32, M*N)
	vecdotGGUFMatMul(input, weights, want, M, K, N, qt)

	// Compute via SMOPA.
	got := make([]float32, M*N)
	smeGGUFMatMul(input, weights, got, M, K, N, qt)

	// Compare with tolerance (double quantization + float accumulation order).
	maxRelErr := float64(0)
	maxAbsDiff := float64(0)
	for i := range got {
		absDiff := math.Abs(float64(got[i] - want[i]))
		relErr := float64(0)
		if want[i] != 0 {
			relErr = absDiff / math.Abs(float64(want[i]))
		}
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		if absDiff > maxAbsDiff {
			maxAbsDiff = absDiff
		}
		if relErr > 0.05 && absDiff > 0.5 {
			t.Errorf("%s output[%d] (m=%d,n=%d): SMOPA %f != vecdot %f (relErr=%.4f, absDiff=%.4f)",
				name, i, i/N, i%N, got[i], want[i], relErr, absDiff)
		}
	}
	t.Logf("%s [M=%d,K=%d,N=%d]: maxRelErr=%.6f, maxAbsDiff=%.6f", name, M, K, N, maxRelErr, maxAbsDiff)
}

func TestSMEGGUFMatMul_Q4_K(t *testing.T) {
	testSMOPAvsVecdot(t, TypeQ4_K, 2, QK_K, 3, "Q4_K")
	testSMOPAvsVecdot(t, TypeQ4_K, 4, QK_K*2, 16, "Q4_K_larger")
	testSMOPAvsVecdot(t, TypeQ4_K, 32, QK_K, 64, "Q4_K_wide")
}

func TestSMEGGUFMatMul_Q6_K(t *testing.T) {
	testSMOPAvsVecdot(t, TypeQ6_K, 2, QK_K, 3, "Q6_K")
	testSMOPAvsVecdot(t, TypeQ6_K, 4, QK_K*2, 16, "Q6_K_larger")
	testSMOPAvsVecdot(t, TypeQ6_K, 32, QK_K, 64, "Q6_K_wide")
}

func TestSMEGGUFMatMul_Q2_K(t *testing.T) {
	testSMOPAvsVecdot(t, TypeQ2_K, 2, QK_K, 3, "Q2_K")
	testSMOPAvsVecdot(t, TypeQ2_K, 4, QK_K*2, 16, "Q2_K_larger")
	testSMOPAvsVecdot(t, TypeQ2_K, 32, QK_K, 64, "Q2_K_wide")
}

func TestSMEGGUFMatMul_Q3_K(t *testing.T) {
	testSMOPAvsVecdot(t, TypeQ3_K, 2, QK_K, 3, "Q3_K")
	testSMOPAvsVecdot(t, TypeQ3_K, 4, QK_K*2, 16, "Q3_K_larger")
	testSMOPAvsVecdot(t, TypeQ3_K, 32, QK_K, 64, "Q3_K_wide")
}

func TestSMEGGUFMatMul_Q5_K(t *testing.T) {
	testSMOPAvsVecdot(t, TypeQ5_K, 2, QK_K, 3, "Q5_K")
	testSMOPAvsVecdot(t, TypeQ5_K, 4, QK_K*2, 16, "Q5_K_larger")
	testSMOPAvsVecdot(t, TypeQ5_K, 32, QK_K, 64, "Q5_K_wide")
}

func TestSMEGGUFMatMul_NonAligned(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	// Test N not a multiple of 16.
	for _, qt := range []QuantType{TypeQ4_K, TypeQ6_K, TypeQ2_K, TypeQ3_K, TypeQ5_K} {
		name := fmt.Sprintf("N=7_%d", qt)
		testSMOPAvsVecdot(t, qt, 2, QK_K, 7, name)
	}

	// Test M not a multiple of 16.
	for _, qt := range []QuantType{TypeQ4_K, TypeQ6_K} {
		name := fmt.Sprintf("M=5_%d", qt)
		testSMOPAvsVecdot(t, qt, 5, QK_K, 16, name)
	}
}

func TestSMEGGUFMatMul_LargerDims(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}
	// Larger workload: M=16, K=1024, N=64.
	testSMOPAvsVecdot(t, TypeQ4_K, 16, QK_K*4, 64, "Q4_K_large")
	testSMOPAvsVecdot(t, TypeQ6_K, 16, QK_K*4, 64, "Q6_K_large")
}

func TestParallelSMEGGUFMatMul_Q4_K(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	pool := workerpool.New(runtime.GOMAXPROCS(0))
	defer pool.Close()

	M, K, N := 8, QK_K*2, 32
	weights, input := makeTestMatMulData(TypeQ4_K, M, K, N)

	serial := make([]float32, M*N)
	parallel := make([]float32, M*N)

	smeGGUFMatMul(input, weights, serial, M, K, N, TypeQ4_K)
	parallelSMEGGUFMatMul(pool, input, weights, parallel, M, K, N, TypeQ4_K)

	for i := range serial {
		if serial[i] != parallel[i] {
			t.Errorf("output[%d]: serial %f != parallel %f", i, serial[i], parallel[i])
		}
	}
}

func BenchmarkSMEGGUFMatMul_Q4_K(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	for _, tc := range []struct {
		M, K, N int
	}{
		{1, QK_K, 256},
		{4, QK_K, 256},
		{16, QK_K * 4, 256},
		{32, QK_K * 4, 256},
		{32, QK_K * 16, 256},
	} {
		weights, input := makeTestMatMulData(TypeQ4_K, tc.M, tc.K, tc.N)
		output := make([]float32, tc.M*tc.N)

		b.Run(fmt.Sprintf("SMOPA/M=%d_K=%d_N=%d", tc.M, tc.K, tc.N), func(b *testing.B) {
			ops := 2 * int64(tc.M) * int64(tc.K) * int64(tc.N)
			b.SetBytes(ops) // for throughput reporting
			for range b.N {
				smeGGUFMatMul(input, weights, output, tc.M, tc.K, tc.N, TypeQ4_K)
			}
		})

		b.Run(fmt.Sprintf("Vecdot/M=%d_K=%d_N=%d", tc.M, tc.K, tc.N), func(b *testing.B) {
			ops := 2 * int64(tc.M) * int64(tc.K) * int64(tc.N)
			b.SetBytes(ops)
			for range b.N {
				vecdotGGUFMatMul(input, weights, output, tc.M, tc.K, tc.N, TypeQ4_K)
			}
		})
	}
}
