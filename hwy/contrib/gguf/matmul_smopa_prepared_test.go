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

// testPreparedVsVecdot compares PreparedGGUFMatMul output against the vecdot
// reference implementation. Both paths quantize activations to Q8_K and compute
// the same dot products, so results should match within float32 tolerance.
func testPreparedVsVecdot(t *testing.T, qt QuantType, M, K, N int, name string) {
	t.Helper()

	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	weights, input := makeTestMatMulData(qt, M, K, N)

	// Compute reference via vecdot.
	want := make([]float32, M*N)
	vecdotGGUFMatMul(input, weights, want, M, K, N, qt)

	// Prepare weights and compute via 4-tile prepared path.
	pw := PrepareWeights(weights, K, N, qt)
	if pw == nil {
		t.Fatalf("PrepareWeights returned nil for %v", qt)
	}

	got := make([]float32, M*N)
	smePreparedGGUFMatMul(input, pw, got, M)

	// Compare with tolerance.
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
			t.Errorf("%s output[%d] (m=%d,n=%d): prepared %f != vecdot %f (relErr=%.4f, absDiff=%.4f)",
				name, i, i/N, i%N, got[i], want[i], relErr, absDiff)
		}
	}
	t.Logf("%s [M=%d,K=%d,N=%d]: maxRelErr=%.6f, maxAbsDiff=%.6f", name, M, K, N, maxRelErr, maxAbsDiff)
}

func TestPreparedMatMul_Q4_K(t *testing.T) {
	testPreparedVsVecdot(t, TypeQ4_K, 2, QK_K, 3, "Q4_K_small")
	testPreparedVsVecdot(t, TypeQ4_K, 4, QK_K*2, 16, "Q4_K_medium")
	testPreparedVsVecdot(t, TypeQ4_K, 16, QK_K, 64, "Q4_K_wide")
	testPreparedVsVecdot(t, TypeQ4_K, 32, QK_K, 64, "Q4_K_batch")
	testPreparedVsVecdot(t, TypeQ4_K, 48, QK_K*2, 100, "Q4_K_unaligned")
}

func TestPreparedMatMul_Q5_K(t *testing.T) {
	testPreparedVsVecdot(t, TypeQ5_K, 2, QK_K, 3, "Q5_K_small")
	testPreparedVsVecdot(t, TypeQ5_K, 16, QK_K*2, 32, "Q5_K_medium")
	testPreparedVsVecdot(t, TypeQ5_K, 32, QK_K, 64, "Q5_K_wide")
}

func TestPreparedMatMul_Q6_K(t *testing.T) {
	testPreparedVsVecdot(t, TypeQ6_K, 2, QK_K, 3, "Q6_K_small")
	testPreparedVsVecdot(t, TypeQ6_K, 16, QK_K*2, 32, "Q6_K_medium")
	testPreparedVsVecdot(t, TypeQ6_K, 32, QK_K, 64, "Q6_K_wide")
}

func TestPreparedMatMul_Q2_K(t *testing.T) {
	testPreparedVsVecdot(t, TypeQ2_K, 2, QK_K, 3, "Q2_K_small")
	testPreparedVsVecdot(t, TypeQ2_K, 16, QK_K*2, 32, "Q2_K_medium")
	testPreparedVsVecdot(t, TypeQ2_K, 32, QK_K, 64, "Q2_K_wide")
}

func TestPreparedMatMul_Q3_K(t *testing.T) {
	testPreparedVsVecdot(t, TypeQ3_K, 2, QK_K, 3, "Q3_K_small")
	testPreparedVsVecdot(t, TypeQ3_K, 16, QK_K*2, 32, "Q3_K_medium")
	testPreparedVsVecdot(t, TypeQ3_K, 32, QK_K, 64, "Q3_K_wide")
}

func TestPreparedMatMul_NonAligned(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	// Test N not multiple of 64 (partial N-tile-group).
	for _, qt := range []QuantType{TypeQ4_K, TypeQ6_K, TypeQ2_K, TypeQ3_K, TypeQ5_K} {
		testPreparedVsVecdot(t, qt, 2, QK_K, 7, fmt.Sprintf("N=7_%d", qt))
		testPreparedVsVecdot(t, qt, 2, QK_K, 50, fmt.Sprintf("N=50_%d", qt))
	}

	// Test M not multiple of 16.
	for _, qt := range []QuantType{TypeQ4_K, TypeQ6_K} {
		testPreparedVsVecdot(t, qt, 5, QK_K, 16, fmt.Sprintf("M=5_%d", qt))
		testPreparedVsVecdot(t, qt, 17, QK_K, 32, fmt.Sprintf("M=17_%d", qt))
	}
}

func TestPreparedMatMul_LargerDims(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}
	testPreparedVsVecdot(t, TypeQ4_K, 16, QK_K*4, 64, "Q4_K_K1024_N64")
	testPreparedVsVecdot(t, TypeQ4_K, 32, QK_K*4, 128, "Q4_K_K1024_N128")
	testPreparedVsVecdot(t, TypeQ6_K, 16, QK_K*4, 64, "Q6_K_K1024_N64")
}

// testPreparedVsSMOPA verifies that the prepared path matches the existing
// single-tile SMOPA path exactly (same accumulation order).
func testPreparedVsSMOPA(t *testing.T, qt QuantType, M, K, N int, name string) {
	t.Helper()

	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	weights, input := makeTestMatMulData(qt, M, K, N)

	// Compute via single-tile SMOPA.
	want := make([]float32, M*N)
	smeGGUFMatMul(input, weights, want, M, K, N, qt)

	// Compute via 4-tile prepared SMOPA.
	pw := PrepareWeights(weights, K, N, qt)
	got := make([]float32, M*N)
	smePreparedGGUFMatMul(input, pw, got, M)

	// Both paths use SMOPA but different tile counts, so float accumulation
	// order should still be identical (same sub-block iteration order).
	maxRelErr := float64(0)
	for i := range got {
		absDiff := math.Abs(float64(got[i] - want[i]))
		relErr := float64(0)
		if want[i] != 0 {
			relErr = absDiff / math.Abs(float64(want[i]))
		}
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		if relErr > 0.001 && absDiff > 0.01 {
			t.Errorf("%s output[%d] (m=%d,n=%d): prepared %f != smopa %f (relErr=%.6f)",
				name, i, i/N, i%N, got[i], want[i], relErr)
		}
	}
	t.Logf("%s [M=%d,K=%d,N=%d]: maxRelErr=%.8f (prepared vs single-tile)", name, M, K, N, maxRelErr)
}

func TestPreparedVsSMOPA_Q4_K(t *testing.T) {
	testPreparedVsSMOPA(t, TypeQ4_K, 16, QK_K, 64, "Q4_K")
	testPreparedVsSMOPA(t, TypeQ4_K, 32, QK_K*4, 128, "Q4_K_large")
}

func TestPreparedVsSMOPA_Q6_K(t *testing.T) {
	testPreparedVsSMOPA(t, TypeQ6_K, 16, QK_K, 64, "Q6_K")
	testPreparedVsSMOPA(t, TypeQ6_K, 32, QK_K*4, 64, "Q6_K_large")
}

func TestParallelPreparedMatMul_Q4_K(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	pool := workerpool.New(runtime.GOMAXPROCS(0))
	defer pool.Close()

	M, K, N := 16, QK_K*2, 128
	weights, input := makeTestMatMulData(TypeQ4_K, M, K, N)
	pw := PrepareWeights(weights, K, N, TypeQ4_K)

	serial := make([]float32, M*N)
	parallel := make([]float32, M*N)

	smePreparedGGUFMatMul(input, pw, serial, M)
	parallelSMEPreparedGGUFMatMul(pool, input, pw, parallel, M)

	for i := range serial {
		if serial[i] != parallel[i] {
			t.Errorf("output[%d]: serial %f != parallel %f", i, serial[i], parallel[i])
		}
	}
}

// TestPrepareWeights_PanelCorrectness validates that pre-packed panels match
// scalar extraction for each quant type.
func TestPrepareWeights_PanelCorrectness(t *testing.T) {
	for _, qt := range []QuantType{TypeQ4_K, TypeQ5_K, TypeQ6_K, TypeQ2_K, TypeQ3_K} {
		t.Run(fmt.Sprintf("qt=%d", qt), func(t *testing.T) {
			K := QK_K
			N := 70 // not a multiple of 64, tests partial group
			nblocks := K / QK_K
			blockSize := BytesPerBlock(qt)
			wRowBytes := nblocks * blockSize
			info := GetSubBlockInfo(qt)

			weights := make([]uint8, N*wRowBytes)
			for i := range weights {
				weights[i] = uint8((i*7 + 13) % 256)
			}
			// Set valid d values.
			for n := range N {
				off := n * wRowBytes
				switch qt {
				case TypeQ4_K, TypeQ5_K:
					weights[off] = fp16One[0]
					weights[off+1] = fp16One[1]
				case TypeQ6_K:
					weights[off+208] = fp16One[0]
					weights[off+209] = fp16One[1]
				case TypeQ2_K:
					weights[off+16] = fp16One[0]
					weights[off+17] = fp16One[1]
				case TypeQ3_K:
					weights[off+108] = fp16One[0]
					weights[off+109] = fp16One[1]
				}
			}

			pw := PrepareWeights(weights, K, N, qt)
			if pw == nil {
				t.Fatal("PrepareWeights returned nil")
			}

			// Verify panels match scalar extraction.
			var unsignedBuf [32]uint8
			var signedBuf [32]int8
			kGroups := info.SubBlockSize / 4

			for ng := range pw.NTileGroups {
				for kb := range nblocks {
					for j := range info.NumSubBlocks {
						panelOff := pw.PanelOffset(ng, kb, j)

						for tile := range nTilesPerGroup {
							for col := range 16 {
								n := ng*64 + tile*16 + col
								if n >= N {
									continue
								}
								wBlock := weights[n*wRowBytes+kb*blockSize : n*wRowBytes+(kb+1)*blockSize]

								if info.Signed {
									extractSignedSubBlock(qt, wBlock, j, signedBuf[:info.SubBlockSize])
									for k4 := range kGroups {
										for g := range 4 {
											got := int8(pw.Panels[panelOff+k4*256+tile*64+col*4+g])
											want := signedBuf[k4*4+g]
											if got != want {
												t.Errorf("ng=%d kb=%d j=%d tile=%d col=%d k4=%d g=%d: got %d, want %d",
													ng, kb, j, tile, col, k4, g, got, want)
											}
										}
									}
								} else {
									extractUnsignedSubBlock(qt, wBlock, j, unsignedBuf[:info.SubBlockSize])
									for k4 := range kGroups {
										for g := range 4 {
											got := pw.Panels[panelOff+k4*256+tile*64+col*4+g]
											want := unsignedBuf[k4*4+g]
											if got != want {
												t.Errorf("ng=%d kb=%d j=%d tile=%d col=%d k4=%d g=%d: got %d, want %d",
													ng, kb, j, tile, col, k4, g, got, want)
											}
										}
									}
								}
							}
						}
					}
				}
			}
		})
	}
}

// TestPrepareWeights_ScaleCorrectness validates that pre-computed scales match
// parseBlockMeta output.
func TestPrepareWeights_ScaleCorrectness(t *testing.T) {
	for _, qt := range []QuantType{TypeQ4_K, TypeQ5_K, TypeQ6_K, TypeQ2_K, TypeQ3_K} {
		t.Run(fmt.Sprintf("qt=%d", qt), func(t *testing.T) {
			K := QK_K
			N := 70
			nblocks := K / QK_K
			blockSize := BytesPerBlock(qt)
			wRowBytes := nblocks * blockSize
			info := GetSubBlockInfo(qt)

			weights := make([]uint8, N*wRowBytes)
			for i := range weights {
				weights[i] = uint8((i*7 + 13) % 256)
			}
			for n := range N {
				off := n * wRowBytes
				switch qt {
				case TypeQ4_K, TypeQ5_K:
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
				}
			}

			pw := PrepareWeights(weights, K, N, qt)
			if pw == nil {
				t.Fatal("PrepareWeights returned nil")
			}

			for ng := range pw.NTileGroups {
				for kb := range nblocks {
					scaleKBOff := pw.ScaleOffset(ng, kb)

					for tile := range nTilesPerGroup {
						for col := range 16 {
							n := ng*64 + tile*16 + col
							if n >= N {
								continue
							}
							wBlock := weights[n*wRowBytes+kb*blockSize : n*wRowBytes+(kb+1)*blockSize]
							var meta blockMeta
							parseBlockMeta(qt, wBlock, &meta)

							for j := range info.NumSubBlocks {
								scaleSubOff := scaleKBOff + j*nColsPerGroup
								gotScale := pw.Scales[scaleSubOff+tile*16+col]
								wantScale := meta.d * meta.scs[j]
								if gotScale != wantScale {
									t.Errorf("scale ng=%d kb=%d j=%d tile=%d col=%d: got %f, want %f",
										ng, kb, j, tile, col, gotScale, wantScale)
								}

								if !info.Signed {
									minKBOff := pw.MinOffset(ng, kb)
									minSubOff := minKBOff + j*nColsPerGroup
									gotMin := pw.Scales[minSubOff+tile*16+col]
									wantMin := meta.dmin * meta.mns[j]
									if gotMin != wantMin {
										t.Errorf("min ng=%d kb=%d j=%d tile=%d col=%d: got %f, want %f",
											ng, kb, j, tile, col, gotMin, wantMin)
									}
								}
							}
						}
					}
				}
			}
		})
	}
}

func BenchmarkPreparedMatMul_Q4_K(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	for _, tc := range []struct {
		M, K, N int
	}{
		{1, QK_K, 256},
		{16, QK_K * 4, 256},
		{32, QK_K * 4, 256},
		{32, QK_K * 16, 256},
		{32, QK_K * 16, 4096},
	} {
		weights, input := makeTestMatMulData(TypeQ4_K, tc.M, tc.K, tc.N)
		pw := PrepareWeights(weights, tc.K, tc.N, TypeQ4_K)
		outputPrepared := make([]float32, tc.M*tc.N)
		outputSMOPA := make([]float32, tc.M*tc.N)
		outputVecdot := make([]float32, tc.M*tc.N)
		ops := 2 * int64(tc.M) * int64(tc.K) * int64(tc.N)

		b.Run(fmt.Sprintf("Prepared/M=%d_K=%d_N=%d", tc.M, tc.K, tc.N), func(b *testing.B) {
			b.SetBytes(ops)
			for range b.N {
				smePreparedGGUFMatMul(input, pw, outputPrepared, tc.M)
			}
		})

		b.Run(fmt.Sprintf("SMOPA/M=%d_K=%d_N=%d", tc.M, tc.K, tc.N), func(b *testing.B) {
			b.SetBytes(ops)
			for range b.N {
				smeGGUFMatMul(input, weights, outputSMOPA, tc.M, tc.K, tc.N, TypeQ4_K)
			}
		})

		b.Run(fmt.Sprintf("Vecdot/M=%d_K=%d_N=%d", tc.M, tc.K, tc.N), func(b *testing.B) {
			b.SetBytes(ops)
			for range b.N {
				vecdotGGUFMatMul(input, weights, outputVecdot, tc.M, tc.K, tc.N, TypeQ4_K)
			}
		})
	}
}

func BenchmarkPreparedMatMul_Parallel_Q4_K(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	pool := workerpool.New(runtime.GOMAXPROCS(0))
	defer pool.Close()

	for _, tc := range []struct {
		M, K, N int
	}{
		{32, QK_K * 16, 4096},
	} {
		weights, input := makeTestMatMulData(TypeQ4_K, tc.M, tc.K, tc.N)
		pw := PrepareWeights(weights, tc.K, tc.N, TypeQ4_K)
		output := make([]float32, tc.M*tc.N)
		ops := 2 * int64(tc.M) * int64(tc.K) * int64(tc.N)

		b.Run(fmt.Sprintf("Prepared/M=%d_K=%d_N=%d", tc.M, tc.K, tc.N), func(b *testing.B) {
			b.SetBytes(ops)
			for range b.N {
				parallelSMEPreparedGGUFMatMul(pool, input, pw, output, tc.M)
			}
		})

		b.Run(fmt.Sprintf("SMOPA/M=%d_K=%d_N=%d", tc.M, tc.K, tc.N), func(b *testing.B) {
			b.SetBytes(ops)
			for range b.N {
				parallelSMEGGUFMatMul(pool, input, weights, output, tc.M, tc.K, tc.N, TypeQ4_K)
			}
		})
	}
}
