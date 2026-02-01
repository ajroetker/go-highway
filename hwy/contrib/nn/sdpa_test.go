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

package nn

import (
	"fmt"
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/nn/asm"
)

func TestSDPAAuto(t *testing.T) {
	tests := []struct {
		name    string
		seqLen  int
		kvLen   int
		headDim int
		useMask bool
	}{
		{"1x1x32/no_mask", 1, 1, 32, false},
		{"4x4x32/no_mask", 4, 4, 32, false},
		{"4x4x32/mask", 4, 4, 32, true},
		{"8x8x64/no_mask", 8, 8, 64, false},
		{"8x16x64/no_mask", 8, 16, 64, false}, // seqLen != kvLen
		{"16x16x128/no_mask", 16, 16, 128, false},
		{"3x5x7/no_mask", 3, 5, 7, false}, // non-aligned
		{"32x32x64/mask", 32, 32, 64, true},
		{"64x64x64/no_mask", 64, 64, 64, false},
		{"128x128x64/no_mask", 128, 128, 64, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scale := float32(1.0 / stdmath.Sqrt(float64(tt.headDim)))
			q := make([]float32, tt.seqLen*tt.headDim)
			k := make([]float32, tt.kvLen*tt.headDim)
			v := make([]float32, tt.kvLen*tt.headDim)

			for i := range q {
				q[i] = float32(i)*0.01 - 0.5
			}
			for i := range k {
				k[i] = float32(i)*0.008 - 0.4
			}
			for i := range v {
				v[i] = float32(i)*0.006 - 0.3
			}

			var mask []float32
			if tt.useMask {
				mask = make([]float32, tt.seqLen*tt.kvLen)
				for i := range mask {
					mask[i] = float32(i%3) * -0.1
				}
			}

			autoOutput := make([]float32, tt.seqLen*tt.headDim)
			scalarOutput := make([]float32, tt.seqLen*tt.headDim)
			scalarScores := make([]float32, tt.seqLen*tt.kvLen)

			SDPAAuto(q, k, v, mask, autoOutput, tt.seqLen, tt.kvLen, tt.headDim, scale)
			SDPAScalar(q, k, v, mask, scalarScores, scalarOutput, tt.seqLen, tt.kvLen, tt.headDim, scale)

			for i := range autoOutput {
				diff := stdmath.Abs(float64(autoOutput[i] - scalarOutput[i]))
				relTol := stdmath.Max(1e-3, 1e-3*stdmath.Abs(float64(scalarOutput[i])))
				if diff > relTol {
					t.Errorf("output[%d]: auto=%v, scalar=%v, diff=%v", i, autoOutput[i], scalarOutput[i], diff)
				}
			}
		})
	}
}

func TestSDPACausal(t *testing.T) {
	tests := []struct {
		name    string
		seqLen  int
		kvLen   int
		headDim int
	}{
		{"4x4x32", 4, 4, 32},
		{"8x8x64", 8, 8, 64},
		{"4x8x32", 4, 8, 32}, // kvLen > seqLen (prefix caching)
		{"16x16x64", 16, 16, 64},
		{"3x5x7", 3, 5, 7}, // non-aligned
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scale := float32(1.0 / stdmath.Sqrt(float64(tt.headDim)))
			q := make([]float32, tt.seqLen*tt.headDim)
			k := make([]float32, tt.kvLen*tt.headDim)
			v := make([]float32, tt.kvLen*tt.headDim)

			for i := range q {
				q[i] = float32(i)*0.01 - 0.5
			}
			for i := range k {
				k[i] = float32(i)*0.008 - 0.4
			}
			for i := range v {
				v[i] = float32(i)*0.006 - 0.3
			}

			autoOutput := make([]float32, tt.seqLen*tt.headDim)
			scalarOutput := make([]float32, tt.seqLen*tt.headDim)
			scalarScores := make([]float32, tt.seqLen*tt.kvLen)

			SDPACausalAuto(q, k, v, autoOutput, tt.seqLen, tt.kvLen, tt.headDim, scale)
			SDPACausalScalar(q, k, v, scalarScores, scalarOutput, tt.seqLen, tt.kvLen, tt.headDim, scale)

			for i := range autoOutput {
				diff := stdmath.Abs(float64(autoOutput[i] - scalarOutput[i]))
				relTol := stdmath.Max(1e-3, 1e-3*stdmath.Abs(float64(scalarOutput[i])))
				if diff > relTol {
					t.Errorf("output[%d]: auto=%v, scalar=%v, diff=%v", i, autoOutput[i], scalarOutput[i], diff)
				}
			}
		})
	}
}

func TestSDPACausalMasking(t *testing.T) {
	// Verify that causal attention prevents attending to future positions
	seqLen, kvLen, headDim := 4, 4, 4
	scale := float32(1.0 / stdmath.Sqrt(float64(headDim)))

	// Set all Q, K to same values so attention scores would be uniform without masking
	q := make([]float32, seqLen*headDim)
	k := make([]float32, kvLen*headDim)
	v := make([]float32, kvLen*headDim)

	for i := range q {
		q[i] = 0.5
	}
	for i := range k {
		k[i] = 0.5
	}
	// V is identity-like: each row has a unique value
	for i := range kvLen {
		for d := range headDim {
			v[i*headDim+d] = float32(i + 1)
		}
	}

	output := make([]float32, seqLen*headDim)
	SDPACausalAuto(q, k, v, output, seqLen, kvLen, headDim, scale)

	// Row 0 should only attend to position 0 -> output should be v[0,:] = 1.0
	for d := range headDim {
		if stdmath.Abs(float64(output[d]-1.0)) > 1e-3 {
			t.Errorf("row 0, dim %d: got %v, want ~1.0", d, output[d])
		}
	}

	// Row 1 should attend to positions 0-1 -> output is average of v[0,:] and v[1,:] = 1.5
	for d := range headDim {
		if stdmath.Abs(float64(output[headDim+d]-1.5)) > 1e-3 {
			t.Errorf("row 1, dim %d: got %v, want ~1.5", d, output[headDim+d])
		}
	}
}

func TestSDPAProperties(t *testing.T) {
	// Attention weights should sum to 1 per row
	seqLen, kvLen, headDim := 8, 8, 32
	scale := float32(1.0 / stdmath.Sqrt(float64(headDim)))

	q := make([]float32, seqLen*headDim)
	k := make([]float32, kvLen*headDim)
	v := make([]float32, kvLen*headDim)

	for i := range q {
		q[i] = float32(i)*0.01 - 0.5
	}
	for i := range k {
		k[i] = float32(i)*0.008 - 0.4
	}
	for i := range v {
		v[i] = float32(i)*0.006 - 0.3
	}

	scores := make([]float32, seqLen*kvLen)
	output := make([]float32, seqLen*headDim)

	SDPAScalar(q, k, v, nil, scores, output, seqLen, kvLen, headDim, scale)

	// Check scores are valid probability distributions
	for i := range seqLen {
		var rowSum float64
		for j := range kvLen {
			w := scores[i*kvLen+j]
			if w < 0 {
				t.Errorf("scores[%d,%d] = %v, want >= 0", i, j, w)
			}
			rowSum += float64(w)
		}
		if stdmath.Abs(rowSum-1.0) > 1e-5 {
			t.Errorf("row %d sum = %v, want ~1.0", i, rowSum)
		}
	}
}

func TestMultiHeadSDPA(t *testing.T) {
	batchSize := 2
	numHeads := 4
	numKVHeads := 2 // GQA: 2 heads per KV head
	seqLen := 8
	kvLen := 8
	headDim := 16
	scale := float32(1.0 / stdmath.Sqrt(float64(headDim)))

	qSize := batchSize * numHeads * seqLen * headDim
	kvSize := batchSize * numKVHeads * kvLen * headDim
	oSize := batchSize * numHeads * seqLen * headDim

	q := make([]float32, qSize)
	k := make([]float32, kvSize)
	v := make([]float32, kvSize)
	output := make([]float32, oSize)

	for i := range q {
		q[i] = float32(i)*0.01 - 0.5
	}
	for i := range k {
		k[i] = float32(i)*0.008 - 0.4
	}
	for i := range v {
		v[i] = float32(i)*0.006 - 0.3
	}

	MultiHeadSDPAAuto(nil, q, k, v, nil, output, batchSize, numHeads, numKVHeads,
		seqLen, kvLen, headDim, 0, 0, scale, false)

	// Basic sanity: no NaN or Inf
	for i, val := range output {
		if stdmath.IsNaN(float64(val)) || stdmath.IsInf(float64(val), 0) {
			t.Errorf("output[%d] = %v (NaN/Inf)", i, val)
		}
	}

	// GQA: heads 0 and 1 should share KV head 0, heads 2 and 3 should share KV head 1
	// Verify that query heads sharing a KV head produce different outputs
	// (they have different Q, same K/V)
	qHeadStride := seqLen * headDim
	head0 := output[:qHeadStride]
	head1 := output[qHeadStride : 2*qHeadStride]
	allSame := true
	for i := range head0 {
		if head0[i] != head1[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("GQA: heads 0 and 1 produced identical outputs (should differ due to different Q)")
	}
}

func TestMultiHeadSDPACausal(t *testing.T) {
	batchSize := 1
	numHeads := 2
	numKVHeads := 2
	seqLen := 4
	kvLen := 4
	headDim := 8
	scale := float32(1.0 / stdmath.Sqrt(float64(headDim)))

	qSize := batchSize * numHeads * seqLen * headDim
	kvSize := batchSize * numKVHeads * kvLen * headDim

	q := make([]float32, qSize)
	k := make([]float32, kvSize)
	v := make([]float32, kvSize)
	output := make([]float32, qSize)

	for i := range q {
		q[i] = float32(i)*0.01 - 0.5
	}
	for i := range k {
		k[i] = float32(i)*0.008 - 0.4
	}
	for i := range v {
		v[i] = float32(i)*0.006 - 0.3
	}

	MultiHeadSDPAAuto(nil, q, k, v, nil, output, batchSize, numHeads, numKVHeads,
		seqLen, kvLen, headDim, 0, 0, scale, true)

	for i, val := range output {
		if stdmath.IsNaN(float64(val)) || stdmath.IsInf(float64(val), 0) {
			t.Errorf("output[%d] = %v (NaN/Inf)", i, val)
		}
	}
}

func TestSDPAAuto64(t *testing.T) {
	seqLen, kvLen, headDim := 8, 8, 32
	scale := 1.0 / stdmath.Sqrt(float64(headDim))

	q := make([]float64, seqLen*headDim)
	k := make([]float64, kvLen*headDim)
	v := make([]float64, kvLen*headDim)

	for i := range q {
		q[i] = float64(i)*0.01 - 0.5
	}
	for i := range k {
		k[i] = float64(i)*0.008 - 0.4
	}
	for i := range v {
		v[i] = float64(i)*0.006 - 0.3
	}

	autoOutput := make([]float64, seqLen*headDim)
	scalarOutput := make([]float64, seqLen*headDim)
	scalarScores := make([]float64, seqLen*kvLen)

	SDPAAuto(q, k, v, nil, autoOutput, seqLen, kvLen, headDim, scale)
	SDPAScalar(q, k, v, nil, scalarScores, scalarOutput, seqLen, kvLen, headDim, scale)

	for i := range autoOutput {
		if stdmath.Abs(autoOutput[i]-scalarOutput[i]) > 1e-8 {
			t.Errorf("output[%d]: auto=%v, scalar=%v", i, autoOutput[i], scalarOutput[i])
		}
	}
}

// TestSDPASMEDirect calls the SME FMOPA assembly directly with trivial inputs
// to isolate numerical issues from the adapter layer.
func TestSDPASMEDirect(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("no SME support")
	}

	// 16x16 Q@KT with headDim=16, all ones → each score = 16 * scale
	seqLen, kvLen, headDim := 16, 16, 16
	scale := float32(1.0) / float32(headDim) // = 0.0625, so each score = 1.0

	q := make([]float32, seqLen*headDim)
	kt := make([]float32, headDim*kvLen) // already transposed [headDim, kvLen]
	v := make([]float32, kvLen*headDim)
	output := make([]float32, seqLen*headDim)

	for i := range q {
		q[i] = 1.0
	}
	for i := range kt {
		kt[i] = 1.0
	}
	// V[r, d] = float32(r) for all d
	for r := 0; r < kvLen; r++ {
		for d := 0; d < headDim; d++ {
			v[r*headDim+d] = float32(r)
		}
	}

	// With Q=1, KT=1: Q@KT = [[16,16,...],[16,16,...],...]
	// After scale (0.0625): scores = [[1,1,...],[1,1,...],...]
	// softmax([1,1,...,1]) = [1/16, 1/16, ..., 1/16]
	// output = (1/16) * sum(V rows) = (1/16) * (0+1+...+15) * ones = 7.5
	expected := float32(7.5)

	asm.SDPAFMOPAF32(q, kt, v, nil, output, seqLen, kvLen, headDim, scale)

	t.Logf("output[0]=%v, expected=%v", output[0], expected)
	t.Logf("output[15]=%v", output[15])
	t.Logf("first row: %v", output[:headDim])

	for i := range output {
		diff := stdmath.Abs(float64(output[i] - expected))
		if diff > 0.1 {
			t.Errorf("output[%d]=%v, want ~%v (diff=%v)", i, output[i], expected, diff)
			if i > 5 {
				break
			}
		}
	}

	// Now test with 32x32x64 (the failing case dimensions)
	seqLen2, kvLen2, headDim2 := 32, 32, 64
	scale2 := float32(1.0) / float32(headDim2) // 1/64 = 0.015625

	q2 := make([]float32, seqLen2*headDim2)
	kt2 := make([]float32, headDim2*kvLen2)
	v2 := make([]float32, kvLen2*headDim2)
	output2 := make([]float32, seqLen2*headDim2)

	for i := range q2 {
		q2[i] = 1.0
	}
	for i := range kt2 {
		kt2[i] = 1.0
	}
	for r := 0; r < kvLen2; r++ {
		for d := 0; d < headDim2; d++ {
			v2[r*headDim2+d] = float32(r)
		}
	}

	// Q@KT = 64 for all entries, scaled by 1/64 = 1.0, softmax = 1/32
	// output = (1/32) * sum(0..31) = (1/32) * 496 = 15.5
	expected2 := float32(15.5)

	asm.SDPAFMOPAF32(q2, kt2, v2, nil, output2, seqLen2, kvLen2, headDim2, scale2)

	t.Logf("32x32x64: output[0]=%v, expected=%v", output2[0], expected2)
	t.Logf("32x32x64: output[63]=%v", output2[63])
	t.Logf("32x32x64: first 8: %v", output2[:8])

	for i := range output2 {
		diff := stdmath.Abs(float64(output2[i] - expected2))
		if diff > 0.1 {
			t.Errorf("32x32x64: output[%d]=%v, want ~%v (diff=%v)", i, output2[i], expected2, diff)
			if i > 5 {
				break
			}
		}
	}

	// Test with actual test data but manual transpose (like the adapter does)
	seqLen3, kvLen3, headDim3 := 32, 32, 64
	scale3 := float32(1.0 / stdmath.Sqrt(float64(headDim3)))

	q3 := make([]float32, seqLen3*headDim3)
	k3 := make([]float32, kvLen3*headDim3)
	v3 := make([]float32, kvLen3*headDim3)
	for i := range q3 {
		q3[i] = float32(i)*0.01 - 0.5
	}
	for i := range k3 {
		k3[i] = float32(i)*0.008 - 0.4
	}
	for i := range v3 {
		v3[i] = float32(i)*0.006 - 0.3
	}

	// Manual transpose K [kvLen3, headDim3] → KT [headDim3, kvLen3]
	kt3 := make([]float32, headDim3*kvLen3)
	for i := 0; i < kvLen3; i++ {
		for j := 0; j < headDim3; j++ {
			kt3[j*kvLen3+i] = k3[i*headDim3+j]
		}
	}

	// Also test through adapter (Transpose2D)
	kt3b := make([]float32, headDim3*kvLen3)
	matmul.Transpose2D(k3, kvLen3, headDim3, kt3b)

	// Check transpose match
	for i := range kt3 {
		if kt3[i] != kt3b[i] {
			t.Errorf("transpose mismatch at %d: manual=%v, Transpose2D=%v", i, kt3[i], kt3b[i])
			break
		}
	}

	// Get reference from scalar
	scalarOutput3 := make([]float32, seqLen3*headDim3)
	scalarScores3 := make([]float32, seqLen3*kvLen3)
	SDPAScalar(q3, k3, v3, nil, scalarScores3, scalarOutput3, seqLen3, kvLen3, headDim3, scale3)

	// Call SME directly
	smeOutput3 := make([]float32, seqLen3*headDim3)
	asm.SDPAFMOPAF32(q3, kt3, v3, nil, smeOutput3, seqLen3, kvLen3, headDim3, scale3)

	t.Logf("direct SME: output[0]=%v, scalar=%v", smeOutput3[0], scalarOutput3[0])
	t.Logf("direct SME: output[64]=%v, scalar=%v", smeOutput3[64], scalarOutput3[64])

	// Call through adapter (SDPAAuto)
	autoOutput3 := make([]float32, seqLen3*headDim3)
	SDPAAuto(q3, k3, v3, nil, autoOutput3, seqLen3, kvLen3, headDim3, scale3)

	t.Logf("adapter: output[0]=%v, scalar=%v", autoOutput3[0], scalarOutput3[0])
	t.Logf("adapter: output[64]=%v, scalar=%v", autoOutput3[64], scalarOutput3[64])

	// Compare direct SME vs scalar
	for i := 0; i < 5; i++ {
		diff := stdmath.Abs(float64(smeOutput3[i] - scalarOutput3[i]))
		t.Logf("  [%d] sme=%v scalar=%v auto=%v diff_sme=%v diff_auto=%v",
			i, smeOutput3[i], scalarOutput3[i], autoOutput3[i],
			diff, stdmath.Abs(float64(autoOutput3[i]-scalarOutput3[i])))
	}

	// Diagnostic: test if Q row variation affects output (tests gather correctness)
	// Q[r, :] = r for all d, KT = 1, V[r, :] = r
	// Different Q rows produce different attention weights → different outputs
	seqLen4, kvLen4, headDim4 := 32, 32, 64
	scale4 := float32(1.0 / stdmath.Sqrt(float64(headDim4)))
	q4 := make([]float32, seqLen4*headDim4)
	kt4 := make([]float32, headDim4*kvLen4)
	v4 := make([]float32, kvLen4*headDim4)
	for r := 0; r < seqLen4; r++ {
		for d := 0; d < headDim4; d++ {
			q4[r*headDim4+d] = float32(r)
		}
	}
	for i := range kt4 {
		kt4[i] = 1.0
	}
	for r := 0; r < kvLen4; r++ {
		for d := 0; d < headDim4; d++ {
			v4[r*headDim4+d] = float32(r)
		}
	}

	smeOut4 := make([]float32, seqLen4*headDim4)
	asm.SDPAFMOPAF32(q4, kt4, v4, nil, smeOut4, seqLen4, kvLen4, headDim4, scale4)

	// Row 0: Q=0, all scores = 0, uniform softmax → avg(V) = 15.5
	// Row 31: Q=31, all scores = 31*64*scale = 31*64*0.125=248, uniform → 15.5
	// (Since KT=1, all kv positions get same score per row, so softmax is always uniform)
	// So all rows should produce 15.5 regardless of Q variation.
	// This doesn't test gather! Need asymmetric KT.
	t.Logf("symmetric KT: row0=%v row31=%v (expect 15.5)", smeOut4[0], smeOut4[31*headDim4])

	// Better test: make KT asymmetric
	// KT[dd=0, kj=0] = 1000, rest = 0
	// Q[r, d=0] = r, Q[r, d>0] = 0
	// Then score[r, kj=0] = r * 1000, score[r, kj>0] = 0
	// For r=0: all scores = 0, uniform → avg(V) = 15.5
	// For r=31: score[0] >> others, softmax peaks at kj=0 → output ≈ V[0] = 0
	for i := range q4 {
		q4[i] = 0
	}
	for r := 0; r < seqLen4; r++ {
		q4[r*headDim4+0] = float32(r) // only d=0 has value
	}
	for i := range kt4 {
		kt4[i] = 0
	}
	kt4[0*kvLen4+0] = 100 // KT[dd=0, kj=0] = 100

	for i := range v4 {
		v4[i] = 0
	}
	for r := 0; r < kvLen4; r++ {
		for d := 0; d < headDim4; d++ {
			v4[r*headDim4+d] = float32(r)
		}
	}

	smeOut5 := make([]float32, seqLen4*headDim4)
	asm.SDPAFMOPAF32(q4, kt4, v4, nil, smeOut5, seqLen4, kvLen4, headDim4, scale4)

	// Row 0: Q[0,0]=0 → all scores=0 → uniform → output=15.5
	// Row 31: Q[31,0]=31 → score[kj=0]=31*100*scale=31*100*0.125=387.5 (huge!)
	//   → softmax peaks hard at kj=0 → output ≈ V[0,:] = 0
	t.Logf("asymmetric: row0[0]=%v (expect ~15.5), row31[0]=%v (expect ~0.0)",
		smeOut5[0], smeOut5[31*headDim4])
	t.Logf("asymmetric: row1[0]=%v, row16[0]=%v",
		smeOut5[1*headDim4], smeOut5[16*headDim4])

	// Isolate: Q has realistic data, KT=1, V=incrementing
	// If Q gather works: different Q rows produce same score (all KT=1),
	// BUT different Q rows have different magnitudes, so scores differ between rows
	// Actually no — with KT=1, score[r, c] = sum(Q[r,:]) which varies by row
	// This causes different softmax weights across rows
	q6 := make([]float32, 32*64)
	kt6 := make([]float32, 64*32)
	v6 := make([]float32, 32*64)
	out6 := make([]float32, 32*64)
	for i := range q6 {
		q6[i] = float32(i)*0.01 - 0.5 // same as failing test
	}
	for i := range kt6 {
		kt6[i] = 1.0
	}
	for r := 0; r < 32; r++ {
		for d := 0; d < 64; d++ {
			v6[r*64+d] = float32(r)
		}
	}
	asm.SDPAFMOPAF32(q6, kt6, v6, nil, out6, 32, 32, 64, float32(0.125))
	// With KT=1: score[r, c] = sum_d(Q[r,d]) * 1 = same for all c
	// So softmax is uniform for each row → output = 15.5 for all rows
	// This doesn't distinguish rows either! Need non-uniform KT.
	t.Logf("Q=data,KT=1: row0=%v row31=%v (expect 15.5)", out6[0], out6[31*64])

	// OK let me try: Q=1, KT has one big element at [0,0], V[0]=100, V[else]=0
	for i := range q6 {
		q6[i] = 1.0
	}
	for i := range kt6 {
		kt6[i] = 0.0
	}
	kt6[0*32+0] = 100 // KT[dd=0, kj=0] = 100
	for i := range v6 {
		v6[i] = 0
	}
	v6[0*64+0] = 100 // V[kj=0, d=0] = 100

	asm.SDPAFMOPAF32(q6, kt6, v6, nil, out6, 32, 32, 64, float32(0.125))
	// Q=1, KT[0,0]=100: score[r, 0] = Q[r,0]*KT[0,0] = 1*100 = 100
	// After scale: 100*0.125 = 12.5 (very large)
	// softmax: w[0] ≈ 1, w[1:] ≈ 0 → output[r, 0] ≈ V[0,0] = 100
	t.Logf("Q=1,KT[0,0]=100: out[0]=%v (expect ~100), out[1]=%v (expect ~0)",
		out6[0], out6[1])

	// Now test: Q[0,0]=1 rest 0, KT[0,0]=100, V same
	for i := range q6 {
		q6[i] = 0
	}
	q6[0] = 1.0 // Only Q[row=0, d=0] = 1
	asm.SDPAFMOPAF32(q6, kt6, v6, nil, out6, 32, 32, 64, float32(0.125))
	// Row 0: score[0,0] = 1*100=100, scaled=12.5 → peaked at kj=0
	// Row 1: score[1,c] = 0 for all c → uniform → output[1,0] ≈ V_avg = 0
	t.Logf("Q[0,0]=1: row0[0]=%v (expect ~100), row1[0]=%v (expect ~0)",
		out6[0], out6[1*64])
}

func BenchmarkSDPA(b *testing.B) {
	configs := []struct {
		seqLen, kvLen, headDim int
	}{
		{16, 16, 64},
		{64, 64, 64},
		{128, 128, 64},
		{128, 128, 128},
		{512, 512, 64},
	}

	for _, c := range configs {
		scale := float32(1.0 / stdmath.Sqrt(float64(c.headDim)))
		q := make([]float32, c.seqLen*c.headDim)
		k := make([]float32, c.kvLen*c.headDim)
		v := make([]float32, c.kvLen*c.headDim)
		output := make([]float32, c.seqLen*c.headDim)

		for i := range q {
			q[i] = float32(i) * 0.001
		}
		for i := range k {
			k[i] = float32(i) * 0.001
		}
		for i := range v {
			v[i] = float32(i) * 0.001
		}

		label := fmt.Sprintf("s%d_kv%d_d%d", c.seqLen, c.kvLen, c.headDim)

		b.Run("Auto/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SDPAAuto(q, k, v, nil, output, c.seqLen, c.kvLen, c.headDim, scale)
			}
		})

		b.Run("CausalAuto/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SDPACausalAuto(q, k, v, output, c.seqLen, c.kvLen, c.headDim, scale)
			}
		})

		b.Run("Scalar/"+label, func(b *testing.B) {
			scores := make([]float32, c.seqLen*c.kvLen)
			for i := 0; i < b.N; i++ {
				SDPAScalar(q, k, v, nil, scores, output, c.seqLen, c.kvLen, c.headDim, scale)
			}
		})
	}
}
