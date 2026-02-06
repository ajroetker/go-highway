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

//go:build darwin && arm64

package nn

import (
	"fmt"
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/nn/asm"
)

// transposeF32 transposes a [rows, cols] matrix to [cols, rows].
func transposeF32(src []float32, rows, cols int) []float32 {
	dst := make([]float32, cols*rows)
	matmul.Transpose2D(src, rows, cols, dst)
	return dst
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
	for r := range kvLen {
		for d := range headDim {
			v[r*headDim+d] = float32(r)
		}
	}

	// qt = transpose(q) — all ones, so transpose is identity
	qt := transposeF32(q, seqLen, headDim)

	// With Q=1, KT=1: Q@KT = [[16,16,...],[16,16,...],...]
	// After scale (0.0625): scores = [[1,1,...],[1,1,...],...]
	// softmax([1,1,...,1]) = [1/16, 1/16, ..., 1/16]
	// output = (1/16) * sum(V rows) = (1/16) * (0+1+...+15) * ones = 7.5
	expected := float32(7.5)

	asm.SDPAFMOPAF32(qt, kt, v, nil, output, seqLen, kvLen, headDim, scale)

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

	// Now test with 32x32x64 (uses full 4-tile path)
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
	for r := range kvLen2 {
		for d := range headDim2 {
			v2[r*headDim2+d] = float32(r)
		}
	}

	qt2 := transposeF32(q2, seqLen2, headDim2)

	// Q@KT = 64 for all entries, scaled by 1/64 = 1.0, softmax = 1/32
	// output = (1/32) * sum(0..31) = (1/32) * 496 = 15.5
	expected2 := float32(15.5)

	asm.SDPAFMOPAF32(qt2, kt2, v2, nil, output2, seqLen2, kvLen2, headDim2, scale2)

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

	// Test with actual test data
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

	// Transpose Q and K
	qt3 := transposeF32(q3, seqLen3, headDim3)
	kt3 := transposeF32(k3, kvLen3, headDim3)

	// Get reference from scalar
	scalarOutput3 := make([]float32, seqLen3*headDim3)
	scalarScores3 := make([]float32, seqLen3*kvLen3)
	SDPAScalar(q3, k3, v3, nil, scalarScores3, scalarOutput3, seqLen3, kvLen3, headDim3, scale3)

	// Call SME directly
	smeOutput3 := make([]float32, seqLen3*headDim3)
	asm.SDPAFMOPAF32(qt3, kt3, v3, nil, smeOutput3, seqLen3, kvLen3, headDim3, scale3)

	t.Logf("direct SME: output[0]=%v, scalar=%v", smeOutput3[0], scalarOutput3[0])
	t.Logf("direct SME: output[64]=%v, scalar=%v", smeOutput3[64], scalarOutput3[64])

	// Call through adapter (SDPAAuto)
	autoOutput3 := make([]float32, seqLen3*headDim3)
	SDPAAuto(q3, k3, v3, nil, autoOutput3, seqLen3, kvLen3, headDim3, scale3)

	t.Logf("adapter: output[0]=%v, scalar=%v", autoOutput3[0], scalarOutput3[0])
	t.Logf("adapter: output[64]=%v, scalar=%v", autoOutput3[64], scalarOutput3[64])

	// Compare direct SME vs scalar
	for i := range 5 {
		diff := stdmath.Abs(float64(smeOutput3[i] - scalarOutput3[i]))
		t.Logf("  [%d] sme=%v scalar=%v auto=%v diff_sme=%v diff_auto=%v",
			i, smeOutput3[i], scalarOutput3[i], autoOutput3[i],
			diff, stdmath.Abs(float64(autoOutput3[i]-scalarOutput3[i])))
	}
}

// TestSDPACausalSME tests the causal SME flash attention kernel with dimensions
// large enough to trigger the SME path (>= minDimForSDPASME=32).
func TestSDPACausalSME(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("no SME support")
	}

	tests := []struct {
		name    string
		seqLen  int
		kvLen   int
		headDim int
	}{
		{"32x32x64", 32, 32, 64},
		{"32x64x64", 32, 64, 64}, // kvLen > seqLen (prefix caching)
		{"64x64x64", 64, 64, 64},
		{"64x64x128", 64, 64, 128},
		{"128x128x64", 128, 128, 64},
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

			// Reference: scalar causal
			scalarOutput := make([]float32, tt.seqLen*tt.headDim)
			scalarScores := make([]float32, tt.seqLen*tt.kvLen)
			SDPACausalScalar(q, k, v, scalarScores, scalarOutput, tt.seqLen, tt.kvLen, tt.headDim, scale)

			// SME causal via dispatch
			smeOutput := make([]float32, tt.seqLen*tt.headDim)
			SDPACausalAuto(q, k, v, smeOutput, tt.seqLen, tt.kvLen, tt.headDim, scale)

			t.Logf("scalar[0]=%v sme[0]=%v", scalarOutput[0], smeOutput[0])
			t.Logf("scalar[%d]=%v sme[%d]=%v", tt.headDim, scalarOutput[tt.headDim], tt.headDim, smeOutput[tt.headDim])

			maxDiff := float64(0)
			for i := range smeOutput {
				diff := stdmath.Abs(float64(smeOutput[i] - scalarOutput[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff > 1e-3 {
					t.Errorf("output[%d]=%v, want ~%v (diff=%v)", i, smeOutput[i], scalarOutput[i], diff)
					if i > 10 {
						t.Fatalf("too many errors, stopping")
					}
				}
			}
			t.Logf("max diff: %e", maxDiff)
		})
	}
}

// BenchmarkSDPAHandwrittenVsGenerated compares the handwritten C/assembly SDPA
// against the hwygen-generated Go SIMD version.
func BenchmarkSDPAHandwrittenVsGenerated(b *testing.B) {
	configs := []struct {
		seqLen, kvLen, headDim int
	}{
		{16, 16, 64},
		{64, 64, 64},
		{128, 128, 64},
		{256, 256, 64},
	}

	for _, c := range configs {
		scale := float32(1.0 / stdmath.Sqrt(float64(c.headDim)))
		q := make([]float32, c.seqLen*c.headDim)
		k := make([]float32, c.kvLen*c.headDim)
		v := make([]float32, c.kvLen*c.headDim)
		scores := make([]float32, c.seqLen*c.kvLen)
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

		// hwygen-generated SIMD
		b.Run("Generated/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SDPAAuto(q, k, v, nil, output, c.seqLen, c.kvLen, c.headDim, scale)
			}
		})

		// Handwritten C/assembly (via GOAT)
		b.Run("Handwritten/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				asm.SDPANeonF32(q, k, v, nil, scores, output, c.seqLen, c.kvLen, c.headDim, scale)
			}
		})
	}
}
