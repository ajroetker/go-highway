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
		{"3x5x7/no_mask", 3, 5, 7, false},       // non-aligned (below SME threshold)
		{"32x32x64/mask", 32, 32, 64, true},
		{"64x64x64/no_mask", 64, 64, 64, false},
		{"128x128x64/no_mask", 128, 128, 64, false},
		// SME-eligible but non-aligned to tile boundary (exercises padding)
		{"33x33x33/no_mask", 33, 33, 33, false},
		{"50x50x50/no_mask", 50, 50, 50, false},
		{"33x50x37/no_mask", 33, 50, 37, false},
		{"33x33x33/mask", 33, 33, 33, true},
		{"100x100x100/no_mask", 100, 100, 100, false},
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
		{"3x5x7", 3, 5, 7},   // non-aligned (below SME threshold)
		{"33x33x33", 33, 33, 33},     // SME-eligible, non-aligned
		{"50x50x50", 50, 50, 50},     // SME-eligible, non-aligned
		{"33x50x37", 33, 50, 37},     // all different, non-aligned
		{"100x100x100", 100, 100, 100},
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

func TestSDPAAuto64UnalignedSME(t *testing.T) {
	// f64 tile size is 8, so dims not divisible by 8 but >= 32 exercise padding
	testCases := []struct {
		seqLen, kvLen, headDim int
	}{
		{33, 33, 33},
		{50, 50, 50},
		{33, 50, 37},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("%dx%dx%d", tc.seqLen, tc.kvLen, tc.headDim)
		t.Run(name, func(t *testing.T) {
			scale := 1.0 / stdmath.Sqrt(float64(tc.headDim))
			q := make([]float64, tc.seqLen*tc.headDim)
			k := make([]float64, tc.kvLen*tc.headDim)
			v := make([]float64, tc.kvLen*tc.headDim)

			for i := range q {
				q[i] = float64(i)*0.01 - 0.5
			}
			for i := range k {
				k[i] = float64(i)*0.008 - 0.4
			}
			for i := range v {
				v[i] = float64(i)*0.006 - 0.3
			}

			autoOutput := make([]float64, tc.seqLen*tc.headDim)
			scalarOutput := make([]float64, tc.seqLen*tc.headDim)
			scalarScores := make([]float64, tc.seqLen*tc.kvLen)

			SDPAAuto(q, k, v, nil, autoOutput, tc.seqLen, tc.kvLen, tc.headDim, scale)
			SDPAScalar(q, k, v, nil, scalarScores, scalarOutput, tc.seqLen, tc.kvLen, tc.headDim, scale)

			for i := range autoOutput {
				if stdmath.Abs(autoOutput[i]-scalarOutput[i]) > 1e-6 {
					t.Errorf("output[%d]: auto=%v, scalar=%v", i, autoOutput[i], scalarOutput[i])
				}
			}
		})
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

