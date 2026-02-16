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

// cosineSimilarity computes the cosine similarity between two float32 vectors.
// Returns 1.0 for identical direction, 0.0 for orthogonal, -1.0 for opposite.
func cosineSimilarity(a, b []float32) float64 {
	var dotAB, dotAA, dotBB float64
	for i := range a {
		dotAB += float64(a[i]) * float64(b[i])
		dotAA += float64(a[i]) * float64(a[i])
		dotBB += float64(b[i]) * float64(b[i])
	}
	denom := stdmath.Sqrt(dotAA) * stdmath.Sqrt(dotBB)
	if denom < 1e-12 {
		return 0
	}
	return dotAB / denom
}

func TestQuantizedSDPA(t *testing.T) {
	tests := []struct {
		name    string
		seqLen  int
		kvLen   int
		headDim int
		useMask bool
	}{
		{"4x4x32/no_mask", 4, 4, 32, false},
		{"4x4x32/mask", 4, 4, 32, true},
		{"8x8x64/no_mask", 8, 8, 64, false},
		{"8x16x64/no_mask", 8, 16, 64, false},
		{"16x16x128/no_mask", 16, 16, 128, false},
		{"32x32x64/mask", 32, 32, 64, true},
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

			// Quantized output
			qOutput := make([]float32, tt.seqLen*tt.headDim)
			QuantizedSDPA(q, k, v, mask, qOutput, tt.seqLen, tt.kvLen, tt.headDim, scale)

			// Scalar reference output
			scalarOutput := make([]float32, tt.seqLen*tt.headDim)
			scalarScores := make([]float32, tt.seqLen*tt.kvLen)
			SDPAScalar(q, k, v, mask, scalarScores, scalarOutput, tt.seqLen, tt.kvLen, tt.headDim, scale)

			// Quantized SDPA has significant error due to two uint8
			// quantize-matmul-dequantize stages plus softmax. Use cosine
			// similarity as the primary quality metric (direction agreement),
			// which is robust to scale differences from quantization.
			cosSim := cosineSimilarity(qOutput, scalarOutput)
			if cosSim < 0.90 {
				t.Errorf("Cosine similarity: %.4f (want >= 0.90)", cosSim)
			}

			// No NaN or Inf
			for i, val := range qOutput {
				if stdmath.IsNaN(float64(val)) || stdmath.IsInf(float64(val), 0) {
					t.Errorf("qOutput[%d] = %v (NaN/Inf)", i, val)
				}
			}
		})
	}
}

func TestQuantizedSDPACausal(t *testing.T) {
	tests := []struct {
		name    string
		seqLen  int
		kvLen   int
		headDim int
	}{
		{"4x4x32", 4, 4, 32},
		{"8x8x64", 8, 8, 64},
		{"4x8x32", 4, 8, 32},
		{"16x16x64", 16, 16, 64},
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

			// Quantized causal output
			qOutput := make([]float32, tt.seqLen*tt.headDim)
			QuantizedSDPACausal(q, k, v, qOutput, tt.seqLen, tt.kvLen, tt.headDim, scale)

			// Scalar causal reference
			scalarOutput := make([]float32, tt.seqLen*tt.headDim)
			scalarScores := make([]float32, tt.seqLen*tt.kvLen)
			SDPACausalScalar(q, k, v, scalarScores, scalarOutput, tt.seqLen, tt.kvLen, tt.headDim, scale)

			cosSim := cosineSimilarity(qOutput, scalarOutput)
			if cosSim < 0.90 {
				t.Errorf("Cosine similarity: %.4f (want >= 0.90)", cosSim)
			}

			for i, val := range qOutput {
				if stdmath.IsNaN(float64(val)) || stdmath.IsInf(float64(val), 0) {
					t.Errorf("qOutput[%d] = %v (NaN/Inf)", i, val)
				}
			}
		})
	}
}

func TestMultiHeadQuantizedSDPA(t *testing.T) {
	batchSize := 2
	numHeads := 4
	numKVHeads := 2 // GQA
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

	MultiHeadQuantizedSDPA(nil, q, k, v, nil, output, batchSize, numHeads, numKVHeads,
		seqLen, kvLen, headDim, 0, 0, scale, false)

	// No NaN or Inf
	for i, val := range output {
		if stdmath.IsNaN(float64(val)) || stdmath.IsInf(float64(val), 0) {
			t.Errorf("output[%d] = %v (NaN/Inf)", i, val)
		}
	}

	// GQA: heads 0 and 1 should share KV head 0, produce different outputs
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

func TestMultiHeadQuantizedSDPACausal(t *testing.T) {
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

	MultiHeadQuantizedSDPA(nil, q, k, v, nil, output, batchSize, numHeads, numKVHeads,
		seqLen, kvLen, headDim, 0, 0, scale, true)

	for i, val := range output {
		if stdmath.IsNaN(float64(val)) || stdmath.IsInf(float64(val), 0) {
			t.Errorf("output[%d] = %v (NaN/Inf)", i, val)
		}
	}
}

func TestMultiHeadQuantizedSDPAStrided(t *testing.T) {
	tests := []struct {
		name       string
		batchSize  int
		numHeads   int
		numKVHeads int
		seqLen     int
		kvLen      int
		headDim    int
		causal     bool
	}{
		{"b2_h4_kv2_s8_d16/non_causal", 2, 4, 2, 8, 8, 16, false},
		{"b2_h4_kv2_s8_d16/causal", 2, 4, 2, 8, 8, 16, true},
		{"b1_h2_kv2_s4_d8/non_causal", 1, 2, 2, 4, 8, 8, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scale := float32(1.0 / stdmath.Sqrt(float64(tt.headDim)))

			// Create data in BSHD layout.
			qBSHD := make([]float32, tt.batchSize*tt.seqLen*tt.numHeads*tt.headDim)
			kBSHD := make([]float32, tt.batchSize*tt.kvLen*tt.numKVHeads*tt.headDim)
			vBSHD := make([]float32, tt.batchSize*tt.kvLen*tt.numKVHeads*tt.headDim)

			for i := range qBSHD {
				qBSHD[i] = float32(float64(i)*0.01 - 0.5)
			}
			for i := range kBSHD {
				kBSHD[i] = float32(float64(i)*0.008 - 0.4)
			}
			for i := range vBSHD {
				vBSHD[i] = float32(float64(i)*0.006 - 0.3)
			}

			// Permute BSHD → BHSD for reference.
			qBHSD := make([]float32, len(qBSHD))
			kBHSD := make([]float32, len(kBSHD))
			vBHSD := make([]float32, len(vBSHD))

			for b := range tt.batchSize {
				for s := range tt.seqLen {
					for h := range tt.numHeads {
						for d := range tt.headDim {
							srcIdx := b*tt.seqLen*tt.numHeads*tt.headDim + s*tt.numHeads*tt.headDim + h*tt.headDim + d
							dstIdx := b*tt.numHeads*tt.seqLen*tt.headDim + h*tt.seqLen*tt.headDim + s*tt.headDim + d
							qBHSD[dstIdx] = qBSHD[srcIdx]
						}
					}
				}
			}
			for b := range tt.batchSize {
				for s := range tt.kvLen {
					for h := range tt.numKVHeads {
						for d := range tt.headDim {
							srcIdx := b*tt.kvLen*tt.numKVHeads*tt.headDim + s*tt.numKVHeads*tt.headDim + h*tt.headDim + d
							dstIdx := b*tt.numKVHeads*tt.kvLen*tt.headDim + h*tt.kvLen*tt.headDim + s*tt.headDim + d
							kBHSD[dstIdx] = kBSHD[srcIdx]
							vBHSD[dstIdx] = vBSHD[srcIdx]
						}
					}
				}
			}

			// Reference: BHSD contiguous.
			refOutput := make([]float32, tt.batchSize*tt.numHeads*tt.seqLen*tt.headDim)
			MultiHeadQuantizedSDPA(nil, qBHSD, kBHSD, vBHSD, nil, refOutput,
				tt.batchSize, tt.numHeads, tt.numKVHeads, tt.seqLen, tt.kvLen, tt.headDim,
				0, 0, scale, tt.causal)

			// Strided: BSHD.
			stridedOutput := make([]float32, len(qBSHD))
			qBatchStride := tt.seqLen * tt.numHeads * tt.headDim
			qHeadStride := tt.headDim
			qSeqStride := tt.numHeads * tt.headDim
			kvBatchStride := tt.kvLen * tt.numKVHeads * tt.headDim
			kvHeadStride := tt.headDim
			kvSeqStride := tt.numKVHeads * tt.headDim

			MultiHeadQuantizedSDPAStrided(nil,
				qBSHD, kBSHD, vBSHD, nil, stridedOutput,
				tt.batchSize, tt.numHeads, tt.numKVHeads, tt.seqLen, tt.kvLen, tt.headDim,
				qBatchStride, qHeadStride, qSeqStride,
				kvBatchStride, kvHeadStride, kvSeqStride,
				0, 0,
				scale, tt.causal,
			)

			// Permute strided output (BSHD) → BHSD for comparison.
			stridedBHSD := make([]float32, len(stridedOutput))
			for b := range tt.batchSize {
				for s := range tt.seqLen {
					for h := range tt.numHeads {
						for d := range tt.headDim {
							srcIdx := b*tt.seqLen*tt.numHeads*tt.headDim + s*tt.numHeads*tt.headDim + h*tt.headDim + d
							dstIdx := b*tt.numHeads*tt.seqLen*tt.headDim + h*tt.seqLen*tt.headDim + s*tt.headDim + d
							stridedBHSD[dstIdx] = stridedOutput[srcIdx]
						}
					}
				}
			}

			// Compare.
			for i := range refOutput {
				diff := stdmath.Abs(float64(refOutput[i] - stridedBHSD[i]))
				if diff > 1e-3 {
					t.Errorf("output[%d]: ref=%v, strided=%v, diff=%v", i, refOutput[i], stridedBHSD[i], diff)
				}
			}
		})
	}
}

func BenchmarkQuantizedSDPA(b *testing.B) {
	configs := []struct {
		seqLen, kvLen, headDim int
	}{
		{16, 16, 64},
		{64, 64, 64},
		{128, 128, 64},
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

		b.Run("Quantized/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				QuantizedSDPA(q, k, v, nil, output, c.seqLen, c.kvLen, c.headDim, scale)
			}
		})

		b.Run("QuantizedCausal/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				QuantizedSDPACausal(q, k, v, output, c.seqLen, c.kvLen, c.headDim, scale)
			}
		})

		b.Run("ScalarRef/"+label, func(b *testing.B) {
			scores := make([]float32, c.seqLen*c.kvLen)
			for i := 0; i < b.N; i++ {
				SDPAScalar(q, k, v, nil, scores, output, c.seqLen, c.kvLen, c.headDim, scale)
			}
		})
	}
}
