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

package loss

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// testRNG returns a seeded random number generator for reproducible tests.
func testRNG() *rand.Rand {
	return rand.New(rand.NewSource(42))
}

// standardCrossEntropy computes cross-entropy by materializing the full logits matrix.
// This is the O(numPositions * vocabSize) memory approach for comparison.
func standardCrossEntropy(
	hiddenStates []float32,
	embeddings []float32,
	labels []int32,
	numPositions, hiddenDim, vocabSize int,
) float32 {
	totalLoss := float64(0)
	validCount := 0

	// Materialize full logits matrix
	logits := make([]float32, numPositions*vocabSize)
	for pos := 0; pos < numPositions; pos++ {
		for v := 0; v < vocabSize; v++ {
			dot := float32(0)
			for d := 0; d < hiddenDim; d++ {
				dot += hiddenStates[pos*hiddenDim+d] * embeddings[v*hiddenDim+d]
			}
			logits[pos*vocabSize+v] = dot
		}
	}

	// Compute loss for each position
	for pos := 0; pos < numPositions; pos++ {
		label := labels[pos]
		if label < 0 || int(label) >= vocabSize {
			continue
		}

		row := logits[pos*vocabSize : (pos+1)*vocabSize]

		// Numerically stable logsumexp
		maxLogit := row[0]
		for _, v := range row[1:] {
			if v > maxLogit {
				maxLogit = v
			}
		}

		sumExp := float64(0)
		for _, v := range row {
			sumExp += math.Exp(float64(v - maxLogit))
		}

		lse := float64(maxLogit) + math.Log(sumExp)
		loss := lse - float64(row[label])
		totalLoss += loss
		validCount++
	}

	if validCount == 0 {
		return 0
	}
	return float32(totalLoss / float64(validCount))
}

// TestBaseCutCrossEntropyCorrectness verifies CCE against standard cross-entropy.
func TestBaseCutCrossEntropyCorrectness(t *testing.T) {
	numPositions := 4
	hiddenDim := 8
	vocabSize := 16

	rng := testRNG()

	hiddenStates := make([]float32, numPositions*hiddenDim)
	for i := range hiddenStates {
		hiddenStates[i] = rng.Float32()*2 - 1
	}

	embeddings := make([]float32, vocabSize*hiddenDim)
	for i := range embeddings {
		embeddings[i] = rng.Float32()*2 - 1
	}

	labels := []int32{3, 7, 0, 12}

	// CCE result
	cceLoss := BaseCutCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)

	// Reference: standard cross-entropy with materialized logits
	refLoss := standardCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)

	diff := math.Abs(float64(cceLoss - refLoss))
	if diff > 1e-4 {
		t.Errorf("CCE loss %f != reference loss %f (diff %f)", cceLoss, refLoss, diff)
	}
}

// TestBaseCutCrossEntropyWithIgnored verifies that -1 labels are ignored.
func TestBaseCutCrossEntropyWithIgnored(t *testing.T) {
	numPositions := 6
	hiddenDim := 8
	vocabSize := 16

	rng := testRNG()

	hiddenStates := make([]float32, numPositions*hiddenDim)
	for i := range hiddenStates {
		hiddenStates[i] = rng.Float32()*2 - 1
	}

	embeddings := make([]float32, vocabSize*hiddenDim)
	for i := range embeddings {
		embeddings[i] = rng.Float32()*2 - 1
	}

	// Mix of valid and ignored labels
	labels := []int32{3, -1, 7, -1, 0, 12}

	cceLoss := BaseCutCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)
	refLoss := standardCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)

	diff := math.Abs(float64(cceLoss - refLoss))
	if diff > 1e-4 {
		t.Errorf("CCE loss %f != reference loss %f (diff %f)", cceLoss, refLoss, diff)
	}
}

// TestBaseCutCrossEntropyAllIgnored verifies zero loss when all positions ignored.
func TestBaseCutCrossEntropyAllIgnored(t *testing.T) {
	numPositions := 4
	hiddenDim := 8
	vocabSize := 16

	hiddenStates := make([]float32, numPositions*hiddenDim)
	embeddings := make([]float32, vocabSize*hiddenDim)
	labels := []int32{-1, -1, -1, -1}

	loss := BaseCutCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)
	if loss != 0 {
		t.Errorf("expected zero loss for all-ignored positions, got %f", loss)
	}
}

// TestBaseCutCrossEntropyEmpty verifies zero loss for empty inputs.
func TestBaseCutCrossEntropyEmpty(t *testing.T) {
	loss := BaseCutCrossEntropy(nil, nil, nil, 0, 0, 0)
	if loss != 0 {
		t.Errorf("expected zero loss for empty input, got %f", loss)
	}
}

// TestBaseCutCrossEntropyGradBasic verifies gradient computation.
func TestBaseCutCrossEntropyGradBasic(t *testing.T) {
	numPositions := 2
	hiddenDim := 4
	vocabSize := 8

	rng := testRNG()

	hiddenStates := make([]float32, numPositions*hiddenDim)
	for i := range hiddenStates {
		hiddenStates[i] = rng.Float32()*0.5 - 0.25
	}

	embeddings := make([]float32, vocabSize*hiddenDim)
	for i := range embeddings {
		embeddings[i] = rng.Float32()*0.5 - 0.25
	}

	labels := []int32{2, 5}

	gradOutput := make([]float32, numPositions*hiddenDim)
	BaseCutCrossEntropyGrad(hiddenStates, embeddings, labels, gradOutput, numPositions, hiddenDim, vocabSize)

	// Verify gradient via finite differences
	eps := float32(1e-3)
	for pos := 0; pos < numPositions; pos++ {
		for d := 0; d < hiddenDim; d++ {
			idx := pos*hiddenDim + d

			// f(x + eps)
			hiddenStates[idx] += eps
			lossPlus := BaseCutCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)

			// f(x - eps)
			hiddenStates[idx] -= 2 * eps
			lossMinus := BaseCutCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)

			// Restore
			hiddenStates[idx] += eps

			numericalGrad := (lossPlus - lossMinus) / (2 * eps)
			analyticGrad := gradOutput[idx]

			diff := math.Abs(float64(numericalGrad - analyticGrad))
			relDiff := diff / (math.Abs(float64(numericalGrad)) + 1e-8)

			if relDiff > 0.1 && diff > 1e-3 {
				t.Errorf("Gradient mismatch at pos=%d, dim=%d: numerical=%f, analytic=%f (relDiff=%f)",
					pos, d, numericalGrad, analyticGrad, relDiff)
			}
		}
	}
}

// TestCutCrossEntropyParallel verifies the parallel version matches sequential.
func TestCutCrossEntropyParallel(t *testing.T) {
	numPositions := 128
	hiddenDim := 16
	vocabSize := 32

	rng := testRNG()

	hiddenStates := make([]float32, numPositions*hiddenDim)
	for i := range hiddenStates {
		hiddenStates[i] = rng.Float32()*2 - 1
	}

	embeddings := make([]float32, vocabSize*hiddenDim)
	for i := range embeddings {
		embeddings[i] = rng.Float32()*2 - 1
	}

	labels := make([]int32, numPositions)
	for i := range labels {
		labels[i] = int32(rng.Intn(vocabSize))
	}

	// Sequential
	seqLoss := BaseCutCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)

	// Parallel with different worker counts
	for _, workers := range []int{1, 2, 4, 8} {
		parLoss := CutCrossEntropyParallel(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize, workers)
		diff := math.Abs(float64(seqLoss - parLoss))
		if diff > 1e-4 {
			t.Errorf("workers=%d: parallel loss %f != sequential loss %f (diff %f)",
				workers, parLoss, seqLoss, diff)
		}
	}
}

// TestCutCrossEntropyParallelWithIgnored verifies parallel CCE handles ignored positions correctly.
// This is an edge case where some workers may have all-ignored positions.
func TestCutCrossEntropyParallelWithIgnored(t *testing.T) {
	numPositions := 8
	hiddenDim := 8
	vocabSize := 16

	rng := testRNG()

	hiddenStates := make([]float32, numPositions*hiddenDim)
	for i := range hiddenStates {
		hiddenStates[i] = rng.Float32()*2 - 1
	}

	embeddings := make([]float32, vocabSize*hiddenDim)
	for i := range embeddings {
		embeddings[i] = rng.Float32()*2 - 1
	}

	// First half all ignored, second half valid
	// This tests the edge case where some workers have all-ignored positions
	labels := []int32{-1, -1, -1, -1, 3, 7, 0, 12}

	// Sequential loss
	seqLoss := BaseCutCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)

	// Parallel with 4 workers - first worker will have all ignored positions
	parLoss := CutCrossEntropyParallel(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize, 4)

	diff := math.Abs(float64(seqLoss - parLoss))
	if diff > 1e-4 {
		t.Errorf("parallel loss %f != sequential loss %f (diff %f)", parLoss, seqLoss, diff)
	}

	// Also test with 2 workers - half ignored, half valid
	parLoss2 := CutCrossEntropyParallel(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize, 2)
	diff2 := math.Abs(float64(seqLoss - parLoss2))
	if diff2 > 1e-4 {
		t.Errorf("parallel (2 workers) loss %f != sequential loss %f (diff %f)", parLoss2, seqLoss, diff2)
	}
}

// TestCutCrossEntropyParallelAllIgnored tests parallel CCE with all positions ignored.
func TestCutCrossEntropyParallelAllIgnored(t *testing.T) {
	numPositions := 128
	hiddenDim := 8
	vocabSize := 16

	hiddenStates := make([]float32, numPositions*hiddenDim)
	embeddings := make([]float32, vocabSize*hiddenDim)
	labels := make([]int32, numPositions)
	for i := range labels {
		labels[i] = -1
	}

	for _, workers := range []int{2, 4, 8} {
		loss := CutCrossEntropyParallel(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize, workers)
		if loss != 0 {
			t.Errorf("workers=%d: expected zero loss for all-ignored positions, got %f", workers, loss)
		}
	}
}

// TestCutCrossEntropyParallelSmallInput verifies parallel falls back to sequential for small inputs.
func TestCutCrossEntropyParallelSmallInput(t *testing.T) {
	numPositions := 32 // Below the 64 threshold
	hiddenDim := 8
	vocabSize := 16

	rng := testRNG()

	hiddenStates := make([]float32, numPositions*hiddenDim)
	for i := range hiddenStates {
		hiddenStates[i] = rng.Float32()*2 - 1
	}

	embeddings := make([]float32, vocabSize*hiddenDim)
	for i := range embeddings {
		embeddings[i] = rng.Float32()*2 - 1
	}

	labels := make([]int32, numPositions)
	for i := range labels {
		labels[i] = int32(rng.Intn(vocabSize))
	}

	seqLoss := BaseCutCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)
	parLoss := CutCrossEntropyParallel(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize, 8)

	diff := math.Abs(float64(seqLoss - parLoss))
	if diff > 1e-6 {
		t.Errorf("small input: parallel loss %f != sequential loss %f (diff %f)", parLoss, seqLoss, diff)
	}
}

// TestMemorySavingsEstimate verifies memory estimate function.
func TestMemorySavingsEstimate(t *testing.T) {
	// 2B model-like parameters
	numPositions := 32768
	hiddenDim := 2048
	vocabSize := 32000

	stdMem, cceMem := MemorySavingsEstimate(numPositions, hiddenDim, vocabSize)

	// Standard should be much larger than CCE
	if cceMem >= stdMem {
		t.Errorf("CCE memory %d should be less than standard %d", cceMem, stdMem)
	}

	// Standard: 32K * 32K * 4 * 2 = ~8 GB
	expectedStdApprox := int64(numPositions) * int64(vocabSize) * 4 * 2
	if stdMem != expectedStdApprox {
		t.Errorf("Expected standard memory %d, got %d", expectedStdApprox, stdMem)
	}

	ratio := float64(cceMem) / float64(stdMem)
	t.Logf("Memory savings: standard=%d bytes, CCE=%d bytes, ratio=%.6f (%.0fx reduction)",
		stdMem, cceMem, ratio, 1.0/ratio)
}

// BenchmarkCutCrossEntropy benchmarks CCE vs standard CE.
func BenchmarkCutCrossEntropy(b *testing.B) {
	numPositions := 64
	hiddenDim := 128
	vocabSize := 1024

	rng := testRNG()

	hiddenStates := make([]float32, numPositions*hiddenDim)
	for i := range hiddenStates {
		hiddenStates[i] = rng.Float32()*2 - 1
	}

	embeddings := make([]float32, vocabSize*hiddenDim)
	for i := range embeddings {
		embeddings[i] = rng.Float32()*2 - 1
	}

	labels := make([]int32, numPositions)
	for i := range labels {
		labels[i] = int32(rng.Intn(vocabSize))
	}

	b.Run("CutCrossEntropy", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = BaseCutCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)
		}
	})

	b.Run("StandardCrossEntropy", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = standardCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)
		}
	})

	b.Run("CutCrossEntropy_MemoryUsage", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = BaseCutCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)
		}
	})
}

// BenchmarkCutCrossEntropyParallel benchmarks parallel CCE with different worker counts.
func BenchmarkCutCrossEntropyParallel(b *testing.B) {
	numPositions := 256
	hiddenDim := 128
	vocabSize := 1024

	rng := testRNG()

	hiddenStates := make([]float32, numPositions*hiddenDim)
	for i := range hiddenStates {
		hiddenStates[i] = rng.Float32()*2 - 1
	}

	embeddings := make([]float32, vocabSize*hiddenDim)
	for i := range embeddings {
		embeddings[i] = rng.Float32()*2 - 1
	}

	labels := make([]int32, numPositions)
	for i := range labels {
		labels[i] = int32(rng.Intn(vocabSize))
	}

	b.Run("Sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = BaseCutCrossEntropy(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize)
		}
	})

	for _, workers := range []int{2, 4, 8} {
		b.Run(fmt.Sprintf("Parallel_%dworkers", workers), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = CutCrossEntropyParallel(hiddenStates, embeddings, labels, numPositions, hiddenDim, vocabSize, workers)
			}
		})
	}
}
