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

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestQKVDenseAuto(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	tests := []struct {
		name       string
		batchSize  int
		inFeatures int
		qDim       int
		kvDim      int
		useBias    bool
	}{
		{"1x64x64x64/bias", 1, 64, 64, 64, true},
		{"1x64x64x64/no_bias", 1, 64, 64, 64, false},
		{"2x128x64x64/bias", 2, 128, 64, 64, true},
		{"4x256x128x64/bias", 4, 256, 128, 64, true},
		{"3x7x5x5/bias", 3, 7, 5, 5, true}, // non-aligned dimensions
		{"8x64x32x32/bias", 8, 64, 32, 32, true},
		{"1x128x64x32/bias", 1, 128, 64, 32, true}, // qDim != kvDim (GQA-style)
		{"2x768x256x64/bias", 2, 768, 256, 64, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			totalOut := tt.qDim + 2*tt.kvDim
			x := make([]float32, tt.batchSize*tt.inFeatures)
			wQKV := make([]float32, totalOut*tt.inFeatures)
			var biasQ, biasK, biasV []float32
			if tt.useBias {
				biasQ = make([]float32, tt.qDim)
				biasK = make([]float32, tt.kvDim)
				biasV = make([]float32, tt.kvDim)
				for i := range biasQ {
					biasQ[i] = float32(i) * 0.1
				}
				for i := range biasK {
					biasK[i] = float32(i) * 0.05
				}
				for i := range biasV {
					biasV[i] = float32(i) * 0.02
				}
			}

			for i := range x {
				x[i] = float32(i)*0.01 - 0.5
			}
			for i := range wQKV {
				wQKV[i] = float32(i)*0.005 - 0.25
			}

			autoQ := make([]float32, tt.batchSize*tt.qDim)
			autoK := make([]float32, tt.batchSize*tt.kvDim)
			autoV := make([]float32, tt.batchSize*tt.kvDim)

			scalarQ := make([]float32, tt.batchSize*tt.qDim)
			scalarK := make([]float32, tt.batchSize*tt.kvDim)
			scalarV := make([]float32, tt.batchSize*tt.kvDim)

			QKVDenseAuto(pool, x, wQKV, biasQ, biasK, biasV, autoQ, autoK, autoV,
				tt.batchSize, tt.inFeatures, tt.qDim, tt.kvDim)
			QKVDenseScalar(x, wQKV, biasQ, biasK, biasV, scalarQ, scalarK, scalarV,
				tt.batchSize, tt.inFeatures, tt.qDim, tt.kvDim)

			compareSlices(t, "Q", autoQ, scalarQ)
			compareSlices(t, "K", autoK, scalarK)
			compareSlices(t, "V", autoV, scalarV)
		})
	}
}

func TestQKVDenseBase(t *testing.T) {
	batchSize, inFeatures, qDim, kvDim := 4, 32, 16, 16

	x := make([]float32, batchSize*inFeatures)
	wQKV := make([]float32, (qDim+2*kvDim)*inFeatures)
	biasQ := make([]float32, qDim)
	biasK := make([]float32, kvDim)
	biasV := make([]float32, kvDim)

	for i := range x {
		x[i] = float32(i)*0.01 - 0.5
	}
	for i := range wQKV {
		wQKV[i] = float32(i)*0.005 - 0.25
	}
	for i := range biasQ {
		biasQ[i] = float32(i) * 0.1
	}
	for i := range biasK {
		biasK[i] = float32(i) * 0.05
	}
	for i := range biasV {
		biasV[i] = float32(i) * 0.02
	}

	baseQ := make([]float32, batchSize*qDim)
	baseK := make([]float32, batchSize*kvDim)
	baseV := make([]float32, batchSize*kvDim)

	scalarQ := make([]float32, batchSize*qDim)
	scalarK := make([]float32, batchSize*kvDim)
	scalarV := make([]float32, batchSize*kvDim)

	QKVDense(x, wQKV, biasQ, biasK, biasV, baseQ, baseK, baseV,
		batchSize, inFeatures, qDim, kvDim)
	QKVDenseScalar(x, wQKV, biasQ, biasK, biasV, scalarQ, scalarK, scalarV,
		batchSize, inFeatures, qDim, kvDim)

	compareSlices(t, "Q", baseQ, scalarQ)
	compareSlices(t, "K", baseK, scalarK)
	compareSlices(t, "V", baseV, scalarV)
}

func TestQKVDenseEquivalence(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	// Verify fused QKV == 3 separate DenseAuto calls
	batchSize, inFeatures, qDim, kvDim := 2, 64, 32, 32
	totalOut := qDim + 2*kvDim

	x := make([]float32, batchSize*inFeatures)
	wQKV := make([]float32, totalOut*inFeatures)
	biasQ := make([]float32, qDim)
	biasK := make([]float32, kvDim)
	biasV := make([]float32, kvDim)

	for i := range x {
		x[i] = float32(i)*0.01 - 0.5
	}
	for i := range wQKV {
		wQKV[i] = float32(i)*0.005 - 0.25
	}
	for i := range biasQ {
		biasQ[i] = float32(i) * 0.1
	}
	for i := range biasK {
		biasK[i] = float32(i) * 0.05
	}
	for i := range biasV {
		biasV[i] = float32(i) * 0.02
	}

	// Fused QKV
	fusedQ := make([]float32, batchSize*qDim)
	fusedK := make([]float32, batchSize*kvDim)
	fusedV := make([]float32, batchSize*kvDim)
	QKVDenseAuto(pool, x, wQKV, biasQ, biasK, biasV, fusedQ, fusedK, fusedV,
		batchSize, inFeatures, qDim, kvDim)

	// 3 separate DenseAuto calls
	wQ := wQKV[:qDim*inFeatures]
	wK := wQKV[qDim*inFeatures : (qDim+kvDim)*inFeatures]
	wV := wQKV[(qDim+kvDim)*inFeatures:]

	sepQ := make([]float32, batchSize*qDim)
	sepK := make([]float32, batchSize*kvDim)
	sepV := make([]float32, batchSize*kvDim)

	DenseAuto(pool, x, wQ, biasQ, sepQ, batchSize, inFeatures, qDim)
	DenseAuto(pool, x, wK, biasK, sepK, batchSize, inFeatures, kvDim)
	DenseAuto(pool, x, wV, biasV, sepV, batchSize, inFeatures, kvDim)

	compareSlices(t, "Q", fusedQ, sepQ)
	compareSlices(t, "K", fusedK, sepK)
	compareSlices(t, "V", fusedV, sepV)
}

func TestQKVDenseAuto64(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batchSize, inFeatures, qDim, kvDim := 2, 16, 8, 8

	x := make([]float64, batchSize*inFeatures)
	wQKV := make([]float64, (qDim+2*kvDim)*inFeatures)
	biasQ := make([]float64, qDim)
	biasK := make([]float64, kvDim)
	biasV := make([]float64, kvDim)

	for i := range x {
		x[i] = float64(i)*0.01 - 0.5
	}
	for i := range wQKV {
		wQKV[i] = float64(i)*0.005 - 0.25
	}
	for i := range biasQ {
		biasQ[i] = float64(i) * 0.1
	}
	for i := range biasK {
		biasK[i] = float64(i) * 0.05
	}
	for i := range biasV {
		biasV[i] = float64(i) * 0.02
	}

	autoQ := make([]float64, batchSize*qDim)
	autoK := make([]float64, batchSize*kvDim)
	autoV := make([]float64, batchSize*kvDim)

	scalarQ := make([]float64, batchSize*qDim)
	scalarK := make([]float64, batchSize*kvDim)
	scalarV := make([]float64, batchSize*kvDim)

	QKVDenseAuto(pool, x, wQKV, biasQ, biasK, biasV, autoQ, autoK, autoV,
		batchSize, inFeatures, qDim, kvDim)
	QKVDenseScalar(x, wQKV, biasQ, biasK, biasV, scalarQ, scalarK, scalarV,
		batchSize, inFeatures, qDim, kvDim)

	for i := range autoQ {
		if stdmath.Abs(autoQ[i]-scalarQ[i]) > 1e-10 {
			t.Errorf("Q[%d]: auto=%v, scalar=%v", i, autoQ[i], scalarQ[i])
		}
	}
	for i := range autoK {
		if stdmath.Abs(autoK[i]-scalarK[i]) > 1e-10 {
			t.Errorf("K[%d]: auto=%v, scalar=%v", i, autoK[i], scalarK[i])
		}
	}
	for i := range autoV {
		if stdmath.Abs(autoV[i]-scalarV[i]) > 1e-10 {
			t.Errorf("V[%d]: auto=%v, scalar=%v", i, autoV[i], scalarV[i])
		}
	}
}

func BenchmarkQKVDense(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	configs := []struct {
		batch, in, qDim, kvDim int
	}{
		{1, 768, 768, 768},
		{8, 768, 768, 768},
		{1, 768, 256, 64},  // GQA-style
		{32, 768, 768, 768},
	}

	for _, c := range configs {
		totalOut := c.qDim + 2*c.kvDim
		x := make([]float32, c.batch*c.in)
		wQKV := make([]float32, totalOut*c.in)
		biasQ := make([]float32, c.qDim)
		biasK := make([]float32, c.kvDim)
		biasV := make([]float32, c.kvDim)
		q := make([]float32, c.batch*c.qDim)
		k := make([]float32, c.batch*c.kvDim)
		v := make([]float32, c.batch*c.kvDim)

		for i := range x {
			x[i] = float32(i) * 0.001
		}
		for i := range wQKV {
			wQKV[i] = float32(i) * 0.0005
		}

		label := fmt.Sprintf("b%d_%dx%dx%d", c.batch, c.in, c.qDim, c.kvDim)

		b.Run("Auto/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				QKVDenseAuto(pool, x, wQKV, biasQ, biasK, biasV, q, k, v,
					c.batch, c.in, c.qDim, c.kvDim)
			}
		})

		b.Run("Base/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				QKVDense(x, wQKV, biasQ, biasK, biasV, q, k, v,
					c.batch, c.in, c.qDim, c.kvDim)
			}
		})

		// Benchmark 3 separate DenseAuto calls for comparison
		wQ := wQKV[:c.qDim*c.in]
		wK := wQKV[c.qDim*c.in : (c.qDim+c.kvDim)*c.in]
		wV := wQKV[(c.qDim+c.kvDim)*c.in:]

		b.Run("Separate/"+label, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				DenseAuto(pool, x, wQ, biasQ, q, c.batch, c.in, c.qDim)
				DenseAuto(pool, x, wK, biasK, k, c.batch, c.in, c.kvDim)
				DenseAuto(pool, x, wV, biasV, v, c.batch, c.in, c.kvDim)
			}
		})
	}
}

// compareSlices is a test helper that compares two float32 slices with relative tolerance.
func compareSlices(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: length mismatch: got %d, want %d", name, len(got), len(want))
		return
	}
	for i := range got {
		diff := stdmath.Abs(float64(got[i] - want[i]))
		relTol := stdmath.Max(1e-4, 1e-4*stdmath.Abs(float64(want[i])))
		if diff > relTol {
			t.Errorf("%s[%d]: got=%v, want=%v, diff=%v", name, i, got[i], want[i], diff)
		}
	}
}
