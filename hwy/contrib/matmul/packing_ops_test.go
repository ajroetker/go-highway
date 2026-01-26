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

package matmul

import (
	"math"
	"testing"
)

func TestPackRHSFast(t *testing.T) {
	// Test matrix B: 4x8 (K=4, N=8)
	// Row-major layout
	b := []float32{
		1, 2, 3, 4, 5, 6, 7, 8,
		9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24,
		25, 26, 27, 28, 29, 30, 31, 32,
	}

	k, n := 4, 8
	nr := 4 // micro-tile column dimension

	// Expected packed layout: [num_micro_panels, panelK, nr]
	// For nr=4, we have 2 micro-panels
	// Panel 0: cols 0-3
	// Panel 1: cols 4-7
	expected := []float32{
		// Panel 0, K=0: cols 0-3 of row 0
		1, 2, 3, 4,
		// Panel 0, K=1: cols 0-3 of row 1
		9, 10, 11, 12,
		// Panel 0, K=2: cols 0-3 of row 2
		17, 18, 19, 20,
		// Panel 0, K=3: cols 0-3 of row 3
		25, 26, 27, 28,
		// Panel 1, K=0: cols 4-7 of row 0
		5, 6, 7, 8,
		// Panel 1, K=1: cols 4-7 of row 1
		13, 14, 15, 16,
		// Panel 1, K=2: cols 4-7 of row 2
		21, 22, 23, 24,
		// Panel 1, K=3: cols 4-7 of row 3
		29, 30, 31, 32,
	}

	packed := make([]float32, len(expected))
	PackRHSFast(b, packed, n, 0, 0, k, n, nr)

	for i := range expected {
		if packed[i] != expected[i] {
			t.Errorf("packed[%d] = %v, want %v", i, packed[i], expected[i])
		}
	}
}

func TestPackRHSFastPartialPanel(t *testing.T) {
	// Test matrix B: 4x6 (K=4, N=6)
	// Last panel will be partial (2 valid cols + 2 zero padding)
	b := []float32{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
	}

	k, n := 4, 6
	nr := 4

	expected := []float32{
		// Panel 0: cols 0-3
		1, 2, 3, 4,
		7, 8, 9, 10,
		13, 14, 15, 16,
		19, 20, 21, 22,
		// Panel 1: cols 4-5 + zero padding
		5, 6, 0, 0,
		11, 12, 0, 0,
		17, 18, 0, 0,
		23, 24, 0, 0,
	}

	packed := make([]float32, len(expected))
	PackRHSFast(b, packed, n, 0, 0, k, n, nr)

	for i := range expected {
		if packed[i] != expected[i] {
			t.Errorf("packed[%d] = %v, want %v", i, packed[i], expected[i])
		}
	}
}

func TestApplyPackedOutput(t *testing.T) {
	// Test: output = alpha * packed + beta * output
	packed := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	output := []float32{
		10, 20, 30, 40, 50, 60, 70, 80,
		100, 200, 300, 400, 500, 600, 700, 800,
	}

	// Apply to a 2x4 region starting at (0, 2)
	alpha := float32(2.0)
	beta := float32(0.5)

	// After: output[r][2:6] = 2.0 * packed[r][:4] + 0.5 * output[r][2:6]
	// Row 0: [10, 20, 2*1+0.5*30, 2*2+0.5*40, 2*3+0.5*50, 2*4+0.5*60, 70, 80]
	//      = [10, 20, 17, 24, 31, 38, 70, 80]
	// Row 1: [100, 200, 2*5+0.5*300, 2*6+0.5*400, 2*7+0.5*500, 2*8+0.5*600, 700, 800]
	//      = [100, 200, 160, 212, 264, 316, 700, 800]

	expected := []float32{
		10, 20, 17, 24, 31, 38, 70, 80,
		100, 200, 160, 212, 264, 316, 700, 800,
	}

	ApplyPackedOutput(packed, output, alpha, beta, 4, 0, 2, 8, 2, 4)

	for i := range expected {
		if math.Abs(float64(output[i]-expected[i])) > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, output[i], expected[i])
		}
	}
}

func TestApplyPackedOutputSimple(t *testing.T) {
	// Test: output = packed (alpha=1, beta=0)
	packed := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	output := []float32{
		99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99,
	}

	// After applying to (0,1) with width=4:
	// Row 0: [99, 1, 2, 3, 4, 99]
	// Row 1: [99, 5, 6, 7, 8, 99]

	expected := []float32{
		99, 1, 2, 3, 4, 99,
		99, 5, 6, 7, 8, 99,
	}

	ApplyPackedOutputSimple(packed, output, 4, 0, 1, 6, 2, 4)

	for i := range expected {
		if output[i] != expected[i] {
			t.Errorf("output[%d] = %v, want %v", i, output[i], expected[i])
		}
	}
}

func TestApplyPackedOutputAccum(t *testing.T) {
	// Test: output += packed (alpha=1, beta=1)
	packed := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	output := []float32{
		10, 20, 30, 40,
		50, 60, 70, 80,
	}

	// After: output += packed
	expected := []float32{
		11, 22, 33, 44,
		55, 66, 77, 88,
	}

	ApplyPackedOutputAccum(packed, output, 4, 0, 0, 4, 2, 4)

	for i := range expected {
		if output[i] != expected[i] {
			t.Errorf("output[%d] = %v, want %v", i, output[i], expected[i])
		}
	}
}

func TestApplyPackedOutputFloat64(t *testing.T) {
	packed := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	output := []float64{
		10, 20, 30, 40,
		50, 60, 70, 80,
	}

	alpha := 2.0
	beta := 0.5

	// output = 2 * packed + 0.5 * output
	expected := []float64{
		2*1 + 0.5*10, 2*2 + 0.5*20, 2*3 + 0.5*30, 2*4 + 0.5*40,
		2*5 + 0.5*50, 2*6 + 0.5*60, 2*7 + 0.5*70, 2*8 + 0.5*80,
	}

	ApplyPackedOutput(packed, output, alpha, beta, 4, 0, 0, 4, 2, 4)

	for i := range expected {
		if math.Abs(output[i]-expected[i]) > 1e-10 {
			t.Errorf("output[%d] = %v, want %v", i, output[i], expected[i])
		}
	}
}

// Benchmark PackRHSFast vs PackRHS
func BenchmarkPackRHS(b *testing.B) {
	k, n := 512, 512
	nr := 32
	src := make([]float32, k*n)
	for i := range src {
		src[i] = float32(i)
	}
	packed := make([]float32, k*n)

	b.Run("PackRHS", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			PackRHS(src, packed, k, n, 0, 0, k, n, nr)
		}
	})

	b.Run("PackRHSFast", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			PackRHSFast(src, packed, n, 0, 0, k, n, nr)
		}
	})
}

// Benchmark ApplyPackedOutput variants
func BenchmarkApplyPackedOutput(b *testing.B) {
	height, width := 256, 256
	packed := make([]float32, height*width)
	output := make([]float32, height*width)
	for i := range packed {
		packed[i] = float32(i)
		output[i] = float32(i * 2)
	}

	b.Run("General", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ApplyPackedOutput(packed, output, 2.0, 0.5, width, 0, 0, width, height, width)
		}
	})

	b.Run("Simple", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ApplyPackedOutputSimple(packed, output, width, 0, 0, width, height, width)
		}
	})

	b.Run("Accum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ApplyPackedOutputAccum(packed, output, width, 0, 0, width, height, width)
		}
	})
}
