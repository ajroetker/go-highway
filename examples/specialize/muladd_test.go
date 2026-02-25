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

package specialize

import (
	"fmt"
	"math"
	"testing"
)

// referenceMulAdd computes out[i] += x[i] * y[i] in scalar float64.
func referenceMulAdd(x, y, out []float32) {
	for i := range out {
		out[i] += x[i] * y[i]
	}
}

func TestBaseMulAdd(t *testing.T) {
	tests := []struct {
		name string
		x    []float32
		y    []float32
		out  []float32
	}{
		{
			name: "zeros",
			x:    []float32{0, 0, 0, 0},
			y:    []float32{0, 0, 0, 0},
			out:  []float32{0, 0, 0, 0},
		},
		{
			name: "ones",
			x:    []float32{1, 1, 1, 1},
			y:    []float32{1, 1, 1, 1},
			out:  []float32{0, 0, 0, 0},
		},
		{
			name: "accumulate",
			x:    []float32{1, 2, 3, 4},
			y:    []float32{5, 6, 7, 8},
			out:  []float32{10, 20, 30, 40},
		},
		{
			name: "negative",
			x:    []float32{-1, -2, -3, -4},
			y:    []float32{1, 2, 3, 4},
			out:  []float32{0, 0, 0, 0},
		},
		{
			name: "simd width",
			x:    []float32{1, 2, 3, 4, 5, 6, 7, 8},
			y:    []float32{8, 7, 6, 5, 4, 3, 2, 1},
			out:  []float32{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name: "larger than simd with tail",
			x:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			y:    []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			out:  []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Compute expected
			expected := make([]float32, len(tt.out))
			copy(expected, tt.out)
			referenceMulAdd(tt.x, tt.y, expected)

			// Compute actual
			actual := make([]float32, len(tt.out))
			copy(actual, tt.out)
			BaseMulAdd(tt.x, tt.y, actual)

			for i := range actual {
				diff := math.Abs(float64(actual[i] - expected[i]))
				if diff > 1e-5 {
					t.Errorf("index %d: got %v, want %v (diff: %v)", i, actual[i], expected[i], diff)
				}
			}
		})
	}
}

func TestBaseMulAdd64(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{5, 4, 3, 2, 1}
	out := []float64{10, 20, 30, 40, 50}

	expected := make([]float64, len(out))
	copy(expected, out)
	for i := range expected {
		expected[i] += x[i] * y[i]
	}

	BaseMulAdd(x, y, out)

	for i := range out {
		diff := math.Abs(out[i] - expected[i])
		if diff > 1e-10 {
			t.Errorf("index %d: got %v, want %v (diff: %v)", i, out[i], expected[i], diff)
		}
	}
}

func TestBaseMulAddEmpty(t *testing.T) {
	// Should not panic
	BaseMulAdd[float32](nil, nil, nil)
	BaseMulAdd([]float32{}, []float32{}, []float32{})
}

func BenchmarkBaseMulAdd(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		x := make([]float32, size)
		y := make([]float32, size)
		out := make([]float32, size)
		for i := range x {
			x[i] = float32(i) * 0.01
			y[i] = float32(size-i) * 0.01
		}

		b.Run(fmt.Sprintf("f32_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				BaseMulAdd(x, y, out)
			}
		})
	}

	for _, size := range sizes {
		x := make([]float64, size)
		y := make([]float64, size)
		out := make([]float64, size)
		for i := range x {
			x[i] = float64(i) * 0.01
			y[i] = float64(size-i) * 0.01
		}

		b.Run(fmt.Sprintf("f64_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				BaseMulAdd(x, y, out)
			}
		})
	}
}
