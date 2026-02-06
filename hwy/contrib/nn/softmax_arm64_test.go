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

//go:build !noasm && arm64

package nn

import (
	"fmt"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/nn/asm"
)

// softmaxHandwrittenF32 wraps the handwritten C/assembly softmax.
func softmaxHandwrittenF32(input, output []float32) {
	asm.SoftmaxNeonF32(input, output, len(input))
}

// BenchmarkSoftmaxHandwrittenVsGenerated compares the handwritten C/assembly
// softmax (3-pass fused) against the hwygen-generated Go SIMD version (5-pass).
func BenchmarkSoftmaxHandwrittenVsGenerated(b *testing.B) {
	sizes := []int{8, 64, 256, 1024, 4096}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i) * 0.1
		}

		// hwygen-generated SIMD (uses Go's simd package, 5-pass)
		b.Run(fmt.Sprintf("Generated/%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Softmax(input, output)
			}
		})

		// Handwritten C/assembly (3-pass fused via GOAT)
		b.Run(fmt.Sprintf("Handwritten/%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				softmaxHandwrittenF32(input, output)
			}
		})
	}
}
