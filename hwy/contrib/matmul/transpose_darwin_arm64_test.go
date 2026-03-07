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

package matmul

import (
	"fmt"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// BenchmarkTransposePrefill benchmarks transpose for prefill-like shapes
// (small M, large K) that currently fall through to scalar due to both-dims
// threshold checks. Compares scalar, NEON, and SME directly.
func BenchmarkTransposePrefill(b *testing.B) {
	type shape struct {
		m, k int
	}
	shapes := []shape{
		{16, 256}, {16, 640}, {16, 1024}, {16, 2048},
		{13, 640}, {10, 2048},
		{4, 1024}, {8, 512},
	}

	for _, s := range shapes {
		src := make([]float32, s.m*s.k)
		dst := make([]float32, s.k*s.m)
		for i := range src {
			src[i] = float32(i)
		}
		bytes := int64(s.m * s.k * 4 * 2)

		b.Run(fmt.Sprintf("Scalar/%dx%d", s.m, s.k), func(b *testing.B) {
			b.SetBytes(bytes)
			for i := 0; i < b.N; i++ {
				transposeScalar(src, s.m, s.k, dst)
			}
		})

		b.Run(fmt.Sprintf("NEON/%dx%d", s.m, s.k), func(b *testing.B) {
			b.SetBytes(bytes)
			for i := 0; i < b.N; i++ {
				transpose2DAsmF32(src, s.m, s.k, dst)
			}
		})

		if hwy.HasSME() {
			b.Run(fmt.Sprintf("SME/%dx%d", s.m, s.k), func(b *testing.B) {
				b.SetBytes(bytes)
				for i := 0; i < b.N; i++ {
					asm.TransposeSMEF32(src, s.m, s.k, dst)
				}
			})
		}
	}
}

// BenchmarkTransposePrefillF64 benchmarks f64 transpose for prefill-like shapes.
func BenchmarkTransposePrefillF64(b *testing.B) {
	type shape struct {
		m, k int
	}
	shapes := []shape{
		{8, 256}, {8, 640}, {16, 256}, {16, 640},
	}

	for _, s := range shapes {
		src := make([]float64, s.m*s.k)
		dst := make([]float64, s.k*s.m)
		for i := range src {
			src[i] = float64(i)
		}
		bytes := int64(s.m * s.k * 8 * 2)

		b.Run(fmt.Sprintf("Scalar/%dx%d", s.m, s.k), func(b *testing.B) {
			b.SetBytes(bytes)
			for i := 0; i < b.N; i++ {
				transposeScalar(src, s.m, s.k, dst)
			}
		})

		b.Run(fmt.Sprintf("NEON/%dx%d", s.m, s.k), func(b *testing.B) {
			b.SetBytes(bytes)
			for i := 0; i < b.N; i++ {
				transpose2DAsmF64(src, s.m, s.k, dst)
			}
		})

		if hwy.HasSME() {
			b.Run(fmt.Sprintf("SME/%dx%d", s.m, s.k), func(b *testing.B) {
				b.SetBytes(bytes)
				for i := 0; i < b.N; i++ {
					asm.TransposeSMEF64(src, s.m, s.k, dst)
				}
			})
		}
	}
}
