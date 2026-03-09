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

//go:build !noasm && darwin && arm64

package vec

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/matvec"
	"github.com/ajroetker/go-highway/hwy/contrib/vec/asm"
)

// Benchmark to find optimal thresholds for batch operations.
// Compares:
// - GoAT loop: Loop over optimized single-vector GoAT assembly
// - MatVec FMOPA: Matrix-vector multiply using SME outer product tiles
// - SME streaming: Batch operation in SME streaming mode

func generateBenchData(count, dims int) (query []float32, data []float32, result []float32) {
	query = make([]float32, dims)
	data = make([]float32, count*dims)
	result = make([]float32, count)
	for i := range query {
		query[i] = rand.Float32()*2 - 1
	}
	for i := range data {
		data[i] = rand.Float32()*2 - 1
	}
	return
}

// batchDotGoATLoop is the GoAT single-vector loop implementation
func batchDotGoATLoop(query []float32, data []float32, dots []float32, count int, dims int) {
	for i := range count {
		offset := i * dims
		dots[i] = DotFloat32(query, data[offset:offset+dims])
	}
}

// batchDotMatVec uses MatVec FMOPA
func batchDotMatVec(query []float32, data []float32, dots []float32, count int, dims int) {
	matvec.MatVecFloat32(data, count, dims, query, dots)
}

// batchDotSMEStreaming uses SME streaming mode
func batchDotSMEStreaming(query []float32, data []float32, dots []float32, count int, dims int) {
	asm.BatchDotSME(query, data, dots, count, dims)
}

// batchL2GoATLoop is the GoAT single-vector loop implementation
func batchL2GoATLoop(query []float32, data []float32, distances []float32, count int, dims int) {
	for i := range count {
		offset := i * dims
		distances[i] = L2SquaredDistanceFloat32(query, data[offset:offset+dims])
	}
}

// batchL2SMEStreaming uses SME streaming mode
func batchL2SMEStreaming(query []float32, data []float32, distances []float32, count int, dims int) {
	asm.BatchL2SquaredSME(query, data, distances, count, dims)
}

// batchL2ViaFMOPA uses MatVec FMOPA with identity: L2²(a,b) = ||a||² + ||b||² - 2(a·b)
func batchL2ViaFMOPA(query []float32, data []float32, distances []float32, count int, dims int) {
	// Compute ||query||² once
	queryNormSq := SquaredNormFloat32(query[:dims])

	// Compute all dot products using MatVec FMOPA
	matvec.MatVecFloat32(data, count, dims, query, distances)

	// Transform: distances[i] = queryNormSq + ||data[i]||² - 2*distances[i]
	for i := range count {
		offset := i * dims
		dataNormSq := SquaredNormFloat32(data[offset : offset+dims])
		distances[i] = queryNormSq + dataNormSq - 2*distances[i]
	}
}

// BenchmarkBatchDotThreshold compares implementations across dimensions
// to find optimal crossover points for MatVec FMOPA.
func BenchmarkBatchDotThreshold(b *testing.B) {
	// Test dimensions around potential crossover points
	dims := []int{32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024}
	counts := []int{16, 32, 64, 128, 256, 512, 1024}

	for _, dim := range dims {
		for _, count := range counts {
			query, data, result := generateBenchData(count, dim)

			b.Run(dimCountName("GoAT", dim, count), func(b *testing.B) {
				for b.Loop() {
					batchDotGoATLoop(query, data, result, count, dim)
				}
			})

			// MatVec requires 16-aligned count for float32
			if count%16 == 0 {
				b.Run(dimCountName("MatVec", dim, count), func(b *testing.B) {
					for b.Loop() {
						batchDotMatVec(query, data, result, count, dim)
					}
				})
			}
		}
	}
}

// BenchmarkBatchDotSMEStreamingThreshold tests when SME streaming mode becomes beneficial
func BenchmarkBatchDotSMEStreamingThreshold(b *testing.B) {
	// Test large dimensions where SME streaming might help
	dims := []int{1024, 1536, 2048, 2560, 3000, 3072, 4096, 8192}
	counts := []int{100, 500, 1000}

	for _, dim := range dims {
		for _, count := range counts {
			query, data, result := generateBenchData(count, dim)

			b.Run(dimCountName("GoAT", dim, count), func(b *testing.B) {
				for b.Loop() {
					batchDotGoATLoop(query, data, result, count, dim)
				}
			})

			b.Run(dimCountName("SMEStream", dim, count), func(b *testing.B) {
				for b.Loop() {
					batchDotSMEStreaming(query, data, result, count, dim)
				}
			})
		}
	}
}

// BenchmarkBatchL2Threshold compares L2 implementations across dimensions
// Tests GoAT loop vs FMOPA-based approach (using L2² = ||a||² + ||b||² - 2(a·b))
func BenchmarkBatchL2Threshold(b *testing.B) {
	// Test dimensions around potential crossover points
	dims := []int{32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024}
	counts := []int{16, 32, 64, 128, 256, 512, 1024}

	for _, dim := range dims {
		for _, count := range counts {
			query, data, result := generateBenchData(count, dim)

			b.Run(dimCountName("GoAT", dim, count), func(b *testing.B) {
				for b.Loop() {
					batchL2GoATLoop(query, data, result, count, dim)
				}
			})

			// FMOPA requires 16-aligned count for float32
			if count%16 == 0 {
				b.Run(dimCountName("FMOPA", dim, count), func(b *testing.B) {
					for b.Loop() {
						batchL2ViaFMOPA(query, data, result, count, dim)
					}
				})
			}
		}
	}
}

// BenchmarkBatchL2SMEStreamingThreshold tests when SME streaming mode becomes beneficial for L2
func BenchmarkBatchL2SMEStreamingThreshold(b *testing.B) {
	dims := []int{1024, 1536, 2048, 2560, 3000, 3072, 4096, 8192}
	counts := []int{100, 500, 1000}

	for _, dim := range dims {
		for _, count := range counts {
			query, data, result := generateBenchData(count, dim)

			b.Run(dimCountName("GoAT", dim, count), func(b *testing.B) {
				for b.Loop() {
					batchL2GoATLoop(query, data, result, count, dim)
				}
			})

			b.Run(dimCountName("SMEStream", dim, count), func(b *testing.B) {
				for b.Loop() {
					batchL2SMEStreaming(query, data, result, count, dim)
				}
			})
		}
	}
}

func dimCountName(impl string, dim, count int) string {
	return impl + "/" + strconv.Itoa(dim) + "d_n" + strconv.Itoa(count)
}

