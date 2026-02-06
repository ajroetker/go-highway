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

package wavelet

import (
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/image"
)

// 1D benchmark sizes
var bench1DSizes = []int{64, 256, 1024, 4096}

// 2D benchmark sizes
var bench2DSizes = []struct {
	name   string
	width  int
	height int
}{
	{"256x256", 256, 256},
	{"1080p", 1920, 1080},
	{"4K", 3840, 2160},
}

func BenchmarkSynthesize53_1D(b *testing.B) {
	for _, size := range bench1DSizes {
		b.Run(benchSizeName(size), func(b *testing.B) {
			data := make([]int32, size)
			for i := range data {
				data[i] = int32(i % 256)
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Synthesize53(data, 0)
			}
			b.SetBytes(int64(size * 4)) // int32 = 4 bytes
		})
	}
}

func BenchmarkAnalyze53_1D(b *testing.B) {
	for _, size := range bench1DSizes {
		b.Run(benchSizeName(size), func(b *testing.B) {
			data := make([]int32, size)
			for i := range data {
				data[i] = int32(i % 256)
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Analyze53(data, 0)
			}
			b.SetBytes(int64(size * 4))
		})
	}
}

func BenchmarkSynthesize97_1D_Float32(b *testing.B) {
	for _, size := range bench1DSizes {
		b.Run(benchSizeName(size), func(b *testing.B) {
			data := make([]float32, size)
			for i := range data {
				data[i] = float32(i) / float32(size)
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Synthesize97(data, 0)
			}
			b.SetBytes(int64(size * 4))
		})
	}
}

func BenchmarkAnalyze97_1D_Float32(b *testing.B) {
	for _, size := range bench1DSizes {
		b.Run(benchSizeName(size), func(b *testing.B) {
			data := make([]float32, size)
			for i := range data {
				data[i] = float32(i) / float32(size)
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Analyze97(data, 0)
			}
			b.SetBytes(int64(size * 4))
		})
	}
}

func BenchmarkSynthesize2D_53(b *testing.B) {
	for _, size := range bench2DSizes {
		b.Run(size.name, func(b *testing.B) {
			img := image.NewImage[int32](size.width, size.height)
			for y := 0; y < size.height; y++ {
				row := img.Row(y)
				for x := 0; x < size.width; x++ {
					row[x] = int32((x + y) % 256)
				}
			}

			// Pre-analyze for synthesis benchmark
			Analyze2D_53(img, 3, zeroPhase)

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Synthesize2D_53(img, 3, zeroPhase)
				// Re-analyze for next iteration
				if i < b.N-1 {
					Analyze2D_53(img, 3, zeroPhase)
				}
			}
			b.SetBytes(int64(size.width * size.height * 4))
		})
	}
}

func BenchmarkAnalyze2D_53(b *testing.B) {
	for _, size := range bench2DSizes {
		b.Run(size.name, func(b *testing.B) {
			img := image.NewImage[int32](size.width, size.height)
			for y := 0; y < size.height; y++ {
				row := img.Row(y)
				for x := 0; x < size.width; x++ {
					row[x] = int32((x + y) % 256)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Analyze2D_53(img, 3, zeroPhase)
				// Re-synthesize for next iteration
				if i < b.N-1 {
					Synthesize2D_53(img, 3, zeroPhase)
				}
			}
			b.SetBytes(int64(size.width * size.height * 4))
		})
	}
}

func BenchmarkSynthesize2D_97_Float32(b *testing.B) {
	for _, size := range bench2DSizes {
		b.Run(size.name, func(b *testing.B) {
			img := image.NewImage[float32](size.width, size.height)
			for y := 0; y < size.height; y++ {
				row := img.Row(y)
				for x := 0; x < size.width; x++ {
					row[x] = float32((x + y) % 256) / 255.0
				}
			}

			// Pre-analyze for synthesis benchmark
			Analyze2D_97(img, 3, zeroPhase)

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Synthesize2D_97(img, 3, zeroPhase)
				if i < b.N-1 {
					Analyze2D_97(img, 3, zeroPhase)
				}
			}
			b.SetBytes(int64(size.width * size.height * 4))
		})
	}
}

func BenchmarkAnalyze2D_97_Float32(b *testing.B) {
	for _, size := range bench2DSizes {
		b.Run(size.name, func(b *testing.B) {
			img := image.NewImage[float32](size.width, size.height)
			for y := 0; y < size.height; y++ {
				row := img.Row(y)
				for x := 0; x < size.width; x++ {
					row[x] = float32((x + y) % 256) / 255.0
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Analyze2D_97(img, 3, zeroPhase)
				if i < b.N-1 {
					Synthesize2D_97(img, 3, zeroPhase)
				}
			}
			b.SetBytes(int64(size.width * size.height * 4))
		})
	}
}

func benchSizeName(size int) string {
	switch size {
	case 64:
		return "64"
	case 256:
		return "256"
	case 1024:
		return "1024"
	case 4096:
		return "4096"
	default:
		return "unknown"
	}
}
