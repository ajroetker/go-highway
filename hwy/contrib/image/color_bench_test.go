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

package image

import (
	"testing"
)

// Benchmark sizes for color transforms
var colorBenchSizes = []struct {
	name   string
	width  int
	height int
}{
	{"64x64", 64, 64},
	{"256x256", 256, 256},
	{"1080p", 1920, 1080},
	{"4K", 3840, 2160},
}

func BenchmarkForwardRCT(b *testing.B) {
	for _, size := range colorBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			r := NewImage[int32](size.width, size.height)
			g := NewImage[int32](size.width, size.height)
			blueImg := NewImage[int32](size.width, size.height)
			outY := NewImage[int32](size.width, size.height)
			outCb := NewImage[int32](size.width, size.height)
			outCr := NewImage[int32](size.width, size.height)

			// Fill with test data
			for y := 0; y < size.height; y++ {
				rRow := r.Row(y)
				gRow := g.Row(y)
				bRow := blueImg.Row(y)
				for x := 0; x < size.width; x++ {
					rRow[x] = int32((x + y*size.width) % 256)
					gRow[x] = int32((x + y*size.width + 85) % 256)
					bRow[x] = int32((x + y*size.width + 170) % 256)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				ForwardRCT(r, g, blueImg, outY, outCb, outCr)
			}
			// 3 reads + 3 writes, 4 bytes each
			b.SetBytes(int64(size.width * size.height * 4 * 6))
		})
	}
}

func BenchmarkInverseRCT(b *testing.B) {
	for _, size := range colorBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			yImg := NewImage[int32](size.width, size.height)
			cb := NewImage[int32](size.width, size.height)
			cr := NewImage[int32](size.width, size.height)
			outR := NewImage[int32](size.width, size.height)
			outG := NewImage[int32](size.width, size.height)
			outB := NewImage[int32](size.width, size.height)

			// Fill with test data
			for y := 0; y < size.height; y++ {
				yRow := yImg.Row(y)
				cbRow := cb.Row(y)
				crRow := cr.Row(y)
				for x := 0; x < size.width; x++ {
					yRow[x] = int32(128 + (x+y*size.width)%64)
					cbRow[x] = int32((x + y*size.width) % 128 - 64)
					crRow[x] = int32((x + y*size.width + 32) % 128 - 64)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				InverseRCT(yImg, cb, cr, outR, outG, outB)
			}
			b.SetBytes(int64(size.width * size.height * 4 * 6))
		})
	}
}

func BenchmarkForwardICT_Float32(b *testing.B) {
	for _, size := range colorBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			r := NewImage[float32](size.width, size.height)
			g := NewImage[float32](size.width, size.height)
			blueImg := NewImage[float32](size.width, size.height)
			outY := NewImage[float32](size.width, size.height)
			outCb := NewImage[float32](size.width, size.height)
			outCr := NewImage[float32](size.width, size.height)

			// Fill with test data
			for y := 0; y < size.height; y++ {
				rRow := r.Row(y)
				gRow := g.Row(y)
				bRow := blueImg.Row(y)
				for x := 0; x < size.width; x++ {
					rRow[x] = float32(x+y*size.width) / float32(size.width*size.height)
					gRow[x] = float32((x+y*size.width)+size.width*size.height/3) / float32(size.width*size.height*2)
					bRow[x] = float32((x+y*size.width)+size.width*size.height*2/3) / float32(size.width*size.height*2)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				ForwardICT(r, g, blueImg, outY, outCb, outCr)
			}
			b.SetBytes(int64(size.width * size.height * 4 * 6))
		})
	}
}

func BenchmarkInverseICT_Float32(b *testing.B) {
	for _, size := range colorBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			yImg := NewImage[float32](size.width, size.height)
			cb := NewImage[float32](size.width, size.height)
			cr := NewImage[float32](size.width, size.height)
			outR := NewImage[float32](size.width, size.height)
			outG := NewImage[float32](size.width, size.height)
			outB := NewImage[float32](size.width, size.height)

			// Fill with test data
			for y := 0; y < size.height; y++ {
				yRow := yImg.Row(y)
				cbRow := cb.Row(y)
				crRow := cr.Row(y)
				for x := 0; x < size.width; x++ {
					yRow[x] = float32(x+y*size.width) / float32(size.width*size.height)
					cbRow[x] = float32(x-size.width/2) / float32(size.width)
					crRow[x] = float32(y-size.height/2) / float32(size.height)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				InverseICT(yImg, cb, cr, outR, outG, outB)
			}
			b.SetBytes(int64(size.width * size.height * 4 * 6))
		})
	}
}

func BenchmarkForwardICT_Float64(b *testing.B) {
	for _, size := range colorBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			r := NewImage[float64](size.width, size.height)
			g := NewImage[float64](size.width, size.height)
			blueImg := NewImage[float64](size.width, size.height)
			outY := NewImage[float64](size.width, size.height)
			outCb := NewImage[float64](size.width, size.height)
			outCr := NewImage[float64](size.width, size.height)

			// Fill with test data
			for y := 0; y < size.height; y++ {
				rRow := r.Row(y)
				gRow := g.Row(y)
				bRow := blueImg.Row(y)
				for x := 0; x < size.width; x++ {
					rRow[x] = float64(x+y*size.width) / float64(size.width*size.height)
					gRow[x] = float64((x+y*size.width)+size.width*size.height/3) / float64(size.width*size.height*2)
					bRow[x] = float64((x+y*size.width)+size.width*size.height*2/3) / float64(size.width*size.height*2)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				ForwardICT(r, g, blueImg, outY, outCb, outCr)
			}
			// 8 bytes for float64
			b.SetBytes(int64(size.width * size.height * 8 * 6))
		})
	}
}

func BenchmarkInverseICT_Float64(b *testing.B) {
	for _, size := range colorBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			yImg := NewImage[float64](size.width, size.height)
			cb := NewImage[float64](size.width, size.height)
			cr := NewImage[float64](size.width, size.height)
			outR := NewImage[float64](size.width, size.height)
			outG := NewImage[float64](size.width, size.height)
			outB := NewImage[float64](size.width, size.height)

			// Fill with test data
			for y := 0; y < size.height; y++ {
				yRow := yImg.Row(y)
				cbRow := cb.Row(y)
				crRow := cr.Row(y)
				for x := 0; x < size.width; x++ {
					yRow[x] = float64(x+y*size.width) / float64(size.width*size.height)
					cbRow[x] = float64(x-size.width/2) / float64(size.width)
					crRow[x] = float64(y-size.height/2) / float64(size.height)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				InverseICT(yImg, cb, cr, outR, outG, outB)
			}
			b.SetBytes(int64(size.width * size.height * 8 * 6))
		})
	}
}
