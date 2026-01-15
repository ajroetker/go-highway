package image

import (
	"testing"
)

// Benchmark sizes
var benchSizes = []struct {
	name   string
	width  int
	height int
}{
	{"64x64", 64, 64},
	{"256x256", 256, 256},
	{"1080p", 1920, 1080},
	{"4K", 3840, 2160},
}

func BenchmarkBrightnessContrast(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			img := NewImage[float32](size.width, size.height)
			out := NewImage[float32](size.width, size.height)

			// Fill with test data
			for y := 0; y < size.height; y++ {
				row := img.Row(y)
				for x := 0; x < size.width; x++ {
					row[x] = float32(x+y) / float32(size.width+size.height)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				BrightnessContrast(img, out, 1.5, 0.1)
			}
			b.SetBytes(int64(size.width * size.height * 4 * 2)) // read + write
		})
	}
}

func BenchmarkClampImage(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			img := NewImage[float32](size.width, size.height)
			out := NewImage[float32](size.width, size.height)

			for y := 0; y < size.height; y++ {
				row := img.Row(y)
				for x := 0; x < size.width; x++ {
					row[x] = float32(x-size.width/2) / float32(size.width)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				ClampImage(img, out, 0.0, 1.0)
			}
			b.SetBytes(int64(size.width * size.height * 4 * 2))
		})
	}
}

func BenchmarkThreshold(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			img := NewImage[float32](size.width, size.height)
			out := NewImage[float32](size.width, size.height)

			for y := 0; y < size.height; y++ {
				row := img.Row(y)
				for x := 0; x < size.width; x++ {
					row[x] = float32(x) / float32(size.width)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Threshold(img, out, 0.5, 0.0, 1.0)
			}
			b.SetBytes(int64(size.width * size.height * 4 * 2))
		})
	}
}

func BenchmarkScale(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			img := NewImage[float32](size.width, size.height)
			out := NewImage[float32](size.width, size.height)

			for y := 0; y < size.height; y++ {
				row := img.Row(y)
				for x := 0; x < size.width; x++ {
					row[x] = float32(x) / float32(size.width)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Scale(img, out, 2.5)
			}
			b.SetBytes(int64(size.width * size.height * 4 * 2))
		})
	}
}

func BenchmarkAbs(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			img := NewImage[float32](size.width, size.height)
			out := NewImage[float32](size.width, size.height)

			for y := 0; y < size.height; y++ {
				row := img.Row(y)
				for x := 0; x < size.width; x++ {
					row[x] = float32(x-size.width/2) / float32(size.width)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Abs(img, out)
			}
			b.SetBytes(int64(size.width * size.height * 4 * 2))
		})
	}
}

func BenchmarkMinImage(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			a := NewImage[float32](size.width, size.height)
			bb := NewImage[float32](size.width, size.height)
			out := NewImage[float32](size.width, size.height)

			for y := 0; y < size.height; y++ {
				rowA := a.Row(y)
				rowB := bb.Row(y)
				for x := 0; x < size.width; x++ {
					rowA[x] = float32(x) / float32(size.width)
					rowB[x] = float32(size.width-x) / float32(size.width)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				MinImage(a, bb, out)
			}
			b.SetBytes(int64(size.width * size.height * 4 * 3)) // 2 reads + 1 write
		})
	}
}

func BenchmarkGamma(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			img := NewImage[float32](size.width, size.height)
			out := NewImage[float32](size.width, size.height)

			for y := 0; y < size.height; y++ {
				row := img.Row(y)
				for x := 0; x < size.width; x++ {
					row[x] = float32(x+1) / float32(size.width+1)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Gamma(img, out, 2.2)
			}
			b.SetBytes(int64(size.width * size.height * 4 * 2))
		})
	}
}
