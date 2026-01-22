package matmul

import (
	"fmt"
	"testing"
	"github.com/ajroetker/go-highway/hwy"
)

func BenchmarkMatMulFloat16(b *testing.B) {
	b.Logf("Dispatch level: %s, HasSME: %v", hwy.CurrentName(), hwy.HasSME())
	sizes := []int{64, 128, 256, 512}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := make([]hwy.Float16, n*n)
			bb := make([]hwy.Float16, n*n)
			c := make([]hwy.Float16, n*n)
			for i := range a {
				a[i] = hwy.Float32ToFloat16(float32(i%7) + 0.5)
				bb[i] = hwy.Float32ToFloat16(float32(i%11) + 0.25)
			}
			flops := float64(2*n*n*n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulAuto(a, bb, c, n, n, n)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func BenchmarkMatMulBFloat16(b *testing.B) {
	b.Logf("Dispatch level: %s, HasSME: %v", hwy.CurrentName(), hwy.HasSME())
	sizes := []int{64, 128, 256, 512}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := make([]hwy.BFloat16, n*n)
			bb := make([]hwy.BFloat16, n*n)
			c := make([]hwy.BFloat16, n*n)
			for i := range a {
				a[i] = hwy.Float32ToBFloat16(float32(i%7) + 0.5)
				bb[i] = hwy.Float32ToBFloat16(float32(i%11) + 0.25)
			}
			flops := float64(2*n*n*n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulAuto(a, bb, c, n, n, n)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func BenchmarkParallelMatMulFloat16(b *testing.B) {
	b.Logf("Dispatch level: %s, HasSME: %v", hwy.CurrentName(), hwy.HasSME())
	sizes := []int{256, 512, 1024}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := make([]hwy.Float16, n*n)
			bb := make([]hwy.Float16, n*n)
			c := make([]hwy.Float16, n*n)
			for i := range a {
				a[i] = hwy.Float32ToFloat16(float32(i%7) + 0.5)
				bb[i] = hwy.Float32ToFloat16(float32(i%11) + 0.25)
			}
			flops := float64(2*n*n*n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ParallelMatMul(a, bb, c, n, n, n)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func BenchmarkParallelMatMulBFloat16(b *testing.B) {
	b.Logf("Dispatch level: %s, HasSME: %v", hwy.CurrentName(), hwy.HasSME())
	sizes := []int{256, 512, 1024}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := make([]hwy.BFloat16, n*n)
			bb := make([]hwy.BFloat16, n*n)
			c := make([]hwy.BFloat16, n*n)
			for i := range a {
				a[i] = hwy.Float32ToBFloat16(float32(i%7) + 0.5)
				bb[i] = hwy.Float32ToBFloat16(float32(i%11) + 0.25)
			}
			flops := float64(2*n*n*n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ParallelMatMul(a, bb, c, n, n, n)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}
