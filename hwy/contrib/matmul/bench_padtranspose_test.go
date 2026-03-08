package matmul

import (
	"math/rand"
	"testing"
)

func BenchmarkMatMulNonAligned(b *testing.B) {
	type tc struct {
		m, n, k int
	}
	cases := []tc{
		{33, 33, 33},
		{50, 50, 50},
		{100, 100, 100},
		{200, 200, 200},
		{17, 128, 64},
		// LLM-like shapes: small M (prefill tokens), large K/N (hidden dim)
		{10, 4096, 4096},
		{17, 4096, 4096},
		{33, 4096, 4096},
	}

	for _, c := range cases {
		m, n, k := c.m, c.n, c.k
		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		cMat := make([]float32, m*n)
		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}
		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(m)+"x"+sizeStr(n)+"x"+sizeStr(k), func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMul(a, bMat, cMat, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}
