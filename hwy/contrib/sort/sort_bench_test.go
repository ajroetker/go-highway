package sort

import (
	"math/rand"
	"slices"
	"testing"
)

// Generate random data for benchmarks
func generateFloat32(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32() * 1000
	}
	return data
}

func generateFloat64(n int) []float64 {
	data := make([]float64, n)
	for i := range data {
		data[i] = rand.Float64() * 1000
	}
	return data
}

func generateInt32(n int) []int32 {
	data := make([]int32, n)
	for i := range data {
		data[i] = rand.Int31n(10000) - 5000
	}
	return data
}

func generateInt64(n int) []int64 {
	data := make([]int64, n)
	for i := range data {
		data[i] = rand.Int63n(10000) - 5000
	}
	return data
}

// Float32 benchmarks
func BenchmarkSort_Float32_100(b *testing.B) {
	benchmarkSortFloat32(b, 100)
}

func BenchmarkSort_Float32_1000(b *testing.B) {
	benchmarkSortFloat32(b, 1000)
}

func BenchmarkSort_Float32_10000(b *testing.B) {
	benchmarkSortFloat32(b, 10000)
}

func BenchmarkSort_Float32_100000(b *testing.B) {
	benchmarkSortFloat32(b, 100000)
}

func benchmarkSortFloat32(b *testing.B, n int) {
	// Generate reference data
	ref := generateFloat32(n)
	data := make([]float32, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		Sort(data)
	}
}

// Float64 benchmarks
func BenchmarkSort_Float64_100(b *testing.B) {
	benchmarkSortFloat64(b, 100)
}

func BenchmarkSort_Float64_1000(b *testing.B) {
	benchmarkSortFloat64(b, 1000)
}

func BenchmarkSort_Float64_10000(b *testing.B) {
	benchmarkSortFloat64(b, 10000)
}

func BenchmarkSort_Float64_100000(b *testing.B) {
	benchmarkSortFloat64(b, 100000)
}

func benchmarkSortFloat64(b *testing.B, n int) {
	ref := generateFloat64(n)
	data := make([]float64, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		Sort(data)
	}
}

// Int32 benchmarks
func BenchmarkSort_Int32_100(b *testing.B) {
	benchmarkSortInt32(b, 100)
}

func BenchmarkSort_Int32_1000(b *testing.B) {
	benchmarkSortInt32(b, 1000)
}

func BenchmarkSort_Int32_10000(b *testing.B) {
	benchmarkSortInt32(b, 10000)
}

func BenchmarkSort_Int32_100000(b *testing.B) {
	benchmarkSortInt32(b, 100000)
}

func benchmarkSortInt32(b *testing.B, n int) {
	ref := generateInt32(n)
	data := make([]int32, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		Sort(data)
	}
}

// VQSort Int32 benchmarks (to compare with RadixSort for integers)
func BenchmarkVQSort_Int32_100(b *testing.B) {
	benchmarkVQSortInt32(b, 100)
}

func BenchmarkVQSort_Int32_1000(b *testing.B) {
	benchmarkVQSortInt32(b, 1000)
}

func BenchmarkVQSort_Int32_10000(b *testing.B) {
	benchmarkVQSortInt32(b, 10000)
}

func BenchmarkVQSort_Int32_100000(b *testing.B) {
	benchmarkVQSortInt32(b, 100000)
}

func benchmarkVQSortInt32(b *testing.B, n int) {
	ref := generateInt32(n)
	data := make([]int32, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		VQSort(data)
	}
}

// Int64 benchmarks
func BenchmarkSort_Int64_100(b *testing.B) {
	benchmarkSortInt64(b, 100)
}

func BenchmarkSort_Int64_1000(b *testing.B) {
	benchmarkSortInt64(b, 1000)
}

func BenchmarkSort_Int64_10000(b *testing.B) {
	benchmarkSortInt64(b, 10000)
}

func BenchmarkSort_Int64_100000(b *testing.B) {
	benchmarkSortInt64(b, 100000)
}

func benchmarkSortInt64(b *testing.B, n int) {
	ref := generateInt64(n)
	data := make([]int64, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		Sort(data)
	}
}

// Standard library comparison benchmarks
func BenchmarkStdlib_Float32_100(b *testing.B) {
	benchmarkStdlibFloat32(b, 100)
}

func BenchmarkStdlib_Float32_1000(b *testing.B) {
	benchmarkStdlibFloat32(b, 1000)
}

func BenchmarkStdlib_Float32_10000(b *testing.B) {
	benchmarkStdlibFloat32(b, 10000)
}

func BenchmarkStdlib_Float32_100000(b *testing.B) {
	benchmarkStdlibFloat32(b, 100000)
}

func benchmarkStdlibFloat32(b *testing.B, n int) {
	ref := generateFloat32(n)
	data := make([]float32, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		slices.Sort(data)
	}
}

func BenchmarkStdlib_Float64_100(b *testing.B) {
	benchmarkStdlibFloat64(b, 100)
}

func BenchmarkStdlib_Float64_1000(b *testing.B) {
	benchmarkStdlibFloat64(b, 1000)
}

func BenchmarkStdlib_Float64_10000(b *testing.B) {
	benchmarkStdlibFloat64(b, 10000)
}

func BenchmarkStdlib_Float64_100000(b *testing.B) {
	benchmarkStdlibFloat64(b, 100000)
}

func benchmarkStdlibFloat64(b *testing.B, n int) {
	ref := generateFloat64(n)
	data := make([]float64, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		slices.Sort(data)
	}
}

func BenchmarkStdlib_Int32_100(b *testing.B) {
	benchmarkStdlibInt32(b, 100)
}

func BenchmarkStdlib_Int32_1000(b *testing.B) {
	benchmarkStdlibInt32(b, 1000)
}

func BenchmarkStdlib_Int32_10000(b *testing.B) {
	benchmarkStdlibInt32(b, 10000)
}

func BenchmarkStdlib_Int32_100000(b *testing.B) {
	benchmarkStdlibInt32(b, 100000)
}

func benchmarkStdlibInt32(b *testing.B, n int) {
	ref := generateInt32(n)
	data := make([]int32, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		slices.Sort(data)
	}
}

func BenchmarkStdlib_Int64_100(b *testing.B) {
	benchmarkStdlibInt64(b, 100)
}

func BenchmarkStdlib_Int64_1000(b *testing.B) {
	benchmarkStdlibInt64(b, 1000)
}

func BenchmarkStdlib_Int64_10000(b *testing.B) {
	benchmarkStdlibInt64(b, 10000)
}

func BenchmarkStdlib_Int64_100000(b *testing.B) {
	benchmarkStdlibInt64(b, 100000)
}

func benchmarkStdlibInt64(b *testing.B, n int) {
	ref := generateInt64(n)
	data := make([]int64, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		slices.Sort(data)
	}
}

// Partition benchmarks
func BenchmarkPartition3Way_Float32_10000(b *testing.B) {
	ref := generateFloat32(10000)
	data := make([]float32, len(ref))
	pivot := PivotSampled(ref)

	for b.Loop() {
		copy(data, ref)
		Partition3Way(data, pivot)
	}
}

func BenchmarkPartition_Float32_10000(b *testing.B) {
	ref := generateFloat32(10000)
	data := make([]float32, len(ref))
	pivot := PivotSampled(ref)

	for b.Loop() {
		copy(data, ref)
		Partition(data, pivot)
	}
}

// Compress partition benchmarks
func BenchmarkCompressPartition3Way_Float32_10000(b *testing.B) {
	ref := generateFloat32(10000)
	data := make([]float32, len(ref))
	pivot := PivotSampled(ref)

	for b.Loop() {
		copy(data, ref)
		CompressPartition3WayFloat32(data, pivot)
	}
}

// Scalar baseline for partition comparison
func scalarPartition3WayBench(data []float32, pivot float32) (int, int) {
	lt := 0
	gt := len(data)
	i := 0
	for i < gt {
		if data[i] < pivot {
			data[lt], data[i] = data[i], data[lt]
			lt++
			i++
		} else if data[i] > pivot {
			gt--
			data[i], data[gt] = data[gt], data[i]
		} else {
			i++
		}
	}
	return lt, gt
}

func BenchmarkScalarPartition3Way_Float32_10000(b *testing.B) {
	ref := generateFloat32(10000)
	data := make([]float32, len(ref))
	pivot := PivotSampled(ref)

	for b.Loop() {
		copy(data, ref)
		scalarPartition3WayBench(data, pivot)
	}
}

// Radix sort benchmarks - compare against stdlib

func BenchmarkRadixSortInt32_100(b *testing.B) {
	benchmarkRadixSortInt32(b, 100)
}

func BenchmarkRadixSortInt32_1000(b *testing.B) {
	benchmarkRadixSortInt32(b, 1000)
}

func BenchmarkRadixSortInt32_10000(b *testing.B) {
	benchmarkRadixSortInt32(b, 10000)
}

func BenchmarkRadixSortInt32_100000(b *testing.B) {
	benchmarkRadixSortInt32(b, 100000)
}

func BenchmarkRadixSortInt32_1000000(b *testing.B) {
	benchmarkRadixSortInt32(b, 1000000)
}

func benchmarkRadixSortInt32(b *testing.B, n int) {
	ref := generateInt32(n)
	data := make([]int32, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		RadixSort(data)
	}
}

func BenchmarkStdlibSortInt32_100(b *testing.B) {
	benchmarkStdlibSortInt32(b, 100)
}

func BenchmarkStdlibSortInt32_1000(b *testing.B) {
	benchmarkStdlibSortInt32(b, 1000)
}

func BenchmarkStdlibSortInt32_10000(b *testing.B) {
	benchmarkStdlibSortInt32(b, 10000)
}

func BenchmarkStdlibSortInt32_100000(b *testing.B) {
	benchmarkStdlibSortInt32(b, 100000)
}

func BenchmarkStdlibSortInt32_1000000(b *testing.B) {
	benchmarkStdlibSortInt32(b, 1000000)
}

func benchmarkStdlibSortInt32(b *testing.B, n int) {
	ref := generateInt32(n)
	data := make([]int32, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		slices.Sort(data)
	}
}

// Int64 radix sort benchmarks
func BenchmarkRadixSortInt64_10000(b *testing.B) {
	benchmarkRadixSortInt64(b, 10000)
}

func BenchmarkRadixSortInt64_20000(b *testing.B) {
	benchmarkRadixSortInt64(b, 20000)
}

func BenchmarkRadixSortInt64_30000(b *testing.B) {
	benchmarkRadixSortInt64(b, 30000)
}

func BenchmarkRadixSortInt64_50000(b *testing.B) {
	benchmarkRadixSortInt64(b, 50000)
}

func BenchmarkRadixSortInt64_100000(b *testing.B) {
	benchmarkRadixSortInt64(b, 100000)
}

func benchmarkRadixSortInt64(b *testing.B, n int) {
	ref := generateInt64(n)
	data := make([]int64, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		RadixSort(data)
	}
}

func BenchmarkStdlibSortInt64_10000(b *testing.B) {
	benchmarkStdlibSortInt64(b, 10000)
}

func BenchmarkStdlibSortInt64_20000(b *testing.B) {
	benchmarkStdlibSortInt64(b, 20000)
}

func BenchmarkStdlibSortInt64_30000(b *testing.B) {
	benchmarkStdlibSortInt64(b, 30000)
}

func BenchmarkStdlibSortInt64_50000(b *testing.B) {
	benchmarkStdlibSortInt64(b, 50000)
}

func BenchmarkStdlibSortInt64_100000(b *testing.B) {
	benchmarkStdlibSortInt64(b, 100000)
}

func benchmarkStdlibSortInt64(b *testing.B, n int) {
	ref := generateInt64(n)
	data := make([]int64, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(data, ref)
		slices.Sort(data)
	}
}
