//go:build !noasm && arm64

package asm

import (
	"testing"
)

// BenchmarkFloat16x8MulAddReturn benchmarks Float16x8 MulAdd with return value.
func BenchmarkFloat16x8MulAddReturn(b *testing.B) {
	vA := ZeroFloat16x8()
	vB := ZeroFloat16x8()
	acc := ZeroFloat16x8()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		acc = vA.MulAdd(vB, acc)
	}
	b.StopTimer()

	// Prevent compiler from optimizing away
	_ = acc
}

// BenchmarkFloat16x8MulAddInPlace benchmarks Float16x8 MulAddAcc in-place.
func BenchmarkFloat16x8MulAddInPlace(b *testing.B) {
	vA := ZeroFloat16x8()
	vB := ZeroFloat16x8()
	acc := ZeroFloat16x8()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vA.MulAddAcc(vB, &acc)
	}
	b.StopTimer()

	// Prevent compiler from optimizing away
	_ = acc
}

// BenchmarkFloat16x8AddReturn benchmarks Float16x8 Add with return value.
func BenchmarkFloat16x8AddReturn(b *testing.B) {
	vA := ZeroFloat16x8()
	vB := ZeroFloat16x8()
	var result Float16x8

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result = vA.Add(vB)
	}
	b.StopTimer()

	_ = result
}

// BenchmarkFloat16x8AddInPlace benchmarks Float16x8 AddInto in-place.
func BenchmarkFloat16x8AddInPlace(b *testing.B) {
	vA := ZeroFloat16x8()
	vB := ZeroFloat16x8()
	var result Float16x8

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vA.AddInto(vB, &result)
	}
	b.StopTimer()

	_ = result
}

// BenchmarkBFloat16x8MulAddReturn benchmarks BFloat16x8 MulAdd with return value.
func BenchmarkBFloat16x8MulAddReturn(b *testing.B) {
	vA := ZeroBFloat16x8()
	vB := ZeroBFloat16x8()
	acc := ZeroBFloat16x8()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		acc = vA.MulAdd(vB, acc)
	}
	b.StopTimer()

	_ = acc
}

// BenchmarkBFloat16x8MulAddInPlace benchmarks BFloat16x8 MulAddAcc in-place.
func BenchmarkBFloat16x8MulAddInPlace(b *testing.B) {
	vA := ZeroBFloat16x8()
	vB := ZeroBFloat16x8()
	acc := ZeroBFloat16x8()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vA.MulAddAcc(vB, &acc)
	}
	b.StopTimer()

	_ = acc
}

// BenchmarkBFloat16x8BFDot benchmarks BFloat16x8 BFDOT with F32 accumulator.
// This is the preferred pattern for ML workloads.
func BenchmarkBFloat16x8BFDot(b *testing.B) {
	vA := ZeroBFloat16x8()
	vB := ZeroBFloat16x8()
	acc := ZeroFloat32x4()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vA.BFDotF32Acc(vB, &acc)
	}
	b.StopTimer()

	_ = acc
}
