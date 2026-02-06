//go:build !noasm && arm64

package asm

import (
	"testing"
)

// BenchmarkMulAddReturn benchmarks the traditional return-value approach.
// Each MulAdd call returns a new [16]byte that gets allocated on stack.
func BenchmarkMulAddReturn(b *testing.B) {
	vA := BroadcastFloat32x4(1.5)
	vB := BroadcastFloat32x4(2.0)
	acc := ZeroFloat32x4()

	for b.Loop() {
		// Traditional approach: returns a value, causes stack allocation
		acc = vA.MulAdd(vB, acc)
	}
	b.StopTimer()

	// Prevent compiler from optimizing away
	if acc.Get(0) == 0 {
		b.Fatal("unexpected zero")
	}
}

// BenchmarkMulAddInPlace benchmarks the new in-place approach.
// MulAddAcc modifies acc directly without return allocation.
func BenchmarkMulAddInPlace(b *testing.B) {
	vA := BroadcastFloat32x4(1.5)
	vB := BroadcastFloat32x4(2.0)
	acc := ZeroFloat32x4()

	for b.Loop() {
		// New approach: modifies acc in-place, no return allocation
		vA.MulAddAcc(vB, &acc)
	}
	b.StopTimer()

	// Prevent compiler from optimizing away
	if acc.Get(0) == 0 {
		b.Fatal("unexpected zero")
	}
}

// BenchmarkAddReturn benchmarks the traditional Add with return value.
func BenchmarkAddReturn(b *testing.B) {
	vA := BroadcastFloat32x4(1.0)
	vB := BroadcastFloat32x4(2.0)
	var result Float32x4

	for b.Loop() {
		result = vA.Add(vB)
	}
	b.StopTimer()

	if result.Get(0) == 0 {
		b.Fatal("unexpected zero")
	}
}

// BenchmarkAddInPlace benchmarks the new AddInto without return allocation.
func BenchmarkAddInPlace(b *testing.B) {
	vA := BroadcastFloat32x4(1.0)
	vB := BroadcastFloat32x4(2.0)
	var result Float32x4

	for b.Loop() {
		vA.AddInto(vB, &result)
	}
	b.StopTimer()

	if result.Get(0) == 0 {
		b.Fatal("unexpected zero")
	}
}

// BenchmarkMulAddChainReturn benchmarks a chain of MulAdd ops (typical inner loop pattern).
func BenchmarkMulAddChainReturn(b *testing.B) {
	v0 := BroadcastFloat32x4(1.1)
	v1 := BroadcastFloat32x4(1.2)
	v2 := BroadcastFloat32x4(1.3)
	v3 := BroadcastFloat32x4(1.4)
	a0 := BroadcastFloat32x4(2.0)
	a1 := BroadcastFloat32x4(2.1)
	a2 := BroadcastFloat32x4(2.2)
	a3 := BroadcastFloat32x4(2.3)
	acc0 := ZeroFloat32x4()
	acc1 := ZeroFloat32x4()
	acc2 := ZeroFloat32x4()
	acc3 := ZeroFloat32x4()

	for b.Loop() {
		// 4x unrolled MulAdd - typical matmul inner loop
		acc0 = v0.MulAdd(a0, acc0)
		acc1 = v1.MulAdd(a1, acc1)
		acc2 = v2.MulAdd(a2, acc2)
		acc3 = v3.MulAdd(a3, acc3)
	}
	b.StopTimer()

	if acc0.Get(0)+acc1.Get(0)+acc2.Get(0)+acc3.Get(0) == 0 {
		b.Fatal("unexpected zero")
	}
}

// BenchmarkMulAddChainInPlace benchmarks a chain of in-place MulAdd ops.
func BenchmarkMulAddChainInPlace(b *testing.B) {
	v0 := BroadcastFloat32x4(1.1)
	v1 := BroadcastFloat32x4(1.2)
	v2 := BroadcastFloat32x4(1.3)
	v3 := BroadcastFloat32x4(1.4)
	a0 := BroadcastFloat32x4(2.0)
	a1 := BroadcastFloat32x4(2.1)
	a2 := BroadcastFloat32x4(2.2)
	a3 := BroadcastFloat32x4(2.3)
	acc0 := ZeroFloat32x4()
	acc1 := ZeroFloat32x4()
	acc2 := ZeroFloat32x4()
	acc3 := ZeroFloat32x4()

	for b.Loop() {
		// 4x unrolled MulAdd with in-place - no return allocations
		v0.MulAddAcc(a0, &acc0)
		v1.MulAddAcc(a1, &acc1)
		v2.MulAddAcc(a2, &acc2)
		v3.MulAddAcc(a3, &acc3)
	}
	b.StopTimer()

	if acc0.Get(0)+acc1.Get(0)+acc2.Get(0)+acc3.Get(0) == 0 {
		b.Fatal("unexpected zero")
	}
}

// BenchmarkFloat64MulAddReturn benchmarks Float64x2 MulAdd with return.
func BenchmarkFloat64MulAddReturn(b *testing.B) {
	vA := BroadcastFloat64x2(1.5)
	vB := BroadcastFloat64x2(2.0)
	acc := ZeroFloat64x2()

	for b.Loop() {
		acc = vA.MulAdd(vB, acc)
	}
	b.StopTimer()

	if acc.Get(0) == 0 {
		b.Fatal("unexpected zero")
	}
}

// BenchmarkFloat64MulAddInPlace benchmarks Float64x2 MulAdd in-place.
func BenchmarkFloat64MulAddInPlace(b *testing.B) {
	vA := BroadcastFloat64x2(1.5)
	vB := BroadcastFloat64x2(2.0)
	acc := ZeroFloat64x2()

	for b.Loop() {
		vA.MulAddAcc(vB, &acc)
	}
	b.StopTimer()

	if acc.Get(0) == 0 {
		b.Fatal("unexpected zero")
	}
}
