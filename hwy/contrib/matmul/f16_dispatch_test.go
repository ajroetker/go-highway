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

//go:build arm64

package matmul

import (
	"runtime"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// TestF16DispatchPath tests the exact dispatch path used by BenchmarkMatMulFloat16
// to help diagnose why benchmarks crash but asm tests pass.
func TestF16DispatchPath(t *testing.T) {
	if !hwy.HasARMFP16() {
		t.Skip("CPU does not support ARM FP16")
	}

	pool := workerpool.New(0)
	defer pool.Close()

	n := 64

	a := make([]hwy.Float16, n*n)
	b := make([]hwy.Float16, n*n)
	c := make([]hwy.Float16, n*n)

	for i := range a {
		a[i] = hwy.Float32ToFloat16(float32(i%7) + 0.5)
		b[i] = hwy.Float32ToFloat16(float32(i%11) + 0.25)
	}

	// This is exactly what the benchmark calls
	t.Log("Calling MatMulAuto (same as benchmark)...")
	MatMulAuto(pool, a, b, c, n, n, n)
	t.Log("MatMulAuto completed successfully")
}

// TestF16BlockedMatMulDispatch tests the BlockedMatMul dispatch path
func TestF16BlockedMatMulDispatch(t *testing.T) {
	if !hwy.HasARMFP16() {
		t.Skip("CPU does not support ARM FP16")
	}

	n := 64

	a := make([]hwy.Float16, n*n)
	b := make([]hwy.Float16, n*n)
	c := make([]hwy.Float16, n*n)

	for i := range a {
		a[i] = hwy.Float32ToFloat16(float32(i%7) + 0.5)
		b[i] = hwy.Float32ToFloat16(float32(i%11) + 0.25)
	}

	// Call through the BlockedMatMul dispatch (function pointer)
	t.Log("Calling BlockedMatMul via dispatch...")
	BlockedMatMul(a, b, c, n, n, n)
	t.Log("BlockedMatMul completed successfully")
}

// TestF16ParallelMatMul tests the ParallelMatMul path directly
func TestF16ParallelMatMul(t *testing.T) {
	if !hwy.HasARMFP16() {
		t.Skip("CPU does not support ARM FP16")
	}

	pool := workerpool.New(0)
	defer pool.Close()

	n := 64

	a := make([]hwy.Float16, n*n)
	b := make([]hwy.Float16, n*n)
	c := make([]hwy.Float16, n*n)

	for i := range a {
		a[i] = hwy.Float32ToFloat16(float32(i%7) + 0.5)
		b[i] = hwy.Float32ToFloat16(float32(i%11) + 0.25)
	}

	// This is what MatMulAuto calls for 64x64
	t.Log("Calling ParallelMatMul (64x64 goes here because 64^3 >= MinParallelOps)...")
	ParallelMatMul(pool, a, b, c, n, n, n)
	t.Log("ParallelMatMul completed successfully")
}

// TestF16MatMulMultipleIterations tests calling multiple times like a benchmark
func TestF16MatMulMultipleIterations(t *testing.T) {
	if !hwy.HasARMFP16() {
		t.Skip("CPU does not support ARM FP16")
	}

	pool := workerpool.New(0)
	defer pool.Close()

	n := 64

	a := make([]hwy.Float16, n*n)
	b := make([]hwy.Float16, n*n)
	c := make([]hwy.Float16, n*n)

	for i := range a {
		a[i] = hwy.Float32ToFloat16(float32(i%7) + 0.5)
		b[i] = hwy.Float32ToFloat16(float32(i%11) + 0.25)
	}

	// Run MANY iterations - benchmarks might run hundreds of times
	// The crash might only occur after many iterations
	iterations := 1000
	t.Logf("Running %d iterations of MatMulAuto...", iterations)
	for i := 0; i < iterations; i++ {
		MatMulAuto(pool, a, b, c, n, n, n)
		if i > 0 && i%100 == 0 {
			t.Logf("Completed %d iterations", i)
		}
	}
	t.Logf("All %d iterations completed successfully", iterations)
}

// TestF16MatMulManyIterationsInGoroutine tests many iterations in a goroutine
func TestF16MatMulManyIterationsInGoroutine(t *testing.T) {
	if !hwy.HasARMFP16() {
		t.Skip("CPU does not support ARM FP16")
	}

	pool := workerpool.New(0)
	defer pool.Close()

	n := 64

	a := make([]hwy.Float16, n*n)
	b := make([]hwy.Float16, n*n)
	c := make([]hwy.Float16, n*n)

	for i := range a {
		a[i] = hwy.Float32ToFloat16(float32(i%7) + 0.5)
		b[i] = hwy.Float32ToFloat16(float32(i%11) + 0.25)
	}

	iterations := 1000
	t.Logf("Running %d iterations in a goroutine...", iterations)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < iterations; i++ {
			MatMulAuto(pool, a, b, c, n, n, n)
		}
	}()
	<-done
	t.Logf("All %d goroutine iterations completed successfully", iterations)
}

// TestF16WithLockOSThread tests with the goroutine locked to OS thread
// This prevents goroutine migration which might affect SIMD state
func TestF16WithLockOSThread(t *testing.T) {
	if !hwy.HasARMFP16() {
		t.Skip("CPU does not support ARM FP16")
	}

	pool := workerpool.New(0)
	defer pool.Close()

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	n := 64

	a := make([]hwy.Float16, n*n)
	b := make([]hwy.Float16, n*n)
	c := make([]hwy.Float16, n*n)

	for i := range a {
		a[i] = hwy.Float32ToFloat16(float32(i%7) + 0.5)
		b[i] = hwy.Float32ToFloat16(float32(i%11) + 0.25)
	}

	iterations := 1000
	t.Logf("Running %d iterations with LockOSThread...", iterations)
	for i := 0; i < iterations; i++ {
		MatMulAuto(pool, a, b, c, n, n, n)
	}
	t.Logf("All %d iterations completed successfully", iterations)
}

// TestF16StreamingMatMul tests the streaming (non-blocked) matmul path
func TestF16StreamingMatMul(t *testing.T) {
	if !hwy.HasARMFP16() {
		t.Skip("CPU does not support ARM FP16")
	}

	n := 64

	a := make([]hwy.Float16, n*n)
	b := make([]hwy.Float16, n*n)
	c := make([]hwy.Float16, n*n)

	for i := range a {
		a[i] = hwy.Float32ToFloat16(float32(i%7) + 0.5)
		b[i] = hwy.Float32ToFloat16(float32(i%11) + 0.25)
	}

	// Call through MatMul dispatch (streaming, not blocked)
	t.Log("Calling MatMul (streaming) via dispatch...")
	MatMul(a, b, c, n, n, n)
	t.Log("MatMul completed successfully")
}
