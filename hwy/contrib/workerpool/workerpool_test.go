// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

package workerpool

import (
	"runtime"
	"sync/atomic"
	"testing"
)

func TestNew(t *testing.T) {
	pool := New(4)
	defer pool.Close()

	if pool.NumWorkers() != 4 {
		t.Errorf("NumWorkers() = %d, want 4", pool.NumWorkers())
	}
}

func TestNewDefault(t *testing.T) {
	pool := New(0)
	defer pool.Close()

	if pool.NumWorkers() != runtime.GOMAXPROCS(0) {
		t.Errorf("NumWorkers() = %d, want %d", pool.NumWorkers(), runtime.GOMAXPROCS(0))
	}
}

func TestParallelFor(t *testing.T) {
	pool := New(4)
	defer pool.Close()

	n := 100
	results := make([]int, n)

	pool.ParallelFor(n, func(start, end int) {
		for i := start; i < end; i++ {
			results[i] = i * 2
		}
	})

	for i := 0; i < n; i++ {
		if results[i] != i*2 {
			t.Errorf("results[%d] = %d, want %d", i, results[i], i*2)
		}
	}
}

func TestParallelForAtomic(t *testing.T) {
	pool := New(4)
	defer pool.Close()

	n := 100
	results := make([]int, n)

	pool.ParallelForAtomic(n, func(i int) {
		results[i] = i * 2
	})

	for i := 0; i < n; i++ {
		if results[i] != i*2 {
			t.Errorf("results[%d] = %d, want %d", i, results[i], i*2)
		}
	}
}

func TestParallelForAtomicBatched(t *testing.T) {
	pool := New(4)
	defer pool.Close()

	n := 100
	results := make([]int, n)

	pool.ParallelForAtomicBatched(n, 10, func(start, end int) {
		for i := start; i < end; i++ {
			results[i] = i * 2
		}
	})

	for i := 0; i < n; i++ {
		if results[i] != i*2 {
			t.Errorf("results[%d] = %d, want %d", i, results[i], i*2)
		}
	}
}

func TestParallelForSmallN(t *testing.T) {
	pool := New(8)
	defer pool.Close()

	// Test with n smaller than workers
	n := 3
	var count atomic.Int32

	pool.ParallelFor(n, func(start, end int) {
		count.Add(int32(end - start))
	})

	if count.Load() != int32(n) {
		t.Errorf("count = %d, want %d", count.Load(), n)
	}
}

func TestParallelForZeroN(t *testing.T) {
	pool := New(4)
	defer pool.Close()

	var called bool
	pool.ParallelFor(0, func(start, end int) {
		called = true
	})

	if called {
		t.Error("ParallelFor with n=0 should not call fn")
	}
}

func TestCloseMultipleTimes(t *testing.T) {
	pool := New(4)
	pool.Close()
	pool.Close() // Should not panic
}

func TestClosedPoolFallback(t *testing.T) {
	pool := New(4)
	pool.Close()

	n := 100
	results := make([]int, n)

	// Should still work (sequential fallback)
	pool.ParallelFor(n, func(start, end int) {
		for i := start; i < end; i++ {
			results[i] = i * 2
		}
	})

	for i := 0; i < n; i++ {
		if results[i] != i*2 {
			t.Errorf("results[%d] = %d, want %d", i, results[i], i*2)
		}
	}
}

func BenchmarkParallelFor(b *testing.B) {
	pool := New(0) // Use GOMAXPROCS
	defer pool.Close()

	n := 1000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool.ParallelFor(n, func(start, end int) {
			// Simulate work
			for j := start; j < end; j++ {
				_ = j * j
			}
		})
	}
}

func BenchmarkParallelForAtomic(b *testing.B) {
	pool := New(0)
	defer pool.Close()

	n := 1000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool.ParallelForAtomic(n, func(i int) {
			_ = i * i
		})
	}
}

func BenchmarkParallelForAtomicBatched(b *testing.B) {
	pool := New(0)
	defer pool.Close()

	n := 1000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool.ParallelForAtomicBatched(n, 10, func(start, end int) {
			for j := start; j < end; j++ {
				_ = j * j
			}
		})
	}
}

// BenchmarkPoolOverhead measures the overhead of using the pool vs inline spawn
func BenchmarkPoolOverhead(b *testing.B) {
	pool := New(0)
	defer pool.Close()

	b.Run("Pool", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			pool.ParallelFor(10, func(start, end int) {
				// Minimal work
			})
		}
	})
}
