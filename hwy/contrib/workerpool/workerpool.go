// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

// Package workerpool provides a persistent, reusable worker pool for parallel
// computation. Unlike per-call goroutine spawning, a Pool is created once and
// reused across many operations, eliminating allocation and spawn overhead.
//
// This is critical for performance in transformer inference where ~50+ matrix
// multiplications occur per forward pass. Per-call channel allocation and
// goroutine spawning dominates compute time for smaller matrices.
//
// Usage:
//
//	pool := workerpool.New(runtime.GOMAXPROCS(0))
//	defer pool.Close()
//
//	// Reuse pool across many operations
//	for _, layer := range layers {
//	    pool.ParallelFor(m, func(start, end int) {
//	        processRows(start, end)
//	    })
//	}
package workerpool

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// Pool is a persistent worker pool that can be reused across many parallel
// operations. Workers are spawned once at creation and reused.
type Pool struct {
	numWorkers int
	workC      chan workItem
	closeOnce  sync.Once
	closed     atomic.Bool
}

// workItem represents a single parallel operation to execute.
type workItem struct {
	fn      func()
	barrier *sync.WaitGroup
}

// New creates a new worker pool with the specified number of workers.
// Workers are spawned immediately and persist until Close is called.
// If numWorkers <= 0, uses GOMAXPROCS.
func New(numWorkers int) *Pool {
	if numWorkers <= 0 {
		numWorkers = runtime.GOMAXPROCS(0)
	}

	p := &Pool{
		numWorkers: numWorkers,
		// Buffer enough for all workers to have pending work
		workC: make(chan workItem, numWorkers*2),
	}

	// Spawn persistent workers
	for range numWorkers {
		go p.worker()
	}

	return p
}

// worker is the main loop for each persistent worker goroutine.
func (p *Pool) worker() {
	for item := range p.workC {
		item.fn()
		item.barrier.Done()
	}
}

// NumWorkers returns the number of workers in the pool.
func (p *Pool) NumWorkers() int {
	return p.numWorkers
}

// Close shuts down the worker pool. All pending work will complete.
// Calling Close multiple times is safe.
func (p *Pool) Close() {
	p.closeOnce.Do(func() {
		p.closed.Store(true)
		close(p.workC)
	})
}

// ParallelFor executes fn for each index in [0, n) using the worker pool.
// Each worker processes a contiguous range of indices.
// Blocks until all work completes.
//
// fn receives (start, end) indices where work should process [start, end).
func (p *Pool) ParallelFor(n int, fn func(start, end int)) {
	if n <= 0 {
		return
	}

	if p.closed.Load() {
		// Fallback to sequential if pool is closed
		fn(0, n)
		return
	}

	// Determine number of workers to use (don't use more workers than items)
	workers := min(p.numWorkers, n)

	// For very small n, just run sequentially
	if workers == 1 {
		fn(0, n)
		return
	}

	// Calculate chunk size (ensure all items are covered)
	chunkSize := (n + workers - 1) / workers

	var wg sync.WaitGroup
	wg.Add(workers)

	for i := range workers {
		start := i * chunkSize
		end := min(start+chunkSize, n)
		if start >= n {
			// No work for this worker
			wg.Done()
			continue
		}

		p.workC <- workItem{
			fn: func() {
				fn(start, end)
			},
			barrier: &wg,
		}
	}

	wg.Wait()
}

// ParallelForAtomic executes fn for each index in [0, n) using atomic work
// stealing. This provides better load balancing when work per item varies.
// Blocks until all work completes.
//
// fn receives the index to process.
func (p *Pool) ParallelForAtomic(n int, fn func(i int)) {
	if n <= 0 {
		return
	}

	if p.closed.Load() {
		// Fallback to sequential if pool is closed
		for i := range n {
			fn(i)
		}
		return
	}

	workers := min(p.numWorkers, n)

	if workers == 1 {
		for i := range n {
			fn(i)
		}
		return
	}

	var nextIdx atomic.Int32
	var wg sync.WaitGroup
	wg.Add(workers)

	for range workers {
		p.workC <- workItem{
			fn: func() {
				for {
					idx := int(nextIdx.Add(1)) - 1
					if idx >= n {
						return
					}
					fn(idx)
				}
			},
			barrier: &wg,
		}
	}

	wg.Wait()
}

// ParallelForAtomicBatched executes fn for batches of indices using atomic
// work stealing. Combines the load balancing of atomic distribution with
// reduced atomic operation overhead by processing multiple items per grab.
//
// fn receives (start, end) indices where work should process [start, end).
// batchSize controls how many items are grabbed per atomic operation.
func (p *Pool) ParallelForAtomicBatched(n int, batchSize int, fn func(start, end int)) {
	if n <= 0 {
		return
	}

	if batchSize <= 0 {
		batchSize = 1
	}

	if p.closed.Load() {
		fn(0, n)
		return
	}

	// Calculate number of batches
	numBatches := (n + batchSize - 1) / batchSize
	workers := min(p.numWorkers, numBatches)

	if workers == 1 {
		fn(0, n)
		return
	}

	var nextBatch atomic.Int32
	var wg sync.WaitGroup
	wg.Add(workers)

	for range workers {
		p.workC <- workItem{
			fn: func() {
				for {
					batch := int(nextBatch.Add(1)) - 1
					start := batch * batchSize
					if start >= n {
						return
					}
					end := min(start+batchSize, n)
					fn(start, end)
				}
			},
			barrier: &wg,
		}
	}

	wg.Wait()
}
