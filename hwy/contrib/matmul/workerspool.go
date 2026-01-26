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

package matmul

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// WorkersPool manages a pool of workers for parallel execution.
// It provides controlled parallelism with proper coordination.
//
// This is inspired by gomlx's workerspool implementation for packgemm.
type WorkersPool struct {
	// maxParallelism is the soft target for parallel workers.
	// 0 = disabled, -1 = unlimited, >0 = limited
	maxParallelism int

	mu         sync.Mutex
	cond       sync.Cond
	numRunning int

	// extraParallelism temporarily increases when a worker sleeps
	extraParallelism atomic.Int32
}

// NewWorkersPool creates a new pool with default parallelism (2 * GOMAXPROCS).
func NewWorkersPool() *WorkersPool {
	p := &WorkersPool{
		maxParallelism: 2 * runtime.GOMAXPROCS(0),
	}
	p.cond = sync.Cond{L: &p.mu}
	return p
}

// NewWorkersPoolWithMax creates a pool with specified max parallelism.
func NewWorkersPoolWithMax(maxParallelism int) *WorkersPool {
	p := &WorkersPool{
		maxParallelism: maxParallelism,
	}
	p.cond = sync.Cond{L: &p.mu}
	return p
}

// IsEnabled returns whether parallelism is enabled.
func (p *WorkersPool) IsEnabled() bool {
	return p.maxParallelism != 0
}

// MaxParallelism returns the configured max parallelism.
func (p *WorkersPool) MaxParallelism() int {
	return p.maxParallelism
}

// AdjustedMaxParallelism returns the effective parallelism (>= 1).
// For unlimited (-1), returns GOMAXPROCS.
// For disabled (0), returns 1.
// Otherwise, returns min(maxParallelism, GOMAXPROCS).
func (p *WorkersPool) AdjustedMaxParallelism() int {
	if p.maxParallelism < 0 {
		return runtime.GOMAXPROCS(0)
	}
	return min(max(p.maxParallelism, 1), runtime.GOMAXPROCS(0))
}

// SetMaxParallelism updates the max parallelism.
// Should only be called before any workers start.
func (p *WorkersPool) SetMaxParallelism(maxParallelism int) {
	p.maxParallelism = maxParallelism
}

// lockedIsFull returns whether all workers are busy (must hold lock).
func (p *WorkersPool) lockedIsFull() bool {
	if p.maxParallelism == 0 {
		return true // disabled
	}
	if p.maxParallelism < 0 {
		return false // unlimited
	}
	return p.numRunning >= p.maxParallelism+int(p.extraParallelism.Load())
}

// lockedRunTask starts a task in a goroutine (must hold lock).
func (p *WorkersPool) lockedRunTask(task func()) {
	p.numRunning++
	go func() {
		task()
		p.mu.Lock()
		p.numRunning--
		p.cond.Signal()
		p.mu.Unlock()
	}()
}

// StartIfAvailable runs the task if workers are available.
// Returns true if task was started, false if pool is full.
func (p *WorkersPool) StartIfAvailable(task func()) bool {
	if p.maxParallelism < 0 {
		// Unlimited: always start
		go task()
		return true
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.lockedIsFull() {
		return false
	}

	p.lockedRunTask(task)
	return true
}

// WaitToStart blocks until a worker is available, then runs the task.
// If parallelism is disabled, runs inline.
func (p *WorkersPool) WaitToStart(task func()) {
	if p.maxParallelism < 0 {
		go task()
		return
	}

	if p.maxParallelism == 0 {
		// Disabled: run inline
		task()
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	for p.lockedIsFull() {
		p.cond.Wait()
	}
	p.lockedRunTask(task)
}

// Saturate fans out as many workers as available, each running the given task.
// Workers consume from a shared work source (typically a channel).
// When the first task completes (signaling no more work), it stops spawning.
// Returns when all started tasks have finished.
//
// Usage pattern:
//
//	workChan := make(chan workItem, numItems)
//	// ... fill workChan ...
//	close(workChan)
//	pool.Saturate(func() {
//	    for item := range workChan {
//	        process(item)
//	    }
//	})
func (p *WorkersPool) Saturate(task func()) {
	if p.maxParallelism == 0 {
		// Disabled: run single task
		task()
		return
	}

	limit := p.maxParallelism
	if limit < 0 {
		limit = runtime.GOMAXPROCS(0)
	}

	var wg sync.WaitGroup
	var doneFanningOut atomic.Bool

	p.mu.Lock()
	started := 0

	for !doneFanningOut.Load() {
		// Check limits
		unlimited := p.maxParallelism < 0
		if (unlimited && started >= limit) || (!unlimited && p.lockedIsFull()) {
			p.cond.Wait()
			if doneFanningOut.Load() {
				p.cond.Signal() // propagate to other waiters
				break
			}
			continue
		}

		started++
		wg.Add(1)
		p.lockedRunTask(func() {
			defer wg.Done()
			task()
			doneFanningOut.Store(true)
		})
	}
	p.mu.Unlock()
	wg.Wait()
}

// WorkerIsAsleep indicates a worker is waiting and temporarily
// increases available parallelism. Call WorkerRestarted when done.
func (p *WorkersPool) WorkerIsAsleep() {
	p.extraParallelism.Add(1)
}

// WorkerRestarted indicates a sleeping worker is active again.
func (p *WorkersPool) WorkerRestarted() {
	p.extraParallelism.Add(-1)
}

// workItem represents a chunk of work for parallel GEMM.
type workItem struct {
	batchStart, batchEnd     int
	lhsRowStart, lhsRowEnd   int
	rhsColStart, rhsColEnd   int
}

// feedWorkItems distributes GEMM work into workItems optimized for maxWorkers.
// It prioritizes batch splitting, then splits on LHS or RHS dimension.
// Closes workChan on exit.
//
// This implements the intelligent work splitting from gomlx's packgemm-simd-large-opt.
func feedWorkItems(
	batchSize, lhsCrossSize, rhsCrossSize int,
	params CacheParams,
	maxWorkers int,
	workChan chan<- workItem,
) {
	defer close(workChan)

	if maxWorkers <= 0 {
		maxWorkers = 1
	}

	// If batch size is large enough, split only on batch dimension
	if batchSize >= 2*maxWorkers {
		batchStep := batchSize / maxWorkers
		for batchIdx := 0; batchIdx < batchSize; batchIdx += batchStep {
			workChan <- workItem{
				batchStart:  batchIdx,
				batchEnd:    batchIdx + min(batchStep, batchSize-batchIdx),
				lhsRowStart: 0,
				lhsRowEnd:   lhsCrossSize,
				rhsColStart: 0,
				rhsColEnd:   rhsCrossSize,
			}
		}
		return
	}

	// First handle batches one at a time up to maxWorkers
	batchIdx := 0
	if batchSize >= maxWorkers {
		for ; batchIdx < maxWorkers; batchIdx++ {
			workChan <- workItem{
				batchStart:  batchIdx,
				batchEnd:    batchIdx + 1,
				lhsRowStart: 0,
				lhsRowEnd:   lhsCrossSize,
				rhsColStart: 0,
				rhsColEnd:   rhsCrossSize,
			}
		}
	}

	// Split remaining work on LHS or RHS dimension
	batchCountRemaining := batchSize - batchIdx
	if batchCountRemaining == 0 {
		return
	}

	splitFactor := (maxWorkers + batchCountRemaining - 1) / batchCountRemaining

	if lhsCrossSize > rhsCrossSize {
		// Split on LHS dimension (aligned to Mc)
		lhsSplitSize := (lhsCrossSize + splitFactor - 1) / splitFactor
		lhsSplitSize = max(1, lhsSplitSize/params.Mc) * params.Mc

		batchStart := batchIdx
		for lhsRowIdx := 0; lhsRowIdx < lhsCrossSize; lhsRowIdx += lhsSplitSize {
			for bi := batchStart; bi < batchSize; bi++ {
				workChan <- workItem{
					batchStart:  bi,
					batchEnd:    bi + 1,
					lhsRowStart: lhsRowIdx,
					lhsRowEnd:   lhsRowIdx + min(lhsSplitSize, lhsCrossSize-lhsRowIdx),
					rhsColStart: 0,
					rhsColEnd:   rhsCrossSize,
				}
			}
		}
	} else {
		// Split on RHS dimension (aligned to Nc)
		rhsSplitSize := (rhsCrossSize + splitFactor - 1) / splitFactor
		rhsSplitSize = max(1, rhsSplitSize/params.Nc) * params.Nc

		batchStart := batchIdx
		for rhsColIdx := 0; rhsColIdx < rhsCrossSize; rhsColIdx += rhsSplitSize {
			for bi := batchStart; bi < batchSize; bi++ {
				workChan <- workItem{
					batchStart:  bi,
					batchEnd:    bi + 1,
					lhsRowStart: 0,
					lhsRowEnd:   lhsCrossSize,
					rhsColStart: rhsColIdx,
					rhsColEnd:   rhsColIdx + min(rhsSplitSize, rhsCrossSize-rhsColIdx),
				}
			}
		}
	}
}
