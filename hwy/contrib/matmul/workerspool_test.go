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
	"sync/atomic"
	"testing"
	"time"
)

func TestWorkersPoolBasic(t *testing.T) {
	pool := NewWorkersPoolWithMax(4)

	if !pool.IsEnabled() {
		t.Error("pool should be enabled")
	}
	if pool.MaxParallelism() != 4 {
		t.Errorf("MaxParallelism = %d, want 4", pool.MaxParallelism())
	}
}

func TestWorkersPoolDisabled(t *testing.T) {
	pool := NewWorkersPoolWithMax(0)

	if pool.IsEnabled() {
		t.Error("pool should be disabled")
	}

	// Should run inline
	var ran atomic.Bool
	pool.WaitToStart(func() {
		ran.Store(true)
	})

	if !ran.Load() {
		t.Error("task should have run inline")
	}
}

func TestWorkersPoolSaturate(t *testing.T) {
	pool := NewWorkersPoolWithMax(4)

	// Create work channel with 10 items
	work := make(chan int, 10)
	for i := 0; i < 10; i++ {
		work <- i
	}
	close(work)

	var processed atomic.Int32
	pool.Saturate(func() {
		for range work {
			processed.Add(1)
		}
	})

	if processed.Load() != 10 {
		t.Errorf("processed %d items, want 10", processed.Load())
	}
}

func TestWorkersPoolStartIfAvailable(t *testing.T) {
	pool := NewWorkersPoolWithMax(2)

	var running atomic.Int32
	blocker := make(chan struct{})

	// Start 2 tasks that block
	for i := 0; i < 2; i++ {
		ok := pool.StartIfAvailable(func() {
			running.Add(1)
			<-blocker
			running.Add(-1)
		})
		if !ok {
			t.Errorf("task %d should have started", i)
		}
	}

	// Wait for both to be running
	time.Sleep(10 * time.Millisecond)

	// Third should fail (pool full)
	ok := pool.StartIfAvailable(func() {
		t.Error("this should not run")
	})
	if ok {
		t.Error("third task should not have started")
	}

	// Unblock
	close(blocker)

	// Wait for completion
	time.Sleep(10 * time.Millisecond)

	// Now should work
	var ran atomic.Bool
	ok = pool.StartIfAvailable(func() {
		ran.Store(true)
	})
	if !ok {
		t.Error("task should have started after unblock")
	}

	time.Sleep(10 * time.Millisecond)
	if !ran.Load() {
		t.Error("task should have run")
	}
}

func TestFeedWorkItems_BatchOnly(t *testing.T) {
	params := CacheParams{Mc: 4, Nc: 4}
	workChan := make(chan workItem, 100)

	go feedWorkItems(10, 16, 16, params, 2, workChan)

	var items []workItem
	for item := range workChan {
		items = append(items, item)
	}

	// With batch=10, maxWorkers=2, should split into 2 items of 5 batches each
	if len(items) != 2 {
		t.Errorf("got %d items, want 2", len(items))
	}

	// Check coverage
	batchesCovered := 0
	for _, item := range items {
		batchesCovered += item.batchEnd - item.batchStart
		if item.lhsRowStart != 0 || item.lhsRowEnd != 16 {
			t.Errorf("lhs range should be [0, 16], got [%d, %d]", item.lhsRowStart, item.lhsRowEnd)
		}
		if item.rhsColStart != 0 || item.rhsColEnd != 16 {
			t.Errorf("rhs range should be [0, 16], got [%d, %d]", item.rhsColStart, item.rhsColEnd)
		}
	}
	if batchesCovered != 10 {
		t.Errorf("covered %d batches, want 10", batchesCovered)
	}
}

func TestFeedWorkItems_LHSSplit(t *testing.T) {
	params := CacheParams{Mc: 4, Nc: 4}
	workChan := make(chan workItem, 100)

	// 2 batches, 16 LHS rows, 4 RHS cols, 4 workers
	// LHS > RHS so should split on LHS
	go feedWorkItems(2, 16, 4, params, 4, workChan)

	var items []workItem
	for item := range workChan {
		items = append(items, item)
	}

	// Should have 4 items (2 batches * 2 LHS splits)
	if len(items) != 4 {
		t.Errorf("got %d items, want 4", len(items))
		for i, item := range items {
			t.Logf("item %d: batch[%d,%d] lhs[%d,%d] rhs[%d,%d]",
				i, item.batchStart, item.batchEnd,
				item.lhsRowStart, item.lhsRowEnd,
				item.rhsColStart, item.rhsColEnd)
		}
	}
}

func TestFeedWorkItems_RHSSplit(t *testing.T) {
	params := CacheParams{Mc: 4, Nc: 4}
	workChan := make(chan workItem, 100)

	// 2 batches, 4 LHS rows, 16 RHS cols, 4 workers
	// RHS > LHS so should split on RHS
	go feedWorkItems(2, 4, 16, params, 4, workChan)

	var items []workItem
	for item := range workChan {
		items = append(items, item)
	}

	// Should have 4 items (2 batches * 2 RHS splits)
	if len(items) != 4 {
		t.Errorf("got %d items, want 4", len(items))
		for i, item := range items {
			t.Logf("item %d: batch[%d,%d] lhs[%d,%d] rhs[%d,%d]",
				i, item.batchStart, item.batchEnd,
				item.lhsRowStart, item.lhsRowEnd,
				item.rhsColStart, item.rhsColEnd)
		}
	}
}

func TestFeedWorkItems_ExactMatch(t *testing.T) {
	params := CacheParams{Mc: 4, Nc: 4}
	workChan := make(chan workItem, 100)

	// 4 batches, 4 workers -> one batch per worker
	go feedWorkItems(4, 8, 8, params, 4, workChan)

	var items []workItem
	for item := range workChan {
		items = append(items, item)
	}

	// Should have 4 items, one per batch
	if len(items) != 4 {
		t.Errorf("got %d items, want 4", len(items))
	}

	for i, item := range items {
		if item.batchEnd-item.batchStart != 1 {
			t.Errorf("item %d: batch size = %d, want 1", i, item.batchEnd-item.batchStart)
		}
	}
}
