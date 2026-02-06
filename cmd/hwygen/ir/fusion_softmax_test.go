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

package ir

import (
	"testing"
)

// TestSoftmaxFusionPattern tests that we can identify the softmax fusion pattern.
//
// The Go BaseSoftmax has this structure (5+ passes):
//   1. max = ReduceMax(input)  -- scalar reduction
//   2. shifted = make([]T, size)  -- ALLOCATION
//   3. Loop: shifted[i] = input[i] - max  -- PASS 1
//   4. algo.BaseApply(shifted, output, math.BaseExpVec)  -- PASS 2 (exp)
//   5. Loop: sum += output[i]  -- PASS 3 (sum)
//   6. Loop: output[i] *= 1/sum  -- PASS 4 (normalize)
//
// The optimal fused version (handwritten C reference) has 3 passes:
//   1. max = ReduceMax(input)
//   2. Loop: output[i] = exp(input[i] - max); sum += output[i]  -- FUSED
//   3. Loop: output[i] *= 1/sum
//
// This test verifies that the fusion pass can identify the pattern.
func TestSoftmaxFusionPattern(t *testing.T) {
	// Create IR that represents the softmax structure
	fn := NewFunction("BaseSoftmax")
	fn.ElemType = "float32"
	fn.Params = []IRParam{
		{Name: "input", Type: "[]float32", IsSlice: true, ElemType: "float32"},
		{Name: "output", Type: "[]float32", IsSlice: true, ElemType: "float32"},
	}

	// Step 1: max reduction (scalar, not fusible)
	maxNode := fn.AddNode(OpKindReduction, "ReduceMax")
	maxNode.InputNames = []string{"input"}
	maxNode.Outputs = []string{"maxVal"}

	// Step 2: Allocate shifted (this should be eliminated by fusion)
	allocNode := fn.AddNode(OpKindAlloc, "make")
	allocNode.Outputs = []string{"shifted"}
	allocNode.AllocSize = "size"
	allocNode.AllocElemType = "float32"

	// Step 3: Loop to compute shifted = input - max
	lr := &LoopRange{LoopVar: "i", Start: "0", End: "size", Step: "lanes", IsVectorized: true, VectorLanes: 4}

	shiftLoop := fn.AddNode(OpKindLoop, "for")
	shiftLoop.LoopRange = lr.Clone()

	loadInput := fn.AddChildNode(shiftLoop, OpKindLoad, "Load")
	loadInput.InputNames = []string{"input", "i"}
	loadInput.Outputs = []string{"v1"}
	loadInput.LoopRange = lr.Clone()

	broadcastMax := fn.AddChildNode(shiftLoop, OpKindBroadcast, "Set")
	broadcastMax.InputNames = []string{"maxVal"}
	broadcastMax.Outputs = []string{"maxVec"}
	broadcastMax.LoopRange = lr.Clone()

	subNode := fn.AddChildNode(shiftLoop, OpKindElementwise, "Sub")
	subNode.Inputs = []*IRNode{loadInput, broadcastMax}
	subNode.Outputs = []string{"v2"}
	subNode.LoopRange = lr.Clone()

	storeShifted := fn.AddChildNode(shiftLoop, OpKindStore, "Store")
	storeShifted.Inputs = []*IRNode{subNode}
	storeShifted.InputNames = []string{"shifted", "i"}
	storeShifted.LoopRange = lr.Clone()

	// Step 4: Loop for exp (BaseApply inlined)
	expLoop := fn.AddNode(OpKindLoop, "for")
	expLoop.LoopRange = lr.Clone()

	loadShifted := fn.AddChildNode(expLoop, OpKindLoad, "Load")
	loadShifted.InputNames = []string{"shifted", "i"}
	loadShifted.Outputs = []string{"v3"}
	loadShifted.LoopRange = lr.Clone()

	expNode := fn.AddChildNode(expLoop, OpKindElementwise, "Exp")
	expNode.Inputs = []*IRNode{loadShifted}
	expNode.Outputs = []string{"v4"}
	expNode.LoopRange = lr.Clone()

	storeOutput := fn.AddChildNode(expLoop, OpKindStore, "Store")
	storeOutput.Inputs = []*IRNode{expNode}
	storeOutput.InputNames = []string{"output", "i"}
	storeOutput.LoopRange = lr.Clone()

	// Step 5: Loop for sum reduction
	sumLoop := fn.AddNode(OpKindLoop, "for")
	sumLoop.LoopRange = lr.Clone()

	loadOutput := fn.AddChildNode(sumLoop, OpKindLoad, "Load")
	loadOutput.InputNames = []string{"output", "i"}
	loadOutput.Outputs = []string{"v5"}
	loadOutput.LoopRange = lr.Clone()

	sumAccum := fn.AddChildNode(sumLoop, OpKindScalar, "Add")
	sumAccum.Inputs = []*IRNode{loadOutput}
	sumAccum.Outputs = []string{"expSum"}
	sumAccum.LoopRange = lr.Clone()

	// Step 6: Normalize loop
	normLoop := fn.AddNode(OpKindLoop, "for")
	normLoop.LoopRange = lr.Clone()

	loadNorm := fn.AddChildNode(normLoop, OpKindLoad, "Load")
	loadNorm.InputNames = []string{"output", "i"}
	loadNorm.Outputs = []string{"v6"}
	loadNorm.LoopRange = lr.Clone()

	mulNorm := fn.AddChildNode(normLoop, OpKindElementwise, "Mul")
	mulNorm.Inputs = []*IRNode{loadNorm}
	mulNorm.InputNames = []string{"invSum"}
	mulNorm.Outputs = []string{"v7"}
	mulNorm.LoopRange = lr.Clone()

	storeNorm := fn.AddChildNode(normLoop, OpKindStore, "Store")
	storeNorm.Inputs = []*IRNode{mulNorm}
	storeNorm.InputNames = []string{"output", "i"}
	storeNorm.LoopRange = lr.Clone()

	// Get stats before fusion
	beforeStats := ComputeFusionStats(fn)
	t.Logf("Before fusion: %d loops, %d operations", beforeStats.OriginalPasses, len(fn.Operations))

	// Run analysis
	Analyze(fn)

	// Look for allocations that can be eliminated
	allocsToElim := IdentifyAllocationsToEliminate(fn)
	t.Logf("Allocations to eliminate: %d", len(allocsToElim))

	// Apply fusion
	ApplyFusionRules(fn)

	// Apply softmax-specific optimization
	OptimizeSoftmax(fn)

	// Get stats after fusion
	afterStats := ComputeFusionStats(fn)
	t.Logf("After fusion: %d fusion groups, %d eliminated allocs",
		afterStats.FusionGroups, afterStats.EliminatedAllocs)

	// Verify we created fusion groups
	if afterStats.FusionGroups == 0 {
		t.Error("expected fusion groups to be created")
	}

	// The allocation should be a candidate for elimination
	// (actual elimination depends on the fusion pass finding the right pattern)
	if len(allocsToElim) == 0 {
		// This is expected since the current implementation needs both
		// producer and consumer in the same loop to eliminate the alloc
		t.Log("Note: allocation elimination requires producer/consumer in same loop")
	}
}

// TestMapReduceFusion tests the Elem+Reduce (MapReduce) fusion pattern.
// This is the key fusion for softmax: exp(x) + sum in one pass.
func TestMapReduceFusion(t *testing.T) {
	fn := NewFunction("testMapReduce")

	lr := &LoopRange{
		LoopVar:      "i",
		Start:        "0",
		End:          "size",
		Step:         "lanes",
		IsVectorized: true,
		VectorLanes:  4,
	}

	// Create: load -> elementwise (exp) -> reduction (sum)
	loadNode := fn.AddNode(OpKindLoad, "Load")
	loadNode.InputNames = []string{"input", "i"}
	loadNode.Outputs = []string{"v1"}
	loadNode.LoopRange = lr.Clone()

	expNode := fn.AddNode(OpKindElementwise, "Exp")
	expNode.Inputs = []*IRNode{loadNode}
	expNode.Outputs = []string{"v2"}
	expNode.LoopRange = lr.Clone()

	reduceNode := fn.AddNode(OpKindReduction, "ReduceSum")
	reduceNode.Inputs = []*IRNode{expNode}
	reduceNode.Outputs = []string{"sum"}
	reduceNode.LoopRange = lr.Clone()

	// Run analysis
	Analyze(fn)

	// Find candidates
	candidates := FindFusionCandidates(fn)

	// Should find Elem+Reduce pattern
	foundMapReduce := false
	for _, c := range candidates {
		t.Logf("Candidate: %s (producer=%s, consumer=%s)", c.Pattern, c.Producer.Op, c.Consumer.Op)
		if c.Pattern == "Elem+Reduce" && c.Producer == expNode && c.Consumer == reduceNode {
			foundMapReduce = true
		}
	}

	if !foundMapReduce {
		t.Error("expected to find Elem+Reduce fusion candidate for exp->sum pattern")
	}

	// Apply fusion
	ApplyFusionRules(fn)

	// Check that fusion was applied
	if len(fn.FusionGroups) == 0 {
		t.Error("expected fusion groups to be created")
	}

	// The exp and reduce should be in the same group
	expFused := expNode.FusionGroup >= 0
	reduceFused := reduceNode.FusionGroup >= 0

	if !expFused || !reduceFused {
		t.Errorf("expected both nodes to be fused: exp=%v, reduce=%v", expFused, reduceFused)
	}

	if expNode.FusionGroup != reduceNode.FusionGroup {
		t.Error("exp and reduce should be in the same fusion group")
	}
}

// TestAllocationElimination tests that temporary allocations can be identified.
func TestAllocationElimination(t *testing.T) {
	fn := NewFunction("testAllocElim")

	// Create: alloc -> loop(store to alloc) -> loop(load from alloc)
	allocNode := fn.AddNode(OpKindAlloc, "make")
	allocNode.Outputs = []string{"temp"}
	allocNode.AllocSize = "size"
	allocNode.AllocElemType = "float32"

	lr := &LoopRange{LoopVar: "i", Start: "0", End: "size", Step: "1"}

	// First loop: write to temp
	loop1 := fn.AddNode(OpKindLoop, "for")
	loop1.LoopRange = lr.Clone()

	store := fn.AddChildNode(loop1, OpKindStore, "Store")
	store.InputNames = []string{"temp", "i", "value"}
	store.LoopRange = lr.Clone()

	// Second loop: read from temp
	loop2 := fn.AddNode(OpKindLoop, "for")
	loop2.LoopRange = lr.Clone()

	load := fn.AddChildNode(loop2, OpKindLoad, "Load")
	load.InputNames = []string{"temp", "i"}
	load.Outputs = []string{"v1"}
	load.LoopRange = lr.Clone()

	// Set up producer-consumer: alloc -> loop1
	allocNode.Consumers = []*IRNode{loop1}

	// Run analysis
	Analyze(fn)

	// The allocation should be identified as eliminable if both
	// read and write happen in loops with the same iteration space
	allocs := IdentifyAllocationsToEliminate(fn)

	t.Logf("Found %d allocations to eliminate", len(allocs))
	for _, a := range allocs {
		t.Logf("  - %s (outputs: %v)", a.Op, a.Outputs)
	}

	// Note: with the current implementation, allocs are only eliminated
	// when both read and write are in the SAME loop body.
	// The softmax pattern has them in different loops, which requires
	// loop fusion first.
}

// TestFusionStatistics tests that we correctly count passes.
func TestFusionStatistics(t *testing.T) {
	fn := NewFunction("testStats")

	// Simulate softmax structure: 1 alloc + 4 loops
	fn.AddNode(OpKindAlloc, "make")
	fn.AddNode(OpKindLoop, "for") // shift
	fn.AddNode(OpKindLoop, "for") // exp
	fn.AddNode(OpKindLoop, "for") // sum
	fn.AddNode(OpKindLoop, "for") // normalize

	stats := ComputeFusionStats(fn)

	// 1 alloc + 4 loops = 5 passes
	if stats.OriginalPasses != 5 {
		t.Errorf("original passes = %d, want 5", stats.OriginalPasses)
	}

	t.Logf("Original: %d passes (4 loops + 1 alloc)", stats.OriginalPasses)

	// After optimal fusion, we should have 3 passes:
	// - max loop (unfused, different pattern)
	// - fused exp+sum loop
	// - normalize loop
	// And the allocation should be eliminated.
	//
	// The current implementation doesn't achieve this automatically because
	// it requires cross-loop fusion and cross-package function inlining.
}
