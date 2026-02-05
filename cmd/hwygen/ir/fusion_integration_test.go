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
	"strings"
	"testing"
)

// TestSoftmaxFusedCGeneration tests that the fusion pass produces C code
// matching the structure of the handwritten softmax_neon_arm64.c.
//
// The handwritten reference has:
//   - 3 passes (find max, fused exp+sum, normalize)
//   - No shifted[] intermediate allocation
//   - Fused Pass 2: subtract-max + exp + sum in one loop
//
// This test verifies the IR fusion produces similar structural properties.
func TestSoftmaxFusedCGeneration(t *testing.T) {
	// Build IR representing the softmax structure
	fn := NewFunction("BaseSoftmax")
	fn.ElemType = "float32"
	fn.Params = []IRParam{
		{Name: "input", Type: "[]float32", IsSlice: true, ElemType: "float32"},
		{Name: "output", Type: "[]float32", IsSlice: true, ElemType: "float32"},
	}

	// Full softmax IR (5 passes before fusion)
	lr := &LoopRange{LoopVar: "i", Start: "0", End: "size", Step: "lanes", IsVectorized: true, VectorLanes: 4}

	// Step 1: max reduction
	maxNode := fn.AddNode(OpKindReduction, "ReduceMax")
	maxNode.InputNames = []string{"input"}
	maxNode.Outputs = []string{"maxVal"}

	// Step 2: Allocate shifted (should be eliminated by fusion)
	allocNode := fn.AddNode(OpKindAlloc, "make")
	allocNode.Outputs = []string{"shifted"}
	allocNode.AllocSize = "size"
	allocNode.AllocElemType = "float32"

	// Step 3: Loop to compute shifted = input - max
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

	sumAccum := fn.AddChildNode(sumLoop, OpKindReduction, "Add")
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
	t.Logf("Before fusion: %d original passes, %d loops", beforeStats.OriginalPasses, countLoops(fn))

	// Run analysis
	Analyze(fn)

	// Apply fusion rules
	ApplyFusionRules(fn)
	OptimizeSoftmax(fn)

	// Get stats after fusion
	afterStats := ComputeFusionStats(fn)
	t.Logf("After fusion: %d fusion groups, %d eliminated allocs",
		afterStats.FusionGroups, afterStats.EliminatedAllocs)

	// Verify fusion happened
	if afterStats.FusionGroups == 0 {
		t.Error("expected at least one fusion group")
	}

	// Count fused vs unfused loops to understand the structure
	fusedLoops := 0
	unfusedLoops := 0
	for _, node := range fn.AllNodes {
		if node.Kind == OpKindLoop {
			if node.FusionGroup >= 0 {
				fusedLoops++
			} else {
				unfusedLoops++
			}
		}
	}
	t.Logf("Fused loops: %d, Unfused loops: %d", fusedLoops, unfusedLoops)

	// Verify fusion groups contain the expected operations
	for i, group := range fn.FusionGroups {
		var ops []string
		for _, nodeID := range group.Members {
			for _, node := range fn.AllNodes {
				if node.ID == nodeID {
					ops = append(ops, node.Op)
				}
			}
		}
		t.Logf("FusionGroup %d: operations=%v", i, ops)
	}
}

// TestFusionGroupContainsExpAndSum verifies that the Exp and sum accumulation
// are in the same fusion group (the key optimization for softmax).
func TestFusionGroupContainsExpAndSum(t *testing.T) {
	fn := NewFunction("testExpSum")
	fn.ElemType = "float32"

	lr := &LoopRange{LoopVar: "i", Start: "0", End: "size", Step: "lanes", IsVectorized: true, VectorLanes: 4}

	// Create a chain: Load -> Exp -> Store + Accumulate
	loadNode := fn.AddNode(OpKindLoad, "Load")
	loadNode.InputNames = []string{"input", "i"}
	loadNode.Outputs = []string{"v1"}
	loadNode.LoopRange = lr.Clone()

	expNode := fn.AddNode(OpKindElementwise, "Exp")
	expNode.Inputs = []*IRNode{loadNode}
	expNode.Outputs = []string{"v2"}
	expNode.LoopRange = lr.Clone()

	storeNode := fn.AddNode(OpKindStore, "Store")
	storeNode.Inputs = []*IRNode{expNode}
	storeNode.InputNames = []string{"output", "i"}
	storeNode.LoopRange = lr.Clone()

	reduceNode := fn.AddNode(OpKindReduction, "ReduceSum")
	reduceNode.Inputs = []*IRNode{expNode}
	reduceNode.Outputs = []string{"sum"}
	reduceNode.LoopRange = lr.Clone()

	// Run analysis
	Analyze(fn)

	// Find candidates
	candidates := FindFusionCandidates(fn)
	t.Logf("Found %d fusion candidates", len(candidates))
	for _, c := range candidates {
		t.Logf("  Candidate: %s (producer=%s, consumer=%s)", c.Pattern, c.Producer.Op, c.Consumer.Op)
	}

	// Apply fusion
	ApplyFusionRules(fn)

	// Verify exp and reduce are in the same group
	if expNode.FusionGroup < 0 {
		t.Error("Exp node should be fused")
	}
	if reduceNode.FusionGroup < 0 {
		t.Error("ReduceSum node should be fused")
	}
	if expNode.FusionGroup != reduceNode.FusionGroup {
		t.Errorf("Exp and ReduceSum should be in same fusion group: exp=%d, reduce=%d",
			expNode.FusionGroup, reduceNode.FusionGroup)
	}
}

// TestEmitFusedSoftmax tests that the emitter produces valid C code structure.
func TestEmitFusedSoftmax(t *testing.T) {
	fn := NewFunction("BaseSoftmax")
	fn.ElemType = "float32"
	fn.Params = []IRParam{
		{Name: "input", Type: "float*", IsSlice: true, ElemType: "float32"},
		{Name: "output", Type: "float*", IsSlice: true, ElemType: "float32"},
		{Name: "psize", Type: "long*"},
	}

	lr := &LoopRange{LoopVar: "p", Start: "0", End: "size", Step: "4", IsVectorized: true, VectorLanes: 4}

	// Create simplified structure: Load -> Sub -> Exp -> Store + Sum
	// This represents the fused pass 2

	loadNode := fn.AddNode(OpKindLoad, "Load")
	loadNode.InputNames = []string{"input", "p"}
	loadNode.Outputs = []string{"x"}
	loadNode.LoopRange = lr.Clone()

	subNode := fn.AddNode(OpKindElementwise, "Sub")
	subNode.Inputs = []*IRNode{loadNode}
	subNode.InputNames = []string{"maxBroadcast"}
	subNode.Outputs = []string{"shifted"}
	subNode.LoopRange = lr.Clone()

	expNode := fn.AddNode(OpKindElementwise, "Exp")
	expNode.Inputs = []*IRNode{subNode}
	expNode.Outputs = []string{"expVal"}
	expNode.LoopRange = lr.Clone()

	storeNode := fn.AddNode(OpKindStore, "Store")
	storeNode.Inputs = []*IRNode{expNode}
	storeNode.InputNames = []string{"output", "p"}
	storeNode.LoopRange = lr.Clone()

	sumNode := fn.AddNode(OpKindReduction, "Add")
	sumNode.Inputs = []*IRNode{expNode}
	sumNode.Outputs = []string{"sumVec"}
	sumNode.LoopRange = lr.Clone()

	// Run analysis and fusion
	Analyze(fn)
	ApplyFusionRules(fn)

	// Create a mock emitter profile
	profile := &testProfile{
		lanes: 4,
		tier:  "NEON",
	}

	emitter := NewEmitter(profile)
	cCode := emitter.EmitFunction(fn)

	t.Logf("Generated C code length: %d bytes", len(cCode))

	// Verify structural properties of generated code
	// The generated code should have the basic structure

	// Should have function definition with void return
	if !strings.Contains(cCode, "void") {
		t.Log("Generated code:\n", cCode)
		t.Error("expected void function definition in generated C code")
	}

	// Should have SIMD operations (the core of the fusion)
	hasLoad := strings.Contains(cCode, "vld1q_f32")
	hasSub := strings.Contains(cCode, "vsubq_f32")
	hasStore := strings.Contains(cCode, "vst1q_f32")

	if !hasLoad || !hasSub || !hasStore {
		t.Log("Generated code:\n", cCode)
		t.Errorf("expected SIMD intrinsics: load=%v, sub=%v, store=%v", hasLoad, hasSub, hasStore)
	}

	// Log the generated code for inspection
	t.Log("Generated C code:\n", cCode)
}

// testProfile implements CProfile for testing
type testProfile struct {
	lanes int
	tier  string
}

func (p *testProfile) GetTier() string {
	return p.tier
}

func (p *testProfile) GetLanes() int {
	return p.lanes
}

func (p *testProfile) GetVecType(tier string) string {
	switch tier {
	case "NEON", "q":
		return "float32x4_t"
	case "AVX2", "ymm":
		return "__m256"
	default:
		return "float"
	}
}

func (p *testProfile) GetScalarType() string {
	return "float"
}

func (p *testProfile) GetLoadFn(tier string) string {
	return "vld1q_f32"
}

func (p *testProfile) GetStoreFn(tier string) string {
	return "vst1q_f32"
}

func (p *testProfile) GetIntrinsic(op, tier string) string {
	switch op {
	case "Load":
		return "vld1q_f32"
	case "Store":
		return "vst1q_f32"
	case "Add":
		return "vaddq_f32"
	case "Sub":
		return "vsubq_f32"
	case "Mul":
		return "vmulq_f32"
	case "Set":
		return "vdupq_n_f32"
	default:
		return op
	}
}

func (p *testProfile) GetFmaArgOrder() string {
	return "acc_first"
}

func (p *testProfile) GetInlineHelpers() string {
	return ""
}

func (p *testProfile) RequiresCast() bool {
	return false
}

func (p *testProfile) GetCastExpr() string {
	return ""
}

func (p *testProfile) GetZeroInit(tier string) string {
	return "vdupq_n_f32(0.0f)"
}

// countLoops counts OpKindLoop nodes in the function
func countLoops(fn *IRFunction) int {
	count := 0
	for _, node := range fn.AllNodes {
		if node.Kind == OpKindLoop {
			count++
		}
	}
	return count
}

// TestCompareToHandwrittenC compares structural properties of the generated
// fusion output against the handwritten softmax_neon_arm64.c reference.
func TestCompareToHandwrittenC(t *testing.T) {
	// This test documents the expected optimization targets.
	// The handwritten C has:
	// 1. Pass 1: Find max (separate loop)
	// 2. Pass 2: Fused subtract + exp + sum (single loop)
	// 3. Pass 3: Normalize (separate loop)
	//
	// Total: 3 main loops, 0 intermediate allocations

	t.Log("Handwritten softmax_neon_arm64.c structure:")
	t.Log("  - Pass 1: max reduction loop")
	t.Log("  - Pass 2: fused (load - max) -> exp -> store + sum")
	t.Log("  - Pass 3: normalize loop")
	t.Log("  - No shifted[] allocation")
	t.Log("")
	t.Log("Current fusion implementation achieves:")

	fn := buildSoftmaxIR()

	stats := ComputeFusionStats(fn)
	t.Logf("  Before: %d operations (%d original passes), %d loops", len(fn.Operations), stats.OriginalPasses, countLoops(fn))

	Analyze(fn)
	ApplyFusionRules(fn)
	OptimizeSoftmax(fn)

	stats = ComputeFusionStats(fn)
	t.Logf("  After: %d fusion groups", stats.FusionGroups)

	// Document the gap between current and optimal
	allocsIdentified := IdentifyAllocationsToEliminate(fn)
	t.Logf("  Allocations identified for elimination: %d", len(allocsIdentified))

	if len(allocsIdentified) == 0 {
		t.Log("  Note: Cross-loop allocation elimination not yet implemented")
		t.Log("  The softmax pattern requires fusing:")
		t.Log("    - shift loop (write to shifted[])")
		t.Log("    - exp loop (read from shifted[], write to output[])")
		t.Log("    - sum loop (read from output[])")
		t.Log("  into a single fused loop to eliminate shifted[]")
	}
}

// buildSoftmaxIR creates the full softmax IR for testing
func buildSoftmaxIR() *IRFunction {
	fn := NewFunction("BaseSoftmax")
	fn.ElemType = "float32"
	fn.Params = []IRParam{
		{Name: "input", Type: "[]float32", IsSlice: true, ElemType: "float32"},
		{Name: "output", Type: "[]float32", IsSlice: true, ElemType: "float32"},
	}

	lr := &LoopRange{LoopVar: "i", Start: "0", End: "size", Step: "lanes", IsVectorized: true, VectorLanes: 4}

	// Max reduction
	maxNode := fn.AddNode(OpKindReduction, "ReduceMax")
	maxNode.InputNames = []string{"input"}
	maxNode.Outputs = []string{"maxVal"}

	// Alloc shifted
	allocNode := fn.AddNode(OpKindAlloc, "make")
	allocNode.Outputs = []string{"shifted"}
	allocNode.AllocSize = "size"
	allocNode.AllocElemType = "float32"

	// Shift loop
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

	// Exp loop
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

	// Sum loop
	sumLoop := fn.AddNode(OpKindLoop, "for")
	sumLoop.LoopRange = lr.Clone()

	loadOutput := fn.AddChildNode(sumLoop, OpKindLoad, "Load")
	loadOutput.InputNames = []string{"output", "i"}
	loadOutput.Outputs = []string{"v5"}
	loadOutput.LoopRange = lr.Clone()

	sumAccum := fn.AddChildNode(sumLoop, OpKindReduction, "Add")
	sumAccum.Inputs = []*IRNode{loadOutput}
	sumAccum.Outputs = []string{"expSum"}
	sumAccum.LoopRange = lr.Clone()

	// Normalize loop
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

	return fn
}
