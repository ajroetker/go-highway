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
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

// TestIRNodeTypes verifies the OpKind classification.
func TestIRNodeTypes(t *testing.T) {
	tests := []struct {
		op   string
		want OpKind
	}{
		{"Add", OpKindElementwise},
		{"Sub", OpKindElementwise},
		{"Mul", OpKindElementwise},
		{"MulAdd", OpKindElementwise},
		{"ReduceSum", OpKindReduction},
		{"ReduceMax", OpKindReduction},
		{"Load", OpKindLoad},
		{"Store", OpKindStore},
		{"Set", OpKindBroadcast},
		{"Const", OpKindBroadcast},
	}

	for _, tt := range tests {
		t.Run(tt.op, func(t *testing.T) {
			got := ClassifyHwyOp(tt.op)
			if got != tt.want {
				t.Errorf("ClassifyHwyOp(%q) = %v, want %v", tt.op, got, tt.want)
			}
		})
	}
}

// TestIRBuilderBasic tests basic IR building from a simple function.
func TestIRBuilderBasic(t *testing.T) {
	src := `
package test

func testFunc(input, output []float32, size int) {
	for i := 0; i < size; i++ {
		output[i] = input[i] * 2.0
	}
}
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	// Find the function
	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Name.Name == "testFunc" {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("function not found")
	}

	// Build ParsedFunc
	pf := &ParsedFunc{
		Name: funcDecl.Name.Name,
		Body: funcDecl.Body,
		Params: []ParamInput{
			{Name: "input", Type: "[]float32"},
			{Name: "output", Type: "[]float32"},
			{Name: "size", Type: "int"},
		},
	}

	// Build IR
	builder := NewBuilder(WithElemType("float32"))
	irFunc, err := builder.Build(pf)
	if err != nil {
		t.Fatalf("build IR: %v", err)
	}

	// Verify basics
	if irFunc.Name != "testFunc" {
		t.Errorf("name = %q, want %q", irFunc.Name, "testFunc")
	}
	if len(irFunc.Params) != 3 {
		t.Errorf("params = %d, want 3", len(irFunc.Params))
	}
	if len(irFunc.Operations) == 0 {
		t.Error("expected at least one operation (loop)")
	}

	// Should have a loop
	hasLoop := false
	for _, op := range irFunc.Operations {
		if op.Kind == OpKindLoop {
			hasLoop = true
			break
		}
	}
	if !hasLoop {
		t.Error("expected loop operation")
	}
}

// TestIRBuilderWithHwyCalls tests IR building with hwy.* calls.
func TestIRBuilderWithHwyCalls(t *testing.T) {
	// This test simulates what the builder would produce from parsed hwy calls
	builder := NewBuilder(WithElemType("float32"))
	builder.fn = NewFunction("testVectorOp")
	builder.fn.ElemType = "float32"

	// Create a simple IR manually to test the structure
	loopNode := builder.fn.AddNode(OpKindLoop, "for")
	loopNode.LoopRange = &LoopRange{
		LoopVar:      "i",
		Start:        "0",
		End:          "size",
		Step:         "lanes",
		IsVectorized: true,
		VectorLanes:  4,
	}

	// Add a load operation
	loadNode := builder.fn.AddChildNode(loopNode, OpKindLoad, "Load")
	loadNode.InputNames = []string{"input", "i"}
	loadNode.Outputs = []string{"v1"}
	loadNode.LoopRange = loopNode.LoopRange.Clone()

	// Add an elementwise operation
	mulNode := builder.fn.AddChildNode(loopNode, OpKindElementwise, "Mul")
	mulNode.Inputs = []*IRNode{loadNode}
	mulNode.Outputs = []string{"v2"}
	mulNode.LoopRange = loopNode.LoopRange.Clone()

	// Add a store operation
	storeNode := builder.fn.AddChildNode(loopNode, OpKindStore, "Store")
	storeNode.Inputs = []*IRNode{mulNode}
	storeNode.InputNames = []string{"output", "i"}
	storeNode.LoopRange = loopNode.LoopRange.Clone()

	// Verify structure
	if len(builder.fn.Operations) != 1 {
		t.Errorf("operations = %d, want 1 (loop)", len(builder.fn.Operations))
	}
	if len(loopNode.Children) != 3 {
		t.Errorf("loop children = %d, want 3", len(loopNode.Children))
	}

	// Verify node IDs are unique
	ids := make(map[int]bool)
	for _, n := range builder.fn.AllNodes {
		if ids[n.ID] {
			t.Errorf("duplicate node ID: %d", n.ID)
		}
		ids[n.ID] = true
	}
}

// TestLoopRangeSame tests LoopRange comparison.
func TestLoopRangeSame(t *testing.T) {
	lr1 := &LoopRange{Start: "0", End: "size", Step: "lanes"}
	lr2 := &LoopRange{Start: "0", End: "size", Step: "lanes"}
	lr3 := &LoopRange{Start: "0", End: "size", Step: "1"}

	if !lr1.Same(lr2) {
		t.Error("lr1 and lr2 should be same")
	}
	if lr1.Same(lr3) {
		t.Error("lr1 and lr3 should not be same")
	}
	if lr1.Same(nil) {
		t.Error("lr1 and nil should not be same")
	}
}

// TestAnalyze tests data flow analysis.
func TestAnalyze(t *testing.T) {
	fn := NewFunction("testAnalysis")

	// Create a producer node
	producer := fn.AddNode(OpKindElementwise, "Add")
	producer.Outputs = []string{"result"}

	// Create a consumer node
	consumer := fn.AddNode(OpKindElementwise, "Mul")
	consumer.Inputs = []*IRNode{producer}

	// Run analysis
	Analyze(fn)

	// Verify producer-consumer relationship
	if len(producer.Consumers) != 1 {
		t.Errorf("producer consumers = %d, want 1", len(producer.Consumers))
	}
	if producer.Consumers[0] != consumer {
		t.Error("producer should have consumer as its consumer")
	}
	if len(consumer.Producers) != 1 {
		t.Errorf("consumer producers = %d, want 1", len(consumer.Producers))
	}
	if consumer.Producers[0] != producer {
		t.Error("consumer should have producer as its producer")
	}
}

// TestFusionCandidates tests fusion candidate detection.
func TestFusionCandidates(t *testing.T) {
	fn := NewFunction("testFusion")

	// Create two elementwise nodes that can be fused
	lr := &LoopRange{Start: "0", End: "size", Step: "lanes"}

	node1 := fn.AddNode(OpKindElementwise, "Add")
	node1.Outputs = []string{"temp"}
	node1.LoopRange = lr.Clone()

	node2 := fn.AddNode(OpKindElementwise, "Mul")
	node2.Inputs = []*IRNode{node1}
	node2.Outputs = []string{"result"}
	node2.LoopRange = lr.Clone()

	// Run analysis to establish relationships
	Analyze(fn)

	// Find candidates
	candidates := FindFusionCandidates(fn)

	if len(candidates) == 0 {
		t.Error("expected at least one fusion candidate")
	}

	// Check that we found the Elem+Elem pattern
	found := false
	for _, c := range candidates {
		if c.Pattern == "Elem+Elem" {
			found = true
			if c.Producer != node1 || c.Consumer != node2 {
				t.Error("incorrect producer/consumer in candidate")
			}
			break
		}
	}
	if !found {
		t.Error("expected Elem+Elem fusion candidate")
	}
}

// TestApplyFusionRules tests the fusion application.
func TestApplyFusionRules(t *testing.T) {
	fn := NewFunction("testApplyFusion")

	lr := &LoopRange{Start: "0", End: "size", Step: "lanes"}

	// Create a chain: Load -> Add -> Mul -> Store
	loadNode := fn.AddNode(OpKindLoad, "Load")
	loadNode.Outputs = []string{"v0"}
	loadNode.LoopRange = lr.Clone()

	addNode := fn.AddNode(OpKindElementwise, "Add")
	addNode.Inputs = []*IRNode{loadNode}
	addNode.Outputs = []string{"v1"}
	addNode.LoopRange = lr.Clone()

	mulNode := fn.AddNode(OpKindElementwise, "Mul")
	mulNode.Inputs = []*IRNode{addNode}
	mulNode.Outputs = []string{"v2"}
	mulNode.LoopRange = lr.Clone()

	storeNode := fn.AddNode(OpKindStore, "Store")
	storeNode.Inputs = []*IRNode{mulNode}
	storeNode.LoopRange = lr.Clone()

	// Apply fusion
	ApplyFusionRules(fn)

	// Check that fusion groups were created
	if len(fn.FusionGroups) == 0 {
		t.Error("expected fusion groups to be created")
	}

	// Check that at least some nodes are fused
	fusedCount := 0
	for _, node := range fn.AllNodes {
		if node.FusionGroup >= 0 {
			fusedCount++
		}
	}
	if fusedCount < 2 {
		t.Errorf("expected at least 2 fused nodes, got %d", fusedCount)
	}
}

// TestComputeFusionStats tests fusion statistics computation.
func TestComputeFusionStats(t *testing.T) {
	fn := NewFunction("testStats")

	// Add some operations
	fn.AddNode(OpKindLoop, "for")
	fn.AddNode(OpKindAlloc, "make")
	fn.AddNode(OpKindLoop, "for")

	stats := ComputeFusionStats(fn)

	if stats.OriginalPasses != 3 { // 2 loops + 1 alloc
		t.Errorf("original passes = %d, want 3", stats.OriginalPasses)
	}
}

// TestClassifyMathOp tests math operation classification.
func TestClassifyMathOp(t *testing.T) {
	tests := []struct {
		op   string
		want OpKind
	}{
		{"BaseExpVec", OpKindElementwise},
		{"BaseSigmoidVec", OpKindElementwise},
		{"BaseTanhVec", OpKindElementwise},
		{"Sqrt", OpKindScalar},
		{"Exp", OpKindScalar},
	}

	for _, tt := range tests {
		t.Run(tt.op, func(t *testing.T) {
			got := ClassifyMathOp(tt.op)
			if got != tt.want {
				t.Errorf("ClassifyMathOp(%q) = %v, want %v", tt.op, got, tt.want)
			}
		})
	}
}

// TestClassifyAlgoOp tests algo operation classification.
func TestClassifyAlgoOp(t *testing.T) {
	tests := []struct {
		op   string
		want OpKind
	}{
		{"BaseApply", OpKindCall},
		{"BaseSum", OpKindReduction},
		{"BaseMax", OpKindReduction},
	}

	for _, tt := range tests {
		t.Run(tt.op, func(t *testing.T) {
			got := ClassifyAlgoOp(tt.op)
			if got != tt.want {
				t.Errorf("ClassifyAlgoOp(%q) = %v, want %v", tt.op, got, tt.want)
			}
		})
	}
}

// TestIRFunctionString tests the String() methods.
func TestIRFunctionString(t *testing.T) {
	fn := NewFunction("testFunc")
	fn.Params = []IRParam{
		{Name: "input", Type: "[]float32", IsSlice: true},
		{Name: "size", Type: "int", IsInt: true},
	}
	fn.AddNode(OpKindLoop, "for")

	str := fn.String()
	if !strings.Contains(str, "testFunc") {
		t.Error("function string should contain function name")
	}
	if !strings.Contains(str, "input") {
		t.Error("function string should contain param names")
	}
}

// TestIRNodeString tests the IRNode.String() method.
func TestIRNodeString(t *testing.T) {
	node := NewNode(1, OpKindElementwise, "Add")
	node.Outputs = []string{"result"}

	str := node.String()
	if !strings.Contains(str, "Add") {
		t.Error("node string should contain operation name")
	}
	if !strings.Contains(str, "result") {
		t.Error("node string should contain output name")
	}
}

// TestHasSingleConsumer tests the HasSingleConsumer method.
func TestHasSingleConsumer(t *testing.T) {
	node := NewNode(1, OpKindElementwise, "Add")

	// No consumers
	if node.HasSingleConsumer() {
		t.Error("node with 0 consumers should not have single consumer")
	}

	// One consumer
	consumer := NewNode(2, OpKindElementwise, "Mul")
	node.Consumers = []*IRNode{consumer}
	if !node.HasSingleConsumer() {
		t.Error("node with 1 consumer should have single consumer")
	}

	// Two consumers
	consumer2 := NewNode(3, OpKindStore, "Store")
	node.Consumers = append(node.Consumers, consumer2)
	if node.HasSingleConsumer() {
		t.Error("node with 2 consumers should not have single consumer")
	}
}
