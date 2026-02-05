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
	"bytes"
	"fmt"
	"slices"
	"strings"
)

// CProfile abstracts the target-specific intrinsic profile.
// This interface is implemented by the main package's CIntrinsicProfile.
type CProfile interface {
	// GetTier returns the primary SIMD tier name (e.g., "q", "ymm", "zmm").
	GetTier() string

	// GetLanes returns the number of elements per vector.
	GetLanes() int

	// GetVecType returns the C vector type for the tier.
	GetVecType(tier string) string

	// GetScalarType returns the C scalar type.
	GetScalarType() string

	// GetLoadFn returns the load intrinsic.
	GetLoadFn(tier string) string

	// GetStoreFn returns the store intrinsic.
	GetStoreFn(tier string) string

	// GetIntrinsic returns the intrinsic for an operation.
	GetIntrinsic(op, tier string) string

	// GetFmaArgOrder returns "acc_first" or "acc_last" for FMA.
	GetFmaArgOrder() string

	// GetInlineHelpers returns helper functions to include.
	GetInlineHelpers() string

	// RequiresCast returns true if pointer casts are needed for this type.
	RequiresCast() bool

	// GetCastExpr returns the cast expression for a type.
	GetCastExpr() string

	// GetZeroInit returns the zero initialization for a vector.
	GetZeroInit(tier string) string
}

// Emitter generates C code from IR.
type Emitter struct {
	profile CProfile
	buf     *bytes.Buffer
	indent  int

	// varTypes tracks emitted variable types
	varTypes map[string]string

	// emittedVars tracks which variables have been declared
	emittedVars map[string]bool

	// fusedGroups tracks which fusion groups have been emitted
	fusedGroups map[int]bool
}

// NewEmitter creates a new C emitter.
func NewEmitter(profile CProfile) *Emitter {
	return &Emitter{
		profile:     profile,
		buf:         &bytes.Buffer{},
		varTypes:    make(map[string]string),
		emittedVars: make(map[string]bool),
		fusedGroups: make(map[int]bool),
	}
}

// EmitFunction generates C code for an IRFunction.
func (e *Emitter) EmitFunction(fn *IRFunction) string {
	e.buf.Reset()
	e.varTypes = make(map[string]string)
	e.emittedVars = make(map[string]bool)
	e.fusedGroups = make(map[int]bool)
	e.indent = 0

	// Emit inline helpers if needed
	if helpers := e.profile.GetInlineHelpers(); helpers != "" {
		e.writef("%s\n", helpers)
	}

	// Emit function signature
	e.emitFunctionSignature(fn)

	// Emit function body
	e.indent = 1
	e.emitParamDerefs(fn)
	e.emitOperations(fn.Operations, fn)

	// Close function
	e.indent = 0
	e.writef("}\n")

	return e.buf.String()
}

// emitFunctionSignature emits the C function signature.
func (e *Emitter) emitFunctionSignature(fn *IRFunction) {
	// Build parameter list
	var params []string

	for _, p := range fn.Params {
		cType, cName := e.paramToCType(p)
		params = append(params, cType+" "+cName)
	}

	// Function returns void in GOAT convention; outputs are via pointers
	e.writef("void %s(%s) {\n", e.cFuncName(fn), strings.Join(params, ", "))
}

// paramToCType converts an IR parameter to C type and name.
func (e *Emitter) paramToCType(p IRParam) (cType, cName string) {
	if p.IsSlice {
		// Slice → pointer
		elemType := e.goTypeToCType(p.ElemType)
		return elemType + " *", p.Name
	} else if p.IsInt {
		// Integer → long pointer (GOAT convention)
		return "long *", "p" + p.Name
	} else if p.IsFloat {
		// Float scalar → pointer
		scalarType := e.goTypeToCType(p.Type)
		return scalarType + " *", "p" + p.Name
	}
	return "void *", p.Name
}

// goTypeToCType converts a Go type to C type.
func (e *Emitter) goTypeToCType(goType string) string {
	switch goType {
	case "float32":
		return "float"
	case "float64":
		return "double"
	case "int", "int64":
		return "long"
	case "int32":
		return "int"
	case "uint64":
		return "unsigned long"
	case "uint32":
		return "unsigned int"
	case "uint8", "byte":
		return "unsigned char"
	case "T":
		// Generic - use profile's scalar type
		return e.profile.GetScalarType()
	default:
		return goType
	}
}

// cFuncName generates the C function name.
func (e *Emitter) cFuncName(fn *IRFunction) string {
	// Convert BaseFoo to foo_c_<type>_<target>
	name := fn.Name
	if strings.HasPrefix(name, "Base") {
		name = strings.ToLower(name[4:5]) + name[5:]
	}
	return fmt.Sprintf("%s_c_%s_%s", name, fn.ElemType, e.profile.GetTier())
}

// emitParamDerefs emits dereferences for pointer parameters.
func (e *Emitter) emitParamDerefs(fn *IRFunction) {
	for _, p := range fn.Params {
		if p.IsInt {
			e.writef("long %s = *p%s;\n", p.Name, p.Name)
		} else if p.IsFloat && !p.IsSlice {
			cType := e.goTypeToCType(p.Type)
			e.writef("%s %s = *p%s;\n", cType, p.Name, p.Name)
		}
	}
}

// emitOperations emits a sequence of IR operations.
func (e *Emitter) emitOperations(ops []*IRNode, fn *IRFunction) {
	for _, op := range ops {
		// Skip nodes that are part of a fused group (emitted with root)
		if op.FusionGroup >= 0 && !op.IsFusionRoot {
			continue
		}

		// If this is a fusion root, emit the entire fused group
		if op.IsFusionRoot && op.FusionGroup >= 0 {
			e.emitFusedGroup(fn, op.FusionGroup)
			continue
		}

		// Emit individual operation
		e.emitOperation(op, fn)
	}
}

// emitOperation emits a single IR operation.
func (e *Emitter) emitOperation(op *IRNode, fn *IRFunction) {
	switch op.Kind {
	case OpKindLoop:
		e.emitLoop(op, fn)
	case OpKindAlloc:
		e.emitAlloc(op)
	case OpKindBroadcast:
		e.emitBroadcast(op)
	case OpKindLoad:
		e.emitLoad(op)
	case OpKindStore:
		e.emitStore(op)
	case OpKindElementwise:
		e.emitElementwise(op)
	case OpKindReduction:
		e.emitReduction(op)
	case OpKindScalar:
		e.emitScalar(op)
	case OpKindCall:
		e.emitCall(op)
	case OpKindControl:
		e.emitControl(op)
	case OpKindNoop:
		// Skip
	}
}

// emitLoop emits a for loop.
func (e *Emitter) emitLoop(op *IRNode, fn *IRFunction) {
	lr := op.LoopRange
	if lr == nil {
		return
	}

	// Emit loop header
	if lr.IsVectorized {
		lanes := e.profile.GetLanes()
		e.writef("for (long %s = %s; %s + %d <= %s; %s += %d) {\n",
			lr.LoopVar, lr.Start, lr.LoopVar, lanes, lr.End, lr.LoopVar, lanes)
	} else {
		e.writef("for (long %s = %s; %s < %s; %s += %s) {\n",
			lr.LoopVar, lr.Start, lr.LoopVar, lr.End, lr.LoopVar, lr.Step)
	}

	// Emit body
	e.indent++
	e.emitOperations(op.Children, fn)
	e.indent--

	e.writef("}\n")
}

// emitAlloc emits an allocation (C99 VLA).
func (e *Emitter) emitAlloc(op *IRNode) {
	if len(op.Outputs) == 0 {
		return
	}

	varName := op.Outputs[0]
	elemType := e.goTypeToCType(op.AllocElemType)

	e.writef("%s %s[%s];\n", elemType, varName, op.AllocSize)
	e.emittedVars[varName] = true
	e.varTypes[varName] = elemType + "[]"
}

// emitBroadcast emits a scalar-to-vector broadcast.
func (e *Emitter) emitBroadcast(op *IRNode) {
	if len(op.Outputs) == 0 {
		return
	}

	tier := e.profile.GetTier()
	vecType := e.profile.GetVecType(tier)
	varName := op.Outputs[0]

	// Get the broadcast value
	var value string
	if len(op.InputNames) > 0 {
		value = op.InputNames[0]
	} else {
		value = op.Op // For Const, the value is stored in Op
	}

	intrinsic := e.profile.GetIntrinsic("Set", tier)
	if intrinsic != "" {
		e.writef("%s %s = %s(%s);\n", vecType, varName, intrinsic, value)
	} else {
		e.writef("%s %s = %s;\n", vecType, varName, value)
	}

	e.emittedVars[varName] = true
	e.varTypes[varName] = vecType
}

// emitLoad emits a vector load.
func (e *Emitter) emitLoad(op *IRNode) {
	if len(op.Outputs) == 0 || len(op.InputNames) < 1 {
		return
	}

	tier := e.profile.GetTier()
	vecType := e.profile.GetVecType(tier)
	varName := op.Outputs[0]
	ptr := op.InputNames[0]

	loadFn := e.profile.GetLoadFn(tier)

	// Handle indexed loads
	if len(op.InputNames) >= 2 {
		idx := op.InputNames[1]
		if e.profile.RequiresCast() {
			cast := e.profile.GetCastExpr()
			e.writef("%s %s = %s((%s)(&%s[%s]));\n", vecType, varName, loadFn, cast, ptr, idx)
		} else {
			e.writef("%s %s = %s(&%s[%s]);\n", vecType, varName, loadFn, ptr, idx)
		}
	} else {
		e.writef("%s %s = %s(%s);\n", vecType, varName, loadFn, ptr)
	}

	e.emittedVars[varName] = true
	e.varTypes[varName] = vecType
}

// emitStore emits a vector store.
func (e *Emitter) emitStore(op *IRNode) {
	if len(op.InputNames) < 2 {
		return
	}

	tier := e.profile.GetTier()
	storeFn := e.profile.GetStoreFn(tier)
	ptr := op.InputNames[0]
	idx := op.InputNames[1]

	// Get the value to store
	var value string
	if len(op.Inputs) > 0 && op.Inputs[0] != nil {
		value = op.Inputs[0].Outputs[0]
	} else if len(op.InputNames) >= 3 {
		value = op.InputNames[2]
	}

	// NEON: store(ptr, val) vs AVX: store(val, ptr)
	if strings.Contains(tier, "q") || strings.Contains(tier, "d") {
		// NEON style
		if e.profile.RequiresCast() {
			cast := e.profile.GetCastExpr()
			e.writef("%s((%s)(&%s[%s]), %s);\n", storeFn, cast, ptr, idx, value)
		} else {
			e.writef("%s(&%s[%s], %s);\n", storeFn, ptr, idx, value)
		}
	} else {
		// AVX style
		e.writef("%s(&%s[%s], %s);\n", storeFn, ptr, idx, value)
	}
}

// emitElementwise emits an elementwise operation.
func (e *Emitter) emitElementwise(op *IRNode) {
	if len(op.Outputs) == 0 {
		return
	}

	tier := e.profile.GetTier()
	vecType := e.profile.GetVecType(tier)
	varName := op.Outputs[0]

	intrinsic := e.profile.GetIntrinsic(op.Op, tier)
	if intrinsic == "" {
		// Fallback to operator
		e.emitBinaryOp(op, vecType)
		return
	}

	// Build arguments
	var args []string
	for _, input := range op.Inputs {
		if input != nil && len(input.Outputs) > 0 {
			args = append(args, input.Outputs[0])
		}
	}
	for _, name := range op.InputNames {
		args = append(args, name)
	}

	// Handle FMA argument order
	if op.Op == "MulAdd" && e.profile.GetFmaArgOrder() == "acc_first" {
		// NEON: fma(acc, a, b)
		if len(args) >= 3 {
			args = []string{args[2], args[0], args[1]}
		}
	}

	// Emit
	if !e.emittedVars[varName] {
		e.writef("%s %s = %s(%s);\n", vecType, varName, intrinsic, strings.Join(args, ", "))
		e.emittedVars[varName] = true
		e.varTypes[varName] = vecType
	} else {
		e.writef("%s = %s(%s);\n", varName, intrinsic, strings.Join(args, ", "))
	}
}

// emitBinaryOp emits a binary operation using operators.
func (e *Emitter) emitBinaryOp(op *IRNode, vecType string) {
	if len(op.Outputs) == 0 {
		return
	}

	varName := op.Outputs[0]

	var left, right string
	if len(op.Inputs) >= 2 {
		if op.Inputs[0] != nil && len(op.Inputs[0].Outputs) > 0 {
			left = op.Inputs[0].Outputs[0]
		}
		if op.Inputs[1] != nil && len(op.Inputs[1].Outputs) > 0 {
			right = op.Inputs[1].Outputs[0]
		}
	}
	if left == "" && len(op.InputNames) > 0 {
		left = op.InputNames[0]
	}
	if right == "" && len(op.InputNames) > 1 {
		right = op.InputNames[1]
	}

	// Map operation to C operator
	var opStr string
	switch op.Op {
	case "Add":
		opStr = "+"
	case "Sub":
		opStr = "-"
	case "Mul":
		opStr = "*"
	case "Div":
		opStr = "/"
	default:
		opStr = op.Op
	}

	if !e.emittedVars[varName] {
		e.writef("%s %s = %s %s %s;\n", vecType, varName, left, opStr, right)
		e.emittedVars[varName] = true
	} else {
		e.writef("%s = %s %s %s;\n", varName, left, opStr, right)
	}
}

// emitReduction emits a reduction operation.
func (e *Emitter) emitReduction(op *IRNode) {
	if len(op.Outputs) == 0 {
		return
	}

	tier := e.profile.GetTier()
	scalarType := e.profile.GetScalarType()
	varName := op.Outputs[0]

	intrinsic := e.profile.GetIntrinsic(op.Op, tier)

	// Get input
	var input string
	if len(op.Inputs) > 0 && op.Inputs[0] != nil && len(op.Inputs[0].Outputs) > 0 {
		input = op.Inputs[0].Outputs[0]
	} else if len(op.InputNames) > 0 {
		input = op.InputNames[0]
	}

	if intrinsic != "" {
		if !e.emittedVars[varName] {
			e.writef("%s %s = %s(%s);\n", scalarType, varName, intrinsic, input)
			e.emittedVars[varName] = true
		} else {
			e.writef("%s = %s(%s);\n", varName, intrinsic, input)
		}
	}

	e.varTypes[varName] = scalarType
}

// emitScalar emits a scalar operation.
func (e *Emitter) emitScalar(op *IRNode) {
	if len(op.Outputs) == 0 {
		return
	}

	scalarType := e.profile.GetScalarType()

	// Handle different scalar operations
	switch op.Op {
	case "len":
		// Already handled as parameter
		return
	case "min", "max":
		e.emitMinMax(op)
	case "Add", "Sub", "Mul", "Div":
		e.emitScalarArith(op, scalarType)
	default:
		// Generic scalar operation
		e.emitScalarArith(op, scalarType)
	}
}

// emitMinMax emits min/max operations.
func (e *Emitter) emitMinMax(op *IRNode) {
	if len(op.Outputs) == 0 || len(op.InputNames) < 2 {
		return
	}

	varName := op.Outputs[0]
	scalarType := e.profile.GetScalarType()

	a, b := op.InputNames[0], op.InputNames[1]

	if op.Op == "min" {
		e.writef("%s %s = (%s < %s) ? %s : %s;\n", scalarType, varName, a, b, a, b)
	} else {
		e.writef("%s %s = (%s > %s) ? %s : %s;\n", scalarType, varName, a, b, a, b)
	}

	e.emittedVars[varName] = true
	e.varTypes[varName] = scalarType
}

// emitScalarArith emits scalar arithmetic.
func (e *Emitter) emitScalarArith(op *IRNode, scalarType string) {
	if len(op.Outputs) == 0 {
		return
	}

	varName := op.Outputs[0]

	var left, right string
	if len(op.Inputs) >= 1 && op.Inputs[0] != nil && len(op.Inputs[0].Outputs) > 0 {
		left = op.Inputs[0].Outputs[0]
	}
	if len(op.Inputs) >= 2 && op.Inputs[1] != nil && len(op.Inputs[1].Outputs) > 0 {
		right = op.Inputs[1].Outputs[0]
	}
	if left == "" && len(op.InputNames) > 0 {
		left = op.InputNames[0]
	}
	if right == "" && len(op.InputNames) > 1 {
		right = op.InputNames[1]
	}

	var opStr string
	switch op.Op {
	case "Add":
		opStr = "+"
	case "Sub":
		opStr = "-"
	case "Mul":
		opStr = "*"
	case "Div":
		opStr = "/"
	default:
		opStr = op.Op
	}

	if !e.emittedVars[varName] {
		e.writef("%s %s = %s %s %s;\n", scalarType, varName, left, opStr, right)
		e.emittedVars[varName] = true
	} else {
		e.writef("%s = %s %s %s;\n", varName, left, opStr, right)
	}

	e.varTypes[varName] = scalarType
}

// emitCall emits a cross-package function call.
func (e *Emitter) emitCall(op *IRNode) {
	// Cross-package calls are inlined by the resolver
	// This should only be reached for unresolved calls
	e.writef("// TODO: inline %s\n", op.CallTarget)
}

// emitControl emits control flow.
func (e *Emitter) emitControl(op *IRNode) {
	switch op.Op {
	case "if":
		// Skip - control flow breaks fusion
	case "return":
		if len(op.InputNames) > 0 {
			e.writef("return %s;\n", op.InputNames[0])
		} else {
			e.writef("return;\n")
		}
	}
}

// emitFusedGroup emits a fused group of operations.
func (e *Emitter) emitFusedGroup(fn *IRFunction, groupID int) {
	if e.fusedGroups[groupID] {
		return
	}
	e.fusedGroups[groupID] = true

	// Find the fusion group info
	var group *FusionGroup
	for i := range fn.FusionGroups {
		if fn.FusionGroups[i].ID == groupID {
			group = &fn.FusionGroups[i]
			break
		}
	}

	if group == nil {
		return
	}

	// Emit based on pattern
	switch group.Pattern {
	case "Elem+Elem":
		e.emitFusedElemElem(fn, group)
	case "Elem+Reduce":
		e.emitFusedMapReduce(fn, group)
	case "AllocElim":
		e.emitWithAllocElimination(fn, group)
	default:
		// Emit unfused
		for _, id := range group.Members {
			if node := fn.GetNode(id); node != nil {
				e.emitOperation(node, fn)
			}
		}
	}
}

// emitFusedElemElem emits vertically fused elementwise operations.
func (e *Emitter) emitFusedElemElem(fn *IRFunction, group *FusionGroup) {
	// All elementwise ops in the group share the same loop
	lr := group.LoopRange
	if lr == nil {
		return
	}

	// Emit single loop with all operations
	lanes := e.profile.GetLanes()

	e.writef("// Fused: %s\n", group.Pattern)
	e.writef("for (long %s = %s; %s + %d <= %s; %s += %d) {\n",
		lr.LoopVar, lr.Start, lr.LoopVar, lanes, lr.End, lr.LoopVar, lanes)

	e.indent++

	// Emit operations in topological order
	for _, id := range group.Members {
		if node := fn.GetNode(id); node != nil {
			e.emitOperation(node, fn)
		}
	}

	e.indent--
	e.writef("}\n")
}

// emitFusedMapReduce emits a fused compute-and-reduce loop.
func (e *Emitter) emitFusedMapReduce(fn *IRFunction, group *FusionGroup) {
	lr := group.LoopRange
	if lr == nil {
		return
	}

	tier := e.profile.GetTier()
	lanes := e.profile.GetLanes()
	vecType := e.profile.GetVecType(tier)
	scalarType := e.profile.GetScalarType()

	// Find the reduction node
	var reduceNode *IRNode
	for _, id := range group.Members {
		if node := fn.GetNode(id); node != nil && node.Kind == OpKindReduction {
			reduceNode = node
			break
		}
	}

	if reduceNode == nil {
		return
	}

	// Initialize vector accumulator
	accName := "_vacc"
	zeroInit := e.profile.GetZeroInit(tier)
	e.writef("// Fused: %s (MapReduce)\n", group.Pattern)
	e.writef("%s %s = %s;\n", vecType, accName, zeroInit)

	// Emit fused loop
	e.writef("for (long %s = %s; %s + %d <= %s; %s += %d) {\n",
		lr.LoopVar, lr.Start, lr.LoopVar, lanes, lr.End, lr.LoopVar, lanes)

	e.indent++

	// Emit elementwise operations
	for _, id := range group.Members {
		node := fn.GetNode(id)
		if node == nil || node.Kind == OpKindReduction {
			continue
		}
		e.emitOperation(node, fn)
	}

	// Accumulate into vector
	addIntrinsic := e.profile.GetIntrinsic("Add", tier)
	if len(reduceNode.Inputs) > 0 && reduceNode.Inputs[0] != nil {
		inputName := reduceNode.Inputs[0].Outputs[0]
		e.writef("%s = %s(%s, %s);\n", accName, addIntrinsic, accName, inputName)
	}

	e.indent--
	e.writef("}\n")

	// Final horizontal reduction
	reduceIntrinsic := e.profile.GetIntrinsic(reduceNode.Op, tier)
	if len(reduceNode.Outputs) > 0 {
		outName := reduceNode.Outputs[0]
		e.writef("%s %s = %s(%s);\n", scalarType, outName, reduceIntrinsic, accName)
		e.emittedVars[outName] = true
	}
}

// emitWithAllocElimination emits with temporary allocation eliminated.
func (e *Emitter) emitWithAllocElimination(fn *IRFunction, group *FusionGroup) {
	// Find operations excluding the allocation
	e.writef("// Fused: %s (allocation eliminated)\n", group.Pattern)

	for _, id := range group.Members {
		node := fn.GetNode(id)
		if node == nil {
			continue
		}

		// Skip eliminated allocations
		if slices.Contains(group.EliminatedAllocs, id) {
			continue
		}

		e.emitOperation(node, fn)
	}
}

// writef writes a formatted line with indentation.
func (e *Emitter) writef(format string, args ...any) {
	for i := 0; i < e.indent; i++ {
		e.buf.WriteString("\t")
	}
	fmt.Fprintf(e.buf, format, args...)
}
