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

// Package ir provides an intermediate representation for Go SIMD code,
// enabling cross-cutting optimizations like operation fusion before
// emitting target-specific C code.
package ir

import (
	"fmt"
	"go/ast"
	"strings"
)

// OpKind categorizes IR operations for pattern matching during fusion.
type OpKind int

const (
	// OpKindElementwise represents element-by-element operations that can be
	// fused vertically (Add, Sub, Mul, Exp, etc.).
	OpKindElementwise OpKind = iota

	// OpKindReduction represents operations that reduce a vector/slice to a
	// scalar (ReduceSum, ReduceMax, etc.). Can fuse with preceding elementwise
	// for MapReduce patterns.
	OpKindReduction

	// OpKindLoad represents memory loads (hwy.Load, slice indexing).
	OpKindLoad

	// OpKindStore represents memory stores (hwy.Store, slice assignment).
	OpKindStore

	// OpKindAlloc represents memory allocation (make([]T, n)).
	OpKindAlloc

	// OpKindBroadcast represents scalar-to-vector broadcasts (hwy.Set, hwy.Const).
	OpKindBroadcast

	// OpKindLoop represents a for loop containing vectorized operations.
	OpKindLoop

	// OpKindCall represents a cross-package function call that needs resolution
	// (e.g., algo.BaseApply, math.BaseExpVec).
	OpKindCall

	// OpKindScalar represents scalar operations that don't participate in
	// vector fusion (comparisons for control flow, scalar arithmetic).
	OpKindScalar

	// OpKindControl represents control flow (if, return) that breaks fusion chains.
	OpKindControl

	// OpKindNoop represents no-op nodes used for structuring (block boundaries).
	OpKindNoop
)

// String returns a human-readable name for the OpKind.
func (k OpKind) String() string {
	switch k {
	case OpKindElementwise:
		return "Elementwise"
	case OpKindReduction:
		return "Reduction"
	case OpKindLoad:
		return "Load"
	case OpKindStore:
		return "Store"
	case OpKindAlloc:
		return "Alloc"
	case OpKindBroadcast:
		return "Broadcast"
	case OpKindLoop:
		return "Loop"
	case OpKindCall:
		return "Call"
	case OpKindScalar:
		return "Scalar"
	case OpKindControl:
		return "Control"
	case OpKindNoop:
		return "Noop"
	default:
		return fmt.Sprintf("OpKind(%d)", k)
	}
}

// IRNode represents a single operation in the IR graph.
type IRNode struct {
	// ID is a unique identifier for this node within its function.
	ID int

	// Kind categorizes this operation for fusion pattern matching.
	Kind OpKind

	// Op is the specific operation name (e.g., "Add", "Exp", "ReduceSum", "Load").
	Op string

	// Inputs are the nodes whose outputs feed into this operation.
	// For elementwise ops: operands. For Store: value and address.
	Inputs []*IRNode

	// InputNames are the variable names corresponding to inputs when
	// inputs are external (parameters) or when tracking names is useful.
	InputNames []string

	// Outputs are the variable names this node produces.
	// Most nodes have a single output; Load4 has multiple.
	Outputs []string

	// OutputTypes are the types of each output (e.g., "float32", "float32x4_t").
	OutputTypes []string

	// LoopRange describes the iteration space if this is a loop node or
	// the containing loop's range for nodes inside loops.
	LoopRange *LoopRange

	// Children contains the body operations for OpKindLoop nodes.
	Children []*IRNode

	// CallTarget holds cross-package function reference for OpKindCall nodes.
	// Format: "package.FuncName" (e.g., "math.BaseExpVec").
	CallTarget string

	// CallTypeArgs holds type arguments for generic calls.
	// E.g., ["float32"] for BaseExpVec[float32].
	CallTypeArgs []string

	// FuncArg holds the function argument for higher-order functions like BaseApply.
	// Format: "package.FuncName" (e.g., "math.BaseExpVec").
	FuncArg string

	// AllocSize holds the allocation size expression for OpKindAlloc nodes.
	AllocSize string

	// AllocElemType holds the element type for allocations.
	AllocElemType string

	// ---- Data flow (populated by analysis) ----

	// Producers are nodes that produce inputs to this node.
	// Computed during data flow analysis.
	Producers []*IRNode

	// Consumers are nodes that use this node's outputs.
	// Computed during data flow analysis.
	Consumers []*IRNode

	// ---- Fusion state ----

	// FusionGroup identifies which fusion group this node belongs to.
	// -1 means unfused. Nodes with the same positive group ID will be
	// emitted together in a single fused loop.
	FusionGroup int

	// IsFusionRoot is true if this node is the root of its fusion group
	// (typically the reduction or final store).
	IsFusionRoot bool

	// IsFusionEliminated is true if this node should be eliminated during
	// code emission (e.g., store/load pair replaced by register passing).
	IsFusionEliminated bool

	// ---- Source tracking ----

	// ASTNode is the original Go AST node, preserved for error messages.
	ASTNode ast.Node
}

// NewNode creates a new IRNode with the given kind and operation.
func NewNode(id int, kind OpKind, op string) *IRNode {
	return &IRNode{
		ID:          id,
		Kind:        kind,
		Op:          op,
		FusionGroup: -1,
	}
}

// IsVector returns true if this node operates on vectors (not scalars).
func (n *IRNode) IsVector() bool {
	switch n.Kind {
	case OpKindElementwise, OpKindLoad, OpKindStore, OpKindBroadcast:
		return true
	case OpKindLoop:
		return true // loops contain vector ops
	case OpKindReduction:
		// Reductions consume vectors but produce scalars
		return false
	default:
		return false
	}
}

// HasSingleConsumer returns true if this node's output is used by exactly one node.
// Important for allocation elimination—temps with single consumers can often be removed.
func (n *IRNode) HasSingleConsumer() bool {
	return len(n.Consumers) == 1
}

// LoopRange describes the iteration space for a loop or operation within a loop.
type LoopRange struct {
	// LoopVar is the iteration variable name (e.g., "i", "ii").
	LoopVar string

	// Start is the starting value expression (e.g., "0").
	Start string

	// End is the ending value expression (e.g., "size", "len(input)").
	End string

	// Step is the increment expression (e.g., "lanes", "1").
	Step string

	// IsVectorized indicates if this loop processes vector-sized chunks.
	IsVectorized bool

	// VectorLanes is the number of elements per vector iteration.
	// 0 for scalar loops.
	VectorLanes int
}

// Clone creates a deep copy of the LoopRange.
func (lr *LoopRange) Clone() *LoopRange {
	if lr == nil {
		return nil
	}
	return &LoopRange{
		LoopVar:      lr.LoopVar,
		Start:        lr.Start,
		End:          lr.End,
		Step:         lr.Step,
		IsVectorized: lr.IsVectorized,
		VectorLanes:  lr.VectorLanes,
	}
}

// Same returns true if two LoopRanges represent the same iteration space.
// Important for fusion—operations with different ranges cannot be fused.
func (lr *LoopRange) Same(other *LoopRange) bool {
	if lr == nil || other == nil {
		return lr == other
	}
	return lr.Start == other.Start &&
		lr.End == other.End &&
		lr.Step == other.Step
}

// IRParam represents a function parameter in the IR.
type IRParam struct {
	// Name is the parameter name.
	Name string

	// Type is the Go type (e.g., "[]float32", "int").
	Type string

	// IsSlice indicates if this is a slice parameter.
	IsSlice bool

	// IsInt indicates if this is an integer parameter.
	IsInt bool

	// IsFloat indicates if this is a floating-point scalar parameter.
	IsFloat bool

	// ElemType is the element type for slices (e.g., "float32").
	ElemType string
}

// IRFunction represents a function in the IR.
type IRFunction struct {
	// Name is the function name.
	Name string

	// TypeParams holds generic type parameters.
	TypeParams []TypeParam

	// Params are the function parameters.
	Params []IRParam

	// Returns are the return value types.
	Returns []IRParam

	// Operations is the linear sequence of IR nodes in the function body.
	// Loop nodes contain their body in Children.
	Operations []*IRNode

	// AllNodes is a map from ID to node for quick lookup.
	AllNodes map[int]*IRNode

	// FusionGroups describes the fusion groups after fusion pass.
	// Each group is a slice of node IDs that will be emitted together.
	FusionGroups []FusionGroup

	// ElemType is the concrete element type for this instantiation
	// (e.g., "float32" when T=float32).
	ElemType string

	// ---- Metadata ----

	// Package is the containing package name.
	Package string

	// ImportPath is the full import path (e.g., "github.com/ajroetker/go-highway/hwy/contrib/nn").
	ImportPath string

	// NextID is used to allocate unique node IDs.
	nextID int
}

// TypeParam mirrors the parser's TypeParam for generic parameters.
type TypeParam struct {
	Name       string // T
	Constraint string // hwy.Floats
}

// NewFunction creates a new IRFunction with the given name.
func NewFunction(name string) *IRFunction {
	return &IRFunction{
		Name:     name,
		AllNodes: make(map[int]*IRNode),
	}
}

// NewNodeID allocates and returns a new unique node ID.
func (f *IRFunction) NewNodeID() int {
	id := f.nextID
	f.nextID++
	return id
}

// AddNode creates a new node, adds it to the function, and returns it.
func (f *IRFunction) AddNode(kind OpKind, op string) *IRNode {
	node := NewNode(f.NewNodeID(), kind, op)
	f.Operations = append(f.Operations, node)
	f.AllNodes[node.ID] = node
	return node
}

// AddChildNode creates a new node and adds it to a parent loop's children.
func (f *IRFunction) AddChildNode(parent *IRNode, kind OpKind, op string) *IRNode {
	node := NewNode(f.NewNodeID(), kind, op)
	parent.Children = append(parent.Children, node)
	f.AllNodes[node.ID] = node
	return node
}

// GetNode returns the node with the given ID, or nil if not found.
func (f *IRFunction) GetNode(id int) *IRNode {
	return f.AllNodes[id]
}

// FusionGroup describes a set of nodes that will be fused together.
type FusionGroup struct {
	// ID is the fusion group identifier (matches IRNode.FusionGroup).
	ID int

	// Root is the ID of the root node (final output of the fused computation).
	Root int

	// Members are the IDs of all nodes in this group.
	Members []int

	// Pattern describes the fusion pattern (e.g., "Elem+Elem", "Elem+Reduce").
	Pattern string

	// LoopRange is the shared iteration space for all members.
	LoopRange *LoopRange

	// EliminatedAllocs are the IDs of allocation nodes eliminated by this fusion.
	EliminatedAllocs []int
}

// String returns a debug string representation of the IRNode.
func (n *IRNode) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "IRNode{ID:%d Kind:%s Op:%q", n.ID, n.Kind, n.Op)
	if len(n.Outputs) > 0 {
		fmt.Fprintf(&sb, " Out:%v", n.Outputs)
	}
	if len(n.Inputs) > 0 {
		ids := make([]int, len(n.Inputs))
		for i, in := range n.Inputs {
			ids[i] = in.ID
		}
		fmt.Fprintf(&sb, " In:%v", ids)
	}
	if n.CallTarget != "" {
		fmt.Fprintf(&sb, " Call:%s", n.CallTarget)
	}
	if n.FusionGroup >= 0 {
		fmt.Fprintf(&sb, " Fused:%d", n.FusionGroup)
	}
	sb.WriteString("}")
	return sb.String()
}

// String returns a debug string representation of the IRFunction.
func (f *IRFunction) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "IRFunction{Name:%s Params:[", f.Name)
	for i, p := range f.Params {
		if i > 0 {
			sb.WriteString(", ")
		}
		fmt.Fprintf(&sb, "%s:%s", p.Name, p.Type)
	}
	fmt.Fprintf(&sb, "] Ops:%d", len(f.Operations))
	if len(f.FusionGroups) > 0 {
		fmt.Fprintf(&sb, " FusionGroups:%d", len(f.FusionGroups))
	}
	sb.WriteString("}")
	return sb.String()
}

// ClassifyHwyOp returns the OpKind for a hwy.* operation.
func ClassifyHwyOp(op string) OpKind {
	switch op {
	// Elementwise arithmetic
	case "Add", "Sub", "Mul", "Div", "Neg", "Abs",
		"MulAdd", "MulSub", "NegMulAdd", "NegMulSub",
		"Min", "Max", "Sqrt", "Rsqrt",
		"Floor", "Ceil", "Round", "RoundToEven", "Trunc":
		return OpKindElementwise

	// Elementwise bitwise (for masks and integer ops)
	case "And", "Or", "Xor", "AndNot", "Not",
		"ShiftLeft", "ShiftRight":
		return OpKindElementwise

	// Elementwise comparison (produce masks)
	case "Equal", "NotEqual", "Less", "LessEqual",
		"Greater", "GreaterEqual", "LessThan", "GreaterThan":
		return OpKindElementwise

	// Elementwise selection
	case "IfThenElse", "Merge":
		return OpKindElementwise

	// Elementwise conversions
	case "ConvertToInt32", "ConvertToInt64", "ConvertToFloat32", "ConvertToFloat64":
		return OpKindElementwise

	// Reductions
	case "ReduceSum", "ReduceMin", "ReduceMax", "ReduceAnd", "ReduceOr":
		return OpKindReduction

	// Loads
	case "Load", "LoadSlice", "Load4":
		return OpKindLoad

	// Stores
	case "Store", "StoreSlice":
		return OpKindStore

	// Broadcasts
	case "Set", "Const", "Zero":
		return OpKindBroadcast

	// Special operations that need individual handling
	case "Pow2", "GetExponent", "GetMantissa", "ConvertExponentToFloat":
		return OpKindElementwise

	case "PopCount":
		return OpKindElementwise

	case "BitsFromMask":
		return OpKindScalar // produces scalar from mask

	default:
		return OpKindScalar
	}
}

// ClassifyMathOp returns the OpKind for a math.* or contrib/math operation.
func ClassifyMathOp(op string) OpKind {
	switch op {
	// Vec-to-Vec math functions (all elementwise)
	case "BaseExpVec", "BaseLogVec", "BaseSigmoidVec", "BaseTanhVec",
		"BaseSinVec", "BaseCosVec", "BaseErfVec",
		"BaseLog2Vec", "BaseLog10Vec", "BaseExp2Vec",
		"BaseSinhVec", "BaseCoshVec", "BaseAsinhVec", "BaseAcoshVec", "BaseAtanhVec",
		"BasePowVec":
		return OpKindElementwise

	// Standard math library scalar functions
	case "Sqrt", "Exp", "Log", "Sin", "Cos", "Tan",
		"Sinh", "Cosh", "Tanh", "Asin", "Acos", "Atan",
		"Abs", "Floor", "Ceil", "Round", "Trunc",
		"Pow", "Mod", "Max", "Min", "Inf":
		return OpKindScalar

	default:
		return OpKindCall // Unknown - treat as cross-package call
	}
}

// ClassifyAlgoOp returns the OpKind for an algo.* operation.
func ClassifyAlgoOp(op string) OpKind {
	switch op {
	// Higher-order functions that apply transforms
	case "BaseApply", "BaseApplyInPlace":
		return OpKindCall // needs resolution

	// Reductions
	case "BaseSum", "BaseMax", "BaseMin":
		return OpKindReduction

	default:
		return OpKindCall
	}
}
