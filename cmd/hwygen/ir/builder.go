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
	"fmt"
	"go/ast"
	"go/token"
	"strings"
)

// Builder walks Go AST and produces IR nodes.
// It is similar to CASTTranslator but produces IR instead of C strings,
// enabling fusion and other cross-cutting optimizations.
type Builder struct {
	// fn is the function being built.
	fn *IRFunction

	// imports maps local package names to import paths.
	imports map[string]string

	// vars tracks variable name → producing node for data flow.
	vars map[string]*IRNode

	// varTypes tracks variable name → type for type inference.
	varTypes map[string]string

	// currentLoop is the loop node we're currently building, if any.
	currentLoop *IRNode

	// resolver resolves cross-package function references.
	resolver Resolver

	// elemType is the concrete element type (e.g., "float32").
	elemType string
}

// Resolver is an interface for resolving cross-package function calls.
// It is implemented by FunctionRegistry in resolver.go.
type Resolver interface {
	// Resolve looks up a function by its qualified name (e.g., "math.BaseExpVec").
	// Returns the resolved function's IR or nil if not found.
	Resolve(qualifiedName string, typeArgs []string) (*IRFunction, error)
}

// BuilderOption configures the Builder.
type BuilderOption func(*Builder)

// WithResolver sets the cross-package resolver.
func WithResolver(r Resolver) BuilderOption {
	return func(b *Builder) {
		b.resolver = r
	}
}

// WithImports sets the import map.
func WithImports(imports map[string]string) BuilderOption {
	return func(b *Builder) {
		b.imports = imports
	}
}

// WithElemType sets the concrete element type.
func WithElemType(elemType string) BuilderOption {
	return func(b *Builder) {
		b.elemType = elemType
	}
}

// NewBuilder creates a new IR builder.
func NewBuilder(opts ...BuilderOption) *Builder {
	b := &Builder{
		imports:  make(map[string]string),
		vars:     make(map[string]*IRNode),
		varTypes: make(map[string]string),
		elemType: "float32",
	}
	for _, opt := range opts {
		opt(b)
	}
	return b
}

// ParsedFunc mirrors the main package's ParsedFunc to avoid import cycles.
// The main package should convert its ParsedFunc to this type before calling Build.
type ParsedFunc struct {
	Name       string
	TypeParams []TypeParamInput
	Params     []ParamInput
	Returns    []ParamInput
	Body       *ast.BlockStmt
}

// TypeParamInput mirrors the main package's TypeParam.
type TypeParamInput struct {
	Name       string
	Constraint string
}

// ParamInput mirrors the main package's Param.
type ParamInput struct {
	Name string
	Type string
}

// Build transforms a ParsedFunc into an IRFunction.
func (b *Builder) Build(pf *ParsedFunc) (*IRFunction, error) {
	b.fn = NewFunction(pf.Name)
	b.fn.ElemType = b.elemType
	b.vars = make(map[string]*IRNode)
	b.varTypes = make(map[string]string)

	// Copy type parameters
	for _, tp := range pf.TypeParams {
		b.fn.TypeParams = append(b.fn.TypeParams, TypeParam{
			Name:       tp.Name,
			Constraint: tp.Constraint,
		})
	}

	// Build parameter list and track their types
	for _, p := range pf.Params {
		param := IRParam{
			Name: p.Name,
			Type: p.Type,
		}
		if strings.HasPrefix(p.Type, "[]") {
			param.IsSlice = true
			param.ElemType = strings.TrimPrefix(p.Type, "[]")
			// Substitute generic type parameter
			if param.ElemType == "T" {
				param.ElemType = b.elemType
			}
		} else if p.Type == "int" || p.Type == "int64" {
			param.IsInt = true
		} else if p.Type == "float32" || p.Type == "float64" || p.Type == "T" {
			param.IsFloat = true
		}
		b.fn.Params = append(b.fn.Params, param)
		b.varTypes[p.Name] = p.Type
	}

	// Build returns
	for _, r := range pf.Returns {
		ret := IRParam{
			Name: r.Name,
			Type: r.Type,
		}
		if strings.HasPrefix(r.Type, "[]") {
			ret.IsSlice = true
			ret.ElemType = strings.TrimPrefix(r.Type, "[]")
		}
		b.fn.Returns = append(b.fn.Returns, ret)
	}

	// Build function body
	if pf.Body != nil {
		if err := b.buildBlock(pf.Body); err != nil {
			return nil, fmt.Errorf("build body: %w", err)
		}
	}

	return b.fn, nil
}

// buildBlock processes a block statement.
func (b *Builder) buildBlock(block *ast.BlockStmt) error {
	for _, stmt := range block.List {
		if err := b.buildStmt(stmt); err != nil {
			return err
		}
	}
	return nil
}

// buildStmt processes a single statement.
func (b *Builder) buildStmt(stmt ast.Stmt) error {
	switch s := stmt.(type) {
	case *ast.AssignStmt:
		return b.buildAssign(s)
	case *ast.DeclStmt:
		return b.buildDecl(s)
	case *ast.ForStmt:
		return b.buildFor(s)
	case *ast.RangeStmt:
		return b.buildRange(s)
	case *ast.IfStmt:
		return b.buildIf(s)
	case *ast.ReturnStmt:
		return b.buildReturn(s)
	case *ast.ExprStmt:
		return b.buildExprStmt(s)
	case *ast.IncDecStmt:
		return b.buildIncDec(s)
	case *ast.BlockStmt:
		return b.buildBlock(s)
	default:
		// Skip unsupported statements
		return nil
	}
}

// buildAssign processes an assignment statement.
func (b *Builder) buildAssign(stmt *ast.AssignStmt) error {
	// Handle different assignment types
	if len(stmt.Lhs) != len(stmt.Rhs) && len(stmt.Rhs) == 1 {
		// Multi-value return (e.g., x, y := fn())
		return b.buildMultiAssign(stmt)
	}

	for i, lhs := range stmt.Lhs {
		rhs := stmt.Rhs[i]

		// Handle compound assignments (+=, -=, etc.)
		if stmt.Tok != token.ASSIGN && stmt.Tok != token.DEFINE {
			return b.buildCompoundAssign(stmt, i)
		}

		// Get the target variable name
		lhsName := b.exprToName(lhs)

		// Check if RHS is a make() call for allocation
		if call, ok := rhs.(*ast.CallExpr); ok {
			if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "make" {
				return b.buildMakeAlloc(lhsName, call)
			}
		}

		// Build the RHS expression
		rhsNode, err := b.buildExpr(rhs)
		if err != nil {
			return err
		}

		if rhsNode != nil {
			// Record the variable → node mapping
			b.vars[lhsName] = rhsNode
			if len(rhsNode.Outputs) == 0 {
				rhsNode.Outputs = []string{lhsName}
			}
		}

		// Infer and record type
		if stmt.Tok == token.DEFINE {
			b.varTypes[lhsName] = b.inferType(rhs)
		}
	}
	return nil
}

// buildCompoundAssign handles +=, -=, etc.
func (b *Builder) buildCompoundAssign(stmt *ast.AssignStmt, idx int) error {
	lhs := stmt.Lhs[idx]
	rhs := stmt.Rhs[idx]
	lhsName := b.exprToName(lhs)

	// Get the current value
	prevNode := b.vars[lhsName]

	// Build the RHS
	rhsNode, err := b.buildExpr(rhs)
	if err != nil {
		return err
	}

	// Map token to operation
	var op string
	switch stmt.Tok {
	case token.ADD_ASSIGN:
		op = "Add"
	case token.SUB_ASSIGN:
		op = "Sub"
	case token.MUL_ASSIGN:
		op = "Mul"
	case token.QUO_ASSIGN:
		op = "Div"
	default:
		op = "Unknown"
	}

	// Create the compound operation node
	node := b.fn.AddNode(OpKindElementwise, op)
	node.Inputs = []*IRNode{prevNode, rhsNode}
	node.Outputs = []string{lhsName}
	node.ASTNode = stmt

	// Check if this is a reduction pattern (scalar += vector reduction)
	if rhsNode != nil && rhsNode.Kind == OpKindReduction {
		// This is accumulating reduction results
		node.Kind = OpKindScalar
	}

	b.vars[lhsName] = node
	return nil
}

// buildMakeAlloc handles make([]T, size) allocations.
func (b *Builder) buildMakeAlloc(name string, call *ast.CallExpr) error {
	if len(call.Args) < 2 {
		return nil
	}

	// Extract type and size
	var elemType string
	if arrType, ok := call.Args[0].(*ast.ArrayType); ok {
		elemType = b.typeToString(arrType.Elt)
	}

	size := b.exprToString(call.Args[1])

	node := b.fn.AddNode(OpKindAlloc, "make")
	node.Outputs = []string{name}
	node.AllocSize = size
	node.AllocElemType = elemType
	node.ASTNode = call

	b.vars[name] = node
	b.varTypes[name] = "[]" + elemType
	return nil
}

// buildMultiAssign handles multi-value returns like Load4.
func (b *Builder) buildMultiAssign(stmt *ast.AssignStmt) error {
	call, ok := stmt.Rhs[0].(*ast.CallExpr)
	if !ok {
		return nil
	}

	// Build the call node
	node, err := b.buildExpr(call)
	if err != nil {
		return err
	}

	if node == nil {
		return nil
	}

	// Record all output variables
	for _, lhs := range stmt.Lhs {
		name := b.exprToName(lhs)
		node.Outputs = append(node.Outputs, name)
		b.vars[name] = node
	}

	return nil
}

// buildDecl handles variable declarations.
func (b *Builder) buildDecl(stmt *ast.DeclStmt) error {
	genDecl, ok := stmt.Decl.(*ast.GenDecl)
	if !ok || genDecl.Tok != token.VAR {
		return nil
	}

	for _, spec := range genDecl.Specs {
		valueSpec, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}

		typeStr := b.typeToString(valueSpec.Type)
		for i, name := range valueSpec.Names {
			b.varTypes[name.Name] = typeStr

			// Handle initializers
			if i < len(valueSpec.Values) {
				node, err := b.buildExpr(valueSpec.Values[i])
				if err != nil {
					return err
				}
				if node != nil {
					node.Outputs = []string{name.Name}
					b.vars[name.Name] = node
				}
			}
		}
	}
	return nil
}

// buildFor handles for loops.
func (b *Builder) buildFor(stmt *ast.ForStmt) error {
	loopNode := b.fn.AddNode(OpKindLoop, "for")
	loopNode.ASTNode = stmt

	// Parse loop range
	lr := &LoopRange{}

	// Init: i := 0
	if init, ok := stmt.Init.(*ast.AssignStmt); ok && len(init.Lhs) > 0 {
		lr.LoopVar = b.exprToName(init.Lhs[0])
		if len(init.Rhs) > 0 {
			lr.Start = b.exprToString(init.Rhs[0])
		}
	}

	// Cond: i < size
	if binExpr, ok := stmt.Cond.(*ast.BinaryExpr); ok {
		lr.End = b.exprToString(binExpr.Y)
	}

	// Post: i += lanes
	if post, ok := stmt.Post.(*ast.AssignStmt); ok {
		if post.Tok == token.ADD_ASSIGN && len(post.Rhs) > 0 {
			lr.Step = b.exprToString(post.Rhs[0])
			// Check if this is a vectorized loop
			stepStr := lr.Step
			if strings.Contains(stepStr, "lanes") ||
				strings.Contains(stepStr, "MaxLanes") ||
				strings.Contains(stepStr, "NumLanes") {
				lr.IsVectorized = true
			}
		}
	}

	loopNode.LoopRange = lr

	// Save current loop context
	prevLoop := b.currentLoop
	b.currentLoop = loopNode

	// Build loop body into children
	if stmt.Body != nil {
		for _, bodyStmt := range stmt.Body.List {
			if err := b.buildStmtInLoop(loopNode, bodyStmt); err != nil {
				return err
			}
		}
	}

	// Restore loop context
	b.currentLoop = prevLoop

	return nil
}

// buildStmtInLoop processes a statement inside a loop, adding to the loop's children.
func (b *Builder) buildStmtInLoop(loopNode *IRNode, stmt ast.Stmt) error {
	switch s := stmt.(type) {
	case *ast.AssignStmt:
		return b.buildAssignInLoop(loopNode, s)
	case *ast.ExprStmt:
		return b.buildExprStmtInLoop(loopNode, s)
	case *ast.IfStmt:
		// Control flow inside loop
		node := b.fn.AddChildNode(loopNode, OpKindControl, "if")
		node.ASTNode = s
		return nil
	default:
		return nil
	}
}

// buildAssignInLoop processes assignments inside a loop.
func (b *Builder) buildAssignInLoop(loopNode *IRNode, stmt *ast.AssignStmt) error {
	for i, lhs := range stmt.Lhs {
		// Handle index expressions (slice[i] = value)
		if indexExpr, ok := lhs.(*ast.IndexExpr); ok {
			// This is a store
			sliceName := b.exprToName(indexExpr.X)
			indexStr := b.exprToString(indexExpr.Index)

			var rhsNode *IRNode
			if i < len(stmt.Rhs) {
				var err error
				rhsNode, err = b.buildExprInLoop(loopNode, stmt.Rhs[i])
				if err != nil {
					return err
				}
			}

			storeNode := b.fn.AddChildNode(loopNode, OpKindStore, "Store")
			storeNode.Inputs = []*IRNode{rhsNode}
			storeNode.InputNames = []string{sliceName, indexStr}
			storeNode.LoopRange = loopNode.LoopRange.Clone()
			storeNode.ASTNode = stmt
			continue
		}

		// Regular assignment
		lhsName := b.exprToName(lhs)

		// Handle compound assignments
		if stmt.Tok != token.ASSIGN && stmt.Tok != token.DEFINE {
			return b.buildCompoundAssignInLoop(loopNode, stmt, i)
		}

		if i < len(stmt.Rhs) {
			rhsNode, err := b.buildExprInLoop(loopNode, stmt.Rhs[i])
			if err != nil {
				return err
			}
			if rhsNode != nil {
				rhsNode.Outputs = []string{lhsName}
				b.vars[lhsName] = rhsNode
			}
		}
	}
	return nil
}

// buildCompoundAssignInLoop handles += etc inside loops.
func (b *Builder) buildCompoundAssignInLoop(loopNode *IRNode, stmt *ast.AssignStmt, idx int) error {
	lhs := stmt.Lhs[idx]
	rhs := stmt.Rhs[idx]
	lhsName := b.exprToName(lhs)

	// Get current value
	prevNode := b.vars[lhsName]

	// Build RHS
	rhsNode, err := b.buildExprInLoop(loopNode, rhs)
	if err != nil {
		return err
	}

	// Map token to op
	var op string
	switch stmt.Tok {
	case token.ADD_ASSIGN:
		op = "Add"
	case token.SUB_ASSIGN:
		op = "Sub"
	case token.MUL_ASSIGN:
		op = "Mul"
	case token.QUO_ASSIGN:
		op = "Div"
	default:
		op = "Unknown"
	}

	// Check if this is accumulation (scalar += ...)
	// This is important for fusion - identifies reduction patterns
	node := b.fn.AddChildNode(loopNode, OpKindScalar, op)
	node.Inputs = []*IRNode{prevNode, rhsNode}
	node.Outputs = []string{lhsName}
	node.LoopRange = loopNode.LoopRange.Clone()
	node.ASTNode = stmt

	b.vars[lhsName] = node
	return nil
}

// buildExprInLoop processes an expression inside a loop.
func (b *Builder) buildExprInLoop(loopNode *IRNode, expr ast.Expr) (*IRNode, error) {
	switch e := expr.(type) {
	case *ast.CallExpr:
		return b.buildCallInLoop(loopNode, e)
	case *ast.BinaryExpr:
		return b.buildBinaryInLoop(loopNode, e)
	case *ast.IndexExpr:
		return b.buildIndexInLoop(loopNode, e)
	case *ast.Ident:
		// Variable reference
		if node, ok := b.vars[e.Name]; ok {
			return node, nil
		}
		return nil, nil
	case *ast.BasicLit:
		// Literal value - create broadcast
		node := b.fn.AddChildNode(loopNode, OpKindBroadcast, "Const")
		node.Outputs = []string{e.Value}
		node.ASTNode = e
		return node, nil
	default:
		return nil, nil
	}
}

// buildCallInLoop processes a function call inside a loop.
func (b *Builder) buildCallInLoop(loopNode *IRNode, call *ast.CallExpr) (*IRNode, error) {
	// Check for package-qualified call
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		pkgIdent, ok := sel.X.(*ast.Ident)
		if !ok {
			return nil, nil
		}
		pkgName := pkgIdent.Name
		funcName := sel.Sel.Name

		// Handle hwy.* calls
		if pkgName == "hwy" {
			return b.buildHwyCallInLoop(loopNode, funcName, call)
		}

		// Handle cross-package calls (algo.*, math.*)
		return b.buildCrossPackageCallInLoop(loopNode, pkgName, funcName, call)
	}

	// Local function call
	if ident, ok := call.Fun.(*ast.Ident); ok {
		// Check if it's a type conversion
		if isTypeName(ident.Name) {
			// Type conversion - pass through
			if len(call.Args) > 0 {
				return b.buildExprInLoop(loopNode, call.Args[0])
			}
		}
	}

	return nil, nil
}

// buildHwyCallInLoop processes an hwy.* call inside a loop.
func (b *Builder) buildHwyCallInLoop(loopNode *IRNode, funcName string, call *ast.CallExpr) (*IRNode, error) {
	kind := ClassifyHwyOp(funcName)
	node := b.fn.AddChildNode(loopNode, kind, funcName)
	node.LoopRange = loopNode.LoopRange.Clone()
	node.ASTNode = call

	// Build inputs
	for _, arg := range call.Args {
		input, err := b.buildExprInLoop(loopNode, arg)
		if err != nil {
			return nil, err
		}
		if input != nil {
			node.Inputs = append(node.Inputs, input)
		} else {
			// Record as string for parameters/variables not in IR
			node.InputNames = append(node.InputNames, b.exprToString(arg))
		}
	}

	return node, nil
}

// buildCrossPackageCallInLoop handles algo.*, math.* calls inside loops.
func (b *Builder) buildCrossPackageCallInLoop(loopNode *IRNode, pkgName, funcName string, call *ast.CallExpr) (*IRNode, error) {
	qualifiedName := pkgName + "." + funcName

	// Extract type arguments if generic
	var typeArgs []string
	if indexExpr, ok := call.Fun.(*ast.IndexExpr); ok {
		if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
			funcName = sel.Sel.Name
			qualifiedName = pkgName + "." + funcName
		}
		typeArgs = append(typeArgs, b.typeToString(indexExpr.Index))
	}

	// Classify the operation
	var kind OpKind
	switch pkgName {
	case "math":
		kind = ClassifyMathOp(funcName)
	case "algo":
		kind = ClassifyAlgoOp(funcName)
	default:
		kind = OpKindCall
	}

	node := b.fn.AddChildNode(loopNode, kind, funcName)
	node.CallTarget = qualifiedName
	node.CallTypeArgs = typeArgs
	node.LoopRange = loopNode.LoopRange.Clone()
	node.ASTNode = call

	// Build inputs
	for _, arg := range call.Args {
		// Check if arg is a function reference (for higher-order functions)
		if sel, ok := arg.(*ast.SelectorExpr); ok {
			if pkgIdent, ok := sel.X.(*ast.Ident); ok {
				// This is a function argument like math.BaseExpVec
				node.FuncArg = pkgIdent.Name + "." + sel.Sel.Name
				continue
			}
		}
		if indexExpr, ok := arg.(*ast.IndexExpr); ok {
			// Generic function like math.BaseExpVec[T]
			if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
				if pkgIdent, ok := sel.X.(*ast.Ident); ok {
					node.FuncArg = pkgIdent.Name + "." + sel.Sel.Name
					continue
				}
			}
		}

		input, err := b.buildExprInLoop(loopNode, arg)
		if err != nil {
			return nil, err
		}
		if input != nil {
			node.Inputs = append(node.Inputs, input)
		} else {
			node.InputNames = append(node.InputNames, b.exprToString(arg))
		}
	}

	return node, nil
}

// buildBinaryInLoop processes a binary expression inside a loop.
func (b *Builder) buildBinaryInLoop(loopNode *IRNode, expr *ast.BinaryExpr) (*IRNode, error) {
	left, err := b.buildExprInLoop(loopNode, expr.X)
	if err != nil {
		return nil, err
	}
	right, err := b.buildExprInLoop(loopNode, expr.Y)
	if err != nil {
		return nil, err
	}

	// Map operator to hwy operation
	var op string
	switch expr.Op {
	case token.ADD:
		op = "Add"
	case token.SUB:
		op = "Sub"
	case token.MUL:
		op = "Mul"
	case token.QUO:
		op = "Div"
	case token.LSS:
		op = "Less"
	case token.GTR:
		op = "Greater"
	case token.LEQ:
		op = "LessEqual"
	case token.GEQ:
		op = "GreaterEqual"
	case token.EQL:
		op = "Equal"
	case token.NEQ:
		op = "NotEqual"
	default:
		op = expr.Op.String()
	}

	node := b.fn.AddChildNode(loopNode, OpKindElementwise, op)
	node.Inputs = []*IRNode{left, right}
	node.LoopRange = loopNode.LoopRange.Clone()
	node.ASTNode = expr

	return node, nil
}

// buildIndexInLoop processes slice indexing inside a loop.
func (b *Builder) buildIndexInLoop(loopNode *IRNode, expr *ast.IndexExpr) (*IRNode, error) {
	sliceName := b.exprToName(expr.X)
	indexStr := b.exprToString(expr.Index)

	// Check if this is vectorized access (hwy.Load pattern)
	// For now, treat as scalar load
	node := b.fn.AddChildNode(loopNode, OpKindLoad, "LoadScalar")
	node.InputNames = []string{sliceName, indexStr}
	node.LoopRange = loopNode.LoopRange.Clone()
	node.ASTNode = expr

	return node, nil
}

// buildRange handles for-range loops.
func (b *Builder) buildRange(stmt *ast.RangeStmt) error {
	loopNode := b.fn.AddNode(OpKindLoop, "range")
	loopNode.ASTNode = stmt

	// Parse range
	lr := &LoopRange{
		Start: "0",
	}

	// Key (i) in "for i := range n"
	if key, ok := stmt.Key.(*ast.Ident); ok {
		lr.LoopVar = key.Name
	}

	// Range expression
	lr.End = b.exprToString(stmt.X)
	lr.Step = "1"

	loopNode.LoopRange = lr

	// Build body
	prevLoop := b.currentLoop
	b.currentLoop = loopNode

	if stmt.Body != nil {
		for _, bodyStmt := range stmt.Body.List {
			if err := b.buildStmtInLoop(loopNode, bodyStmt); err != nil {
				return err
			}
		}
	}

	b.currentLoop = prevLoop
	return nil
}

// buildIf handles if statements.
func (b *Builder) buildIf(stmt *ast.IfStmt) error {
	// Create control flow node
	node := b.fn.AddNode(OpKindControl, "if")
	node.ASTNode = stmt

	// We don't deeply analyze if bodies for now - they break fusion chains
	// In the future, we could handle simple conditionals that don't affect
	// the main vectorized computation

	return nil
}

// buildReturn handles return statements.
func (b *Builder) buildReturn(stmt *ast.ReturnStmt) error {
	node := b.fn.AddNode(OpKindControl, "return")
	node.ASTNode = stmt

	for _, result := range stmt.Results {
		node.InputNames = append(node.InputNames, b.exprToString(result))
	}

	return nil
}

// buildExprStmt handles expression statements (function calls for side effects).
func (b *Builder) buildExprStmt(stmt *ast.ExprStmt) error {
	if call, ok := stmt.X.(*ast.CallExpr); ok {
		_, err := b.buildExpr(call)
		return err
	}
	return nil
}

// buildExprStmtInLoop handles expression statements inside loops.
func (b *Builder) buildExprStmtInLoop(loopNode *IRNode, stmt *ast.ExprStmt) error {
	if call, ok := stmt.X.(*ast.CallExpr); ok {
		_, err := b.buildCallInLoop(loopNode, call)
		return err
	}
	return nil
}

// buildIncDec handles increment/decrement statements.
func (b *Builder) buildIncDec(stmt *ast.IncDecStmt) error {
	name := b.exprToName(stmt.X)
	prevNode := b.vars[name]

	op := "Add"
	if stmt.Tok == token.DEC {
		op = "Sub"
	}

	node := b.fn.AddNode(OpKindScalar, op)
	node.Inputs = []*IRNode{prevNode}
	node.Outputs = []string{name}
	node.ASTNode = stmt

	b.vars[name] = node
	return nil
}

// buildExpr processes a top-level expression.
func (b *Builder) buildExpr(expr ast.Expr) (*IRNode, error) {
	switch e := expr.(type) {
	case *ast.CallExpr:
		return b.buildCall(e)
	case *ast.BinaryExpr:
		return b.buildBinary(e)
	case *ast.IndexExpr:
		return b.buildIndex(e)
	case *ast.Ident:
		if node, ok := b.vars[e.Name]; ok {
			return node, nil
		}
		return nil, nil
	case *ast.BasicLit:
		node := b.fn.AddNode(OpKindBroadcast, "Const")
		node.Outputs = []string{e.Value}
		node.ASTNode = e
		return node, nil
	case *ast.UnaryExpr:
		return b.buildUnary(e)
	default:
		return nil, nil
	}
}

// buildCall processes a function call.
func (b *Builder) buildCall(call *ast.CallExpr) (*IRNode, error) {
	// Check for package-qualified call
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if pkgIdent, ok := sel.X.(*ast.Ident); ok {
			pkgName := pkgIdent.Name
			funcName := sel.Sel.Name

			if pkgName == "hwy" {
				return b.buildHwyCall(funcName, call)
			}

			return b.buildCrossPackageCall(pkgName, funcName, call)
		}
	}

	// Generic call with type parameters
	if indexExpr, ok := call.Fun.(*ast.IndexExpr); ok {
		if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
			if pkgIdent, ok := sel.X.(*ast.Ident); ok {
				pkgName := pkgIdent.Name
				funcName := sel.Sel.Name
				typeArg := b.typeToString(indexExpr.Index)
				return b.buildCrossPackageCallWithType(pkgName, funcName, typeArg, call)
			}
		}
	}

	// Local function call
	if ident, ok := call.Fun.(*ast.Ident); ok {
		// Built-in functions
		switch ident.Name {
		case "len":
			node := b.fn.AddNode(OpKindScalar, "len")
			if len(call.Args) > 0 {
				node.InputNames = []string{b.exprToString(call.Args[0])}
			}
			node.ASTNode = call
			return node, nil
		case "min", "max":
			node := b.fn.AddNode(OpKindScalar, ident.Name)
			for _, arg := range call.Args {
				node.InputNames = append(node.InputNames, b.exprToString(arg))
			}
			node.ASTNode = call
			return node, nil
		case "copy":
			node := b.fn.AddNode(OpKindNoop, "copy")
			for _, arg := range call.Args {
				node.InputNames = append(node.InputNames, b.exprToString(arg))
			}
			node.ASTNode = call
			return node, nil
		}

		// Type conversion
		if isTypeName(ident.Name) && len(call.Args) > 0 {
			return b.buildExpr(call.Args[0])
		}
	}

	return nil, nil
}

// buildHwyCall processes an hwy.* call outside a loop.
func (b *Builder) buildHwyCall(funcName string, call *ast.CallExpr) (*IRNode, error) {
	kind := ClassifyHwyOp(funcName)
	node := b.fn.AddNode(kind, funcName)
	node.ASTNode = call

	// Build inputs
	for _, arg := range call.Args {
		input, err := b.buildExpr(arg)
		if err != nil {
			return nil, err
		}
		if input != nil {
			node.Inputs = append(node.Inputs, input)
		} else {
			node.InputNames = append(node.InputNames, b.exprToString(arg))
		}
	}

	return node, nil
}

// buildCrossPackageCall processes a cross-package call.
func (b *Builder) buildCrossPackageCall(pkgName, funcName string, call *ast.CallExpr) (*IRNode, error) {
	qualifiedName := pkgName + "." + funcName

	var kind OpKind
	switch pkgName {
	case "math", "stdmath":
		kind = ClassifyMathOp(funcName)
	case "algo":
		kind = ClassifyAlgoOp(funcName)
	default:
		kind = OpKindCall
	}

	node := b.fn.AddNode(kind, funcName)
	node.CallTarget = qualifiedName
	node.ASTNode = call

	// Build inputs
	for _, arg := range call.Args {
		// Check for function argument
		if sel, ok := arg.(*ast.SelectorExpr); ok {
			if pkgIdent, ok := sel.X.(*ast.Ident); ok {
				node.FuncArg = pkgIdent.Name + "." + sel.Sel.Name
				continue
			}
		}

		input, err := b.buildExpr(arg)
		if err != nil {
			return nil, err
		}
		if input != nil {
			node.Inputs = append(node.Inputs, input)
		} else {
			node.InputNames = append(node.InputNames, b.exprToString(arg))
		}
	}

	return node, nil
}

// buildCrossPackageCallWithType handles generic cross-package calls.
func (b *Builder) buildCrossPackageCallWithType(pkgName, funcName, typeArg string, call *ast.CallExpr) (*IRNode, error) {
	qualifiedName := pkgName + "." + funcName

	var kind OpKind
	switch pkgName {
	case "math":
		kind = ClassifyMathOp(funcName)
	case "algo":
		kind = ClassifyAlgoOp(funcName)
	default:
		kind = OpKindCall
	}

	node := b.fn.AddNode(kind, funcName)
	node.CallTarget = qualifiedName
	node.CallTypeArgs = []string{typeArg}
	node.ASTNode = call

	// Build inputs
	for _, arg := range call.Args {
		// Check for function argument (higher-order)
		if indexExpr, ok := arg.(*ast.IndexExpr); ok {
			if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
				if pkgIdent, ok := sel.X.(*ast.Ident); ok {
					node.FuncArg = pkgIdent.Name + "." + sel.Sel.Name
					continue
				}
			}
		}
		if sel, ok := arg.(*ast.SelectorExpr); ok {
			if pkgIdent, ok := sel.X.(*ast.Ident); ok {
				node.FuncArg = pkgIdent.Name + "." + sel.Sel.Name
				continue
			}
		}

		input, err := b.buildExpr(arg)
		if err != nil {
			return nil, err
		}
		if input != nil {
			node.Inputs = append(node.Inputs, input)
		} else {
			node.InputNames = append(node.InputNames, b.exprToString(arg))
		}
	}

	return node, nil
}

// buildBinary processes a binary expression.
func (b *Builder) buildBinary(expr *ast.BinaryExpr) (*IRNode, error) {
	left, err := b.buildExpr(expr.X)
	if err != nil {
		return nil, err
	}
	right, err := b.buildExpr(expr.Y)
	if err != nil {
		return nil, err
	}

	var op string
	switch expr.Op {
	case token.ADD:
		op = "Add"
	case token.SUB:
		op = "Sub"
	case token.MUL:
		op = "Mul"
	case token.QUO:
		op = "Div"
	case token.LSS:
		op = "Less"
	case token.GTR:
		op = "Greater"
	default:
		op = expr.Op.String()
	}

	node := b.fn.AddNode(OpKindScalar, op)
	node.Inputs = []*IRNode{left, right}
	node.ASTNode = expr

	return node, nil
}

// buildUnary processes a unary expression.
func (b *Builder) buildUnary(expr *ast.UnaryExpr) (*IRNode, error) {
	operand, err := b.buildExpr(expr.X)
	if err != nil {
		return nil, err
	}

	var op string
	switch expr.Op {
	case token.SUB:
		op = "Neg"
	case token.NOT:
		op = "Not"
	default:
		op = expr.Op.String()
	}

	node := b.fn.AddNode(OpKindScalar, op)
	node.Inputs = []*IRNode{operand}
	node.ASTNode = expr

	return node, nil
}

// buildIndex processes an index expression.
func (b *Builder) buildIndex(expr *ast.IndexExpr) (*IRNode, error) {
	sliceName := b.exprToName(expr.X)
	indexStr := b.exprToString(expr.Index)

	node := b.fn.AddNode(OpKindLoad, "LoadScalar")
	node.InputNames = []string{sliceName, indexStr}
	node.ASTNode = expr

	return node, nil
}

// Helper functions

func (b *Builder) exprToName(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.IndexExpr:
		return b.exprToName(e.X)
	case *ast.SelectorExpr:
		return b.exprToName(e.X) + "." + e.Sel.Name
	default:
		return ""
	}
}

func (b *Builder) exprToString(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.BasicLit:
		return e.Value
	case *ast.BinaryExpr:
		return b.exprToString(e.X) + e.Op.String() + b.exprToString(e.Y)
	case *ast.CallExpr:
		return b.exprToString(e.Fun) + "(...)"
	case *ast.SelectorExpr:
		return b.exprToString(e.X) + "." + e.Sel.Name
	case *ast.IndexExpr:
		return b.exprToString(e.X) + "[" + b.exprToString(e.Index) + "]"
	case *ast.ParenExpr:
		return "(" + b.exprToString(e.X) + ")"
	case *ast.UnaryExpr:
		return e.Op.String() + b.exprToString(e.X)
	default:
		return fmt.Sprintf("%T", expr)
	}
}

func (b *Builder) typeToString(expr ast.Expr) string {
	if expr == nil {
		return ""
	}
	switch e := expr.(type) {
	case *ast.Ident:
		name := e.Name
		// Substitute T with concrete type
		if name == "T" {
			return b.elemType
		}
		return name
	case *ast.ArrayType:
		return "[]" + b.typeToString(e.Elt)
	case *ast.SelectorExpr:
		return b.exprToString(e.X) + "." + e.Sel.Name
	case *ast.IndexExpr:
		return b.typeToString(e.X) + "[" + b.typeToString(e.Index) + "]"
	default:
		return fmt.Sprintf("%T", expr)
	}
}

func (b *Builder) inferType(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.BasicLit:
		switch e.Kind {
		case token.INT:
			return "int"
		case token.FLOAT:
			return b.elemType
		case token.STRING:
			return "string"
		}
	case *ast.CallExpr:
		// For hwy calls, infer vector type
		if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
			if pkgIdent, ok := sel.X.(*ast.Ident); ok && pkgIdent.Name == "hwy" {
				switch sel.Sel.Name {
				case "Load", "Add", "Mul", "Sub", "Div":
					return "hwy.Vec[" + b.elemType + "]"
				case "ReduceSum", "ReduceMax", "ReduceMin":
					return b.elemType
				}
			}
		}
	case *ast.IndexExpr:
		// Slice indexing returns element type
		sliceType := b.inferType(e.X)
		if elemType, ok := strings.CutPrefix(sliceType, "[]"); ok {
			return elemType
		}
	case *ast.Ident:
		if t, ok := b.varTypes[e.Name]; ok {
			return t
		}
	}
	return b.elemType
}

func isTypeName(name string) bool {
	switch name {
	case "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64",
		"float32", "float64", "complex64", "complex128",
		"byte", "rune", "string", "bool", "T":
		return true
	}
	return false
}

