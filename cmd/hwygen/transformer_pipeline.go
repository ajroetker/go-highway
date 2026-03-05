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

package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"maps"
	"strconv"
	"strings"
)


// OperationComplexity categorizes operations by their register pressure and latency.
type OperationComplexity int

const (
	// ComplexitySimple: basic arithmetic (Add, Sub, Mul, FMA) - low register pressure
	ComplexitySimple OperationComplexity = iota
	// ComplexityMedium: comparisons, blends, shuffles - moderate register pressure
	ComplexityMedium
	// ComplexityComplex: transcendentals (Exp, Log, Sin, etc.) - high register pressure
	ComplexityComplex
	// ComplexityReduction: reductions (Sum, Min, Max) - data dependencies limit ILP
	ComplexityReduction
)

// goBuiltinIdents contains Go's predeclared identifiers (types, functions, constants).
var goBuiltinIdents = map[string]bool{
	"true": true, "false": true, "nil": true,
	"int": true, "int8": true, "int16": true, "int32": true, "int64": true,
	"uint": true, "uint8": true, "uint16": true, "uint32": true, "uint64": true,
	"float32": true, "float64": true, "complex64": true, "complex128": true,
	"string": true, "bool": true, "byte": true, "rune": true,
	"len": true, "cap": true, "make": true, "new": true, "append": true,
	"copy": true, "delete": true, "panic": true, "recover": true, "close": true,
	"print": true, "println": true,
}

// simpleOps are operations with low register pressure that can be heavily unrolled.
var simpleOps = map[string]bool{
	"Add": true, "Sub": true, "Mul": true, "Div": true,
	"FMA": true, "MulAdd": true, "MulSub": true,
	"Neg": true, "Abs": true, "Min": true, "Max": true,
	"And": true, "Or": true, "Xor": true, "AndNot": true, "Not": true,
	"Load": true, "LoadSlice": true, "Store": true, "StoreSlice": true, "Set": true, "Zero": true,
	// Meta-operations that don't affect complexity
	"MaxLanes": true, "NumLanes": true, "Lanes": true,
	"Vec": true, "Mask": true, // Type references
}

// complexOps are operations that use many registers (polynomial coefficients, etc.).
var complexOps = map[string]bool{
	"Exp": true, "Exp2": true, "Exp10": true,
	"Log": true, "Log2": true, "Log10": true,
	"Sin": true, "Cos": true, "SinCos": true,
	"Tanh": true, "Sinh": true, "Cosh": true,
	"Asinh": true, "Acosh": true, "Atanh": true,
	"Sigmoid": true, "Erf": true, "Pow": true,
	"Sqrt": true, "RSqrt": true,
}

// reductionOps have data dependencies that limit instruction-level parallelism.
var reductionOps = map[string]bool{
	"ReduceSum": true, "ReduceMin": true, "ReduceMax": true,
}

// analyzeLoopComplexity determines the complexity of operations in a loop body.
func analyzeLoopComplexity(hwyCalls []HwyCall) OperationComplexity {
	hasComplex := false
	hasReduction := false
	hasMedium := false

	for _, call := range hwyCalls {
		if complexOps[call.FuncName] {
			hasComplex = true
		}
		if reductionOps[call.FuncName] {
			hasReduction = true
		}
		if !simpleOps[call.FuncName] && !complexOps[call.FuncName] && !reductionOps[call.FuncName] {
			hasMedium = true
		}
	}

	// Return the highest complexity found
	if hasComplex {
		return ComplexityComplex
	}
	if hasReduction {
		return ComplexityReduction
	}
	if hasMedium {
		return ComplexityMedium
	}
	return ComplexitySimple
}

// computeUnrollFactor determines the automatic unroll factor based on operation complexity
// and target architecture. Returns 1 if unrolling should be disabled.
func computeUnrollFactor(loopInfo *LoopInfo, hwyCalls []HwyCall, target Target) int {
	if loopInfo == nil {
		return 1
	}

	// Honor explicit //hwy:unroll directive
	if loopInfo.UnrollHint > 0 {
		return loopInfo.UnrollHint
	}
	// //hwy:unroll 0 or //hwy:unroll 1 disables unrolling
	if loopInfo.UnrollHint == 0 {
		// No directive - use automatic heuristics
	} else {
		return 1 // Explicit disable
	}

	// Analyze operation complexity
	complexity := analyzeLoopComplexity(hwyCalls)

	// Base unroll factors by complexity
	var baseFactor int
	switch complexity {
	case ComplexitySimple:
		baseFactor = 4 // Simple ops can be heavily unrolled
	case ComplexityMedium:
		baseFactor = 2 // Moderate unrolling
	case ComplexityComplex:
		baseFactor = 2 // Limited by register pressure from polynomial coefficients
	case ComplexityReduction:
		baseFactor = 2 // Data dependencies limit ILP anyway
	default:
		baseFactor = 2
	}

	// Adjust for target architecture
	// AVX-512 has 32 registers vs AVX2's 16, so can be more aggressive
	switch target.Name {
	case "AVX512":
		if baseFactor < 4 && complexity != ComplexityComplex {
			baseFactor = min(baseFactor+1, 4)
		}
	case "NEON":
		// NEON has 32 V registers but narrower, keep moderate
		baseFactor = min(baseFactor, 4)
	case "Fallback":
		// No unrolling for fallback - it's scalar anyway
		return 1
	}

	return baseFactor
}

// unrollLoopWithCleanup applies loop unrolling and inserts a cleanup loop for remaining elements.
// After unrolling with factor N, the main loop processes N*lanes elements per iteration.
// A cleanup loop is inserted to process any remaining full vector chunks (< N*lanes but >= lanes),
// UNLESS the function already has an explicit tail loop after the main loop.
func unrollLoopWithCleanup(body *ast.BlockStmt, forStmt *ast.ForStmt, loopInfo *LoopInfo, unrollFactor int, lanes int) {
	if body == nil || forStmt == nil || loopInfo == nil || unrollFactor <= 1 {
		return
	}

	// Check if there's already a tail loop after the main loop (explicit tail handling).
	// If so, the cleanup loop is unnecessary since the existing tail loop handles all remaining elements.
	needsCleanupLoop := !hasExplicitTailLoop(body, forStmt, loopInfo.Iterator)

	// Clone the original loop body before unrolling (for the cleanup loop)
	var origBodyClone []ast.Stmt
	var origCond ast.Expr
	var origPost ast.Stmt

	if needsCleanupLoop {
		origBodyClone = make([]ast.Stmt, len(forStmt.Body.List))
		for i, stmt := range forStmt.Body.List {
			origBodyClone[i] = cloneStmt(stmt)
		}
		if forStmt.Cond != nil {
			origCond = cloneExpr(forStmt.Cond)
		}
		if forStmt.Post != nil {
			origPost = cloneStmt(forStmt.Post)
		}
	}

	// Check if iterator is declared in the loop's Init (e.g., "for ii := 0; ...")
	// If so, we need to hoist it to allow cleanup loop (or existing tail loop) access
	var hoistedDecl ast.Stmt
	if forStmt.Init != nil {
		if assign, ok := forStmt.Init.(*ast.AssignStmt); ok && assign.Tok == token.DEFINE {
			// Check if this declares the iterator we're tracking
			for _, lhs := range assign.Lhs {
				if ident, ok := lhs.(*ast.Ident); ok && ident.Name == loopInfo.Iterator {
					// Hoist the declaration: create "ii := 0" before the loop
					hoistedDecl = cloneStmt(forStmt.Init)
					// Remove Init from the main loop (it becomes "for ; cond; post")
					forStmt.Init = nil
					break
				}
			}
		}
	}

	// Apply unrolling to the main loop (this modifies forStmt in place)
	unrollLoop(forStmt, loopInfo, unrollFactor, lanes)

	// Find the position of the unrolled loop and insert cleanup loop (if needed) after it
	for i, stmt := range body.List {
		if stmt == forStmt {
			// Build new statement list
			newList := make([]ast.Stmt, 0, len(body.List)+2)
			newList = append(newList, body.List[:i]...)

			// Insert hoisted declaration if needed
			if hoistedDecl != nil {
				newList = append(newList, hoistedDecl)
			}

			// Insert main (unrolled) loop
			newList = append(newList, forStmt)

			// Insert cleanup loop only if function doesn't have its own tail handling
			if needsCleanupLoop {
				cleanupLoop := &ast.ForStmt{
					Cond: origCond,
					Post: origPost,
					Body: &ast.BlockStmt{
						List: origBodyClone,
					},
				}
				newList = append(newList, cleanupLoop)
			}

			// Insert remaining statements
			newList = append(newList, body.List[i+1:]...)
			body.List = newList
			return
		}
	}
}

// hasExplicitTailLoop checks if there's another for loop after the given loop
// that uses the same iterator, indicating explicit tail handling.
func hasExplicitTailLoop(body *ast.BlockStmt, mainLoop *ast.ForStmt, iterator string) bool {
	foundMain := false
	for _, stmt := range body.List {
		if stmt == mainLoop {
			foundMain = true
			continue
		}
		if foundMain {
			if fl, ok := stmt.(*ast.ForStmt); ok {
				if matchesLoopIterator(fl, iterator) {
					return true
				}
			}
		}
	}
	return false
}

// unrollLoop applies loop unrolling to a for loop, creating N copies of the body.
// It modifies the loop in place:
// - Multiplies the stride by unrollFactor
// - Replicates the body with adjusted indices (i, i+lanes, i+2*lanes, ...)
// - Renames variables to avoid redeclaration (x -> x0, x1, x2, ...)
func unrollLoop(forStmt *ast.ForStmt, loopInfo *LoopInfo, unrollFactor int, lanes int) {
	if forStmt == nil || loopInfo == nil || unrollFactor <= 1 {
		return
	}

	// Clone the original body statements
	origBody := forStmt.Body.List

	// Collect variable names declared in the loop body (need renaming for unrolled copies)
	declaredVars := collectDeclaredVars(origBody)

	// Build the unrolled body
	unrolledBody := make([]ast.Stmt, 0, len(origBody)*unrollFactor)

	for u := range unrollFactor {
		for _, stmt := range origBody {
			// Clone the statement
			cloned := cloneStmt(stmt)

			// Rename variables for unrolled iterations (x -> x0, x1, x2, ...)
			if u > 0 {
				renameVarsInStmt(cloned, declaredVars, u)
			}

			// Adjust indices for all iterations except the first
			if u > 0 {
				adjustLoopIndices(cloned, loopInfo.Iterator, u, lanes)
			}

			unrolledBody = append(unrolledBody, cloned)
		}
	}

	// Update the loop body
	forStmt.Body.List = unrolledBody

	// Update the stride: i += lanes -> i += lanes * unrollFactor
	if assignStmt, ok := forStmt.Post.(*ast.AssignStmt); ok {
		if len(assignStmt.Rhs) == 1 {
			// Check if it's already a constant (transformed by transformForStmt)
			if lit, ok := assignStmt.Rhs[0].(*ast.BasicLit); ok && lit.Kind == token.INT {
				// Multiply the stride
				oldStride, _ := strconv.Atoi(lit.Value)
				lit.Value = strconv.Itoa(oldStride * unrollFactor)
			} else {
				// Wrap in multiplication: stride * unrollFactor
				assignStmt.Rhs[0] = &ast.BinaryExpr{
					X:  assignStmt.Rhs[0],
					Op: token.MUL,
					Y:  &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(unrollFactor)},
				}
			}
		}
	}

	// Update the condition to account for unrolled stride
	// Change: i+lanes <= n -> i+lanes*unrollFactor <= n
	if binExpr, ok := forStmt.Cond.(*ast.BinaryExpr); ok {
		if innerBin, ok := binExpr.X.(*ast.BinaryExpr); ok {
			if innerBin.Op == token.ADD {
				// Handle both literal and variable lanes
				switch y := innerBin.Y.(type) {
				case *ast.BasicLit:
					if y.Kind == token.INT {
						oldLanes, _ := strconv.Atoi(y.Value)
						y.Value = strconv.Itoa(oldLanes * unrollFactor)
					}
				case *ast.Ident:
					// lanes variable - wrap in multiplication: lanes * unrollFactor
					innerBin.Y = &ast.BinaryExpr{
						X:  y,
						Op: token.MUL,
						Y:  &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(unrollFactor)},
					}
				}
			}
		}
	}
}

// collectDeclaredVars finds all variable names declared with := or var in the statements.
// It excludes the blank identifier "_" which should never be renamed.
func collectDeclaredVars(stmts []ast.Stmt) map[string]bool {
	vars := collectDeclaredNames(&ast.BlockStmt{List: stmts})
	delete(vars, "_")
	return vars
}

// renameVarsInStmt renames declared variables and their uses by appending the iteration number.
// E.g., for iteration 2: x -> x2, result -> result2
func renameVarsInStmt(stmt ast.Stmt, declaredVars map[string]bool, iteration int) {
	suffix := strconv.Itoa(iteration)

	ast.Inspect(stmt, func(n ast.Node) bool {
		if ident, ok := n.(*ast.Ident); ok {
			if declaredVars[ident.Name] {
				ident.Name = ident.Name + suffix
			}
		}
		return true
	})
}

// adjustLoopIndices adjusts array/slice indices in a statement by adding offset*lanes.
// For iteration u (0-indexed), transforms:
//   - data[i:] -> data[i+u*lanes:]
//   - Load(data[i:]) -> Load(data[i+u*lanes:])
func adjustLoopIndices(stmt ast.Stmt, iterator string, iteration int, lanes int) {
	offset := iteration * lanes

	ast.Inspect(stmt, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.SliceExpr:
			// Transform src[i:] or src[i:n] where low uses the iterator
			if node.Low != nil {
				node.Low = addOffsetToExpr(node.Low, iterator, offset)
			}
		case *ast.IndexExpr:
			// Transform src[i] where index uses the iterator
			if node.Index != nil {
				node.Index = addOffsetToExpr(node.Index, iterator, offset)
			}
		}
		return true
	})
}

// addOffsetToExpr adds an offset to an expression if it references the iterator.
// E.g., if iterator="i" and offset=8: i -> i+8, i+lanes -> i+lanes+8, i-1 -> i-1+8
func addOffsetToExpr(expr ast.Expr, iterator string, offset int) ast.Expr {
	// Check if expr directly references the iterator
	if ident, ok := expr.(*ast.Ident); ok && ident.Name == iterator {
		return &ast.BinaryExpr{
			X:  expr,
			Op: token.ADD,
			Y:  &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(offset)},
		}
	}

	// Check if expr is i+something or i-something
	if binExpr, ok := expr.(*ast.BinaryExpr); ok && (binExpr.Op == token.ADD || binExpr.Op == token.SUB) {
		if ident, ok := binExpr.X.(*ast.Ident); ok && ident.Name == iterator {
			// Transform i+N to i+N+offset, i-N to i-N+offset
			return &ast.BinaryExpr{
				X:  binExpr,
				Op: token.ADD,
				Y:  &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(offset)},
			}
		}
	}

	return expr
}

// findMainSimdLoop finds the main SIMD loop in a function body that matches the given LoopInfo.
func findMainSimdLoop(body *ast.BlockStmt, loopInfo *LoopInfo) *ast.ForStmt {
	if body == nil || loopInfo == nil {
		return nil
	}

	for _, stmt := range body.List {
		forStmt, ok := stmt.(*ast.ForStmt)
		if !ok {
			continue
		}

		// Check if this loop's iterator matches loopInfo.Iterator
		if matchesLoopIterator(forStmt, loopInfo.Iterator) {
			return forStmt
		}
	}

	return nil
}


// matchesLoopIterator checks if a for loop uses the given iterator name.
// It checks the init statement, condition, and post statement.
func matchesLoopIterator(forStmt *ast.ForStmt, iteratorName string) bool {
	// Check init statement: for ii := 0
	if forStmt.Init != nil {
		if assign, ok := forStmt.Init.(*ast.AssignStmt); ok {
			for _, lhs := range assign.Lhs {
				if ident, ok := lhs.(*ast.Ident); ok && ident.Name == iteratorName {
					return true
				}
			}
		}
	}

	// Check condition: ii < size, ii+N <= size, or (ii+N)+M <= size
	if forStmt.Cond != nil {
		if binExpr, ok := forStmt.Cond.(*ast.BinaryExpr); ok {
			// Check LHS directly (ii < size)
			if ident, ok := binExpr.X.(*ast.Ident); ok && ident.Name == iteratorName {
				return true
			}
			// Check LHS if it's a binary expression (ii+N <= size)
			if innerBin, ok := binExpr.X.(*ast.BinaryExpr); ok {
				if ident, ok := innerBin.X.(*ast.Ident); ok && ident.Name == iteratorName {
					return true
				}
				// Check deeper nesting: (ii+N)+M <= size (after transformForStmt wraps condition)
				if deeperBin, ok := innerBin.X.(*ast.BinaryExpr); ok {
					if ident, ok := deeperBin.X.(*ast.Ident); ok && ident.Name == iteratorName {
						return true
					}
				}
			}
		}
	}

	// Check post statement: ii += lanes or ii++
	if forStmt.Post != nil {
		if assignStmt, ok := forStmt.Post.(*ast.AssignStmt); ok {
			for _, lhs := range assignStmt.Lhs {
				if ident, ok := lhs.(*ast.Ident); ok && ident.Name == iteratorName {
					return true
				}
			}
		}
		if incDecStmt, ok := forStmt.Post.(*ast.IncDecStmt); ok {
			if ident, ok := incDecStmt.X.(*ast.Ident); ok && ident.Name == iteratorName {
				return true
			}
		}
	}

	return false
}

// insertTailHandling adds scalar tail handling after the vectorized loop.
func insertTailHandling(body *ast.BlockStmt, loopInfo *LoopInfo, elemType string, target Target, funcName string, params []Param, typeParams []TypeParam) {
	if body == nil || loopInfo == nil {
		return
	}

	// For fallback, no tail handling needed - callers must provide inputs >= vector width
	if target.IsFallback() {
		return
	}

	// Count SIMD loops that use the same iterator. If there are multiple SIMD loops,
	// the function has a multi-phase algorithm (e.g., Normalize: accumulate then scale)
	// and automatic tail handling would break the data dependencies between phases.
	// In such cases, the template must handle tails manually.
	simdLoopCount := 0
	for _, stmt := range body.List {
		if forStmt, ok := stmt.(*ast.ForStmt); ok {
			if matchesLoopIterator(forStmt, loopInfo.Iterator) && isSimdStyleLoop(forStmt) {
				simdLoopCount++
			}
		}
	}
	if simdLoopCount > 1 {
		// Multiple SIMD loops - don't insert automatic tail handling
		return
	}

	// Find the SIMD loop that uses loopInfo.Iterator as its iterator.
	// This ensures we don't add tail handling after unrelated loops (e.g., scalar loops).
	var loopIdx int = -1
	var mainLoop *ast.ForStmt
	for i, stmt := range body.List {
		if forStmt, ok := stmt.(*ast.ForStmt); ok {
			// Check if this loop's iterator matches loopInfo.Iterator
			if matchesLoopIterator(forStmt, loopInfo.Iterator) {
				loopIdx = i
				mainLoop = forStmt
				break
			}
		}
	}

	if mainLoop == nil || loopIdx < 0 {
		return
	}

	// Declare the iterator before the loop so it's in scope for the tail
	// Change: for ii := 0; ... to: ii := 0; for ; ...
	var initStmt ast.Stmt
	if mainLoop.Init != nil {
		initStmt = mainLoop.Init
		mainLoop.Init = nil
	}

	// Build tail handling that calls the fallback function for remaining elements
	// if ii < size {
	//     BaseSigmoid_fallback(in[ii:size], out[ii:size])
	// }
	fallbackFuncName := funcName + "_fallback"
	// Add type suffix for non-float32 types only for generic functions
	// (matches how generator.go names functions in generator.go:100-102)
	if elemType != "float32" && len(typeParams) > 0 {
		fallbackFuncName = fallbackFuncName + "_" + typeNameToSuffix(elemType)
	}

	// Build arguments for the fallback call
	// For slice parameters: param[ii:size]
	// For non-slice parameters: pass as-is
	var callArgs []ast.Expr
	for _, param := range params {
		if strings.HasPrefix(param.Type, "[]") {
			// Create param[ii:size] for slice parameters
			sliceExpr := &ast.SliceExpr{
				X:    ast.NewIdent(param.Name),
				Low:  ast.NewIdent(loopInfo.Iterator),
				High: ast.NewIdent(loopInfo.End),
			}
			callArgs = append(callArgs, sliceExpr)
		} else {
			// Pass non-slice parameters as-is
			callArgs = append(callArgs, ast.NewIdent(param.Name))
		}
	}

	// Create the fallback call: BasePoly2_fallback(x[ii:size], c0, c1, c2, result[ii:size])
	fallbackCall := &ast.CallExpr{
		Fun:  ast.NewIdent(fallbackFuncName),
		Args: callArgs,
	}

	// Wrap in if statement: if ii < size { ... }
	tailIf := &ast.IfStmt{
		Cond: &ast.BinaryExpr{
			X:  ast.NewIdent(loopInfo.Iterator),
			Op: token.LSS,
			Y:  ast.NewIdent(loopInfo.End),
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				&ast.ExprStmt{X: fallbackCall},
			},
		},
	}

	// Insert init statement, main loop, and tail handling
	// Check if the next statement is a scalar tail loop that can be replaced by fallback
	nextIdx := loopIdx + 1
	canReplaceTailLoop := false
	if nextIdx < len(body.List) {
		if isScalarTailLoop(body.List[nextIdx], loopInfo.Iterator, loopInfo.End) {
			canReplaceTailLoop = true
			nextIdx++ // Skip the scalar tail loop (it will be replaced by fallback call)
		}
	}

	newStmts := make([]ast.Stmt, 0, len(body.List)+2)
	newStmts = append(newStmts, body.List[:loopIdx]...)
	if initStmt != nil {
		newStmts = append(newStmts, initStmt)
	}
	newStmts = append(newStmts, mainLoop)
	// Only add the fallback call if we're replacing the scalar tail loop.
	// If the tail loop uses external variables (like 'scale' computed from full array),
	// we must keep the original loop which correctly uses those variables.
	if canReplaceTailLoop {
		newStmts = append(newStmts, tailIf)
	}
	newStmts = append(newStmts, body.List[nextIdx:]...)
	body.List = newStmts
}

// isScalarTailLoop checks if a statement is a scalar tail loop that should be
// replaced by the fallback call. A scalar tail loop has the form:
//
//	for ; i < n; i++ { ... }
//
// where i is the iterator and n is the end variable from the SIMD loop.
// Returns false if the loop body assigns to local variables (other than indexed
// array elements), as these indicate state that the fallback cannot handle.
func isScalarTailLoop(stmt ast.Stmt, iterator, end string) bool {
	forStmt, ok := stmt.(*ast.ForStmt)
	if !ok {
		return false
	}

	// Scalar tail loops have no Init (the iterator is already declared)
	if forStmt.Init != nil {
		return false
	}

	// Check condition: i < n
	cond, ok := forStmt.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.LSS {
		return false
	}

	// Left side should be the iterator
	leftIdent, ok := cond.X.(*ast.Ident)
	if !ok || leftIdent.Name != iterator {
		return false
	}

	// Right side should be the end variable (can be identifier like "n" or call like "len(dst)")
	if exprToString(cond.Y) != end {
		return false
	}

	// Check post: i++ (increment expression)
	post, ok := forStmt.Post.(*ast.IncDecStmt)
	if !ok || post.Tok != token.INC {
		return false
	}

	postIdent, ok := post.X.(*ast.Ident)
	if !ok || postIdent.Name != iterator {
		return false
	}

	// Check if the loop body assigns to local variables (not array elements).
	// If so, this loop has state that the fallback cannot handle correctly.
	// Example: "prev = src[i]" indicates state tracking that needs the manual loop.
	if hasLocalVariableAssignment(forStmt.Body, iterator) {
		return false
	}

	// Check if the loop body uses external variables (not just the iterator and arrays).
	// If so, those variables were computed from the full input and the fallback would
	// recalculate them incorrectly from just the tail.
	// Example: "dst[i] *= scale" uses external variable "scale" computed from full array.
	if usesExternalVariables(forStmt.Body, iterator) {
		return false
	}

	return true
}

// hasLocalVariableAssignment checks if a block contains assignments to local
// variables (identifiers) rather than just indexed array/slice elements.
// Assignments like "prev = src[i]" return true.
// Assignments like "dst[i] = x" return false (these are array element assignments).
func hasLocalVariableAssignment(body *ast.BlockStmt, iterator string) bool {
	if body == nil {
		return false
	}

	hasLocalAssign := false
	ast.Inspect(body, func(n ast.Node) bool {
		assign, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}

		for _, lhs := range assign.Lhs {
			// Check if this is an assignment to a plain identifier (not array index)
			if ident, ok := lhs.(*ast.Ident); ok {
				// Skip the iterator variable itself
				if ident.Name != iterator {
					hasLocalAssign = true
					return false
				}
			}
		}
		return true
	})

	return hasLocalAssign
}

// usesExternalVariables checks if a loop body uses variables that were defined
// outside the loop (excluding the iterator and slice/array variables used in index expressions).
// For example, "dst[i] *= scale" uses external variable "scale".
// The fallback function would recalculate such variables from just the tail, which is wrong.
func usesExternalVariables(body *ast.BlockStmt, iterator string) bool {
	if body == nil {
		return false
	}

	// Collect identifiers that are OK to use:
	// 1. Slices/arrays being indexed (function parameters)
	// 2. Identifiers that are part of selector expressions (package.Func, obj.Method)
	okIdents := make(map[*ast.Ident]bool)

	ast.Inspect(body, func(n ast.Node) bool {
		switch expr := n.(type) {
		case *ast.IndexExpr:
			// Mark the slice/array being indexed as OK
			if ident, ok := expr.X.(*ast.Ident); ok {
				okIdents[ident] = true
			}
		case *ast.SelectorExpr:
			// Mark both parts of selector expressions as OK
			// e.g., hwy.Float32ToFloat16 or dst[i].Float32()
			if ident, ok := expr.X.(*ast.Ident); ok {
				okIdents[ident] = true
			}
			okIdents[expr.Sel] = true
		case *ast.CallExpr:
			// Mark function name in direct calls as OK
			if ident, ok := expr.Fun.(*ast.Ident); ok {
				okIdents[ident] = true
			}
		}
		return true
	})

	hasExternal := false
	ast.Inspect(body, func(n ast.Node) bool {
		ident, ok := n.(*ast.Ident)
		if !ok {
			return true
		}

		// Skip if already marked as OK
		if okIdents[ident] {
			return true
		}

		name := ident.Name

		// Skip the iterator variable
		if name == iterator {
			return true
		}

		if goBuiltinIdents[name] {
			return true
		}

		// Skip blank identifier
		if name == "_" {
			return true
		}

		// This is an external variable - flag it
		hasExternal = true
		return false
	})

	return hasExternal
}

// isSimdStyleLoop checks if a for loop appears to be a SIMD-style loop (as opposed
// to a scalar tail loop). SIMD loops typically have:
// - A condition like i+lanes <= len(dst) (not i < len)
// - A stride like i += lanes (not i++)
func isSimdStyleLoop(forStmt *ast.ForStmt) bool {
	if forStmt == nil || forStmt.Cond == nil || forStmt.Post == nil {
		return false
	}

	// Check condition: should be i+lanes <= len (not i < len)
	cond, ok := forStmt.Cond.(*ast.BinaryExpr)
	if !ok {
		return false
	}

	// SIMD loop condition is typically <= (not <)
	// Or it's < with a +lanes on the left side
	if cond.Op == token.LEQ {
		return true
	}

	// Check if left side is i+lanes (binary expr with +)
	if cond.Op == token.LSS {
		if _, ok := cond.X.(*ast.BinaryExpr); ok {
			// i+lanes < len pattern
			return true
		}
	}

	// Check post: should be i += lanes (not i++)
	switch post := forStmt.Post.(type) {
	case *ast.AssignStmt:
		// i += lanes
		if post.Tok == token.ADD_ASSIGN {
			return true
		}
	case *ast.IncDecStmt:
		// i++ is NOT a SIMD loop
		return false
	}

	return false
}


// buildResultsWithTarget builds the return type list with target-specific Vec types.
func (pf *ParsedFunc) buildResultsWithTarget(elemType string, target Target, skipHalfPrec bool, typeMap map[string]string) *ast.FieldList {
	if len(pf.Returns) == 0 {
		return nil
	}

	fieldList := &ast.FieldList{
		List: make([]*ast.Field, 0, len(pf.Returns)),
	}

	for _, ret := range pf.Returns {
		retType := specializeTypeWithMap(ret.Type, pf.TypeParams, elemType, typeMap)
		// Transform hwy.Vec[T] to concrete vector types for SIMD targets
		vecElemType := extractVecElemType(retType, elemType)
		retType = specializeVecType(retType, vecElemType, target, skipHalfPrec)
		field := &ast.Field{
			Type: parseTypeExpr(retType),
		}
		if ret.Name != "" {
			field.Names = []*ast.Ident{ast.NewIdent(ret.Name)}
		}
		fieldList.List = append(fieldList.List, field)
	}

	return fieldList
}

// postProcessSIMD walks the AST and replaces NumLanes() calls with constants
// and transforms ReduceSum() calls to store+sum patterns.
func postProcessSIMD(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	defaultLanes := ctx.target.LanesFor(ctx.elemType)

	// Walk all statements and expressions, replacing as needed
	ast.Inspect(node, func(n ast.Node) bool {
		switch stmt := n.(type) {
		case *ast.IfStmt:
			// Replace comparisons like: remaining >= v.NumLanes()
			if binExpr, ok := stmt.Cond.(*ast.BinaryExpr); ok {
				replaceNumLanesInExpr(binExpr, defaultLanes, ctx)
			}
		case *ast.AssignStmt:
			// Replace: sum += v.ReduceSum() or sum += hwy.ReduceSum(v)
			// Skip for Float16/BFloat16 - hwy.Vec doesn't have StoreSlice(),
			// and hwy.ReduceSumF16/BF16 work directly.
			// Also skip if target has native ReduceSum support (e.g., NEON has v.ReduceSum() method)
			hasNativeReduceSum := false
			if opInfo, ok := ctx.target.OpMap["ReduceSum"]; ok {
				// Native if it's a method with no package prefix (direct method on vector type)
				hasNativeReduceSum = opInfo.Package == "" && opInfo.IsMethod
			}
			if !ctx.isHalfPrec && !hasNativeReduceSum {
				for i, rhs := range stmt.Rhs {
					if call, ok := rhs.(*ast.CallExpr); ok {
						if isReduceSumCall(call) {
							// Transform to store + sum pattern
							stmt.Rhs[i] = createReduceSumExpr(call, defaultLanes, ctx.elemType)
						}
					}
				}
			}
		case *ast.ExprStmt:
			// Handle standalone expressions if needed
		}
		return true
	})
}

// replaceNumLanesInExpr replaces v.NumLanes() with a constant in a binary expression.
// It uses the context to look up the actual element type of vector variables.
func replaceNumLanesInExpr(binExpr *ast.BinaryExpr, defaultLanes int, ctx *transformContext) {
	// Check RHS
	if call, ok := binExpr.Y.(*ast.CallExpr); ok {
		if lanes := getLanesForNumLanesCall(call, defaultLanes, ctx); lanes > 0 {
			binExpr.Y = &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.Itoa(lanes),
			}
		}
	}
	// Check LHS (less common but possible)
	if call, ok := binExpr.X.(*ast.CallExpr); ok {
		if lanes := getLanesForNumLanesCall(call, defaultLanes, ctx); lanes > 0 {
			binExpr.X = &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.Itoa(lanes),
			}
		}
	}
}

// getLanesForNumLanesCall returns the lane count for a NumLanes() call,
// taking into account the actual element type of the vector variable.
// Returns 0 if the call is not a NumLanes call.
func getLanesForNumLanesCall(call *ast.CallExpr, defaultLanes int, ctx *transformContext) int {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return 0
	}
	if sel.Sel.Name != "NumLanes" && sel.Sel.Name != "NumElements" {
		return 0
	}
	// Try to look up the element type of the vector variable
	if varIdent, ok := sel.X.(*ast.Ident); ok {
		if varElemType, ok := ctx.varVecElemType[varIdent.Name]; ok {
			return ctx.target.LanesFor(varElemType)
		}
	}
	return defaultLanes
}

// isReduceSumCall checks if a call expression is v.ReduceSum(), hwy.ReduceSum(v),
// or the F16/BF16 variants (ReduceSumF16, ReduceSumBF16).
func isReduceSumCall(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	name := sel.Sel.Name
	return name == "ReduceSum" || name == "ReduceSumF16" || name == "ReduceSumBF16" ||
		name == "ReduceMin" || name == "ReduceMax"
}

// createReduceSumExpr creates an expression that stores the vector and sums elements.
// For now, we generate a function call that we'll define in a helper.
// Actually, archsimd vectors don't have a built-in ReduceSum, so we need to
// generate inline code that stores to temp and sums.
// Since we can't inject statements here, we'll generate a compound expression.
func createReduceSumExpr(call *ast.CallExpr, lanes int, elemType string) ast.Expr {
	// Get the vector argument
	var vecExpr ast.Expr
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if ident, ok := sel.X.(*ast.Ident); ok {
			// Check if it's a package name (hwy, asm) or a vector variable
			// Package names are lowercase and known; vectors are variables
			if ident.Name != "hwy" && ident.Name != "asm" && ident.Name != "archsimd" {
				// It's v.ReduceSum() - the receiver is the vector
				vecExpr = sel.X
			}
		}
	}
	if vecExpr == nil && len(call.Args) > 0 {
		// It's hwy.ReduceSum(v) or hwy.ReduceSumF16(v) - first arg is the vector
		vecExpr = call.Args[0]
	}
	if vecExpr == nil {
		return call // Can't transform, leave as-is
	}

	// Generate a function call to a helper we'll need to add
	// For now, generate inline reduction: func() T { var t [N]T; v.StoreSlice(t[:]); return t[0]+t[1]+... }()
	// This is verbose but works without injecting statements

	// Build: t[0] + t[1] + ... + t[lanes-1]
	var sumExpr ast.Expr
	for i := range lanes {
		indexExpr := &ast.IndexExpr{
			X: ast.NewIdent("_simd_temp"),
			Index: &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.Itoa(i),
			},
		}
		if sumExpr == nil {
			sumExpr = indexExpr
		} else {
			sumExpr = &ast.BinaryExpr{
				X:  sumExpr,
				Op: token.ADD,
				Y:  indexExpr,
			}
		}
	}

	// Build the function literal:
	// func() elemType {
	//     var _simd_temp [lanes]elemType
	//     vec.StoreSlice(_simd_temp[:])
	//     return t[0] + t[1] + ...
	// }()
	funcLit := &ast.FuncLit{
		Type: &ast.FuncType{
			Results: &ast.FieldList{
				List: []*ast.Field{
					{Type: ast.NewIdent(elemType)},
				},
			},
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				// var _simd_temp [lanes]elemType
				&ast.DeclStmt{
					Decl: &ast.GenDecl{
						Tok: token.VAR,
						Specs: []ast.Spec{
							&ast.ValueSpec{
								Names: []*ast.Ident{ast.NewIdent("_simd_temp")},
								Type: &ast.ArrayType{
									Len: &ast.BasicLit{
										Kind:  token.INT,
										Value: strconv.Itoa(lanes),
									},
									Elt: ast.NewIdent(elemType),
								},
							},
						},
					},
				},
				// vec.StoreSlice(_simd_temp[:])
				&ast.ExprStmt{
					X: &ast.CallExpr{
						Fun: &ast.SelectorExpr{
							X:   vecExpr,
							Sel: ast.NewIdent("StoreSlice"),
						},
						Args: []ast.Expr{
							&ast.SliceExpr{
								X: ast.NewIdent("_simd_temp"),
							},
						},
					},
				},
				// return sum expression
				&ast.ReturnStmt{
					Results: []ast.Expr{sumExpr},
				},
			},
		},
	}

	// Call the function literal immediately
	return &ast.CallExpr{
		Fun: funcLit,
	}
}

// filterConditionalBlocks filters statements based on //hwy:if, //hwy:else, //hwy:endif directives.
// It returns a new BlockStmt with only the statements that match the current target and element type.
// The original AST is not modified.
func filterConditionalBlocks(body *ast.BlockStmt, blocks []ConditionalBlock, fset *token.FileSet, targetName, elemType string) *ast.BlockStmt {
	if body == nil || len(blocks) == 0 {
		return body
	}

	// Create a new block with filtered statements
	newBody := &ast.BlockStmt{
		Lbrace: body.Lbrace,
		Rbrace: body.Rbrace,
	}

	for _, stmt := range body.List {
		// Get the line number of this statement
		stmtLine := fset.Position(stmt.Pos()).Line

		// Check if this statement is within any conditional block
		included := true
		for _, block := range blocks {
			if stmtLine > block.StartLine && stmtLine < block.EndLine {
				// Statement is within this conditional block
				conditionMatches := block.ParsedCondition.Evaluate(targetName, elemType)

				if block.ElseLine > 0 {
					// Block has an else clause
					if stmtLine < block.ElseLine {
						// Statement is in the "if" part
						included = conditionMatches
					} else {
						// Statement is in the "else" part
						included = !conditionMatches
					}
				} else {
					// No else clause - include only if condition matches
					included = conditionMatches
				}
				break // Found the innermost containing block
			}
		}

		if included {
			// Recursively filter nested blocks (e.g., for statements, if statements)
			filteredStmt := filterNestedConditionalBlocks(stmt, blocks, fset, targetName, elemType)
			newBody.List = append(newBody.List, filteredStmt)
		}
	}

	return newBody
}

// filterNestedConditionalBlocks recursively filters conditional blocks within nested statements.
func filterNestedConditionalBlocks(stmt ast.Stmt, blocks []ConditionalBlock, fset *token.FileSet, targetName, elemType string) ast.Stmt {
	switch s := stmt.(type) {
	case *ast.BlockStmt:
		return filterConditionalBlocks(s, blocks, fset, targetName, elemType)
	case *ast.IfStmt:
		newIf := *s // shallow copy
		if s.Body != nil {
			newIf.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		if s.Else != nil {
			newIf.Else = filterNestedConditionalBlocks(s.Else, blocks, fset, targetName, elemType)
		}
		return &newIf
	case *ast.ForStmt:
		newFor := *s // shallow copy
		if s.Body != nil {
			newFor.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		return &newFor
	case *ast.RangeStmt:
		newRange := *s // shallow copy
		if s.Body != nil {
			newRange.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		return &newRange
	case *ast.SwitchStmt:
		newSwitch := *s // shallow copy
		if s.Body != nil {
			newSwitch.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		return &newSwitch
	case *ast.TypeSwitchStmt:
		// TypeSwitchStmt is resolved later by resolveTypeSwitches once the
		// concrete elemType is known. Just recurse into the body here.
		newSwitch := *s // shallow copy
		if s.Body != nil {
			newSwitch.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		return &newSwitch
	case *ast.SelectStmt:
		newSelect := *s // shallow copy
		if s.Body != nil {
			newSelect.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		return &newSelect
	default:
		return stmt
	}
}

// resolveTypeSpecificConst resolves type-specific constant references.
// It supports two patterns:
//
// Pattern 1 (base name): "expC0" -> "expC0_f32" or "expC0_f64"
//   - Looks up base name in typeSpecificConsts map
//   - Resolves to variant matching target element type
//
// Pattern 2 (suffix swap): "expC0_f32" -> "expC0_f64"
//   - Detects existing type suffix in the name
//   - Swaps to suffix matching target element type
//   - This allows base files to be compilable while hwygen adjusts for other types
func resolveTypeSpecificConst(name string, ctx *transformContext) string {
	targetSuffix := GetTypeSuffix(ctx.elemType)

	// Pattern 1: Check if this is a base name with type-specific variants
	if ctx.typeSpecificConsts != nil {
		if tsc, ok := ctx.typeSpecificConsts[name]; ok {
			if resolved, exists := tsc.Variants[targetSuffix]; exists {
				return resolved
			}
			// Fallback: if no exact match, try f32 for Float16/BFloat16 (compute type)
			if targetSuffix == "f16" || targetSuffix == "bf16" {
				if resolved, exists := tsc.Variants["f32"]; exists {
					return resolved
				}
			}
		}
	}

	// Pattern 2: Check if name already has a type suffix that needs swapping
	for _, suffix := range typeSuffixes {
		if before, ok := strings.CutSuffix(name, suffix); ok {
			// Extract base name and swap suffix
			baseName := before
			newSuffix := "_" + targetSuffix

			// Only swap if target suffix is different
			if suffix != newSuffix {
				return baseName + newSuffix
			}
			return name // Same suffix, no change needed
		}
	}

	return name
}

// transformIdentifiers walks the AST and resolves type-specific constant references
// and type parameter substitutions.
// This handles both Pattern 1 (base name lookup) and Pattern 2 (suffix swapping).
func transformIdentifiers(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		switch expr := n.(type) {
		case *ast.Ident:
			// First check if it's a type parameter that should be replaced
			for _, tp := range ctx.typeParams {
				if expr.Name == tp.Name {
					expr.Name = ctx.elemType
					return true
				}
			}
			// Otherwise check if it's a constant reference
			resolved := resolveTypeSpecificConst(expr.Name, ctx)
			if resolved != expr.Name {
				expr.Name = resolved
			}
		case *ast.SelectorExpr:
			// Rename math.X to stdmath.X to avoid package name conflict
			// since generated files are in the math package but need stdlib math.
			// Only rename if "math" actually refers to the stdlib "math" import,
			// not a local variable or a different package aliased as "math".
			if ident, ok := expr.X.(*ast.Ident); ok && ident.Name == "math" {
				if importPath, isImport := ctx.imports[ident.Name]; isImport && importPath == "math" {
					ident.Name = "stdmath"
				}
			}
		}
		return true
	})
}

// convertStackArrayUsages converts stack array variable usages to slice expressions.
// For example, if buf is a stack array, convert:
//   - copy(buf, ...) -> copy(buf[:], ...)
//   - archsimd.LoadFloat32x8Slice(buf) -> archsimd.LoadFloat32x8Slice(buf[:])
//   - v.StoreSlice(buf) -> v.StoreSlice(buf[:])
func convertStackArrayUsages(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}

		// Check each argument
		for i, arg := range call.Args {
			// Skip if it's already a slice expression
			if _, ok := arg.(*ast.SliceExpr); ok {
				continue
			}

			// Check if the argument is a stack array variable
			if ident, ok := arg.(*ast.Ident); ok {
				if ctx.stackArrayVars[ident.Name] {
					// Replace buf with buf[:]
					call.Args[i] = &ast.SliceExpr{
						X: ident,
					}
				}
			}
		}

		return true
	})
}

// transformFuncRefArgs transforms function references passed as arguments.
// For example: BaseApply(in, out, math.BaseExpVec)
// The math.BaseExpVec should become math.BaseExpVec_avx2 for SIMD targets,
// or math.BaseExpVec_fallback for fallback targets.
func transformFuncRefArgs(call *ast.CallExpr, ctx *transformContext) {
	for i, arg := range call.Args {
		// Handle package.BaseFuncName (SelectorExpr)
		if sel, ok := arg.(*ast.SelectorExpr); ok {
			if ident, ok := sel.X.(*ast.Ident); ok {
				// Check if it's a contrib package with a Base* function
				if ctx.isContribPackage(ident.Name) && strings.HasPrefix(sel.Sel.Name, "Base") {
					// Transform math.BaseExpVec to math.BaseExpVec_avx2
					sel.Sel.Name = sel.Sel.Name + ctx.target.Suffix() + getHwygenTypeSuffix(ctx.elemType)
				}
			}
		}

		// Handle package.BaseFuncName[T] (IndexExpr wrapping SelectorExpr)
		if indexExpr, ok := arg.(*ast.IndexExpr); ok {
			if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
				if ident, ok := sel.X.(*ast.Ident); ok {
					if ctx.isContribPackage(ident.Name) && strings.HasPrefix(sel.Sel.Name, "Base") {
						// Replace the IndexExpr with just the SelectorExpr (strip type param)
						sel.Sel.Name = sel.Sel.Name + ctx.target.Suffix() + getHwygenTypeSuffix(ctx.elemType)
						call.Args[i] = sel
					}
				}
			}
		}

		// Handle local BaseFuncName[T] (IndexExpr wrapping Ident)
		if indexExpr, ok := arg.(*ast.IndexExpr); ok {
			if ident, ok := indexExpr.X.(*ast.Ident); ok {
				if strings.HasPrefix(ident.Name, "Base") {
					// Replace the IndexExpr with just the Ident
					call.Args[i] = ast.NewIdent(ident.Name + ctx.target.Suffix() + getHwygenTypeSuffix(ctx.elemType))
				}
			}
		}
	}
}

// hasPredicateParam returns true if the function has a predicate-type parameter
// (i.e., a type parameter with a non-Lanes constraint like Predicate[T]).
func hasPredicateParam(pf *ParsedFunc) bool {
	_, interfaceTypeParams := classifyTypeParams(pf.TypeParams)
	return len(interfaceTypeParams) > 0
}

// generateScalarPredicateBody generates a scalar loop body for predicate functions
// in fallback mode. Returns nil if this function doesn't need scalar generation.
func generateScalarPredicateBody(pf *ParsedFunc, elemType string) *ast.BlockStmt {
	// Map function names to their scalar implementations
	switch pf.Name {
	case "BaseAll":
		return generateScalarAll(pf)
	case "BaseAny":
		return generateScalarAny(pf)
	case "BaseFindIf":
		return generateScalarFindIf(pf)
	case "BaseCountIf":
		return generateScalarCountIf(pf)
	default:
		return nil
	}
}

// generateScalarAll generates: for _, v := range slice { if !pred.Test(v) { return false } } return true
func generateScalarAll(pf *ParsedFunc) *ast.BlockStmt {
	sliceParam := pf.Params[0].Name
	predParam := pf.Params[1].Name

	return &ast.BlockStmt{
		List: []ast.Stmt{
			// for _, v := range slice { if !pred.Test(v) { return false } }
			&ast.RangeStmt{
				Key:   ast.NewIdent("_"),
				Value: ast.NewIdent("v"),
				Tok:   token.DEFINE,
				X:     ast.NewIdent(sliceParam),
				Body: &ast.BlockStmt{
					List: []ast.Stmt{
						&ast.IfStmt{
							Cond: &ast.UnaryExpr{
								Op: token.NOT,
								X: &ast.CallExpr{
									Fun: &ast.SelectorExpr{
										X:   ast.NewIdent(predParam),
										Sel: ast.NewIdent("Test"),
									},
									Args: []ast.Expr{ast.NewIdent("v")},
								},
							},
							Body: &ast.BlockStmt{
								List: []ast.Stmt{
									&ast.ReturnStmt{
										Results: []ast.Expr{ast.NewIdent("false")},
									},
								},
							},
						},
					},
				},
			},
			// return true
			&ast.ReturnStmt{
				Results: []ast.Expr{ast.NewIdent("true")},
			},
		},
	}
}

// generateScalarAny generates: for _, v := range slice { if pred.Test(v) { return true } } return false
func generateScalarAny(pf *ParsedFunc) *ast.BlockStmt {
	sliceParam := pf.Params[0].Name
	predParam := pf.Params[1].Name

	return &ast.BlockStmt{
		List: []ast.Stmt{
			&ast.RangeStmt{
				Key:   ast.NewIdent("_"),
				Value: ast.NewIdent("v"),
				Tok:   token.DEFINE,
				X:     ast.NewIdent(sliceParam),
				Body: &ast.BlockStmt{
					List: []ast.Stmt{
						&ast.IfStmt{
							Cond: &ast.CallExpr{
								Fun: &ast.SelectorExpr{
									X:   ast.NewIdent(predParam),
									Sel: ast.NewIdent("Test"),
								},
								Args: []ast.Expr{ast.NewIdent("v")},
							},
							Body: &ast.BlockStmt{
								List: []ast.Stmt{
									&ast.ReturnStmt{
										Results: []ast.Expr{ast.NewIdent("true")},
									},
								},
							},
						},
					},
				},
			},
			&ast.ReturnStmt{
				Results: []ast.Expr{ast.NewIdent("false")},
			},
		},
	}
}

// generateScalarFindIf generates: for i, v := range slice { if pred.Test(v) { return i } } return -1
func generateScalarFindIf(pf *ParsedFunc) *ast.BlockStmt {
	sliceParam := pf.Params[0].Name
	predParam := pf.Params[1].Name

	return &ast.BlockStmt{
		List: []ast.Stmt{
			&ast.RangeStmt{
				Key:   ast.NewIdent("i"),
				Value: ast.NewIdent("v"),
				Tok:   token.DEFINE,
				X:     ast.NewIdent(sliceParam),
				Body: &ast.BlockStmt{
					List: []ast.Stmt{
						&ast.IfStmt{
							Cond: &ast.CallExpr{
								Fun: &ast.SelectorExpr{
									X:   ast.NewIdent(predParam),
									Sel: ast.NewIdent("Test"),
								},
								Args: []ast.Expr{ast.NewIdent("v")},
							},
							Body: &ast.BlockStmt{
								List: []ast.Stmt{
									&ast.ReturnStmt{
										Results: []ast.Expr{ast.NewIdent("i")},
									},
								},
							},
						},
					},
				},
			},
			&ast.ReturnStmt{
				Results: []ast.Expr{
					&ast.UnaryExpr{Op: token.SUB, X: &ast.BasicLit{Kind: token.INT, Value: "1"}},
				},
			},
		},
	}
}

// generateScalarCountIf generates: count := 0; for _, v := range slice { if pred.Test(v) { count++ } } return count
func generateScalarCountIf(pf *ParsedFunc) *ast.BlockStmt {
	sliceParam := pf.Params[0].Name
	predParam := pf.Params[1].Name

	return &ast.BlockStmt{
		List: []ast.Stmt{
			// count := 0
			&ast.AssignStmt{
				Lhs: []ast.Expr{ast.NewIdent("count")},
				Tok: token.DEFINE,
				Rhs: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "0"}},
			},
			// for _, v := range slice { if pred.Test(v) { count++ } }
			&ast.RangeStmt{
				Key:   ast.NewIdent("_"),
				Value: ast.NewIdent("v"),
				Tok:   token.DEFINE,
				X:     ast.NewIdent(sliceParam),
				Body: &ast.BlockStmt{
					List: []ast.Stmt{
						&ast.IfStmt{
							Cond: &ast.CallExpr{
								Fun: &ast.SelectorExpr{
									X:   ast.NewIdent(predParam),
									Sel: ast.NewIdent("Test"),
								},
								Args: []ast.Expr{ast.NewIdent("v")},
							},
							Body: &ast.BlockStmt{
								List: []ast.Stmt{
									&ast.IncDecStmt{
										X:   ast.NewIdent("count"),
										Tok: token.INC,
									},
								},
							},
						},
					},
				},
			},
			// return count
			&ast.ReturnStmt{
				Results: []ast.Expr{ast.NewIdent("count")},
			},
		},
	}
}

// inlineHelperCalls recursively inlines local helper function calls in a block.
// Local helpers are non-Base* functions defined in the same file that use hwy operations.
// This ensures the entire code path gets specialized for each target architecture.
func inlineHelperCalls(block *ast.BlockStmt, ctx *transformContext) {
	if block == nil || ctx.allFuncs == nil {
		return
	}

	// Process statements in reverse order so we can safely replace them
	for i := 0; i < len(block.List); i++ {
		stmt := block.List[i]

		// Check for expression statement that is a helper call
		exprStmt, ok := stmt.(*ast.ExprStmt)
		if !ok {
			// Recursively process nested blocks
			inlineHelperCallsInStmt(stmt, ctx)
			continue
		}

		// Check if this is a call expression
		callExpr, ok := exprStmt.X.(*ast.CallExpr)
		if !ok {
			continue
		}

		// Get the function name being called
		var funcName string
		switch fun := callExpr.Fun.(type) {
		case *ast.Ident:
			funcName = fun.Name
		case *ast.IndexExpr:
			// Generic call like func[T](...)
			if ident, ok := fun.X.(*ast.Ident); ok {
				funcName = ident.Name
			}
		}

		if funcName == "" {
			continue
		}

		// Skip Base* functions - they're handled separately with target suffix
		if hasBasePrefix(funcName) {
			continue
		}

		// Check if this is a local helper we can inline
		helper, exists := ctx.allFuncs[funcName]
		if !exists {
			continue
		}

		// Skip if helper has no hwy operations (pure scalar helper)
		if len(helper.HwyCalls) == 0 && !hasHwyLanesConstraint(helper.TypeParams) {
			continue
		}

		// Inline the helper
		inlinedStmts := inlineHelper(helper, callExpr, ctx)
		if inlinedStmts == nil {
			continue
		}

		// Replace the call statement with the inlined statements
		// Wrap in a BlockStmt to keep variable scope contained
		block.List[i] = &ast.BlockStmt{List: inlinedStmts}
	}
}

// inlineHelperCallsInStmt recursively processes statements to find nested helper calls.
func inlineHelperCallsInStmt(stmt ast.Stmt, ctx *transformContext) {
	if stmt == nil {
		return
	}

	switch s := stmt.(type) {
	case *ast.BlockStmt:
		inlineHelperCalls(s, ctx)
	case *ast.ForStmt:
		inlineHelperCalls(s.Body, ctx)
	case *ast.IfStmt:
		inlineHelperCalls(s.Body, ctx)
		if s.Else != nil {
			if elseBlock, ok := s.Else.(*ast.BlockStmt); ok {
				inlineHelperCalls(elseBlock, ctx)
			} else if elseIf, ok := s.Else.(*ast.IfStmt); ok {
				inlineHelperCallsInStmt(elseIf, ctx)
			}
		}
	case *ast.RangeStmt:
		inlineHelperCalls(s.Body, ctx)
	case *ast.SwitchStmt:
		inlineHelperCalls(s.Body, ctx)
	}
}

// inlineHelper transforms a helper function and returns its body statements with
// parameters substituted with actual arguments.
func inlineHelper(helper *ParsedFunc, call *ast.CallExpr, ctx *transformContext) []ast.Stmt {
	if helper.Body == nil || len(helper.Body.List) == 0 {
		return nil
	}

	// Clone the helper's body to avoid modifying the original
	clonedBody := cloneBlockStmt(helper.Body)

	// Build parameter -> argument mapping
	paramMap := make(map[string]ast.Expr)
	for i, param := range helper.Params {
		if i < len(call.Args) {
			paramMap[param.Name] = call.Args[i]
		}
	}

	// Create a unique suffix for this inline site to avoid variable conflicts
	ctx.inlineCounter++
	suffix := fmt.Sprintf("_%d", ctx.inlineCounter)

	// Collect local variables defined in the helper to rename them
	localVars := make(map[string]bool)
	collectLocalVariablesFromBlock(clonedBody, localVars)

	// Substitute parameters and rename local variables
	substituteAndRename(clonedBody, paramMap, localVars, suffix)

	// Now transform the cloned body for the current target/elemType
	helperCtx := ctx.clone()
	helperCtx.typeParams = helper.TypeParams
	helperCtx.typeMap = nil
	helperCtx.loopInfo = helper.LoopInfo

	// Copy relevant tracking from parent context
	maps.Copy(helperCtx.halfPrecisionSlices, ctx.halfPrecisionSlices)

	// Transform the helper body - same transformations as the main function
	transformIdentifiers(clonedBody, helperCtx)
	transformNode(clonedBody, helperCtx)

	// Recursively inline any nested helper calls
	inlineHelperCalls(clonedBody, helperCtx)

	// Update parent context's inline counter
	ctx.inlineCounter = helperCtx.inlineCounter

	return clonedBody.List
}

// collectDeclaredNames walks an AST and returns all variable names introduced by
// := assignments, var declarations, for-range, and for-init statements.
func collectDeclaredNames(node ast.Node) map[string]bool {
	names := make(map[string]bool)
	ast.Inspect(node, func(n ast.Node) bool {
		switch stmt := n.(type) {
		case *ast.AssignStmt:
			if stmt.Tok == token.DEFINE {
				for _, lhs := range stmt.Lhs {
					if ident, ok := lhs.(*ast.Ident); ok {
						names[ident.Name] = true
					}
				}
			}
		case *ast.DeclStmt:
			if genDecl, ok := stmt.Decl.(*ast.GenDecl); ok && genDecl.Tok == token.VAR {
				for _, spec := range genDecl.Specs {
					if vs, ok := spec.(*ast.ValueSpec); ok {
						for _, name := range vs.Names {
							names[name.Name] = true
						}
					}
				}
			}
		case *ast.RangeStmt:
			if stmt.Tok == token.DEFINE {
				if ident, ok := stmt.Key.(*ast.Ident); ok {
					names[ident.Name] = true
				}
				if stmt.Value != nil {
					if ident, ok := stmt.Value.(*ast.Ident); ok {
						names[ident.Name] = true
					}
				}
			}
		case *ast.ForStmt:
			if stmt.Init != nil {
				if assign, ok := stmt.Init.(*ast.AssignStmt); ok && assign.Tok == token.DEFINE {
					for _, lhs := range assign.Lhs {
						if ident, ok := lhs.(*ast.Ident); ok {
							names[ident.Name] = true
						}
					}
				}
			}
		}
		return true
	})
	return names
}

// collectLocalVariablesFromBlock collects all variable names defined in a block.
func collectLocalVariablesFromBlock(block *ast.BlockStmt, vars map[string]bool) {
	if block == nil {
		return
	}
	maps.Copy(vars, collectDeclaredNames(block))
}

// substituteAndRename walks the AST and:
// 1. Renames local variables with a unique suffix to avoid conflicts
// 2. Replaces parameter references with actual argument expressions
func substituteAndRename(block *ast.BlockStmt, paramMap map[string]ast.Expr, localVars map[string]bool, suffix string) {
	// First pass: rename local variables
	ast.Inspect(block, func(n ast.Node) bool {
		if ident, ok := n.(*ast.Ident); ok {
			// Check if this is a local variable - rename with suffix
			if localVars[ident.Name] {
				ident.Name = ident.Name + suffix
			}
		}
		return true
	})

	// Second pass: perform parameter substitution
	substituteParams(block, paramMap)
}

// substituteParams replaces parameter identifiers with their argument expressions.
// It uses a post-order traversal approach to avoid visiting newly-inserted nodes,
// which could cause infinite expansion if replacement expressions contain identifiers
// that match parameter names.
func substituteParams(node ast.Node, paramMap map[string]ast.Expr) {
	substituteParamsPostOrder(node, paramMap)
}

// substituteParamsPostOrder does a depth-first post-order traversal,
// processing children before parents to avoid re-visiting modified nodes.
func substituteParamsPostOrder(node ast.Node, paramMap map[string]ast.Expr) {
	if node == nil {
		return
	}

	// First, recursively process all children
	switch n := node.(type) {
	case *ast.BlockStmt:
		for _, stmt := range n.List {
			substituteParamsPostOrder(stmt, paramMap)
		}
	case *ast.ExprStmt:
		substituteParamsPostOrder(n.X, paramMap)
	case *ast.AssignStmt:
		for _, expr := range n.Lhs {
			substituteParamsPostOrder(expr, paramMap)
		}
		for _, expr := range n.Rhs {
			substituteParamsPostOrder(expr, paramMap)
		}
	case *ast.DeclStmt:
		substituteParamsPostOrder(n.Decl, paramMap)
	case *ast.GenDecl:
		for _, spec := range n.Specs {
			substituteParamsPostOrder(spec, paramMap)
		}
	case *ast.ValueSpec:
		for _, val := range n.Values {
			substituteParamsPostOrder(val, paramMap)
		}
	case *ast.IfStmt:
		substituteParamsPostOrder(n.Init, paramMap)
		substituteParamsPostOrder(n.Cond, paramMap)
		substituteParamsPostOrder(n.Body, paramMap)
		substituteParamsPostOrder(n.Else, paramMap)
	case *ast.ForStmt:
		substituteParamsPostOrder(n.Init, paramMap)
		substituteParamsPostOrder(n.Cond, paramMap)
		substituteParamsPostOrder(n.Post, paramMap)
		substituteParamsPostOrder(n.Body, paramMap)
	case *ast.RangeStmt:
		substituteParamsPostOrder(n.Key, paramMap)
		substituteParamsPostOrder(n.Value, paramMap)
		substituteParamsPostOrder(n.X, paramMap)
		substituteParamsPostOrder(n.Body, paramMap)
	case *ast.ReturnStmt:
		for _, expr := range n.Results {
			substituteParamsPostOrder(expr, paramMap)
		}
	case *ast.IncDecStmt:
		substituteParamsPostOrder(n.X, paramMap)
	case *ast.SwitchStmt:
		substituteParamsPostOrder(n.Init, paramMap)
		substituteParamsPostOrder(n.Tag, paramMap)
		substituteParamsPostOrder(n.Body, paramMap)
	case *ast.TypeSwitchStmt:
		substituteParamsPostOrder(n.Init, paramMap)
		substituteParamsPostOrder(n.Assign, paramMap)
		substituteParamsPostOrder(n.Body, paramMap)
	case *ast.CaseClause:
		for _, expr := range n.List {
			substituteParamsPostOrder(expr, paramMap)
		}
		for _, stmt := range n.Body {
			substituteParamsPostOrder(stmt, paramMap)
		}
	case *ast.BranchStmt:
		// nothing to recurse into
	case *ast.CallExpr:
		substituteParamsPostOrder(n.Fun, paramMap)
		for _, arg := range n.Args {
			substituteParamsPostOrder(arg, paramMap)
		}
	case *ast.BinaryExpr:
		substituteParamsPostOrder(n.X, paramMap)
		substituteParamsPostOrder(n.Y, paramMap)
	case *ast.UnaryExpr:
		substituteParamsPostOrder(n.X, paramMap)
	case *ast.IndexExpr:
		substituteParamsPostOrder(n.X, paramMap)
		substituteParamsPostOrder(n.Index, paramMap)
	case *ast.SliceExpr:
		substituteParamsPostOrder(n.X, paramMap)
		substituteParamsPostOrder(n.Low, paramMap)
		substituteParamsPostOrder(n.High, paramMap)
		substituteParamsPostOrder(n.Max, paramMap)
	case *ast.SelectorExpr:
		substituteParamsPostOrder(n.X, paramMap)
	case *ast.ParenExpr:
		substituteParamsPostOrder(n.X, paramMap)
	case *ast.StarExpr:
		substituteParamsPostOrder(n.X, paramMap)
	case *ast.CompositeLit:
		for _, elt := range n.Elts {
			substituteParamsPostOrder(elt, paramMap)
		}
	case *ast.KeyValueExpr:
		substituteParamsPostOrder(n.Key, paramMap)
		substituteParamsPostOrder(n.Value, paramMap)
	case *ast.TypeAssertExpr:
		substituteParamsPostOrder(n.X, paramMap)
	case *ast.Ident, *ast.BasicLit:
		// leaf nodes, nothing to recurse into
	}

	// Now, perform substitutions at this node level (post-order)
	switch parent := node.(type) {
	case *ast.CallExpr:
		for i, arg := range parent.Args {
			if ident, ok := arg.(*ast.Ident); ok {
				if replacement, isParam := paramMap[ident.Name]; isParam {
					parent.Args[i] = cloneExpr(replacement)
				}
			}
		}
	case *ast.BinaryExpr:
		if ident, ok := parent.X.(*ast.Ident); ok {
			if replacement, isParam := paramMap[ident.Name]; isParam {
				parent.X = cloneExpr(replacement)
			}
		}
		if ident, ok := parent.Y.(*ast.Ident); ok {
			if replacement, isParam := paramMap[ident.Name]; isParam {
				parent.Y = cloneExpr(replacement)
			}
		}
	case *ast.IndexExpr:
		if ident, ok := parent.X.(*ast.Ident); ok {
			if replacement, isParam := paramMap[ident.Name]; isParam {
				parent.X = cloneExpr(replacement)
			}
		}
		if ident, ok := parent.Index.(*ast.Ident); ok {
			if replacement, isParam := paramMap[ident.Name]; isParam {
				parent.Index = cloneExpr(replacement)
			}
		}
	case *ast.SliceExpr:
		if ident, ok := parent.X.(*ast.Ident); ok {
			if replacement, isParam := paramMap[ident.Name]; isParam {
				parent.X = cloneExpr(replacement)
			}
		}
		if ident, ok := parent.Low.(*ast.Ident); ok {
			if replacement, isParam := paramMap[ident.Name]; isParam {
				parent.Low = cloneExpr(replacement)
			}
		}
		if ident, ok := parent.High.(*ast.Ident); ok {
			if replacement, isParam := paramMap[ident.Name]; isParam {
				parent.High = cloneExpr(replacement)
			}
		}
	case *ast.UnaryExpr:
		if ident, ok := parent.X.(*ast.Ident); ok {
			if replacement, isParam := paramMap[ident.Name]; isParam {
				parent.X = cloneExpr(replacement)
			}
		}
	case *ast.StarExpr:
		// Handle pointer dereference: *bitPos where bitPos is a pointer parameter
		if ident, ok := parent.X.(*ast.Ident); ok {
			if replacement, isParam := paramMap[ident.Name]; isParam {
				parent.X = cloneExpr(replacement)
			}
		}
	case *ast.AssignStmt:
		for i, rhs := range parent.Rhs {
			if ident, ok := rhs.(*ast.Ident); ok {
				if replacement, isParam := paramMap[ident.Name]; isParam {
					parent.Rhs[i] = cloneExpr(replacement)
				}
			}
		}
	case *ast.ReturnStmt:
		for i, result := range parent.Results {
			if ident, ok := result.(*ast.Ident); ok {
				if replacement, isParam := paramMap[ident.Name]; isParam {
					parent.Results[i] = cloneExpr(replacement)
				}
			}
		}
	}
}

// containsTypeSwitchOrAssert reports whether the block contains any
// TypeSwitchStmt or TypeAssertExpr that would need resolution.
func containsTypeSwitchOrAssert(block *ast.BlockStmt) bool {
	if block == nil {
		return false
	}
	found := false
	ast.Inspect(block, func(n ast.Node) bool {
		if found {
			return false
		}
		switch n.(type) {
		case *ast.TypeSwitchStmt, *ast.TypeAssertExpr:
			found = true
			return false
		}
		return true
	})
	return found
}

// resolveTypeSwitches replaces TypeSwitchStmt nodes with the body of the
// matching CaseClause, based on the known concrete elemType. This must run
// after type parameters are specialized so that the element type is concrete.
//
// For example, given elemType="float32" and:
//
//	switch any(x).(type) {
//	case float32: return float32(math.Sqrt(float64(x)))
//	case float64: return math.Sqrt(x)
//	default:      return float32(math.Sqrt(float64(x)))
//	}
//
// The entire TypeSwitchStmt is replaced by the statements from `case float32:`.
func resolveTypeSwitches(block *ast.BlockStmt, ctx *transformContext) {
	if block == nil {
		return
	}
	resolveTypeSwitchesInBlock(block, ctx.elemType)
	// After resolving, the case body may contain any(x).(Type) patterns
	// that are no-ops in monomorphized code. Simplify them so the C
	// translator (which doesn't handle TypeAssertExpr) can process the AST.
	simplifyTypeAssertions(block)
}

// simplifyTypeAssertions walks the AST and simplifies patterns that arise from
// resolved type switches in generic code:
//
//   - any(expr).(Type) → Type(expr)   (type assertion becomes a conversion)
//   - any(expr)         → expr        (no-op interface boxing stripped)
//
// These patterns are valid Go but opaque to the C translator, which has no
// support for TypeAssertExpr or the any() builtin.
func simplifyTypeAssertions(block *ast.BlockStmt) {
	if block == nil {
		return
	}
	for i, stmt := range block.List {
		block.List[i] = simplifyTypeAssertionsInStmt(stmt)
	}
}

// simplifyTypeAssertionExpr recursively simplifies type assertion patterns in an expression.
func simplifyTypeAssertionExpr(expr ast.Expr) ast.Expr {
	if expr == nil {
		return nil
	}

	switch e := expr.(type) {
	case *ast.TypeAssertExpr:
		// any(inner).(Type) → Type(inner)
		inner := simplifyTypeAssertionExpr(e.X)
		// Unwrap any(x) → x
		inner = unwrapAnyCall(inner)
		if e.Type != nil {
			// Convert to Type(inner) — a type conversion call
			return &ast.CallExpr{
				Fun:  e.Type,
				Args: []ast.Expr{inner},
			}
		}
		return inner

	case *ast.CallExpr:
		// Recursively simplify arguments
		for i, arg := range e.Args {
			e.Args[i] = simplifyTypeAssertionExpr(arg)
		}
		// Simplify the function expression itself
		e.Fun = simplifyTypeAssertionExpr(e.Fun)
		// Unwrap any(x) when used as a standalone call (not in type assertion)
		result := unwrapAnyCall(expr)
		return result

	case *ast.ParenExpr:
		e.X = simplifyTypeAssertionExpr(e.X)
		return e

	case *ast.UnaryExpr:
		e.X = simplifyTypeAssertionExpr(e.X)
		return e

	case *ast.BinaryExpr:
		e.X = simplifyTypeAssertionExpr(e.X)
		e.Y = simplifyTypeAssertionExpr(e.Y)
		return e

	case *ast.IndexExpr:
		e.X = simplifyTypeAssertionExpr(e.X)
		e.Index = simplifyTypeAssertionExpr(e.Index)
		return e

	case *ast.SliceExpr:
		e.X = simplifyTypeAssertionExpr(e.X)
		if e.Low != nil {
			e.Low = simplifyTypeAssertionExpr(e.Low)
		}
		if e.High != nil {
			e.High = simplifyTypeAssertionExpr(e.High)
		}
		return e

	case *ast.StarExpr:
		e.X = simplifyTypeAssertionExpr(e.X)
		return e

	case *ast.CompositeLit:
		for i, elt := range e.Elts {
			e.Elts[i] = simplifyTypeAssertionExpr(elt)
		}
		return e

	case *ast.KeyValueExpr:
		e.Value = simplifyTypeAssertionExpr(e.Value)
		return e
	}

	return expr
}

// unwrapAnyCall simplifies any(x) → x. The any() builtin is a no-op identity
// in monomorphized code (it wraps a concrete value in an empty interface).
func unwrapAnyCall(expr ast.Expr) ast.Expr {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return expr
	}
	ident, ok := call.Fun.(*ast.Ident)
	if !ok || ident.Name != "any" {
		return expr
	}
	if len(call.Args) != 1 {
		return expr
	}
	return call.Args[0]
}

// simplifyTypeAssertionsInStmt recurses into statement blocks.
func simplifyTypeAssertionsInStmt(stmt ast.Stmt) ast.Stmt {
	switch s := stmt.(type) {
	case *ast.ReturnStmt:
		for i, r := range s.Results {
			s.Results[i] = simplifyTypeAssertionExpr(r)
		}
	case *ast.AssignStmt:
		for i, r := range s.Rhs {
			s.Rhs[i] = simplifyTypeAssertionExpr(r)
		}
	case *ast.ExprStmt:
		s.X = simplifyTypeAssertionExpr(s.X)
	case *ast.IfStmt:
		if s.Body != nil {
			simplifyTypeAssertions(s.Body)
		}
		if s.Else != nil {
			s.Else = simplifyTypeAssertionsInStmt(s.Else)
		}
	case *ast.BlockStmt:
		simplifyTypeAssertions(s)
	case *ast.ForStmt:
		if s.Body != nil {
			simplifyTypeAssertions(s.Body)
		}
	case *ast.RangeStmt:
		if s.Body != nil {
			simplifyTypeAssertions(s.Body)
		}
	case *ast.DeclStmt:
		if gd, ok := s.Decl.(*ast.GenDecl); ok {
			for _, spec := range gd.Specs {
				if vs, ok := spec.(*ast.ValueSpec); ok {
					for i, v := range vs.Values {
						vs.Values[i] = simplifyTypeAssertionExpr(v)
					}
				}
			}
		}
	}
	return stmt
}

// resolveTypeSwitchesInBlock replaces TypeSwitchStmt nodes in the block with
// the body of the matching case clause, then recurses into nested blocks.
func resolveTypeSwitchesInBlock(block *ast.BlockStmt, elemType string) {
	var newList []ast.Stmt
	for _, stmt := range block.List {
		if ts, ok := stmt.(*ast.TypeSwitchStmt); ok {
			matched := matchTypeSwitchCase(ts, elemType)
			if matched != nil {
				// Recursively resolve any nested type switches in the matched body.
				for _, s := range matched {
					resolveTypeSwitchesInStmt(s, elemType)
				}
				newList = append(newList, matched...)
			}
			continue
		}
		resolveTypeSwitchesInStmt(stmt, elemType)
		newList = append(newList, stmt)
	}
	block.List = newList
}

// resolveTypeSwitchesInStmt recurses into nested block-containing statements.
func resolveTypeSwitchesInStmt(stmt ast.Stmt, elemType string) {
	switch s := stmt.(type) {
	case *ast.BlockStmt:
		resolveTypeSwitchesInBlock(s, elemType)
	case *ast.IfStmt:
		if s.Body != nil {
			resolveTypeSwitchesInBlock(s.Body, elemType)
		}
		if s.Else != nil {
			resolveTypeSwitchesInStmt(s.Else, elemType)
		}
	case *ast.ForStmt:
		if s.Body != nil {
			resolveTypeSwitchesInBlock(s.Body, elemType)
		}
	case *ast.RangeStmt:
		if s.Body != nil {
			resolveTypeSwitchesInBlock(s.Body, elemType)
		}
	case *ast.SwitchStmt:
		if s.Body != nil {
			resolveTypeSwitchesInBlock(s.Body, elemType)
		}
	case *ast.CaseClause:
		// Recurse into case clause bodies (from regular switch statements).
		for i, bodyStmt := range s.Body {
			if ts, ok := bodyStmt.(*ast.TypeSwitchStmt); ok {
				matched := matchTypeSwitchCase(ts, elemType)
				if matched != nil {
					// Replace the TypeSwitchStmt with matched body inline.
					newBody := make([]ast.Stmt, 0, len(s.Body)-1+len(matched))
					newBody = append(newBody, s.Body[:i]...)
					newBody = append(newBody, matched...)
					newBody = append(newBody, s.Body[i+1:]...)
					s.Body = newBody
					// Recurse on the newly inserted statements.
					for _, m := range matched {
						resolveTypeSwitchesInStmt(m, elemType)
					}
					return // restart not needed — only one type switch per clause expected
				}
			} else {
				resolveTypeSwitchesInStmt(bodyStmt, elemType)
			}
		}
	}
}

// matchTypeSwitchCase finds the CaseClause in a TypeSwitchStmt that matches
// elemType, and returns its body statements. Returns nil if no match is found.
//
// Matching rules:
//   - "float32" matches `case float32:`
//   - "float64" matches `case float64:`
//   - "hwy.Float16" and "hwy.BFloat16" match `default:` first, then fall back
//     to `case float32:` (since half-precision types compute through float32)
//   - If no specific case matches, `default:` is used
func matchTypeSwitchCase(ts *ast.TypeSwitchStmt, elemType string) []ast.Stmt {
	if ts.Body == nil {
		return nil
	}

	// Normalize elemType to the bare Go type name used in case clauses.
	bareType := elemType
	switch elemType {
	case "hwy.Float16", "hwy.BFloat16":
		bareType = "" // no direct case match; rely on default or float32 fallback
	}

	var defaultClause *ast.CaseClause
	var float32Clause *ast.CaseClause

	for _, stmt := range ts.Body.List {
		cc, ok := stmt.(*ast.CaseClause)
		if !ok {
			continue
		}
		if cc.List == nil {
			// default: clause
			defaultClause = cc
			continue
		}
		for _, typeExpr := range cc.List {
			typeName := exprToString(typeExpr)
			if typeName == bareType {
				return cc.Body
			}
			if typeName == "float32" {
				float32Clause = cc
			}
		}
	}

	// For half-precision types, prefer default, then float32.
	if bareType == "" {
		if defaultClause != nil {
			return defaultClause.Body
		}
		if float32Clause != nil {
			return float32Clause.Body
		}
	}

	// Fall back to default clause for any unmatched type.
	if defaultClause != nil {
		return defaultClause.Body
	}

	return nil
}
