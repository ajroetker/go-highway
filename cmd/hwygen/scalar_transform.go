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
	"go/ast"
	"go/token"
	"strings"
)

// scalarizableHwyOps is the set of hwy operations that can be scalarized.
// These are simple operations that map directly to scalar Go code.
var scalarizableHwyOps = map[string]bool{
	// Memory operations
	"Load":      true,
	"LoadSlice": true,
	"Store":     true,
	"StoreSlice": true,

	// Initialization
	"Zero":  true,
	"Set":   true,
	"Const": true,

	// Arithmetic
	"Add":    true,
	"Sub":    true,
	"Mul":    true,
	"Div":    true,
	"FMA":    true,
	"MulAdd": true,
	"Neg":    true,

	// Min/Max (map to Go builtins)
	"Min": true,
	"Max": true,

	// Reductions (become identity in scalar code)
	"ReduceSum": true,
	"ReduceMin": true,
	"ReduceMax": true,

	// Lane count (becomes 1)
	"NumLanes": true,
	"MaxLanes": true,

	// Interleave operations (identity for scalar, but with lanes=1 loops never execute)
	"InterleaveLower": true,
	"InterleaveUpper": true,

	// Float16/BFloat16 conversion functions (pass through unchanged in scalar code)
	"Float32ToFloat16":  true,
	"Float32ToBFloat16": true,
	"Float16ToFloat32":  true,
	"BFloat16ToFloat32": true,
}

// nonScalarizableHwyOps is the set of hwy operations that prevent scalarization.
// If any of these are found, we keep the hwy.Vec fallback code.
var nonScalarizableHwyOps = map[string]bool{
	// Mask operations
	"Mask":        true,
	"FirstN":      true,
	"IfThenElse":  true,
	"Merge":       true,
	"Compress":    true,
	"MaskFromVec": true,
	"VecFromMask": true,
	"AllTrue":     true,
	"AllFalse":    true,
	"CountTrue":   true,

	// Comparison operations that return masks
	"Equal":              true,
	"NotEqual":           true,
	"Less":               true,
	"LessThan":           true,
	"LessEqual":          true,
	"LessThanOrEqual":    true,
	"Greater":            true,
	"GreaterThan":        true,
	"GreaterEqual":       true,
	"GreaterThanOrEqual": true,
	"IsNaN":              true,
	"IsInf":              true,

	// Shuffle/permute operations
	"Shuffle":     true,
	"Reverse":     true,
	"Rotate":      true,
	"InterleaveLo": true,
	"InterleaveHi": true,
	"ConcatLo":     true,
	"ConcatHi":     true,
	"OddEven":      true,

	// Shift operations
	"ShiftLeft":     true,
	"ShiftRight":    true,
	"ShiftLeftLane": true,
	"ShiftRightLane": true,

	// Bitwise operations (complex to handle correctly for floats)
	"And":    true,
	"Or":     true,
	"Xor":    true,
	"AndNot": true,
	"Not":    true,

	// Type reinterpretation
	"BitCast":       true,
	"ReinterpretCast": true,

	// Vector-specific operations
	"Iota":        true,
	"Data":        true,
	"Gather":      true,
	"Scatter":     true,
	"MaskedLoad":  true,
	"MaskedStore": true,

	// Special math operations (complex polynomials)
	"Sqrt":               true,
	"RSqrt":              true,
	"RSqrtNewtonRaphson": true,
	"RSqrtPrecise":       true,
	"Abs":                true, // could be scalarized but has edge cases

	// Half-precision specific
	"AddF16":    true,
	"SubF16":    true,
	"MulF16":    true,
	"DivF16":    true,
	"FMAF16":    true,
	"NegF16":    true,
	"MinF16":    true,
	"MaxF16":    true,
	"AddBF16":   true,
	"SubBF16":   true,
	"MulBF16":   true,
	"DivBF16":   true,
	"FMABF16":   true,
	"NegBF16":   true,
	"MinBF16":   true,
	"MaxBF16":   true,
}

// scalarizableContribMath maps contrib math functions to stdlib equivalents.
// These are Vec-to-Vec functions that can be replaced with stdlib calls.
var scalarizableContribMath = map[string]string{
	// Exp family
	"BaseExpVec":   "Exp",
	"BaseExp2Vec":  "Exp2",
	"BaseExp10Vec": "Exp10",
	// Log family
	"BaseLogVec":   "Log",
	"BaseLog2Vec":  "Log2",
	"BaseLog10Vec": "Log10",
	// Trig functions
	"BaseSinVec":  "Sin",
	"BaseCosVec":  "Cos",
	"BaseTanhVec": "Tanh",
	"BaseSinhVec": "Sinh",
	"BaseCoshVec": "Cosh",
	// Inverse hyperbolic
	"BaseAsinhVec": "Asinh",
	"BaseAcoshVec": "Acosh",
	"BaseAtanhVec": "Atanh",
	// Error function
	"BaseErfVec": "Erf",
}

// containsVecType checks if a type expression contains hwy.Vec or hwy.Mask.
func containsVecType(typeExpr ast.Expr) bool {
	if typeExpr == nil {
		return false
	}

	found := false
	ast.Inspect(typeExpr, func(n ast.Node) bool {
		if found {
			return false
		}

		switch t := n.(type) {
		case *ast.SelectorExpr:
			// Check for hwy.Vec or hwy.Mask
			if pkg, ok := t.X.(*ast.Ident); ok && pkg.Name == "hwy" {
				if t.Sel.Name == "Vec" || t.Sel.Name == "Mask" {
					found = true
					return false
				}
			}
		case *ast.Ident:
			// Check for Vec or Mask (unqualified)
			if t.Name == "Vec" || t.Name == "Mask" {
				found = true
				return false
			}
		}
		return true
	})

	return found
}

// hasMinMaxVariables checks if a function body has local variables named 'min' or 'max'.
// These would shadow the builtin min/max functions after scalarization.
func hasMinMaxVariables(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}

		switch node := n.(type) {
		case *ast.AssignStmt:
			// Check short variable declarations: min := ...
			if node.Tok == token.DEFINE {
				for _, lhs := range node.Lhs {
					if ident, ok := lhs.(*ast.Ident); ok {
						if ident.Name == "min" || ident.Name == "max" {
							found = true
							return false
						}
					}
				}
			}
		case *ast.DeclStmt:
			// Check var declarations: var min float32
			if genDecl, ok := node.Decl.(*ast.GenDecl); ok && genDecl.Tok == token.VAR {
				for _, spec := range genDecl.Specs {
					if valueSpec, ok := spec.(*ast.ValueSpec); ok {
						for _, name := range valueSpec.Names {
							if name.Name == "min" || name.Name == "max" {
								found = true
								return false
							}
						}
					}
				}
			}
		}
		return true
	})
	return found
}

// hasComplexLanesUsage checks if 'lanes' is used in problematic contexts.
// Returns true if lanes is used in contexts that would change semantics:
// - Array indices: arr[lanes] (semantic difference when lanes becomes 1)
// - Slice bounds: arr[:lanes] (semantic difference when lanes becomes 1)
//
// Note: These patterns are NOT considered problematic:
// - For loop conditions (i+lanes <= n) and post (i += lanes) - handled by scalarization
// - If conditions (if len(v) < lanes) - becomes dead code but harmless
func hasComplexLanesUsage(body *ast.BlockStmt) bool {
	hasProblematicUsage := false

	ast.Inspect(body, func(n ast.Node) bool {
		if hasProblematicUsage {
			return false
		}

		switch node := n.(type) {
		case *ast.IndexExpr:
			// Check if lanes is used as array index (e.g., arr[lanes])
			// This would change from arr[N] to arr[1] which is different
			if containsLanesIdent(node.Index) {
				hasProblematicUsage = true
				return false
			}
		case *ast.SliceExpr:
			// Check if lanes is used in slice bounds (e.g., arr[:lanes])
			// This would change from arr[:N] to arr[:1] which is different
			if containsLanesIdent(node.Low) || containsLanesIdent(node.High) {
				hasProblematicUsage = true
				return false
			}
		}
		return true
	})

	return hasProblematicUsage
}

// containsLanesIdent checks if an expression contains the 'lanes' identifier.
func containsLanesIdent(expr ast.Expr) bool {
	if expr == nil {
		return false
	}
	found := false
	ast.Inspect(expr, func(n ast.Node) bool {
		if found {
			return false
		}
		if ident, ok := n.(*ast.Ident); ok && ident.Name == "lanes" {
			found = true
			return false
		}
		return true
	})
	return found
}

// canScalarizeFallback checks if a function can be scalarized.
// Returns true only if:
// - The function does NOT have hwy.Vec parameters or return types (Vec-to-Vec functions)
// - ALL hwy operations in the body are in the scalarizable set
// - The function doesn't have local variables named 'min' or 'max' (would shadow builtins)
// - The function doesn't use 'lanes' outside of simple loop patterns
func canScalarizeFallback(funcDecl *ast.FuncDecl) bool {
	if funcDecl.Body == nil {
		return false
	}

	// Check if function signature uses hwy.Vec types - if so, don't scalarize
	if funcDecl.Type != nil {
		// Check parameters
		if funcDecl.Type.Params != nil {
			for _, field := range funcDecl.Type.Params.List {
				if containsVecType(field.Type) {
					return false
				}
			}
		}
		// Check return types
		if funcDecl.Type.Results != nil {
			for _, field := range funcDecl.Type.Results.List {
				if containsVecType(field.Type) {
					return false
				}
				// Check for named return values that would shadow min/max builtins
				for _, name := range field.Names {
					if name.Name == "min" || name.Name == "max" {
						return false
					}
				}
			}
		}
	}

	// Check for local variables named 'min' or 'max' that would shadow builtins
	if hasMinMaxVariables(funcDecl.Body) {
		return false
	}

	// Check if 'lanes' is used outside simple loop patterns (condition/post)
	// If lanes is used elsewhere (e.g., array sizes, other statements), don't scalarize
	if hasComplexLanesUsage(funcDecl.Body) {
		return false
	}

	canScalarize := true
	ast.Inspect(funcDecl.Body, func(n ast.Node) bool {
		if !canScalarize {
			return false
		}

		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}

		// Check for hwy.* calls
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
				opName := sel.Sel.Name
				if nonScalarizableHwyOps[opName] {
					canScalarize = false
					return false
				}
				if !scalarizableHwyOps[opName] {
					// Unknown op - be conservative
					canScalarize = false
					return false
				}
			}
		}

		// Check for hwy.Op[T](...) indexed calls (generic calls)
		if idx, ok := call.Fun.(*ast.IndexExpr); ok {
			if sel, ok := idx.X.(*ast.SelectorExpr); ok {
				if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
					opName := sel.Sel.Name
					if nonScalarizableHwyOps[opName] {
						canScalarize = false
						return false
					}
					if !scalarizableHwyOps[opName] {
						canScalarize = false
						return false
					}
				}
			}
		}

		// Check for math.Base*Vec calls
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "math" {
				funcName := sel.Sel.Name
				// Check if it's a Vec function with fallback suffix
				for baseFunc := range scalarizableContribMath {
					if strings.HasPrefix(funcName, baseFunc) {
						return true // scalarizable
					}
				}
				// Non-Vec math functions or unknown ones - check if they're contrib functions
				if strings.HasPrefix(funcName, "Base") && strings.Contains(funcName, "Vec") {
					// Unknown Vec function - be conservative
					canScalarize = false
					return false
				}
			}
		}

		// Check for method calls like v.NumLanes()
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
			methodName := sel.Sel.Name
			if methodName == "NumLanes" || methodName == "Data" {
				if methodName == "Data" {
					canScalarize = false
					return false
				}
				// NumLanes is scalarizable
			}
		}

		return true
	})

	return canScalarize
}

// scalarizeFallback transforms a fallback function to use pure scalar operations.
// This modifies the function declaration in place.
func scalarizeFallback(funcDecl *ast.FuncDecl, elemType string) {
	if funcDecl.Body == nil {
		return
	}

	funcDecl.Body.List = scalarizeStmts(funcDecl.Body.List, elemType)
}

// scalarizeStmts transforms a list of statements to use scalar operations.
func scalarizeStmts(stmts []ast.Stmt, elemType string) []ast.Stmt {
	var result []ast.Stmt

	for _, stmt := range stmts {
		newStmts := scalarizeStmt(stmt, elemType)
		result = append(result, newStmts...)
	}

	return result
}

// scalarizeStmt transforms a single statement.
// Returns a slice because some statements may be removed or split.
func scalarizeStmt(stmt ast.Stmt, elemType string) []ast.Stmt {
	switch s := stmt.(type) {
	case *ast.AssignStmt:
		return scalarizeAssign(s, elemType)

	case *ast.DeclStmt:
		return scalarizeDecl(s, elemType)

	case *ast.ExprStmt:
		return scalarizeExprStmt(s, elemType)

	case *ast.ForStmt:
		return []ast.Stmt{scalarizeFor(s, elemType)}

	case *ast.RangeStmt:
		s.X = scalarizeExpr(s.X, elemType)
		s.Body = &ast.BlockStmt{List: scalarizeStmts(s.Body.List, elemType)}
		return []ast.Stmt{s}

	case *ast.IfStmt:
		s.Cond = scalarizeExpr(s.Cond, elemType)
		s.Body = &ast.BlockStmt{List: scalarizeStmts(s.Body.List, elemType)}
		if s.Else != nil {
			switch e := s.Else.(type) {
			case *ast.BlockStmt:
				s.Else = &ast.BlockStmt{List: scalarizeStmts(e.List, elemType)}
			case *ast.IfStmt:
				elseStmts := scalarizeStmt(e, elemType)
				if len(elseStmts) == 1 {
					s.Else = elseStmts[0]
				}
			}
		}
		return []ast.Stmt{s}

	case *ast.BlockStmt:
		s.List = scalarizeStmts(s.List, elemType)
		return []ast.Stmt{s}

	case *ast.ReturnStmt:
		for i, r := range s.Results {
			s.Results[i] = scalarizeExpr(r, elemType)
		}
		return []ast.Stmt{s}

	case *ast.SwitchStmt:
		if s.Tag != nil {
			s.Tag = scalarizeExpr(s.Tag, elemType)
		}
		s.Body = &ast.BlockStmt{List: scalarizeStmts(s.Body.List, elemType)}
		return []ast.Stmt{s}

	case *ast.CaseClause:
		for i, e := range s.List {
			s.List[i] = scalarizeExpr(e, elemType)
		}
		s.Body = scalarizeStmts(s.Body, elemType)
		return []ast.Stmt{s}

	default:
		return []ast.Stmt{stmt}
	}
}

// scalarizeAssign transforms an assignment statement.
func scalarizeAssign(s *ast.AssignStmt, elemType string) []ast.Stmt {
	// Check if this is a Store/StoreSlice call on the RHS
	// Pattern: hwy.Store(val, dst[i:]) should become dst[i] = val
	if len(s.Rhs) == 1 {
		if call, ok := s.Rhs[0].(*ast.CallExpr); ok {
			if isHwyStoreCall(call) {
				return scalarizeStoreCall(call, elemType)
			}
		}
	}

	// Check if LHS is assigning to `lanes` variable from NumLanes()
	// Pattern: lanes := sum.NumLanes() or lanes := hwy.Zero[T]().NumLanes()
	// These should be removed since we don't need lanes in scalar code
	if len(s.Lhs) == 1 && len(s.Rhs) == 1 {
		if ident, ok := s.Lhs[0].(*ast.Ident); ok && ident.Name == "lanes" {
			if isNumLanesCall(s.Rhs[0]) {
				// Remove this statement entirely
				return nil
			}
		}
	}

	// Transform RHS expressions
	for i, rhs := range s.Rhs {
		s.Rhs[i] = scalarizeExpr(rhs, elemType)
	}

	// Transform LHS expressions, but only for complex expressions like index/slice
	// Don't transform simple identifiers - they should remain as-is
	for i, lhs := range s.Lhs {
		switch lhs.(type) {
		case *ast.IndexExpr, *ast.SliceExpr, *ast.SelectorExpr:
			s.Lhs[i] = scalarizeExpr(lhs, elemType)
		// Don't transform *ast.Ident - identifiers should stay as-is on LHS
		}
	}

	return []ast.Stmt{s}
}

// scalarizeDecl transforms a declaration statement.
func scalarizeDecl(s *ast.DeclStmt, elemType string) []ast.Stmt {
	if genDecl, ok := s.Decl.(*ast.GenDecl); ok && genDecl.Tok == token.VAR {
		for _, spec := range genDecl.Specs {
			if valueSpec, ok := spec.(*ast.ValueSpec); ok {
				// Transform the type: hwy.Vec[T] → T
				if valueSpec.Type != nil {
					valueSpec.Type = scalarizeVecTypeExpr(valueSpec.Type, elemType)
				}
				for i, val := range valueSpec.Values {
					valueSpec.Values[i] = scalarizeExpr(val, elemType)
				}
			}
		}
	}
	return []ast.Stmt{s}
}

// scalarizeVecTypeExpr converts hwy.Vec[T] type expressions to scalar type T.
// E.g., hwy.Vec[float32] → float32, hwy.Vec[int32] → int32.
func scalarizeVecTypeExpr(typeExpr ast.Expr, elemType string) ast.Expr {
	// Match hwy.Vec[T] which is an IndexExpr wrapping a SelectorExpr
	if idx, ok := typeExpr.(*ast.IndexExpr); ok {
		if sel, ok := idx.X.(*ast.SelectorExpr); ok {
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
				if sel.Sel.Name == "Vec" || sel.Sel.Name == "Mask" {
					// Return the type argument (e.g., float32, int32)
					return idx.Index
				}
			}
		}
	}
	return typeExpr
}

// scalarizeExprStmt transforms an expression statement.
func scalarizeExprStmt(s *ast.ExprStmt, elemType string) []ast.Stmt {
	// Check if this is a Store/StoreSlice call
	if call, ok := s.X.(*ast.CallExpr); ok {
		if isHwyStoreCall(call) {
			return scalarizeStoreCall(call, elemType)
		}
	}

	s.X = scalarizeExpr(s.X, elemType)
	return []ast.Stmt{s}
}

// scalarizeFor transforms a for loop for scalar execution.
func scalarizeFor(s *ast.ForStmt, elemType string) *ast.ForStmt {
	// Transform init
	if s.Init != nil {
		if init, ok := s.Init.(*ast.AssignStmt); ok {
			initStmts := scalarizeAssign(init, elemType)
			if len(initStmts) == 1 {
				if newInit, ok := initStmts[0].(*ast.AssignStmt); ok {
					s.Init = newInit
				}
			} else if len(initStmts) == 0 {
				s.Init = nil
			}
		}
	}

	// Transform the condition
	// Pattern: i+lanes <= n -> i < n
	if s.Cond != nil {
		s.Cond = scalarizeLoopCond(s.Cond, elemType)
	}

	// Transform the post
	// Pattern: i += lanes -> i++
	if s.Post != nil {
		s.Post = scalarizeLoopPost(s.Post, elemType)
	}

	// Transform the body
	s.Body = &ast.BlockStmt{List: scalarizeStmts(s.Body.List, elemType)}

	return s
}

// scalarizeLoopCond transforms a loop condition from SIMD to scalar.
// Pattern: i+lanes <= n -> i < n
func scalarizeLoopCond(cond ast.Expr, elemType string) ast.Expr {
	binExpr, ok := cond.(*ast.BinaryExpr)
	if !ok {
		return scalarizeExpr(cond, elemType)
	}

	// Check for pattern: i+lanes <= n or i+lanes < n
	if binExpr.Op == token.LEQ || binExpr.Op == token.LSS {
		if addExpr, ok := binExpr.X.(*ast.BinaryExpr); ok && addExpr.Op == token.ADD {
			if isLanesIdent(addExpr.Y) {
				// Transform i+lanes <= n to i < n
				return &ast.BinaryExpr{
					X:  addExpr.X, // i
					Op: token.LSS,
					Y:  binExpr.Y, // n
				}
			}
		}
	}

	return scalarizeExpr(cond, elemType)
}

// scalarizeLoopPost transforms a loop post statement from SIMD to scalar.
// Patterns handled:
// - i += lanes -> i++
// - i += v.NumLanes() -> i++
// - i += hwy.NumLanes[T]() -> i++
func scalarizeLoopPost(post ast.Stmt, elemType string) ast.Stmt {
	switch p := post.(type) {
	case *ast.AssignStmt:
		if p.Tok == token.ADD_ASSIGN && len(p.Lhs) == 1 && len(p.Rhs) == 1 {
			rhs := p.Rhs[0]

			// Pattern: i += lanes
			if isLanesIdent(rhs) {
				return &ast.IncDecStmt{
					X:   p.Lhs[0],
					Tok: token.INC,
				}
			}

			// Pattern: i += v.NumLanes() or i += hwy.NumLanes[T]()
			// Check if RHS is a NumLanes call and transform to i++
			if call, ok := rhs.(*ast.CallExpr); ok {
				if isNumLanesCall(call) {
					return &ast.IncDecStmt{
						X:   p.Lhs[0],
						Tok: token.INC,
					}
				}
			}

			// For other patterns, scalarize the RHS (it might contain lanes or NumLanes calls)
			scalarizedRhs := scalarizeExpr(rhs, elemType)
			// If the scalarized RHS is just 1, convert to i++
			if lit, ok := scalarizedRhs.(*ast.BasicLit); ok && lit.Kind == token.INT && lit.Value == "1" {
				return &ast.IncDecStmt{
					X:   p.Lhs[0],
					Tok: token.INC,
				}
			}
			p.Rhs[0] = scalarizedRhs
		}
	case *ast.IncDecStmt:
		// Already scalar form
		return p
	}

	return post
}

// scalarizeExpr transforms an expression to use scalar operations.
func scalarizeExpr(expr ast.Expr, elemType string) ast.Expr {
	if expr == nil {
		return nil
	}

	switch e := expr.(type) {
	case *ast.CallExpr:
		return scalarizeCallExpr(e, elemType)

	case *ast.BinaryExpr:
		e.X = scalarizeExpr(e.X, elemType)
		e.Y = scalarizeExpr(e.Y, elemType)
		return e

	case *ast.UnaryExpr:
		e.X = scalarizeExpr(e.X, elemType)
		return e

	case *ast.ParenExpr:
		e.X = scalarizeExpr(e.X, elemType)
		return e

	case *ast.IndexExpr:
		e.X = scalarizeExpr(e.X, elemType)
		e.Index = scalarizeExpr(e.Index, elemType)
		return e

	case *ast.SliceExpr:
		e.X = scalarizeExpr(e.X, elemType)
		if e.Low != nil {
			e.Low = scalarizeExpr(e.Low, elemType)
		}
		if e.High != nil {
			e.High = scalarizeExpr(e.High, elemType)
		}
		if e.Max != nil {
			e.Max = scalarizeExpr(e.Max, elemType)
		}
		return e

	case *ast.SelectorExpr:
		// Handle method calls like v.NumLanes()
		// If this is part of a call, it will be handled by scalarizeCallExpr
		e.X = scalarizeExpr(e.X, elemType)
		return e

	case *ast.Ident:
		// Check if this is the `lanes` variable - replace with 1
		if e.Name == "lanes" {
			return &ast.BasicLit{Kind: token.INT, Value: "1"}
		}
		return e

	case *ast.BasicLit:
		return e

	case *ast.CompositeLit:
		for i, elt := range e.Elts {
			e.Elts[i] = scalarizeExpr(elt, elemType)
		}
		return e

	case *ast.KeyValueExpr:
		e.Key = scalarizeExpr(e.Key, elemType)
		e.Value = scalarizeExpr(e.Value, elemType)
		return e

	default:
		return expr
	}
}

// scalarizeCallExpr transforms a call expression to scalar form.
func scalarizeCallExpr(call *ast.CallExpr, elemType string) ast.Expr {
	// First, recursively scalarize arguments
	for i, arg := range call.Args {
		call.Args[i] = scalarizeExpr(arg, elemType)
	}

	// Check for make([]hwy.Vec[T], ...) calls - transform to make([]T, ...)
	if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "make" {
		if len(call.Args) >= 1 {
			if arrayType, ok := call.Args[0].(*ast.ArrayType); ok {
				if scalarizedElem := scalarizeVecType(arrayType.Elt, elemType); scalarizedElem != nil {
					arrayType.Elt = scalarizedElem
				}
			}
		}
		return call
	}

	// Check for hwy.Op(...) calls
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
			return scalarizeHwyCall(sel.Sel.Name, call.Args, elemType)
		}

		// Check for method calls like v.NumLanes()
		if sel.Sel.Name == "NumLanes" {
			return &ast.BasicLit{Kind: token.INT, Value: "1"}
		}
	}

	// Check for hwy.Op[T](...) indexed calls
	if idx, ok := call.Fun.(*ast.IndexExpr); ok {
		if sel, ok := idx.X.(*ast.SelectorExpr); ok {
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
				return scalarizeHwyCall(sel.Sel.Name, call.Args, elemType)
			}
		}
	}

	// Check for contrib math calls
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "math" {
			funcName := sel.Sel.Name
			for baseFunc, stdlibFunc := range scalarizableContribMath {
				if strings.HasPrefix(funcName, baseFunc) {
					return scalarizeContribMathCall(stdlibFunc, call.Args, elemType)
				}
			}
		}
	}

	return call
}

// scalarizeHwyCall transforms an hwy.* call to scalar form.
func scalarizeHwyCall(opName string, args []ast.Expr, elemType string) ast.Expr {
	switch opName {
	case "Add":
		if len(args) >= 2 {
			return &ast.BinaryExpr{X: args[0], Op: token.ADD, Y: args[1]}
		}
	case "Sub":
		if len(args) >= 2 {
			return &ast.BinaryExpr{X: args[0], Op: token.SUB, Y: args[1]}
		}
	case "Mul":
		if len(args) >= 2 {
			return &ast.BinaryExpr{X: args[0], Op: token.MUL, Y: args[1]}
		}
	case "Div":
		if len(args) >= 2 {
			return &ast.BinaryExpr{X: args[0], Op: token.QUO, Y: args[1]}
		}
	case "FMA", "MulAdd":
		// FMA(a, b, c) = a*b + c
		if len(args) >= 3 {
			mulExpr := &ast.BinaryExpr{X: args[0], Op: token.MUL, Y: args[1]}
			return &ast.BinaryExpr{X: mulExpr, Op: token.ADD, Y: args[2]}
		}
	case "Neg":
		if len(args) >= 1 {
			return &ast.UnaryExpr{Op: token.SUB, X: args[0]}
		}
	case "Min":
		if len(args) >= 2 {
			return &ast.CallExpr{
				Fun:  ast.NewIdent("min"),
				Args: []ast.Expr{args[0], args[1]},
			}
		}
	case "Max":
		if len(args) >= 2 {
			return &ast.CallExpr{
				Fun:  ast.NewIdent("max"),
				Args: []ast.Expr{args[0], args[1]},
			}
		}
	case "Load", "LoadSlice":
		// Load(src[i:]) -> src[i]
		// LoadSlice(src) -> src[0]
		if len(args) >= 1 {
			return extractIndexFromSlice(args[0])
		}
	case "Zero":
		// Zero[T]() -> T(0)
		return makeTypedZero(elemType)
	case "Set", "Const":
		// Set(val) or Const(val) -> T(val)
		// For half-precision types, use conversion functions since T(val) won't compile
		if len(args) >= 1 {
			if elemType == "hwy.Float16" {
				// hwy.Float32ToFloat16(float32(val))
				return &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   ast.NewIdent("hwy"),
						Sel: ast.NewIdent("Float32ToFloat16"),
					},
					Args: []ast.Expr{args[0]},
				}
			} else if elemType == "hwy.BFloat16" {
				// hwy.Float32ToBFloat16(float32(val))
				return &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   ast.NewIdent("hwy"),
						Sel: ast.NewIdent("Float32ToBFloat16"),
					},
					Args: []ast.Expr{args[0]},
				}
			}
			// For native types, use type conversion
			return &ast.CallExpr{
				Fun:  parseTypeExpr(elemType),
				Args: []ast.Expr{args[0]},
			}
		}
	case "ReduceSum", "ReduceMin", "ReduceMax":
		// In scalar code, reduce is identity
		if len(args) >= 1 {
			return args[0]
		}
	case "NumLanes", "MaxLanes":
		return &ast.BasicLit{Kind: token.INT, Value: "1"}
	case "InterleaveLower":
		// With lanes=1, interleave lower is identity (take first element)
		if len(args) >= 1 {
			return args[0]
		}
	case "InterleaveUpper":
		// With lanes=1, interleave upper takes from second arg
		if len(args) >= 2 {
			return args[1]
		}
	case "Float32ToFloat16", "Float32ToBFloat16", "Float16ToFloat32", "BFloat16ToFloat32":
		// Type conversion functions - pass through unchanged
		return &ast.CallExpr{
			Fun: &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent(opName),
			},
			Args: args,
		}
	}

	// If we don't know how to scalarize, return as-is (shouldn't happen if canScalarize worked)
	return &ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X:   ast.NewIdent("hwy"),
			Sel: ast.NewIdent(opName),
		},
		Args: args,
	}
}

// scalarizeContribMathCall transforms a contrib math Vec function to stdlib.
func scalarizeContribMathCall(stdlibFunc string, args []ast.Expr, elemType string) ast.Expr {
	if len(args) < 1 {
		return &ast.CallExpr{
			Fun: &ast.SelectorExpr{
				X:   ast.NewIdent("stdmath"),
				Sel: ast.NewIdent(stdlibFunc),
			},
			Args: args,
		}
	}

	// Pattern: T(stdmath.Func(float64(x)))
	// Convert input to float64, call stdlib, convert back to T
	argFloat64 := &ast.CallExpr{
		Fun:  ast.NewIdent("float64"),
		Args: []ast.Expr{args[0]},
	}
	stdCall := &ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X:   ast.NewIdent("stdmath"),
			Sel: ast.NewIdent(stdlibFunc),
		},
		Args: []ast.Expr{argFloat64},
	}
	return &ast.CallExpr{
		Fun:  parseTypeExpr(elemType),
		Args: []ast.Expr{stdCall},
	}
}

// scalarizeStoreCall transforms a Store/StoreSlice call to an assignment.
// hwy.Store(val, dst[i:]) -> dst[i] = val
func scalarizeStoreCall(call *ast.CallExpr, elemType string) []ast.Stmt {
	if len(call.Args) < 2 {
		return nil
	}

	val := scalarizeExpr(call.Args[0], elemType)
	dst := call.Args[1]

	// Extract index from dst[i:]
	target := extractIndexFromSlice(dst)

	return []ast.Stmt{
		&ast.AssignStmt{
			Lhs: []ast.Expr{target},
			Tok: token.ASSIGN,
			Rhs: []ast.Expr{val},
		},
	}
}

// isHwyStoreCall checks if a call is hwy.Store or hwy.StoreSlice.
func isHwyStoreCall(call *ast.CallExpr) bool {
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
			return sel.Sel.Name == "Store" || sel.Sel.Name == "StoreSlice"
		}
	}
	return false
}

// isNumLanesCall checks if an expression is a NumLanes() call.
func isNumLanesCall(expr ast.Expr) bool {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return false
	}

	// Check for v.NumLanes()
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if sel.Sel.Name == "NumLanes" {
			return true
		}
	}

	// Check for hwy.NumLanes[T]() or hwy.MaxLanes[T]()
	if idx, ok := call.Fun.(*ast.IndexExpr); ok {
		if sel, ok := idx.X.(*ast.SelectorExpr); ok {
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
				return sel.Sel.Name == "MaxLanes" || sel.Sel.Name == "NumLanes"
			}
		}
	}

	return false
}

// isLanesIdent checks if an expression is the `lanes` identifier.
func isLanesIdent(expr ast.Expr) bool {
	if ident, ok := expr.(*ast.Ident); ok {
		return ident.Name == "lanes"
	}
	return false
}

// extractIndexFromSlice extracts an index expression from a slice.
// src[i:] -> src[i]
// src -> src[0]
func extractIndexFromSlice(expr ast.Expr) ast.Expr {
	if sliceExpr, ok := expr.(*ast.SliceExpr); ok {
		if sliceExpr.Low != nil {
			// src[i:] -> src[i]
			return &ast.IndexExpr{
				X:     sliceExpr.X,
				Index: sliceExpr.Low,
			}
		}
		// src[:] -> src[0]
		return &ast.IndexExpr{
			X:     sliceExpr.X,
			Index: &ast.BasicLit{Kind: token.INT, Value: "0"},
		}
	}
	// Not a slice - assume it's a slice variable, use index 0
	return &ast.IndexExpr{
		X:     expr,
		Index: &ast.BasicLit{Kind: token.INT, Value: "0"},
	}
}

// scalarizeVecType transforms hwy.Vec[T] to T (the scalar element type).
// Returns nil if the type is not hwy.Vec[T].
func scalarizeVecType(typeExpr ast.Expr, elemType string) ast.Expr {
	// Check for hwy.Vec[T] - IndexExpr where X is hwy.Vec selector
	if idx, ok := typeExpr.(*ast.IndexExpr); ok {
		if sel, ok := idx.X.(*ast.SelectorExpr); ok {
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" && sel.Sel.Name == "Vec" {
				// Return the type parameter (T)
				return idx.Index
			}
		}
	}
	return nil
}

// makeTypedZero creates a zero value for the given element type.
// Always uses explicit type conversion to ensure correct type inference.
func makeTypedZero(elemType string) ast.Expr {
	// For half-precision types, use conversion functions
	switch elemType {
	case "hwy.Float16":
		// hwy.Float32ToFloat16(0)
		return &ast.CallExpr{
			Fun: &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent("Float32ToFloat16"),
			},
			Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "0"}},
		}
	case "hwy.BFloat16":
		// hwy.Float32ToBFloat16(0)
		return &ast.CallExpr{
			Fun: &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent("Float32ToBFloat16"),
			},
			Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "0"}},
		}
	default:
		// Always use explicit type conversion: T(0)
		// This is necessary because untyped literals would default to
		// int or float64, causing type mismatches in short variable declarations.
		return &ast.CallExpr{
			Fun:  parseTypeExpr(elemType),
			Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "0"}},
		}
	}
}
