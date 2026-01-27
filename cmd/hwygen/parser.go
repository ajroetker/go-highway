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
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"strings"
)

// ParsedFunc represents a function that has been parsed from the input file.
type ParsedFunc struct {
	Name       string            // Function name
	TypeParams []TypeParam       // Generic type parameters
	Params     []Param           // Function parameters
	Returns    []Param           // Return values
	Body       *ast.BlockStmt    // Function body
	HwyCalls   []HwyCall         // Detected hwy.* and contrib.* calls
	LoopInfo   *LoopInfo         // Main processing loop info
	Doc        *ast.CommentGroup // Function documentation
	Private    bool              // true if base function uses lowercase "base" prefix (generates unexported dispatch)
}

// TypeParam represents a generic type parameter.
type TypeParam struct {
	Name       string // T
	Constraint string // hwy.Floats
}

// hasHwyLanesConstraint checks if any type parameter has an hwy.Lanes-related constraint.
func hasHwyLanesConstraint(typeParams []TypeParam) bool {
	for _, tp := range typeParams {
		// Check for hwy.Lanes, hwy.Floats, hwy.Ints, hwy.SignedInts, hwy.UnsignedInts
		if strings.Contains(tp.Constraint, "hwy.") {
			return true
		}
	}
	return false
}

// Param represents a function parameter or return value.
type Param struct {
	Name string // parameter name (may be empty for returns)
	Type string // type expression as string
}

// HwyCall represents a call to hwy.* or contrib.* package.
type HwyCall struct {
	Package  string    // "hwy" or "contrib"
	FuncName string    // "Load", "Add", "Exp", etc.
	Position token.Pos // Source position
}

// LoopInfo represents information about the main vectorized loop.
type LoopInfo struct {
	Iterator   string // "ii", "i", etc.
	Start      string // "0"
	End        string // "size", "len(data)"
	Stride     string // "vOne.NumElements()", "lanes", etc.
	UnrollHint int    // Explicit unroll factor from //hwy:unroll directive (0 = auto)
}

// TypeSpecificConst represents a constant with type-specific variants.
// E.g., expLn2Hi_f32 and expLn2Hi_f64 are variants of base name "expLn2Hi".
type TypeSpecificConst struct {
	BaseName string            // e.g., "expLn2Hi"
	Variants map[string]string // map[type_suffix]full_name, e.g., {"f32": "expLn2Hi_f32", "f64": "expLn2Hi_f64"}
}

// ParsedCondition represents a parsed conditional expression.
// Supports type conditions (f32, f64), target conditions (avx2, avx512, neon, fallback),
// and compound conditions with && (AND) and || (OR).
type ParsedCondition struct {
	// For simple conditions (leaf nodes)
	IsType     bool   // true if this is a type condition (f32, f64)
	IsTarget   bool   // true if this is a target condition (avx2, avx512, neon, fallback)
	IsCategory bool   // true if this is a category condition (float, int, uint)
	Value      string // the condition value (e.g., "f64", "avx2", "float")
	Negated    bool   // true if condition is negated (e.g., "!avx2")

	// For compound conditions (internal nodes)
	Op    string           // "" for leaf, "&&" for AND, "||" for OR
	Left  *ParsedCondition // left operand for compound
	Right *ParsedCondition // right operand for compound
}

// ConditionalBlock represents a //hwy:if ... //hwy:endif block.
type ConditionalBlock struct {
	Condition       string           // Original condition string (e.g., "f64 && avx2")
	ParsedCondition *ParsedCondition // Parsed condition tree
	StartLine       int              // Line number of //hwy:if
	ElseLine        int              // Line number of //hwy:else (0 if no else)
	EndLine         int              // Line number of //hwy:endif
}

// ParseResult contains all parsed information from a source file.
type ParseResult struct {
	Funcs              []ParsedFunc
	PackageName        string
	TypeSpecificConsts map[string]*TypeSpecificConst // map[base_name]variants
	ConditionalBlocks  []ConditionalBlock
	FileSet            *token.FileSet    // For converting positions to line numbers
	Imports            map[string]string // map[local_name]import_path (e.g., "math" -> "math", "stdmath" -> "math")
}

// Parse parses a Go source file and extracts functions with hwy operations.
func Parse(filename string) (*ParseResult, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse file: %w", err)
	}

	result := &ParseResult{
		PackageName:        file.Name.Name,
		TypeSpecificConsts: make(map[string]*TypeSpecificConst),
		FileSet:            fset,
		Imports:            make(map[string]string),
	}

	// Extract imports to map local names to import paths
	for _, imp := range file.Imports {
		// Get the import path (remove quotes)
		importPath := strings.Trim(imp.Path.Value, `"`)

		// Determine the local name used in code
		var localName string
		if imp.Name != nil {
			// Explicit alias: import foo "path/to/bar"
			localName = imp.Name.Name
		} else {
			// Default package name: "math" -> "math"; "path/to/foo" -> "foo"
			localName = filepath.Base(importPath)
		}

		// Skip blank imports (import _ "pkg")
		if localName != "_" {
			result.Imports[localName] = importPath
		}
	}

	// First pass: collect type-specific constants from var declarations
	for _, decl := range file.Decls {
		genDecl, ok := decl.(*ast.GenDecl)
		if !ok || (genDecl.Tok != token.VAR && genDecl.Tok != token.CONST) {
			continue
		}

		for _, spec := range genDecl.Specs {
			valueSpec, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}

			for _, name := range valueSpec.Names {
				parseTypeSpecificConst(name.Name, result.TypeSpecificConsts)
			}
		}
	}

	// Parse conditional directives from comments
	result.ConditionalBlocks = parseConditionalDirectives(file, fset)

	// Parse unroll directives from comments
	unrollDirectives := parseUnrollDirectives(file, fset)

	for _, decl := range file.Decls {
		funcDecl, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}

		// Only process functions (not methods) that start with "Base" or "base"
		name := funcDecl.Name.Name
		isExportedBase := strings.HasPrefix(name, "Base")
		isPrivateBase := !isExportedBase && strings.HasPrefix(name, "base")
		if funcDecl.Recv != nil || (!isExportedBase && !isPrivateBase) {
			continue
		}

		pf := ParsedFunc{
			Name:    name,
			Body:    funcDecl.Body,
			Doc:     funcDecl.Doc,
			Private: isPrivateBase,
		}

		// Extract type parameters
		if funcDecl.Type.TypeParams != nil {
			for _, field := range funcDecl.Type.TypeParams.List {
				for _, name := range field.Names {
					constraint := exprToString(field.Type)
					pf.TypeParams = append(pf.TypeParams, TypeParam{
						Name:       name.Name,
						Constraint: constraint,
					})
				}
			}
		}

		// Extract parameters
		if funcDecl.Type.Params != nil {
			for _, field := range funcDecl.Type.Params.List {
				typeStr := exprToString(field.Type)
				if len(field.Names) == 0 {
					// Unnamed parameter
					pf.Params = append(pf.Params, Param{Name: "", Type: typeStr})
				} else {
					for _, name := range field.Names {
						pf.Params = append(pf.Params, Param{Name: name.Name, Type: typeStr})
					}
				}
			}
		}

		// Extract return values
		if funcDecl.Type.Results != nil {
			for _, field := range funcDecl.Type.Results.List {
				typeStr := exprToString(field.Type)
				if len(field.Names) == 0 {
					pf.Returns = append(pf.Returns, Param{Name: "", Type: typeStr})
				} else {
					for _, name := range field.Names {
						pf.Returns = append(pf.Returns, Param{Name: name.Name, Type: typeStr})
					}
				}
			}
		}

		// Find hwy.* and contrib.* calls
		pf.HwyCalls = findHwyCalls(funcDecl.Body)

		// Detect main vectorized loop (with unroll directive support)
		pf.LoopInfo = detectLoopWithUnroll(funcDecl.Body, fset, unrollDirectives)

		// Include functions that use hwy operations OR have hwy.Lanes type parameters
		// (generic functions with hwy.Lanes need type specialization even without hwy ops)
		hasHwyLanesTypeParam := hasHwyLanesConstraint(pf.TypeParams)
		if len(pf.HwyCalls) > 0 || hasHwyLanesTypeParam {
			result.Funcs = append(result.Funcs, pf)
		}
	}

	return result, nil
}

// hasBasePrefix returns true if the name starts with "Base" or "base".
func hasBasePrefix(name string) bool {
	return strings.HasPrefix(name, "Base") || strings.HasPrefix(name, "base")
}

// findHwyCalls walks the AST and finds all hwy.* and contrib.* calls and references.
// Also detects calls to Base*/base* functions within the same package.
func findHwyCalls(node ast.Node) []HwyCall {
	var calls []HwyCall
	seen := make(map[string]bool) // Avoid duplicates

	ast.Inspect(node, func(n ast.Node) bool {
		var selExpr *ast.SelectorExpr
		var pos token.Pos

		switch expr := n.(type) {
		case *ast.CallExpr:
			// Check for same-package Base*/base* function calls
			if ident, ok := expr.Fun.(*ast.Ident); ok {
				if hasBasePrefix(ident.Name) {
					key := "local." + ident.Name
					if !seen[key] {
						seen[key] = true
						calls = append(calls, HwyCall{
							Package:  "local",
							FuncName: ident.Name,
							Position: expr.Pos(),
						})
					}
				}
			}
			// Check for same-package generic Base*/base* function calls: BaseApply[T](...) / baseApply[T](...)
			if indexExpr, ok := expr.Fun.(*ast.IndexExpr); ok {
				if ident, ok := indexExpr.X.(*ast.Ident); ok {
					if hasBasePrefix(ident.Name) {
						key := "local." + ident.Name
						if !seen[key] {
							seen[key] = true
							calls = append(calls, HwyCall{
								Package:  "local",
								FuncName: ident.Name,
								Position: expr.Pos(),
							})
						}
					}
				}
			}
			pos = expr.Pos()
			// Check for hwy.Function or contrib.Function
			// Also handle generic calls like hwy.Pow2[T]()
			switch fun := expr.Fun.(type) {
			case *ast.SelectorExpr:
				// Direct call: hwy.Load(...)
				selExpr = fun
			case *ast.IndexExpr:
				// Generic call: hwy.Pow2[T](...)
				if sel, ok := fun.X.(*ast.SelectorExpr); ok {
					selExpr = sel
				}
			case *ast.IndexListExpr:
				// Generic call with multiple type params: hwy.Func[T, U](...)
				if sel, ok := fun.X.(*ast.SelectorExpr); ok {
					selExpr = sel
				}
			}

		case *ast.SelectorExpr:
			// Function reference (not a call): math.BaseExpVec passed as argument
			pos = expr.Pos()
			selExpr = expr

		case *ast.IndexExpr:
			// Generic reference: math.BaseExpVec[T] passed as argument
			pos = expr.Pos()
			if sel, ok := expr.X.(*ast.SelectorExpr); ok {
				selExpr = sel
			}
		}

		if selExpr == nil {
			return true
		}

		ident, ok := selExpr.X.(*ast.Ident)
		if !ok {
			return true
		}

		// Recognize hwy package and contrib subpackages
		// Also recognize "stdmath" as an alias for stdlib math package
		switch ident.Name {
		case "hwy", "contrib", "math", "vec", "matvec", "matmul", "algo", "image", "bitpack", "sort", "stdmath":
			key := ident.Name + "." + selExpr.Sel.Name
			if !seen[key] {
				seen[key] = true
				calls = append(calls, HwyCall{
					Package:  ident.Name,
					FuncName: selExpr.Sel.Name,
					Position: pos,
				})
			}
		}

		return true
	})

	return calls
}

// UnrollDirective represents a parsed //hwy:unroll directive.
type UnrollDirective struct {
	Line   int // Line number of the directive
	Factor int // Unroll factor (1 = no unroll, 0 = disable)
}

// parseUnrollDirectives parses //hwy:unroll N comments from the file.
func parseUnrollDirectives(file *ast.File, fset *token.FileSet) []UnrollDirective {
	var directives []UnrollDirective

	for _, cg := range file.Comments {
		for _, c := range cg.List {
			text := strings.TrimSpace(strings.TrimPrefix(c.Text, "//"))
			line := fset.Position(c.Pos()).Line

			if after, ok := strings.CutPrefix(text, "hwy:unroll "); ok {
				factor := 0
				if _, err := fmt.Sscanf(after, "%d", &factor); err == nil {
					directives = append(directives, UnrollDirective{
						Line:   line,
						Factor: factor,
					})
				}
			}
		}
	}

	return directives
}

// detectLoopWithUnroll attempts to find the main vectorized loop pattern
// and also checks for //hwy:unroll directives.
// Looks for: for ii := 0; ii < size; ii += stride
// Skips auxiliary loops that only contain Store operations (like zeroing loops).
func detectLoopWithUnroll(body *ast.BlockStmt, fset *token.FileSet, unrollDirectives []UnrollDirective) *LoopInfo {
	if body == nil {
		return nil
	}

	for _, stmt := range body.List {
		forStmt, ok := stmt.(*ast.ForStmt)
		if !ok {
			continue
		}

		info := &LoopInfo{}

		// Parse init: ii := 0
		if forStmt.Init != nil {
			if assignStmt, ok := forStmt.Init.(*ast.AssignStmt); ok {
				if len(assignStmt.Lhs) == 1 && len(assignStmt.Rhs) == 1 {
					if ident, ok := assignStmt.Lhs[0].(*ast.Ident); ok {
						info.Iterator = ident.Name
					}
					info.Start = exprToString(assignStmt.Rhs[0])
				}
			}
		}

		// If no init (iterator declared before loop), try to get iterator from Post
		// Pattern: for ; i+lanes <= n; i += lanes
		if info.Iterator == "" && forStmt.Post != nil {
			if assignStmt, ok := forStmt.Post.(*ast.AssignStmt); ok {
				if len(assignStmt.Lhs) == 1 {
					if ident, ok := assignStmt.Lhs[0].(*ast.Ident); ok {
						info.Iterator = ident.Name
					}
				}
			}
		}

		// Parse condition: ii < size or ii+lanes <= size
		if forStmt.Cond != nil {
			if binExpr, ok := forStmt.Cond.(*ast.BinaryExpr); ok {
				switch binExpr.Op {
				case token.LSS:
					// ii < size
					info.End = exprToString(binExpr.Y)
				case token.LEQ:
					// ii+lanes <= size - extract size from RHS
					info.End = exprToString(binExpr.Y)
				}
			}
		}

		// Parse post: ii += stride
		if forStmt.Post != nil {
			if assignStmt, ok := forStmt.Post.(*ast.AssignStmt); ok {
				if len(assignStmt.Rhs) == 1 {
					info.Stride = exprToString(assignStmt.Rhs[0])
				}
			}
		}

		// Only consider it a SIMD loop if it strides by lanes/NumLanes
		isSimdLoop := strings.Contains(info.Stride, "lanes") ||
			strings.Contains(info.Stride, "NumLanes") ||
			strings.Contains(info.Stride, "NumElements")

		// Skip auxiliary loops that only contain Store operations (like zeroing loops).
		// These don't need tail handling added - they handle their own tails.
		if isAuxiliaryLoop(forStmt.Body) {
			continue
		}

		if info.Iterator != "" && info.End != "" && isSimdLoop {
			// Check for //hwy:unroll directive on the line before the loop
			if fset != nil {
				loopLine := fset.Position(forStmt.Pos()).Line
				for _, ud := range unrollDirectives {
					// Directive should be on the line immediately before the loop
					if ud.Line == loopLine-1 || ud.Line == loopLine-2 {
						info.UnrollHint = ud.Factor
						break
					}
				}
			}
			return info
		}
	}

	return nil
}

// isAuxiliaryLoop checks if a loop body only contains Store operations without Load or arithmetic.
// Such loops (like zeroing loops) are auxiliary and shouldn't trigger tail handling.
func isAuxiliaryLoop(body *ast.BlockStmt) bool {
	if body == nil {
		return false
	}

	hasStore := false
	hasLoad := false
	hasArithmetic := false

	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}

		// Check for hwy.Store, .Store, .StoreSlice
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
			name := sel.Sel.Name
			if name == "Store" || name == "StoreSlice" {
				hasStore = true
			}
			if name == "Load" || name == "LoadSlice" {
				hasLoad = true
			}
			// Arithmetic operations
			if name == "Add" || name == "Sub" || name == "Mul" || name == "Div" ||
				name == "MulAdd" || name == "MulSub" {
				hasArithmetic = true
			}
		}

		// Check for hwy.Store(v, dst), hwy.Load(src)
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
			if ident, ok := sel.X.(*ast.Ident); ok && ident.Name == "hwy" {
				name := sel.Sel.Name
				if name == "Store" || name == "MaskStore" {
					hasStore = true
				}
				if name == "Load" || name == "MaskLoad" {
					hasLoad = true
				}
				if name == "Add" || name == "Sub" || name == "Mul" || name == "Div" ||
					name == "MulAdd" || name == "MulSub" {
					hasArithmetic = true
				}
			}
		}

		return true
	})

	// A loop is auxiliary if it only has Store (no Load or arithmetic)
	return hasStore && !hasLoad && !hasArithmetic
}

// exprToString converts an AST expression to a string representation.
func exprToString(expr ast.Expr) string {
	if expr == nil {
		return ""
	}

	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:
		return exprToString(e.X) + "." + e.Sel.Name
	case *ast.StarExpr:
		return "*" + exprToString(e.X)
	case *ast.ArrayType:
		if e.Len == nil {
			return "[]" + exprToString(e.Elt)
		}
		return "[" + exprToString(e.Len) + "]" + exprToString(e.Elt)
	case *ast.IndexExpr:
		return exprToString(e.X) + "[" + exprToString(e.Index) + "]"
	case *ast.IndexListExpr:
		// Generic type instantiation T[U, V]
		args := make([]string, len(e.Indices))
		for i, idx := range e.Indices {
			args[i] = exprToString(idx)
		}
		return exprToString(e.X) + "[" + strings.Join(args, ", ") + "]"
	case *ast.CallExpr:
		args := make([]string, len(e.Args))
		for i, arg := range e.Args {
			args[i] = exprToString(arg)
		}
		return exprToString(e.Fun) + "(" + strings.Join(args, ", ") + ")"
	case *ast.BasicLit:
		return e.Value
	case *ast.BinaryExpr:
		return exprToString(e.X) + " " + e.Op.String() + " " + exprToString(e.Y)
	case *ast.ParenExpr:
		return "(" + exprToString(e.X) + ")"
	case *ast.InterfaceType:
		// For constraints like "hwy.Floats | hwy.Integers"
		if len(e.Methods.List) == 0 {
			return "interface{}"
		}
		// Just return a simplified version
		return "constraint"
	case *ast.FuncType:
		// Handle function types like func(hwy.Vec[T]) hwy.Vec[T]
		var params []string
		if e.Params != nil {
			for _, field := range e.Params.List {
				paramType := exprToString(field.Type)
				if len(field.Names) == 0 {
					params = append(params, paramType)
				} else {
					for range field.Names {
						params = append(params, paramType)
					}
				}
			}
		}
		var results []string
		if e.Results != nil {
			for _, field := range e.Results.List {
				resultType := exprToString(field.Type)
				if len(field.Names) == 0 {
					results = append(results, resultType)
				} else {
					for range field.Names {
						results = append(results, resultType)
					}
				}
			}
		}
		// Build function type string
		funcStr := "func(" + strings.Join(params, ", ") + ")"
		if len(results) == 1 {
			funcStr += " " + results[0]
		} else if len(results) > 1 {
			funcStr += " (" + strings.Join(results, ", ") + ")"
		}
		return funcStr
	default:
		return fmt.Sprintf("%T", expr)
	}
}

// GetConcreteTypes returns the concrete types that should be generated
// from a generic type parameter constraint.
// Handles union constraints like "hwy.Integers | hwy.FloatsNative" by returning
// all applicable types.
func GetConcreteTypes(constraint string) []string {
	// Handle union constraints by collecting all applicable types
	var types []string
	seen := make(map[string]bool)

	addTypes := func(newTypes []string) {
		for _, t := range newTypes {
			if !seen[t] {
				seen[t] = true
				types = append(types, t)
			}
		}
	}

	// Check for Lanes first (covers all types)
	if strings.Contains(constraint, "Lanes") {
		return []string{"float32", "float64", "int32", "int64", "uint32", "uint64"}
	}

	// Check for float constraints
	if strings.Contains(constraint, "FloatsNative") {
		// FloatsNative is float32/float64 only (no Float16/BFloat16)
		addTypes([]string{"float32", "float64"})
	} else if strings.Contains(constraint, "Floats") {
		// Full Floats constraint includes Float16/BFloat16
		addTypes([]string{"hwy.Float16", "hwy.BFloat16", "float32", "float64"})
	}

	// Check for integer constraints
	if strings.Contains(constraint, "SignedInts") {
		addTypes([]string{"int32", "int64"})
	}
	if strings.Contains(constraint, "UnsignedInts") {
		addTypes([]string{"uint32", "uint64"})
	}
	// Integers covers both signed and unsigned (but only if not already added)
	if strings.Contains(constraint, "Integers") && !strings.Contains(constraint, "SignedInts") && !strings.Contains(constraint, "UnsignedInts") {
		addTypes([]string{"int32", "int64", "uint32", "uint64"})
	}

	// If no types were added, default to common types
	if len(types) == 0 {
		return []string{"float32", "float64"}
	}

	return types
}

// typeSuffixes are recognized type suffixes for type-specific constants.
var typeSuffixes = []string{"_f16", "_bf16", "_f32", "_f64", "_i32", "_i64", "_u32", "_u64"}

// parseTypeSpecificConst checks if a variable name has a type suffix and registers it.
// E.g., "expLn2Hi_f32" -> base name "expLn2Hi", suffix "f32"
func parseTypeSpecificConst(name string, consts map[string]*TypeSpecificConst) {
	for _, suffix := range typeSuffixes {
		if before, ok := strings.CutSuffix(name, suffix); ok {
			baseName := before
			typeSuffix := suffix[1:] // Remove leading underscore

			if consts[baseName] == nil {
				consts[baseName] = &TypeSpecificConst{
					BaseName: baseName,
					Variants: make(map[string]string),
				}
			}
			consts[baseName].Variants[typeSuffix] = name
			return
		}
	}
}

// parseConditionalDirectives parses //hwy:if, //hwy:else, //hwy:endif comments.
func parseConditionalDirectives(file *ast.File, fset *token.FileSet) []ConditionalBlock {
	var blocks []ConditionalBlock
	var stack []*ConditionalBlock // Stack for nested blocks

	for _, cg := range file.Comments {
		for _, c := range cg.List {
			text := strings.TrimSpace(strings.TrimPrefix(c.Text, "//"))
			line := fset.Position(c.Pos()).Line

			if after, ok := strings.CutPrefix(text, "hwy:if "); ok {
				condition := strings.TrimSpace(after)
				block := &ConditionalBlock{
					Condition:       condition,
					ParsedCondition: parseCondition(condition),
					StartLine:       line,
				}
				stack = append(stack, block)
			} else if text == "hwy:else" {
				if len(stack) > 0 {
					stack[len(stack)-1].ElseLine = line
				}
			} else if text == "hwy:endif" {
				if len(stack) > 0 {
					block := stack[len(stack)-1]
					block.EndLine = line
					blocks = append(blocks, *block)
					stack = stack[:len(stack)-1]
				}
			}
		}
	}

	return blocks
}

// typeConditions are valid type condition values
var typeConditions = map[string]bool{
	"f16": true, "bf16": true, "f32": true, "f64": true,
	"i8": true, "i16": true, "i32": true, "i64": true,
	"u8": true, "u16": true, "u32": true, "u64": true,
}

// targetConditions are valid target condition values
var targetConditions = map[string]bool{
	"avx2": true, "avx512": true, "neon": true, "fallback": true,
}

// categoryConditions are type category predicates
var categoryConditions = map[string]bool{
	"float": true, // matches f16, bf16, f32, f64
	"int":   true, // matches i8, i16, i32, i64
	"uint":  true, // matches u8, u16, u32, u64
}

// categoryTypes maps category names to their member type suffixes
var categoryTypes = map[string][]string{
	"float": {"f16", "bf16", "f32", "f64"},
	"int":   {"i8", "i16", "i32", "i64"},
	"uint":  {"u8", "u16", "u32", "u64"},
}

// parseCondition parses a condition string into a ParsedCondition tree.
// Supports: simple conditions (f64, avx2), negation (!avx2),
// AND (f64 && avx2), and OR (f32 || f64).
// AND has higher precedence than OR.
func parseCondition(cond string) *ParsedCondition {
	cond = strings.TrimSpace(cond)
	if cond == "" {
		return nil
	}

	// Handle OR (lowest precedence) - split on ||
	if parts := splitOnOperator(cond, "||"); len(parts) > 1 {
		result := parseCondition(parts[0])
		for i := 1; i < len(parts); i++ {
			result = &ParsedCondition{
				Op:    "||",
				Left:  result,
				Right: parseCondition(parts[i]),
			}
		}
		return result
	}

	// Handle AND (higher precedence) - split on &&
	if parts := splitOnOperator(cond, "&&"); len(parts) > 1 {
		result := parseCondition(parts[0])
		for i := 1; i < len(parts); i++ {
			result = &ParsedCondition{
				Op:    "&&",
				Left:  result,
				Right: parseCondition(parts[i]),
			}
		}
		return result
	}

	// Handle parentheses
	if strings.HasPrefix(cond, "(") && strings.HasSuffix(cond, ")") {
		return parseCondition(cond[1 : len(cond)-1])
	}

	// Handle negation
	negated := false
	if strings.HasPrefix(cond, "!") {
		negated = true
		cond = strings.TrimSpace(cond[1:])
	}

	// Simple condition (leaf node)
	pc := &ParsedCondition{
		Value:   cond,
		Negated: negated,
	}

	if typeConditions[cond] {
		pc.IsType = true
	} else if targetConditions[cond] {
		pc.IsTarget = true
	} else if categoryConditions[cond] {
		pc.IsCategory = true
	}

	return pc
}

// splitOnOperator splits a condition string on an operator, respecting parentheses.
func splitOnOperator(cond, op string) []string {
	var parts []string
	depth := 0
	start := 0

	for i := 0; i < len(cond); i++ {
		switch cond[i] {
		case '(':
			depth++
		case ')':
			depth--
		default:
			if depth == 0 && i+len(op) <= len(cond) && cond[i:i+len(op)] == op {
				parts = append(parts, strings.TrimSpace(cond[start:i]))
				start = i + len(op)
				i += len(op) - 1
			}
		}
	}

	if start < len(cond) {
		parts = append(parts, strings.TrimSpace(cond[start:]))
	}

	return parts
}

// Evaluate evaluates a ParsedCondition against a target and element type.
// targetName is like "AVX2", "AVX512", "NEON", "Fallback"
// elemType is like "float32", "float64"
func (pc *ParsedCondition) Evaluate(targetName, elemType string) bool {
	if pc == nil {
		return true // No condition = always true
	}

	// Handle compound conditions
	if pc.Op == "&&" {
		return pc.Left.Evaluate(targetName, elemType) && pc.Right.Evaluate(targetName, elemType)
	}
	if pc.Op == "||" {
		return pc.Left.Evaluate(targetName, elemType) || pc.Right.Evaluate(targetName, elemType)
	}

	// Handle simple conditions
	var result bool
	if pc.IsType {
		typeSuffix := GetTypeSuffix(elemType)
		result = (pc.Value == typeSuffix)
	} else if pc.IsTarget {
		// Normalize target name for comparison (case-insensitive)
		normalizedTarget := strings.ToLower(targetName)
		normalizedValue := strings.ToLower(pc.Value)
		result = (normalizedTarget == normalizedValue)
	} else if pc.IsCategory {
		// Check if the element type belongs to this category
		typeSuffix := GetTypeSuffix(elemType)
		if types, ok := categoryTypes[pc.Value]; ok {
			if slices.Contains(types, typeSuffix) {
				result = true
			}
		}
	} else {
		// Unknown condition type - try both
		typeSuffix := GetTypeSuffix(elemType)
		normalizedTarget := strings.ToLower(targetName)
		normalizedValue := strings.ToLower(pc.Value)
		result = (pc.Value == typeSuffix) || (normalizedTarget == normalizedValue)
	}

	if pc.Negated {
		return !result
	}
	return result
}

// GetTypeSuffix returns the type suffix for a given element type.
// E.g., "float32" -> "f32", "float64" -> "f64"
func GetTypeSuffix(elemType string) string {
	switch elemType {
	case "float16", "hwy.Float16", "Float16":
		return "f16"
	case "bfloat16", "hwy.BFloat16", "BFloat16":
		return "bf16"
	case "float32":
		return "f32"
	case "float64":
		return "f64"
	case "int8":
		return "i8"
	case "int16":
		return "i16"
	case "int32":
		return "i32"
	case "int64":
		return "i64"
	case "uint8":
		return "u8"
	case "uint16":
		return "u16"
	case "uint32":
		return "u32"
	case "uint64":
		return "u64"
	default:
		return "f32"
	}
}
