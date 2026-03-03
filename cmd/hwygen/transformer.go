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
	"slices"
	"strconv"
	"strings"
)



// TransformResult contains the transformed function and any hoisted constants.
type TransformResult struct {
	FuncDecl      *ast.FuncDecl
	HoistedConsts []HoistedConst
}

// HoistedConst represents a constant that was hoisted from a function to package level.
type HoistedConst struct {
	VarName   string // Package-level var name (e.g., "BaseSigmoid_one_f32")
	Value     string // The constant value (e.g., "1.0")
	VecType   string // Vector type (e.g., "Float32x8")
	Broadcast string // Broadcast function (e.g., "archsimd.BroadcastFloat32x8")
}

// TransformOptions contains additional context for transformation.
type TransformOptions struct {
	TypeSpecificConsts map[string]*TypeSpecificConst
	ConditionalBlocks  []ConditionalBlock
	FileSet            *token.FileSet         // For resolving line numbers in conditional blocks
	Imports            map[string]string      // map[local_name]import_path for resolving package references
	AllFuncs           map[string]*ParsedFunc // All functions in file for inlining helpers
	SkipHalfPrecNEON   bool                   // Skip NEON asm specialization for this half-precision function
	TypeMap            map[string]string      // Per-type-param concrete types (from //hwy:gen); nil for single-type functions
	PackageConsts      []PackageConst         // Package-level constants from sibling files for deterministic hoisting
	PackageConstsMap   map[string]bool        // Pre-built lookup; if set, takes precedence over PackageConsts
}

// Transform transforms a parsed function for a specific target and element type.
// It clones the AST, specializes generics, and transforms hwy operations.
func Transform(pf *ParsedFunc, target Target, elemType string) *TransformResult {
	return TransformWithOptions(pf, target, elemType, nil)
}

// TransformWithOptions transforms a parsed function with additional options.
func TransformWithOptions(pf *ParsedFunc, target Target, elemType string, opts *TransformOptions) *TransformResult {
	if opts == nil {
		opts = &TransformOptions{}
	}

	// First, filter the original body based on conditional blocks.
	// We need to do this BEFORE cloning because the original AST has valid positions.
	var filteredBody *ast.BlockStmt
	if len(opts.ConditionalBlocks) > 0 && opts.FileSet != nil {
		filteredBody = filterConditionalBlocks(pf.Body, opts.ConditionalBlocks, opts.FileSet, target.Name, elemType)
	} else {
		filteredBody = pf.Body
	}

	// Create new function declaration (don't copy Doc - emitter handles comments)
	funcDecl := &ast.FuncDecl{
		Name: ast.NewIdent(pf.Name + target.Suffix()),
		Type: &ast.FuncType{
			Params:  &ast.FieldList{},
			Results: pf.buildResultsWithTarget(elemType, target, opts.SkipHalfPrecNEON, opts.TypeMap),
		},
		Body: cloneBlockStmt(filteredBody),
	}

	// Build parameter list with specialized types
	for _, param := range pf.Params {
		paramType := specializeTypeWithMap(param.Type, pf.TypeParams, elemType, opts.TypeMap)
		// Also transform hwy.Vec[T] to concrete vector types for SIMD targets
		// Extract the element type for this specific parameter's Vec type
		vecElemType := extractVecElemType(paramType, elemType)
		paramType = specializeVecType(paramType, vecElemType, target, opts.SkipHalfPrecNEON)
		field := &ast.Field{
			Names: []*ast.Ident{ast.NewIdent(param.Name)},
			Type:  parseTypeExpr(paramType),
		}
		funcDecl.Type.Params.List = append(funcDecl.Type.Params.List, field)
	}

	// For fallback target with predicate functions, generate scalar loop body
	// to avoid allocations from hwy.Load/pred.Apply
	if target.IsFallback() && hasPredicateParam(pf) {
		if scalarBody := generateScalarPredicateBody(pf, elemType); scalarBody != nil {
			funcDecl.Body = scalarBody
			return &TransformResult{
				FuncDecl:      funcDecl,
				HoistedConsts: nil,
			}
		}
	}

	// Build package-level constant lookup map for deterministic hoisting
	packageConsts := opts.PackageConstsMap
	if packageConsts == nil {
		packageConsts = make(map[string]bool, len(opts.PackageConsts))
		for _, pc := range opts.PackageConsts {
			packageConsts[pc.Name] = true
		}
	}

	// Transform the function body
	ctx := &transformContext{
		target:                  target,
		elemType:                elemType,
		typeParams:              pf.TypeParams,
		typeMap:                 opts.TypeMap,
		loopInfo:                pf.LoopInfo,
		lanesVars:               make(map[string]bool),
		localVars:               make(map[string]bool),
		stackArrayVars:          make(map[string]bool),
		hoistedConsts:           make(map[string]HoistedConst),
		funcName:                pf.Name,
		typeSpecificConsts:      opts.TypeSpecificConsts,
		conditionalBlocks:       opts.ConditionalBlocks,
		fset:                    opts.FileSet,
		imports:                 opts.Imports,
		varTypes:                make(map[string]string),
		halfPrecisionSlices:     make(map[string]bool),
		halfPrecisionScalarVars: make(map[string]bool),
		varVecLanes:             make(map[string]int),
		varVecElemType:          make(map[string]string),
		allFuncs:                opts.AllFuncs,
		skipHalfPrecNEON:        opts.SkipHalfPrecNEON,
		packageConsts:           packageConsts,
		isHalfPrec:              isHalfPrecisionType(elemType),
		isAVXPromoted:           isAVXPromotedHalfPrec(target, elemType),
		inPlaceLookup:           buildInPlaceLookup(target.OpMap),
	}

	// Add function parameters to localVars to prevent them from being hoisted
	// Also track half-precision slice and scalar parameters, and parameter types
	for _, param := range pf.Params {
		ctx.localVars[param.Name] = true
		// Track parameter types for type inference (needed for inferring slice element types)
		ctx.varTypes[param.Name] = param.Type
		// Check if parameter is a slice of half-precision type
		if isHalfPrecisionSliceType(param.Type, elemType) {
			ctx.halfPrecisionSlices[param.Name] = true
		}
		// Check if parameter is a scalar half-precision type
		if isHalfPrecisionScalarType(param.Type, elemType) {
			ctx.halfPrecisionScalarVars[param.Name] = true
		}
	}

	// Also track named return values as half-precision scalars
	// For functions like BaseMinMax[T hwy.Floats](v []T) (min, max T),
	// the named return values min and max should be tracked as half-precision scalars
	for _, ret := range pf.Returns {
		if ret.Name != "" && isHalfPrecisionScalarType(ret.Type, elemType) {
			ctx.halfPrecisionScalarVars[ret.Name] = true
		}
	}

	// Collect all locally-defined variable names to avoid hoisting them as constants
	collectLocalVariables(funcDecl.Body, ctx)

	// Pre-scan for Load sizes to determine inferredFuncLanes before processing Set calls.
	// This ensures hoisted constants match the actual vector width used by Load operations.
	if loadSize := findMaxLoadSizeForElemType(funcDecl.Body, elemType); loadSize > 0 {
		ctx.inferredFuncLanes = loadSize
	}

	// Inline local helper function calls before main transformation.
	// This ensures helper bodies get specialized for the target/elemType.
	inlineHelperCalls(funcDecl.Body, ctx)

	// Resolve type-specific constant references
	// Pattern 1: expC0 -> expC0_f32 (base name lookup)
	// Pattern 2: expC0_f32 -> expC0_f64 (suffix swapping for compilable base files)
	transformIdentifiers(funcDecl.Body, ctx)

	transformNode(funcDecl.Body, ctx)

	// Post-process: scalarize fallback functions that only use simple ops.
	// This converts hwy.Vec operations to pure scalar Go code for better performance
	// by eliminating the allocation overhead of 1-element Vec wrappers.
	// NOTE: Don't scalarize Float16/BFloat16 functions - their arithmetic operators
	// do integer math (since they're uint16 under the hood), which produces wrong results.
	// The non-scalarized path uses transformHalfPrecisionFallback to fix this.
	wasScalarized := false
	if target.IsFallback() && !isHalfPrecisionType(elemType) {
		if canScalarizeFallback(funcDecl) {
			scalarizeFallback(funcDecl, elemType)
			wasScalarized = true
		}
	}

	// Post-process: convert "_ = expr" assignments to expression statements.
	// This is needed because tryTransformToInPlace marks in-place ops with _ = voidFunc()
	// which is invalid Go when the function returns nothing (e.g., MulAddAcc).
	if target.IsNEON() {
		convertBlankAssignToExprStmt(funcDecl.Body)
	}

	// Post-process to replace NumLanes() calls and ReduceSum() calls
	if target.Name != "Fallback" {
		postProcessSIMD(funcDecl.Body, ctx)
	}

	// Post-process to convert stack array usages to slice expressions
	if target.Name != "Fallback" && len(ctx.stackArrayVars) > 0 {
		convertStackArrayUsages(funcDecl.Body, ctx)
	}

	// Post-process to transform scalar operations for Float16/BFloat16.
	// Scalar Go operations (+, -, *, /, >, <, etc.) don't work on Float16/BFloat16
	// (they're uint16 under the hood), so we convert to float32 for computation.
	// This applies to all targets (Fallback, NEON, AVX2, AVX512) since scalar tail
	// loops exist in all targets.
	// Skip Vec-returning functions - they use hwy.Vec operations which already work.
	// Skip scalarized fallback functions - they just copy values, no arithmetic needed.
	if isHalfPrecisionType(elemType) && !returnsVecType(pf.Returns) && !wasScalarized {
		transformHalfPrecisionFallback(funcDecl.Body, ctx)
	}

	// Apply loop unrolling if there's a SIMD loop (not for fallback)
	if pf.LoopInfo != nil && target.Name != "Fallback" {
		lanes := target.LanesFor(elemType)
		unrollFactor := computeUnrollFactor(pf.LoopInfo, pf.HwyCalls, target)
		if unrollFactor > 1 {
			// Find the main SIMD loop and unroll it
			if mainLoop := findMainSimdLoop(funcDecl.Body, pf.LoopInfo); mainLoop != nil {
				unrollLoopWithCleanup(funcDecl.Body, mainLoop, pf.LoopInfo, unrollFactor, lanes)
			}
		}
	}

	// Insert tail handling if there's a loop and function doesn't return a value
	// (functions that return values have their own tail handling in the template)
	if pf.LoopInfo != nil && len(pf.Returns) == 0 {
		insertTailHandling(funcDecl.Body, pf.LoopInfo, elemType, target, pf.Name, pf.Params, pf.TypeParams)
	}

	// Collect hoisted constants in deterministic order
	var hoisted []HoistedConst
	keys := make([]string, 0, len(ctx.hoistedConsts))
	for k := range ctx.hoistedConsts {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		hoisted = append(hoisted, ctx.hoistedConsts[k])
	}

	return &TransformResult{
		FuncDecl:      funcDecl,
		HoistedConsts: hoisted,
	}
}

// transformContext stores context information for a function transformation process.
type transformContext struct {
	target                  Target
	elemType                string
	typeParams              []TypeParam
	typeMap                 map[string]string             // Per-type-param concrete types (from //hwy:gen); nil for single-type functions
	lanesVars               map[string]bool               // Variables assigned from NumLanes()
	localVars               map[string]bool               // Variables defined locally in the function
	stackArrayVars          map[string]bool               // Variables that are stack arrays (need [:] when used as slice)
	loopInfo                *LoopInfo
	hoistedConsts           map[string]HoistedConst       // Hoisted constants (key is local var name)
	funcName                string                        // Current function name for generating unique hoisted names
	typeSpecificConsts      map[string]*TypeSpecificConst // Type-specific constant registry
	conditionalBlocks       []ConditionalBlock            // Conditional blocks to process
	fset                    *token.FileSet                // For resolving line numbers
	imports                 map[string]string             // map[local_name]import_path for resolving package references
	varTypes                map[string]string             // map[var_name]type for type inference (e.g., "int32", "hwy.Float16")
	halfPrecisionScalarVars map[string]bool               // Variables assigned from half-precision slice reads
	halfPrecisionSlices     map[string]bool               // Slice variables that hold half-precision elements
	varVecLanes             map[string]int                // map[var_name]lanes for detected vector sizes from Load
	varVecElemType          map[string]string             // map[var_name]elemType for detected element types from Load
	inferredFuncLanes       int                           // Inferred lane count for function (from first detected Load size)
	allFuncs                map[string]*ParsedFunc        // All functions in file for inlining helpers
	inlineCounter           int                           // Counter for unique variable naming during inlining
	skipHalfPrecNEON        bool                          // Skip NEON asm specialization for half-precision (use generic hwy.Vec[T] path)
	packageConsts           map[string]bool               // Package-level constant/var names for deterministic hoisting
	isHalfPrec              bool                          // Cached isHalfPrecisionType(elemType)
	isAVXPromoted           bool                          // Cached isAVXPromotedHalfPrec(target, elemType)
	inPlaceLookup           map[string]inPlaceEntry       // Reverse lookup: InPlaceOf value → in-place op
}

// inPlaceEntry maps a method name to the in-place operation that replaces it.
type inPlaceEntry struct {
	OpName string
	OpInfo OpInfo
}

// isNEONHalfPrec returns true when targeting NEON with native half-precision
// asm types (not the generic hwy.Vec[T] path).
func (ctx *transformContext) isNEONHalfPrec() bool {
	return ctx.target.IsNEON() && !ctx.skipHalfPrecNEON && ctx.isHalfPrec
}

// clone creates a shallow copy of the context with fresh maps for per-function
// mutable state. Shared state (hoistedConsts, imports, allFuncs, etc.) is
// preserved by reference.
func (ctx *transformContext) clone() *transformContext {
	c := *ctx
	c.lanesVars = make(map[string]bool)
	c.localVars = make(map[string]bool)
	c.stackArrayVars = make(map[string]bool)
	c.varTypes = make(map[string]string)
	c.halfPrecisionSlices = make(map[string]bool)
	c.halfPrecisionScalarVars = make(map[string]bool)
	c.varVecLanes = make(map[string]int)
	c.varVecElemType = make(map[string]string)
	return &c
}

// buildInPlaceLookup creates a reverse lookup from method name to the in-place
// operation that replaces it, for O(1) lookup in tryTransformToInPlace.
func buildInPlaceLookup(opMap map[string]OpInfo) map[string]inPlaceEntry {
	m := make(map[string]inPlaceEntry)
	for opName, opInfo := range opMap {
		if opInfo.InPlaceOf != "" {
			m[opInfo.InPlaceOf] = inPlaceEntry{OpName: opName, OpInfo: opInfo}
		}
	}
	return m
}

// vecLoadInfo contains inferred information from an hwy.Load call.
type vecLoadInfo struct {
	lanes    int    // Number of vector lanes (0 if not detected)
	elemType string // Element type (empty if not detected or same as function's elemType)
}

// inferVecLanesFromLoad checks if an expression is an hwy.Load call with a detectable slice size.
// Returns the number of lanes and element type if detected.
func inferVecLanesFromLoad(expr ast.Expr, ctx *transformContext) vecLoadInfo {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return vecLoadInfo{}
	}

	// Check for hwy.LoadSlice(...) or hwy.Load[T](...) call
	var funcName string
	var explicitElemType string // Explicit type param from hwy.Load[uint8] style calls
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		// hwy.LoadSlice(...)
		pkgIdent, ok := fun.X.(*ast.Ident)
		if !ok || pkgIdent.Name != "hwy" {
			return vecLoadInfo{}
		}
		funcName = fun.Sel.Name
	case *ast.IndexExpr:
		// hwy.Load[T](...) - generic call with explicit type param
		sel, ok := fun.X.(*ast.SelectorExpr)
		if !ok {
			return vecLoadInfo{}
		}
		pkgIdent, ok := sel.X.(*ast.Ident)
		if !ok || pkgIdent.Name != "hwy" {
			return vecLoadInfo{}
		}
		funcName = sel.Sel.Name
		// Extract explicit type parameter
		if typeIdent, ok := fun.Index.(*ast.Ident); ok {
			explicitElemType = typeIdent.Name
		}
	default:
		return vecLoadInfo{}
	}

	if funcName != "LoadSlice" {
		return vecLoadInfo{}
	}

	// Check if we have an argument with a detectable slice size
	if len(call.Args) == 0 {
		return vecLoadInfo{}
	}

	sliceBytes := getSliceSize(call.Args[0])
	if sliceBytes <= 0 {
		return vecLoadInfo{}
	}

	// Determine element type: explicit type param > function's default
	effectiveElemType := ctx.elemType
	if explicitElemType != "" {
		effectiveElemType = explicitElemType
	}

	elemSize := elemTypeSize(effectiveElemType)
	if elemSize <= 0 {
		return vecLoadInfo{}
	}

	// Return explicit element type if provided
	returnElemType := ""
	if explicitElemType != "" {
		returnElemType = explicitElemType
	}

	return vecLoadInfo{
		lanes:    sliceBytes / elemSize,
		elemType: returnElemType,
	}
}

// inferTypeFromExpr analyzes an expression and returns its inferred type.
// Returns "int32" for expressions like hwy.ConvertToInt32(...), hwy.Set[int32](...), etc.
// Returns empty string if type cannot be inferred.
func inferTypeFromExpr(expr ast.Expr, ctx *transformContext) string {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return ""
	}

	// Check for hwy.Set[int32](...) or similar indexed expressions
	if indexExpr, ok := call.Fun.(*ast.IndexExpr); ok {
		if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
			if pkgIdent, ok := sel.X.(*ast.Ident); ok && pkgIdent.Name == "hwy" {
				// Check the type parameter
				if ident, ok := indexExpr.Index.(*ast.Ident); ok {
					if ident.Name == "int32" {
						return "int32"
					}
				}
			}
		}
	}

	// Check for hwy.ConvertToInt32(...) or method call .ConvertToInt32()
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		funcName := sel.Sel.Name
		switch funcName {
		case "ConvertToInt32":
			return "int32"
		case "And", "Or", "Xor", "AndNot", "ShiftLeft", "ShiftRight", "Add", "Sub", "Mul":
			// For bitwise and arithmetic operations, check if BOTH arguments are int32
			// This handles expressions like hwy.And(hwy.Add(kInt, intOne), intThree)
			if len(call.Args) >= 2 {
				arg0IsInt32 := isInt32Expr(call.Args[0], ctx)
				arg1IsInt32 := isInt32Expr(call.Args[1], ctx)
				if arg0IsInt32 && arg1IsInt32 {
					return "int32"
				}
			}
		}
	}

	return ""
}

// isInt32Expr checks if an expression is of int32 type based on tracked variable types.
func isInt32Expr(expr ast.Expr, ctx *transformContext) bool {
	switch e := expr.(type) {
	case *ast.Ident:
		return ctx.varTypes[e.Name] == "int32"
	case *ast.CallExpr:
		// Check if this is a function call that returns int32
		return inferTypeFromExpr(e, ctx) == "int32"
	}
	return false
}

// isComparisonOp returns true if the operation is a comparison operation.
func isComparisonOp(opName string) bool {
	switch opName {
	case "Equal", "Greater", "GreaterThan", "Less", "LessThan",
		"GreaterEqual", "GreaterThanOrEqual", "LessEqual", "LessThanOrEqual":
		return true
	}
	return false
}

// collectLocalVariables walks the AST and collects all locally-defined variable names.
// This is used to exclude local variables from constant hoisting.
func collectLocalVariables(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		switch stmt := n.(type) {
		case *ast.AssignStmt:
			// Collect all LHS identifiers from := and = assignments
			// Only := definitely defines new variables, but we track both to be safe
			if stmt.Tok == token.DEFINE {
				for i, lhs := range stmt.Lhs {
					if ident, ok := lhs.(*ast.Ident); ok {
						ctx.localVars[ident.Name] = true
						// Track variable types for type inference
						if i < len(stmt.Rhs) {
							if inferredType := inferTypeFromExpr(stmt.Rhs[i], ctx); inferredType != "" {
								ctx.varTypes[ident.Name] = inferredType
							}
							// Track vector lanes and element type for variables assigned from Load
							if loadInfo := inferVecLanesFromLoad(stmt.Rhs[i], ctx); loadInfo.lanes > 0 {
								ctx.varVecLanes[ident.Name] = loadInfo.lanes
								if loadInfo.elemType != "" {
									ctx.varVecElemType[ident.Name] = loadInfo.elemType
								}
								// Set function-wide inferred lanes on first detection
								if ctx.inferredFuncLanes == 0 {
									ctx.inferredFuncLanes = loadInfo.lanes
								}
							}
						}
					}
				}
			}
		case *ast.DeclStmt:
			// var declarations
			if genDecl, ok := stmt.Decl.(*ast.GenDecl); ok {
				if genDecl.Tok == token.VAR {
					for _, spec := range genDecl.Specs {
						if valueSpec, ok := spec.(*ast.ValueSpec); ok {
							for _, name := range valueSpec.Names {
								ctx.localVars[name.Name] = true
							}
						}
					}
				}
			}
		case *ast.RangeStmt:
			// for k, v := range ...
			if stmt.Tok == token.DEFINE {
				if ident, ok := stmt.Key.(*ast.Ident); ok && ident.Name != "_" {
					ctx.localVars[ident.Name] = true
				}
				if ident, ok := stmt.Value.(*ast.Ident); ok && ident.Name != "_" {
					ctx.localVars[ident.Name] = true
				}
			}
		case *ast.ForStmt:
			// for i := 0; ... - the init statement
			if stmt.Init != nil {
				if assign, ok := stmt.Init.(*ast.AssignStmt); ok && assign.Tok == token.DEFINE {
					for _, lhs := range assign.Lhs {
						if ident, ok := lhs.(*ast.Ident); ok {
							ctx.localVars[ident.Name] = true
						}
					}
				}
			}
		}
		return true
	})
}

// transformNode recursively transforms AST nodes.
func transformNode(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.CallExpr:
			transformCallExpr(node, ctx)
			// Also check for type conversions like T(1)
			transformTypeConversion(node, ctx)
			// Transform function references passed as arguments
			transformFuncRefArgs(node, ctx)
		case *ast.DeclStmt:
			// Transform variable declarations
			if genDecl, ok := node.Decl.(*ast.GenDecl); ok {
				transformGenDecl(genDecl, ctx)
			}
		case *ast.AssignStmt:
			// Transform assignments
			transformAssignStmt(node, ctx)
		case *ast.ForStmt:
			// Transform for loop for SIMD (stride, condition)
			transformForStmt(node, ctx)
		case *ast.CompositeLit:
			// Transform composite literal types (e.g., [4]hwy.Vec[float32]{} -> [4]asm.Float32x4{})
			transformCompositeLit(node, ctx)
		}
		return true
	})
}

// transformForStmt transforms for loops for SIMD targets.
// Changes: for ii := 0; ii < size; ii += v.NumLanes()
// To:      for ii := 0; ii+8 <= size; ii += 8
// Also handles: for ii := 0; ii < size; ii += lanes (where lanes was from NumLanes())
// Only transforms loops that use NumLanes() stride (not scalar tail loops).
func transformForStmt(stmt *ast.ForStmt, ctx *transformContext) {
	if ctx.target.IsFallback() || ctx.loopInfo == nil {
		return
	}

	lanes := ctx.target.LanesFor(ctx.elemType)

	// Check if this loop uses NumLanes() stride - only transform those loops
	isSimdLoop := false
	if assignStmt, ok := stmt.Post.(*ast.AssignStmt); ok {
		if len(assignStmt.Rhs) == 1 {
			// Case 1: ii += v.NumLanes() - direct call
			if call, ok := assignStmt.Rhs[0].(*ast.CallExpr); ok {
				if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
					if sel.Sel.Name == "NumElements" || sel.Sel.Name == "NumLanes" {
						isSimdLoop = true
						// Transform post: ii += v.NumLanes() -> ii += lanes
						assignStmt.Rhs[0] = &ast.BasicLit{
							Kind:  token.INT,
							Value: strconv.Itoa(lanes),
						}
					}
				}
			}
			// Case 2: ii += lanes - variable assigned from NumLanes()
			if ident, ok := assignStmt.Rhs[0].(*ast.Ident); ok {
				if ctx.lanesVars[ident.Name] {
					isSimdLoop = true
					// The variable was already replaced with a constant in transformAssignStmt,
					// but we still need to transform the loop condition
				}
			}
			// Case 3: ii += 8 (or other constant) - already transformed
			if lit, ok := assignStmt.Rhs[0].(*ast.BasicLit); ok {
				if lit.Kind == token.INT {
					// Check if the value matches our lanes - this means it was already transformed
					if lit.Value == strconv.Itoa(lanes) {
						isSimdLoop = true
					}
				}
			}
		}
	}

	// Only transform condition for SIMD loops (not scalar tail loops)
	if isSimdLoop {
		// Transform condition: ii < size -> ii+lanes <= size
		if binExpr, ok := stmt.Cond.(*ast.BinaryExpr); ok {
			if binExpr.Op == token.LSS {
				// Change ii < size to ii+lanes <= size
				binExpr.Op = token.LEQ
				binExpr.X = &ast.BinaryExpr{
					X:  binExpr.X,
					Op: token.ADD,
					Y: &ast.BasicLit{
						Kind:  token.INT,
						Value: strconv.Itoa(lanes),
					},
				}
			}
		}
	}
}

// transformCompositeLit transforms composite literal types for SIMD targets.
// Converts types like [4]hwy.Vec[float32]{} to [4]asm.Float32x4{} for NEON
// or [8]archsimd.Float32x8{} for AVX2.
func transformCompositeLit(lit *ast.CompositeLit, ctx *transformContext) {
	if lit.Type == nil {
		return
	}

	// For fallback target, don't transform hwy.Vec types
	if ctx.target.IsFallback() {
		return
	}

	// Check if the type is an array with hwy.Vec element type
	arrayType, ok := lit.Type.(*ast.ArrayType)
	if !ok {
		return
	}

	// Transform the element type if it's hwy.Vec[T] or similar
	typeStr := exprToString(arrayType.Elt)

	// First specialize generic type parameters (T -> float32)
	specialized := specializeType(typeStr, ctx.typeParams, ctx.elemType)

	// Then transform hwy.Vec[float32] -> asm.Float32x4 for SIMD targets
	specialized = specializeVecType(specialized, ctx.elemType, ctx.target, ctx.skipHalfPrecNEON)

	if specialized != typeStr {
		arrayType.Elt = parseTypeExpr(specialized)
	}
}

// transformTypeConversion converts T(1) to float32(1) for generic type parameters.
func transformTypeConversion(call *ast.CallExpr, ctx *transformContext) {
	// Check if this is a type conversion T(value) where T is a type parameter
	ident, ok := call.Fun.(*ast.Ident)
	if !ok {
		return
	}

	// Check if the identifier is a type parameter
	for _, tp := range ctx.typeParams {
		if ident.Name == tp.Name {
			// Replace T with the concrete element type
			ident.Name = ctx.elemType
			return
		}
	}
}

// transformCallExpr transforms hwy.* and contrib.* function calls.
func transformCallExpr(call *ast.CallExpr, ctx *transformContext) {
	// First, check for calls to other Base* functions and add target suffix
	// This applies to ALL targets including fallback, since generated functions
	// are always concrete (BaseApply_fallback, not generic BaseApply)
	if ident, ok := call.Fun.(*ast.Ident); ok {
		if strings.HasPrefix(ident.Name, "Base") {
			// Transform BaseFoo to BaseFoo_avx2 (or BaseFoo_fallback, etc.)
			suffix := ctx.target.Suffix()
			// Add type suffix for non-float32 types, but ONLY if the current function
			// has type parameters (indicating it's a generic function with type variants).
			// Concrete functions like BaseEncodeStreamVByte32GroupSIMD([]uint32) don't
			// have type variants, so their internal Base* calls shouldn't add type suffix.
			if len(ctx.typeParams) > 0 {
				suffix += getHwygenTypeSuffix(ctx.elemType)
			}
			ident.Name = ident.Name + suffix
		}
	}

	// Transform Vec method calls like .Store() -> .StoreSlice() for SIMD targets
	// This handles cases like fn(x).Store(dst) where fn returns a Vec
	// Skip this for package-level function calls like hwy.StoreSlice() which are handled later
	if ctx.target.Name != "Fallback" {
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
			// Don't transform package-level function calls
			if ident, ok := sel.X.(*ast.Ident); ok && ident.Name == "hwy" {
				// Skip - this is a package-level function, handled later
			} else {
				switch sel.Sel.Name {
				case "Store":
					// Transform .Store(dst) -> .StoreSlice(dst)
					// But skip if the argument is a pointer type (from Store transformation)
					// Store produces v.Store((*[N]T)(unsafe.Pointer(...))) which should stay as Store
					isPointerArg := false
					if len(call.Args) == 1 {
						if _, isCall := call.Args[0].(*ast.CallExpr); isCall {
							// Argument is a call expression like (*[N]T)(ptr) - likely from Store
							isPointerArg = true
						}
					}
					if !isPointerArg {
						sel.Sel.Name = "StoreSlice"
						// Cast []hwy.Float16/[]hwy.BFloat16 -> []uint16 for half-precision
						if ctx.isHalfPrec && len(call.Args) == 1 {
							call.Args[0] = halfPrecSliceToUint16(call.Args[0])
						}
					}
				case "Data":
					transformDataMethod(call, ctx)
					return
				case "GetBit":
					transformGetBitMethod(call, ctx)
					return
				}
			}
		}
	}

	// Also handle generic Base* calls like BaseFoo[T](x)
	// All targets get the suffix since generated functions are concrete
	if indexExpr, ok := call.Fun.(*ast.IndexExpr); ok {
		if ident, ok := indexExpr.X.(*ast.Ident); ok {
			if strings.HasPrefix(ident.Name, "Base") {
				suffix := ctx.target.Suffix() + getHwygenTypeSuffix(ctx.elemType)
				// Strip the type param and add suffix
				call.Fun = ast.NewIdent(ident.Name + suffix)
			}
		}
	}

	var selExpr *ast.SelectorExpr
	var ok bool
	var hasExplicitTypeParam bool // Track if we have an explicit type param to preserve

	// Handle both regular calls (hwy.Load) and generic calls (hwy.Zero[T])
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		selExpr = fun
	case *ast.IndexExpr:
		// Generic function call like hwy.Zero[T]() or hwy.Load[uint8]()
		// The IndexExpr wraps the SelectorExpr
		selExpr, ok = fun.X.(*ast.SelectorExpr)
		if !ok {
			return
		}
		// Check if the type parameter is a concrete type (not a generic type param like T)
		if typeIdent, ok := fun.Index.(*ast.Ident); ok {
			typeName := typeIdent.Name
			isTypeParam := false
			for _, tp := range ctx.typeParams {
				if typeName == tp.Name {
					isTypeParam = true
					break
				}
			}
			if !isTypeParam {
				// This is an explicit concrete type like uint8, float32, etc.
				// Keep the IndexExpr so transformToFunction can use it
				hasExplicitTypeParam = true
			}
		}
		// Transform hwy.Const[T](val) to hwy.Set(val) for non-float32 types.
		// hwy.Const takes float32, which loses precision for float64 targets.
		// Named constants are suffix-transformed to the correct type by the generator.
		// Literals (like 0.5) are untyped in Go, so Set[float64](0.5) preserves
		// full precision. For half-precision types, literals must stay with Const
		// since Set[Float16](0.5) won't compile (no implicit conversion).
		// Note: We don't return here - let the transformation continue so Set gets
		// transformed to asm.Broadcast* for SIMD targets.
		if selExpr.Sel.Name == "Const" {
			if ident, ok := selExpr.X.(*ast.Ident); ok && ident.Name == "hwy" {
				if ctx.elemType != "float32" && len(call.Args) > 0 {
					// Named constants and binary expressions: always convert to Set
					if _, isIdent := call.Args[0].(*ast.Ident); isIdent {
						selExpr.Sel.Name = "Set"
					} else if _, isBinary := call.Args[0].(*ast.BinaryExpr); isBinary {
						selExpr.Sel.Name = "Set"
					}
					// Literals: convert to Set for native types (float64) to avoid
					// float32 precision truncation. Half-precision types must keep
					// Const because Go can't convert untyped floats to Float16/BFloat16.
					if !ctx.isHalfPrec {
						if _, isLit := call.Args[0].(*ast.BasicLit); isLit {
							selExpr.Sel.Name = "Set"
						}
					}
				}
			}
		}
		if ctx.target.IsFallback() {
			// For fallback, replace type param with concrete type
			// hwy.Zero[T]() -> hwy.Zero[float32]()
			for _, tp := range ctx.typeParams {
				if ident, ok := fun.Index.(*ast.Ident); ok && ident.Name == tp.Name {
					ident.Name = ctx.elemType
				}
			}
			// Keep the IndexExpr (with type param), just update the type
		} else if ctx.isHalfPrec {
			// For Float16/BFloat16 on SIMD targets, keep the type param for functions
			// like Const, Set, Zero that need it for type inference
			funcName := selExpr.Sel.Name
			switch funcName {
			case "Const":
				// For NEON target with asm types, convert to asm.BroadcastFloat16x8/BFloat16x8
				if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON {
					broadcastFuncName := "BroadcastFloat16x8"
					if isBFloat16Type(ctx.elemType) {
						broadcastFuncName = "BroadcastBFloat16x8"
					}
					call.Fun = &ast.SelectorExpr{
						X:   ast.NewIdent("asm"),
						Sel: ast.NewIdent(broadcastFuncName),
					}
					if len(call.Args) > 0 {
						call.Args[0] = wrapConstForHalfPrecBroadcast(call.Args[0], ctx.elemType)
					}
					return
				}
				// For AVX promoted types, convert to asm.BroadcastFloat16x8AVX2(...) etc.
				if ctx.isAVXPromoted {
					typeName := ctx.target.TypeMap[ctx.elemType]
					call.Fun = &ast.SelectorExpr{
						X:   ast.NewIdent("asm"),
						Sel: ast.NewIdent("Broadcast" + typeName),
					}
					if len(call.Args) > 0 {
						call.Args[0] = wrapConstForHalfPrecBroadcast(call.Args[0], ctx.elemType)
					}
					return
				}
				// For Fallback or skip: replace type param with concrete type
				for _, tp := range ctx.typeParams {
					if ident, ok := fun.Index.(*ast.Ident); ok && ident.Name == tp.Name {
						ident.Name = ctx.elemType
					}
				}
			case "Set", "Zero":
				// For NEON without skip, these are handled later in the SelectorExpr path
				// Replace type param with concrete type (e.g., hwy.Set[T] -> hwy.Set[hwy.Float16])
				for _, tp := range ctx.typeParams {
					if ident, ok := fun.Index.(*ast.Ident); ok && ident.Name == tp.Name {
						ident.Name = ctx.elemType
					}
				}
				// Keep the IndexExpr with the concrete type
			case "ConvertExponentToFloat":
				if ctx.isAVXPromoted {
					// For AVX promoted: asm.Float16x8AVX2FromFloat32x8(e.ConvertToFloat32())
					wrapFunc := fmt.Sprintf("%sFromFloat32x%d", ctx.target.TypeMap[ctx.elemType], ctx.target.LanesFor("float32"))
					call.Fun = &ast.SelectorExpr{
						X:   ast.NewIdent("asm"),
						Sel: ast.NewIdent(wrapFunc),
					}
					call.Args = []ast.Expr{
						&ast.CallExpr{
							Fun: &ast.SelectorExpr{
								X:   call.Args[0],
								Sel: ast.NewIdent("ConvertToFloat32"),
							},
						},
					}
					return
				}
				// Transform to non-generic ConvertToF16/ConvertToBF16
				if ctx.elemType == "hwy.Float16" {
					call.Fun = &ast.SelectorExpr{
						X:   ast.NewIdent("hwy"),
						Sel: ast.NewIdent("ConvertToF16"),
					}
				} else {
					call.Fun = &ast.SelectorExpr{
						X:   ast.NewIdent("hwy"),
						Sel: ast.NewIdent("ConvertToBF16"),
					}
				}
				return // Already handled, don't continue transformation
			default:
				// For other functions, strip the type param (will be transformed later)
				call.Fun = selExpr
			}
		} else {
			// For SIMD targets with native types, handle special cases first
			funcName := selExpr.Sel.Name
			switch funcName {
			case "ConvertExponentToFloat":
				// Transform to method call: e.ConvertToFloat32() or e.ConvertToFloat64()
				if len(call.Args) >= 1 {
					var methodName string
					if ctx.elemType == "float64" {
						methodName = "ConvertToFloat64"
					} else {
						methodName = "ConvertToFloat32"
					}
					convertToUnaryMethodCall(call, methodName)
				}
				return
			default:
				// Strip the type param (will be transformed later)
				// BUT preserve IndexExpr if we have an explicit concrete type param
				// (e.g., hwy.Load[uint8]) so transformToFunction can use it
				if !hasExplicitTypeParam {
					call.Fun = selExpr
				}
			}
		}
	case *ast.IndexListExpr:
		// Generic function call with multiple type params like hwy.Func[T, U]()
		selExpr, ok = fun.X.(*ast.SelectorExpr)
		if !ok {
			return
		}
		if ctx.target.IsFallback() {
			// For fallback, replace type params with concrete types
			for i, idx := range fun.Indices {
				if ident, ok := idx.(*ast.Ident); ok {
					for _, tp := range ctx.typeParams {
						if ident.Name == tp.Name {
							fun.Indices[i] = ast.NewIdent(ctx.elemType)
						}
					}
				}
			}
		} else {
			call.Fun = selExpr
		}
	default:
		return
	}

	ident, ok := selExpr.X.(*ast.Ident)
	if !ok {
		return
	}

	// Handle hwy.* and contrib subpackage calls
	switch ident.Name {
	case "hwy", "contrib", "math", "vec", "matvec", "matmul", "algo", "image", "bitpack", "sort":
		// Continue processing
	default:
		return
	}

	funcName := selExpr.Sel.Name

	// Handle cross-package Base* function calls (e.g., algo.BaseApply, math.BaseExpVec)
	// These need target suffix added, similar to same-package Base* calls
	if strings.HasPrefix(funcName, "Base") {
		suffix := ctx.target.Suffix() + getHwygenTypeSuffix(ctx.elemType)
		selExpr.Sel.Name = funcName + suffix
		// Strip the IndexExpr (type parameter) if present, since the
		// target-specific variant is a concrete function, not generic.
		// e.g., math.BaseSigmoidVec[float32](x) -> math.BaseSigmoidVec_neon(x)
		call.Fun = selExpr
		return
	}

	opInfo, ok := ctx.target.OpMap[funcName]
	if !ok {
		// Unknown operation, leave as-is
		return
	}

	// Transform based on operation type
	if opInfo.IsMethod {
		transformToMethod(call, funcName, opInfo, ctx)
	} else {
		transformToFunction(call, funcName, opInfo, ctx)
	}
}

// transformDataMethod transforms v.Data() to a temporary slice.
func transformDataMethod(call *ast.CallExpr, ctx *transformContext) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	vecExpr := sel.X
	lanes := ctx.target.LanesFor(ctx.elemType)

	// func() []T { var tmp [lanes]T; v.StoreSlice(tmp[:]); return tmp[:] }()

	// var tmp [lanes]T
	decl := &ast.DeclStmt{
		Decl: &ast.GenDecl{
			Tok: token.VAR,
			Specs: []ast.Spec{
				&ast.ValueSpec{
					Names: []*ast.Ident{ast.NewIdent("_simd_tmp")},
					Type: &ast.ArrayType{
						Len: &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(lanes)},
						Elt: ast.NewIdent(ctx.elemType),
					},
				},
			},
		},
	}

	// v.StoreSlice(tmp[:]) or hwy.StoreSlice(v, tmp[:]) for half-precision
	var storeCall *ast.CallExpr
	if ctx.isHalfPrec && !ctx.isAVXPromoted {
		// hwy.StoreSlice(v, tmp[:]) for half-precision types on Fallback
		storeFun := &ast.SelectorExpr{
			X:   ast.NewIdent("hwy"),
			Sel: ast.NewIdent("Store"),
		}
		storeCall = &ast.CallExpr{
			Fun: storeFun,
			Args: []ast.Expr{
				cloneExpr(vecExpr),
				&ast.SliceExpr{
					X: ast.NewIdent("_simd_tmp"),
				},
			},
		}
	} else if ctx.isAVXPromoted {
		// v.StoreSlice(cast(tmp[:])) for AVX promoted half-precision
		storeCall = &ast.CallExpr{
			Fun: &ast.SelectorExpr{
				X:   cloneExpr(vecExpr),
				Sel: ast.NewIdent("StoreSlice"),
			},
			Args: []ast.Expr{
				halfPrecSliceToUint16(&ast.SliceExpr{
					X: ast.NewIdent("_simd_tmp"),
				}),
			},
		}
	} else {
		// v.StoreSlice(tmp[:]) for native SIMD types
		storeCall = &ast.CallExpr{
			Fun: &ast.SelectorExpr{
				X:   cloneExpr(vecExpr),
				Sel: ast.NewIdent("StoreSlice"),
			},
			Args: []ast.Expr{
				&ast.SliceExpr{
					X: ast.NewIdent("_simd_tmp"),
				},
			},
		}
	}

	// return tmp[:]
	retStmt := &ast.ReturnStmt{
		Results: []ast.Expr{
			&ast.SliceExpr{
				X: ast.NewIdent("_simd_tmp"),
			},
		},
	}

	// Function literal
	funcLit := &ast.FuncLit{
		Type: &ast.FuncType{
			Results: &ast.FieldList{
				List: []*ast.Field{
					{Type: &ast.ArrayType{Elt: ast.NewIdent(ctx.elemType)}},
				},
			},
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				decl,
				&ast.ExprStmt{X: storeCall},
				retStmt,
			},
		},
	}

	// Replace call with invocation - modify fields directly instead of replacing entire struct
	call.Fun = funcLit
	call.Args = nil
	call.Ellipsis = 0
}

// transformGetBitMethod transforms mask.GetBit(i) to check the i-th element.
func transformGetBitMethod(call *ast.CallExpr, ctx *transformContext) {
	if len(call.Args) != 1 {
		return
	}
	indexExpr := call.Args[0]
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	maskExpr := sel.X
	lanes := ctx.target.LanesFor(ctx.elemType)

	// For half-precision types on Fallback/NEON (non-AVX-promoted), keep as hwy.Mask.GetBit
	if ctx.isHalfPrec && !ctx.isAVXPromoted {
		transformGetBitMethodHalfPrecision(call, maskExpr, indexExpr, lanes, ctx)
		return
	}
	// For AVX promoted half-precision, fall through to use SIMD extraction (same as float32)

	// Use Int32 vector for extraction to match most masks used with GetBit
	intVecTypeName := getVectorTypeNameForInt("int32", ctx.elemType, ctx.target)
	pkgName := getVecPackageName(ctx.target)

	// func() bool {
	//   vOne := pkg.BroadcastInt32x4(1)
	//   vZero := pkg.BroadcastInt32x4(0)
	//   vMasked := vOne.Merge(vZero, mask)
	//   var tmp [lanes]int32
	//   vMasked.StoreSlice(tmp[:])
	//   return tmp[i] != 0
	// }()

	// 1. vOne := pkg.BroadcastInt32x*(1)
	vOneDecl := &ast.AssignStmt{
		Lhs: []ast.Expr{ast.NewIdent("_vOne")},
		Tok: token.DEFINE,
		Rhs: []ast.Expr{
			&ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "1"}},
			},
		},
	}

	// 2. vZero := pkg.BroadcastInt32x*(0)
	vZeroDecl := &ast.AssignStmt{
		Lhs: []ast.Expr{ast.NewIdent("_vZero")},
		Tok: token.DEFINE,
		Rhs: []ast.Expr{
			&ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "0"}},
			},
		},
	}

	// 3. vMasked := vOne.Merge(vZero, mask)
	vMaskedDecl := &ast.AssignStmt{
		Lhs: []ast.Expr{ast.NewIdent("_vMasked")},
		Tok: token.DEFINE,
		Rhs: []ast.Expr{
			&ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent("_vOne"),
					Sel: ast.NewIdent("Merge"),
				},
				Args: []ast.Expr{
					ast.NewIdent("_vZero"),
					cloneExpr(maskExpr),
				},
			},
		},
	}

	// 4. var tmp [lanes]int32
	tmpDecl := &ast.DeclStmt{
		Decl: &ast.GenDecl{
			Tok: token.VAR,
			Specs: []ast.Spec{
				&ast.ValueSpec{
					Names: []*ast.Ident{ast.NewIdent("_simd_mask_tmp")},
					Type: &ast.ArrayType{
						Len: &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(lanes)},
						Elt: ast.NewIdent("int32"),
					},
				},
			},
		},
	}

	// 5. vMasked.StoreSlice(tmp[:])
	storeCall := &ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X:   ast.NewIdent("_vMasked"),
			Sel: ast.NewIdent("StoreSlice"),
		},
		Args: []ast.Expr{
			&ast.SliceExpr{
				X: ast.NewIdent("_simd_mask_tmp"),
			},
		},
	}

	// 6. return tmp[i] != 0
	checkExpr := &ast.BinaryExpr{
		X: &ast.IndexExpr{
			X:     ast.NewIdent("_simd_mask_tmp"),
			Index: cloneExpr(indexExpr),
		},
		Op: token.NEQ,
		Y:  &ast.BasicLit{Kind: token.INT, Value: "0"},
	}

	retStmt := &ast.ReturnStmt{
		Results: []ast.Expr{checkExpr},
	}

	funcLit := &ast.FuncLit{
		Type: &ast.FuncType{
			Results: &ast.FieldList{
				List: []*ast.Field{
					{Type: ast.NewIdent("bool")},
				},
			},
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				vOneDecl,
				vZeroDecl,
				vMaskedDecl,
				tmpDecl,
				&ast.ExprStmt{X: storeCall},
				retStmt,
			},
		},
	}

	*call = ast.CallExpr{
		Fun: funcLit,
	}
}



// transformGenDecl transforms variable declarations with generic types.
func transformGenDecl(decl *ast.GenDecl, ctx *transformContext) {
	if decl.Tok != token.VAR && decl.Tok != token.CONST {
		return
	}

	for _, spec := range decl.Specs {
		valueSpec, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}

		// Transform type if present
		if valueSpec.Type != nil {
			typeStr := exprToString(valueSpec.Type)
			// First specialize generic type parameters (T -> float32)
			specialized := specializeType(typeStr, ctx.typeParams, ctx.elemType)
			// Then transform hwy.Vec[float32] -> asm.Float32x4 for SIMD targets
			specialized = specializeVecType(specialized, ctx.elemType, ctx.target, ctx.skipHalfPrecNEON)
			if specialized != typeStr {
				valueSpec.Type = parseTypeExpr(specialized)
			}
		}
	}
}

// transformAssignStmt transforms assignments, particularly for loop stride calculations
// and hoisting hwy.Set calls with constant values.
func transformAssignStmt(stmt *ast.AssignStmt, ctx *transformContext) {
	// For fallback, don't replace NumLanes with a constant - keep it dynamic
	if ctx.target.IsFallback() {
		return
	}

	// For NEON target, detect accumulator patterns and use in-place operations.
	// Pattern: acc = v.MulAdd(a, acc) -> v.MulAddAcc(a, &acc)
	// This avoids return value allocation overhead on ARM64.
	// Skip when skipHalfPrecNEON is true because operands stay as hwy.Vec[T]
	// (not asm.Float16x8), and MulAddAcc is only defined on asm types.
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON {
		if transformed := tryTransformToInPlace(stmt, ctx); transformed {
			return
		}
	}

	// Look for v.NumElements(), hwy.Lanes[T](), or similar and replace with constant
	for i, rhs := range stmt.Rhs {
		if call, ok := rhs.(*ast.CallExpr); ok {
			// Check for hwy.Lanes[T]() or hwy.NumLanes[T]() - IndexExpr wrapping SelectorExpr
			if indexExpr, ok := call.Fun.(*ast.IndexExpr); ok {
				if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
					if pkgIdent, ok := sel.X.(*ast.Ident); ok {
						if pkgIdent.Name == "hwy" && (sel.Sel.Name == "Lanes" || sel.Sel.Name == "MaxLanes" || sel.Sel.Name == "NumLanes") {
							// Extract the actual type parameter from hwy.NumLanes[T]()
							// Use it instead of ctx.elemType to get correct lane count
							effectiveElemType := ctx.elemType
							if typeIdent, ok := indexExpr.Index.(*ast.Ident); ok {
								effectiveElemType = typeIdent.Name
							}
							// Replace with constant lane count for the actual type parameter
							lanes := ctx.target.LanesFor(effectiveElemType)
							stmt.Rhs[i] = &ast.BasicLit{
								Kind:  token.INT,
								Value: strconv.Itoa(lanes),
							}
							// Track the variable name
							if len(stmt.Lhs) > i {
								if ident, ok := stmt.Lhs[i].(*ast.Ident); ok {
									ctx.lanesVars[ident.Name] = true
								}
							}
							continue
						}
					}
				}
			}
			// Check for v.NumElements() or v.NumLanes()
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if sel.Sel.Name == "NumElements" || sel.Sel.Name == "NumLanes" {
					// Try to look up the element type of the vector variable
					// If the receiver is a variable we've tracked from a Load call,
					// use its element type instead of the function's default
					effectiveElemType := ctx.elemType
					if varIdent, ok := sel.X.(*ast.Ident); ok {
						if varElemType, ok := ctx.varVecElemType[varIdent.Name]; ok {
							effectiveElemType = varElemType
						}
					}
					// Replace with constant lane count
					lanes := ctx.target.LanesFor(effectiveElemType)
					stmt.Rhs[i] = &ast.BasicLit{
						Kind:  token.INT,
						Value: strconv.Itoa(lanes),
					}
					// Track the variable name so we can recognize it in loop strides
					if len(stmt.Lhs) > i {
						if ident, ok := stmt.Lhs[i].(*ast.Ident); ok {
							ctx.lanesVars[ident.Name] = true
						}
					}
				}
			}
		}

		// Check for make([]T, ...) calls
		if call, ok := rhs.(*ast.CallExpr); ok {
			if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "make" {
				if len(call.Args) >= 2 {
					// Check if first arg is []T (slice type)
					if arrayType, ok := call.Args[0].(*ast.ArrayType); ok {
						if arrayType.Len == nil { // It's a slice, not an array
							// Specialize the element type (T -> float32/float64)
							elemTypeStr := exprToString(arrayType.Elt)
							specializedType := specializeType(elemTypeStr, ctx.typeParams, ctx.elemType)
							// Also specialize hwy.Vec[float32] -> asm.Float32x4 for SIMD targets
							specializedType = specializeVecType(specializedType, ctx.elemType, ctx.target, ctx.skipHalfPrecNEON)

							// Check if second arg is a lanes variable or literal for stack array optimization
							var lanesCount int
							switch sizeArg := call.Args[1].(type) {
							case *ast.Ident:
								if ctx.lanesVars[sizeArg.Name] {
									lanesCount = ctx.target.LanesFor(ctx.elemType)
								}
							case *ast.BasicLit:
								if sizeArg.Kind == token.INT {
									lanesCount, _ = strconv.Atoi(sizeArg.Value)
								}
							}

							if lanesCount > 0 {
								// Replace make([]T, lanes) with [lanes]T{} (zero-valued array literal)
								stmt.Rhs[i] = &ast.CompositeLit{
									Type: &ast.ArrayType{
										Len: &ast.BasicLit{
											Kind:  token.INT,
											Value: strconv.Itoa(lanesCount),
										},
										Elt: parseTypeExpr(specializedType),
									},
								}
								// Track this variable as a stack array
								if len(stmt.Lhs) > i {
									if ident, ok := stmt.Lhs[i].(*ast.Ident); ok {
										ctx.stackArrayVars[ident.Name] = true
									}
								}
							} else if elemTypeStr != specializedType {
								// Just replace T with concrete type in make call
								arrayType.Elt = parseTypeExpr(specializedType)
							}
						}
					}
				}
			}
		}

		// Check for hwy.Set[T](constant) calls that can be hoisted
		if hoistedName := tryHoistSetCall(stmt, i, rhs, ctx); hoistedName != "" {
			// Replace RHS with reference to hoisted variable
			stmt.Rhs[i] = ast.NewIdent(hoistedName)
		}
	}
}

// findMaxLoadSizeForElemType scans the function body for hwy.Load[T](slice) calls
// and returns the maximum slice size found for the given element type.
// This is used to determine the appropriate vector width for constant hoisting.
func findMaxLoadSizeForElemType(body *ast.BlockStmt, elemType string) int {
	maxSize := 0
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		// Check for hwy.Load[T](slice) pattern
		indexExpr, ok := call.Fun.(*ast.IndexExpr)
		if !ok {
			return true
		}
		selExpr, ok := indexExpr.X.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		ident, ok := selExpr.X.(*ast.Ident)
		if !ok || ident.Name != "hwy" || selExpr.Sel.Name != "LoadSlice" {
			return true
		}
		// Check type parameter matches
		typeIdent, ok := indexExpr.Index.(*ast.Ident)
		if !ok || typeIdent.Name != elemType {
			return true
		}
		// Get slice size from argument
		if len(call.Args) == 1 {
			if size := getSliceSize(call.Args[0]); size > 0 && size > maxSize {
				maxSize = size
			}
		}
		return true
	})
	return maxSize
}

// convertBlankAssignToExprStmt walks a block statement and replaces any
// "_ = expr" assignments with just "expr" as an expression statement.
// This is needed because tryTransformToInPlace converts assignments like
// "acc = v.MulAdd(a, acc)" to "_ = v.MulAddAcc(a, &acc)", but MulAddAcc
// returns void, making "_ = voidFunc()" invalid Go.
func convertBlankAssignToExprStmt(block *ast.BlockStmt) {
	if block == nil {
		return
	}
	for i, stmt := range block.List {
		switch s := stmt.(type) {
		case *ast.AssignStmt:
			// Check for _ = expr pattern where expr is a call (void function)
			if len(s.Lhs) == 1 && len(s.Rhs) == 1 {
				if ident, ok := s.Lhs[0].(*ast.Ident); ok && ident.Name == "_" {
					// Only convert if the RHS is a function/method call
					// (bounds check hints like _ = slice[i] must stay as-is)
					if _, isCall := s.Rhs[0].(*ast.CallExpr); isCall {
						block.List[i] = &ast.ExprStmt{X: s.Rhs[0]}
					}
				}
			}
		case *ast.BlockStmt:
			convertBlankAssignToExprStmt(s)
		case *ast.IfStmt:
			convertBlankAssignToExprStmt(s.Body)
			if s.Else != nil {
				if elseBlock, ok := s.Else.(*ast.BlockStmt); ok {
					convertBlankAssignToExprStmt(elseBlock)
				}
			}
		case *ast.ForStmt:
			convertBlankAssignToExprStmt(s.Body)
		case *ast.RangeStmt:
			convertBlankAssignToExprStmt(s.Body)
		}
	}
}

// tryTransformToInPlace detects accumulator patterns and transforms them to in-place operations.
// Pattern: acc = v.MulAdd(a, acc) -> v.MulAddAcc(a, &acc)
// This only applies to NEON target where in-place operations avoid allocation overhead.
// Returns true if the statement was transformed.
func tryTransformToInPlace(stmt *ast.AssignStmt, ctx *transformContext) bool {
	// Only handle simple assignments with one LHS and one RHS
	if len(stmt.Lhs) != 1 || len(stmt.Rhs) != 1 {
		return false
	}

	// LHS must be an identifier (the accumulator variable)
	lhsIdent, ok := stmt.Lhs[0].(*ast.Ident)
	if !ok {
		return false
	}
	accName := lhsIdent.Name

	// RHS must be a method call
	call, ok := stmt.Rhs[0].(*ast.CallExpr)
	if !ok {
		return false
	}

	// Get the method name
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	methodName := sel.Sel.Name

	// Check if this operation has an in-place variant
	var inPlaceOp OpInfo
	var foundInPlace bool

	// When the call is a function call like hwy.MulAdd(vA, vB, vC) that will be converted
	// to a method call vA.MulAdd(vB, vC), the first arg becomes receiver, shifting indices.
	// Detect this by checking if the call is hwy.* (function) vs receiver.Method() (already method).
	accArgAdjustment := 0
	if pkgIdent, ok := sel.X.(*ast.Ident); ok {
		// hwy.MulAdd(...) is a function call - after transformation, arg[0] becomes receiver
		if pkgIdent.Name == "hwy" || pkgIdent.Name == "asm" {
			accArgAdjustment = 1
		}
	}

	entry, ok := ctx.inPlaceLookup[methodName]
	if !ok {
		return false
	}
	// Check if the last argument is the same as LHS (accumulator pattern)
	if len(call.Args) > 0 {
		lastArg := call.Args[len(call.Args)-1]
		if argIdent, ok := lastArg.(*ast.Ident); ok && argIdent.Name == accName {
			// Verify AccArg matches the actual last argument index after transformation.
			// For half-precision NEON: hwy.MulAdd(vA, vB, vC) -> vA.MulAddAcc(vB, &vC)
			// Original: 3 args, lastArg at index 2
			// After transformation: 2 args (vB, &vC), lastArg at index 1
			// MulAddAcc.AccArg = 1, so check: 1 == (3-1) - 1 = 1 ✓
			expectedAccArg := len(call.Args) - 1 - accArgAdjustment
			if entry.OpInfo.AccArg == expectedAccArg {
				inPlaceOp = entry.OpInfo
				inPlaceOp.Name = entry.OpName
				foundInPlace = true
			}
		}
	}

	if !foundInPlace {
		return false
	}

	// Transform: acc = v.MulAdd(a, acc) -> v.MulAddAcc(a, &acc)
	// The receiver stays the same, we change the method name and wrap the last arg with &

	// Change method name to in-place version
	sel.Sel.Name = inPlaceOp.Name

	// Wrap the accumulator argument with &
	lastIdx := len(call.Args) - 1
	call.Args[lastIdx] = &ast.UnaryExpr{
		Op: token.AND,
		X:  call.Args[lastIdx],
	}

	// Remove the assignment - convert to expression statement
	// We need to replace the AssignStmt with an ExprStmt in the parent
	// Since we can't easily do that here, we'll use a workaround:
	// Set the LHS to a blank identifier and the RHS to the call
	// This isn't ideal, but Go will optimize away the blank assignment
	stmt.Lhs[0] = ast.NewIdent("_")

	return true
}

// tryHoistSetCall checks if an expression is a hwy.Set[T](constant) call
// and if so, registers it for hoisting and returns the hoisted variable name.
func tryHoistSetCall(stmt *ast.AssignStmt, rhsIndex int, rhs ast.Expr, ctx *transformContext) string {
	call, ok := rhs.(*ast.CallExpr)
	if !ok {
		return ""
	}

	// Check for hwy.Set[T](arg) pattern - could be IndexExpr wrapping SelectorExpr
	var selExpr *ast.SelectorExpr
	var typeParam string
	switch fun := call.Fun.(type) {
	case *ast.IndexExpr:
		// hwy.Set[T](arg)
		selExpr, ok = fun.X.(*ast.SelectorExpr)
		if !ok {
			return ""
		}
		// Extract the type parameter
		if typeIdent, ok := fun.Index.(*ast.Ident); ok {
			typeParam = typeIdent.Name
		}
	case *ast.SelectorExpr:
		// hwy.Set(arg) - non-generic, shouldn't happen but handle it
		selExpr = fun
	default:
		return ""
	}

	// Verify it's hwy.Set or hwy.Const
	ident, ok := selExpr.X.(*ast.Ident)
	if !ok || ident.Name != "hwy" {
		return ""
	}
	isConst := selExpr.Sel.Name == "Const"
	if selExpr.Sel.Name != "Set" && !isConst {
		return ""
	}

	// Determine the actual element type for this call
	actualElemType := ctx.elemType
	if !isConst && typeParam == "int32" {
		actualElemType = "int32"
		// For non-AVX-promoted half-precision types, don't hoist int32 constants to native SIMD types
		// because hwy.ConvertToInt32 returns hwy.Vec[int32], not native SIMD types.
		// Keeping them as hwy.Set[int32] ensures type compatibility.
		// For AVX promoted half-precision, ConvertToInt32 returns archsimd.Int32x8/Int32x16,
		// so hoisting as archsimd.BroadcastInt32xN is correct.
		if ctx.isHalfPrec && !ctx.isAVXPromoted {
			return ""
		}
	}
	// For all half-precision types (NEON, AVX promoted, Fallback), skip hoisting float constants.
	// NEON and Fallback: constants stay as inline hwy.Set/Const calls.
	// AVX promoted: the inline Set→asm.Broadcast transformation produces the correct promoted type.
	// Int32 constants for AVX promoted are handled above (line 5139) and are fine to hoist.
	if ctx.isHalfPrec && actualElemType != "int32" {
		return ""
	}

	// Check if the argument is a constant (literal or type conversion of constant)
	if len(call.Args) != 1 {
		return ""
	}
	arg := call.Args[0]
	constValue := extractConstantValue(arg, actualElemType, ctx)
	if constValue == "" {
		return ""
	}

	// Get the local variable name being assigned
	if rhsIndex >= len(stmt.Lhs) {
		return ""
	}
	localIdent, ok := stmt.Lhs[rhsIndex].(*ast.Ident)
	if !ok {
		return ""
	}
	localVarName := localIdent.Name

	// Generate unique hoisted variable name (include target to avoid conflicts)
	// For int32 constants, we need separate versions for f32 and f64 functions
	// because they have different lane counts
	elemSuffix := "f32"
	if ctx.elemType == "float64" {
		elemSuffix = "f64"
	}
	if actualElemType == "int32" {
		// Include parent element type in suffix for proper lane matching
		if ctx.elemType == "float64" {
			elemSuffix = "i32_f64"
		} else {
			elemSuffix = "i32_f32"
		}
	}
	hoistedName := fmt.Sprintf("%s_%s_%s_%s", ctx.funcName, ctx.target.Name, localVarName, elemSuffix)

	// Get vector type and broadcast function for this target
	// For int32 types used in float operations, match the lane count of the parent element type
	// If inferredFuncLanes is set and smaller than target width, use it to match Load sizes
	var useLanes int
	if actualElemType == "int32" || actualElemType == "int64" {
		// Int32/int64 constants used in float functions should match the parent type's lane count
		// e.g., int32 constants in float64 functions need 2 lanes on NEON, not 4
		useLanes = ctx.target.LanesFor(ctx.elemType)
	} else {
		targetLanes := ctx.target.LanesFor(actualElemType)
		useLanes = targetLanes
		if ctx.inferredFuncLanes > 0 && ctx.inferredFuncLanes < targetLanes {
			useLanes = ctx.inferredFuncLanes
		}
	}
	vecTypeName := getVectorTypeNameForLanes(actualElemType, useLanes)
	pkgName := getVecPackageName(ctx.target)
	broadcastFunc := fmt.Sprintf("%s.Broadcast%s", pkgName, vecTypeName)

	// Register the hoisted constant
	ctx.hoistedConsts[localVarName] = HoistedConst{
		VarName:   hoistedName,
		Value:     constValue,
		VecType:   vecTypeName,
		Broadcast: broadcastFunc,
	}

	return hoistedName
}

// extractConstantValue extracts the string representation of a constant value.
// Returns empty string if the expression is not a constant.
// The elemType parameter is used to add type conversion when needed.
// The ctx is used to check if a variable is locally-defined (not a constant).
func extractConstantValue(expr ast.Expr, elemType string, ctx *transformContext) string {
	switch e := expr.(type) {
	case *ast.BasicLit:
		// Literal like 1.0, 0.5, etc.
		return e.Value
	case *ast.UnaryExpr:
		// Handle negative literals like -1.0
		if e.Op == token.SUB {
			if inner := extractConstantValueRaw(e.X, ctx); inner != "" {
				return "-" + inner
			}
		}
	case *ast.CallExpr:
		// Type conversion like T(1.0) or float32(sigmoidC1)
		// Get the inner value without adding another type conversion
		if len(e.Args) == 1 {
			inner := extractConstantValueRaw(e.Args[0], ctx)
			if inner != "" {
				// Add the target type conversion
				return fmt.Sprintf("%s(%s)", elemType, inner)
			}
		}
	case *ast.Ident:
		// Variable reference - only hoist if it's a known package-level constant
		name := e.Name
		if ctx.localVars[name] {
			return ""
		}
		if ctx.packageConsts[name] {
			// Add type conversion in case the var type differs from target type
			return fmt.Sprintf("%s(%s)", elemType, name)
		}
	}
	return ""
}

// extractConstantValueRaw extracts the raw constant value without type conversion.
func extractConstantValueRaw(expr ast.Expr, ctx *transformContext) string {
	switch e := expr.(type) {
	case *ast.BasicLit:
		return e.Value
	case *ast.UnaryExpr:
		if e.Op == token.SUB {
			if inner := extractConstantValueRaw(e.X, ctx); inner != "" {
				return "-" + inner
			}
		}
	case *ast.Ident:
		name := e.Name
		if ctx != nil && ctx.localVars[name] {
			return ""
		}
		if ctx != nil && ctx.packageConsts[name] {
			return name
		}
	}
	return ""
}

