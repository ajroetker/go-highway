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
	"sort"
	"strconv"
	"strings"
)

// isFloat16Type returns true if the element type is Float16.
func isFloat16Type(elemType string) bool {
	return elemType == "hwy.Float16" || elemType == "Float16"
}

// isBFloat16Type returns true if the element type is BFloat16.
func isBFloat16Type(elemType string) bool {
	return elemType == "hwy.BFloat16" || elemType == "BFloat16"
}

// isHalfPrecisionType returns true if the element type is Float16 or BFloat16.
func isHalfPrecisionType(elemType string) bool {
	return isFloat16Type(elemType) || isBFloat16Type(elemType)
}

// isUnsignedIntType returns true if the element type is an unsigned integer type.
func isUnsignedIntType(elemType string) bool {
	return elemType == "uint8" || elemType == "uint16" || elemType == "uint32" || elemType == "uint64"
}

// is64BitIntType returns true if the element type is a 64-bit integer (signed or unsigned).
func is64BitIntType(elemType string) bool {
	return elemType == "int64" || elemType == "uint64"
}

// isHalfPrecisionSliceType checks if a parameter type is a slice of half-precision elements.
// It handles both concrete types like "[]hwy.Float16" and generic types like "[]T" when
// elemType is a half-precision type.
func isHalfPrecisionSliceType(paramType, elemType string) bool {
	// Check for concrete half-precision slice types
	if paramType == "[]hwy.Float16" || paramType == "[]hwy.BFloat16" ||
		paramType == "[]Float16" || paramType == "[]BFloat16" {
		return true
	}
	// Check for generic slice type when elem type is half-precision
	// e.g., "[]T" with elemType="hwy.Float16"
	if strings.HasPrefix(paramType, "[]") && isHalfPrecisionType(elemType) {
		// The slice element type should match the function's element type
		sliceElem := strings.TrimPrefix(paramType, "[]")
		// It's a generic type param like "T" or it matches the concrete type
		if len(sliceElem) == 1 || sliceElem == elemType {
			return true
		}
	}
	return false
}

// isHalfPrecisionScalarType checks if a parameter type is a scalar (non-slice) half-precision type.
// It handles both concrete types like "hwy.Float16" and generic types like "T" when
// elemType is a half-precision type.
func isHalfPrecisionScalarType(paramType, elemType string) bool {
	// Don't match slices or arrays
	if strings.HasPrefix(paramType, "[]") || strings.HasPrefix(paramType, "[") {
		return false
	}
	// Check for concrete half-precision types
	if paramType == "hwy.Float16" || paramType == "hwy.BFloat16" ||
		paramType == "Float16" || paramType == "BFloat16" {
		return true
	}
	// Check for generic type param (single letter like "T") when elem type is half-precision
	if len(paramType) == 1 && isHalfPrecisionType(elemType) {
		return true
	}
	return false
}

// returnsVecType checks if any return type contains "Vec" or "Mask".
// Vec-returning functions use hwy.Vec operations which already work for Float16/BFloat16.
func returnsVecType(returns []Param) bool {
	for _, ret := range returns {
		if strings.Contains(ret.Type, "Vec") || strings.Contains(ret.Type, "Mask") {
			return true
		}
	}
	return false
}

// getHalfPrecisionFuncName returns the hwy function name for Float16/BFloat16 operations.
// For example, "Add" with Float16 returns "AddF16", "Add" with BFloat16 returns "AddBF16".
// Returns empty string for operations that don't have F16/BF16 specific versions.
func getHalfPrecisionFuncName(opName string, elemType string) string {
	suffix := "F16"
	if isBFloat16Type(elemType) {
		suffix = "BF16"
	}

	// Map operation names to their F16/BF16 counterparts
	switch opName {
	case "Add":
		return "Add" + suffix
	case "Sub":
		return "Sub" + suffix
	case "Mul":
		return "Mul" + suffix
	case "Div":
		return "Div" + suffix
	case "FMA", "MulAdd":
		return "FMA" + suffix
	case "Neg":
		return "Neg" + suffix
	case "Abs":
		return "Abs" + suffix
	case "Min":
		return "Min" + suffix
	case "Max":
		return "Max" + suffix
	case "Sqrt":
		return "Sqrt" + suffix
	case "RSqrt":
		return "RSqrt" + suffix
	case "RSqrtNewtonRaphson":
		return "RSqrtNewtonRaphson" + suffix
	case "RSqrtPrecise":
		return "RSqrtPrecise" + suffix
	case "Greater", "GreaterThan":
		return "GreaterThan" + suffix
	case "Less", "LessThan":
		return "LessThan" + suffix
	case "GreaterEqual", "GreaterThanOrEqual":
		return "GreaterThanOrEqual" + suffix
	case "LessEqual", "LessThanOrEqual":
		return "LessThanOrEqual" + suffix
	case "Equal":
		return "Equal" + suffix
	case "ReduceSum":
		return "ReduceSum" + suffix
	case "ReduceMin":
		return "ReduceMin" + suffix
	case "ReduceMax":
		return "ReduceMax" + suffix
	case "IsNaN":
		return "IsNaN" + suffix
	case "IsInf":
		return "IsInf" + suffix
	default:
		// For operations without specific F16/BF16 variants, return empty
		return ""
	}
}

// isHalfPrecisionMergeOp returns true if the operation is Merge/IfThenElse for F16/BF16.
// These need special handling because argument order differs between hwy.Merge and IfThenElseF16.
func isHalfPrecisionMergeOp(opName string) bool {
	return opName == "Merge" || opName == "IfThenElse"
}

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
	FileSet            *token.FileSet    // For resolving line numbers in conditional blocks
	Imports            map[string]string // map[local_name]import_path for resolving package references
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
			Results: pf.buildResultsWithTarget(elemType, target),
		},
		Body: cloneBlockStmt(filteredBody),
	}

	// Build parameter list with specialized types
	for _, param := range pf.Params {
		paramType := specializeType(param.Type, pf.TypeParams, elemType)
		// Also transform hwy.Vec[T] to concrete vector types for SIMD targets
		paramType = specializeVecType(paramType, elemType, target)
		field := &ast.Field{
			Names: []*ast.Ident{ast.NewIdent(param.Name)},
			Type:  parseTypeExpr(paramType),
		}
		funcDecl.Type.Params.List = append(funcDecl.Type.Params.List, field)
	}

	// For fallback target with predicate functions, generate scalar loop body
	// to avoid allocations from hwy.Load/pred.Apply
	if target.Name == "Fallback" && hasPredicateParam(pf) {
		if scalarBody := generateScalarPredicateBody(pf, elemType); scalarBody != nil {
			funcDecl.Body = scalarBody
			return &TransformResult{
				FuncDecl:      funcDecl,
				HoistedConsts: nil,
			}
		}
	}

	// Transform the function body
	ctx := &transformContext{
		target:                  target,
		elemType:                elemType,
		typeParams:              pf.TypeParams,
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
	}

	// Add function parameters to localVars to prevent them from being hoisted
	// Also track half-precision slice and scalar parameters
	for _, param := range pf.Params {
		ctx.localVars[param.Name] = true
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

	// Resolve type-specific constant references
	// Pattern 1: expC0 -> expC0_f32 (base name lookup)
	// Pattern 2: expC0_f32 -> expC0_f64 (suffix swapping for compilable base files)
	transformIdentifiers(funcDecl.Body, ctx)

	transformNode(funcDecl.Body, ctx)

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
	if isHalfPrecisionType(elemType) && !returnsVecType(pf.Returns) {
		transformHalfPrecisionFallback(funcDecl.Body, ctx)
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
	sort.Strings(keys)
	for _, k := range keys {
		hoisted = append(hoisted, ctx.hoistedConsts[k])
	}

	return &TransformResult{
		FuncDecl:      funcDecl,
		HoistedConsts: hoisted,
	}
}

type transformContext struct {
	target                  Target
	elemType                string
	typeParams              []TypeParam
	lanesVars               map[string]bool // Variables assigned from NumLanes()
	localVars               map[string]bool // Variables defined locally in the function
	stackArrayVars          map[string]bool // Variables that are stack arrays (need [:] when used as slice)
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

	// Check for hwy.Load(...) or hwy.Load[T](...) call
	var funcName string
	var explicitElemType string // Explicit type param from hwy.Load[uint8] style calls
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		// hwy.Load(...)
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

	if funcName != "Load" {
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

	// Use explicit type param if present, otherwise fall back to function's elemType
	effectiveElemType := ctx.elemType
	if explicitElemType != "" {
		effectiveElemType = explicitElemType
	}

	elemSize := elemTypeSize(effectiveElemType)
	if elemSize <= 0 {
		return vecLoadInfo{}
	}

	return vecLoadInfo{
		lanes:    sliceBytes / elemSize,
		elemType: explicitElemType, // Only set if different from function's elemType
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
				arg0IsInt32 := isInt32ExprHelper(call.Args[0], ctx)
				arg1IsInt32 := isInt32ExprHelper(call.Args[1], ctx)
				if arg0IsInt32 && arg1IsInt32 {
					return "int32"
				}
			}
		}
	}

	return ""
}

// isInt32ExprHelper is a helper that checks if an expression is int32 without causing recursion.
func isInt32ExprHelper(expr ast.Expr, ctx *transformContext) bool {
	switch e := expr.(type) {
	case *ast.Ident:
		return ctx.varTypes[e.Name] == "int32"
	case *ast.CallExpr:
		// Recursively check for int32-returning calls
		return inferTypeFromExpr(e, ctx) == "int32"
	}
	return false
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
	if ctx.target.Name == "Fallback" || ctx.loopInfo == nil {
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
				switch ctx.elemType {
				case "float64":
					suffix = suffix + "_Float64"
				case "hwy.Float16":
					suffix = suffix + "_Float16"
				case "hwy.BFloat16":
					suffix = suffix + "_BFloat16"
				case "int32":
					suffix = suffix + "_Int32"
				case "int64":
					suffix = suffix + "_Int64"
				case "uint32":
					suffix = suffix + "_Uint32"
				case "uint64":
					suffix = suffix + "_Uint64"
				}
			}
			ident.Name = ident.Name + suffix
		}
	}

	// Transform Vec method calls like .Store() -> .StoreSlice() for SIMD targets
	// This handles cases like fn(x).Store(dst) where fn returns a Vec
	// Skip this for package-level function calls like hwy.Store() which are handled later
	if ctx.target.Name != "Fallback" {
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
			// Don't transform package-level function calls
			if ident, ok := sel.X.(*ast.Ident); ok && ident.Name == "hwy" {
				// Skip - this is a package-level function, handled later
			} else {
				switch sel.Sel.Name {
				case "Store":
					// Transform .Store(dst) -> .StoreSlice(dst)
					sel.Sel.Name = "StoreSlice"
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
				suffix := ctx.target.Suffix()
				// Add type suffix for non-float32 types
				switch ctx.elemType {
				case "float64":
					suffix = suffix + "_Float64"
				case "hwy.Float16":
					suffix = suffix + "_Float16"
				case "hwy.BFloat16":
					suffix = suffix + "_BFloat16"
				case "int32":
					suffix = suffix + "_Int32"
				case "int64":
					suffix = suffix + "_Int64"
				case "uint32":
					suffix = suffix + "_Uint32"
				case "uint64":
					suffix = suffix + "_Uint64"
				}
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
		// Transform hwy.Const[T](val) to hwy.Set(val) for non-float32 types
		// ONLY when val is a named constant (identifier), not a literal.
		// Named constants have been suffix-transformed to the correct type.
		// Literals should stay with Const which handles the conversion.
		// Note: We don't return here - let the transformation continue so Set gets
		// transformed to asm.Broadcast* for SIMD targets.
		if selExpr.Sel.Name == "Const" {
			if ident, ok := selExpr.X.(*ast.Ident); ok && ident.Name == "hwy" {
				if ctx.elemType != "float32" && len(call.Args) > 0 {
					// Only transform if the argument is an identifier (named constant)
					// or binary expression like `constant * 2`
					if _, isIdent := call.Args[0].(*ast.Ident); isIdent {
						selExpr.Sel.Name = "Set"
					} else if _, isBinary := call.Args[0].(*ast.BinaryExpr); isBinary {
						selExpr.Sel.Name = "Set"
					}
				}
			}
		}
		if ctx.target.Name == "Fallback" {
			// For fallback, replace type param with concrete type
			// hwy.Zero[T]() -> hwy.Zero[float32]()
			for _, tp := range ctx.typeParams {
				if ident, ok := fun.Index.(*ast.Ident); ok && ident.Name == tp.Name {
					ident.Name = ctx.elemType
				}
			}
			// Keep the IndexExpr (with type param), just update the type
		} else if isHalfPrecisionType(ctx.elemType) {
			// For Float16/BFloat16 on SIMD targets, keep the type param for functions
			// like Const, Set, Zero that need it for type inference
			funcName := selExpr.Sel.Name
			switch funcName {
			case "Const", "Set", "Zero":
				// Replace type param with concrete type (e.g., hwy.Const[T] -> hwy.Const[hwy.Float16])
				for _, tp := range ctx.typeParams {
					if ident, ok := fun.Index.(*ast.Ident); ok && ident.Name == tp.Name {
						ident.Name = ctx.elemType
					}
				}
				// Keep the IndexExpr with the concrete type
			case "ConvertExponentToFloat":
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
					call.Fun = &ast.SelectorExpr{
						X:   call.Args[0],
						Sel: ast.NewIdent(methodName),
					}
					call.Args = nil
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
		if ctx.target.Name == "Fallback" {
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
		suffix := ctx.target.Suffix()
		if ctx.elemType == "float64" {
			suffix = suffix + "_Float64"
		} else if isFloat16Type(ctx.elemType) {
			suffix = suffix + "_Float16"
		} else if isBFloat16Type(ctx.elemType) {
			suffix = suffix + "_BFloat16"
		}
		selExpr.Sel.Name = funcName + suffix
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

	// v.StoreSlice(tmp[:]) or hwy.Store(v, tmp[:]) for half-precision
	var storeCall *ast.CallExpr
	if isHalfPrecisionType(ctx.elemType) {
		// hwy.Store(v, tmp[:]) for half-precision types
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

	// For half-precision types, use hwy package functions instead of native SIMD
	if isHalfPrecisionType(ctx.elemType) {
		transformGetBitMethodHalfPrecision(call, maskExpr, indexExpr, lanes, ctx)
		return
	}

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

// transformGetBitMethodHalfPrecision handles mask.GetBit(i) for half-precision types.
// Uses hwy package functions instead of native SIMD types.
func transformGetBitMethodHalfPrecision(call *ast.CallExpr, maskExpr, indexExpr ast.Expr, lanes int, ctx *transformContext) {
	// For half-precision, use a simpler scalar extraction approach
	// func() bool {
	//   return mask.GetBit(i)  // Keep as-is for hwy.Mask[Float16]
	// }()
	//
	// Actually, hwy.Mask already has GetBit, so we can keep the call as-is
	// but we need to ensure the mask is properly typed.

	// The mask.GetBit(i) call should work directly for hwy.Mask types
	// No transformation needed for half-precision - keep the original call
	return
}

// transformToMethod converts hwy.Add(a, b) to a.Add(b) for SIMD targets.
// For Fallback, keeps hwy.Add(a, b) as-is.
func transformToMethod(call *ast.CallExpr, funcName string, opInfo OpInfo, ctx *transformContext) {
	if len(call.Args) < 1 {
		return
	}

	// For fallback, keep hwy calls as-is (don't convert to method calls)
	if ctx.target.Name == "Fallback" {
		// Just update the package name if needed
		switch fun := call.Fun.(type) {
		case *ast.SelectorExpr:
			fun.X = ast.NewIdent("hwy")
		case *ast.IndexExpr:
			if sel, ok := fun.X.(*ast.SelectorExpr); ok {
				sel.X = ast.NewIdent("hwy")
			}
		}
		return
	}

	// For Float16/BFloat16 on SIMD targets, use hwy package functions instead of methods.
	// archsimd doesn't have native support for half-precision types.
	if isHalfPrecisionType(ctx.elemType) {
		// Handle Merge specially - needs argument reordering
		// hwy.Merge(yes, no, mask) -> hwy.IfThenElseF16(mask, yes, no)
		if funcName == "Merge" && len(call.Args) >= 3 {
			suffix := "F16"
			if isBFloat16Type(ctx.elemType) {
				suffix = "BF16"
			}
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent("IfThenElse" + suffix),
			}
			// Reorder: (yes, no, mask) -> (mask, yes, no)
			call.Args = []ast.Expr{call.Args[2], call.Args[0], call.Args[1]}
			return
		}
		// Handle IfThenElse - same signature as IfThenElseF16, no reordering needed
		// hwy.IfThenElse(mask, yes, no) -> hwy.IfThenElseF16(mask, yes, no)
		if funcName == "IfThenElse" && len(call.Args) >= 3 {
			suffix := "F16"
			if isBFloat16Type(ctx.elemType) {
				suffix = "BF16"
			}
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent("IfThenElse" + suffix),
			}
			// Args stay in same order
			return
		}

		if f16FuncName := getHalfPrecisionFuncName(funcName, ctx.elemType); f16FuncName != "" {
			// For operations with 2 operands, check if both are int32 - if so, keep generic hwy function
			// This handles comparisons (Equal, Greater), arithmetic (Add, Sub, Mul), and bitwise (And, Or)
			// operations that may operate on int32 intermediate values (like octant calculations in trig functions)
			if len(call.Args) >= 2 {
				if isInt32Expr(call.Args[0], ctx) && isInt32Expr(call.Args[1], ctx) {
					// Keep as generic hwy.Add, hwy.Equal, hwy.And, etc. for int32 operands
					call.Fun = &ast.SelectorExpr{
						X:   ast.NewIdent("hwy"),
						Sel: ast.NewIdent(funcName),
					}
					return
				}
			}
			// Transform to hwy.AddF16(a, b), hwy.MulF16(a, b), etc.
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent(f16FuncName),
			}
			// Args stay as-is (already in the correct order for function calls)
			return
		}

		// For operations that don't have F16/BF16 variants, keep as hwy function calls
		// instead of converting to method calls (which don't exist on hwy.Vec[Float16])
		switch funcName {
		case "RoundToEven", "ConvertToInt32", "ConvertToFloat32":
			// Keep as hwy function call - do NOT convert to method
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent(funcName),
			}
			return
		case "And", "Or", "Xor", "Not", "AndNot":
			// Bitwise operations on hwy.Vec[Float16/BFloat16] don't have method forms,
			// so keep them as hwy.And, hwy.Or, hwy.Xor, hwy.Not, hwy.AndNot
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent(funcName),
			}
			return
		case "NotEqual":
			// hwy.NotEqual on hwy.Vec[Float16/BFloat16] doesn't have a method form,
			// keep as hwy.NotEqual(a, b)
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent("NotEqual"),
			}
			return
		case "Pow":
			// hwy.Pow on hwy.Vec[Float16/BFloat16] doesn't have a method form,
			// keep as hwy.Pow(base, exp)
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent("Pow"),
			}
			return
		case "MaskAnd", "MaskOr", "MaskXor", "MaskAndNot":
			// Mask operations on hwy.Mask[Float16/BFloat16] don't have method forms,
			// so keep them as hwy.MaskAnd, hwy.MaskOr, etc.
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent(funcName),
			}
			return
		case "Store":
			// Keep hwy.Store(v, dst) as-is for half-precision types
			// (hwy.Vec[Float16] has a Store method that handles the conversion)
			return
		case "Pow2":
			// Pow2 needs a type parameter: hwy.Pow2[hwy.Float16](kInt)
			call.Fun = &ast.IndexExpr{
				X: &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent("Pow2"),
				},
				Index: ast.NewIdent(ctx.elemType),
			}
			return
		case "SignBit":
			// SignBit needs a type parameter: hwy.SignBit[hwy.Float16]()
			call.Fun = &ast.IndexExpr{
				X: &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent("SignBit"),
				},
				Index: ast.NewIdent(ctx.elemType),
			}
			return
		}

		// For operations without F16/BF16 variants, fall through to regular handling
		// but this may cause issues if they try to use method calls
	}

	// For AVX2/AVX512, use wrapper functions for ReduceMax (unsigned only) and GetLane (all types).
	// archsimd doesn't have ReduceMax for unsigned types or a direct GetLane method.
	if ctx.target.Name == "AVX2" || ctx.target.Name == "AVX512" {
		switch funcName {
		case "ReduceMax":
			// hwy.ReduceMax(v) -> hwy.ReduceMax_AVX2_Uint32x8(v) (unsigned only)
			if isUnsignedIntType(ctx.elemType) && len(call.Args) >= 1 {
				vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
				wrapperName := fmt.Sprintf("ReduceMax_%s_%s", ctx.target.Name, vecTypeName)
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent(wrapperName),
				}
				// Args stay as-is
				return
			}
		case "GetLane":
			// hwy.GetLane(v, i) -> hwy.GetLane_AVX2_Float32x8(v, i) etc.
			if len(call.Args) >= 2 {
				vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
				wrapperName := fmt.Sprintf("GetLane_%s_%s", ctx.target.Name, vecTypeName)
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent(wrapperName),
				}
				// Args stay as-is
				return
			}
		}
	}

	// For 64-bit integer types on AVX2, use wrapper functions for Max and Min.
	// AVX2 doesn't have VPMAXSQ/VPMINUQ/VPMAXUQ/VPMINSQ instructions (only AVX-512 has them).
	if is64BitIntType(ctx.elemType) && ctx.target.Name == "AVX2" {
		switch funcName {
		case "Max":
			// hwy.Max(a, b) -> hwy.Max_AVX2_Uint64x4(a, b) or hwy.Max_AVX2_Int64x4(a, b)
			if len(call.Args) >= 2 {
				vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
				wrapperName := fmt.Sprintf("Max_%s_%s", ctx.target.Name, vecTypeName)
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent(wrapperName),
				}
				// Args stay as-is
				return
			}
		case "Min":
			// hwy.Min(a, b) -> hwy.Min_AVX2_Uint64x4(a, b) or hwy.Min_AVX2_Int64x4(a, b)
			if len(call.Args) >= 2 {
				vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
				wrapperName := fmt.Sprintf("Min_%s_%s", ctx.target.Name, vecTypeName)
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent(wrapperName),
				}
				// Args stay as-is
				return
			}
		}
	}

	// For SIMD targets, convert to method calls on archsimd types
	switch funcName {
	case "Store":
		// hwy.Store(v, dst) -> v.StoreSlice(dst)
		if len(call.Args) >= 2 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent("StoreSlice"),
			}
			call.Args = call.Args[1:]
		}

	case "MaskStore":
		// hwy.MaskStore(mask, v, dst) -> v.MaskStoreSlice(mask, dst)
		if len(call.Args) >= 3 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[1],
				Sel: ast.NewIdent("MaskStoreSlice"),
			}
			call.Args = []ast.Expr{call.Args[0], call.Args[2]}
		}

	case "Neg":
		// hwy.Neg(x) -> pkg.BroadcastFloat32x8(0).Sub(x) for SIMD
		// (archsimd/asm types don't have a Neg method, so we use 0 - x)
		if len(call.Args) >= 1 {
			vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
			pkgName := getVecPackageName(ctx.target)
			// Create pkg.BroadcastFloat32x8(0)
			zeroLit := &ast.BasicLit{Kind: token.INT, Value: "0"}
			zeroCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{zeroLit},
			}
			call.Fun = &ast.SelectorExpr{
				X:   zeroCall,
				Sel: ast.NewIdent("Sub"),
			}
			// Args stays as [x]
		}

	case "Pow2":
		// hwy.Pow2[T](kInt) -> kInt.Pow2Float32() or kInt.Pow2Float64()
		// based on context element type
		if len(call.Args) >= 1 {
			var methodName string
			switch ctx.elemType {
			case "float32":
				methodName = "Pow2Float32"
			case "float64":
				methodName = "Pow2Float64"
			default:
				methodName = "Pow2Float32" // fallback
			}
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent(methodName),
			}
			call.Args = nil
		}

	case "GetExponent":
		// For Float16/BFloat16, use hwy.GetExponent which has proper handling
		if isHalfPrecisionType(ctx.elemType) {
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent("GetExponent"),
			}
			return
		}
		if len(call.Args) >= 1 {
			x := call.Args[0]
			intVecTypeName := getVectorTypeNameForInt("int32", ctx.elemType, ctx.target)
			if ctx.elemType == "float64" {
				intVecTypeName = getVectorTypeNameForInt("int64", ctx.elemType, ctx.target)
			}
			pkgName := getVecPackageName(ctx.target)

			// 1. x.AsInt32() / x.AsInt64()
			var asIntMethod string
			var shift int
			var mask string
			var bias string

			if ctx.elemType == "float32" {
				asIntMethod = "AsInt32x8"
				// Check targets.go OpMap["AsInt32"].Name
				if op, ok := ctx.target.OpMap["AsInt32"]; ok {
					asIntMethod = op.Name
				}
				shift = 23
				mask = "255" // 0xFF
				bias = "127"
			} else {
				asIntMethod = "AsInt64x4"
				if op, ok := ctx.target.OpMap["AsInt64"]; ok {
					asIntMethod = op.Name
				}
				shift = 52
				mask = "2047" // 0x7FF
				bias = "1023"
			}

			// x.AsInt32()
			expr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   cloneExpr(x),
					Sel: ast.NewIdent(asIntMethod),
				},
			}

			// .ShiftAllRight(shift)
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("ShiftAllRight"),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(shift)}},
			}

			// .And(Broadcast(mask))
			broadcastMask := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: mask}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("And"),
				},
				Args: []ast.Expr{broadcastMask},
			}

			// .Sub(Broadcast(bias))
			broadcastBias := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: bias}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("Sub"),
				},
				Args: []ast.Expr{broadcastBias},
			}

			// NOTE: Don't add .ConvertToFloat() here - let ConvertExponentToFloat handle the
			// int-to-float conversion. This keeps GetExponent returning integers as expected.

			*call = *expr
		}

	case "GetMantissa":
		// For Float16/BFloat16, use hwy.GetMantissa which has proper handling
		if isHalfPrecisionType(ctx.elemType) {
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent("GetMantissa"),
			}
			return
		}
		if len(call.Args) >= 1 {
			x := call.Args[0]
			intVecTypeName := getVectorTypeNameForInt("int32", ctx.elemType, ctx.target)
			if ctx.elemType == "float64" {
				intVecTypeName = getVectorTypeNameForInt("int64", ctx.elemType, ctx.target)
			}
			pkgName := getVecPackageName(ctx.target)

			var asIntMethod string
			var mask string
			var one string
			var asFloatMethod string

			if ctx.elemType == "float32" {
				asIntMethod = "AsInt32x8"
				if op, ok := ctx.target.OpMap["AsInt32"]; ok {
					asIntMethod = op.Name
				}
				mask = "8388607"   // 0x7FFFFF
				one = "1065353216" // 0x3F800000
				asFloatMethod = "AsFloat32x8"
				if op, ok := ctx.target.OpMap["AsFloat32"]; ok {
					asFloatMethod = op.Name
				}
			} else {
				asIntMethod = "AsInt64x4"
				if op, ok := ctx.target.OpMap["AsInt64"]; ok {
					asIntMethod = op.Name
				}
				mask = "4503599627370495"   // 0x000FFFFFFFFFFFFF
				one = "4607182418800017408" // 0x3FF0000000000000
				asFloatMethod = "AsFloat64x4"
				if op, ok := ctx.target.OpMap["AsFloat64"]; ok {
					asFloatMethod = op.Name
				}
			}

			// x.AsInt32()
			expr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   cloneExpr(x),
					Sel: ast.NewIdent(asIntMethod),
				},
			}

			// .And(Broadcast(mask))
			broadcastMask := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: mask}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("And"),
				},
				Args: []ast.Expr{broadcastMask},
			}

			// .Or(Broadcast(one))
			broadcastOne := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: one}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("Or"),
				},
				Args: []ast.Expr{broadcastOne},
			}

			// .AsFloat32()
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent(asFloatMethod),
				},
			}

			*call = *expr
		}

	case "Abs":
		// hwy.Abs(x) -> x.Max(negX) where negX = pkg.Broadcast*(0).Sub(x)
		// archsimd doesn't have Abs method, so we implement |x| = max(x, -x)
		if opInfo.Package == "special" && len(call.Args) >= 1 {
			vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
			pkgName := getVecPackageName(ctx.target)
			x := call.Args[0]
			// Create pkg.Broadcast*(0)
			zeroLit := &ast.BasicLit{Kind: token.INT, Value: "0"}
			zeroCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{zeroLit},
			}
			// Create pkg.Broadcast*(0).Sub(x) = -x
			negX := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   zeroCall,
					Sel: ast.NewIdent("Sub"),
				},
				Args: []ast.Expr{cloneExpr(x)},
			}
			// Create x.Max(-x)
			call.Fun = &ast.SelectorExpr{
				X:   x,
				Sel: ast.NewIdent("Max"),
			}
			call.Args = []ast.Expr{negX}
		} else {
			// Normal Abs method call
			if len(call.Args) >= 1 {
				call.Fun = &ast.SelectorExpr{
					X:   call.Args[0],
					Sel: ast.NewIdent(opInfo.Name),
				}
				call.Args = nil
			}
		}

	case "IsNaN":
		// hwy.IsNaN(x) -> x.Equal(x).Xor(one.Equal(one))
		// NaN != NaN, so x.Equal(x) is false (all 0s) for NaN elements
		// We XOR with all-true mask to invert, giving true for NaN
		if opInfo.Package == "special" && len(call.Args) >= 1 {
			vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
			pkgName := getVecPackageName(ctx.target)
			x := call.Args[0]
			// Create pkg.Broadcast*(1.0)
			oneLit := &ast.BasicLit{Kind: token.FLOAT, Value: "1.0"}
			oneCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{oneLit},
			}
			// Create one.Equal(one) to get all-true mask
			allTrue := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   oneCall,
					Sel: ast.NewIdent("Equal"),
				},
				Args: []ast.Expr{cloneExpr(oneCall)},
			}
			// Create x.Equal(x)
			xEqX := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   x,
					Sel: ast.NewIdent("Equal"),
				},
				Args: []ast.Expr{cloneExpr(x)},
			}
			// Create x.Equal(x).Xor(allTrue) to invert
			call.Fun = &ast.SelectorExpr{
				X:   xEqX,
				Sel: ast.NewIdent("Xor"),
			}
			call.Args = []ast.Expr{allTrue}
		}

	case "MaskNot":
		// hwy.MaskNot(mask) -> mask.Xor(allTrue)
		// where allTrue = one.Equal(one) (comparing 1.0 == 1.0 gives all-true mask)
		if opInfo.Package == "special" && len(call.Args) >= 1 {
			vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
			pkgName := getVecPackageName(ctx.target)
			mask := call.Args[0]

			// Create pkg.Broadcast*(1.0) for float types or 1 for int types
			var oneLit ast.Expr
			if ctx.elemType == "float32" || ctx.elemType == "float64" {
				oneLit = &ast.BasicLit{Kind: token.FLOAT, Value: "1.0"}
			} else {
				oneLit = &ast.BasicLit{Kind: token.INT, Value: "1"}
			}
			oneCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{oneLit},
			}
			// Create one.Equal(one) to get all-true mask
			allTrue := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   oneCall,
					Sel: ast.NewIdent("Equal"),
				},
				Args: []ast.Expr{cloneExpr(oneCall)},
			}
			// Create mask.Xor(allTrue) to invert
			call.Fun = &ast.SelectorExpr{
				X:   mask,
				Sel: ast.NewIdent("Xor"),
			}
			call.Args = []ast.Expr{allTrue}
		}

	case "IsInf":
		// hwy.IsInf(x, sign) -> compare with +Inf and/or -Inf
		// sign=0: either +Inf or -Inf, sign=1: +Inf only, sign=-1: -Inf only
		if opInfo.Package == "special" && len(call.Args) >= 2 {
			vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
			pkgName := getVecPackageName(ctx.target)
			x := call.Args[0]
			signArg := call.Args[1]

			// Determine sign value (0, 1, or -1)
			signVal := 0
			if lit, ok := signArg.(*ast.BasicLit); ok && lit.Kind == token.INT {
				if lit.Value == "1" {
					signVal = 1
				} else if lit.Value == "-1" {
					signVal = -1
				}
			} else if unary, ok := signArg.(*ast.UnaryExpr); ok && unary.Op == token.SUB {
				if lit, ok := unary.X.(*ast.BasicLit); ok && lit.Kind == token.INT && lit.Value == "1" {
					signVal = -1
				}
			}

			// Create math.Inf(1) with type conversion for float32
			posInfExpr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent("stdmath"),
					Sel: ast.NewIdent("Inf"),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "1"}},
			}
			// For float32, wrap in type conversion
			var posInfArg ast.Expr = posInfExpr
			if ctx.elemType == "float32" {
				posInfArg = &ast.CallExpr{
					Fun:  ast.NewIdent("float32"),
					Args: []ast.Expr{posInfExpr},
				}
			}

			// Create pkg.Broadcast*(posInf) for +Inf
			posInfCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{posInfArg},
			}

			// Create math.Inf(-1) with type conversion for float32
			negInfExpr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent("stdmath"),
					Sel: ast.NewIdent("Inf"),
				},
				Args: []ast.Expr{
					&ast.UnaryExpr{
						Op: token.SUB,
						X:  &ast.BasicLit{Kind: token.INT, Value: "1"},
					},
				},
			}
			// For float32, wrap in type conversion
			var negInfArg ast.Expr = negInfExpr
			if ctx.elemType == "float32" {
				negInfArg = &ast.CallExpr{
					Fun:  ast.NewIdent("float32"),
					Args: []ast.Expr{negInfExpr},
				}
			}

			// Create pkg.Broadcast*(negInf) for -Inf
			negInfCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{negInfArg},
			}

			switch signVal {
			case 1:
				// Check +Inf only: x.Equal(posInf)
				call.Fun = &ast.SelectorExpr{
					X:   x,
					Sel: ast.NewIdent("Equal"),
				}
				call.Args = []ast.Expr{posInfCall}
			case -1:
				// Check -Inf only: x.Equal(negInf)
				call.Fun = &ast.SelectorExpr{
					X:   x,
					Sel: ast.NewIdent("Equal"),
				}
				call.Args = []ast.Expr{negInfCall}
			default:
				// Check either: x.Equal(posInf).Or(x.Equal(negInf))
				posInfMask := &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   cloneExpr(x),
						Sel: ast.NewIdent("Equal"),
					},
					Args: []ast.Expr{posInfCall},
				}
				negInfMask := &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   x,
						Sel: ast.NewIdent("Equal"),
					},
					Args: []ast.Expr{negInfCall},
				}
				call.Fun = &ast.SelectorExpr{
					X:   posInfMask,
					Sel: ast.NewIdent("Or"),
				}
				call.Args = []ast.Expr{negInfMask}
			}
		}

	case "ShiftRight", "ShiftLeft", "ShiftAllRight", "ShiftAllLeft":
		// hwy.ShiftRight(v, shift) -> v.ShiftAllRight(uint64(shift))
		// archsimd's ShiftAllRight/ShiftAllLeft expect uint64, but hwy uses int
		if len(call.Args) >= 2 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent(opInfo.Name),
			}
			shiftArg := call.Args[1]
			// Wrap shift in uint64() cast for archsimd targets
			if ctx.target.VecPackage == "archsimd" {
				shiftArg = &ast.CallExpr{
					Fun:  ast.NewIdent("uint64"),
					Args: []ast.Expr{shiftArg},
				}
			}
			call.Args = []ast.Expr{shiftArg}
		}

	case "And", "Xor":
		// archsimd float types don't have And/Xor methods, only int types do.
		// For float types on archsimd, use hwy wrappers.
		// BUT: if both operands are int32 vectors, use method call since Int32x8 has And/Xor.
		if ctx.target.VecPackage == "archsimd" && (ctx.elemType == "float32" || ctx.elemType == "float64") {
			// Check if both operands are int32 - if so, use method call
			if len(call.Args) >= 2 && isInt32Expr(call.Args[0], ctx) && isInt32Expr(call.Args[1], ctx) {
				// Int32x8 has .And() method - use method call a.And(b)
				call.Fun = &ast.SelectorExpr{
					X:   call.Args[0],
					Sel: ast.NewIdent(opInfo.Name),
				}
				call.Args = call.Args[1:]
			} else {
				// hwy.And(a, b) -> hwy.And_AVX2_F32x8(a, b)
				fullName := fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent(fullName),
				}
				// Keep args as [a, b]
			}
		} else if isHalfPrecisionType(ctx.elemType) {
			// For half-precision contexts, integer operations (like octant masking in sin/cos)
			// use hwy.Vec[int32] which doesn't have And method. Keep as generic hwy function.
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent(opInfo.Name),
			}
			// Keep args as [a, b]
		} else {
			// Integer types or non-archsimd: use method call a.And(b)
			if len(call.Args) >= 2 {
				call.Fun = &ast.SelectorExpr{
					X:   call.Args[0],
					Sel: ast.NewIdent(opInfo.Name),
				}
				call.Args = call.Args[1:]
			}
		}

	case "Not":
		// archsimd float types don't have Not method, only int types do.
		// For float types on archsimd, use hwy wrappers.
		if ctx.target.VecPackage == "archsimd" && (ctx.elemType == "float32" || ctx.elemType == "float64") {
			// hwy.Not(a) -> hwy.Not_AVX2_F32x8(a)
			fullName := fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent(fullName),
			}
			// Keep args as [a]
		} else {
			// Integer types or non-archsimd: use method call a.Not()
			if len(call.Args) >= 1 {
				call.Fun = &ast.SelectorExpr{
					X:   call.Args[0],
					Sel: ast.NewIdent(opInfo.Name),
				}
				call.Args = nil
			}
		}

	default:
		// Binary operations: hwy.Add(a, b) -> a.Add(b)
		if len(call.Args) >= 2 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent(opInfo.Name),
			}
			call.Args = call.Args[1:]
		} else if len(call.Args) == 1 {
			// Other unary operations
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent(opInfo.Name),
			}
			call.Args = nil
		}
	}
}

// transformToFunction converts hwy.Load(src) to archsimd.LoadFloat32x8Slice(src).
func transformToFunction(call *ast.CallExpr, funcName string, opInfo OpInfo, ctx *transformContext) {
	// Handle both SelectorExpr (hwy.Load) and IndexExpr (hwy.Zero[float32])
	var selExpr *ast.SelectorExpr
	var explicitTypeParam string // Explicit type parameter from hwy.Load[uint8] style calls
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		selExpr = fun
	case *ast.IndexExpr:
		// For generic functions like hwy.Load[uint8]() or hwy.Zero[float32]()
		selExpr = fun.X.(*ast.SelectorExpr)
		// Extract explicit type parameter if it's a concrete type (not a generic type param like T)
		if typeIdent, ok := fun.Index.(*ast.Ident); ok {
			typeName := typeIdent.Name
			// Check if this is a concrete type, not a generic type parameter
			isTypeParam := false
			for _, tp := range ctx.typeParams {
				if typeName == tp.Name {
					isTypeParam = true
					break
				}
			}
			if !isTypeParam {
				// This is an explicit concrete type like uint8, float32, etc.
				explicitTypeParam = typeName
			}
		}
	default:
		return
	}

	if ctx.target.Name == "Fallback" {
		// For fallback, use the appropriate package
		if opInfo.SubPackage != "" {
			// Contrib functions use their subpackage with target suffix
			// e.g., contrib.Sigmoid -> math.BaseSigmoidVec_fallback
			selExpr.X = ast.NewIdent(opInfo.SubPackage)
			fullName := fmt.Sprintf("%s_%s%s", opInfo.Name, strings.ToLower(ctx.target.Name), getHwygenTypeSuffix(ctx.elemType))
			selExpr.Sel.Name = fullName
		} else {
			// Core ops use hwy package
			selExpr.X = ast.NewIdent("hwy")
			// Use opInfo.Name if it differs from the source funcName (e.g., ShiftAllRight -> ShiftRight)
			if opInfo.Name != "" {
				selExpr.Sel.Name = opInfo.Name
			} else {
				selExpr.Sel.Name = funcName
			}
		}
		return
	}

	// For Float16/BFloat16 on SIMD targets, use hwy package functions instead of archsimd calls.
	// archsimd doesn't have native support for half-precision types.
	if isHalfPrecisionType(ctx.elemType) {
		// Handle Merge specially - needs argument reordering
		// hwy.Merge(yes, no, mask) -> hwy.IfThenElseF16(mask, yes, no)
		if funcName == "Merge" && len(call.Args) >= 3 {
			suffix := "F16"
			if isBFloat16Type(ctx.elemType) {
				suffix = "BF16"
			}
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "IfThenElse" + suffix
			// Reorder: (yes, no, mask) -> (mask, yes, no)
			call.Args = []ast.Expr{call.Args[2], call.Args[0], call.Args[1]}
			return
		}
		// Handle IfThenElse - same signature as IfThenElseF16, no reordering needed
		// hwy.IfThenElse(mask, yes, no) -> hwy.IfThenElseF16(mask, yes, no)
		if funcName == "IfThenElse" && len(call.Args) >= 3 {
			suffix := "F16"
			if isBFloat16Type(ctx.elemType) {
				suffix = "BF16"
			}
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "IfThenElse" + suffix
			// Args stay in same order
			return
		}

		if f16FuncName := getHalfPrecisionFuncName(funcName, ctx.elemType); f16FuncName != "" {
			// For comparison operations, check if operands are int32 - if so, keep generic hwy function
			if isComparisonOp(funcName) && len(call.Args) >= 2 {
				if isInt32Expr(call.Args[0], ctx) && isInt32Expr(call.Args[1], ctx) {
					// Keep as generic hwy.Equal, hwy.Greater, etc. for int32 operands
					selExpr.X = ast.NewIdent("hwy")
					selExpr.Sel.Name = funcName
					return
				}
			}
			// Transform to hwy.AddF16(a, b), hwy.MulF16(a, b), etc.
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = f16FuncName
			return
		}
		// For Load/Store/Set/Zero on F16/BF16, use generic hwy functions
		switch funcName {
		case "Load":
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "Load"
			return
		case "Store":
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "Store"
			return
		case "Set":
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "Set"
			// Note: For half-precision, argument wrapping is handled in
			// transformHalfPrecisionFallback after scalar variables are tracked.
			return
		case "Zero":
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "Zero"
			return
		case "RoundToEven":
			// hwy.RoundToEven doesn't have an F16/BF16 variant, use generic
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "RoundToEven"
			return
		case "ConvertToInt32":
			// hwy.ConvertToInt32 doesn't have an F16/BF16 variant, use generic
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "ConvertToInt32"
			return
		case "ConvertExponentToFloat":
			// For Float16/BFloat16, use dedicated conversion functions
			if ctx.elemType == "hwy.Float16" {
				selExpr.X = ast.NewIdent("hwy")
				selExpr.Sel.Name = "ConvertToF16"
			} else {
				selExpr.X = ast.NewIdent("hwy")
				selExpr.Sel.Name = "ConvertToBF16"
			}
			// Strip the type parameter if present
			if indexExpr, ok := call.Fun.(*ast.IndexExpr); ok {
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent(selExpr.Sel.Name),
				}
				_ = indexExpr // used to strip type param
			}
			return
		case "Pow2":
			// Pow2 needs a type parameter: hwy.Pow2[hwy.Float16](kInt)
			call.Fun = &ast.IndexExpr{
				X: &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent("Pow2"),
				},
				Index: ast.NewIdent(ctx.elemType),
			}
			return
		case "Const":
			// Keep hwy.Const[T] with type parameter for half-precision types
			// hwy.Const handles float64-to-T conversion, while hwy.Set expects T
			// This is handled earlier in the IndexExpr case - just return here
			return
		case "And", "Or", "Xor", "Not", "AndNot":
			// Bitwise operations on hwy.Vec[Float16/BFloat16] don't have method forms,
			// so keep them as hwy.And, hwy.Or, hwy.Xor, hwy.Not, hwy.AndNot
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = funcName
			return
		case "NotEqual":
			// hwy.NotEqual on hwy.Vec[Float16/BFloat16] doesn't have a method form,
			// keep as hwy.NotEqual(a, b)
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "NotEqual"
			return
		case "Pow":
			// hwy.Pow on hwy.Vec[Float16/BFloat16] doesn't have a method form,
			// keep as hwy.Pow(base, exp)
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "Pow"
			return
		case "SignBit":
			// For half-precision types, use hwy.SignBit[T]() which returns hwy.Vec[T]
			// The generic function handles the sign bit correctly for Float16/BFloat16
			call.Fun = &ast.IndexExpr{
				X: &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent("SignBit"),
				},
				Index: ast.NewIdent(ctx.elemType),
			}
			return
		}
		// For other operations without F16/BF16 variants, fall through
	}

	// For SIMD targets, transform to package calls (archsimd for AVX, asm for NEON)
	var fullName string
	// Use explicit type parameter if present (e.g., hwy.Load[uint8]), otherwise use function's elemType
	effectiveElemType := ctx.elemType
	if explicitTypeParam != "" {
		effectiveElemType = explicitTypeParam
	}
	vecTypeName := getVectorTypeName(effectiveElemType, ctx.target)
	pkgName := getVecPackageName(ctx.target)

	// Check if this op should be redirected to hwy wrappers (archsimd doesn't have it)
	if opInfo.Package == "hwy" && opInfo.SubPackage == "" {
		// Use hwy wrapper instead of archsimd
		// Try to infer lanes and element type from any argument (for operations like TableLookupBytes)
		shortTypeName := getShortTypeName(effectiveElemType, ctx.target)
		inferredLanes := 0
		inferredElemType := effectiveElemType
		for _, arg := range call.Args {
			if argIdent, ok := arg.(*ast.Ident); ok {
				if lanes, found := ctx.varVecLanes[argIdent.Name]; found {
					inferredLanes = lanes
					// Also check if we have an element type for this variable
					if elemType, hasType := ctx.varVecElemType[argIdent.Name]; hasType {
						inferredElemType = elemType
					}
					break // Use the first known lanes
				}
			}
		}
		if inferredLanes > 0 {
			shortTypeName = getShortTypeNameForLanes(inferredElemType, inferredLanes)
		} else if ctx.inferredFuncLanes > 0 {
			// Fall back to function-level inferred lanes (from Load calls)
			// Cap at target's max lanes to avoid generating invalid types (e.g., Uint8x32 on NEON)
			useLanes := ctx.inferredFuncLanes
			targetLanes := ctx.target.LanesFor(effectiveElemType)
			if useLanes > targetLanes {
				useLanes = targetLanes
			}
			shortTypeName = getShortTypeNameForLanes(effectiveElemType, useLanes)
		}
		fullName = fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, shortTypeName)
		selExpr.X = ast.NewIdent("hwy")
		selExpr.Sel.Name = fullName
		// Strip the IndexExpr if call.Fun was hwy.Func[T]() - the wrapper doesn't use type params
		call.Fun = selExpr
		return
	}

	switch funcName {
	case "Load":
		// Check if we can determine the slice size from the argument
		// For example, hwy.Load(data[:16]) with uint8 should use Uint8x16, not Uint8x32
		loadVecTypeName := vecTypeName
		if len(call.Args) > 0 {
			sliceBytes := getSliceSize(call.Args[0])
			elemSize := elemTypeSize(effectiveElemType)
			targetLanes := ctx.target.LanesFor(effectiveElemType)
			if sliceBytes > 0 && elemSize > 0 {
				detectedLanes := sliceBytes / elemSize
				// Only use smaller type if detected lanes is less than target default
				// and is a valid vector size (power of 2, typically 2, 4, 8, 16, 32, 64)
				if detectedLanes < targetLanes && detectedLanes > 0 {
					loadVecTypeName = getVectorTypeNameForLanes(effectiveElemType, detectedLanes)
				}
			} else if ctx.inferredFuncLanes > 0 && ctx.inferredFuncLanes < targetLanes {
				// No explicit size, but we have inferred lanes from earlier in the function
				loadVecTypeName = getVectorTypeNameForLanes(effectiveElemType, ctx.inferredFuncLanes)
			}
		}
		fullName = fmt.Sprintf("Load%sSlice", loadVecTypeName)
		selExpr.X = ast.NewIdent(pkgName)
	case "Load4":
		// For Vec types (Float16/BFloat16), use hwy wrapper since asm doesn't have Load4VecSlice
		if strings.HasPrefix(vecTypeName, "Vec") || strings.HasPrefix(vecTypeName, "hwy.Vec") {
			fullName = fmt.Sprintf("Load4_%s_Vec", ctx.target.Name)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			// For NEON: asm.Load4Float32x4Slice (single ld1 instruction)
			// For AVX2/512/Fallback: handled by hwy wrapper at line 2094-2100
			fullName = fmt.Sprintf("Load4%sSlice", vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "Set", "Const":
		// Both Set and Const broadcast a scalar value to all lanes
		fullName = fmt.Sprintf("Broadcast%s", vecTypeName)
		selExpr.X = ast.NewIdent(pkgName)
	case "Zero":
		if opInfo.Package == "special" {
			// archsimd doesn't have Zero*, use Broadcast with 0
			fullName = fmt.Sprintf("Broadcast%s", vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)
			// Add 0 as argument
			call.Args = []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "0"}}
		} else {
			fullName = fmt.Sprintf("Zero%s", vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "SlideUpLanes":
		// For NEON: hwy.SlideUpLanes(v, offset) -> asm.SlideUpLanesFloat32x4(v, offset)
		// For AVX2/AVX512: hwy.SlideUpLanes(v, offset) -> hwy.SlideUpLanes_AVX2_F32x8(v, offset)
		if ctx.target.Name == "AVX2" || ctx.target.Name == "AVX512" {
			shortTypeName := getShortTypeName(ctx.elemType, ctx.target)
			fullName = fmt.Sprintf("SlideUpLanes_%s_%s", ctx.target.Name, shortTypeName)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			fullName = fmt.Sprintf("SlideUpLanes%s", vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "SlideDownLanes":
		// For NEON: hwy.SlideDownLanes(v, offset) -> asm.SlideDownLanesFloat32x4(v, offset)
		// For AVX2/AVX512: hwy.SlideDownLanes(v, offset) -> hwy.SlideDownLanes_AVX2_F32x8(v, offset)
		if ctx.target.Name == "AVX2" || ctx.target.Name == "AVX512" {
			shortTypeName := getShortTypeName(ctx.elemType, ctx.target)
			fullName = fmt.Sprintf("SlideDownLanes_%s_%s", ctx.target.Name, shortTypeName)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			fullName = fmt.Sprintf("SlideDownLanes%s", vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "InsertLane":
		// hwy.InsertLane(v, idx, val) -> asm.InsertLaneFloat32x4(v, idx, val)
		fullName = fmt.Sprintf("InsertLane%s", vecTypeName)
		selExpr.X = ast.NewIdent(pkgName)
	case "MaskLoad":
		fullName = fmt.Sprintf("MaskLoad%sSlice", vecTypeName)
		selExpr.X = ast.NewIdent(pkgName)
	case "Compress":
		// Use hwy wrapper if configured
		if opInfo.Package == "hwy" {
			fullName = fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
			selExpr.X = ast.NewIdent("hwy")
		} else {
			// Compress returns (Vec, int). Maps to CompressKeysF32x4, etc.
			switch ctx.elemType {
			case "float32":
				fullName = "CompressKeysF32x4"
			case "float64":
				fullName = "CompressKeysF64x2"
			case "int32":
				fullName = "CompressKeysI32x4"
			case "int64":
				fullName = "CompressKeysI64x2"
			case "uint32":
				fullName = "CompressKeysU32x4"
			case "uint64":
				fullName = "CompressKeysU64x2"
			default:
				fullName = "CompressKeysF32x4"
			}
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "CompressStore":
		// Use hwy wrapper if configured
		if opInfo.Package == "hwy" {
			fullName = fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
			selExpr.X = ast.NewIdent("hwy")
		} else {
			// CompressStore has type-specific versions: CompressStore (float32), CompressStoreFloat64, etc.
			switch ctx.elemType {
			case "float32":
				fullName = "CompressStore"
			case "float64":
				fullName = "CompressStoreFloat64"
			case "int32":
				fullName = "CompressStoreInt32"
			case "int64":
				fullName = "CompressStoreInt64"
			case "uint32":
				fullName = "CompressStoreUint32"
			case "uint64":
				fullName = "CompressStoreUint64"
			default:
				fullName = "CompressStore"
			}
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "FirstN":
		// Use hwy wrapper if configured
		if opInfo.Package == "hwy" {
			fullName = fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
			selExpr.X = ast.NewIdent("hwy")
		} else {
			// FirstN returns a mask type: Int32x4 for 4-lane, Int64x2 for 2-lane
			switch ctx.elemType {
			case "float32":
				fullName = "FirstN"
			case "float64":
				fullName = "FirstNFloat64"
			case "int32", "uint32":
				fullName = "FirstN" // Int32x4 mask for 32-bit types
			case "int64", "uint64":
				fullName = "FirstNInt64" // Int64x2 mask for 64-bit types
			default:
				fullName = "FirstN"
			}
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "IfThenElse":
		// Use hwy wrapper if configured
		if opInfo.Package == "hwy" {
			fullName = fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
			selExpr.X = ast.NewIdent("hwy")
		} else {
			// IfThenElse has type-specific versions for NEON
			switch ctx.elemType {
			case "float32":
				fullName = "IfThenElse"
			case "float64":
				fullName = "IfThenElseFloat64"
			case "int32":
				fullName = "IfThenElseInt32"
			case "int64":
				fullName = "IfThenElseInt64"
			default:
				fullName = "IfThenElse"
			}
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "AllTrue":
		// AllTrue has type-specific versions for inlining:
		// AllTrueVal for Int32x4 masks, AllTrueValFloat64 for Int64x2 masks
		switch ctx.elemType {
		case "float32", "int32":
			fullName = "AllTrueVal"
		case "float64", "int64":
			fullName = "AllTrueValFloat64"
		case "uint32":
			fullName = "AllTrueValUint32"
		case "uint64":
			fullName = "AllTrueValUint64"
		default:
			fullName = "AllTrueVal"
		}
		selExpr.X = ast.NewIdent(pkgName)
	case "AllFalse":
		// AllFalse has type-specific versions for inlining:
		// AllFalseVal for Int32x4 masks, AllFalseValFloat64 for Int64x2 masks
		switch ctx.elemType {
		case "float32", "int32":
			fullName = "AllFalseVal"
		case "float64", "int64":
			fullName = "AllFalseValFloat64"
		case "uint32":
			fullName = "AllFalseValUint32"
		case "uint64":
			fullName = "AllFalseValUint64"
		default:
			fullName = "AllFalseVal"
		}
		selExpr.X = ast.NewIdent(pkgName)
	case "SignBit":
		// SignBit has type-specific versions for NEON: SignBitFloat32x4, SignBitFloat64x2
		// For AVX2/AVX512, archsimd.SignBit() is generic
		if ctx.target.Name == "NEON" {
			switch ctx.elemType {
			case "float32":
				fullName = "SignBitFloat32x4"
			case "float64":
				fullName = "SignBitFloat64x2"
			default:
				fullName = "SignBitFloat32x4"
			}
		} else {
			fullName = "SignBit"
		}
		selExpr.X = ast.NewIdent(pkgName)
	case "Iota":
		// Iota needs target-specific handling since archsimd doesn't have a generic Iota.
		// NEON: type-specific asm functions (IotaFloat32x4, IotaFloat64x2, etc.)
		// AVX2/AVX512: hwy wrapper functions (Iota_AVX2_F32x8, Iota_AVX512_F32x16, etc.)
		// Float16/BFloat16 on any target: hwy.Iota[T]() generic function
		if isHalfPrecisionType(effectiveElemType) {
			// Half-precision types use hwy.Iota[T]() on all targets
			call.Fun = &ast.IndexExpr{
				X: &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent("Iota"),
				},
				Index: ast.NewIdent(ctx.elemType),
			}
			return
		}
		if ctx.target.Name == "NEON" {
			switch ctx.elemType {
			case "float32":
				fullName = "IotaFloat32x4"
			case "float64":
				fullName = "IotaFloat64x2"
			case "uint32":
				fullName = "IotaUint32x4"
			case "uint64":
				fullName = "IotaUint64x2"
			default:
				fullName = "Iota"
			}
			selExpr.X = ast.NewIdent(pkgName)
		} else if ctx.target.VecPackage == "archsimd" {
			// AVX2/AVX512: use hwy.Iota_{target}_{shortType}()
			shortTypeName := getShortTypeName(effectiveElemType, ctx.target)
			fullName = fmt.Sprintf("Iota_%s_%s", ctx.target.Name, shortTypeName)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			fullName = opInfo.Name
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "MaskNot":
		// MaskNot(mask) -> mask.Xor(allTrue)
		// where allTrue = one.Equal(one) (comparing 1.0 == 1.0 gives all-true mask)
		if opInfo.Package == "special" && len(call.Args) >= 1 {
			vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
			mask := call.Args[0]

			// Create pkg.Broadcast*(1.0) for float types or 1 for int types
			var oneLit ast.Expr
			if ctx.elemType == "float32" || ctx.elemType == "float64" {
				oneLit = &ast.BasicLit{Kind: token.FLOAT, Value: "1.0"}
			} else {
				oneLit = &ast.BasicLit{Kind: token.INT, Value: "1"}
			}
			oneCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{oneLit},
			}
			// Create one.Equal(one) to get all-true mask
			allTrue := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   oneCall,
					Sel: ast.NewIdent("Equal"),
				},
				Args: []ast.Expr{cloneExpr(oneCall)},
			}
			// Create mask.Xor(allTrue) to invert
			call.Fun = &ast.SelectorExpr{
				X:   mask,
				Sel: ast.NewIdent("Xor"),
			}
			call.Args = []ast.Expr{allTrue}
		}
		return // Don't set fullName, we've already transformed the call
	case "ShiftRight", "ShiftLeft", "ShiftAllRight", "ShiftAllLeft":
		// archsimd's ShiftAllRight/ShiftAllLeft expect uint64, but hwy uses int.
		// After function-to-method transformation, the shift is the last arg.
		// Wrap it in a uint64() cast for archsimd targets.
		if ctx.target.VecPackage == "archsimd" && len(call.Args) >= 1 {
			lastIdx := len(call.Args) - 1
			call.Args[lastIdx] = &ast.CallExpr{
				Fun:  ast.NewIdent("uint64"),
				Args: []ast.Expr{call.Args[lastIdx]},
			}
		}
		fullName = opInfo.Name
		selExpr.X = ast.NewIdent(pkgName)
	case "ConvertExponentToFloat":
		// Convert Vec[int32] to Vec[T] for the target float type
		// For native float types, transform to e.ConvertToFloat32() method call
		if len(call.Args) >= 1 {
			var methodName string
			switch ctx.elemType {
			case "float32":
				methodName = "ConvertToFloat32"
			case "float64":
				methodName = "ConvertToFloat64"
			default:
				// Half-precision handled earlier in the isHalfPrecisionType block
				methodName = "ConvertToFloat32"
			}
			// Transform hwy.ConvertExponentToFloat[T](e) to e.ConvertToFloat32()
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent(methodName),
			}
			call.Args = nil
		}
		return
	default:
		// For contrib functions (SubPackage), use hwygen's naming convention:
		// lowercase target, type suffix only for non-default types
		// e.g., math.BaseExpVec_avx2, math.BaseExpVec_avx2_Float64
		if opInfo.SubPackage != "" {
			fullName = fmt.Sprintf("%s_%s%s", opInfo.Name, strings.ToLower(ctx.target.Name), getHwygenTypeSuffix(ctx.elemType))
			selExpr.X = ast.NewIdent(opInfo.SubPackage) // math, vec, matvec, algo
		} else if opInfo.Package == "hwy" {
			// Core ops from hwy package (e.g., hwy.Sqrt_AVX2_F32x8)
			fullName = fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
			selExpr.X = ast.NewIdent("hwy")
		} else {
			fullName = opInfo.Name
			selExpr.X = ast.NewIdent(pkgName)
		}
	}

	selExpr.Sel.Name = fullName

	// If call.Fun is an IndexExpr (from explicit type param like hwy.Load[uint8]),
	// strip the IndexExpr since asm/archsimd package functions don't use type params
	if _, ok := call.Fun.(*ast.IndexExpr); ok {
		call.Fun = selExpr
	}
}

// getVecPackageName returns the package name for vector types based on target.
// Returns "archsimd" for AVX targets, "asm" for NEON.
func getVecPackageName(target Target) string {
	switch target.VecPackage {
	case "archsimd":
		return "archsimd"
	case "asm":
		return "asm"
	default:
		return "archsimd" // default for compatibility
	}
}

// getShortTypeName returns the short type name like F32x8 for contrib functions.
func getShortTypeName(elemType string, target Target) string {
	lanes := target.LanesFor(elemType)
	return getShortTypeNameForLanes(elemType, lanes)
}

// getShortTypeNameForLanes returns the short type name for a specific lane count.
func getShortTypeNameForLanes(elemType string, lanes int) string {
	switch elemType {
	case "float32":
		return fmt.Sprintf("F32x%d", lanes)
	case "float64":
		return fmt.Sprintf("F64x%d", lanes)
	case "int32":
		return fmt.Sprintf("I32x%d", lanes)
	case "int64":
		return fmt.Sprintf("I64x%d", lanes)
	case "uint8":
		return fmt.Sprintf("Uint8x%d", lanes)
	case "uint16":
		return fmt.Sprintf("Uint16x%d", lanes)
	case "uint32":
		return fmt.Sprintf("Uint32x%d", lanes)
	case "uint64":
		return fmt.Sprintf("Uint64x%d", lanes)
	default:
		return "Vec"
	}
}

// getHwygenTypeSuffix returns the type suffix used by hwygen for generated functions.
// float32 is the default (no suffix), other types get _Float64, _Int32, _Int64.
func getHwygenTypeSuffix(elemType string) string {
	switch elemType {
	case "float32":
		return "" // default type, no suffix
	case "float64":
		return "_Float64"
	case "int32":
		return "_Int32"
	case "int64":
		return "_Int64"
	default:
		return ""
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
			specialized = specializeVecType(specialized, ctx.elemType, ctx.target)
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
	if ctx.target.Name == "Fallback" {
		return
	}

	// Look for v.NumElements(), hwy.Lanes[T](), or similar and replace with constant
	for i, rhs := range stmt.Rhs {
		if call, ok := rhs.(*ast.CallExpr); ok {
			// Check for hwy.Lanes[T]() - IndexExpr wrapping SelectorExpr
			if indexExpr, ok := call.Fun.(*ast.IndexExpr); ok {
				if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
					if pkgIdent, ok := sel.X.(*ast.Ident); ok {
						if pkgIdent.Name == "hwy" && (sel.Sel.Name == "Lanes" || sel.Sel.Name == "MaxLanes") {
							// Replace with constant lane count
							lanes := ctx.target.LanesFor(ctx.elemType)
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
					// Replace with constant lane count
					lanes := ctx.target.LanesFor(ctx.elemType)
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
		if !ok || ident.Name != "hwy" || selExpr.Sel.Name != "Load" {
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

	// Verify it's hwy.Set
	ident, ok := selExpr.X.(*ast.Ident)
	if !ok || ident.Name != "hwy" {
		return ""
	}
	if selExpr.Sel.Name != "Set" {
		return ""
	}

	// Determine the actual element type for this Set call
	// If the type parameter is explicitly "int32", use that instead of ctx.elemType
	actualElemType := ctx.elemType
	if typeParam == "int32" {
		actualElemType = "int32"
		// For half-precision types, don't hoist int32 constants to native SIMD types
		// because hwy.ConvertToInt32 returns hwy.Vec[int32], not native SIMD types.
		// Keeping them as hwy.Set[int32] ensures type compatibility.
		if isHalfPrecisionType(ctx.elemType) {
			return ""
		}
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
		// Variable reference - only hoist if it's NOT a local variable
		name := e.Name
		if ctx.localVars[name] {
			// This is a locally-defined variable, not a package-level constant
			return ""
		}
		if isLikelyConstant(name) {
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
		// Skip if it's a locally-defined variable
		if ctx != nil && ctx.localVars[name] {
			return ""
		}
		if isLikelyConstant(name) {
			return name
		}
	}
	return ""
}

// isLikelyConstant checks if a name looks like a package-level constant.
// This is a heuristic - constants typically have specific naming patterns.
// Note: This function is only called AFTER checking that the name is not
// in ctx.localVars, so locally-defined variables are already excluded.
func isLikelyConstant(name string) bool {
	// Skip very common short names that are almost never constants
	skipNames := map[string]bool{
		"i": true, "j": true, "k": true, "ii": true, "jj": true,
		"x": true, "y": true, "z": true, "n": true, "m": true,
		"a": true, "b": true, "c": true, "v": true, "w": true,
		"err": true, "ok": true,
	}
	if skipNames[name] {
		return false
	}

	// Constants typically:
	// 1. Contain digits (e.g., sigmoidC1, ln2Hi, exp2bias, c1, c2)
	// 2. Are all uppercase (e.g., PI, MAX_VALUE)
	hasDigit := false
	hasLower := false
	hasUpper := false
	for _, r := range name {
		if r >= '0' && r <= '9' {
			hasDigit = true
		} else if r >= 'a' && r <= 'z' {
			hasLower = true
		} else if r >= 'A' && r <= 'Z' {
			hasUpper = true
		}
	}

	// Accept if it has a digit (like sigmoidC1, ln2Hi)
	if hasDigit {
		return true
	}

	// Accept if all uppercase (like PI, MAX_VALUE)
	if hasUpper && !hasLower {
		return true
	}

	// Reject short lowercase names that look like local variables
	if len(name) <= 4 && hasLower {
		return false
	}

	// Accept longer camelCase names that look like package-level constants
	// (e.g., sigmoidScale, expBias, tanhClamp)
	return len(name) > 4
}

// matchesLoopIterator checks if a for loop uses the given iterator name.
// It checks both the init statement (for ii := 0) and the condition (ii < size).
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

	// Check condition: ii < size or ii+N <= size
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
	if target.Name == "Fallback" {
		return
	}

	// Count SIMD loops that use the same iterator. If there are multiple SIMD loops,
	// the function has a multi-phase algorithm (e.g., Normalize: accumulate then scale)
	// and automatic tail handling would break the data dependencies between phases.
	// In such cases, the template must handle tails manually.
	simdLoopCount := 0
	for _, stmt := range body.List {
		if forStmt, ok := stmt.(*ast.ForStmt); ok {
			if matchesLoopIterator(forStmt, loopInfo.Iterator) && isSimdStyleLoop(forStmt, loopInfo) {
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

		// Skip built-in identifiers
		builtins := map[string]bool{
			"true": true, "false": true, "nil": true,
			"int": true, "int8": true, "int16": true, "int32": true, "int64": true,
			"uint": true, "uint8": true, "uint16": true, "uint32": true, "uint64": true,
			"float32": true, "float64": true, "complex64": true, "complex128": true,
			"string": true, "bool": true, "byte": true, "rune": true,
			"len": true, "cap": true, "make": true, "new": true, "append": true,
			"copy": true, "delete": true, "panic": true, "recover": true, "close": true,
			"print": true, "println": true,
		}
		if builtins[name] {
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
func isSimdStyleLoop(forStmt *ast.ForStmt, loopInfo *LoopInfo) bool {
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

// specializeType replaces generic type parameters with concrete types.
// For SIMD targets, also transforms hwy.Vec[T] to archsimd/asm vector types.
func specializeType(typeStr string, typeParams []TypeParam, elemType string) string {
	// First, identify which type parameters are element types vs interface types
	elementTypeParams := make(map[string]bool)
	interfaceTypeParams := make(map[string]string) // maps param name to constraint

	for _, tp := range typeParams {
		// Element type constraints
		if strings.Contains(tp.Constraint, "Lanes") ||
			strings.Contains(tp.Constraint, "Floats") ||
			strings.Contains(tp.Constraint, "Integers") ||
			strings.Contains(tp.Constraint, "SignedInts") ||
			strings.Contains(tp.Constraint, "UnsignedInts") {
			elementTypeParams[tp.Name] = true
		} else {
			// Interface constraint (like Predicate[T])
			interfaceTypeParams[tp.Name] = tp.Constraint
		}
	}

	// Replace element type parameters and hwy.Vec[T]/hwy.Mask[T]
	for _, tp := range typeParams {
		if elementTypeParams[tp.Name] {
			// Replace hwy.Vec[T] with concrete vector type placeholder
			typeStr = strings.ReplaceAll(typeStr, "hwy.Vec["+tp.Name+"]", "hwy.Vec["+elemType+"]")
			// Replace hwy.Mask[T] with concrete mask type placeholder
			typeStr = strings.ReplaceAll(typeStr, "hwy.Mask["+tp.Name+"]", "hwy.Mask["+elemType+"]")
			// Replace []T with []float32, etc.
			typeStr = strings.ReplaceAll(typeStr, "[]"+tp.Name, "[]"+elemType)
			// Replace standalone T with concrete type
			typeStr = replaceTypeParam(typeStr, tp.Name, elemType)
		}
	}

	// For interface type parameters, specialize the generic type within the constraint
	// e.g., Predicate[T] -> Predicate[float32]
	for paramName, constraint := range interfaceTypeParams {
		// Check if this parameter's type is exactly its constraint (e.g., "P" -> "Predicate[T]")
		if typeStr == paramName {
			// Specialize the constraint's type parameters
			specializedConstraint := constraint
			for _, tp := range typeParams {
				if elementTypeParams[tp.Name] {
					specializedConstraint = strings.ReplaceAll(specializedConstraint, "["+tp.Name+"]", "["+elemType+"]")
				}
			}
			typeStr = specializedConstraint
		}
	}

	return typeStr
}

// replaceTypeParam replaces a type parameter name with a concrete type,
// being careful to only replace it when it appears as a standalone type
// (not as part of another identifier).
func replaceTypeParam(typeStr, paramName, elemType string) string {
	// Simple approach: replace when the parameter appears alone or as a type argument
	// This handles cases like "T", "[T]", "[]T", "func(T)"
	result := typeStr

	// Replace [T] with [elemType]
	result = strings.ReplaceAll(result, "["+paramName+"]", "["+elemType+"]")

	// Replace T when it's the whole string
	if result == paramName {
		return elemType
	}

	// Replace T in slice types []T
	result = strings.ReplaceAll(result, "[]"+paramName, "[]"+elemType)

	// Replace T in function types and map value types - look for patterns like "T)" or "T," or "(T" or "]T"
	// This is a simple heuristic that works for most cases
	for _, suffix := range []string{")", ",", " ", ""} {
		for _, prefix := range []string{"(", ",", " ", "]"} {
			old := prefix + paramName + suffix
			new := prefix + elemType + suffix
			result = strings.ReplaceAll(result, old, new)
		}
	}

	return result
}

// specializeVecType transforms hwy.Vec[elemType] and hwy.Mask[elemType] to concrete archsimd/asm types.
// For example: hwy.Vec[float32] -> archsimd.Float32x8 (for AVX2)
//
//	hwy.Mask[float32] -> archsimd.Int32x8 (for AVX2)
func specializeVecType(typeStr string, elemType string, target Target) string {
	if target.Name == "Fallback" {
		// For fallback, keep hwy.Vec[float32], hwy.Mask[float32] etc.
		return typeStr
	}

	// For Float16/BFloat16 on SIMD targets, keep hwy.Vec[hwy.Float16] etc.
	// since archsimd doesn't have native support for half-precision types.
	if isHalfPrecisionType(elemType) {
		// Keep the hwy generic Vec/Mask types as-is
		return typeStr
	}

	pkgName := target.VecPackage
	if pkgName == "" {
		pkgName = "archsimd" // default
	}

	// Transform hwy.Vec[elemType]
	vecPlaceholder := "hwy.Vec[" + elemType + "]"
	if strings.Contains(typeStr, vecPlaceholder) {
		vecTypeName, ok := target.TypeMap[elemType]
		if ok {
			concreteType := pkgName + "." + vecTypeName
			typeStr = strings.ReplaceAll(typeStr, vecPlaceholder, concreteType)
		}
	}

	// Transform hwy.Mask[elemType] to integer vector type (masks are represented as integer vectors)
	maskPlaceholder := "hwy.Mask[" + elemType + "]"
	if strings.Contains(typeStr, maskPlaceholder) {
		maskTypeName := getMaskTypeName(elemType, target)
		if maskTypeName != "" {
			concreteMaskType := pkgName + "." + maskTypeName
			typeStr = strings.ReplaceAll(typeStr, maskPlaceholder, concreteMaskType)
		}
	}

	return typeStr
}

// getMaskTypeName returns the mask type name for a given element type and target.
// For archsimd (AVX2/AVX512), masks are dedicated Mask32xN or Mask64xN types.
// For NEON and fallback, masks may use integer vector types.
func getMaskTypeName(elemType string, target Target) string {
	lanes := target.LanesFor(elemType)
	// For archsimd targets, use proper Mask types
	if target.VecPackage == "archsimd" {
		switch elemType {
		case "float32", "int32", "uint32":
			return fmt.Sprintf("Mask32x%d", lanes)
		case "float64", "int64", "uint64":
			return fmt.Sprintf("Mask64x%d", lanes)
		default:
			return ""
		}
	}
	// For other targets (NEON, fallback), use integer vector types
	switch elemType {
	case "float32":
		return fmt.Sprintf("Int32x%d", lanes)
	case "float64":
		return fmt.Sprintf("Int64x%d", lanes)
	case "int32", "uint32":
		return fmt.Sprintf("Int32x%d", lanes)
	case "int64", "uint64":
		return fmt.Sprintf("Int64x%d", lanes)
	default:
		return ""
	}
}

// getVectorTypeName returns the vector type name for archsimd functions.
func getVectorTypeName(elemType string, target Target) string {
	lanes := target.LanesFor(elemType)
	return getVectorTypeNameForLanes(elemType, lanes)
}

// getVectorTypeNameForLanes returns the vector type name for a specific lane count.
func getVectorTypeNameForLanes(elemType string, lanes int) string {
	switch elemType {
	case "float32":
		return fmt.Sprintf("Float32x%d", lanes)
	case "float64":
		return fmt.Sprintf("Float64x%d", lanes)
	case "int32":
		return fmt.Sprintf("Int32x%d", lanes)
	case "int64":
		return fmt.Sprintf("Int64x%d", lanes)
	case "uint8":
		return fmt.Sprintf("Uint8x%d", lanes)
	case "uint16":
		return fmt.Sprintf("Uint16x%d", lanes)
	case "uint32":
		return fmt.Sprintf("Uint32x%d", lanes)
	case "uint64":
		return fmt.Sprintf("Uint64x%d", lanes)
	default:
		return "Vec"
	}
}

// getSliceSize extracts the size from a slice expression like data[:16], data[0:16], or data[1:17].
// Returns 0 if the size cannot be determined.
func getSliceSize(expr ast.Expr) int {
	sliceExpr, ok := expr.(*ast.SliceExpr)
	if !ok {
		return 0
	}
	// Need a high bound to determine size
	if sliceExpr.High == nil {
		return 0
	}
	highLit, ok := sliceExpr.High.(*ast.BasicLit)
	if !ok || highLit.Kind != token.INT {
		return 0
	}
	high, err := strconv.Atoi(highLit.Value)
	if err != nil {
		return 0
	}
	// If there's a low bound, subtract it from high to get actual size
	// For [:N] or [0:N], size is N
	// For [1:17], size is 17-1=16
	low := 0
	if sliceExpr.Low != nil {
		lowLit, ok := sliceExpr.Low.(*ast.BasicLit)
		if !ok || lowLit.Kind != token.INT {
			return 0 // Non-literal low bound, can't determine effective size
		}
		low, err = strconv.Atoi(lowLit.Value)
		if err != nil {
			return 0
		}
	}
	return high - low
}

// elemTypeSize returns the size in bytes of an element type.
func elemTypeSize(elemType string) int {
	switch elemType {
	case "float32", "int32", "uint32":
		return 4
	case "float64", "int64", "uint64":
		return 8
	case "uint8", "int8":
		return 1
	case "uint16", "int16":
		return 2
	default:
		return 0
	}
}

// getVectorTypeNameForInt returns the vector type name for int types used
// in float operations. The lane count matches the parent element type.
// For example, int32 in a float64 function needs Int32x2 (matching Float64x2 lanes).
func getVectorTypeNameForInt(intType, parentElemType string, target Target) string {
	if intType != "int32" && intType != "int64" {
		// Non-integer type, use regular logic
		return getVectorTypeName(intType, target)
	}

	// Match lanes to parent element type
	lanes := target.LanesFor(parentElemType)
	switch intType {
	case "int32":
		return fmt.Sprintf("Int32x%d", lanes)
	case "int64":
		return fmt.Sprintf("Int64x%d", lanes)
	default:
		return getVectorTypeName(intType, target)
	}
}

// parseTypeExpr converts a type string back to an AST expression.
func parseTypeExpr(typeStr string) ast.Expr {
	// Handle slice types
	if strings.HasPrefix(typeStr, "[]") {
		return &ast.ArrayType{
			Elt: parseTypeExpr(typeStr[2:]),
		}
	}

	// Handle array types like [4]uint32 or [16]uint8
	// Must check before generic types since both use brackets
	if strings.HasPrefix(typeStr, "[") {
		closeBracket := strings.Index(typeStr, "]")
		if closeBracket > 0 {
			sizeStr := typeStr[1:closeBracket]
			elemType := typeStr[closeBracket+1:]
			// Check if it's an array type (size is a number) vs generic (size is a type)
			if _, err := strconv.Atoi(sizeStr); err == nil {
				return &ast.ArrayType{
					Len: &ast.BasicLit{Kind: token.INT, Value: sizeStr},
					Elt: parseTypeExpr(elemType),
				}
			}
		}
	}

	// Handle pointer types
	if strings.HasPrefix(typeStr, "*") {
		return &ast.StarExpr{
			X: parseTypeExpr(typeStr[1:]),
		}
	}

	// Handle function types like func(archsimd.Float32x8) archsimd.Float32x8
	if strings.HasPrefix(typeStr, "func(") {
		return parseFuncType(typeStr)
	}

	// Handle generic types like hwy.Vec[float32] or Vec[float32]
	if bracketIdx := strings.Index(typeStr, "["); bracketIdx >= 0 {
		closeBracket := strings.LastIndex(typeStr, "]")
		if closeBracket > bracketIdx {
			baseType := typeStr[:bracketIdx]
			typeArg := typeStr[bracketIdx+1 : closeBracket]

			// Parse the base type (could be pkg.Type or just Type)
			baseExpr := parseTypeExpr(baseType)

			// Create IndexExpr for the generic instantiation
			return &ast.IndexExpr{
				X:     baseExpr,
				Index: parseTypeExpr(typeArg),
			}
		}
	}

	// Handle qualified names (pkg.Type)
	if before, after, ok := strings.Cut(typeStr, "."); ok {
		return &ast.SelectorExpr{
			X:   ast.NewIdent(before),
			Sel: ast.NewIdent(after),
		}
	}

	// Simple identifier
	return ast.NewIdent(typeStr)
}

// parseFuncType parses a function type string like "func(archsimd.Float32x8) archsimd.Float32x8"
func parseFuncType(typeStr string) *ast.FuncType {
	// Find the matching closing paren for the params
	parenDepth := 0
	paramsEnd := -1
	for i := 5; i < len(typeStr); i++ { // Start after "func("
		switch typeStr[i] {
		case '(':
			parenDepth++
		case ')':
			if parenDepth == 0 {
				paramsEnd = i
				break
			}
			parenDepth--
		}
		if paramsEnd >= 0 {
			break
		}
	}
	if paramsEnd < 0 {
		// Malformed, return empty func type
		return &ast.FuncType{}
	}

	// Extract params string (between "func(" and ")")
	paramsStr := typeStr[5:paramsEnd]

	// Extract results string (after ")")
	resultsStr := strings.TrimSpace(typeStr[paramsEnd+1:])
	// Remove surrounding parens from results if present
	if strings.HasPrefix(resultsStr, "(") && strings.HasSuffix(resultsStr, ")") {
		resultsStr = resultsStr[1 : len(resultsStr)-1]
	}

	// Parse params
	var params []*ast.Field
	if paramsStr != "" {
		for _, paramType := range splitTypeList(paramsStr) {
			paramType = strings.TrimSpace(paramType)
			if paramType != "" {
				params = append(params, &ast.Field{
					Type: parseTypeExpr(paramType),
				})
			}
		}
	}

	// Parse results
	var results []*ast.Field
	if resultsStr != "" {
		for _, resultType := range splitTypeList(resultsStr) {
			resultType = strings.TrimSpace(resultType)
			if resultType != "" {
				results = append(results, &ast.Field{
					Type: parseTypeExpr(resultType),
				})
			}
		}
	}

	funcType := &ast.FuncType{
		Params: &ast.FieldList{List: params},
	}
	if len(results) > 0 {
		funcType.Results = &ast.FieldList{List: results}
	}
	return funcType
}

// splitTypeList splits a comma-separated type list, respecting nested brackets and parens.
func splitTypeList(s string) []string {
	var parts []string
	depth := 0
	start := 0
	for i, c := range s {
		switch c {
		case '(', '[':
			depth++
		case ')', ']':
			depth--
		case ',':
			if depth == 0 {
				parts = append(parts, s[start:i])
				start = i + 1
			}
		}
	}
	if start < len(s) {
		parts = append(parts, s[start:])
	}
	return parts
}

// cloneBlockStmt creates a deep copy of a block statement.
func cloneBlockStmt(block *ast.BlockStmt) *ast.BlockStmt {
	if block == nil {
		return nil
	}

	newBlock := &ast.BlockStmt{
		List: make([]ast.Stmt, len(block.List)),
	}

	for i, stmt := range block.List {
		newBlock.List[i] = cloneStmt(stmt)
	}

	return newBlock
}

// cloneStmt creates a deep copy of a statement.
func cloneStmt(stmt ast.Stmt) ast.Stmt {
	if stmt == nil {
		return nil
	}

	switch s := stmt.(type) {
	case *ast.ExprStmt:
		return &ast.ExprStmt{X: cloneExpr(s.X)}
	case *ast.AssignStmt:
		return cloneAssignStmt(s)
	case *ast.DeclStmt:
		return &ast.DeclStmt{Decl: cloneDecl(s.Decl)}
	case *ast.ReturnStmt:
		return cloneReturnStmt(s)
	case *ast.ForStmt:
		return cloneForStmt(s)
	case *ast.IfStmt:
		return cloneIfStmt(s)
	case *ast.IncDecStmt:
		return &ast.IncDecStmt{X: cloneExpr(s.X), Tok: s.Tok}
	case *ast.BranchStmt:
		return &ast.BranchStmt{Tok: s.Tok, Label: s.Label}
	case *ast.BlockStmt:
		return cloneBlockStmt(s)
	case *ast.RangeStmt:
		return &ast.RangeStmt{
			Key:   cloneExpr(s.Key),
			Value: cloneExpr(s.Value),
			Tok:   s.Tok,
			X:     cloneExpr(s.X),
			Body:  cloneBlockStmt(s.Body),
		}
	case *ast.SwitchStmt:
		return &ast.SwitchStmt{
			Init: cloneStmt(s.Init),
			Tag:  cloneExpr(s.Tag),
			Body: cloneBlockStmt(s.Body),
		}
	case *ast.TypeSwitchStmt:
		return &ast.TypeSwitchStmt{
			Init:   cloneStmt(s.Init),
			Assign: cloneStmt(s.Assign),
			Body:   cloneBlockStmt(s.Body),
		}
	case *ast.CaseClause:
		// For default clause, List is nil; preserve that
		var exprs []ast.Expr
		if len(s.List) > 0 {
			exprs = make([]ast.Expr, len(s.List))
			for i, e := range s.List {
				exprs[i] = cloneExpr(e)
			}
		}
		stmts := make([]ast.Stmt, len(s.Body))
		for i, st := range s.Body {
			stmts[i] = cloneStmt(st)
		}
		return &ast.CaseClause{
			List: exprs,
			Body: stmts,
		}
	default:
		// For other statement types, return as-is
		return stmt
	}
}

// cloneExpr creates a deep copy of an expression.
func cloneExpr(expr ast.Expr) ast.Expr {
	if expr == nil {
		return nil
	}

	switch e := expr.(type) {
	case *ast.Ident:
		return &ast.Ident{Name: e.Name}
	case *ast.BasicLit:
		return &ast.BasicLit{Kind: e.Kind, Value: e.Value}
	case *ast.SelectorExpr:
		return &ast.SelectorExpr{
			X:   cloneExpr(e.X),
			Sel: ast.NewIdent(e.Sel.Name),
		}
	case *ast.CallExpr:
		args := make([]ast.Expr, len(e.Args))
		for i, arg := range e.Args {
			args[i] = cloneExpr(arg)
		}
		return &ast.CallExpr{
			Fun:      cloneExpr(e.Fun),
			Args:     args,
			Ellipsis: e.Ellipsis,
		}
	case *ast.BinaryExpr:
		return &ast.BinaryExpr{
			X:  cloneExpr(e.X),
			Op: e.Op,
			Y:  cloneExpr(e.Y),
		}
	case *ast.UnaryExpr:
		return &ast.UnaryExpr{
			Op: e.Op,
			X:  cloneExpr(e.X),
		}
	case *ast.ParenExpr:
		return &ast.ParenExpr{X: cloneExpr(e.X)}
	case *ast.IndexExpr:
		return &ast.IndexExpr{
			X:     cloneExpr(e.X),
			Index: cloneExpr(e.Index),
		}
	case *ast.SliceExpr:
		return &ast.SliceExpr{
			X:      cloneExpr(e.X),
			Low:    cloneExpr(e.Low),
			High:   cloneExpr(e.High),
			Max:    cloneExpr(e.Max),
			Slice3: e.Slice3,
		}
	case *ast.StarExpr:
		return &ast.StarExpr{X: cloneExpr(e.X)}
	case *ast.TypeAssertExpr:
		return &ast.TypeAssertExpr{
			X:    cloneExpr(e.X),
			Type: cloneExpr(e.Type),
		}
	case *ast.ArrayType:
		return &ast.ArrayType{
			Len: cloneExpr(e.Len),
			Elt: cloneExpr(e.Elt),
		}
	case *ast.CompositeLit:
		elts := make([]ast.Expr, len(e.Elts))
		for i, elt := range e.Elts {
			elts[i] = cloneExpr(elt)
		}
		return &ast.CompositeLit{
			Type: cloneExpr(e.Type),
			Elts: elts,
		}
	default:
		// For unsupported types, return as-is (may cause issues for complex expressions)
		return expr
	}
}

// cloneAssignStmt clones an assignment statement.
func cloneAssignStmt(stmt *ast.AssignStmt) *ast.AssignStmt {
	newStmt := &ast.AssignStmt{
		Lhs: make([]ast.Expr, len(stmt.Lhs)),
		Rhs: make([]ast.Expr, len(stmt.Rhs)),
		Tok: stmt.Tok,
	}
	for i, lhs := range stmt.Lhs {
		newStmt.Lhs[i] = cloneExpr(lhs)
	}
	for i, rhs := range stmt.Rhs {
		newStmt.Rhs[i] = cloneExpr(rhs)
	}
	return newStmt
}

// cloneDecl clones a declaration.
func cloneDecl(decl ast.Decl) ast.Decl {
	if decl == nil {
		return nil
	}

	switch d := decl.(type) {
	case *ast.GenDecl:
		newSpecs := make([]ast.Spec, len(d.Specs))
		for i, spec := range d.Specs {
			newSpecs[i] = cloneSpec(spec)
		}
		return &ast.GenDecl{
			Tok:   d.Tok,
			Specs: newSpecs,
		}
	default:
		return decl
	}
}

// cloneSpec clones a declaration spec (e.g., variable declaration).
func cloneSpec(spec ast.Spec) ast.Spec {
	if spec == nil {
		return nil
	}

	switch s := spec.(type) {
	case *ast.ValueSpec:
		var newValues []ast.Expr
		if len(s.Values) > 0 {
			newValues = make([]ast.Expr, len(s.Values))
			for i, v := range s.Values {
				newValues[i] = cloneExpr(v)
			}
		}
		newNames := make([]*ast.Ident, len(s.Names))
		for i, n := range s.Names {
			newNames[i] = &ast.Ident{Name: n.Name}
		}
		return &ast.ValueSpec{
			Names:  newNames,
			Type:   cloneExpr(s.Type),
			Values: newValues,
		}
	default:
		return spec
	}
}

// cloneReturnStmt clones a return statement.
func cloneReturnStmt(stmt *ast.ReturnStmt) *ast.ReturnStmt {
	newStmt := &ast.ReturnStmt{
		Results: make([]ast.Expr, len(stmt.Results)),
	}
	for i, result := range stmt.Results {
		newStmt.Results[i] = cloneExpr(result)
	}
	return newStmt
}

// cloneForStmt clones a for loop.
func cloneForStmt(stmt *ast.ForStmt) *ast.ForStmt {
	return &ast.ForStmt{
		Init: cloneStmt(stmt.Init),
		Cond: cloneExpr(stmt.Cond),
		Post: cloneStmt(stmt.Post),
		Body: cloneBlockStmt(stmt.Body),
	}
}

// cloneIfStmt clones an if statement.
func cloneIfStmt(stmt *ast.IfStmt) *ast.IfStmt {
	return &ast.IfStmt{
		Init: cloneStmt(stmt.Init),
		Cond: cloneExpr(stmt.Cond),
		Body: cloneBlockStmt(stmt.Body),
		Else: cloneStmt(stmt.Else),
	}
}

// buildResults builds the return type list for a function.
func (pf *ParsedFunc) buildResults(elemType string) *ast.FieldList {
	if len(pf.Returns) == 0 {
		return nil
	}

	fieldList := &ast.FieldList{
		List: make([]*ast.Field, 0, len(pf.Returns)),
	}

	for _, ret := range pf.Returns {
		retType := specializeType(ret.Type, pf.TypeParams, elemType)
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

// buildResultsWithTarget builds the return type list with target-specific Vec types.
func (pf *ParsedFunc) buildResultsWithTarget(elemType string, target Target) *ast.FieldList {
	if len(pf.Returns) == 0 {
		return nil
	}

	fieldList := &ast.FieldList{
		List: make([]*ast.Field, 0, len(pf.Returns)),
	}

	for _, ret := range pf.Returns {
		retType := specializeType(ret.Type, pf.TypeParams, elemType)
		// Transform hwy.Vec[T] to concrete vector types for SIMD targets
		retType = specializeVecType(retType, elemType, target)
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

	lanes := ctx.target.LanesFor(ctx.elemType)
	vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)

	// Walk all statements and expressions, replacing as needed
	ast.Inspect(node, func(n ast.Node) bool {
		switch stmt := n.(type) {
		case *ast.IfStmt:
			// Replace comparisons like: remaining >= v.NumLanes()
			if binExpr, ok := stmt.Cond.(*ast.BinaryExpr); ok {
				replaceNumLanesInExpr(binExpr, lanes)
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
			if !isHalfPrecisionType(ctx.elemType) && !hasNativeReduceSum {
				for i, rhs := range stmt.Rhs {
					if call, ok := rhs.(*ast.CallExpr); ok {
						if isReduceSumCall(call) {
							// Transform to store + sum pattern
							stmt.Rhs[i] = createReduceSumExpr(call, lanes, vecTypeName, ctx.elemType)
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
func replaceNumLanesInExpr(binExpr *ast.BinaryExpr, lanes int) {
	// Check RHS
	if call, ok := binExpr.Y.(*ast.CallExpr); ok {
		if isNumLanesCall(call) {
			binExpr.Y = &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.Itoa(lanes),
			}
		}
	}
	// Check LHS (less common but possible)
	if call, ok := binExpr.X.(*ast.CallExpr); ok {
		if isNumLanesCall(call) {
			binExpr.X = &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.Itoa(lanes),
			}
		}
	}
}

// isNumLanesCall checks if a call expression is v.NumLanes() or v.NumElements().
func isNumLanesCall(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	return sel.Sel.Name == "NumLanes" || sel.Sel.Name == "NumElements"
}

// isReduceSumCall checks if a call expression is v.ReduceSum(), hwy.ReduceSum(v),
// or the F16/BF16 variants (ReduceSumF16, ReduceSumBF16).
func isReduceSumCall(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	name := sel.Sel.Name
	return name == "ReduceSum" || name == "ReduceSumF16" || name == "ReduceSumBF16"
}

// createReduceSumExpr creates an expression that stores the vector and sums elements.
// For now, we generate a function call that we'll define in a helper.
// Actually, archsimd vectors don't have a built-in ReduceSum, so we need to
// generate inline code that stores to temp and sums.
// Since we can't inject statements here, we'll generate a compound expression.
func createReduceSumExpr(call *ast.CallExpr, lanes int, vecTypeName, elemType string) ast.Expr {
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
				switch ident.Name {
				case "math", "vec", "matvec", "matmul", "algo", "image", "bitpack", "sort":
					if strings.HasPrefix(sel.Sel.Name, "Base") {
						// Transform math.BaseExpVec to math.BaseExpVec_avx2
						suffix := ctx.target.Suffix()
						if ctx.elemType == "float64" {
							suffix = suffix + "_Float64"
						} else if isFloat16Type(ctx.elemType) {
							suffix = suffix + "_Float16"
						} else if isBFloat16Type(ctx.elemType) {
							suffix = suffix + "_BFloat16"
						}
						sel.Sel.Name = sel.Sel.Name + suffix
					}
				}
			}
		}

		// Handle package.BaseFuncName[T] (IndexExpr wrapping SelectorExpr)
		if indexExpr, ok := arg.(*ast.IndexExpr); ok {
			if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
				if ident, ok := sel.X.(*ast.Ident); ok {
					switch ident.Name {
					case "math", "vec", "matvec", "matmul", "algo", "image", "bitpack", "sort":
						if strings.HasPrefix(sel.Sel.Name, "Base") {
							// Transform to non-generic version with suffix
							suffix := ctx.target.Suffix()
							if ctx.elemType == "float64" {
								suffix = suffix + "_Float64"
							} else if isFloat16Type(ctx.elemType) {
								suffix = suffix + "_Float16"
							} else if isBFloat16Type(ctx.elemType) {
								suffix = suffix + "_BFloat16"
							}
							// Replace the IndexExpr with just the SelectorExpr (strip type param)
							sel.Sel.Name = sel.Sel.Name + suffix
							call.Args[i] = sel
						}
					}
				}
			}
		}

		// Handle local BaseFuncName[T] (IndexExpr wrapping Ident)
		if indexExpr, ok := arg.(*ast.IndexExpr); ok {
			if ident, ok := indexExpr.X.(*ast.Ident); ok {
				if strings.HasPrefix(ident.Name, "Base") {
					// Transform BaseFunc[T] to BaseFunc_avx2
					suffix := ctx.target.Suffix()
					if ctx.elemType == "float64" {
						suffix = suffix + "_Float64"
					} else if isFloat16Type(ctx.elemType) {
						suffix = suffix + "_Float16"
					} else if isBFloat16Type(ctx.elemType) {
						suffix = suffix + "_BFloat16"
					}
					// Replace the IndexExpr with just the Ident
					call.Args[i] = ast.NewIdent(ident.Name + suffix)
				}
			}
		}
	}
}

// hasPredicateParam returns true if the function has a predicate-type parameter
// (i.e., a type parameter with a non-Lanes constraint like Predicate[T]).
func hasPredicateParam(pf *ParsedFunc) bool {
	for _, tp := range pf.TypeParams {
		// Skip element type constraints
		if strings.Contains(tp.Constraint, "Lanes") ||
			strings.Contains(tp.Constraint, "Floats") ||
			strings.Contains(tp.Constraint, "Integers") ||
			strings.Contains(tp.Constraint, "SignedInts") ||
			strings.Contains(tp.Constraint, "UnsignedInts") {
			continue
		}
		// This is likely a predicate or other interface type param
		return true
	}
	return false
}

// generateScalarPredicateBody generates a scalar loop body for predicate functions
// in fallback mode. Returns nil if this function doesn't need scalar generation.
func generateScalarPredicateBody(pf *ParsedFunc, elemType string) *ast.BlockStmt {
	// Map function names to their scalar implementations
	switch pf.Name {
	case "BaseAll":
		return generateScalarAll(pf, elemType)
	case "BaseAny":
		return generateScalarAny(pf, elemType)
	case "BaseNone":
		return generateScalarNone(pf, elemType)
	case "BaseFindIf":
		return generateScalarFindIf(pf, elemType)
	case "BaseCountIf":
		return generateScalarCountIf(pf, elemType)
	default:
		return nil
	}
}

// generateScalarAll generates: for _, v := range slice { if !pred.Test(v) { return false } } return true
func generateScalarAll(pf *ParsedFunc, elemType string) *ast.BlockStmt {
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
func generateScalarAny(pf *ParsedFunc, elemType string) *ast.BlockStmt {
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

// generateScalarNone generates: return !BaseAny_fallback...(slice, pred)
func generateScalarNone(pf *ParsedFunc, elemType string) *ast.BlockStmt {
	sliceParam := pf.Params[0].Name
	predParam := pf.Params[1].Name

	// Build the function name: BaseAny_fallback or BaseAny_fallback_Float64, etc.
	funcName := "BaseAny_fallback"
	switch elemType {
	case "float64":
		funcName += "_Float64"
	case "int32":
		funcName += "_Int32"
	case "int64":
		funcName += "_Int64"
	case "uint32":
		funcName += "_Uint32"
	case "uint64":
		funcName += "_Uint64"
	}

	return &ast.BlockStmt{
		List: []ast.Stmt{
			&ast.ReturnStmt{
				Results: []ast.Expr{
					&ast.UnaryExpr{
						Op: token.NOT,
						X: &ast.CallExpr{
							Fun:  ast.NewIdent(funcName),
							Args: []ast.Expr{ast.NewIdent(sliceParam), ast.NewIdent(predParam)},
						},
					},
				},
			},
		},
	}
}

// generateScalarFindIf generates: for i, v := range slice { if pred.Test(v) { return i } } return -1
func generateScalarFindIf(pf *ParsedFunc, elemType string) *ast.BlockStmt {
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
func generateScalarCountIf(pf *ParsedFunc, elemType string) *ast.BlockStmt {
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

// transformHalfPrecisionFallback transforms scalar operations on Float16/BFloat16
// to use float32 conversions. This is necessary because Float16/BFloat16 are uint16
// under the hood and don't have arithmetic operators defined.
//
// Transformations:
// - Scalar variable declarations: var x hwy.Float16 → var x float32
// - Slice reads in expressions: input[i] → input[i].Float32()
// - Slice assignments: shifted[i] = expr → shifted[i] = hwy.Float32ToFloat16(expr)
// - Type conversions: hwy.Float16(1.0) → float32(1.0)
// - hwy.ReduceSum calls: x := hwy.ReduceSum(v) → x := hwy.ReduceSum(v).Float32()
// - Return statements: return x → return hwy.Float32ToFloat16(x)
func transformHalfPrecisionFallback(body *ast.BlockStmt, ctx *transformContext) {
	// Get the conversion function name
	var toFloat32Method string = "Float32"
	var fromFloat32Func string
	if isFloat16Type(ctx.elemType) {
		fromFloat32Func = "hwy.Float32ToFloat16"
	} else {
		fromFloat32Func = "hwy.Float32ToBFloat16"
	}

	// Track variables assigned from ReduceSum so we know they're float32
	reduceSumVars := make(map[string]bool)

	// Track variables computed as float32 that need conversion back to Float16/BFloat16
	// when passed to hwy.Set. This is separate from halfPrecisionScalarVars which
	// tracks original Float16/BFloat16 values (from slice reads or parameters).
	float32ComputedVars := make(map[string]bool)

	// First pass: collect variables assigned from half-precision slice reads
	// and track local slice variables of half-precision type.
	// Note: ctx.halfPrecisionScalarVars is already initialized with scalar function
	// parameters; this pass adds local variables assigned from slice reads.
	halfPrecisionScalarVars := ctx.halfPrecisionScalarVars
	if halfPrecisionScalarVars == nil {
		halfPrecisionScalarVars = make(map[string]bool)
	}
	ast.Inspect(body, func(n ast.Node) bool {
		if assign, ok := n.(*ast.AssignStmt); ok {
			for i, rhs := range assign.Rhs {
				// Track the variable name
				var varName string
				if i < len(assign.Lhs) {
					if ident, ok := assign.Lhs[i].(*ast.Ident); ok {
						varName = ident.Name
					}
				}

				// Check if RHS is a slice index expression on a half-precision slice
				// Only track for := definitions, not compound assignments (+=, etc.)
				// Compound assignments like `expSum += output[i]` should have output[i] wrapped
				if indexExpr, ok := rhs.(*ast.IndexExpr); ok {
					if assign.Tok == token.DEFINE && isHalfPrecisionSliceExpr(indexExpr, ctx) {
						if varName != "" {
							halfPrecisionScalarVars[varName] = true
						}
					}
				}

				// Check if RHS is a type conversion T(x) where T is the half-precision type.
				// After transformNode runs, T(1) becomes hwy.Float16(1) or hwy.BFloat16(1).
				// These need to be converted to/from float32 for scalar operations.
				if call, ok := rhs.(*ast.CallExpr); ok {
					if ident, ok := call.Fun.(*ast.Ident); ok {
						// Check for half-precision type conversions (after transformation)
						// or single-letter type params (before transformation)
						isHalfPrecisionConv := ident.Name == ctx.elemType ||
							ident.Name == "hwy.Float16" || ident.Name == "hwy.BFloat16" ||
							len(ident.Name) == 1
						if isHalfPrecisionConv {
							if varName != "" && assign.Tok == token.DEFINE {
								float32ComputedVars[varName] = true
							}
						}
					}
					// Check for hwy.ReduceMax, hwy.ReduceMin, hwy.ReduceSum etc.
					// Only match base names (not F16/BF16 suffixed versions) because:
					// - Base versions (ReduceSum) return element type T (e.g., Float16)
					//   → result IS half-precision, needs .Float32() for scalar ops
					// - Suffixed versions (ReduceSumF16) already return float32
					//   → result is NOT half-precision, .Float32() on float32 is invalid
					if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
						if pkgIdent, ok := sel.X.(*ast.Ident); ok && pkgIdent.Name == "hwy" {
							funcName := sel.Sel.Name
							if isBaseReduceFunction(funcName) {
								if varName != "" && assign.Tok == token.DEFINE {
									halfPrecisionScalarVars[varName] = true
								}
							}
						}
					}
				}

				// Check if RHS is a binary operation involving type conversions or
				// half-precision scalars, e.g., scale := T(1) / norm
				if binExpr, ok := rhs.(*ast.BinaryExpr); ok {
					if assign.Tok == token.DEFINE && varName != "" {
						// Check if either operand is a type conversion
						hasTypeConv := false
						if call, ok := binExpr.X.(*ast.CallExpr); ok {
							if ident, ok := call.Fun.(*ast.Ident); ok {
								if ident.Name == ctx.elemType ||
									ident.Name == "hwy.Float16" || ident.Name == "hwy.BFloat16" ||
									len(ident.Name) == 1 {
									hasTypeConv = true
								}
							}
						}
						if call, ok := binExpr.Y.(*ast.CallExpr); ok {
							if ident, ok := call.Fun.(*ast.Ident); ok {
								if ident.Name == ctx.elemType ||
									ident.Name == "hwy.Float16" || ident.Name == "hwy.BFloat16" ||
									len(ident.Name) == 1 {
									hasTypeConv = true
								}
							}
						}
						// Check if either operand is a tracked variable
						if ident, ok := binExpr.X.(*ast.Ident); ok {
							if float32ComputedVars[ident.Name] {
								hasTypeConv = true
							}
						}
						if ident, ok := binExpr.Y.(*ast.Ident); ok {
							if float32ComputedVars[ident.Name] {
								hasTypeConv = true
							}
						}
						if hasTypeConv {
							float32ComputedVars[varName] = true
						}
					}
				}

				// Check if RHS is a slice expression on a half-precision slice
				// e.g., row := m[i*cols : (i+1)*cols]
				if sliceExpr, ok := rhs.(*ast.SliceExpr); ok {
					if ident, ok := sliceExpr.X.(*ast.Ident); ok {
						if ctx.halfPrecisionSlices[ident.Name] && varName != "" {
							// This is a sub-slice of a half-precision slice
							ctx.halfPrecisionSlices[varName] = true
						}
					}
				}

				// Check if RHS is make([]T, ...) where T is half-precision
				if callExpr, ok := rhs.(*ast.CallExpr); ok {
					if ident, ok := callExpr.Fun.(*ast.Ident); ok {
						if ident.Name == "make" && len(callExpr.Args) > 0 {
							// Check if it's making a half-precision slice
							if arrayType, ok := callExpr.Args[0].(*ast.ArrayType); ok {
								if arrayType.Len == nil { // slice, not array
									elemTypeStr := exprToString(arrayType.Elt)
									if elemTypeStr == ctx.elemType || elemTypeStr == "hwy.Float16" || elemTypeStr == "hwy.BFloat16" {
										if varName != "" {
											ctx.halfPrecisionSlices[varName] = true
										}
									}
								}
							}
						}
					}
					// Check if RHS is a .Data() call on a vector
					// hwy.Vec[T].Data() returns []T, so for half-precision types this is a half-precision slice
					if sel, ok := callExpr.Fun.(*ast.SelectorExpr); ok {
						if sel.Sel.Name == "Data" && len(callExpr.Args) == 0 {
							// This is a .Data() call - for half-precision element types,
							// the result is a half-precision slice
							if varName != "" && assign.Tok == token.DEFINE {
								ctx.halfPrecisionSlices[varName] = true
							}
						}
					}
					// Check if RHS is an IIFE (from transformed .Data() for NEON target)
					// Pattern: func() []T { var _simd_tmp [N]T; ...; return _simd_tmp[:] }()
					if funcLit, ok := callExpr.Fun.(*ast.FuncLit); ok {
						if funcLit.Type.Results != nil && len(funcLit.Type.Results.List) == 1 {
							if arrType, ok := funcLit.Type.Results.List[0].Type.(*ast.ArrayType); ok {
								if arrType.Len == nil { // slice type (no length)
									if ident, ok := arrType.Elt.(*ast.Ident); ok {
										if isHalfPrecisionType(ident.Name) {
											if varName != "" && assign.Tok == token.DEFINE {
												ctx.halfPrecisionSlices[varName] = true
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		return true
	})

	// Store in context for use in wrapHalfPrecisionExpr
	ctx.halfPrecisionScalarVars = halfPrecisionScalarVars

	// Transform the AST
	ast.Inspect(body, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.DeclStmt:
			// Transform: var expSum hwy.Float16 → var expSum float32
			if genDecl, ok := node.Decl.(*ast.GenDecl); ok && genDecl.Tok == token.VAR {
				for _, spec := range genDecl.Specs {
					if valueSpec, ok := spec.(*ast.ValueSpec); ok {
						typeStr := exprToString(valueSpec.Type)
						if typeStr == ctx.elemType || typeStr == "hwy.Float16" || typeStr == "hwy.BFloat16" {
							valueSpec.Type = ast.NewIdent("float32")
						}
					}
				}
			}

		case *ast.AssignStmt:
			// Check for hwy.ReduceSum assignment and wrap with .Float32()
			// result := hwy.ReduceSum(sum) → result := hwy.ReduceSum(sum).Float32()
			// But ReduceSumF16/ReduceSumBF16 already return float32, so don't wrap those.
			for i, rhs := range node.Rhs {
				if callExpr, ok := rhs.(*ast.CallExpr); ok {
					if isReduceSumCall(callExpr) {
						// Check if it's ReduceSumF16/BF16 which already returns float32
						alreadyFloat32 := false
						if sel, ok := callExpr.Fun.(*ast.SelectorExpr); ok {
							name := sel.Sel.Name
							alreadyFloat32 = name == "ReduceSumF16" || name == "ReduceSumBF16"
						}

						if !alreadyFloat32 {
							// Wrap with .Float32()
							node.Rhs[i] = &ast.CallExpr{
								Fun: &ast.SelectorExpr{
									X:   cloneExpr(callExpr),
									Sel: ast.NewIdent("Float32"),
								},
							}
						}
						// Track as float32 and remove from half-precision tracking,
						// since the variable is now float32 (either already was or just wrapped).
						if len(node.Lhs) > i {
							if ident, ok := node.Lhs[i].(*ast.Ident); ok {
								reduceSumVars[ident.Name] = true
								delete(halfPrecisionScalarVars, ident.Name)
							}
						}
						continue
					}
				}
			}
			// Handle other assignments
			transformHalfPrecisionAssignment(node, ctx, toFloat32Method, fromFloat32Func)

		case *ast.IfStmt:
			// Transform if conditions that involve half-precision comparisons
			if binExpr, ok := node.Cond.(*ast.BinaryExpr); ok {
				binExpr.X = wrapHalfPrecisionExpr(binExpr.X, ctx, toFloat32Method)
				binExpr.Y = wrapHalfPrecisionExpr(binExpr.Y, ctx, toFloat32Method)
			}

		case *ast.ReturnStmt:
			// Transform return statements: return x → return hwy.Float32ToFloat16(x)
			// Only wrap values that we know are float32 (from ReduceSum or float32 computations)
			for i, result := range node.Results {
				if ident, ok := result.(*ast.Ident); ok {
					// Only wrap if we tracked this variable as being float32
					if reduceSumVars[ident.Name] || float32ComputedVars[ident.Name] {
						node.Results[i] = &ast.CallExpr{
							Fun:  parseTypeExpr(fromFloat32Func),
							Args: []ast.Expr{ident},
						}
					}
				}
			}

		case *ast.CallExpr:
			// Transform hwy.Set(scale) where scale is a float32-computed scalar
			// hwy.Set(scale) → hwy.Set(hwy.Float32ToFloat16(scale))
			var sel *ast.SelectorExpr
			var ok bool
			switch fun := node.Fun.(type) {
			case *ast.SelectorExpr:
				sel = fun
				ok = true
			case *ast.IndexExpr:
				// hwy.Set[T](arg) - indexed call
				sel, ok = fun.X.(*ast.SelectorExpr)
			}
			if ok && sel != nil {
				if ident, identOk := sel.X.(*ast.Ident); identOk && ident.Name == "hwy" && sel.Sel.Name == "Set" {
					if len(node.Args) == 1 {
						// Only wrap identifiers that were computed from half-precision type conversions
						if argIdent, argOk := node.Args[0].(*ast.Ident); argOk {
							if float32ComputedVars[argIdent.Name] {
								// Wrap: hwy.Set(x) → hwy.Set(hwy.Float32ToFloat16(x))
								node.Args[0] = &ast.CallExpr{
									Fun:  parseTypeExpr(fromFloat32Func),
									Args: []ast.Expr{argIdent},
								}
							}
						}
					}
				}
			}
		}
		return true
	})
}

// transformHalfPrecisionAssignment transforms assignments involving half-precision types.
func transformHalfPrecisionAssignment(stmt *ast.AssignStmt, ctx *transformContext, toFloat32Method, fromFloat32Func string) {
	// Check for compound assignments (+=, -=, *=, /=) on half-precision slices
	// These need special handling: dst[i] += x becomes dst[i] = Float32ToFloat16(dst[i].Float32() + x.Float32())
	isCompoundAssign := stmt.Tok == token.ADD_ASSIGN || stmt.Tok == token.SUB_ASSIGN ||
		stmt.Tok == token.MUL_ASSIGN || stmt.Tok == token.QUO_ASSIGN

	if isCompoundAssign && len(stmt.Lhs) == 1 && len(stmt.Rhs) == 1 {
		if indexExpr, ok := stmt.Lhs[0].(*ast.IndexExpr); ok {
			if isHalfPrecisionSliceExpr(indexExpr, ctx) {
				// Transform compound assignment on half-precision slice
				// dst[i] op= x → dst[i] = Float32ToFloat16(dst[i].Float32() op x.Float32())

				// Get the binary operator from the compound token
				var binOp token.Token
				switch stmt.Tok {
				case token.ADD_ASSIGN:
					binOp = token.ADD
				case token.SUB_ASSIGN:
					binOp = token.SUB
				case token.MUL_ASSIGN:
					binOp = token.MUL
				case token.QUO_ASSIGN:
					binOp = token.QUO
				}

				// Create: dst[i].Float32()
				lhsFloat32 := &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   cloneExpr(indexExpr),
						Sel: ast.NewIdent(toFloat32Method),
					},
				}

				// Create: x.Float32() (wrap RHS)
				rhsFloat32 := wrapHalfPrecisionExpr(stmt.Rhs[0], ctx, toFloat32Method)

				// Create: dst[i].Float32() op x.Float32()
				binExpr := &ast.BinaryExpr{
					X:  lhsFloat32,
					Op: binOp,
					Y:  rhsFloat32,
				}

				// Create: Float32ToFloat16(...)
				newRhs := &ast.CallExpr{
					Fun:  parseTypeExpr(fromFloat32Func),
					Args: []ast.Expr{binExpr},
				}

				// Transform to regular assignment
				stmt.Tok = token.ASSIGN
				stmt.Rhs[0] = newRhs
				return
			}
		}
	}

	// Transform RHS expressions to add .Float32() for slice index reads.
	// Skip if LHS is a simple identifier tracked as half-precision scalar AND
	// this is a simple assignment (:= or =). For compound assignments (+=, -=, etc.),
	// always wrap because the LHS variable is already float32 (from ReduceSum wrapping)
	// and the RHS needs to be float32 for arithmetic.
	isCompound := stmt.Tok == token.ADD_ASSIGN || stmt.Tok == token.SUB_ASSIGN ||
		stmt.Tok == token.MUL_ASSIGN || stmt.Tok == token.QUO_ASSIGN
	for i, rhs := range stmt.Rhs {
		skipConversion := false
		if !isCompound && i < len(stmt.Lhs) {
			if ident, ok := stmt.Lhs[i].(*ast.Ident); ok {
				if ctx.halfPrecisionScalarVars != nil && ctx.halfPrecisionScalarVars[ident.Name] {
					// Don't convert the slice read - keep the variable as Float16
					// It will be converted later when used in scalar arithmetic
					skipConversion = true
				}
			}
		}
		if !skipConversion {
			stmt.Rhs[i] = wrapHalfPrecisionExpr(rhs, ctx, toFloat32Method)
		}
	}

	// Transform LHS slice assignments to wrap RHS with fromFloat32Func
	for i, lhs := range stmt.Lhs {
		if indexExpr, ok := lhs.(*ast.IndexExpr); ok {
			// This is a slice assignment like shifted[i] = ...
			// We need to wrap the RHS with Float32ToFloat16/Float32ToBFloat16
			// But only if the slice is of half-precision type
			if isHalfPrecisionSliceExpr(indexExpr, ctx) {
				// Wrap RHS: expr → hwy.Float32ToFloat16(expr)
				if i < len(stmt.Rhs) {
					stmt.Rhs[i] = &ast.CallExpr{
						Fun:  parseTypeExpr(fromFloat32Func),
						Args: []ast.Expr{stmt.Rhs[i]},
					}
				}
			}
		}
	}
}

// wrapHalfPrecisionExpr wraps index expressions on half-precision slices with .Float32()
func wrapHalfPrecisionExpr(expr ast.Expr, ctx *transformContext, toFloat32Method string) ast.Expr {
	if expr == nil {
		return nil
	}

	switch e := expr.(type) {
	case *ast.IndexExpr:
		// Check if this is reading from a half-precision slice
		if isHalfPrecisionSliceExpr(e, ctx) {
			// Wrap with .Float32(): input[i] → input[i].Float32()
			return &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   cloneExpr(e),
					Sel: ast.NewIdent(toFloat32Method),
				},
			}
		}
		return e

	case *ast.BinaryExpr:
		// Recursively wrap operands
		return &ast.BinaryExpr{
			X:  wrapHalfPrecisionExpr(e.X, ctx, toFloat32Method),
			Op: e.Op,
			Y:  wrapHalfPrecisionExpr(e.Y, ctx, toFloat32Method),
		}

	case *ast.CallExpr:
		// Check for type conversions like hwy.Float16(1.0)
		if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
			if pkgIdent, ok := sel.X.(*ast.Ident); ok {
				if pkgIdent.Name == "hwy" {
					funcName := sel.Sel.Name
					// Check for type conversion
					if funcName == "Float16" || funcName == "BFloat16" {
						// Transform: hwy.Float16(1.0) → float32(1.0)
						e.Fun = ast.NewIdent("float32")
						return e
					}
					// Skip wrapping arguments for vector operations
					// These need half-precision arguments to produce half-precision vectors
					if isVectorOperation(funcName) {
						return e
					}
				}
			}
		}
		// Also check for simple type conversions like hwy.Float16(x)
		if ident, ok := e.Fun.(*ast.Ident); ok {
			if ident.Name == ctx.elemType || ident.Name == "hwy.Float16" || ident.Name == "hwy.BFloat16" {
				e.Fun = ast.NewIdent("float32")
				return e
			}
		}
		// Recurse into call arguments for type conversions like float64(input[i] - maxVal)
		for i, arg := range e.Args {
			e.Args[i] = wrapHalfPrecisionExpr(arg, ctx, toFloat32Method)
		}
		return e

	case *ast.ParenExpr:
		return &ast.ParenExpr{X: wrapHalfPrecisionExpr(e.X, ctx, toFloat32Method)}

	case *ast.UnaryExpr:
		return &ast.UnaryExpr{
			Op: e.Op,
			X:  wrapHalfPrecisionExpr(e.X, ctx, toFloat32Method),
		}

	case *ast.Ident:
		// Check if this is a half-precision scalar variable
		if ctx.halfPrecisionScalarVars != nil && ctx.halfPrecisionScalarVars[e.Name] {
			// Wrap with .Float32(): aik → aik.Float32()
			return &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   cloneExpr(e),
					Sel: ast.NewIdent(toFloat32Method),
				},
			}
		}
		return e

	default:
		return expr
	}
}

// vectorOperations is a set of hwy function names that operate on vectors.
// Arguments to these functions should NOT be wrapped with .Float32() because
// they need half-precision arguments to produce half-precision vectors.
var vectorOperations = map[string]bool{
	// Vector creation and manipulation
	"Set": true, "Load": true, "Store": true, "Zero": true, "Broadcast": true,
	// Vector arithmetic
	"Add": true, "Sub": true, "Mul": true, "Div": true, "MulAdd": true, "MulSub": true,
	"Neg": true, "Abs": true, "Min": true, "Max": true, "Clamp": true,
	// F16/BF16 specific arithmetic
	"AddF16": true, "SubF16": true, "MulF16": true, "DivF16": true, "MulAddF16": true,
	"AddBF16": true, "SubBF16": true, "MulBF16": true, "DivBF16": true, "MulAddBF16": true,
	// Comparisons
	"Eq": true, "Ne": true, "Lt": true, "Le": true, "Gt": true, "Ge": true,
	"LessThan": true, "LessThanOrEqual": true, "GreaterThan": true, "GreaterThanOrEqual": true,
	"LessThanF16": true, "LessThanOrEqualF16": true, "GreaterThanF16": true, "GreaterThanOrEqualF16": true,
	"LessThanBF16": true, "LessThanOrEqualBF16": true, "GreaterThanBF16": true, "GreaterThanOrEqualBF16": true,
	// Reductions
	"ReduceSum": true, "ReduceMin": true, "ReduceMax": true,
	"ReduceSumF16": true, "ReduceSumBF16": true,
	// Merge/Select
	"Merge": true, "IfThenElse": true, "IfThenElseF16": true, "IfThenElseBF16": true,
	// Bitwise
	"And": true, "Or": true, "Xor": true, "Not": true, "AndNot": true,
	// Shuffle/Permute
	"Shuffle": true, "Reverse": true, "RotateLeft": true, "RotateRight": true,
	// Convert (these produce vectors of the target type)
	"ConvertTo": true, "PromoteTo": true, "DemoteTo": true,
}

// isVectorOperation returns true if the function name is a hwy vector operation
// that needs half-precision arguments to produce half-precision vectors.
func isVectorOperation(funcName string) bool {
	return vectorOperations[funcName]
}

// reduceBaseFunctions is a set of hwy reduce function base names that reduce vectors to scalars
// and return the element type. Variables assigned from these functions need to be
// tracked as half-precision scalars for Float16/BFloat16.
var reduceBaseFunctions = []string{
	"ReduceMax",
	"ReduceMin",
	"ReduceSum",
}

// isBaseReduceFunction returns true if the function name is a base hwy reduce
// operation (ReduceMax, ReduceMin, ReduceSum). These return the element type T,
// so for Float16/BFloat16, the result is a half-precision scalar.
// Does NOT match F16/BF16 suffixed versions which already return float32.
func isBaseReduceFunction(funcName string) bool {
	for _, base := range reduceBaseFunctions {
		if funcName == base {
			return true
		}
	}
	return false
}

// isHalfPrecisionSliceExpr checks if an index expression is accessing a half-precision slice.
// It uses the tracked halfPrecisionSlices set which is populated from function parameters
// and local variable types.
func isHalfPrecisionSliceExpr(indexExpr *ast.IndexExpr, ctx *transformContext) bool {
	if ctx.halfPrecisionSlices == nil {
		return false
	}
	// Get the slice variable name
	if ident, ok := indexExpr.X.(*ast.Ident); ok {
		return ctx.halfPrecisionSlices[ident.Name]
	}
	return false
}

