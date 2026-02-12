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
	"slices"
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

// isAVXPromotedHalfPrec returns true if the target uses promoted float32 storage
// for half-precision types (AVX2 with Float16x8AVX2, AVX512 with Float16x16AVX512).
func isAVXPromotedHalfPrec(target Target, elemType string) bool {
	return isHalfPrecisionType(elemType) && (target.Name == "AVX2" || target.Name == "AVX512")
}

// halfPrecSliceToUint16 wraps a half-precision slice expression with an unsafe
// cast to []uint16. Since hwy.Float16 and hwy.BFloat16 are defined as uint16,
// this is a safe reinterpretation:
//
//	unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
func halfPrecSliceToUint16(sliceExpr ast.Expr) ast.Expr {
	return &ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X:   ast.NewIdent("unsafe"),
			Sel: ast.NewIdent("Slice"),
		},
		Args: []ast.Expr{
			// (*uint16)(unsafe.Pointer(unsafe.SliceData(s)))
			&ast.CallExpr{
				Fun: &ast.ParenExpr{
					X: &ast.StarExpr{X: ast.NewIdent("uint16")},
				},
				Args: []ast.Expr{
					&ast.CallExpr{
						Fun: &ast.SelectorExpr{
							X:   ast.NewIdent("unsafe"),
							Sel: ast.NewIdent("Pointer"),
						},
						Args: []ast.Expr{
							&ast.CallExpr{
								Fun: &ast.SelectorExpr{
									X:   ast.NewIdent("unsafe"),
									Sel: ast.NewIdent("SliceData"),
								},
								Args: []ast.Expr{sliceExpr},
							},
						},
					},
				},
			},
			// len(s)
			&ast.CallExpr{
				Fun:  ast.NewIdent("len"),
				Args: []ast.Expr{sliceExpr},
			},
		},
	}
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
	case "NotEqual":
		return "NotEqual" + suffix
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

// transformHalfPrecisionFallback transforms functions for scalar half-precision implementations.
// hwy.Float16 and hwy.BFloat16 are defined as uint16 aliases, so standard arithmetic
// operators (+, -, *, /) perform integer arithmetic rather than floating-point.
//
// This transformation finds all scalar half-precision operations and wraps them to:
// 1. Convert inputs from hwy.Float16/BFloat16 to float32
// 2. Perform the operation in float32
// 3. Convert the result back to hwy.Float16/BFloat16 (if needed)
//
// This affects:
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

	// Track variables assigned from ReduceSum so we know they're float32.
	// These are float32 by virtue of ReduceSum wrapping, not from type conversions.
	reduceSumVars := make(map[string]bool)

	// Track variables whose values are computed as float32 intermediates from
	// type conversions like T(1.0) or binary expressions involving such conversions.
	// These variables are already float32 and should NOT be wrapped with .Float32().
	// They need conversion back to Float16/BFloat16 only at storage boundaries
	// (e.g., hwy.Set calls, return statements).
	//
	// IMPORTANT: Do not confuse with halfPrecisionScalarVars below. Variables from
	// type conversions like `x := T(1.0)` must go here, not halfPrecisionScalarVars.
	// Putting them in halfPrecisionScalarVars would cause wrapHalfPrecisionExpr to
	// emit x.Float32() (treating x as a uint16-backed half-precision value), and
	// would cause the type conversion itself to become hwy.Float32ToFloat16(1.0)
	// instead of float32(1.0).
	float32ComputedVars := make(map[string]bool)

	// Track variables that hold actual half-precision scalar values (uint16-backed
	// hwy.Float16 or hwy.BFloat16). These come from slice reads like `maxVal := input[0]`
	// or scalar function parameters. When used in arithmetic, wrapHalfPrecisionExpr
	// appends .Float32() to promote them for float32 computation.
	//
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
						// Check if either operand is a tracked variable (either scalar or computed)
						if ident, ok := binExpr.X.(*ast.Ident); ok {
							if float32ComputedVars[ident.Name] || (halfPrecisionScalarVars != nil && halfPrecisionScalarVars[ident.Name]) {
								hasTypeConv = true
							}
						}
						if ident, ok := binExpr.Y.(*ast.Ident); ok {
							if float32ComputedVars[ident.Name] || (halfPrecisionScalarVars != nil && halfPrecisionScalarVars[ident.Name]) {
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
		case *ast.ExprStmt:
			node.X = wrapHalfPrecisionExpr(node.X, ctx, toFloat32Method)

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
						// Check if the call already returns float32:
						// - hwy.ReduceSumF16/BF16 return float32
						// - asm type method .ReduceSum() returns float32 (NEON Float16x8/BFloat16x8)
						// But generic hwy.ReduceSum returns the element type, so don't skip that.
						alreadyFloat32 := false
						if sel, ok := callExpr.Fun.(*ast.SelectorExpr); ok {
							name := sel.Sel.Name
							if name == "ReduceSumF16" || name == "ReduceSumBF16" {
								alreadyFloat32 = true
							} else if name == "ReduceSum" || name == "ReduceMin" || name == "ReduceMax" {
								// Method call on asm vector type (not hwy.ReduceSum package function)
								if _, isIdent := sel.X.(*ast.Ident); isIdent {
									// v.ReduceSum() — receiver is a variable, so it's a method call
									// Check it's not hwy.ReduceSum/ReduceMin/ReduceMax
									if ident, ok := sel.X.(*ast.Ident); ok && ident.Name != "hwy" {
										alreadyFloat32 = true
									}
								}
							}
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
						// For := assignments, the variable's type is inferred as float32.
						// Track it so the return statement wraps it back to element type.
						// For = assignments (e.g., named return params already typed as element type),
						// we must convert the float32 result back to element type at the assignment.
						if len(node.Lhs) > i {
							if ident, ok := node.Lhs[i].(*ast.Ident); ok {
								if node.Tok == token.DEFINE {
									// := assignment: result infers float32 type
									reduceSumVars[ident.Name] = true
									delete(halfPrecisionScalarVars, ident.Name)
								} else {
									// = assignment: variable already has element type (e.g., named return param)
									// Wrap RHS with fromFloat32Func to convert back
									node.Rhs[i] = &ast.CallExpr{
										Fun:  parseTypeExpr(fromFloat32Func),
										Args: []ast.Expr{node.Rhs[i]},
									}
									// Variable stays as element type, no reduceSumVars tracking needed
								}
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
			// Handle expressions that compute in float32 (including tracked variables and expressions rewritten by wrapHalfPrecisionExpr)
			for i, result := range node.Results {
				// Optimization: if returning a simple half-precision variable, don't wrap/unwrap
				// Check if it's a simple identifier (unwrap parens)
				var unwrapped ast.Expr = result
				for {
					if p, ok := unwrapped.(*ast.ParenExpr); ok {
						unwrapped = p.X
					} else {
						break
					}
				}

				isSimpleHalfPrec := false
				if ident, ok := unwrapped.(*ast.Ident); ok {
					if ctx.halfPrecisionScalarVars != nil && ctx.halfPrecisionScalarVars[ident.Name] {
						isSimpleHalfPrec = true
					}
				}

				if isSimpleHalfPrec {
					// Just return the variable directly (it's already the right type)
					// No need to wrap with .Float32() and then convert back
					continue
				}

				// First, apply standard expression wrapping (e.g. input[i] -> input[i].Float32())
				newResult := wrapHalfPrecisionExpr(result, ctx, toFloat32Method)

				shouldWrap := false

				// Helper to check if a node is a tracked float32 variable
				var isTrackedVar func(ast.Expr) bool
				isTrackedVar = func(e ast.Expr) bool {
					switch t := e.(type) {
					case *ast.ParenExpr:
						return isTrackedVar(t.X)
					case *ast.Ident:
						return reduceSumVars[t.Name] || float32ComputedVars[t.Name]
					}
					return false
				}

				if isTrackedVar(result) {
					shouldWrap = true
				}

				// Check if the transformed expression is a float32 operation
				// Recursive check for paren unwrap
				var checkType func(ast.Expr) bool
				checkType = func(e ast.Expr) bool {
					switch t := e.(type) {
					case *ast.BinaryExpr:
						switch t.Op {
						case token.ADD, token.SUB, token.MUL, token.QUO:
							return true
						}
					case *ast.UnaryExpr:
						if t.Op == token.SUB || t.Op == token.ADD {
							return true
						}
					case *ast.CallExpr:
						// Check for .Float32() calls
						if sel, ok := t.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == toFloat32Method {
							return true
						}
						// Check for float32() conversions
						if ident, ok := t.Fun.(*ast.Ident); ok && ident.Name == "float32" {
							return true
						}
					case *ast.ParenExpr:
						return checkType(t.X)
					}
					return false
				}

				if !shouldWrap && checkType(newResult) {
					shouldWrap = true
				}

				if shouldWrap {
					// Optimization: If newResult is explicit float32(...) cast, strip it
					// because Float32ToFloat16 takes float32.
					arg := newResult
					if call, ok := newResult.(*ast.CallExpr); ok {
						if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "float32" && len(call.Args) == 1 {
							arg = call.Args[0]
						}
					}
					node.Results[i] = &ast.CallExpr{
						Fun:  parseTypeExpr(fromFloat32Func),
						Args: []ast.Expr{arg},
					}
				} else {
					node.Results[i] = newResult
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
							if float32ComputedVars[argIdent.Name] || reduceSumVars[argIdent.Name] {
								// Wrap: hwy.Set(x) -> hwy.Set(hwy.Float32ToFloat16(x))
								node.Args[0] = &ast.CallExpr{
									Fun:  parseTypeExpr(fromFloat32Func),
									Args: []ast.Expr{argIdent},
								}
							}
						}
					}
				}
			}
			// Handle AVX promoted asm.Broadcast*(uint16(X)) where X is float32
			if sel != nil {
				if ident, identOk := sel.X.(*ast.Ident); identOk && ident.Name == "asm" {
					funcName := sel.Sel.Name
					if strings.HasPrefix(funcName, "Broadcast") && len(node.Args) == 1 {
						if innerCall, innerOk := node.Args[0].(*ast.CallExpr); innerOk {
							if innerIdent, iOk := innerCall.Fun.(*ast.Ident); iOk && innerIdent.Name == "uint16" {
								if len(innerCall.Args) == 1 {
									if argIdent, argOk := innerCall.Args[0].(*ast.Ident); argOk {
										if float32ComputedVars[argIdent.Name] || reduceSumVars[argIdent.Name] {
											// Wrap: uint16(x) -> uint16(hwy.Float32ToFloat16(x))
											innerCall.Args[0] = &ast.CallExpr{
												Fun:  parseTypeExpr(fromFloat32Func),
												Args: []ast.Expr{argIdent},
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
					// Only skip for IndexExpr (slice read) or Ident (var copy) to allow
					// type conversions T(...) to be processed.
					switch rhs.(type) {
					case *ast.IndexExpr, *ast.Ident:
						skipConversion = true
					}
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
	case *ast.ParenExpr:
		return &ast.ParenExpr{
			Lparen: e.Lparen,
			X:      wrapHalfPrecisionExpr(e.X, ctx, toFloat32Method),
			Rparen: e.Rparen,
		}

	case *ast.UnaryExpr:
		return &ast.UnaryExpr{
			OpPos: e.OpPos,
			Op:    e.Op,
			X:     wrapHalfPrecisionExpr(e.X, ctx, toFloat32Method),
		}

	case *ast.Ident:
		if ctx.halfPrecisionScalarVars != nil && ctx.halfPrecisionScalarVars[e.Name] {
			return &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   cloneExpr(e),
					Sel: ast.NewIdent(toFloat32Method),
				},
			}
		}
		return e

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
		var sel *ast.SelectorExpr
		switch fun := e.Fun.(type) {
		case *ast.SelectorExpr:
			sel = fun
		case *ast.IndexExpr:
			// Handle generic calls like hwy.Set[T](x)
			if s, ok := fun.X.(*ast.SelectorExpr); ok {
				sel = s
			}
		}

		if sel != nil {
			if pkgIdent, ok := sel.X.(*ast.Ident); ok {
				funcName := sel.Sel.Name
				if pkgIdent.Name == "hwy" {
					// Check for type conversion
					if funcName == "Float16" || funcName == "BFloat16" {
						// Transform: hwy.Float16(x) to float32(x)
						// wrapHalfPrecisionExpr promotes everything to float32 for
						// scalar computation. Conversion back to half-precision is
						// handled at the assignment level by transformHalfPrecisionAssignment.
						e.Fun = ast.NewIdent("float32")

						// Recurse into arguments to wrap any half-precision scalars
						for i, arg := range e.Args {
							e.Args[i] = wrapHalfPrecisionExpr(arg, ctx, toFloat32Method)
						}
						return e
					}
					// Skip wrapping arguments for vector operations
					// These need half-precision arguments to produce half-precision vectors
					if isVectorOperation(funcName) {
						return e
					}
				}
				// Also skip wrapping for asm.* vector operations (NEON half-precision types)
				if pkgIdent.Name == "asm" && isVectorOperation(funcName) {
					return e
				}
				// Skip wrapping inside unsafe.Pointer() calls (used for NEON Load/Store)
				if pkgIdent.Name == "unsafe" && funcName == "Pointer" {
					return e
				}
			}
		}
		// Also check for simple type conversions like Float16(x) (without package qualifier)
		if ident, ok := e.Fun.(*ast.Ident); ok {
			if ident.Name == ctx.elemType || ident.Name == "hwy.Float16" || ident.Name == "hwy.BFloat16" {
				// Same as the selector case above: promote to float32 for
				// scalar computation. Assignment-level wrapping handles
				// conversion back to half-precision.
				e.Fun = ast.NewIdent("float32")
				// Recurse into arguments to wrap any half-precision scalars
				for i, arg := range e.Args {
					e.Args[i] = wrapHalfPrecisionExpr(arg, ctx, toFloat32Method)
				}
				return e
			}
			// Skip recursion for uint16 type conversions - these preserve the bit pattern
			// of half-precision values (hwy.Float16/BFloat16 are uint16 aliases)
			if ident.Name == "uint16" {
				return e
			}
		}
		// Recurse into call arguments for type conversions like float64(input[i] - maxVal)
		isBasicType := false
		if ident, ok := e.Fun.(*ast.Ident); ok {
			isBasicType = isBasicNumericType(ident.Name)
		}

		for i, arg := range e.Args {
			// If not a basic type conversion (like float64(x), int(x)),
			// skip wrapping simple half-precision arguments (identifiers, slice reads).
			// This allows passing half-precision values to functions directly without
			// promotion to float32.
			if !isBasicType && isSimpleHalfPrecision(arg, ctx) {
				continue
			}
			e.Args[i] = wrapHalfPrecisionExpr(arg, ctx, toFloat32Method)
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
	"Set": true, "Load": true, "LoadSlice": true, "Store": true, "StoreSlice": true, "Zero": true, "Broadcast": true,
	// NEON asm half-precision types (need uint16 args, not float32)
	"BroadcastFloat16x8": true, "BroadcastBFloat16x8": true,
	"LoadFloat16x8Ptr": true, "LoadBFloat16x8Ptr": true,
	"ZeroFloat16x8": true, "ZeroBFloat16x8": true,
	"StorePtr": true,
	// In-place operations (NEON)
	"MulAddAcc": true, "MulAddInto": true, "AddInto": true, "SubInto": true,
	"MulInto": true, "DivInto": true, "MinInto": true, "MaxInto": true,
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
	return slices.Contains(reduceBaseFunctions, funcName)
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

// isBasicNumericType returns true if the name corresponds to a Go basic numeric type.
func isBasicNumericType(name string) bool {
	switch name {
	case "float32", "float64", "complex64", "complex128",
		"int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64",
		"uintptr", "byte", "rune":
		return true
	}
	return false
}

// isSimpleHalfPrecision returns true if the expression represents a direct usage
// of a half-precision value (variable or slice read) without any computation.
func isSimpleHalfPrecision(expr ast.Expr, ctx *transformContext) bool {
	// Unwrap parens
	for {
		if p, ok := expr.(*ast.ParenExpr); ok {
			expr = p.X
		} else {
			break
		}
	}

	switch e := expr.(type) {
	case *ast.Ident:
		return ctx.halfPrecisionScalarVars != nil && ctx.halfPrecisionScalarVars[e.Name]
	case *ast.IndexExpr:
		return isHalfPrecisionSliceExpr(e, ctx)
	}
	return false
}
