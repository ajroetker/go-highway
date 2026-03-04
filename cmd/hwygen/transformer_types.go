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
	"strconv"
	"strings"
)

// getVecPackageName returns the package name for vector types based on target.
// Returns "archsimd" for AVX targets, "asm" for NEON, defaulting to "archsimd".
func getVecPackageName(target Target) string {
	if target.VecPackage != "" {
		return target.VecPackage
	}
	return "archsimd"
}

// getShortTypeName returns the short type name like F32x8 for contrib functions.
func getShortTypeName(elemType string, target Target) string {
	lanes := target.LanesFor(elemType)
	return getShortTypeNameForLanes(elemType, lanes)
}

// elemTypeEntry holds type metadata for code generation.
type elemTypeEntry struct {
	ShortPrefix  string // Short prefix for function names: "F32", "I32", "Uint8"
	FullPrefix   string // Full prefix for vector type names: "Float32", "Int32", "Uint8"
	HwygenSuffix string // Suffix for hwygen-generated functions: "", "_Float64", "_Int32"
	SizeBytes    int    // Size in bytes: 4, 8, 1, 2
	GoSuffix     string // Go-convention suffix: "f32", "i32", "u8"
	CSuffix      string // C-convention suffix: "f32", "s32", "u8" (uses 's' for signed)
}

// elemTypeTable maps element type strings to their code generation metadata.
var elemTypeTable = map[string]elemTypeEntry{
	"float32":      {"F32", "Float32", "", 4, "f32", "f32"},
	"float64":      {"F64", "Float64", "_Float64", 8, "f64", "f64"},
	"int8":         {"Int8", "Int8", "_Int8", 1, "i8", "s8"},
	"int16":        {"Int16", "Int16", "_Int16", 2, "i16", "s16"},
	"int32":        {"I32", "Int32", "_Int32", 4, "i32", "s32"},
	"int64":        {"I64", "Int64", "_Int64", 8, "i64", "s64"},
	"uint8":        {"Uint8", "Uint8", "_Uint8", 1, "u8", "u8"},
	"uint16":       {"Uint16", "Uint16", "_Uint16", 2, "u16", "u16"},
	"uint32":       {"Uint32", "Uint32", "_Uint32", 4, "u32", "u32"},
	"uint64":       {"Uint64", "Uint64", "_Uint64", 8, "u64", "u64"},
	"hwy.Float16":  {"F16", "Float16", "_Float16", 2, "f16", "f16"},
	"hwy.BFloat16": {"BF16", "BFloat16", "_BFloat16", 2, "bf16", "bf16"},
	// Aliases for unqualified type names
	"float16":  {"F16", "Float16", "_Float16", 2, "f16", "f16"},
	"Float16":  {"F16", "Float16", "_Float16", 2, "f16", "f16"},
	"bfloat16": {"BF16", "BFloat16", "_BFloat16", 2, "bf16", "bf16"},
	"BFloat16": {"BF16", "BFloat16", "_BFloat16", 2, "bf16", "bf16"},
	"byte":     {"Uint8", "Uint8", "_Uint8", 1, "u8", "u8"},
}

// getShortTypeNameForLanes returns the short type name for a specific lane count.
func getShortTypeNameForLanes(elemType string, lanes int) string {
	if info, ok := elemTypeTable[elemType]; ok {
		return info.ShortPrefix + "x" + strconv.Itoa(lanes)
	}
	return "Vec"
}

// getHwygenTypeSuffix returns the type suffix used by hwygen for generated functions.
// float32 is the default (no suffix), other types get a type-specific suffix.
func getHwygenTypeSuffix(elemType string) string {
	if info, ok := elemTypeTable[elemType]; ok {
		return info.HwygenSuffix
	}
	return ""
}

// classifyTypeParams separates type parameters into element-type params
// (constrained by Lanes, Floats, Integers, etc.) and interface-type params
// (constrained by other interfaces like Predicate[T]).
func classifyTypeParams(typeParams []TypeParam) (elementTypeParams map[string]bool, interfaceTypeParams map[string]string) {
	elementTypeParams = make(map[string]bool)
	interfaceTypeParams = make(map[string]string)
	for _, tp := range typeParams {
		if strings.Contains(tp.Constraint, "Lanes") ||
			strings.Contains(tp.Constraint, "Floats") ||
			strings.Contains(tp.Constraint, "Integers") ||
			strings.Contains(tp.Constraint, "SignedInts") ||
			strings.Contains(tp.Constraint, "UnsignedInts") {
			elementTypeParams[tp.Name] = true
		} else {
			interfaceTypeParams[tp.Name] = tp.Constraint
		}
	}
	return
}

// specializeType replaces generic type parameters with concrete types.
// For SIMD targets, also transforms hwy.Vec[T] to archsimd/asm vector types.
func specializeType(typeStr string, typeParams []TypeParam, elemType string) string {
	return specializeTypeWithMap(typeStr, typeParams, elemType, nil)
}

// specializeTypeWithMap is like specializeType but resolves each type parameter
// independently using the provided typeMap. When typeMap is nil, it behaves
// identically to specializeType (all element type params resolve to elemType).
func specializeTypeWithMap(typeStr string, typeParams []TypeParam, elemType string, typeMap map[string]string) string {
	elementTypeParams, interfaceTypeParams := classifyTypeParams(typeParams)

	// Replace element type parameters using typeMap for per-param resolution
	for _, tp := range typeParams {
		if !elementTypeParams[tp.Name] {
			continue
		}
		// Skip string operations when the type parameter name doesn't appear at all
		if !strings.Contains(typeStr, tp.Name) {
			continue
		}
		resolvedType := elemType
		if typeMap != nil {
			if ct, ok := typeMap[tp.Name]; ok {
				resolvedType = ct
			}
		}
		typeStr = strings.ReplaceAll(typeStr, "hwy.Vec["+tp.Name+"]", "hwy.Vec["+resolvedType+"]")
		typeStr = strings.ReplaceAll(typeStr, "hwy.Mask["+tp.Name+"]", "hwy.Mask["+resolvedType+"]")
		typeStr = strings.ReplaceAll(typeStr, "[]"+tp.Name, "[]"+resolvedType)
		typeStr = replaceTypeParam(typeStr, tp.Name, resolvedType)
	}

	// For interface type parameters, specialize using the primary elemType
	for paramName, constraint := range interfaceTypeParams {
		if typeStr == paramName {
			specializedConstraint := constraint
			for _, tp := range typeParams {
				if elementTypeParams[tp.Name] {
					resolvedType := elemType
					if typeMap != nil {
						if ct, ok := typeMap[tp.Name]; ok {
							resolvedType = ct
						}
					}
					specializedConstraint = strings.ReplaceAll(specializedConstraint, "["+tp.Name+"]", "["+resolvedType+"]")
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
	// Replace T when it's the whole string
	if typeStr == paramName {
		return elemType
	}

	result := typeStr

	// Replace [T] with [elemType]
	result = strings.ReplaceAll(result, "["+paramName+"]", "["+elemType+"]")

	// Replace T in slice types []T
	result = strings.ReplaceAll(result, "[]"+paramName, "[]"+elemType)

	// Early exit if paramName no longer appears
	if !strings.Contains(result, paramName) {
		return result
	}

	// Replace T in function types and map value types - look for patterns like "T)" or "T," or "(T" or "]T"
	for _, suffix := range []string{")", ",", " ", ""} {
		for _, prefix := range []string{"(", ",", " ", "]"} {
			old := prefix + paramName + suffix
			new := prefix + elemType + suffix
			result = strings.ReplaceAll(result, old, new)
		}
	}

	return result
}

// complexHalfPrecOps lists hwy.* operations that cannot be converted to asm.Float16x8/BFloat16x8
// method calls. Functions using these must stay on the generic hwy.Vec[T] path for half-precision NEON.
var complexHalfPrecOps = map[string]bool{
	"RoundToEven":            true,
	"ConvertToInt32":         true,
	"ConvertToFloat32":       true,
	"Pow2":                   true,
	"GetExponent":            true,
	"GetMantissa":            true,
	"ConvertExponentToFloat": true,
	"Equal":                  true,
	"MaskAnd":                true,
	"Pow":                    true,
}

// externalGenericHalfPrecPkgs lists package names whose Base*Vec functions use the
// generic hwy.Vec[T] path for half-precision NEON (and thus callers must also use it).
var externalGenericHalfPrecPkgs = map[string]bool{
	"math": true,
}

// NeedsGenericHalfPrecisionPath scans a function body for hwy.* calls that cannot be
// converted to asm types for NEON half-precision, or references to external package
// functions (like math.BaseExpVec) that use the generic path. Returns true if the
// function should use the generic hwy.Vec[T] path instead of asm.Float16x8/BFloat16x8.
func NeedsGenericHalfPrecisionPath(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		// Check all selector expressions (not just call targets) for function value references
		// like math.BaseExpVec passed as argument
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if ident, ok := sel.X.(*ast.Ident); ok {
				if ident.Name == "hwy" {
					if complexHalfPrecOps[sel.Sel.Name] {
						found = true
						return false
					}
				}
				// Detect references to external package functions (e.g., math.BaseExpVec)
				if externalGenericHalfPrecPkgs[ident.Name] && strings.HasPrefix(sel.Sel.Name, "Base") {
					found = true
					return false
				}
			}
		}
		return true
	})
	return found
}

// CollectBaseFuncCalls returns the set of Base* function names called from a function body.
func CollectBaseFuncCalls(body *ast.BlockStmt) map[string]bool {
	calls := make(map[string]bool)
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		// Check direct Base* calls: BaseFoo(args)
		if ident, ok := call.Fun.(*ast.Ident); ok {
			if strings.HasPrefix(ident.Name, "Base") {
				calls[ident.Name] = true
			}
		}
		// Check generic Base* calls: BaseFoo[T](args)
		if indexExpr, ok := call.Fun.(*ast.IndexExpr); ok {
			if ident, ok := indexExpr.X.(*ast.Ident); ok {
				if strings.HasPrefix(ident.Name, "Base") {
					calls[ident.Name] = true
				}
			}
		}
		return true
	})
	return calls
}

// hasFuncParamWithVecType checks if any function parameter is a function type
// containing hwy.Vec. Such functions must use the generic half-precision path
// because callers from other files may pass generic-typed callbacks.
// For example: func BaseApply[T hwy.Floats](in, out []T, fn func(hwy.Vec[T]) hwy.Vec[T])
func hasFuncParamWithVecType(params []Param) bool {
	for _, p := range params {
		if strings.HasPrefix(p.Type, "func(") && strings.Contains(p.Type, "hwy.Vec[") {
			return true
		}
	}
	return false
}

// ComputeGenericHalfPrecFuncs computes the set of function names that need the generic
// hwy.Vec[T] path for half-precision NEON, including transitive dependencies through
// Base* function calls.
func ComputeGenericHalfPrecFuncs(funcs []ParsedFunc) map[string]bool {
	// Pass 1: identify functions that directly use complex ops or have
	// function-typed parameters with hwy.Vec (cross-file compatibility)
	genericFuncs := make(map[string]bool)
	callGraph := make(map[string]map[string]bool) // caller -> set of Base* callees

	for _, pf := range funcs {
		if NeedsGenericHalfPrecisionPath(pf.Body) || hasFuncParamWithVecType(pf.Params) {
			genericFuncs[pf.Name] = true
		}
		callGraph[pf.Name] = CollectBaseFuncCalls(pf.Body)
	}

	// Pass 2: propagate transitively in both directions:
	// - If a function calls a generic-path function, the caller must also be on the generic path
	//   (because it will receive hwy.Vec[T] return values)
	// - If a generic-path function calls a non-generic function, the callee must also be on the
	//   generic path (because it will receive hwy.Vec[T] arguments from the generic caller)
	changed := true
	for changed {
		changed = false
		for _, pf := range funcs {
			if genericFuncs[pf.Name] {
				// Propagate to callees: generic caller forces callees to be generic
				for callee := range callGraph[pf.Name] {
					if !genericFuncs[callee] {
						genericFuncs[callee] = true
						changed = true
					}
				}
				continue
			}
			// Propagate to callers: if any callee is generic, caller must be too
			for callee := range callGraph[pf.Name] {
				if genericFuncs[callee] {
					genericFuncs[pf.Name] = true
					changed = true
					break
				}
			}
		}
	}

	return genericFuncs
}

// specializeVecType transforms hwy.Vec[elemType] and hwy.Mask[elemType] to concrete archsimd/asm types.
// For example: hwy.Vec[float32] -> archsimd.Float32x8 (for AVX2)
//
//	hwy.Mask[float32] -> archsimd.Int32x8 (for AVX2)
//
// extractVecElemType extracts the concrete element type from a specialized type string
// for use with specializeVecType. For types containing "hwy.Vec[X]", returns X.
// Falls back to the provided primaryElemType if no Vec type is found.
//
// Examples:
//
//	"hwy.Vec[float32]" → "float32"
//	"hwy.Vec[hwy.Float16]" → "hwy.Float16"
//	"[]float32" → "float32" (uses primaryElemType)
func extractVecElemType(specializedType, primaryElemType string) string {
	// Look for hwy.Vec[...] pattern
	prefix := "hwy.Vec["
	if _, after, ok := strings.Cut(specializedType, prefix); ok {
		rest := after
		if before, _, ok := strings.Cut(rest, "]"); ok {
			return before
		}
	}
	return primaryElemType
}

// If skipHalfPrec is true, half-precision types on NEON are NOT converted to asm types,
// keeping them on the generic hwy.Vec[T] path (used for functions with complex ops like RoundToEven).
func specializeVecType(typeStr string, elemType string, target Target, skipHalfPrec ...bool) string {
	if target.IsFallback() {
		// For fallback, keep hwy.Vec[float32], hwy.Mask[float32] etc.
		return typeStr
	}

	// Fast path: skip all placeholder construction and scanning for types that
	// cannot contain hwy.Vec or hwy.Mask (scalar types, slice types, etc.).
	if !strings.Contains(typeStr, "hwy.") {
		return typeStr
	}

	// For Float16/BFloat16 on NEON, convert to concrete asm types (Float16x8, BFloat16x8).
	// For Float16/BFloat16 on AVX2/AVX512, convert to promoted asm types that wrap archsimd.Float32x{8,16}.
	// On Fallback, keep hwy.Vec[hwy.Float16].
	if isHalfPrecisionType(elemType) {
		if target.IsNEON() && !(len(skipHalfPrec) > 0 && skipHalfPrec[0]) {
			asmType := "asm.Float16x8"
			if isBFloat16Type(elemType) {
				asmType = "asm.BFloat16x8"
			}
			vecPlaceholder := "hwy.Vec[" + elemType + "]"
			typeStr = strings.ReplaceAll(typeStr, vecPlaceholder, asmType)
			// Also handle Mask types for half-precision on NEON
			maskPlaceholder := "hwy.Mask[" + elemType + "]"
			typeStr = strings.ReplaceAll(typeStr, maskPlaceholder, "asm.Uint16x8")
			return typeStr
		}
		if target.Name == "AVX2" || target.Name == "AVX512" {
			asmType := "asm." + target.TypeMap[elemType]
			vecPlaceholder := "hwy.Vec[" + elemType + "]"
			typeStr = strings.ReplaceAll(typeStr, vecPlaceholder, asmType)
			// Half-precision on AVX uses Mask32x{8,16} since underlying data is float32
			maskPlaceholder := "hwy.Mask[" + elemType + "]"
			lanes := target.LanesFor(elemType)
			maskType := fmt.Sprintf("archsimd.Mask32x%d", lanes)
			typeStr = strings.ReplaceAll(typeStr, maskPlaceholder, maskType)
			return typeStr
		}
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
			// Half-precision on AVX uses float32 promoted storage, so Mask32xN
			if isHalfPrecisionType(elemType) {
				return fmt.Sprintf("Mask32x%d", lanes)
			}
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
	if info, ok := elemTypeTable[elemType]; ok {
		return info.FullPrefix + "x" + strconv.Itoa(lanes)
	}
	return "Vec"
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
	if info, ok := elemTypeTable[elemType]; ok {
		return info.SizeBytes
	}
	return 0
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
func cloneFuncType(ft *ast.FuncType) *ast.FuncType {
	if ft == nil {
		return nil
	}
	return &ast.FuncType{
		Params:  cloneFieldList(ft.Params),
		Results: cloneFieldList(ft.Results),
	}
}

func cloneFieldList(fl *ast.FieldList) *ast.FieldList {
	if fl == nil {
		return nil
	}
	fields := make([]*ast.Field, len(fl.List))
	for i, f := range fl.List {
		names := make([]*ast.Ident, len(f.Names))
		for j, n := range f.Names {
			names[j] = ast.NewIdent(n.Name)
		}
		fields[i] = &ast.Field{
			Names: names,
			Type:  cloneExpr(f.Type),
		}
	}
	return &ast.FieldList{List: fields}
}

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

// genPowIIFE generates an IIFE that computes element-wise Pow for AVX promoted half-precision types.
// It generates: func() asm.Type { var _powBase, _powExp [N]float32; base.AsFloat32xN().StoreSlice(_powBase[:]); ...; return asm.TypeFromFloat32xN(archsimd.LoadFloat32xNSlice(_powBase[:])) }
func genPowIIFE(asmType, wrapFunc, loadFunc, asF32Method, vecPkg, lanesStr string, baseArg, expArg ast.Expr) *ast.FuncLit {
	// base.AsFloat32xN().StoreSlice(_powBase[:])
	storeBase := &ast.ExprStmt{X: &ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X: &ast.CallExpr{
				Fun: &ast.SelectorExpr{X: cloneExpr(baseArg), Sel: ast.NewIdent(asF32Method)},
			},
			Sel: ast.NewIdent("StoreSlice"),
		},
		Args: []ast.Expr{&ast.SliceExpr{X: ast.NewIdent("_powBase")}},
	}}
	// exp.AsFloat32xN().StoreSlice(_powExp[:])
	storeExp := &ast.ExprStmt{X: &ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X: &ast.CallExpr{
				Fun: &ast.SelectorExpr{X: cloneExpr(expArg), Sel: ast.NewIdent(asF32Method)},
			},
			Sel: ast.NewIdent("StoreSlice"),
		},
		Args: []ast.Expr{&ast.SliceExpr{X: ast.NewIdent("_powExp")}},
	}}

	stmts := []ast.Stmt{
		// var _powBase, _powExp [N]float32
		&ast.DeclStmt{Decl: &ast.GenDecl{Tok: token.VAR, Specs: []ast.Spec{
			&ast.ValueSpec{
				Names: []*ast.Ident{ast.NewIdent("_powBase"), ast.NewIdent("_powExp")},
				Type: &ast.ArrayType{
					Len: &ast.BasicLit{Kind: token.INT, Value: lanesStr},
					Elt: ast.NewIdent("float32"),
				},
			},
		}}},
		storeBase,
		storeExp,
		// for _powI := range _powBase { _powBase[_powI] = float32(math.Pow(...)) }
		&ast.RangeStmt{
			Key: ast.NewIdent("_powI"), Tok: token.DEFINE, X: ast.NewIdent("_powBase"),
			Body: &ast.BlockStmt{List: []ast.Stmt{
				&ast.AssignStmt{
					Lhs: []ast.Expr{&ast.IndexExpr{X: ast.NewIdent("_powBase"), Index: ast.NewIdent("_powI")}},
					Tok: token.ASSIGN,
					Rhs: []ast.Expr{&ast.CallExpr{Fun: ast.NewIdent("float32"), Args: []ast.Expr{
						&ast.CallExpr{
							Fun: &ast.SelectorExpr{X: ast.NewIdent("stdmath"), Sel: ast.NewIdent("Pow")},
							Args: []ast.Expr{
								&ast.CallExpr{Fun: ast.NewIdent("float64"), Args: []ast.Expr{&ast.IndexExpr{X: ast.NewIdent("_powBase"), Index: ast.NewIdent("_powI")}}},
								&ast.CallExpr{Fun: ast.NewIdent("float64"), Args: []ast.Expr{&ast.IndexExpr{X: ast.NewIdent("_powExp"), Index: ast.NewIdent("_powI")}}},
							},
						},
					}}},
				},
			}},
		},
		// return asm.TypeFromFloat32xN(archsimd.LoadFloat32xNSlice(_powBase[:]))
		&ast.ReturnStmt{Results: []ast.Expr{
			&ast.CallExpr{
				Fun: &ast.SelectorExpr{X: ast.NewIdent("asm"), Sel: ast.NewIdent(wrapFunc)},
				Args: []ast.Expr{&ast.CallExpr{
					Fun:  &ast.SelectorExpr{X: ast.NewIdent(vecPkg), Sel: ast.NewIdent(loadFunc)},
					Args: []ast.Expr{&ast.SliceExpr{X: ast.NewIdent("_powBase")}},
				}},
			},
		}},
	}

	return &ast.FuncLit{
		Type: &ast.FuncType{
			Results: &ast.FieldList{List: []*ast.Field{{
				Type: &ast.SelectorExpr{X: ast.NewIdent("asm"), Sel: ast.NewIdent(asmType)},
			}}},
		},
		Body: &ast.BlockStmt{List: stmts},
	}
}

// cloneExpr creates a deep copy of an expression.
func cloneExpr(expr ast.Expr) ast.Expr {
	return cloneExprWithDepth(expr, 0)
}

const maxCloneDepth = 1000

func cloneExprWithDepth(expr ast.Expr, depth int) ast.Expr {
	if expr == nil {
		return nil
	}
	if depth > maxCloneDepth {
		panic(fmt.Sprintf("cloneExpr: max depth %d exceeded, expression type: %T", maxCloneDepth, expr))
	}

	switch e := expr.(type) {
	case *ast.Ident:
		return &ast.Ident{Name: e.Name}
	case *ast.BasicLit:
		return &ast.BasicLit{Kind: e.Kind, Value: e.Value}
	case *ast.SelectorExpr:
		return &ast.SelectorExpr{
			X:   cloneExprWithDepth(e.X, depth+1),
			Sel: ast.NewIdent(e.Sel.Name),
		}
	case *ast.CallExpr:
		args := make([]ast.Expr, len(e.Args))
		for i, arg := range e.Args {
			args[i] = cloneExprWithDepth(arg, depth+1)
		}
		return &ast.CallExpr{
			Fun:      cloneExprWithDepth(e.Fun, depth+1),
			Args:     args,
			Ellipsis: e.Ellipsis,
		}
	case *ast.BinaryExpr:
		return &ast.BinaryExpr{
			X:  cloneExprWithDepth(e.X, depth+1),
			Op: e.Op,
			Y:  cloneExprWithDepth(e.Y, depth+1),
		}
	case *ast.UnaryExpr:
		return &ast.UnaryExpr{
			Op: e.Op,
			X:  cloneExprWithDepth(e.X, depth+1),
		}
	case *ast.ParenExpr:
		return &ast.ParenExpr{X: cloneExprWithDepth(e.X, depth+1)}
	case *ast.IndexExpr:
		return &ast.IndexExpr{
			X:     cloneExprWithDepth(e.X, depth+1),
			Index: cloneExprWithDepth(e.Index, depth+1),
		}
	case *ast.SliceExpr:
		return &ast.SliceExpr{
			X:      cloneExprWithDepth(e.X, depth+1),
			Low:    cloneExprWithDepth(e.Low, depth+1),
			High:   cloneExprWithDepth(e.High, depth+1),
			Max:    cloneExprWithDepth(e.Max, depth+1),
			Slice3: e.Slice3,
		}
	case *ast.StarExpr:
		return &ast.StarExpr{X: cloneExprWithDepth(e.X, depth+1)}
	case *ast.TypeAssertExpr:
		return &ast.TypeAssertExpr{
			X:    cloneExprWithDepth(e.X, depth+1),
			Type: cloneExprWithDepth(e.Type, depth+1),
		}
	case *ast.ArrayType:
		return &ast.ArrayType{
			Len: cloneExprWithDepth(e.Len, depth+1),
			Elt: cloneExprWithDepth(e.Elt, depth+1),
		}
	case *ast.CompositeLit:
		elts := make([]ast.Expr, len(e.Elts))
		for i, elt := range e.Elts {
			elts[i] = cloneExprWithDepth(elt, depth+1)
		}
		return &ast.CompositeLit{
			Type: cloneExprWithDepth(e.Type, depth+1),
			Elts: elts,
		}
	case *ast.FuncLit:
		return &ast.FuncLit{
			Type: cloneFuncType(e.Type),
			Body: cloneBlockStmt(e.Body),
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
