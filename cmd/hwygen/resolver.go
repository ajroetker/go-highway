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
	"os"
	"path/filepath"
	"strings"

	"github.com/ajroetker/go-highway/cmd/hwygen/ir"
)

// FunctionRegistry resolves cross-package function references within go-highway.
// It parses *_base.go files from target packages and builds IR for them.
type FunctionRegistry struct {
	// moduleRoot is the absolute path to go-highway root (containing go.mod).
	moduleRoot string

	// moduleName is the Go module name (e.g., "github.com/ajroetker/go-highway").
	moduleName string

	// parsedPkgs caches parsed packages by import path.
	parsedPkgs map[string]*ParseResult

	// resolvedFuncs caches resolved functions by qualified name.
	resolvedFuncs map[string]*ResolvedFunc

	// fset is shared across all parsed files.
	fset *token.FileSet
}

// ResolvedFunc represents a function that has been resolved from another package.
type ResolvedFunc struct {
	// ImportPath is the full import path of the package.
	ImportPath string

	// Name is the function name (e.g., "BaseExpVec").
	Name string

	// ParsedFunc is the parsed function from the source file.
	ParsedFunc *ParsedFunc

	// TypeParams maps generic parameter names to concrete types.
	// E.g., {"T": "float32"} for BaseExpVec[float32].
	TypeParams map[string]string

	// FuncParams maps function parameter names to resolved functions.
	// For higher-order functions like BaseApply(in, out, fn).
	FuncParams map[string]*ResolvedFunc

	// IR is the built IR for this function (lazily computed).
	IR *ir.IRFunction

	// Dependencies are other functions this function calls.
	Dependencies []*ResolvedFunc
}

// NewFunctionRegistry creates a new registry for the given module.
func NewFunctionRegistry(moduleRoot, moduleName string) *FunctionRegistry {
	return &FunctionRegistry{
		moduleRoot:    moduleRoot,
		moduleName:    moduleName,
		parsedPkgs:    make(map[string]*ParseResult),
		resolvedFuncs: make(map[string]*ResolvedFunc),
		fset:          token.NewFileSet(),
	}
}

// Resolve looks up a function by its qualified name (e.g., "math.BaseExpVec")
// and returns the resolved function with the given type arguments.
func (r *FunctionRegistry) Resolve(qualifiedName string, typeArgs []string) (*ir.IRFunction, error) {
	// Parse the qualified name
	parts := strings.Split(qualifiedName, ".")
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid qualified name: %s", qualifiedName)
	}
	pkgAlias := parts[0]
	funcName := parts[1]

	// Build cache key
	cacheKey := qualifiedName
	if len(typeArgs) > 0 {
		cacheKey = fmt.Sprintf("%s[%s]", qualifiedName, strings.Join(typeArgs, ","))
	}

	// Check cache
	if resolved, ok := r.resolvedFuncs[cacheKey]; ok {
		return resolved.IR, nil
	}

	// Resolve the import path from the alias
	importPath, err := r.resolveImportPath(pkgAlias)
	if err != nil {
		return nil, fmt.Errorf("resolve import path for %s: %w", pkgAlias, err)
	}

	// Parse the package
	pkgResult, err := r.parsePackage(importPath)
	if err != nil {
		return nil, fmt.Errorf("parse package %s: %w", importPath, err)
	}

	// Find the function
	var pf *ParsedFunc
	for i := range pkgResult.Funcs {
		if pkgResult.Funcs[i].Name == funcName {
			pf = &pkgResult.Funcs[i]
			break
		}
	}
	if pf == nil {
		return nil, fmt.Errorf("function %s not found in package %s", funcName, importPath)
	}

	// Build type parameter map
	typeParams := make(map[string]string)
	for i, tp := range pf.TypeParams {
		if i < len(typeArgs) {
			typeParams[tp.Name] = typeArgs[i]
		}
	}

	// Resolve to ResolvedFunc
	resolved := &ResolvedFunc{
		ImportPath: importPath,
		Name:       funcName,
		ParsedFunc: pf,
		TypeParams: typeParams,
		FuncParams: make(map[string]*ResolvedFunc),
	}

	// Determine element type for IR building
	elemType := "float32"
	if len(typeArgs) > 0 {
		elemType = normalizeTypeName(typeArgs[0])
	}

	// Build IR
	builder := ir.NewBuilder(
		ir.WithImports(pkgResult.Imports),
		ir.WithElemType(elemType),
		ir.WithResolver(r),
	)

	irFunc, err := builder.Build(convertParsedFunc(pf))
	if err != nil {
		return nil, fmt.Errorf("build IR for %s: %w", qualifiedName, err)
	}

	resolved.IR = irFunc
	r.resolvedFuncs[cacheKey] = resolved

	return irFunc, nil
}

// resolveImportPath converts a package alias to a full import path.
// For go-highway internal packages:
//   - "math" → "github.com/ajroetker/go-highway/hwy/contrib/math"
//   - "algo" → "github.com/ajroetker/go-highway/hwy/contrib/algo"
//   - "hwy"  → "github.com/ajroetker/go-highway/hwy"
func (r *FunctionRegistry) resolveImportPath(alias string) (string, error) {
	// Built-in mappings for go-highway packages
	switch alias {
	case "math":
		return r.moduleName + "/hwy/contrib/math", nil
	case "algo":
		return r.moduleName + "/hwy/contrib/algo", nil
	case "nn":
		return r.moduleName + "/hwy/contrib/nn", nil
	case "hwy":
		return r.moduleName + "/hwy", nil
	case "stdmath":
		return "math", nil
	}

	// Try to find in parsed imports
	for _, pkg := range r.parsedPkgs {
		if path, ok := pkg.Imports[alias]; ok {
			return path, nil
		}
	}

	return "", fmt.Errorf("unknown package alias: %s", alias)
}

// parsePackage parses all *_base.go files in a package.
func (r *FunctionRegistry) parsePackage(importPath string) (*ParseResult, error) {
	// Check cache
	if result, ok := r.parsedPkgs[importPath]; ok {
		return result, nil
	}

	// Convert import path to filesystem path
	pkgPath, err := r.importPathToFilesystem(importPath)
	if err != nil {
		return nil, err
	}

	// Find *_base.go files
	baseFiles, err := filepath.Glob(filepath.Join(pkgPath, "*_base.go"))
	if err != nil {
		return nil, fmt.Errorf("glob base files: %w", err)
	}

	// Also look for files that define the functions we need
	// (some packages might not follow the *_base.go convention)
	allGoFiles, err := filepath.Glob(filepath.Join(pkgPath, "*.go"))
	if err != nil {
		return nil, fmt.Errorf("glob all files: %w", err)
	}

	// Combine, preferring base files
	files := baseFiles
	if len(files) == 0 {
		files = allGoFiles
	}

	// Parse all files and merge results
	result := &ParseResult{
		AllFuncs:           make(map[string]*ParsedFunc),
		TypeSpecificConsts: make(map[string]*TypeSpecificConst),
		Imports:            make(map[string]string),
		FileSet:            r.fset,
	}

	for _, file := range files {
		// Skip test files
		if strings.HasSuffix(file, "_test.go") {
			continue
		}

		parsed, err := r.parseFile(file)
		if err != nil {
			// Skip files that fail to parse (may have build tags, etc.)
			continue
		}

		// Merge results
		if result.PackageName == "" {
			result.PackageName = parsed.PackageName
		}

		for k, v := range parsed.Imports {
			result.Imports[k] = v
		}

		for name, pf := range parsed.AllFuncs {
			result.AllFuncs[name] = pf
		}

		result.Funcs = append(result.Funcs, parsed.Funcs...)

		for k, v := range parsed.TypeSpecificConsts {
			result.TypeSpecificConsts[k] = v
		}
	}

	r.parsedPkgs[importPath] = result
	return result, nil
}

// parseFile parses a single Go file.
func (r *FunctionRegistry) parseFile(filename string) (*ParseResult, error) {
	// Use the main Parse function
	return Parse(filename)
}

// importPathToFilesystem converts an import path to a filesystem path.
func (r *FunctionRegistry) importPathToFilesystem(importPath string) (string, error) {
	// Check if it's a go-highway internal import
	if strings.HasPrefix(importPath, r.moduleName) {
		// Strip module name prefix and join with module root
		relPath := strings.TrimPrefix(importPath, r.moduleName)
		relPath = strings.TrimPrefix(relPath, "/")
		return filepath.Join(r.moduleRoot, relPath), nil
	}

	// External package - not supported
	return "", fmt.Errorf("external packages not supported: %s", importPath)
}

// convertParsedFunc converts the main package's ParsedFunc to ir.ParsedFunc.
func convertParsedFunc(pf *ParsedFunc) *ir.ParsedFunc {
	result := &ir.ParsedFunc{
		Name: pf.Name,
		Body: pf.Body,
	}

	for _, tp := range pf.TypeParams {
		result.TypeParams = append(result.TypeParams, ir.TypeParamInput{
			Name:       tp.Name,
			Constraint: tp.Constraint,
		})
	}

	for _, p := range pf.Params {
		result.Params = append(result.Params, ir.ParamInput{
			Name: p.Name,
			Type: p.Type,
		})
	}

	for _, r := range pf.Returns {
		result.Returns = append(result.Returns, ir.ParamInput{
			Name: r.Name,
			Type: r.Type,
		})
	}

	return result
}

// normalizeTypeName converts type names to canonical form.
// E.g., "T" with constraint hwy.Floats and context float32 → "float32"
func normalizeTypeName(typeName string) string {
	switch typeName {
	case "hwy.Float16", "Float16":
		return "float16"
	case "hwy.BFloat16", "BFloat16":
		return "bfloat16"
	case "float32", "float64", "int32", "int64", "uint32", "uint64":
		return typeName
	default:
		return typeName
	}
}

// FindModuleRoot finds the module root directory containing go.mod.
func FindModuleRoot(startDir string) (string, string, error) {
	dir := startDir
	for {
		goModPath := filepath.Join(dir, "go.mod")
		if _, err := os.Stat(goModPath); err == nil {
			// Found go.mod - read module name
			content, err := os.ReadFile(goModPath)
			if err != nil {
				return "", "", fmt.Errorf("read go.mod: %w", err)
			}

			// Parse module name from first line
			lines := strings.Split(string(content), "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "module ") {
					moduleName := strings.TrimPrefix(line, "module ")
					return dir, moduleName, nil
				}
			}

			return "", "", fmt.Errorf("no module directive in go.mod")
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			return "", "", fmt.Errorf("go.mod not found")
		}
		dir = parent
	}
}

// ResolveHigherOrderFunction handles functions like BaseApply that take function arguments.
// It inlines the function argument and returns a specialized IR.
func (r *FunctionRegistry) ResolveHigherOrderFunction(
	qualifiedName string,
	typeArgs []string,
	funcArg string,
	funcArgTypeArgs []string,
) (*ir.IRFunction, error) {
	// First resolve the higher-order function
	hoFunc, err := r.Resolve(qualifiedName, typeArgs)
	if err != nil {
		return nil, err
	}

	// Then resolve the function argument
	argFunc, err := r.Resolve(funcArg, funcArgTypeArgs)
	if err != nil {
		return nil, err
	}

	// Create a specialized copy of the higher-order function
	// with the function argument inlined
	specialized := cloneIRFunction(hoFunc)
	specialized.Name = fmt.Sprintf("%s_%s", hoFunc.Name, argFunc.Name)

	// Find and replace OpKindCall nodes that reference the function parameter
	inlineFuncArg(specialized, argFunc)

	return specialized, nil
}

// cloneIRFunction creates a deep copy of an IRFunction.
func cloneIRFunction(fn *ir.IRFunction) *ir.IRFunction {
	clone := ir.NewFunction(fn.Name)
	clone.ElemType = fn.ElemType
	clone.Package = fn.Package
	clone.ImportPath = fn.ImportPath

	// Copy type params
	for _, tp := range fn.TypeParams {
		clone.TypeParams = append(clone.TypeParams, ir.TypeParam{
			Name:       tp.Name,
			Constraint: tp.Constraint,
		})
	}

	// Copy params
	for _, p := range fn.Params {
		clone.Params = append(clone.Params, ir.IRParam{
			Name:     p.Name,
			Type:     p.Type,
			IsSlice:  p.IsSlice,
			IsInt:    p.IsInt,
			IsFloat:  p.IsFloat,
			ElemType: p.ElemType,
		})
	}

	// Copy returns
	for _, r := range fn.Returns {
		clone.Returns = append(clone.Returns, ir.IRParam{
			Name:     r.Name,
			Type:     r.Type,
			IsSlice:  r.IsSlice,
			IsInt:    r.IsInt,
			IsFloat:  r.IsFloat,
			ElemType: r.ElemType,
		})
	}

	// Clone operations (deep copy)
	nodeMap := make(map[int]*ir.IRNode)
	var cloneNodes func([]*ir.IRNode) []*ir.IRNode
	cloneNodes = func(nodes []*ir.IRNode) []*ir.IRNode {
		result := make([]*ir.IRNode, len(nodes))
		for i, node := range nodes {
			cloned := clone.AddNode(node.Kind, node.Op)
			cloned.OutputTypes = append([]string{}, node.OutputTypes...)
			cloned.Outputs = append([]string{}, node.Outputs...)
			cloned.InputNames = append([]string{}, node.InputNames...)
			cloned.CallTarget = node.CallTarget
			cloned.CallTypeArgs = append([]string{}, node.CallTypeArgs...)
			cloned.FuncArg = node.FuncArg
			cloned.AllocSize = node.AllocSize
			cloned.AllocElemType = node.AllocElemType
			cloned.FusionGroup = node.FusionGroup
			cloned.IsFusionRoot = node.IsFusionRoot

			if node.LoopRange != nil {
				cloned.LoopRange = node.LoopRange.Clone()
			}

			if len(node.Children) > 0 {
				cloned.Children = cloneNodes(node.Children)
			}

			nodeMap[node.ID] = cloned
			result[i] = cloned
		}
		return result
	}

	clone.Operations = cloneNodes(fn.Operations)

	// Fix up input references
	var fixInputs func([]*ir.IRNode)
	fixInputs = func(nodes []*ir.IRNode) {
		for _, node := range nodes {
			for i, input := range node.Inputs {
				if cloned, ok := nodeMap[input.ID]; ok {
					node.Inputs[i] = cloned
				}
			}
			fixInputs(node.Children)
		}
	}
	fixInputs(clone.Operations)

	return clone
}

// inlineFuncArg inlines a function argument into a higher-order function.
func inlineFuncArg(hoFunc *ir.IRFunction, argFunc *ir.IRFunction) {
	// Find the function parameter name
	var funcParamName string
	for _, p := range hoFunc.Params {
		if strings.HasPrefix(p.Type, "func(") {
			funcParamName = p.Name
			break
		}
	}

	if funcParamName == "" {
		return
	}

	// Walk all nodes and replace calls to the function parameter
	var replaceInNodes func([]*ir.IRNode)
	replaceInNodes = func(nodes []*ir.IRNode) {
		for _, node := range nodes {
			if node.Kind == ir.OpKindCall && node.Op == funcParamName {
				// Replace with the inlined function's operations
				// For simplicity, we just mark this as the target function
				node.CallTarget = argFunc.Name
				node.Kind = argFunc.Operations[0].Kind // Use kind from first op
			}

			replaceInNodes(node.Children)
		}
	}
	replaceInNodes(hoFunc.Operations)
}

// ParsePackageFromPath parses a package given its filesystem path.
func (r *FunctionRegistry) ParsePackageFromPath(pkgPath string) (*ParseResult, error) {
	// Find *_base.go files
	baseFiles, err := filepath.Glob(filepath.Join(pkgPath, "*_base.go"))
	if err != nil {
		return nil, fmt.Errorf("glob base files: %w", err)
	}

	if len(baseFiles) == 0 {
		return nil, fmt.Errorf("no *_base.go files found in %s", pkgPath)
	}

	// Parse all files and merge
	result := &ParseResult{
		AllFuncs:           make(map[string]*ParsedFunc),
		TypeSpecificConsts: make(map[string]*TypeSpecificConst),
		Imports:            make(map[string]string),
		FileSet:            r.fset,
	}

	for _, file := range baseFiles {
		parsed, err := Parse(file)
		if err != nil {
			continue
		}

		if result.PackageName == "" {
			result.PackageName = parsed.PackageName
		}

		for k, v := range parsed.Imports {
			result.Imports[k] = v
		}

		for name, pf := range parsed.AllFuncs {
			result.AllFuncs[name] = pf
		}

		result.Funcs = append(result.Funcs, parsed.Funcs...)
	}

	return result, nil
}

// ExtractFunctionCalls extracts all function calls from an AST node.
func ExtractFunctionCalls(node ast.Node) []string {
	var calls []string

	ast.Inspect(node, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if pkg, ok := sel.X.(*ast.Ident); ok {
					calls = append(calls, pkg.Name+"."+sel.Sel.Name)
				}
			}
		}
		return true
	})

	return calls
}

// ParseFileForCalls parses a file and extracts function calls.
func ParseFileForCalls(filename string) ([]string, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, nil, 0)
	if err != nil {
		return nil, err
	}

	return ExtractFunctionCalls(file), nil
}
