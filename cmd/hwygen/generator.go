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
	"sort"
	"strings"
)

// typeNameToSuffix converts an element type name to a valid function suffix.
// E.g., "float64" -> "Float64", "hwy.Float16" -> "Float16"
func typeNameToSuffix(elemType string) string {
	// Handle qualified names like hwy.Float16
	if idx := strings.LastIndex(elemType, "."); idx >= 0 {
		elemType = elemType[idx+1:]
	}
	// Capitalize first letter
	if len(elemType) > 0 {
		return strings.ToUpper(elemType[:1]) + elemType[1:]
	}
	return elemType
}

// Generator orchestrates the code generation process.
type Generator struct {
	InputFile      string   // Input Go source file
	OutputDir      string   // Output directory
	OutputPrefix   string   // Output file prefix (defaults to input file name without .go)
	Targets        []string // Target architectures (e.g., ["avx2", "fallback"])
	PackageOut     string   // Output package name (defaults to input package)
	DispatchPrefix string   // Dispatch file prefix (defaults to function name)
	BulkMode       bool     // Generate bulk C code for NEON (for GOAT compilation)
}

// Run executes the code generation pipeline.
func (g *Generator) Run() error {
	// 1. Parse the input file
	result, err := Parse(g.InputFile)
	if err != nil {
		return fmt.Errorf("parse input: %w", err)
	}

	if len(result.Funcs) == 0 {
		return fmt.Errorf("no functions with hwy operations found in %s", g.InputFile)
	}

	// Use input package name if output package not specified
	if g.PackageOut == "" {
		g.PackageOut = result.PackageName
	}

	// Handle bulk mode for NEON C code generation
	if g.BulkMode {
		return g.runBulkMode(result)
	}

	// 2. Parse target configurations
	targets, err := g.parseTargets()
	if err != nil {
		return err
	}

	// 3. Transform functions for each target
	targetFuncs := make(map[string][]*ast.FuncDecl)
	targetHoisted := make(map[string][]HoistedConst)

	// Prepare transform options with type-specific constants
	transformOpts := &TransformOptions{
		TypeSpecificConsts: result.TypeSpecificConsts,
		ConditionalBlocks:  result.ConditionalBlocks,
		FileSet:            result.FileSet,
		Imports:            result.Imports,
	}

	for _, target := range targets {
		var transformed []*ast.FuncDecl
		hoistedMap := make(map[string]HoistedConst) // Dedupe by var name

		for _, pf := range result.Funcs {
			// Skip SIMD generation for functions with interface type parameters
			// (like Predicate[T]). These functions can only use the fallback path
			// because the interface signature uses hwy.Vec[T] which is incompatible
			// with SIMD-specific vector types.
			if hasInterfaceTypeParams(pf.TypeParams) && target.Name != "Fallback" {
				continue
			}

			// Determine concrete types to generate
			var concreteTypes []string
			if len(pf.TypeParams) > 0 {
				concreteTypes = GetConcreteTypes(pf.TypeParams[0].Constraint)
			} else {
				// Non-generic function, infer type from parameters
				concreteTypes = inferTypesFromParams(pf.Params)
			}

			// Transform for each concrete type
			for _, elemType := range concreteTypes {
				transformResult := TransformWithOptions(&pf, target, elemType, transformOpts)

				// Add type suffix to function name if not float32
				if elemType != "float32" && len(pf.TypeParams) > 0 {
					transformResult.FuncDecl.Name.Name = transformResult.FuncDecl.Name.Name + "_" + typeNameToSuffix(elemType)
				}

				transformed = append(transformed, transformResult.FuncDecl)

				// Collect hoisted constants
				for _, hc := range transformResult.HoistedConsts {
					hoistedMap[hc.VarName] = hc
				}
			}
		}

		targetFuncs[target.Name] = transformed

		// Convert map to slice in deterministic order
		var hoistedSlice []HoistedConst
		hoistedKeys := make([]string, 0, len(hoistedMap))
		for k := range hoistedMap {
			hoistedKeys = append(hoistedKeys, k)
		}
		sort.Strings(hoistedKeys)
		for _, k := range hoistedKeys {
			hoistedSlice = append(hoistedSlice, hoistedMap[k])
		}
		targetHoisted[target.Name] = hoistedSlice
	}

	// 4. Emit the dispatcher file
	if err := EmitDispatcher(result.Funcs, targets, g.PackageOut, g.OutputDir, g.DispatchPrefix); err != nil {
		return fmt.Errorf("emit dispatcher: %w", err)
	}

	// 5. Emit target-specific files
	baseFilename := g.OutputPrefix
	if baseFilename == "" {
		baseFilename = getBaseFilename(g.InputFile)
	}

	for _, target := range targets {
		funcDecls := targetFuncs[target.Name]
		if len(funcDecls) == 0 {
			continue
		}

		// Detect which contrib subpackages are needed for this specific target
		contribPkgs := detectContribPackagesForTarget(result.Funcs, target)
		hoistedConsts := targetHoisted[target.Name]
		if err := EmitTarget(funcDecls, target, g.PackageOut, baseFilename, g.OutputDir, contribPkgs, hoistedConsts, result.Imports); err != nil {
			return fmt.Errorf("emit target %s: %w", target.Name, err)
		}
	}

	return nil
}

// parseTargets converts target name strings to Target configurations.
func (g *Generator) parseTargets() ([]Target, error) {
	var targets []Target

	for _, name := range g.Targets {
		target, err := GetTarget(name)
		if err != nil {
			return nil, err
		}
		targets = append(targets, target)
	}

	if len(targets) == 0 {
		return nil, fmt.Errorf("no valid targets specified")
	}

	return targets, nil
}

// inferTypesFromParams examines function parameters to infer the element type.
// For non-generic functions like BasePack32(src []uint32, ...), this returns the
// element type of the first slice parameter.
func inferTypesFromParams(params []Param) []string {
	for _, p := range params {
		// Look for slice types like []uint32, []uint64, []float32, etc.
		if after, ok := strings.CutPrefix(p.Type, "[]"); ok {
			elemType := after
			// Handle byte as alias for uint8
			if elemType == "byte" {
				return []string{"uint8"}
			}
			switch elemType {
			case "uint8", "uint16", "uint32", "uint64",
				"int8", "int16", "int32", "int64",
				"float32", "float64":
				return []string{elemType}
			}
		}
	}
	// Default to float32 if no slice parameter found
	return []string{"float32"}
}

// hasInterfaceTypeParams returns true if any type parameter has an interface constraint
// (as opposed to an element type constraint like hwy.Lanes, hwy.Floats, etc.)
func hasInterfaceTypeParams(typeParams []TypeParam) bool {
	for _, tp := range typeParams {
		// Element type constraints - these are NOT interface constraints
		if strings.Contains(tp.Constraint, "Lanes") ||
			strings.Contains(tp.Constraint, "Floats") ||
			strings.Contains(tp.Constraint, "Integers") ||
			strings.Contains(tp.Constraint, "SignedInts") ||
			strings.Contains(tp.Constraint, "UnsignedInts") {
			continue
		}
		// Any other constraint is considered an interface constraint
		return true
	}
	return false
}
