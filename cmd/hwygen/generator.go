package main

import (
	"fmt"
	"go/ast"
	"strings"
)

// Generator orchestrates the code generation process.
type Generator struct {
	InputFile    string   // Input Go source file
	OutputDir    string   // Output directory
	Targets      []string // Target architectures (e.g., ["avx2", "fallback"])
	PackageOut   string   // Output package name (defaults to input package)
	DispatchName string   // Dispatch file prefix (defaults to function name)
	BulkMode     bool     // Generate bulk C code for NEON (for GOAT compilation)
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
				// Non-generic function, use a default type
				concreteTypes = []string{"float32"}
			}

			// Transform for each concrete type
			for _, elemType := range concreteTypes {
				transformResult := TransformWithOptions(&pf, target, elemType, transformOpts)

				// Add type suffix to function name if not float32
				if elemType != "float32" && len(pf.TypeParams) > 0 {
					transformResult.FuncDecl.Name.Name = transformResult.FuncDecl.Name.Name + "_" + strings.Title(elemType)
				}

				transformed = append(transformed, transformResult.FuncDecl)

				// Collect hoisted constants
				for _, hc := range transformResult.HoistedConsts {
					hoistedMap[hc.VarName] = hc
				}
			}
		}

		targetFuncs[target.Name] = transformed

		// Convert map to slice
		var hoistedSlice []HoistedConst
		for _, hc := range hoistedMap {
			hoistedSlice = append(hoistedSlice, hc)
		}
		targetHoisted[target.Name] = hoistedSlice
	}

	// 4. Emit the dispatcher file
	if err := EmitDispatcher(result.Funcs, targets, g.PackageOut, g.OutputDir, g.DispatchName); err != nil {
		return fmt.Errorf("emit dispatcher: %w", err)
	}

	// 5. Emit target-specific files
	baseFilename := getBaseFilename(g.InputFile)

	for _, target := range targets {
		funcDecls := targetFuncs[target.Name]
		if len(funcDecls) == 0 {
			continue
		}

		// Detect which contrib subpackages are needed for this specific target
		contribPkgs := detectContribPackagesForTarget(result.Funcs, target)
		hoistedConsts := targetHoisted[target.Name]
		if err := EmitTarget(funcDecls, target, g.PackageOut, baseFilename, g.OutputDir, contribPkgs, hoistedConsts); err != nil {
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
