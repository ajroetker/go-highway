package main

import (
	"fmt"
	"go/ast"
	"strings"
)

// Generator orchestrates the code generation process.
type Generator struct {
	InputFile  string   // Input Go source file
	OutputDir  string   // Output directory
	Targets    []string // Target architectures (e.g., ["avx2", "fallback"])
	PackageOut string   // Output package name (defaults to input package)
}

// Run executes the code generation pipeline.
func (g *Generator) Run() error {
	// 1. Parse the input file
	funcs, pkgName, err := Parse(g.InputFile)
	if err != nil {
		return fmt.Errorf("parse input: %w", err)
	}

	if len(funcs) == 0 {
		return fmt.Errorf("no functions with hwy operations found in %s", g.InputFile)
	}

	// Use input package name if output package not specified
	if g.PackageOut == "" {
		g.PackageOut = pkgName
	}

	// 2. Parse target configurations
	targets, err := g.parseTargets()
	if err != nil {
		return err
	}

	// 3. Transform functions for each target
	targetFuncs := make(map[string][]*ast.FuncDecl)

	for _, target := range targets {
		var transformed []*ast.FuncDecl

		for _, pf := range funcs {
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
				funcDecl := Transform(&pf, target, elemType)

				// Add type suffix to function name if not float32
				if elemType != "float32" && len(pf.TypeParams) > 0 {
					funcDecl.Name.Name = funcDecl.Name.Name + "_" + strings.Title(elemType)
				}

				transformed = append(transformed, funcDecl)
			}
		}

		targetFuncs[target.Name] = transformed
	}

	// 4. Emit the dispatcher file
	if err := EmitDispatcher(funcs, targets, g.PackageOut, g.OutputDir); err != nil {
		return fmt.Errorf("emit dispatcher: %w", err)
	}

	// 5. Emit target-specific files
	baseFilename := getBaseFilename(g.InputFile)

	// Detect if any function uses contrib
	needsContrib := false
	for _, pf := range funcs {
		for _, call := range pf.HwyCalls {
			if call.Package == "contrib" {
				needsContrib = true
				break
			}
		}
		if needsContrib {
			break
		}
	}

	for _, target := range targets {
		funcDecls := targetFuncs[target.Name]
		if len(funcDecls) == 0 {
			continue
		}

		if err := EmitTarget(funcDecls, target, g.PackageOut, baseFilename, g.OutputDir, needsContrib); err != nil {
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
