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

// AsmAdapterInfo describes one dispatch variable → ASM adapter mapping,
// collected during ASM code generation for use in unified dispatch emission.
type AsmAdapterInfo struct {
	TargetName  string // "NEON", "AVX2", etc.
	Arch        string // "arm64", "amd64"
	DispatchVar string // "LiftUpdate53Int32"
	AdapterFunc string // "liftUpdate53AsmS32"
}

// Generator orchestrates the code generation process.
type Generator struct {
	InputFile      string       // Input Go source file
	OutputDir      string       // Output directory
	OutputPrefix   string       // Output file prefix (defaults to input file name without .go)
	TargetSpecs    []TargetSpec // Target architectures with generation modes
	PackageOut     string       // Output package name (defaults to input package)
	DispatchPrefix string       // Dispatch file prefix (defaults to function name)
	FusionMode     bool         // Enable IR-based fusion optimization
	Verbose        bool         // Verbose output for debugging
}

// Targets returns the list of target name strings (for backward compatibility).
func (g *Generator) Targets() []string {
	var names []string
	for _, ts := range g.TargetSpecs {
		names = append(names, strings.ToLower(ts.Target.Name))
	}
	return names
}

// CMode returns true if any target uses C or ASM mode.
func (g *Generator) CMode() bool {
	for _, ts := range g.TargetSpecs {
		if ts.Mode == TargetModeAsm || ts.Mode == TargetModeC {
			return true
		}
	}
	return false
}

// AsmMode returns true if any target uses ASM mode.
func (g *Generator) AsmMode() bool {
	for _, ts := range g.TargetSpecs {
		if ts.Mode == TargetModeAsm {
			return true
		}
	}
	return false
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

	// Handle legacy C-only mode (all targets are C or ASM, no Go SIMD targets)
	hasGoSimd := false
	for _, ts := range g.TargetSpecs {
		if ts.Mode == TargetModeGoSimd {
			hasGoSimd = true
			break
		}
	}
	if !hasGoSimd {
		return g.runCMode(result)
	}

	// Partition target specs by mode.
	// ASM targets also get Go SIMD generation because not all functions
	// may be ASM-eligible (e.g., Interleave, BFloat16 variants). The ASM
	// adapter init() selectively overrides the ASM-eligible dispatch vars.
	// SVE targets are excluded from Go SIMD generation because they have no
	// OpMap — their dispatch is handled entirely by the z_c_*.gen.go files
	// generated during ASM mode.
	var goSimdSpecs []TargetSpec
	var asmSpecs []TargetSpec
	var cOnlySpecs []TargetSpec
	for _, ts := range g.TargetSpecs {
		switch ts.Mode {
		case TargetModeGoSimd:
			goSimdSpecs = append(goSimdSpecs, ts)
		case TargetModeAsm:
			asmSpecs = append(asmSpecs, ts)
			// Also generate Go SIMD for this target, unless it's an SVE target
			// (SVE has no Go SIMD OpMap; dispatch is handled by C/ASM init files)
			if !isSVETarget(ts.Target) {
				goSimdSpecs = append(goSimdSpecs, TargetSpec{Target: ts.Target, Mode: TargetModeGoSimd})
			}
		case TargetModeC:
			cOnlySpecs = append(cOnlySpecs, ts)
		}
	}

	// 2. Go SIMD path: transform + emit for each Go SIMD target
	var goSimdTargets []Target
	for _, ts := range goSimdSpecs {
		goSimdTargets = append(goSimdTargets, ts.Target)
	}

	targetFuncs := make(map[string][]*ast.FuncDecl)
	targetHoisted := make(map[string][]HoistedConst)

	transformOpts := &TransformOptions{
		TypeSpecificConsts: result.TypeSpecificConsts,
		ConditionalBlocks:  result.ConditionalBlocks,
		FileSet:            result.FileSet,
		Imports:            result.Imports,
		AllFuncs:           result.AllFuncs,
	}

	for _, target := range goSimdTargets {
		var transformed []*ast.FuncDecl
		hoistedMap := make(map[string]HoistedConst)

		var genericHalfPrecFuncs map[string]bool
		if target.Name == "NEON" {
			genericHalfPrecFuncs = ComputeGenericHalfPrecFuncs(result.Funcs)
		}

		for _, pf := range result.Funcs {
			if hasInterfaceTypeParams(pf.TypeParams) && target.Name != "Fallback" {
				continue
			}

			var concreteTypes []string
			if len(pf.TypeParams) > 0 {
				concreteTypes = GetConcreteTypes(pf.TypeParams[0].Constraint)
			} else {
				concreteTypes = inferTypesFromParams(pf.Params)
			}

			for _, elemType := range concreteTypes {
				if target.Name == "NEON" &&
					(elemType == "hwy.Float16" || elemType == "hwy.BFloat16") &&
					genericHalfPrecFuncs[pf.Name] {
					transformOpts.SkipHalfPrecNEON = true
				} else {
					transformOpts.SkipHalfPrecNEON = false
				}

				transformResult := TransformWithOptions(&pf, target, elemType, transformOpts)

				if elemType != "float32" && len(pf.TypeParams) > 0 {
					transformResult.FuncDecl.Name.Name = transformResult.FuncDecl.Name.Name + "_" + typeNameToSuffix(elemType)
				}

				if pf.Private {
					transformResult.FuncDecl.Name.Name = makeUnexported(transformResult.FuncDecl.Name.Name)
				}

				transformed = append(transformed, transformResult.FuncDecl)

				for _, hc := range transformResult.HoistedConsts {
					hoistedMap[hc.VarName] = hc
				}
			}
		}

		targetFuncs[target.Name] = transformed

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

	// 3. ASM path: generate C, compile via GOAT, collect adapter info
	var asmAdapters []AsmAdapterInfo
	if len(asmSpecs) > 0 {
		adapters, err := g.runAsmMode(result, asmSpecs)
		if err != nil {
			return fmt.Errorf("asm mode: %w", err)
		}
		asmAdapters = adapters
	}

	// 4. C-only path
	if len(cOnlySpecs) > 0 {
		if err := g.runCOnlyMode(result, cOnlySpecs); err != nil {
			return fmt.Errorf("c mode: %w", err)
		}
	}

	// 5. Build the full list of targets for dispatch (Go SIMD + ASM targets)
	allTargets := make([]Target, len(goSimdTargets))
	copy(allTargets, goSimdTargets)
	for _, ts := range asmSpecs {
		// Only add if not already present as a Go SIMD target
		found := false
		for _, t := range allTargets {
			if t.Name == ts.Target.Name {
				found = true
				break
			}
		}
		if !found {
			allTargets = append(allTargets, ts.Target)
		}
	}

	// 6. Emit the dispatcher file with ASM adapter info
	if err := EmitDispatcher(result.Funcs, allTargets, g.PackageOut, g.OutputDir, g.DispatchPrefix, asmAdapters); err != nil {
		return fmt.Errorf("emit dispatcher: %w", err)
	}

	// 7. Emit target-specific files (Go SIMD only — ASM targets don't get Go SIMD impl files)
	baseFilename := g.OutputPrefix
	if baseFilename == "" {
		baseFilename = getBaseFilename(g.InputFile)
	}

	for _, target := range goSimdTargets {
		funcDecls := targetFuncs[target.Name]
		if len(funcDecls) == 0 {
			continue
		}

		contribPkgs := detectContribPackagesForTarget(result.Funcs, target)
		hoistedConsts := targetHoisted[target.Name]
		if err := EmitTarget(funcDecls, target, g.PackageOut, baseFilename, g.OutputDir, contribPkgs, hoistedConsts, result.Imports); err != nil {
			return fmt.Errorf("emit target %s: %w", target.Name, err)
		}
	}

	return nil
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
