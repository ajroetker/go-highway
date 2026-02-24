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
	"slices"
	"strings"
)

// getTypeCombinations returns the type combinations for a function.
// If the function has explicit //hwy:gen annotations, those are returned.
// Otherwise, single-type combos are built from GetConcreteTypes on the first type param.
// For non-generic functions, returns a single combo with no type map.
func getTypeCombinations(pf *ParsedFunc) []TypeCombination {
	if len(pf.TypeCombinations) > 0 {
		return pf.TypeCombinations
	}
	if len(pf.TypeParams) == 0 {
		// Non-generic function: single combo with empty type map
		return []TypeCombination{{Types: nil}}
	}
	// Single type param: wrap each concrete type as a single-entry combo
	concreteTypes := GetConcreteTypes(pf.TypeParams[0].Constraint)
	combos := make([]TypeCombination, len(concreteTypes))
	for i, ct := range concreteTypes {
		combos[i] = TypeCombination{Types: map[string]string{pf.TypeParams[0].Name: ct}}
	}
	return combos
}

// comboTypeSuffix builds a function name suffix from a type combination.
// For single-type combos, returns the single type suffix (e.g., "Float32").
// For multi-type combos, concatenates all type suffixes in TypeParams order (e.g., "Float16Float32").
func comboTypeSuffix(combo TypeCombination, typeParams []TypeParam) string {
	if len(combo.Types) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, tp := range typeParams {
		if ct, ok := combo.Types[tp.Name]; ok {
			sb.WriteString(typeNameToSuffix(ct))
		}
	}
	return sb.String()
}

// comboPrimaryType returns the primary element type from a combo (the first type param's value).
// Falls back to "float32" if the combo has no types.
func comboPrimaryType(combo TypeCombination, typeParams []TypeParam) string {
	if len(combo.Types) == 0 {
		return "float32"
	}
	if len(typeParams) > 0 {
		if ct, ok := combo.Types[typeParams[0].Name]; ok {
			return ct
		}
	}
	// Non-generic functions use "" as the key for inferred types
	if ct, ok := combo.Types[""]; ok {
		return ct
	}
	return "float32"
}

// isMultiTypeCombination returns true if the combo has more than one type param.
func isMultiTypeCombination(combo TypeCombination) bool {
	return len(combo.Types) > 1
}

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

// DispatchGroup represents a set of functions that share a single dispatch
// interface. One function is the "primary" (defines the dispatch signature
// and widest constraint), and zero or more are "specializations" that
// override specific (target, combo) slots.
type DispatchGroup struct {
	// GroupName is the dispatch name without "Base" prefix.
	GroupName string

	// Primary is the default function used when no specialization matches.
	Primary *ParsedFunc

	// Specializations are functions with //hwy:specializes pointing to this group.
	Specializations []*ParsedFunc

	// AllCombos is the union of all TypeCombinations across Primary and
	// all Specializations. This determines the dispatch vars.
	AllCombos []TypeCombination

	// AllTypeParams is the type parameters from the Primary (widest constraint).
	AllTypeParams []TypeParam

	// Private reflects whether the primary function uses lowercase "base".
	Private bool
}

// buildDispatchGroups partitions a list of parsed functions into dispatch groups.
// Functions without //hwy:specializes each form their own single-member group.
// Functions with //hwy:specializes join the group of the referenced primary function.
func buildDispatchGroups(funcs []ParsedFunc) ([]DispatchGroup, error) {
	// Step 1: Separate primary functions from specializations
	type primaryEntry struct {
		idx int
		pf  *ParsedFunc
	}
	primaryByGroup := make(map[string]*primaryEntry)
	var specFuncs []*ParsedFunc

	for i := range funcs {
		pf := &funcs[i]
		if pf.SpecializesGroup != "" {
			specFuncs = append(specFuncs, pf)
		} else {
			groupName := deriveFuncGroupName(pf.Name)
			if existing, ok := primaryByGroup[groupName]; ok {
				return nil, fmt.Errorf("duplicate primary function for group %q: %s and %s",
					groupName, existing.pf.Name, pf.Name)
			}
			primaryByGroup[groupName] = &primaryEntry{idx: i, pf: pf}
		}
	}

	// Step 2: Validate specializations reference existing groups
	specsByGroup := make(map[string][]*ParsedFunc)
	for _, pf := range specFuncs {
		if _, ok := primaryByGroup[pf.SpecializesGroup]; !ok {
			return nil, fmt.Errorf("specialization %s references unknown group %q (available: %v)",
				pf.Name, pf.SpecializesGroup, groupNames(primaryByGroup))
		}
		specsByGroup[pf.SpecializesGroup] = append(specsByGroup[pf.SpecializesGroup], pf)
	}

	// Step 3: Build DispatchGroups
	var groups []DispatchGroup
	for groupName, entry := range primaryByGroup {
		primary := entry.pf
		specs := specsByGroup[groupName]

		// Validate signature compatibility
		for _, spec := range specs {
			if err := validateSignatureCompatibility(primary, spec); err != nil {
				return nil, fmt.Errorf("specialization %s incompatible with %s: %w",
					spec.Name, primary.Name, err)
			}
		}

		// Compute union of all combos
		allCombos := computeUnionCombos(primary, specs)

		// Use Primary's type params; widen constraint if specializations add types
		allTypeParams := make([]TypeParam, len(primary.TypeParams))
		copy(allTypeParams, primary.TypeParams)
		if len(specs) > 0 && len(allTypeParams) > 0 {
			widenConstraintForCombos(allTypeParams, allCombos)
		}

		groups = append(groups, DispatchGroup{
			GroupName:       groupName,
			Primary:         primary,
			Specializations: specs,
			AllCombos:       allCombos,
			AllTypeParams:   allTypeParams,
			Private:         primary.Private,
		})
	}

	// Sort groups for deterministic output
	slices.SortFunc(groups, func(a, b DispatchGroup) int {
		return strings.Compare(a.GroupName, b.GroupName)
	})

	return groups, nil
}

// deriveFuncGroupName extracts the dispatch group name from a function name.
// "BaseMatMul" → "MatMul", "baseSigmoid" → "Sigmoid"
func deriveFuncGroupName(name string) string {
	name = strings.TrimPrefix(name, "Base")
	name = strings.TrimPrefix(name, "base")
	// Ensure first letter is uppercase for consistent group names
	if len(name) > 0 {
		name = strings.ToUpper(name[:1]) + name[1:]
	}
	return name
}

// groupNames extracts sorted group names from a map for error messages.
func groupNames[V any](m map[string]*V) []string {
	names := make([]string, 0, len(m))
	for k := range m {
		names = append(names, k)
	}
	slices.Sort(names)
	return names
}

// validateSignatureCompatibility checks that a specialization function has a
// compatible signature with its primary function.
func validateSignatureCompatibility(primary, spec *ParsedFunc) error {
	if len(primary.Params) != len(spec.Params) {
		return fmt.Errorf("parameter count mismatch: primary has %d, specialization has %d",
			len(primary.Params), len(spec.Params))
	}
	if len(primary.Returns) != len(spec.Returns) {
		return fmt.Errorf("return count mismatch: primary has %d, specialization has %d",
			len(primary.Returns), len(spec.Returns))
	}
	return nil
}

// computeUnionCombos returns the union of type combinations from the primary
// function and all its specializations. Duplicate combos are removed.
func computeUnionCombos(primary *ParsedFunc, specs []*ParsedFunc) []TypeCombination {
	seen := make(map[string]bool)
	var result []TypeCombination

	addCombos := func(pf *ParsedFunc) {
		for _, combo := range getTypeCombinations(pf) {
			key := comboKey(combo)
			if !seen[key] {
				seen[key] = true
				result = append(result, combo)
			}
		}
	}

	addCombos(primary)
	for _, spec := range specs {
		addCombos(spec)
	}

	return result
}

// comboKey produces a stable string key for a TypeCombination for deduplication.
func comboKey(combo TypeCombination) string {
	if len(combo.Types) == 0 {
		return ""
	}
	keys := make([]string, 0, len(combo.Types))
	for k := range combo.Types {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	var parts []string
	for _, k := range keys {
		parts = append(parts, k+"="+combo.Types[k])
	}
	return strings.Join(parts, ",")
}

// combosEqual checks if two TypeCombinations have the same type mappings.
func combosEqual(a, b TypeCombination) bool {
	if len(a.Types) != len(b.Types) {
		return false
	}
	for k, v := range a.Types {
		if b.Types[k] != v {
			return false
		}
	}
	return true
}

// widenConstraintForCombos widens type parameter constraints if the combo set
// contains types not covered by the current constraint. For example, if the
// primary has hwy.FloatsNative (float32/float64) but a specialization adds
// hwy.Float16, the constraint is widened to hwy.Floats.
func widenConstraintForCombos(typeParams []TypeParam, combos []TypeCombination) {
	if len(typeParams) == 0 {
		return
	}

	// Collect all concrete types used by the first type param
	firstTP := typeParams[0].Name
	hasHalf := false
	hasNativeFloat := false
	for _, combo := range combos {
		t := combo.Types[firstTP]
		switch t {
		case "hwy.Float16", "hwy.BFloat16":
			hasHalf = true
		case "float32", "float64":
			hasNativeFloat = true
		}
	}

	// Widen if needed
	if hasHalf && hasNativeFloat {
		constraint := typeParams[0].Constraint
		if strings.Contains(constraint, "FloatsNative") || strings.Contains(constraint, "HalfFloats") {
			typeParams[0].Constraint = "hwy.Floats"
		}
	}
}

// selectSourceFunc returns the function whose body should be used to generate
// code for the given target and type combination. Returns nil if no function
// covers this combo on this target.
func selectSourceFunc(group *DispatchGroup, target Target, combo TypeCombination) (*ParsedFunc, error) {
	type candidate struct {
		pf    *ParsedFunc
		score int // higher = more specific
	}

	var candidates []candidate

	// Check specializations
	for _, spec := range group.Specializations {
		if !comboMatchesFunc(spec, combo) {
			continue
		}
		if !targetAllowed(spec, target) {
			continue
		}
		score := 0
		if len(spec.AllowedTargets) > 0 {
			score++ // target-restricted is more specific
		}
		candidates = append(candidates, candidate{pf: spec, score: score})
	}

	// Check primary (always a candidate if it covers this combo)
	if comboMatchesFunc(group.Primary, combo) {
		candidates = append(candidates, candidate{pf: group.Primary, score: -1})
	}

	if len(candidates) == 0 {
		// No function covers this combo on this target — skip silently.
		return nil, nil
	}

	// Sort by score descending
	slices.SortFunc(candidates, func(a, b candidate) int {
		return b.score - a.score
	})

	// Check for ties at the top (among specializations, not primary)
	if len(candidates) > 1 && candidates[0].score == candidates[1].score &&
		candidates[0].score > 0 {
		return nil, fmt.Errorf("ambiguous specialization for (%s, %v) in group %s: "+
			"both %s and %s match with equal specificity",
			target.Name, combo.Types, group.GroupName,
			candidates[0].pf.Name, candidates[1].pf.Name)
	}

	return candidates[0].pf, nil
}

// comboMatchesFunc returns true if the function covers the given type combination.
func comboMatchesFunc(pf *ParsedFunc, combo TypeCombination) bool {
	funcCombos := getTypeCombinations(pf)
	for _, fc := range funcCombos {
		if combosEqual(fc, combo) {
			return true
		}
	}
	return false
}

// targetAllowed returns true if the function is allowed to run on the given target.
// If AllowedTargets is empty, the function runs on all targets.
func targetAllowed(pf *ParsedFunc, target Target) bool {
	if len(pf.AllowedTargets) == 0 {
		return true
	}
	targetLower := strings.ToLower(target.Name)
	for _, allowed := range pf.AllowedTargets {
		if allowed == targetLower {
			return true
		}
	}
	return false
}

// synthPrimaryForDispatch creates a synthetic ParsedFunc from a DispatchGroup
// that has the union of all type combinations and widened constraints. This
// allows the existing emitter to generate correct dispatch without structural changes.
func synthPrimaryForDispatch(group *DispatchGroup) ParsedFunc {
	synth := *group.Primary // shallow copy
	synth.TypeCombinations = group.AllCombos
	synth.TypeParams = group.AllTypeParams
	return synth
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
	KeepCFiles     bool         // Keep intermediate C files in c/ subdirectory
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
			// SVE and NEON:asm targets don't get Go SIMD generation.
			// SVE has no Go SIMD OpMap; NEON:asm relies on the C assembly init
			// to override dispatch vars, with fallback as the base layer.
			// This avoids generating _neon.gen.go files that would be dead code.
		case TargetModeC:
			cOnlySpecs = append(cOnlySpecs, ts)
		}
	}

	// 2. Build dispatch groups from parsed functions.
	// Functions with //hwy:specializes join the referenced group;
	// all others form their own single-member groups.
	groups, err := buildDispatchGroups(result.Funcs)
	if err != nil {
		return fmt.Errorf("build dispatch groups: %w", err)
	}

	// Go SIMD path: transform + emit for each Go SIMD target
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

		for _, group := range groups {
			for _, combo := range group.AllCombos {
				// Select which function body to use for this (target, combo)
				sourcePF, err := selectSourceFunc(&group, target, combo)
				if err != nil {
					return fmt.Errorf("select source: %w", err)
				}
				if sourcePF == nil {
					continue // no coverage on this target
				}

				// Skip interface type params on non-Fallback targets
				if hasInterfaceTypeParams(sourcePF.TypeParams) && target.Name != "Fallback" {
					continue
				}

				elemType := comboPrimaryType(combo, group.AllTypeParams)

				if target.Name == "NEON" &&
					(elemType == "hwy.Float16" || elemType == "hwy.BFloat16") &&
					genericHalfPrecFuncs[sourcePF.Name] {
					transformOpts.SkipHalfPrecNEON = true
				} else {
					transformOpts.SkipHalfPrecNEON = false
				}

				// Set TypeMap for multi-type combos
				if isMultiTypeCombination(combo) {
					transformOpts.TypeMap = combo.Types
				} else {
					transformOpts.TypeMap = nil
				}

				transformResult := TransformWithOptions(sourcePF, target, elemType, transformOpts)

				// Name normalization: always use the Primary function's name for output.
				// This ensures the emitter's impl-name construction works regardless of
				// which source function body (primary or specialization) was used.
				outName := group.Primary.Name + target.Suffix()
				suffix := comboTypeSuffix(combo, group.AllTypeParams)
				if suffix != "" && suffix != "Float32" && len(group.AllTypeParams) > 0 {
					outName = outName + "_" + suffix
				}
				if group.Private {
					outName = makeUnexported(outName)
				}
				transformResult.FuncDecl.Name.Name = outName

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
		slices.Sort(hoistedKeys)
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
			target := ts.Target
			target.Mode = TargetModeAsm
			allTargets = append(allTargets, target)
		}
	}

	// 6. Emit the dispatcher file with ASM adapter info.
	// Build synthetic funcs that merge specialization groups: each group
	// becomes a single ParsedFunc with the union of all type combinations
	// and widened constraints. This lets the existing emitter work unchanged.
	synthFuncs := make([]ParsedFunc, 0, len(groups))
	for _, group := range groups {
		synthFuncs = append(synthFuncs, synthPrimaryForDispatch(&group))
	}
	if err := EmitDispatcher(synthFuncs, allTargets, g.PackageOut, g.OutputDir, g.DispatchPrefix, asmAdapters); err != nil {
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
