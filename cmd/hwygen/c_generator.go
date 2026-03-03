package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/ajroetker/go-highway/cmd/hwygen/ir"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

// runAsmMode generates C code for ASM targets, compiles with GOAT,
// and returns AsmAdapterInfo for unified dispatch wiring.
func (g *Generator) runAsmMode(result *ParseResult, asmSpecs []TargetSpec) ([]AsmAdapterInfo, error) {
	var targets []Target
	for _, ts := range asmSpecs {
		if ts.Target.Name == "Fallback" {
			continue
		}
		targets = append(targets, ts.Target)
	}
	if len(targets) == 0 {
		return nil, nil
	}
	return g.runCModeInternal(result, targets, true)
}

// runCOnlyMode generates C code for inspection only (no GOAT compilation).
func (g *Generator) runCOnlyMode(result *ParseResult, cSpecs []TargetSpec) error {
	var targets []Target
	for _, ts := range cSpecs {
		if ts.Target.Name == "Fallback" {
			continue
		}
		targets = append(targets, ts.Target)
	}
	if len(targets) == 0 {
		return nil
	}
	_, err := g.runCModeInternal(result, targets, false)
	return err
}

// runCMode generates C code for requested targets, compiles with GOAT,
// and generates Go wrapper functions. Used when all targets are C/ASM mode (legacy path).
func (g *Generator) runCMode(result *ParseResult) error {
	var targets []Target
	for _, ts := range g.TargetSpecs {
		if ts.Target.Name == "Fallback" {
			continue
		}
		targets = append(targets, ts.Target)
	}
	if len(targets) == 0 {
		return fmt.Errorf("no valid C generation targets specified")
	}
	isAsm := g.AsmMode()
	_, err := g.runCModeInternal(result, targets, isAsm)
	return err
}

// runCModeInternal is the shared implementation for C/ASM code generation.
// When asmMode is true, it compiles C to Go assembly via GOAT and returns adapter info.
func (g *Generator) runCModeInternal(result *ParseResult, targets []Target, asmMode bool) ([]AsmAdapterInfo, error) {
	if len(targets) == 0 {
		return nil, fmt.Errorf("no valid C generation targets specified")
	}

	// Build dispatch groups to handle //hwy:specializes and //hwy:targets.
	// Groups are built from ALL parsed functions; C-eligibility is checked
	// per (target, combo) when selecting the source function.
	groups, err := buildDispatchGroups(result.Funcs)
	if err != nil {
		return nil, fmt.Errorf("build dispatch groups: %w", err)
	}

	// Filter to groups that have at least one C-eligible member.
	var cGroups []DispatchGroup
	for _, group := range groups {
		hasCEligible := false
		allMembers := append([]*ParsedFunc{group.Primary}, group.Specializations...)
		for _, pf := range allMembers {
			if IsASTCEligible(pf) || IsCEligible(pf) {
				hasCEligible = true
				break
			}
		}
		if hasCEligible {
			cGroups = append(cGroups, group)
		}
	}

	if len(cGroups) == 0 {
		fmt.Println("No C-eligible functions found; all functions will use GoSimd path")
		return nil, nil
	}

	fmt.Printf("Found %d C-eligible dispatch groups:\n", len(cGroups))
	for _, group := range cGroups {
		synth := synthPrimaryForDispatch(&group)
		if IsASTCEligible(&synth) {
			fmt.Printf("  - %s (AST→C)\n", group.Primary.Name)
		} else {
			fmt.Printf("  - %s (Vec→Vec)\n", group.Primary.Name)
		}
		for _, spec := range group.Specializations {
			fmt.Printf("    specialization: %s", spec.Name)
			if len(spec.AllowedTargets) > 0 {
				fmt.Printf(" [targets: %s]", strings.Join(spec.AllowedTargets, ","))
			}
			fmt.Println()
		}
	}

	// Determine the target mode for selectSourceFunc matching
	cMode := TargetModeC
	if asmMode {
		cMode = TargetModeAsm
	}

	// Collect adapter info across all targets
	var allAdapters []AsmAdapterInfo

	// Process each target
	for _, target := range targets {
		fmt.Printf("\nGenerating C for target: %s\n", target.Name)

		// In asm mode, C/assembly files go to asm/ subdirectory
		cOutputDir := g.OutputDir
		if asmMode {
			cOutputDir = filepath.Join(g.OutputDir, "asm")
			if err := os.MkdirAll(cOutputDir, 0o755); err != nil {
				return nil, fmt.Errorf("create asm directory: %w", err)
			}
		}

		// Track generated C files for GOAT compilation
		var cFiles []string

		// Generate C code using dispatch groups. For each group, iterate
		// combos and select the source function per (target, combo). This
		// ensures specializations are used when they match.
		for gi := range cGroups {
			group := &cGroups[gi]
			for _, combo := range group.AllCombos {
				sourcePF, serr := selectSourceFunc(group, target, cMode, combo)
				if serr != nil {
					return nil, fmt.Errorf("select source for %s: %w", group.GroupName, serr)
				}
				if sourcePF == nil {
					continue // no coverage on this target for this combo
				}

				// Check C-eligibility of the selected source
				if !IsASTCEligible(sourcePF) && !IsCEligible(sourcePF) {
					continue
				}

				// For generic functions, use the combo's type parameter.
				// For non-generic functions (nil combo types), infer from
				// slice parameters via getCElemTypes to stay consistent
				// with wrapper generation naming.
				var elemType string
				if len(combo.Types) > 0 {
					elemType = comboPrimaryType(combo, group.AllTypeParams)
				} else {
					ets := getCElemTypes(sourcePF)
					elemType = ets[0]
				}
				profile := GetCProfile(target.Name, elemType)
				if profile == nil {
					continue
				}

				// Create a working copy with the primary's name for output naming
				pf := *sourcePF
				pf.Name = group.Primary.Name

				if IsASTCEligible(&pf) {
					// AST translator emits native arithmetic — skip types that
					// require promoted math unless they have native SIMD arithmetic.
					if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
						continue
					}
					emitter := NewCEmitter(g.PackageOut, elemType, target)
					emitter.profile = profile
					emitter.packageGlobals = result.PackageGlobals
					emitter.packageConsts = result.PackageConsts
					emitter.allFuncs = result.AllFuncs
					if isMultiTypeCombination(combo) {
						emitter.typeMap = combo.Types
					}
					cFile, cerr := emitter.EmitASTTranslatedC(&pf, cOutputDir)
					if cerr != nil {
						return nil, fmt.Errorf("emit AST C for %s (%s, %s): %w", pf.Name, elemType, target.Name, cerr)
					}
					cFiles = append(cFiles, cFile)
				} else if IsCEligible(&pf) {
					emitter := NewCEmitter(g.PackageOut, elemType, target)
					emitter.profile = profile
					if isMultiTypeCombination(combo) {
						emitter.typeMap = combo.Types
					}
					cFile, cerr := emitter.EmitC(&pf, cOutputDir)
					if cerr != nil {
						return nil, fmt.Errorf("emit C for %s (%s, %s): %w", pf.Name, elemType, target.Name, cerr)
					}
					cFiles = append(cFiles, cFile)
				}
			}
		}

		if len(cFiles) == 0 {
			continue
		}

		// If asm mode, compile C files with GOAT and generate wrappers
		if asmMode {
			fmt.Printf("Compiling %d C files with GOAT...\n", len(cFiles))
			var compiledFiles []string
			for _, cFile := range cFiles {
				profile := getCProfileForFile(cFile, target)
				if err := runGOAT(cFile, profile); err != nil {
					fmt.Printf("  WARNING: skipping %s: %v\n", filepath.Base(cFile), err)
					os.Remove(cFile)
					os.Remove(strings.TrimSuffix(cFile, ".c") + ".o")
					continue
				}
				compiledFiles = append(compiledFiles, cFile)
				fmt.Printf("  Compiled: %s\n", filepath.Base(cFile))
			}
			cFiles = compiledFiles

			// Optionally copy C files to c/ subdirectory for debugging
			if g.KeepCFiles {
				cDir := filepath.Join(g.OutputDir, "c")
				if err := os.MkdirAll(cDir, 0o755); err != nil {
					return nil, fmt.Errorf("create c directory: %w", err)
				}
				for _, cFile := range cFiles {
					dst := filepath.Join(cDir, filepath.Base(cFile))
					data, err := os.ReadFile(cFile)
					if err == nil {
						os.WriteFile(dst, data, 0o644)
					}
				}
				fmt.Printf("  Kept %d C files in %s\n", len(cFiles), cDir)
			}

			// Clean up C and object files (Go build doesn't like them)
			for _, cFile := range cFiles {
				os.Remove(cFile)
				os.Remove(strings.TrimSuffix(cFile, ".c") + ".o")
			}

			// Build synthetic primary funcs for wrapper/adapter emission.
			// These have the primary's name and the union of all combos,
			// matching what the Go SIMD path does for EmitDispatcher.
			synthFuncs := make([]ParsedFunc, 0, len(cGroups))
			for _, group := range cGroups {
				synth := synthPrimaryForDispatch(&group)
				if IsASTCEligible(&synth) || IsCEligible(&synth) {
					synthFuncs = append(synthFuncs, synth)
				}
			}

			// Separate struct-ptr and non-struct functions
			var structFuncs, nonStructFuncs []ParsedFunc
			for _, f := range synthFuncs {
				if hasStructPtrParams(&f) {
					structFuncs = append(structFuncs, f)
				} else {
					nonStructFuncs = append(nonStructFuncs, f)
				}
			}

			// Non-struct wrappers go to asm/ subdirectory
			if len(nonStructFuncs) > 0 {
				if err := g.emitCWrappers(nonStructFuncs, target, cOutputDir); err != nil {
					return nil, fmt.Errorf("emit asm wrappers for %s: %w", target.Name, err)
				}
			}

			// Struct-ptr functions need:
			// 1. Thin exported passthrough wrappers in asm/
			// 2. ASM adapter + wiring file in parent
			if len(structFuncs) > 0 {
				if err := g.emitStructAsmPassthrough(structFuncs, target, cOutputDir); err != nil {
					return nil, fmt.Errorf("emit asm passthrough for %s: %w", target.Name, err)
				}
				if err := g.emitZCDispatch(structFuncs, target); err != nil {
					return nil, fmt.Errorf("emit asm adapter for %s: %w", target.Name, err)
				}
			}

			// Non-struct AST-translated functions (slice+int params) need
			// ASM adapter + wiring. Scalar T params (e.g., base T in
			// DeltaDecode, value T in Find) are supported: the adapter
			// converts them to addressable locals for unsafe.Pointer.
			var nonStructASTFuncs []ParsedFunc
			for _, f := range nonStructFuncs {
				if IsASTCEligible(&f) {
					nonStructASTFuncs = append(nonStructASTFuncs, f)
				}
			}
			allSliceDispatchFuncs := nonStructASTFuncs
			if len(allSliceDispatchFuncs) > 0 {
				if err := g.emitSliceAsmPassthrough(allSliceDispatchFuncs, target, cOutputDir); err != nil {
					return nil, fmt.Errorf("emit slice asm passthrough for %s: %w", target.Name, err)
				}
				if err := g.emitZCDispatchForSlices(allSliceDispatchFuncs, target); err != nil {
					return nil, fmt.Errorf("emit asm adapter for slices %s: %w", target.Name, err)
				}
			}

			// Collect adapter info for all ASM-eligible functions
			for _, pf := range append(structFuncs, allSliceDispatchFuncs...) {
				for _, elemType := range getCElemTypes(&pf) {
					profile := GetCProfile(target.Name, elemType)
					if profile == nil {
						continue
					}
					if shouldSkipPromotedType(&pf, profile) {
						continue
					}
					allAdapters = append(allAdapters, AsmAdapterInfo{
						TargetName:  target.Name,
						Arch:        target.Arch(),
						DispatchVar: buildDispatchVarName(pf.Name, elemType, len(pf.TypeParams) > 0),
						AdapterFunc: buildAdapterFuncName(pf.Name, elemType),
					})
				}
			}
		} else {
			fmt.Printf("Generated %d C files (use -asm to compile to Go assembly)\n", len(cFiles))
		}
	}

	return allAdapters, nil
}

// runFusionCMode generates C code with IR-based fusion optimization.
// This path is used when -fusion flag is enabled.
func (g *Generator) runFusionCMode(result *ParseResult, sliceFuncs []ParsedFunc, targets []Target) error {
	// Find module root for cross-package resolution
	moduleRoot, moduleName, err := FindModuleRoot(g.OutputDir)
	if err != nil {
		fmt.Printf("Warning: could not find module root: %v\n", err)
		// Fall back to direct translation without cross-package resolution
		moduleRoot = g.OutputDir
		moduleName = "go-highway"
	}

	// Create function registry for cross-package resolution
	registry := NewFunctionRegistry(moduleRoot, moduleName)

	for _, target := range targets {
		fmt.Printf("\nGenerating C for target: %s (with fusion)\n", target.Name)

		var cFiles []string

		for _, pf := range sliceFuncs {
			elemTypes := getCElemTypes(&pf)
			for _, elemType := range elemTypes {
				profile := GetCProfile(target.Name, elemType)
				if profile == nil {
					continue
				}

				// Skip types that require promoted math and don't have native arithmetic
				if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
					continue
				}

				// Build IR from the parsed function
				builder := ir.NewBuilder(
					ir.WithImports(result.Imports),
					ir.WithElemType(elemType),
					ir.WithResolver(registry),
				)

				irPF := &ir.ParsedFunc{
					Name: pf.Name,
					Body: pf.Body,
				}
				for _, tp := range pf.TypeParams {
					irPF.TypeParams = append(irPF.TypeParams, ir.TypeParamInput{
						Name:       tp.Name,
						Constraint: tp.Constraint,
					})
				}
				for _, p := range pf.Params {
					irPF.Params = append(irPF.Params, ir.ParamInput{
						Name: p.Name,
						Type: p.Type,
					})
				}
				for _, r := range pf.Returns {
					irPF.Returns = append(irPF.Returns, ir.ParamInput{
						Name: r.Name,
						Type: r.Type,
					})
				}

				irFunc, err := builder.Build(irPF)
				if err != nil {
					return fmt.Errorf("build IR for %s: %w", pf.Name, err)
				}

				// Run data flow analysis
				ir.Analyze(irFunc)

				// Apply fusion rules
				ir.ApplyFusionRules(irFunc)

				// Print fusion statistics if verbose
				if g.Verbose {
					stats := ir.ComputeFusionStats(irFunc)
					fmt.Printf("  %s: %d→%d passes, %d allocs eliminated, %d fusion groups\n",
						pf.Name, stats.OriginalPasses, stats.FusedPasses,
						stats.EliminatedAllocs, stats.FusionGroups)
				}

				// Create profile adapter for IR emitter
				profileAdapter := newCProfileAdapter(profile)

				// Emit C code from IR
				emitter := ir.NewEmitter(profileAdapter)
				cCode := emitter.EmitFunction(irFunc)

				// Write C file
				cFileName := fmt.Sprintf("%s_%s_%s.c",
					strings.ToLower(pf.Name),
					elemType,
					strings.ToLower(target.Name))
				cFilePath := filepath.Join(g.OutputDir, cFileName)

				if err := os.WriteFile(cFilePath, []byte(cCode), 0o644); err != nil {
					return fmt.Errorf("write C file: %w", err)
				}

				cFiles = append(cFiles, cFilePath)
				fmt.Printf("  Generated: %s\n", cFileName)
			}
		}

		if len(cFiles) == 0 {
			continue
		}

		if g.AsmMode() {
			fmt.Printf("Compiling %d C files with GOAT...\n", len(cFiles))
			var compiledFiles []string
			for _, cFile := range cFiles {
				profile := getCProfileForFile(cFile, target)
				if err := runGOAT(cFile, profile); err != nil {
					fmt.Printf("  WARNING: skipping %s: %v\n", filepath.Base(cFile), err)
					os.Remove(cFile)
					os.Remove(strings.TrimSuffix(cFile, ".c") + ".o")
					continue
				}
				compiledFiles = append(compiledFiles, cFile)
				fmt.Printf("  Compiled: %s\n", filepath.Base(cFile))
			}
			cFiles = compiledFiles

			// Clean up C and object files
			for _, cFile := range cFiles {
				os.Remove(cFile)
				os.Remove(strings.TrimSuffix(cFile, ".c") + ".o")
			}

			// Generate wrappers (fusion mode outputs to same dir)
			if err := g.emitCWrappers(sliceFuncs, target, g.OutputDir); err != nil {
				return fmt.Errorf("emit wrappers for %s: %w", target.Name, err)
			}
		} else {
			fmt.Printf("Generated %d C files with fusion (use -asm to compile)\n", len(cFiles))
		}
	}

	return nil
}

// cProfileAdapter adapts CIntrinsicProfile to ir.CProfile interface.
type cProfileAdapter struct {
	p *CIntrinsicProfile
}

func newCProfileAdapter(p *CIntrinsicProfile) *cProfileAdapter {
	return &cProfileAdapter{p: p}
}

func (a *cProfileAdapter) GetTier() string {
	for _, t := range a.p.Tiers {
		if !t.IsScalar {
			return t.Name
		}
	}
	return "q"
}

func (a *cProfileAdapter) GetLanes() int {
	for _, t := range a.p.Tiers {
		if !t.IsScalar {
			return t.Lanes
		}
	}
	return 4
}

func (a *cProfileAdapter) GetVecType(tier string) string {
	if vt, ok := a.p.VecTypes[tier]; ok {
		return vt
	}
	return "float32x4_t"
}

func (a *cProfileAdapter) GetScalarType() string {
	if a.p.ScalarArithType != "" {
		return a.p.ScalarArithType
	}
	return a.p.CType
}

func (a *cProfileAdapter) GetLoadFn(tier string) string {
	if fn, ok := a.p.LoadFn[tier]; ok {
		return fn
	}
	return "vld1q_f32"
}

func (a *cProfileAdapter) GetStoreFn(tier string) string {
	if fn, ok := a.p.StoreFn[tier]; ok {
		return fn
	}
	return "vst1q_f32"
}

func (a *cProfileAdapter) GetIntrinsic(op, tier string) string {
	var fnMap map[string]string
	switch op {
	case "Add":
		fnMap = a.p.AddFn
	case "Sub":
		fnMap = a.p.SubFn
	case "Mul":
		fnMap = a.p.MulFn
	case "Div":
		fnMap = a.p.DivFn
	case "MulAdd":
		fnMap = a.p.FmaFn
	case "Neg":
		fnMap = a.p.NegFn
	case "Abs":
		fnMap = a.p.AbsFn
	case "Sqrt":
		fnMap = a.p.SqrtFn
	case "Min":
		fnMap = a.p.MinFn
	case "Max":
		fnMap = a.p.MaxFn
	case "ReduceSum":
		fnMap = a.p.ReduceSumFn
	case "ReduceMin":
		fnMap = a.p.ReduceMinFn
	case "ReduceMax":
		fnMap = a.p.ReduceMaxFn
	case "Set", "Const":
		fnMap = a.p.DupFn
	default:
		return ""
	}
	if fn, ok := fnMap[tier]; ok {
		return fn
	}
	return ""
}

func (a *cProfileAdapter) GetFmaArgOrder() string {
	if a.p.FmaArgOrder != "" {
		return a.p.FmaArgOrder
	}
	return "acc_last"
}

func (a *cProfileAdapter) GetInlineHelpers() string {
	return strings.Join(a.p.InlineHelpers, "\n")
}

func (a *cProfileAdapter) RequiresCast() bool {
	return a.p.CastExpr != ""
}

func (a *cProfileAdapter) GetCastExpr() string {
	return a.p.CastExpr
}

func (a *cProfileAdapter) GetZeroInit(tier string) string {
	// Create a zero vector
	if dupFn, ok := a.p.DupFn[tier]; ok {
		return dupFn + "(0)"
	}
	return "{0}"
}

// getCElemTypes returns the concrete element types for C code generation.
// This includes f16/bf16 types when the constraint allows them.
// For non-generic functions, it infers the element type from slice parameters,
// preferring wider/non-byte types that represent the actual SIMD vector element
// type (byte/uint8 slices are typically scalar staging buffers, not the
// vectorized type).
func getCElemTypes(pf *ParsedFunc) []string {
	// Check for explicit //hwy:elemtype directive override.
	if pf.ElemTypeOverride != "" {
		return []string{pf.ElemTypeOverride}
	}
	if len(pf.TypeParams) > 0 {
		return GetConcreteTypes(pf.TypeParams[0].Constraint)
	}
	// For non-generic functions, collect all slice element types.
	// Prefer wider types over byte/uint8 since byte slices are typically
	// packed output buffers, not the SIMD element type.
	var best string
	for _, p := range pf.Params {
		after, ok := strings.CutPrefix(p.Type, "[]")
		if !ok {
			continue
		}
		var candidate string
		switch after {
		case "int32":
			candidate = "int32"
		case "int64":
			candidate = "int64"
		case "uint64":
			candidate = "uint64"
		case "uint32":
			candidate = "uint32"
		case "uint8", "byte":
			candidate = "uint8"
		case "float64":
			candidate = "float64"
		case "float32":
			candidate = "float32"
		}
		if candidate == "" {
			continue
		}
		if best == "" {
			best = candidate
		} else if best == "uint8" && candidate != "uint8" {
			// Prefer non-byte types: byte slices are typically packed
			// I/O buffers, not the SIMD element type.
			best = candidate
		}
	}
	// If parameter inference yielded only uint8 (packed data), check
	// whether hwy.* calls in the function body use explicit type arguments
	// that indicate a different SIMD element type. This matches C++ Highway's
	// design where the element type is always explicit via the tag parameter.
	if (best == "" || best == "uint8") && pf.Body != nil {
		if bodyType := inferElemTypeFromBody(pf.Body); bodyType != "" {
			best = bodyType
		}
	}
	if best != "" {
		return []string{best}
	}
	return []string{"float32"}
}

// inferElemTypeFromBody walks the function body AST looking for hwy.* calls
// with explicit type arguments (e.g., hwy.Zero[float32](), hwy.NumLanes[float32]()).
// Returns the element type if found, or empty string if not.
//
// This enables correct profile selection for functions that take []uint8 packed
// data but perform float32 SIMD operations internally (e.g., quantized dot
// products), matching C++ Highway's explicit tag-based type dispatch.
func inferElemTypeFromBody(body *ast.BlockStmt) string {
	var found string
	ast.Inspect(body, func(n ast.Node) bool {
		if found != "" {
			return false // already found, stop walking
		}
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		// Look for hwy.Func[T](...) pattern: ast.IndexExpr wrapping ast.SelectorExpr.
		idx, ok := call.Fun.(*ast.IndexExpr)
		if !ok {
			return true
		}
		sel, ok := idx.X.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		pkg, ok := sel.X.(*ast.Ident)
		if !ok || pkg.Name != "hwy" {
			return true
		}
		// Only consider functions whose type argument determines the SIMD
		// element type: Zero[T], NumLanes[T], Set[T], Const[T].
		switch sel.Sel.Name {
		case "Zero", "NumLanes", "Set", "Const":
			// Extract the type argument T.
			if typeIdent, ok := idx.Index.(*ast.Ident); ok {
				found = typeIdent.Name
			}
		}
		return true
	})
	return found
}

// IsSliceFunction checks if a function operates on slices (not Vec) and is
// vectorizable. These are composite functions like GELU that process entire
// arrays using hwy SIMD operations. Pure scalar functions (like Interleave,
// Deinterleave) that operate on slices but have no hwy calls are excluded —
// the composite C template generates incorrect code for them.
func IsSliceFunction(pf *ParsedFunc) bool {
	hasSliceParam := false
	for _, param := range pf.Params {
		if strings.HasPrefix(param.Type, "[]") {
			hasSliceParam = true
			break
		}
	}
	if !hasSliceParam || hasVecInSignature(*pf) {
		return false
	}
	// Must have hwy operations to be vectorizable via the composite C path.
	for _, call := range pf.HwyCalls {
		if call.Package == "hwy" {
			return true
		}
	}
	// Also eligible if it's a recognized math composite (Exp, Sin, GELU, etc.)
	return mathOpFromFuncName(pf.Name) != ""
}

// hasScalarTypeParams returns true if the function has any scalar parameters
// of type T (e.g., the alpha parameter in LeakyReLU/ELU). Functions with
// scalar T params cannot be wired via dispatch because the C function signature
// doesn't include those parameters (they use hardcoded constants instead).
func hasScalarTypeParams(pf *ParsedFunc) bool {
	for _, p := range pf.Params {
		if p.Type == "T" {
			return true
		}
	}
	return false
}

// shouldSkipPromotedType returns true if the given function/type combo should
// be skipped because the C profile requires promoted math and the function's
// C code doesn't handle promotion internally. AST-translated functions
// handle promotion via the split-promote-compute-combine pattern in
// their generated C code when NativeArithmetic is true.
func shouldSkipPromotedType(pf *ParsedFunc, profile *CIntrinsicProfile) bool {
	if profile.MathStrategy != "promoted" || profile.NativeArithmetic {
		return false
	}
	return true
}

// getCProfileForFile determines the CIntrinsicProfile for a generated C file
// based on its filename convention.
func getCProfileForFile(cFile string, target Target) *CIntrinsicProfile {
	base := filepath.Base(cFile)
	for _, et := range []string{
		"float16", "hwy.Float16", "bfloat16", "hwy.BFloat16",
		"float32", "float64",
		"int32", "int64",
		"uint64", "uint32", "uint8",
	} {
		suffix := cTypeSuffix(et)
		if strings.Contains(base, "_"+suffix+"_") {
			p := GetCProfile(target.Name, et)
			if p != nil {
				return p
			}
		}
	}
	// Default to f32
	return GetCProfile(target.Name, "float32")
}

// runGOAT invokes the GOAT tool to compile a C file to Go assembly.
// It uses `go tool github.com/gorse-io/goat` which requires goat to be
// declared as a tool dependency in go.mod (via `go get -tool`).
func runGOAT(cFile string, profile *CIntrinsicProfile) error {
	// Use the Go binary from GOROOT (same toolchain that built hwygen)
	goBin := filepath.Join(runtime.GOROOT(), "bin", "go")

	// Get absolute path for C file since we run from module root
	absCFile, err := filepath.Abs(cFile)
	if err != nil {
		return fmt.Errorf("abs path: %w", err)
	}

	// Find module root (directory containing go.mod with tool directive)
	modRoot, err := findModuleRoot()
	if err != nil {
		return fmt.Errorf("find module root: %w", err)
	}

	// Build GOAT args from profile.
	// -t sets GOAT's target arch (selects parser/prologue with correct type definitions).
	// Without -t, GOAT defaults to runtime.GOARCH which is wrong for cross-compilation.
	// -e flags are passed through to clang for compilation.
	goatTarget := "arm64"
	if profile != nil && profile.GoatTarget != "" {
		goatTarget = profile.GoatTarget
	}
	args := []string{"tool", "github.com/gorse-io/goat", absCFile,
		"-O3",
		"-t", goatTarget,
		"-o", filepath.Dir(absCFile),
	}

	if profile != nil {
		// Don't pass --target via -e: GOAT already derives the correct
		// target triple (e.g. arm64-apple-darwin) from the -t flag and
		// the host OS. Overriding it with a bare "arm64" causes clang
		// to emit ELF-style assembly on macOS, which breaks GOAT's
		// Mach-O constant pool rewriting.
		for _, flag := range profile.GoatExtraFlags {
			args = append(args, "-e="+flag)
		}
	} else {
		// Fallback to arm64 NEON
		args = append(args,
			"-e=-march=armv8-a+simd+fp",
		)
	}
	args = append(args, "-e=-fno-builtin-memset")

	cmd := exec.Command(goBin, args...)
	cmd.Dir = modRoot
	cmd.Env = os.Environ()

	// For cross-compilation, ensure the correct architecture's objdump is on
	// PATH. GOAT invokes bare "objdump" to disassemble the compiled object,
	// which fails when the host objdump doesn't support the target arch.
	if goatTarget != runtime.GOARCH {
		if crossObjdump := crossObjdumpPath(goatTarget); crossObjdump != "" {
			// Prepend a temp directory containing a symlink so "objdump" resolves
			// to the cross-architecture version.
			binDir, err := os.MkdirTemp("", "goat-cross-bin-*")
			if err == nil {
				defer os.RemoveAll(binDir)
				link := filepath.Join(binDir, "objdump")
				if os.Symlink(crossObjdump, link) == nil {
					cmd.Env = append(cmd.Env, "PATH="+binDir+":"+os.Getenv("PATH"))
				}
			}
		}
	}

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%w: %s", err, string(output))
	}

	// Rename GoAT-generated .go file to .gen.go
	// GoAT generates filename_arch.go from filename.c
	cBase := strings.TrimSuffix(filepath.Base(cFile), ".c")
	cDir := filepath.Dir(absCFile)
	goatGoFile := filepath.Join(cDir, cBase+".go")
	genGoFile := filepath.Join(cDir, cBase+".gen.go")
	if _, err := os.Stat(goatGoFile); err == nil {
		if err := os.Rename(goatGoFile, genGoFile); err != nil {
			return fmt.Errorf("rename %s to .gen.go: %w", goatGoFile, err)
		}
	}

	// Fix reserved register names in generated assembly.
	// Go's plan9 assembler reserves "g" for the goroutine pointer, so
	// parameter names like "g+8(FP)" are misinterpreted as register+offset.
	// Rename conflicting parameter names in both .gen.go and .s files.
	sFile := filepath.Join(cDir, cBase+".s")
	if err := fixReservedAsmNames(genGoFile); err != nil {
		return fmt.Errorf("fix reserved names in %s: %w", genGoFile, err)
	}
	if err := fixReservedAsmNames(sFile); err != nil {
		return fmt.Errorf("fix reserved names in %s: %w", sFile, err)
	}

	return nil
}

// crossObjdumpPath returns the path to a cross-architecture objdump binary,
// or "" if none is found. On Linux, cross-compilation toolchains install
// architecture-prefixed binutils (e.g., aarch64-linux-gnu-objdump).
func crossObjdumpPath(targetArch string) string {
	prefixes := map[string]string{
		"arm64":   "aarch64-linux-gnu-objdump",
		"riscv64": "riscv64-linux-gnu-objdump",
	}
	name, ok := prefixes[targetArch]
	if !ok {
		return ""
	}
	path, err := exec.LookPath(name)
	if err != nil {
		return ""
	}
	return path
}

// reservedAsmNames maps Go plan9 assembler reserved names to safe replacements.
// In ARM64 plan9 assembly, "g" is the goroutine register and cannot be used
// as a parameter name in frame references like "g+8(FP)".
var reservedAsmNames = map[string]string{
	"g": "gv",
}

// fixReservedAsmNames renames reserved parameter names in GoAT-generated files.
// It replaces patterns like " g+" (parameter references) and "g " (in func decls)
// with safe alternatives. Replacements are word-boundary-aware to avoid
// corrupting longer names that contain the reserved name as a substring
// (e.g., "img" should not be rewritten to "imgv" when renaming "g" → "gv").
func fixReservedAsmNames(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	content := string(data)
	modified := false
	for reserved, replacement := range reservedAsmNames {
		// Replace parameter references in .s files: " g+8(FP)" → " gv+8(FP)"
		// Use " g+" (space prefix) to avoid matching substrings like "img+".
		// Assembly parameter names always follow a space after the instruction.
		old := " " + reserved + "+"
		new := " " + replacement + "+"
		if strings.Contains(content, old) {
			content = strings.ReplaceAll(content, old, new)
			modified = true
		}
		// Replace parameter names in .go func declarations.
		// Handle first parameter "(g," or "(g ", and middle/later ", g " or ", g,".
		for _, pair := range [][2]string{
			{"(" + reserved + ",", "(" + replacement + ","},
			{"(" + reserved + " ", "(" + replacement + " "},
			{", " + reserved + " ", ", " + replacement + " "},
			{", " + reserved + ",", ", " + replacement + ","},
		} {
			if strings.Contains(content, pair[0]) {
				content = strings.ReplaceAll(content, pair[0], pair[1])
				modified = true
			}
		}
	}

	if modified {
		return os.WriteFile(filename, []byte(content), 0o644)
	}
	return nil
}

// findModuleRoot walks up from the current directory to find go.mod.
func findModuleRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("go.mod not found")
		}
		dir = parent
	}
}

// goatPackageName derives the Go package name from a directory path,
// matching GOAT's behavior of using the directory basename.
func goatPackageName(dir string) string {
	// Handle "." by resolving to absolute path first
	if dir == "." {
		absDir, err := filepath.Abs(dir)
		if err == nil {
			dir = absDir
		}
	}
	base := filepath.Base(dir)
	// Replace hyphens with underscores (GOAT behavior)
	return strings.ReplaceAll(base, "-", "_")
}

// emitCWrappers generates a Go file with wrapper functions that call the assembly.
// outputDir specifies where to write the wrapper file (e.g., asm/ subdirectory).
func (g *Generator) emitCWrappers(funcs []ParsedFunc, target Target, outputDir string) error {
	var buf bytes.Buffer

	// GOAT derives package name from output directory, so we must match that
	pkgName := goatPackageName(outputDir)

	// Determine build tag and file suffix based on target
	buildTag := target.BuildTag
	if buildTag == "" {
		buildTag = "!noasm"
	} else {
		buildTag = "!noasm && " + buildTag
	}
	archSuffix := target.Arch()
	targetSuffix := strings.ToLower(target.Name)

	// Check if we need the hwy import (for f16/bf16 AST-translated functions
	// that actually have native math — promoted math types are skipped)
	needsHwy := false
	for _, pf := range funcs {
		if IsASTCEligible(&pf) {
			for _, et := range getCElemTypes(&pf) {
				if isHalfPrecisionType(et) {
					profile := GetCProfile(target.Name, et)
					if profile != nil && !shouldSkipPromotedType(&pf, profile) {
						needsHwy = true
						break
					}
				}
			}
		}
		if needsHwy {
			break
		}
	}

	// File header
	fmt.Fprintf(&buf, "//go:build %s\n", buildTag)
	fmt.Fprintf(&buf, "// Code generated by hwygen -c. DO NOT EDIT.\n\n")
	fmt.Fprintf(&buf, "package %s\n\n", pkgName)
	if needsHwy {
		fmt.Fprintf(&buf, "import (\n")
		fmt.Fprintf(&buf, "\t\"unsafe\"\n\n")
		fmt.Fprintf(&buf, "\t\"github.com/ajroetker/go-highway/hwy\"\n")
		fmt.Fprintf(&buf, ")\n\n")
	} else {
		fmt.Fprintf(&buf, "import \"unsafe\"\n\n")
	}

	// Emit wrapper functions
	fmt.Fprintf(&buf, "// Public wrapper functions\n")
	for _, pf := range funcs {
		elemTypes := getCElemTypes(&pf)
		for _, elemType := range elemTypes {
			profile := GetCProfile(target.Name, elemType)
			if profile == nil {
				continue
			}
			if IsASTCEligible(&pf) {
				// Skip wrapper for types that didn't get C/asm generated
				if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
					continue
				}
				// Check if function has struct pointer params - these need special
				// wrapper generation to convert Go struct layout to C struct layout
				if hasStructPtrParams(&pf) {
					emitStructPtrAsmWrapper(&buf, &pf, elemType, target)
				} else {
					emitASTCWrapperFunc(&buf, &pf, elemType, targetSuffix)
				}
			} else {
				emitCWrapperFunc(&buf, &pf, elemType, targetSuffix)
			}
		}
	}

	// Format and write
	wrapperDispPrefix := g.DispatchPrefix
	if wrapperDispPrefix == "" {
		wrapperDispPrefix = "dispatch"
	}
	filename := filepath.Join(outputDir, fmt.Sprintf("c_wrappers_%s_%s_%s.gen.go", wrapperDispPrefix, targetSuffix, archSuffix))
	if err := os.WriteFile(filename, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("write wrappers: %w", err)
	}

	fmt.Printf("Generated: %s\n", filename)
	return nil
}

// emitStructAsmPassthrough generates thin exported wrapper functions in the asm/
// subdirectory that re-export the unexported GoAT assembly functions. These take
// unsafe.Pointer params to avoid importing the parent package's types.
func (g *Generator) emitStructAsmPassthrough(funcs []ParsedFunc, target Target, asmDir string) error {
	var buf bytes.Buffer

	pkgName := goatPackageName(asmDir)

	buildTag := target.BuildTag
	if buildTag == "" {
		buildTag = "!noasm"
	} else {
		buildTag = "!noasm && " + buildTag
	}
	archSuffix := target.Arch()
	targetSuffix := strings.ToLower(target.Name)

	// File header
	fmt.Fprintf(&buf, "//go:build %s\n", buildTag)
	fmt.Fprintf(&buf, "// Code generated by hwygen -c. DO NOT EDIT.\n\n")
	fmt.Fprintf(&buf, "package %s\n\n", pkgName)
	fmt.Fprintf(&buf, "import \"unsafe\"\n\n")

	for _, pf := range funcs {
		elemTypes := getCElemTypes(&pf)
		for _, elemType := range elemTypes {
			profile := GetCProfile(target.Name, elemType)
			if profile == nil {
				continue
			}
			if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
				continue
			}

			// Exported name: ForwardICT_F32
			exportedName := structAsmExportedName(pf.Name, elemType)
			// Assembly name: forwardict_c_f32_neon
			asmName := cAsmFuncName(pf.Name, elemType, targetSuffix)

			// Build param list: unsafe.Pointer per parameter.
			// Struct pointer params keep their Go name; scalar params
			// (type-parameter T, int, float) get a "p" prefix to match
			// the GOAT calling convention (passed as pointers).
			typeParamNames := make(map[string]bool, len(pf.TypeParams))
			for _, tp := range pf.TypeParams {
				typeParamNames[tp.Name] = true
			}
			var paramNames []string
			for _, p := range pf.Params {
				if isGenericStructPtr(p.Type) {
					paramNames = append(paramNames, p.Name)
				} else if typeParamNames[p.Type] || isGoScalarIntType(p.Type) || isGoScalarFloatType(p.Type) {
					paramNames = append(paramNames, "p"+p.Name)
				}
			}

			fmt.Fprintf(&buf, "// %s calls the %s SIMD assembly implementation.\n",
				exportedName, strings.ToUpper(target.Name))
			fmt.Fprintf(&buf, "func %s(%s unsafe.Pointer) {\n",
				exportedName, strings.Join(paramNames, ", "))
			fmt.Fprintf(&buf, "\t%s(%s)\n", asmName, strings.Join(paramNames, ", "))
			fmt.Fprintf(&buf, "}\n\n")
		}
	}

	dispPrefix := g.DispatchPrefix
	if dispPrefix == "" {
		dispPrefix = "dispatch"
	}
	filename := filepath.Join(asmDir, fmt.Sprintf("c_struct_wrappers_%s_%s_%s.gen.go", dispPrefix, targetSuffix, archSuffix))
	if err := os.WriteFile(filename, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("write struct asm passthrough: %w", err)
	}
	fmt.Printf("Generated: %s\n", filename)
	return nil
}

// isSVETarget returns true if the target is an SVE target (SVE_DARWIN or SVE_LINUX).
func isSVETarget(target Target) bool {
	return target.Name == "SVE_DARWIN" || target.Name == "SVE_LINUX"
}

func isNeonTarget(target Target) bool {
	return target.Name == "NEON"
}

// isSVEStreamingTarget returns true for SVE targets that require SME streaming
// mode (smstart/smstop). On Darwin, SVE instructions only work in streaming mode,
// so per-function dispatch is impractical due to ~50ns transition overhead.
// These targets should NOT override scalar dispatch vars; instead, SVE assembly
// is exposed for use from batch wrappers that enter streaming mode once.
func isSVEStreamingTarget(target Target) bool {
	return target.Name == "SVE_DARWIN"
}

// sveRuntimeGuard returns the runtime detection function call for SVE targets.
// SVE_DARWIN uses hwy.HasSME(), SVE_LINUX uses hwy.HasSVE().
func sveRuntimeGuard(target Target) string {
	switch target.Name {
	case "SVE_DARWIN":
		return "hwy.HasSME()"
	case "SVE_LINUX":
		return "hwy.HasSVE()"
	default:
		return ""
	}
}

// neonNoSimdGuard returns the NoSimdEnv guard for NEON targets.
// NEON asm is the baseline on ARM64 — SME dispatch files (which must use
// z-prefixed names to sort after z_c_slices_*) override when HasSME() is true.
// The NoSimdEnv check allows HWY_NO_SIMD=1 to disable asm for testing.
func neonNoSimdGuard(target Target) string {
	if isNeonTarget(target) {
		return "hwy.NoSimdEnv()"
	}
	return ""
}

// elemTypeFeatureGuard returns the runtime feature guard expression for ARM64
// element types that require optional CPU extensions. Returns empty string for
// universally-supported types (f32, f64, integers).
// Float16 requires FEAT_FP16 (ARMv8.2-A+), BFloat16 requires FEAT_BF16 (ARMv8.6-A+).
func elemTypeFeatureGuard(elemType string) string {
	if isFloat16Type(elemType) {
		return "hwy.HasARMFP16()"
	}
	if isBFloat16Type(elemType) {
		return "hwy.HasARMBF16()"
	}
	return ""
}

// dispatchAssignment pairs a dispatch variable with its adapter function name.
type dispatchAssignment struct {
	dispatchVar string
	adapterName string
}

// emitGuardedDispatchAssignments writes dispatch variable assignments to buf,
// grouping half-precision types behind runtime feature guards. Universally
// supported types (f32, f64, integers) are emitted directly; Float16 assignments
// are wrapped in `if hwy.HasARMFP16() { ... }` and BFloat16 in
// `if hwy.HasARMBF16() { ... }` to prevent SIGILL on hardware lacking those
// extensions.
func emitGuardedDispatchAssignments(buf *bytes.Buffer, funcs []ParsedFunc, target Target) {
	// Collect assignments grouped by feature guard.
	guardedAssignments := make(map[string][]dispatchAssignment)
	for _, pf := range funcs {
		for _, elemType := range getCElemTypes(&pf) {
			profile := GetCProfile(target.Name, elemType)
			if profile == nil {
				continue
			}
			if shouldSkipPromotedType(&pf, profile) {
				continue
			}
			dv := buildDispatchVarName(pf.Name, elemType, len(pf.TypeParams) > 0)
			an := buildAdapterFuncName(pf.Name, elemType)
			guard := elemTypeFeatureGuard(elemType)
			guardedAssignments[guard] = append(guardedAssignments[guard], dispatchAssignment{dv, an})
		}
	}

	// Emit unguarded assignments first (f32, f64, integers, etc.).
	for _, a := range guardedAssignments[""] {
		fmt.Fprintf(buf, "\t%s = %s\n", a.dispatchVar, a.adapterName)
	}

	// Emit guarded blocks in a stable order.
	for _, guard := range []string{"hwy.HasARMFP16()", "hwy.HasARMBF16()"} {
		assignments := guardedAssignments[guard]
		if len(assignments) == 0 {
			continue
		}
		fmt.Fprintf(buf, "\tif %s {\n", guard)
		for _, a := range assignments {
			fmt.Fprintf(buf, "\t\t%s = %s\n", a.dispatchVar, a.adapterName)
		}
		fmt.Fprintf(buf, "\t}\n")
	}
}

// emitZCDispatch generates a z_c_*.gen.go file in the parent package that
// overrides dispatch variables with assembly implementations. It creates adapter
// functions that convert generic struct pointers (e.g., *Image[T]) to C-compatible
// struct layouts and call the exported asm/ wrapper functions.
func (g *Generator) emitZCDispatch(funcs []ParsedFunc, target Target) error {
	var buf bytes.Buffer

	buildTag := target.BuildTag
	if buildTag == "" {
		buildTag = "!noasm"
	} else {
		buildTag = "!noasm && " + buildTag
	}
	archSuffix := target.Arch()
	targetSuffix := strings.ToLower(target.Name)

	// Determine the asm subpackage import path
	asmImport, err := g.resolveAsmImportPath()
	if err != nil {
		return fmt.Errorf("resolve asm import path: %w", err)
	}

	// Determine if hwy import is needed:
	// - NEON targets need hwy for the NoSimdEnv() guard
	// - Non-streaming SVE targets (Linux) need hwy for HasSVE() guard
	// - Half-precision types need hwy for HasARMFP16()/HasARMBF16() guards
	needsHwy := isNeonTarget(target) || (isSVETarget(target) && !isSVEStreamingTarget(target))
	if !needsHwy {
		// Check if we need the hwy import for half-precision types
		for _, pf := range funcs {
			for _, et := range getCElemTypes(&pf) {
				if isHalfPrecisionType(et) {
					profile := GetCProfile(target.Name, et)
					if profile != nil && !shouldSkipPromotedType(&pf, profile) {
						needsHwy = true
						break
					}
				}
			}
			if needsHwy {
				break
			}
		}
	}

	// File header
	fmt.Fprintf(&buf, "//go:build %s\n", buildTag)
	fmt.Fprintf(&buf, "// Code generated by hwygen -c. DO NOT EDIT.\n\n")
	fmt.Fprintf(&buf, "package %s\n\n", g.PackageOut)
	fmt.Fprintf(&buf, "import (\n")
	fmt.Fprintf(&buf, "\t\"unsafe\"\n\n")
	if needsHwy {
		fmt.Fprintf(&buf, "\t\"github.com/ajroetker/go-highway/hwy\"\n")
	}
	fmt.Fprintf(&buf, "\t\"%s\"\n", asmImport)
	fmt.Fprintf(&buf, ")\n\n")

	// init() to override dispatch variables.
	// SVE streaming targets (Darwin) do NOT override scalar dispatch vars
	// because per-function smstart/smstop overhead (~50ns) makes SVE slower
	// than NEON for scalar calls. The adapter functions are still generated
	// for use from batch wrappers (e.g., sme_wrappers.go).
	// Non-streaming SVE targets (Linux) override dispatch normally.
	if !isSVEStreamingTarget(target) {
		capPrefix := cases.Title(language.English).String(g.DispatchPrefix)
		capTarget := cases.Title(language.English).String(strings.ToLower(target.Name))
		initFn := "init" + capPrefix + capTarget + "CAsm"
		fmt.Fprintf(&buf, "func init() {\n\t%s()\n}\n\n", initFn)
		fmt.Fprintf(&buf, "func %s() {\n", initFn)
		// Emit runtime guards: NoSimdEnv for NEON, feature detection for SVE.
		noSimdGuard := neonNoSimdGuard(target)
		sveGuard := sveRuntimeGuard(target)
		if noSimdGuard != "" {
			fmt.Fprintf(&buf, "\tif %s {\n", noSimdGuard)
			fmt.Fprintf(&buf, "\t\treturn\n")
			fmt.Fprintf(&buf, "\t}\n")
		}
		if sveGuard != "" {
			fmt.Fprintf(&buf, "\tif !%s {\n", sveGuard)
			fmt.Fprintf(&buf, "\t\treturn\n")
			fmt.Fprintf(&buf, "\t}\n")
		}
		emitGuardedDispatchAssignments(&buf, funcs, target)
		fmt.Fprintf(&buf, "}\n\n")
	}

	// Adapter functions: Image[T] → C struct → asm call
	for _, pf := range funcs {
		for _, elemType := range getCElemTypes(&pf) {
			profile := GetCProfile(target.Name, elemType)
			if profile == nil {
				continue
			}
			if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
				continue
			}
			emitZCAdapterFunc(&buf, &pf, elemType, target)
		}
	}

	dispPrefix := g.DispatchPrefix
	if dispPrefix == "" {
		dispPrefix = "dispatch"
	}
	filename := filepath.Join(g.OutputDir, fmt.Sprintf("z_c_%s_%s_%s.gen.go", dispPrefix, targetSuffix, archSuffix))
	if err := os.WriteFile(filename, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("write z_c dispatch: %w", err)
	}
	fmt.Printf("Generated: %s\n", filename)
	return nil
}

// emitSliceAsmPassthrough generates thin exported passthrough functions in asm/
// for non-struct AST-translated functions. These take the same unsafe.Pointer
// params as the raw GoAT stubs, allowing the parent package's z_c_ adapter
// to call them without importing unexported symbols.
func (g *Generator) emitSliceAsmPassthrough(funcs []ParsedFunc, target Target, asmDir string) error {
	var buf bytes.Buffer

	pkgName := goatPackageName(asmDir)

	buildTag := target.BuildTag
	if buildTag == "" {
		buildTag = "!noasm"
	} else {
		buildTag = "!noasm && " + buildTag
	}
	archSuffix := target.Arch()
	targetSuffix := strings.ToLower(target.Name)

	// File header
	fmt.Fprintf(&buf, "//go:build %s\n", buildTag)
	fmt.Fprintf(&buf, "// Code generated by hwygen -c. DO NOT EDIT.\n\n")
	fmt.Fprintf(&buf, "package %s\n\n", pkgName)
	fmt.Fprintf(&buf, "import \"unsafe\"\n\n")

	for _, pf := range funcs {
		elemTypes := getCElemTypes(&pf)
		for _, elemType := range elemTypes {
			profile := GetCProfile(target.Name, elemType)
			if profile == nil {
				continue
			}
			if shouldSkipPromotedType(&pf, profile) {
				continue
			}

			// Exported name: LiftUpdate53_S32
			exportedName := structAsmExportedName(pf.Name, elemType)
			// Assembly name: liftupdate53_c_s32_neon
			asmName := cAsmFuncName(pf.Name, elemType, targetSuffix)

			// Build param list matching the asm stub signature.
			// Slice params → unsafe.Pointer (data pointer)
			// Int params → unsafe.Pointer (pointer to int64)
			// Scalar T params → passed by value
			var paramDefs []string
			var paramNames []string
			for _, p := range pf.Params {
				if strings.HasPrefix(p.Type, "[]") {
					paramDefs = append(paramDefs, p.Name+" unsafe.Pointer")
					paramNames = append(paramNames, p.Name)
				} else if isGoScalarIntType(p.Type) || isGoScalarFloatType(p.Type) {
					ptrName := "p" + p.Name
					paramDefs = append(paramDefs, ptrName+" unsafe.Pointer")
					paramNames = append(paramNames, ptrName)
				} else if p.Type == "T" {
					// Scalar type-parameter param — passed as pointer in GOAT
					ptrName := "p" + p.Name
					paramDefs = append(paramDefs, ptrName+" unsafe.Pointer")
					paramNames = append(paramNames, ptrName)
				}
			}

			// Hidden length params for slice parameters.
			hasIntParams := false
			for _, p := range pf.Params {
				if isGoScalarIntType(p.Type) || isGoScalarFloatType(p.Type) {
					hasIntParams = true
					break
				}
			}
			mixedSlices := hasMixedSliceTypes(&pf)
			if !hasIntParams && !mixedSlices {
				// No explicit int params and same-type slices: single shared length.
				for _, p := range pf.Params {
					if strings.HasPrefix(p.Type, "[]") {
						paramDefs = append(paramDefs, "plen unsafe.Pointer")
						paramNames = append(paramNames, "plen")
						break
					}
				}
			} else {
				// Explicit int params or mixed slice types: per-slice length params.
				for _, p := range pf.Params {
					if strings.HasPrefix(p.Type, "[]") {
						plenName := "plen_" + p.Name
						paramDefs = append(paramDefs, plenName+" unsafe.Pointer")
						paramNames = append(paramNames, plenName)
					}
				}
			}

			// Return value output pointers
			for _, ret := range pf.Returns {
				name := ret.Name
				if name == "" {
					name = "result"
				}
				ptrName := "pout_" + name
				paramDefs = append(paramDefs, ptrName+" unsafe.Pointer")
				paramNames = append(paramNames, ptrName)
			}

			fmt.Fprintf(&buf, "// %s calls the %s SIMD assembly implementation.\n",
				exportedName, strings.ToUpper(target.Name))
			fmt.Fprintf(&buf, "func %s(%s) {\n",
				exportedName, strings.Join(paramDefs, ", "))
			fmt.Fprintf(&buf, "\t%s(%s)\n", asmName, strings.Join(paramNames, ", "))
			fmt.Fprintf(&buf, "}\n\n")
		}
	}

	ptDispPrefix := g.DispatchPrefix
	if ptDispPrefix == "" {
		ptDispPrefix = "dispatch"
	}
	filename := filepath.Join(asmDir, fmt.Sprintf("c_slice_passthrough_%s_%s_%s.gen.go", ptDispPrefix, targetSuffix, archSuffix))
	if err := os.WriteFile(filename, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("write slice asm passthrough: %w", err)
	}
	fmt.Printf("Generated: %s\n", filename)
	return nil
}

// slicePassthroughScalarType returns the Go type for by-value scalar params
// in asm passthrough functions (must match the GoAT stub signature).
func slicePassthroughScalarType(elemType string) string {
	switch elemType {
	case "float16", "hwy.Float16":
		return "uint16" // f16 passed as uint16
	case "bfloat16", "hwy.BFloat16":
		return "uint16" // bf16 passed as uint16
	case "float32":
		return "float32"
	case "float64":
		return "float64"
	default:
		return "float32"
	}
}

// emitZCDispatchForSlices generates a z_c_*.gen.go file in the parent package
// that overrides dispatch variables for non-struct AST-translated functions.
// It creates adapter functions that convert Go slice+int params to
// unsafe.Pointer calls into the asm/ passthrough functions.
func (g *Generator) emitZCDispatchForSlices(funcs []ParsedFunc, target Target) error {
	var buf bytes.Buffer

	buildTag := target.BuildTag
	if buildTag == "" {
		buildTag = "!noasm"
	} else {
		buildTag = "!noasm && " + buildTag
	}
	archSuffix := target.Arch()
	targetSuffix := strings.ToLower(target.Name)

	// Determine the asm subpackage import path
	asmImport, err := g.resolveAsmImportPath()
	if err != nil {
		return fmt.Errorf("resolve asm import path: %w", err)
	}

	// Determine if hwy import is needed:
	// - NEON targets need hwy for the NoSimdEnv() guard
	// - Non-streaming SVE targets (Linux) need hwy for HasSVE() guard
	// - Half-precision types need hwy for HasARMFP16()/HasARMBF16() guards
	needsHwy := isNeonTarget(target) || (isSVETarget(target) && !isSVEStreamingTarget(target))
	if !needsHwy {
		// Check if we need the hwy import for half-precision types
		for _, pf := range funcs {
			for _, et := range getCElemTypes(&pf) {
				if isHalfPrecisionType(et) {
					profile := GetCProfile(target.Name, et)
					if profile != nil && !shouldSkipPromotedType(&pf, profile) {
						needsHwy = true
						break
					}
				}
			}
			if needsHwy {
				break
			}
		}
	}

	// File header
	fmt.Fprintf(&buf, "//go:build %s\n", buildTag)
	fmt.Fprintf(&buf, "// Code generated by hwygen -c. DO NOT EDIT.\n\n")
	fmt.Fprintf(&buf, "package %s\n\n", g.PackageOut)
	fmt.Fprintf(&buf, "import (\n")
	fmt.Fprintf(&buf, "\t\"unsafe\"\n\n")
	if needsHwy {
		fmt.Fprintf(&buf, "\t\"github.com/ajroetker/go-highway/hwy\"\n")
	}
	fmt.Fprintf(&buf, "\t\"%s\"\n", asmImport)
	fmt.Fprintf(&buf, ")\n\n")

	// init() to override dispatch variables.
	// SVE streaming targets (Darwin) do NOT override scalar dispatch vars
	// because per-function smstart/smstop overhead (~50ns) makes SVE slower
	// than NEON for scalar calls. The adapter functions are still generated
	// for use from batch wrappers (e.g., sme_wrappers.go).
	// Non-streaming SVE targets (Linux) override dispatch normally.
	if !isSVEStreamingTarget(target) {
		capPrefix := cases.Title(language.English).String(g.DispatchPrefix)
		capTarget := cases.Title(language.English).String(strings.ToLower(target.Name))
		initFn := "init" + capPrefix + capTarget + "CAsm"
		fmt.Fprintf(&buf, "func init() {\n\t%s()\n}\n\n", initFn)
		fmt.Fprintf(&buf, "func %s() {\n", initFn)
		// Emit runtime guards: NoSimdEnv for NEON, feature detection for SVE.
		noSimdGuard := neonNoSimdGuard(target)
		sveGuard := sveRuntimeGuard(target)
		if noSimdGuard != "" {
			fmt.Fprintf(&buf, "\tif %s {\n", noSimdGuard)
			fmt.Fprintf(&buf, "\t\treturn\n")
			fmt.Fprintf(&buf, "\t}\n")
		}
		if sveGuard != "" {
			fmt.Fprintf(&buf, "\tif !%s {\n", sveGuard)
			fmt.Fprintf(&buf, "\t\treturn\n")
			fmt.Fprintf(&buf, "\t}\n")
		}
		emitGuardedDispatchAssignments(&buf, funcs, target)
		fmt.Fprintf(&buf, "}\n\n")
	}

	// Adapter functions: Go slice+int params → unsafe.Pointer → asm call
	for _, pf := range funcs {
		for _, elemType := range getCElemTypes(&pf) {
			profile := GetCProfile(target.Name, elemType)
			if profile == nil {
				continue
			}
			if shouldSkipPromotedType(&pf, profile) {
				continue
			}
			emitSliceZCAdapterFunc(&buf, &pf, elemType)
		}
	}

	dispPrefix2 := g.DispatchPrefix
	if dispPrefix2 == "" {
		dispPrefix2 = "dispatch"
	}
	filename := filepath.Join(g.OutputDir, fmt.Sprintf("z_c_slices_%s_%s_%s.gen.go", dispPrefix2, targetSuffix, archSuffix))
	if err := os.WriteFile(filename, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("write z_c slice dispatch: %w", err)
	}
	fmt.Printf("Generated: %s\n", filename)
	return nil
}

// extractPanicPreconditions extracts panic guard statements from the top of a
// function body. It collects variable declarations and if-statements that
// contain panic() calls, stopping at the first statement that is neither.
// Returns the Go source code for the preconditions, suitable for emitting
// in the asm adapter function.
func extractPanicPreconditions(body *ast.BlockStmt) string {
	if body == nil {
		return ""
	}
	fset := token.NewFileSet()
	var buf bytes.Buffer
	for _, stmt := range body.List {
		switch s := stmt.(type) {
		case *ast.IfStmt:
			if containsPanic(s.Body) {
				// This is a panic guard — emit it
				printer.Fprint(&buf, fset, s)
				buf.WriteString("\n")
				continue
			}
			// Early return (e.g., if len(x) == 0 { return }) — skip but continue
			if containsReturn(s.Body) {
				continue
			}
			// Some other if — stop collecting
			return buf.String()
		case *ast.AssignStmt:
			// Could be a supporting variable for a subsequent panic guard.
			// Check if any following if-statement uses this var and panics.
			if usedByFollowingPanicGuard(s, body.List) {
				printer.Fprint(&buf, fset, s)
				buf.WriteString("\n")
				continue
			}
			// Not part of a panic guard — stop collecting
			return buf.String()
		case *ast.ReturnStmt:
			// Bare return — stop
			return buf.String()
		default:
			// Any other statement type — stop collecting
			return buf.String()
		}
	}
	return buf.String()
}

// containsPanic checks if a block contains a panic() call.
func containsPanic(block *ast.BlockStmt) bool {
	found := false
	ast.Inspect(block, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "panic" {
				found = true
				return false
			}
		}
		return !found
	})
	return found
}

// containsReturn checks if a block contains a return statement.
func containsReturn(block *ast.BlockStmt) bool {
	for _, stmt := range block.List {
		if _, ok := stmt.(*ast.ReturnStmt); ok {
			return true
		}
	}
	return false
}

// usedByFollowingPanicGuard checks if an assignment's LHS variable is
// referenced by a subsequent if-statement that contains a panic().
func usedByFollowingPanicGuard(assign *ast.AssignStmt, stmts []ast.Stmt) bool {
	// Get the variable name(s) from the assignment LHS
	var varNames []string
	for _, lhs := range assign.Lhs {
		if ident, ok := lhs.(*ast.Ident); ok {
			varNames = append(varNames, ident.Name)
		}
	}
	if len(varNames) == 0 {
		return false
	}

	// Find this statement in the list and check subsequent statements
	found := false
	for _, stmt := range stmts {
		if stmt == assign {
			found = true
			continue
		}
		if !found {
			continue
		}
		if ifStmt, ok := stmt.(*ast.IfStmt); ok && containsPanic(ifStmt.Body) {
			// Check if the condition references any of our variables
			for _, name := range varNames {
				if referencesIdent(ifStmt.Cond, name) {
					return true
				}
			}
		}
	}
	return false
}

// referencesIdent checks if an expression references an identifier by name.
func referencesIdent(expr ast.Expr, name string) bool {
	found := false
	ast.Inspect(expr, func(n ast.Node) bool {
		if ident, ok := n.(*ast.Ident); ok && ident.Name == name {
			found = true
			return false
		}
		return !found
	})
	return found
}

// emitSliceZCAdapterFunc generates an adapter function for a non-struct
// AST-translated function. The adapter converts Go slice/int/scalar params
// to the unsafe.Pointer calling convention expected by the asm passthrough.
func emitSliceZCAdapterFunc(buf *bytes.Buffer, pf *ParsedFunc, elemType string) {
	adapterName := buildAdapterFuncName(pf.Name, elemType)
	asmExportedName := structAsmExportedName(pf.Name, elemType)

	goSliceType := astWrapperGoSliceType(elemType)

	// Build Go function signature matching the dispatch var type
	goSig := groupGoParams(pf, goSliceType, elemType)

	// Classify params.
	var sliceParams []string
	var intParams []string
	var floatParams []string
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			sliceParams = append(sliceParams, p.Name)
		} else if isGoScalarIntType(p.Type) {
			intParams = append(intParams, p.Name)
		} else if isGoScalarFloatType(p.Type) {
			floatParams = append(floatParams, p.Name)
		}
	}
	hasScalarParams := len(intParams) > 0 || len(floatParams) > 0

	// Determine hidden length param strategy.
	// Use per-slice lengths when slices have different element types (e.g.,
	// []byte and []float32) because len() has different semantics per type.
	mixedSlices := hasMixedSliceTypes(pf)
	needsSharedLen := !hasScalarParams && !mixedSlices && len(sliceParams) > 0
	needsPerSliceLen := (hasScalarParams || mixedSlices) && len(sliceParams) > 0

	// Determine if we have return values
	hasReturns := len(pf.Returns) > 0

	// Build return type string
	var returnType string
	if hasReturns {
		if len(pf.Returns) == 1 {
			returnType = goReturnTypeForWrapper(pf.Returns[0].Type, elemType)
		} else {
			var retTypes []string
			for _, ret := range pf.Returns {
				retTypes = append(retTypes, goReturnTypeForWrapper(ret.Type, elemType))
			}
			returnType = "(" + strings.Join(retTypes, ", ") + ")"
		}
	}

	// Function signature
	if hasReturns {
		fmt.Fprintf(buf, "func %s(%s) %s {\n", adapterName, goSig, returnType)
	} else {
		fmt.Fprintf(buf, "func %s(%s) {\n", adapterName, goSig)
	}

	// Emit panic preconditions from the base function (bounds checks, etc.)
	// These ensure the asm adapter preserves the same safety guarantees as
	// the base Go implementation.
	if preconditions := extractPanicPreconditions(pf.Body); preconditions != "" {
		for _, line := range strings.Split(strings.TrimRight(preconditions, "\n"), "\n") {
			fmt.Fprintf(buf, "\t%s\n", line)
		}
	}

	// Build nil-safe pointer variables for each slice parameter.
	// Some params (e.g. bias) may legitimately be nil — pass nil to C code
	// which checks for NULL. Using per-param variables avoids a combined
	// early-return that would skip computation when optional params are nil.
	for _, sp := range sliceParams {
		fmt.Fprintf(buf, "\tvar p_%s unsafe.Pointer\n", sp)
		fmt.Fprintf(buf, "\tif len(%s) > 0 {\n", sp)
		fmt.Fprintf(buf, "\t\tp_%s = unsafe.Pointer(&%s[0])\n", sp, sp)
		fmt.Fprintf(buf, "\t}\n")
	}

	// Convert int params to int64 for unsafe.Pointer
	for _, ip := range intParams {
		fmt.Fprintf(buf, "\t%sVal := int64(%s)\n", ip, ip)
	}

	// Convert scalar T params to addressable locals for unsafe.Pointer
	for _, p := range pf.Params {
		if p.Type == "T" {
			goScalarType := astWrapperGoScalarType(elemType)
			if goScalarType == "hwy.Float16" || goScalarType == "hwy.BFloat16" {
				fmt.Fprintf(buf, "\t%sVal := uint16(%s)\n", p.Name, p.Name)
			} else {
				fmt.Fprintf(buf, "\t%sVal := %s\n", p.Name, p.Name)
			}
		} else if isGoScalarFloatType(p.Type) {
			fmt.Fprintf(buf, "\t%sVal := %s\n", p.Name, p.Name)
		}
	}

	// Hidden length params
	if needsSharedLen {
		fmt.Fprintf(buf, "\tlenVal := int64(len(%s))\n", sliceParams[0])
	} else if needsPerSliceLen {
		for _, sp := range sliceParams {
			fmt.Fprintf(buf, "\tlen_%sVal := int64(len(%s))\n", sp, sp)
		}
	}

	// Output variables for return values.
	// Array returns use the actual type; float scalars use float32/float64
	// so GOAT generates correct FP load/store instructions; int scalars use int64.
	if hasReturns {
		for _, ret := range pf.Returns {
			name := ret.Name
			if name == "" {
				name = "result"
			}
			goType := goReturnTypeForWrapper(ret.Type, elemType)
			if isGoArrayType(goType) {
				fmt.Fprintf(buf, "\tvar out_%s %s\n", name, goType)
			} else {
				outType := goReturnOutVarType(goType)
				fmt.Fprintf(buf, "\tvar out_%s %s\n", name, outType)
			}
		}
	}

	// Call asm passthrough
	fmt.Fprintf(buf, "\tasm.%s(\n", asmExportedName)
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			fmt.Fprintf(buf, "\t\tp_%s,\n", p.Name)
		} else if isGoScalarIntType(p.Type) || isGoScalarFloatType(p.Type) {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&%sVal),\n", p.Name)
		} else if p.Type == "T" {
			// Scalar type-parameter param — passed as pointer in GOAT
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&%sVal),\n", p.Name)
		}
	}
	// Hidden length pointers
	if needsSharedLen {
		fmt.Fprintf(buf, "\t\tunsafe.Pointer(&lenVal),\n")
	} else if needsPerSliceLen {
		for _, sp := range sliceParams {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&len_%sVal),\n", sp)
		}
	}
	// Output pointers
	if hasReturns {
		for _, ret := range pf.Returns {
			name := ret.Name
			if name == "" {
				name = "result"
			}
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&out_%s),\n", name)
		}
	}
	fmt.Fprintf(buf, "\t)\n")

	// Return with type narrowing from int64 to actual return type.
	// Array returns are already the correct type, no conversion needed.
	if hasReturns {
		var retExprs []string
		for _, ret := range pf.Returns {
			name := ret.Name
			if name == "" {
				name = "result"
			}
			goType := goReturnTypeForWrapper(ret.Type, elemType)
			if isGoArrayType(goType) {
				retExprs = append(retExprs, "out_"+name)
			} else {
				retExprs = append(retExprs, goType+"(out_"+name+")")
			}
		}
		fmt.Fprintf(buf, "\treturn %s\n", strings.Join(retExprs, ", "))
	}
	fmt.Fprintf(buf, "}\n\n")
}

// resolveAsmImportPath computes the Go import path for the asm/ subdirectory.
func (g *Generator) resolveAsmImportPath() (string, error) {
	absOutputDir, err := filepath.Abs(g.OutputDir)
	if err != nil {
		return "", fmt.Errorf("abs output dir: %w", err)
	}
	moduleRoot, moduleName, err := FindModuleRoot(absOutputDir)
	if err != nil {
		return "", fmt.Errorf("find module root: %w", err)
	}
	asmAbsDir := filepath.Join(absOutputDir, "asm")
	relPath, err := filepath.Rel(moduleRoot, asmAbsDir)
	if err != nil {
		return "", fmt.Errorf("rel path: %w", err)
	}
	return moduleName + "/" + relPath, nil
}

// structAsmExportedName builds the exported function name for asm/ passthrough.
// E.g., BaseForwardICT + float32 → ForwardICT_F32
func structAsmExportedName(baseName, elemType string) string {
	name := strings.TrimPrefix(baseName, "Base")
	name = strings.TrimPrefix(name, "base")
	return name + "_" + cTypePublicSuffix(elemType)
}

// buildDispatchVarName builds the dispatch variable name from a base function name.
// For generic functions: BaseForwardICT + float32 → ForwardICTFloat32
// For non-generic functions: BaseFusedInt8MatMul + float32 → FusedInt8MatMul (no suffix)
// Unexported functions stay unexported: basePackedMicroKernelGeneral → packedMicroKernelGeneralFloat16
func buildDispatchVarName(baseName, elemType string, isGeneric bool) string {
	private := len(baseName) > 0 && baseName[0] >= 'a' && baseName[0] <= 'z'
	name := strings.TrimPrefix(baseName, "Base")
	name = strings.TrimPrefix(name, "base")
	if private {
		name = makeUnexported(name)
	}
	if isGeneric {
		return name + typeNameToDispatchSuffix(elemType)
	}
	return name
}

// buildAdapterFuncName builds the unexported adapter function name.
// E.g., BaseForwardICT + float32 → forwardICTAsmF32
func buildAdapterFuncName(baseName, elemType string) string {
	name := strings.TrimPrefix(baseName, "Base")
	name = strings.TrimPrefix(name, "base")
	// Lowercase first letter
	if len(name) > 0 {
		name = strings.ToLower(name[:1]) + name[1:]
	}
	return name + "Asm" + cTypePublicSuffix(elemType)
}

// typeNameToDispatchSuffix returns the suffix used in dispatch variable names.
func typeNameToDispatchSuffix(elemType string) string {
	switch elemType {
	case "float32":
		return "Float32"
	case "float64":
		return "Float64"
	case "float16", "hwy.Float16":
		return "Float16"
	case "bfloat16", "hwy.BFloat16":
		return "BFloat16"
	case "int32":
		return "Int32"
	case "int64":
		return "Int64"
	case "uint32":
		return "Uint32"
	case "uint64":
		return "Uint64"
	default:
		return "Float32"
	}
}

// emitZCAdapterFunc generates an adapter function that converts *Image[T] params
// to C-compatible structs and calls the asm/ exported wrapper.
func emitZCAdapterFunc(buf *bytes.Buffer, pf *ParsedFunc, elemType string, target Target) {
	adapterName := buildAdapterFuncName(pf.Name, elemType)

	// Build set of type parameter names for generic substitution.
	typeParamNames := make(map[string]bool, len(pf.TypeParams))
	for _, tp := range pf.TypeParams {
		typeParamNames[tp.Name] = true
	}

	// Build parameter list with specialized types.
	// Struct pointer types: *Image[T] → *Image[float32]
	// Type parameter types: T → float32 (or hwy.Float16, etc.)
	var params []string
	for _, p := range pf.Params {
		paramType := specializeStructPtrType(p.Type, elemType)
		if typeParamNames[p.Type] {
			paramType = goTypeName(elemType)
		}
		params = append(params, p.Name+" "+paramType)
	}
	paramList := strings.Join(params, ", ")

	// Discover unified struct layout from all parameters
	elemCType := goElemTypeToCType(elemType)
	unifiedFields := DiscoverUnifiedStructFields(pf, elemCType)

	fmt.Fprintf(buf, "func %s(%s) {\n", adapterName, paramList)

	// Nil checks
	var nilChecks []string
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			nilChecks = append(nilChecks, p.Name+" == nil")
		}
	}
	if len(nilChecks) > 0 {
		fmt.Fprintf(buf, "\tif %s {\n", strings.Join(nilChecks, " || "))
		fmt.Fprintf(buf, "\t\treturn\n")
		fmt.Fprintf(buf, "\t}\n")
	}

	// Create C-compatible struct instances for each struct pointer param
	for _, p := range pf.Params {
		if !isGenericStructPtr(p.Type) {
			continue
		}
		if len(unifiedFields) == 0 {
			continue
		}
		fmt.Fprintf(buf, "\tc%s := struct {\n", p.Name)
		for _, field := range unifiedFields {
			fmt.Fprintf(buf, "\t\t%s %s\n", field.Name, field.GoType)
		}
		fmt.Fprintf(buf, "\t}{\n")
		for _, field := range unifiedFields {
			if field.IsPtr {
				fmt.Fprintf(buf, "\t\t%s: unsafe.Pointer(&%s.Row(0)[0]),\n", field.Name, p.Name)
			} else {
				fmt.Fprintf(buf, "\t\t%s: %s(%s%s),\n", field.Name, field.GoType, p.Name, field.GoGetter)
			}
		}
		fmt.Fprintf(buf, "\t}\n")
	}

	// Call the asm/ exported function, passing struct pointers and scalar
	// params. Scalar params are passed as unsafe.Pointer(&param) to match
	// the GOAT calling convention (all scalars passed as pointers).
	asmExportedName := structAsmExportedName(pf.Name, elemType)
	fmt.Fprintf(buf, "\tasm.%s(\n", asmExportedName)
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&c%s),\n", p.Name)
		} else if typeParamNames[p.Type] || isGoScalarIntType(p.Type) || isGoScalarFloatType(p.Type) {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&%s),\n", p.Name)
		}
	}
	fmt.Fprintf(buf, "\t)\n")
	fmt.Fprintf(buf, "}\n\n")
}

// emitCWrapperFunc generates a single wrapper function.
func emitCWrapperFunc(buf *bytes.Buffer, pf *ParsedFunc, elemType, targetSuffix string) {
	publicName := buildCPublicName(pf.Name, elemType)
	asmName := cAsmFuncName(pf.Name, elemType, targetSuffix)
	sliceType := cWrapperSliceType(elemType)

	// Determine parameter names based on function signature
	inputParam := "input"
	outputParam := "result"

	// Check if it's an in-place operation (modifies input) or has separate output
	hasOutput := false
	for _, param := range pf.Params {
		if param.Name == "output" || param.Name == "result" {
			hasOutput = true
			outputParam = param.Name
			break
		}
	}

	fmt.Fprintf(buf, "// %s computes %s for entire arrays using %s SIMD.\n", publicName, pf.Name, strings.ToUpper(targetSuffix))
	if hasOutput {
		fmt.Fprintf(buf, "func %s(%s, %s %s) {\n", publicName, inputParam, outputParam, sliceType)
	} else {
		fmt.Fprintf(buf, "func %s(%s %s) %s {\n", publicName, inputParam, sliceType, sliceType)
		fmt.Fprintf(buf, "\t%s := make(%s, len(%s))\n", outputParam, sliceType, inputParam)
	}

	fmt.Fprintf(buf, "\tif len(%s) == 0 {\n", inputParam)
	if hasOutput {
		fmt.Fprintf(buf, "\t\treturn\n")
	} else {
		fmt.Fprintf(buf, "\t\treturn %s\n", outputParam)
	}
	fmt.Fprintf(buf, "\t}\n")

	fmt.Fprintf(buf, "\tn := int64(len(%s))\n", inputParam)
	fmt.Fprintf(buf, "\t%s(unsafe.Pointer(&%s[0]), unsafe.Pointer(&%s[0]), unsafe.Pointer(&n))\n",
		asmName, inputParam, outputParam)

	if !hasOutput {
		fmt.Fprintf(buf, "\treturn %s\n", outputParam)
	}
	fmt.Fprintf(buf, "}\n\n")
}

// buildCPublicName creates the public function name.
// BaseExpVec -> ExpVecCF32, BaseGELU -> GELUCF32
func buildCPublicName(baseName, elemType string) string {
	name := strings.TrimPrefix(baseName, "Base")

	typeSuffix := cTypePublicSuffix(elemType)

	return name + "C" + typeSuffix
}

// cAsmFuncName creates the assembly function name.
// BaseMatMul, float32, neon -> matmul_c_f32_neon
func cAsmFuncName(baseName, elemType, targetSuffix string) string {
	name := strings.ToLower(baseName)
	name = strings.TrimPrefix(name, "base")

	return name + "_c_" + cTypeSuffix(elemType) + "_" + targetSuffix
}

// cTypeSuffix returns the short type suffix for file naming.
func cTypeSuffix(elemType string) string {
	switch elemType {
	case "float32":
		return "f32"
	case "float64":
		return "f64"
	case "float16", "hwy.Float16":
		return "f16"
	case "bfloat16", "hwy.BFloat16":
		return "bf16"
	case "int32":
		return "s32"
	case "int64":
		return "s64"
	case "uint64":
		return "u64"
	case "uint32":
		return "u32"
	case "uint8", "byte":
		return "u8"
	default:
		return "f32"
	}
}

// cTypePublicSuffix returns the public Go suffix for function names.
func cTypePublicSuffix(elemType string) string {
	switch elemType {
	case "float32":
		return "F32"
	case "float64":
		return "F64"
	case "float16", "hwy.Float16":
		return "F16"
	case "bfloat16", "hwy.BFloat16":
		return "BF16"
	case "int32":
		return "S32"
	case "int64":
		return "S64"
	case "uint64":
		return "U64"
	case "uint32":
		return "U32"
	case "uint8", "byte":
		return "U8"
	default:
		return "F32"
	}
}

// isGoScalarIntType returns true for Go integer types that should be passed
// as long* pointers through GoAT (which only accepts int64_t/long, float,
// double, _Bool, or pointer as function arguments).
func isGoScalarIntType(goType string) bool {
	switch goType {
	case "int", "int64", "int32", "int16", "int8",
		"uint", "uint64", "uint8", "uint16", "uint32", "uintptr",
		"byte": // byte is an alias for uint8
		return true
	default:
		return false
	}
}

// hasMixedSliceTypes reports whether a function's slice parameters have
// different element types (e.g., []byte and []float32). When true, each slice
// needs its own length parameter because len([]byte) and len([]float32) have
// different semantics (byte count vs element count).
func hasMixedSliceTypes(pf *ParsedFunc) bool {
	var firstElem string
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			elem := strings.TrimPrefix(p.Type, "[]")
			if firstElem == "" {
				firstElem = elem
			} else if elem != firstElem {
				return true
			}
		}
	}
	return false
}

func isGoScalarFloatType(goType string) bool {
	switch goType {
	case "float32", "float64":
		return true
	default:
		return false
	}
}

// goElemTypeToCType converts a Go element type to a C type.
func goElemTypeToCType(elemType string) string {
	switch elemType {
	case "float32":
		return "float"
	case "float64":
		return "double"
	case "int32":
		return "int"
	case "int64":
		return "long"
	case "uint32":
		return "unsigned int"
	case "uint64":
		return "unsigned long"
	case "uint8", "byte":
		return "unsigned char"
	default:
		return "float"
	}
}

// cWrapperSliceType returns the Go slice type for wrapper functions.
func cWrapperSliceType(elemType string) string {
	switch elemType {
	case "float32":
		return "[]float32"
	case "float64":
		return "[]float64"
	case "float16", "hwy.Float16":
		return "[]uint16" // f16 stored as uint16 in Go
	case "bfloat16", "hwy.BFloat16":
		return "[]uint16" // bf16 stored as uint16 in Go
	default:
		return "[]float32"
	}
}

// astWrapperGoSliceType returns the Go slice type for AST-translated wrapper
// functions. Unlike cWrapperSliceType, this uses the proper hwy types for
// f16/bf16 to match the original Go function signatures.
func astWrapperGoSliceType(elemType string) string {
	switch elemType {
	case "float32":
		return "[]float32"
	case "float64":
		return "[]float64"
	case "float16", "hwy.Float16":
		return "[]hwy.Float16"
	case "bfloat16", "hwy.BFloat16":
		return "[]hwy.BFloat16"
	case "int32":
		return "[]int32"
	case "int64":
		return "[]int64"
	case "uint64":
		return "[]uint64"
	case "uint32":
		return "[]uint32"
	case "uint8", "byte":
		return "[]byte"
	default:
		return "[]float32"
	}
}

// goReturnTypeForWrapper maps Go return types to their Go type name for wrappers.
// elemType is used to resolve generic "T" return types to concrete types.
func goReturnTypeForWrapper(goType, elemType string) string {
	if goType == "T" {
		return astWrapperGoScalarType(elemType)
	}
	switch goType {
	case "uint32":
		return "uint32"
	case "uint64":
		return "uint64"
	case "int", "int64":
		return "int"
	case "int32":
		return "int32"
	case "float32":
		return "float32"
	case "float64":
		return "float64"
	default:
		return goType
	}
}

// isGoArrayType returns true if goType is a fixed-size Go array type (e.g., [4]uint32).
func isGoArrayType(goType string) bool {
	return len(goType) >= 3 && goType[0] == '[' && goType[1] != ']'
}

// goReturnOutVarType returns the Go type for the output variable in the asm wrapper.
// Float types use their native Go type so the pointer passed to GOAT is correctly
// typed (float32 → float *, float64 → double *). Integer types use int64 to match
// GOAT's long * convention.
func goReturnOutVarType(goType string) string {
	switch goType {
	case "float32":
		return "float32"
	case "float64":
		return "float64"
	default:
		return "int64"
	}
}

// goReturnZeroValue returns the zero-value literal for a Go return type.
// For scalar types returns "0", for array types returns e.g. "[4]uint32{}".
func goReturnZeroValue(goType, elemType string) string {
	rt := goReturnTypeForWrapper(goType, elemType)
	if isGoArrayType(rt) {
		return rt + "{}"
	}
	return "0"
}

// emitASTCWrapperFunc generates a wrapper function for an AST-translated function
// with arbitrary parameters (multiple slices + int dimensions).
//
// Handles three parameter patterns:
//  1. Functions with explicit int params (matmul): a, b, c []float32, m, n, k int
//  2. Functions without int params (rabitq): code, q1, q2 []uint64 → adds hidden length param
//  3. Functions with return values (rabitq, varint): uint32 → adds output pointer param
//
// Note: Functions with *Image[T] params are handled through the normal transformer/emitter
// flow via the asmBody generation, not through separate wrapper functions.
func emitASTCWrapperFunc(buf *bytes.Buffer, pf *ParsedFunc, elemType, targetSuffix string) {
	// Skip functions with *Image[T] params - they go through the normal transformer flow
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			return
		}
	}

	publicName := buildCPublicName(pf.Name, elemType)
	asmName := cAsmFuncName(pf.Name, elemType, targetSuffix)
	goSliceType := astWrapperGoSliceType(elemType)

	// Classify params
	var sliceParams []string
	var intParams []string
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			sliceParams = append(sliceParams, p.Name)
		} else if isGoScalarIntType(p.Type) {
			intParams = append(intParams, p.Name)
		}
	}
	hasScalarParams := len(intParams) > 0
	for _, p := range pf.Params {
		if isGoScalarFloatType(p.Type) {
			hasScalarParams = true
			break
		}
	}

	// Determine hidden length param strategy.
	// Use per-slice lengths when slices have different element types (e.g.,
	// []byte and []float32) because len() has different semantics per type.
	mixedSlices := hasMixedSliceTypes(pf)
	needsSharedLen := !hasScalarParams && !mixedSlices && len(sliceParams) > 0
	needsPerSliceLen := (hasScalarParams || mixedSlices) && len(sliceParams) > 0
	firstSlice := ""
	if needsSharedLen && len(sliceParams) > 0 {
		firstSlice = sliceParams[0]
	}

	// Determine if we have return values
	hasReturns := len(pf.Returns) > 0

	// Build Go signature
	goSig := groupGoParams(pf, goSliceType, elemType)

	// Build return type string
	var returnType string
	if hasReturns {
		if len(pf.Returns) == 1 {
			returnType = goReturnTypeForWrapper(pf.Returns[0].Type, elemType)
		} else {
			var retTypes []string
			for _, ret := range pf.Returns {
				retTypes = append(retTypes, goReturnTypeForWrapper(ret.Type, elemType))
			}
			returnType = "(" + strings.Join(retTypes, ", ") + ")"
		}
	}

	// Comment + function signature
	fmt.Fprintf(buf, "// %s computes %s using %s SIMD assembly.\n",
		publicName, strings.TrimPrefix(pf.Name, "Base"), strings.ToUpper(targetSuffix))
	if hasReturns {
		fmt.Fprintf(buf, "func %s(%s) %s {\n", publicName, goSig, returnType)
	} else {
		fmt.Fprintf(buf, "func %s(%s) {\n", publicName, goSig)
	}

	// Zero-length guard for slice-only functions
	if needsSharedLen {
		fmt.Fprintf(buf, "\tif len(%s) == 0 {\n", firstSlice)
		if hasReturns {
			// Return zero values
			var zeros []string
			for _, ret := range pf.Returns {
				zeros = append(zeros, goReturnZeroValue(ret.Type, elemType))
			}
			fmt.Fprintf(buf, "\t\treturn %s\n", strings.Join(zeros, ", "))
		} else {
			fmt.Fprintf(buf, "\t\treturn\n")
		}
		fmt.Fprintf(buf, "\t}\n")
	}

	// Build nil-safe pointer variables for each slice parameter.
	// Some params (e.g. bias) may legitimately be nil — pass nil to C code
	// which checks for NULL. Using per-param variables avoids a combined
	// early-return that would skip computation when optional params are nil.
	for _, sp := range sliceParams {
		fmt.Fprintf(buf, "\tvar p_%s unsafe.Pointer\n", sp)
		fmt.Fprintf(buf, "\tif len(%s) > 0 {\n", sp)
		fmt.Fprintf(buf, "\t\tp_%s = unsafe.Pointer(&%s[0])\n", sp, sp)
		fmt.Fprintf(buf, "\t}\n")
	}

	// Guard against zero dimensions for matmul-like functions (3 slices, 3 ints).
	// If any dimension is zero there is no work to do, and the bounds checks
	// below would compute incorrect products (e.g., len(a) < 0*k).
	if len(sliceParams) == 3 && len(intParams) == 3 {
		var checks []string
		for _, ip := range intParams {
			checks = append(checks, ip+" == 0")
		}
		fmt.Fprintf(buf, "\tif %s {\n", strings.Join(checks, " || "))
		if hasReturns {
			var zeros []string
			for _, ret := range pf.Returns {
				zeros = append(zeros, goReturnZeroValue(ret.Type, elemType))
			}
			fmt.Fprintf(buf, "\t\treturn %s\n", strings.Join(zeros, ", "))
		} else {
			fmt.Fprintf(buf, "\t\treturn\n")
		}
		fmt.Fprintf(buf, "\t}\n")
	}

	// Bounds checks for slices
	if len(sliceParams) > 0 && len(intParams) > 0 {
		emitBoundsChecks(buf, pf, sliceParams, intParams)
	}

	// Convert int params to int64 for unsafe.Pointer
	for _, ip := range intParams {
		fmt.Fprintf(buf, "\t%sVal := int64(%s)\n", ip, ip)
	}

	// Convert scalar T params to addressable locals for unsafe.Pointer
	for _, p := range pf.Params {
		if p.Type == "T" {
			goScalarType := astWrapperGoScalarType(elemType)
			if goScalarType == "hwy.Float16" || goScalarType == "hwy.BFloat16" {
				fmt.Fprintf(buf, "\t%sVal := uint16(%s)\n", p.Name, p.Name)
			} else {
				fmt.Fprintf(buf, "\t%sVal := %s\n", p.Name, p.Name)
			}
		} else if isGoScalarFloatType(p.Type) {
			fmt.Fprintf(buf, "\t%sVal := %s\n", p.Name, p.Name)
		}
	}

	// Hidden length params
	if needsSharedLen {
		fmt.Fprintf(buf, "\tlenVal := int64(len(%s))\n", firstSlice)
	} else if needsPerSliceLen {
		for _, sp := range sliceParams {
			fmt.Fprintf(buf, "\tlen_%sVal := int64(len(%s))\n", sp, sp)
		}
	}

	// Output variables for return values.
	// GOAT always uses 64-bit (long *) for output pointers, so we declare
	// int64 variables and cast to the actual return type.
	// Array returns (e.g., [4]uint32) use the actual type since the C code
	// writes directly through the output pointer.
	if hasReturns {
		for _, ret := range pf.Returns {
			name := ret.Name
			if name == "" {
				name = "result"
			}
			goType := goReturnTypeForWrapper(ret.Type, elemType)
			if isGoArrayType(goType) {
				fmt.Fprintf(buf, "\tvar out_%s %s\n", name, goType)
			} else {
				fmt.Fprintf(buf, "\tvar out_%s int64\n", name)
			}
		}
	}

	// Call assembly function
	fmt.Fprintf(buf, "\t%s(\n", asmName)

	// Regular params
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			fmt.Fprintf(buf, "\t\tp_%s,\n", p.Name)
		} else if isGoScalarIntType(p.Type) || isGoScalarFloatType(p.Type) {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&%sVal),\n", p.Name)
		} else if p.Type == "T" {
			// Scalar type-parameter param — passed as pointer in GOAT
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&%sVal),\n", p.Name)
		}
	}
	// Hidden length params
	if needsSharedLen {
		fmt.Fprintf(buf, "\t\tunsafe.Pointer(&lenVal),\n")
	} else if needsPerSliceLen {
		for _, sp := range sliceParams {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&len_%sVal),\n", sp)
		}
	}
	// Output pointer params
	if hasReturns {
		for _, ret := range pf.Returns {
			name := ret.Name
			if name == "" {
				name = "result"
			}
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&out_%s),\n", name)
		}
	}
	fmt.Fprintf(buf, "\t)\n")

	// Return with type narrowing from int64 to actual return type.
	// Array returns are already the correct type, so no conversion needed.
	if hasReturns {
		var retExprs []string
		for _, ret := range pf.Returns {
			name := ret.Name
			if name == "" {
				name = "result"
			}
			goType := goReturnTypeForWrapper(ret.Type, elemType)
			if isGoArrayType(goType) {
				retExprs = append(retExprs, "out_"+name)
			} else {
				retExprs = append(retExprs, goType+"(out_"+name+")")
			}
		}
		fmt.Fprintf(buf, "\treturn %s\n", strings.Join(retExprs, ", "))
	}
	fmt.Fprintf(buf, "}\n\n")
}

// hasStructPtrParams returns true if the function has any struct pointer parameters
// (e.g., *Image[T], *SomeStruct[T]).
func hasStructPtrParams(pf *ParsedFunc) bool {
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			return true
		}
	}
	return false
}

// isGenericStructPtr returns true if the type is a pointer to a generic struct
// (e.g., *Image[T], *SomeStruct[T]).
func isGenericStructPtr(typeStr string) bool {
	// Check for *Type[...] pattern
	if !strings.HasPrefix(typeStr, "*") {
		return false
	}
	// Must have a type parameter [...]
	return strings.Contains(typeStr, "[") && strings.Contains(typeStr, "]")
}

// StructField describes a field in a C-compatible struct layout.
// Fields are discovered by analyzing method calls in the function body.
type StructField struct {
	Name     string // Field name in C (lowercased method name)
	GoType   string // Go type for wrapper generation (e.g., "int64", "unsafe.Pointer")
	CType    string // C type (e.g., "long" for dimension getters, "%s *" for data pointer)
	GoGetter string // Go expression to get value (e.g., ".Width()", ".Row(0)[0]")
	IsPtr    bool   // Whether this field is a pointer
	IsData   bool   // Whether this is the data pointer field (for Row-like accessors)
}

// DiscoverUnifiedStructFields discovers a unified struct layout from ALL struct
// parameters in the function. This ensures all parameters use the same C struct
// layout, which is required since C uses a single typedef.
func DiscoverUnifiedStructFields(pf *ParsedFunc, elemCType string) []StructField {
	if pf.Body == nil {
		return nil
	}

	// Collect all struct param names
	var structParamNames []string
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			structParamNames = append(structParamNames, p.Name)
		}
	}

	if len(structParamNames) == 0 {
		return nil
	}

	// Track discovered fields (unified across all params)
	discovered := make(map[string]StructField)
	// Track method names to avoid treating them as fields
	methodNames := make(map[string]bool)

	// First pass: find all method calls to identify method names
	ast.Inspect(pf.Body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if ident, ok := sel.X.(*ast.Ident); ok {
					if slices.Contains(structParamNames, ident.Name) {
						methodNames[sel.Sel.Name] = true
					}
				}
			}
		}
		return true
	})

	// Second pass: discover fields from method calls and field accesses
	ast.Inspect(pf.Body, func(n ast.Node) bool {
		// Handle method calls: param.Method(...)
		if call, ok := n.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if ident, ok := sel.X.(*ast.Ident); ok {
					isStructParam := slices.Contains(structParamNames, ident.Name)
					if !isStructParam {
						return true
					}

					methodName := sel.Sel.Name
					fieldName := strings.ToLower(methodName)
					argCount := len(call.Args)

					if argCount == 0 {
						// Simple getter: Width() → width field of type long
						if _, exists := discovered[fieldName]; !exists {
							discovered[fieldName] = StructField{
								Name:     fieldName,
								GoType:   "int64",
								CType:    "long",
								GoGetter: "." + methodName + "()",
								IsPtr:    false,
							}
						}
					} else if argCount == 1 {
						// Row-like accessor: Row(y) → data pointer field
						if _, exists := discovered["data"]; !exists {
							discovered["data"] = StructField{
								Name:     "data",
								GoType:   "unsafe.Pointer",
								CType:    elemCType + " *",
								GoGetter: ".Row(0)[0]",
								IsPtr:    true,
								IsData:   true,
							}
						}
						// Add stride if not already present
						if _, exists := discovered["stride"]; !exists {
							discovered["stride"] = StructField{
								Name:     "stride",
								GoType:   "int64",
								CType:    "long",
								GoGetter: ".Stride()",
								IsPtr:    false,
							}
						}
					}
				}
			}
		}

		// Handle direct field accesses: param.field
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if ident, ok := sel.X.(*ast.Ident); ok {
				isStructParam := slices.Contains(structParamNames, ident.Name)
				if !isStructParam {
					return true
				}

				fieldName := sel.Sel.Name

				// Skip method names - they're not fields
				if methodNames[fieldName] {
					return true
				}

				if _, exists := discovered[fieldName]; !exists {
					if fieldName == "data" {
						discovered[fieldName] = StructField{
							Name:     fieldName,
							GoType:   "unsafe.Pointer",
							CType:    elemCType + " *",
							GoGetter: ".data",
							IsPtr:    true,
							IsData:   true,
						}
					} else {
						discovered[fieldName] = StructField{
							Name:     fieldName,
							GoType:   "int64",
							CType:    "long",
							GoGetter: "." + fieldName,
							IsPtr:    false,
						}
					}
				}
			}
		}

		return true
	})

	// Convert map to slice in deterministic order
	var fields []StructField
	// Data field first (if present)
	if f, ok := discovered["data"]; ok {
		fields = append(fields, f)
		delete(discovered, "data")
	}
	// Then other fields in sorted order
	var names []string
	for name := range discovered {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		fields = append(fields, discovered[name])
	}

	return fields
}

// extractStructBaseName extracts the base name from a struct type.
// E.g., "*Image[T]" -> "Image", "*image.Image[float32]" -> "Image"
func extractStructBaseName(structType string) string {
	baseName := structType
	if strings.HasPrefix(baseName, "*") {
		baseName = baseName[1:]
	}
	if idx := strings.Index(baseName, "["); idx != -1 {
		baseName = baseName[:idx]
	}
	if idx := strings.LastIndex(baseName, "."); idx != -1 {
		baseName = baseName[idx+1:]
	}
	return baseName
}

// emitStructPtrAsmWrapper generates a wrapper function for functions with struct
// pointer parameters. The wrapper creates C-compatible struct layouts and calls
// the assembly function.
//
// This uses transformer-style naming (e.g., BaseForwardICT_neon) so the wrapper
// integrates with the dispatch system.
func emitStructPtrAsmWrapper(buf *bytes.Buffer, pf *ParsedFunc, elemType string, target Target) {
	// Build function name matching transformer convention
	funcName := pf.Name + target.Suffix()
	if elemType != "float32" {
		funcName += "_" + typeNameToSuffix(elemType)
	}

	// Build assembly function name
	targetSuffix := strings.ToLower(target.Name)
	asmName := cAsmFuncName(pf.Name, elemType, targetSuffix)

	// Build parameter list with specialized types
	var params []string
	for _, p := range pf.Params {
		paramType := specializeStructPtrType(p.Type, elemType)
		params = append(params, p.Name+" "+paramType)
	}
	paramList := strings.Join(params, ", ")

	// Emit function signature
	fmt.Fprintf(buf, "// %s calls the %s SIMD assembly implementation.\n",
		funcName, strings.ToUpper(target.Name))
	fmt.Fprintf(buf, "func %s(%s) {\n", funcName, paramList)

	// Nil checks
	var nilChecks []string
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			nilChecks = append(nilChecks, p.Name+" == nil")
		}
	}
	if len(nilChecks) > 0 {
		fmt.Fprintf(buf, "\tif %s {\n", strings.Join(nilChecks, " || "))
		fmt.Fprintf(buf, "\t\treturn\n")
		fmt.Fprintf(buf, "\t}\n")
	}

	// Discover unified struct layout from ALL parameters
	// The C code uses a single typedef, so all parameters must use the same layout
	elemCType := goElemTypeToCType(elemType)
	unifiedFields := DiscoverUnifiedStructFields(pf, elemCType)

	// Create C-compatible struct instances for each struct pointer param
	for _, p := range pf.Params {
		if !isGenericStructPtr(p.Type) {
			continue
		}

		if len(unifiedFields) == 0 {
			continue
		}

		// Emit struct definition using unified layout
		fmt.Fprintf(buf, "\tc%s := struct {\n", p.Name)
		for _, field := range unifiedFields {
			fmt.Fprintf(buf, "\t\t%s %s\n", field.Name, field.GoType)
		}
		fmt.Fprintf(buf, "\t}{\n")

		// Emit field initializers
		for _, field := range unifiedFields {
			if field.IsPtr {
				// For data pointer, get pointer to first element via Row(0)[0]
				fmt.Fprintf(buf, "\t\t%s: unsafe.Pointer(&%s.Row(0)[0]),\n", field.Name, p.Name)
			} else {
				fmt.Fprintf(buf, "\t\t%s: %s(%s%s),\n", field.Name, field.GoType, p.Name, field.GoGetter)
			}
		}
		fmt.Fprintf(buf, "\t}\n")
	}

	// Call assembly function with pointers to the C structs
	fmt.Fprintf(buf, "\t%s(\n", asmName)
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&c%s),\n", p.Name)
		}
	}
	fmt.Fprintf(buf, "\t)\n")
	fmt.Fprintf(buf, "}\n\n")
}

// specializeStructPtrType specializes a generic struct pointer type.
// E.g., *Image[T] with elemType=float32 becomes *Image[float32]
func specializeStructPtrType(typeStr, elemType string) string {
	// Replace T with the concrete type
	// This handles *Image[T] -> *Image[float32]
	if strings.Contains(typeStr, "[T]") {
		return strings.Replace(typeStr, "[T]", "["+goTypeName(elemType)+"]", 1)
	}
	return typeStr
}

// goTypeName returns the Go type name for an element type.
func goTypeName(elemType string) string {
	switch elemType {
	case "float16":
		return "hwy.Float16"
	case "bfloat16":
		return "hwy.BFloat16"
	default:
		return elemType
	}
}

// groupGoParams groups consecutive parameters with the same Go type for cleaner
// function signatures: "a, b, c []float32, m, n, k int" instead of
// "a []float32, b []float32, c []float32, m int, n int, k int".
func groupGoParams(pf *ParsedFunc, goSliceType, elemType string) string {
	// Build set of type parameter names for generic substitution.
	typeParamNames := make(map[string]bool, len(pf.TypeParams))
	for _, tp := range pf.TypeParams {
		typeParamNames[tp.Name] = true
	}

	var groups []string
	var currentNames []string
	var currentType string

	for _, p := range pf.Params {
		var paramType string
		if after, ok := strings.CutPrefix(p.Type, "[]"); ok {
			sliceElem := after
			if typeParamNames[sliceElem] {
				// Generic slice (e.g. []T) — substitute with profile type.
				paramType = goSliceType
			} else {
				// Concrete slice (e.g. []int8) — preserve actual type.
				paramType = p.Type
			}
		} else if p.Type == "int" || p.Type == "int64" {
			paramType = "int"
		} else if typeParamNames[p.Type] {
			paramType = astWrapperGoScalarType(elemType)
		} else {
			paramType = p.Type
		}

		if paramType == currentType {
			currentNames = append(currentNames, p.Name)
		} else {
			if len(currentNames) > 0 {
				groups = append(groups, strings.Join(currentNames, ", ")+" "+currentType)
			}
			currentNames = []string{p.Name}
			currentType = paramType
		}
	}
	if len(currentNames) > 0 {
		groups = append(groups, strings.Join(currentNames, ", ")+" "+currentType)
	}
	return strings.Join(groups, ", ")
}

// astWrapperGoScalarType maps an element type to the Go scalar type used in
// wrapper function signatures for by-value parameters of type T.
func astWrapperGoScalarType(elemType string) string {
	switch elemType {
	case "float32":
		return "float32"
	case "float64":
		return "float64"
	case "float16", "hwy.Float16":
		return "hwy.Float16"
	case "bfloat16", "hwy.BFloat16":
		return "hwy.BFloat16"
	case "int32":
		return "int32"
	case "int64":
		return "int64"
	case "uint32":
		return "uint32"
	case "uint64":
		return "uint64"
	default:
		return "float32"
	}
}

// emitBoundsChecks generates bounds checks for slice parameters based on the
// function's dimension parameters. This is a best-effort heuristic that covers
// common patterns like matmul (a[m*k], b[k*n], c[m*n]).
func emitBoundsChecks(buf *bytes.Buffer, pf *ParsedFunc, sliceParams, intParams []string) {
	// For matmul-like functions with 3 slices and 3 ints (m, n, k),
	// generate standard bounds checks.
	if len(sliceParams) == 3 && len(intParams) == 3 {
		m, n, k := intParams[0], intParams[1], intParams[2]
		fmt.Fprintf(buf, "\tif len(%s) < %s*%s || len(%s) < %s*%s || len(%s) < %s*%s {\n",
			sliceParams[0], m, k,
			sliceParams[1], k, n,
			sliceParams[2], m, n)
		fmt.Fprintf(buf, "\t\treturn\n")
		fmt.Fprintf(buf, "\t}\n")
	}
}
