package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"

	"github.com/ajroetker/go-highway/cmd/hwygen/ir"
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

	// Collect all eligible functions
	var vecFuncs []ParsedFunc   // Vec→Vec functions
	var sliceFuncs []ParsedFunc // Slice→Slice functions (composite like GELU)
	var astFuncs []ParsedFunc   // Functions for AST translation (matmul, transpose, etc.)

	for _, pf := range result.Funcs {
		if IsASTCEligible(&pf) {
			astFuncs = append(astFuncs, pf)
		} else if IsCEligible(&pf) {
			vecFuncs = append(vecFuncs, pf)
		} else if IsSliceFunction(&pf) {
			sliceFuncs = append(sliceFuncs, pf)
		}
	}

	totalFuncs := len(vecFuncs) + len(sliceFuncs) + len(astFuncs)
	if totalFuncs == 0 {
		return nil, fmt.Errorf("no C-eligible functions found")
	}

	fmt.Printf("Found %d C-eligible functions:\n", totalFuncs)
	for _, pf := range vecFuncs {
		fmt.Printf("  - %s (Vec→Vec)\n", pf.Name)
	}
	for _, pf := range sliceFuncs {
		fmt.Printf("  - %s (Slice→Slice)\n", pf.Name)
	}

	// If fusion mode is enabled, use IR-based generation for slice functions
	if g.FusionMode && len(sliceFuncs) > 0 {
		fmt.Println("\nUsing IR-based fusion optimization...")
		if err := g.runFusionCMode(result, sliceFuncs, targets); err != nil {
			return nil, fmt.Errorf("fusion mode: %w", err)
		}
		// Continue with non-slice functions using standard path
		sliceFuncs = nil
	}
	for _, pf := range astFuncs {
		fmt.Printf("  - %s (AST→C)\n", pf.Name)
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

		// Generate C code for Vec→Vec functions
		for _, pf := range vecFuncs {
			elemTypes := getCElemTypes(&pf)
			for _, elemType := range elemTypes {
				profile := GetCProfile(target.Name, elemType)
				if profile == nil {
					continue // No profile for this target/type combo
				}
				emitter := NewCEmitter(g.PackageOut, elemType, target)
				emitter.profile = profile
				cFile, err := emitter.EmitC(&pf, cOutputDir)
				if err != nil {
					return nil, fmt.Errorf("emit C for %s (%s, %s): %w", pf.Name, elemType, target.Name, err)
				}
				cFiles = append(cFiles, cFile)
			}
		}

		// Generate C code for Slice→Slice (composite) functions
		for _, pf := range sliceFuncs {
			elemTypes := getCElemTypes(&pf)
			for _, elemType := range elemTypes {
				profile := GetCProfile(target.Name, elemType)
				if profile == nil {
					continue // No profile for this target/type combo
				}
				emitter := NewCEmitter(g.PackageOut, elemType, target)
				emitter.profile = profile
				cFile, err := emitter.EmitCompositeC(&pf, cOutputDir)
				if err != nil {
					return nil, fmt.Errorf("emit composite C for %s (%s, %s): %w", pf.Name, elemType, target.Name, err)
				}
				cFiles = append(cFiles, cFile)
			}
		}

		// Generate C code for AST-translated functions (matmul, transpose, etc.)
		for _, pf := range astFuncs {
			elemTypes := getCElemTypes(&pf)
			for _, elemType := range elemTypes {
				profile := GetCProfile(target.Name, elemType)
				if profile == nil {
					continue
				}
				// AST translator emits native arithmetic — skip types that
				// require promoted math unless they have native SIMD arithmetic.
				if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
					continue
				}
				emitter := NewCEmitter(g.PackageOut, elemType, target)
				emitter.profile = profile
				cFile, err := emitter.EmitASTTranslatedC(&pf, cOutputDir)
				if err != nil {
					return nil, fmt.Errorf("emit AST C for %s (%s, %s): %w", pf.Name, elemType, target.Name, err)
				}
				cFiles = append(cFiles, cFile)
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

			// Clean up C and object files (Go build doesn't like them)
			for _, cFile := range cFiles {
				os.Remove(cFile)
				os.Remove(strings.TrimSuffix(cFile, ".c") + ".o")
			}

			// Separate struct-ptr and non-struct functions
			allFuncs := make([]ParsedFunc, 0, len(vecFuncs)+len(sliceFuncs)+len(astFuncs))
			allFuncs = append(allFuncs, vecFuncs...)
			allFuncs = append(allFuncs, sliceFuncs...)
			allFuncs = append(allFuncs, astFuncs...)

			var structFuncs, nonStructFuncs []ParsedFunc
			for _, f := range allFuncs {
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

			// Non-struct AST-translated functions (slice+int params) also need
			// ASM adapter + wiring.
			var nonStructASTFuncs []ParsedFunc
			for _, f := range nonStructFuncs {
				if IsASTCEligible(&f) {
					nonStructASTFuncs = append(nonStructASTFuncs, f)
				}
			}
			if len(nonStructASTFuncs) > 0 {
				if err := g.emitSliceAsmPassthrough(nonStructASTFuncs, target, cOutputDir); err != nil {
					return nil, fmt.Errorf("emit slice asm passthrough for %s: %w", target.Name, err)
				}
				if err := g.emitZCDispatchForSlices(nonStructASTFuncs, target); err != nil {
					return nil, fmt.Errorf("emit asm adapter for slices %s: %w", target.Name, err)
				}
			}

			// Collect adapter info for all ASM-eligible functions
			for _, pf := range append(structFuncs, nonStructASTFuncs...) {
				for _, elemType := range getCElemTypes(&pf) {
					profile := GetCProfile(target.Name, elemType)
					if profile == nil {
						continue
					}
					if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
						continue
					}
					allAdapters = append(allAdapters, AsmAdapterInfo{
						TargetName:  target.Name,
						Arch:        target.Arch(),
						DispatchVar: buildDispatchVarName(pf.Name, elemType),
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
// For non-generic functions, it infers the element type from slice parameters.
func getCElemTypes(pf *ParsedFunc) []string {
	if len(pf.TypeParams) > 0 {
		return GetConcreteTypes(pf.TypeParams[0].Constraint)
	}
	// For non-generic functions, detect element type from slice param types
	for _, p := range pf.Params {
		if after, ok := strings.CutPrefix(p.Type, "[]"); ok {
			elemType := after
			switch elemType {
			case "uint64":
				return []string{"uint64"}
			case "uint32":
				return []string{"uint32"}
			case "uint8", "byte":
				return []string{"uint8"}
			case "float64":
				return []string{"float64"}
			case "float32":
				return []string{"float32"}
			}
		}
	}
	return []string{"float32"}
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

// getCProfileForFile determines the CIntrinsicProfile for a generated C file
// based on its filename convention.
func getCProfileForFile(cFile string, target Target) *CIntrinsicProfile {
	base := filepath.Base(cFile)
	for _, et := range []string{
		"float16", "hwy.Float16", "bfloat16", "hwy.BFloat16",
		"float32", "float64",
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
		args = append(args, fmt.Sprintf("-e=--target=%s", profile.GoatTarget))
		for _, flag := range profile.GoatExtraFlags {
			args = append(args, "-e="+flag)
		}
	} else {
		// Fallback to arm64 NEON
		args = append(args,
			"-e=--target=arm64",
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
// It replaces patterns like "g+" (parameter references) and "g " (in func decls)
// with safe alternatives.
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
		// Replace parameter references in .s files: "g+8(FP)" → "gv+8(FP)"
		old := reserved + "+"
		new := replacement + "+"
		if strings.Contains(content, old) {
			content = strings.ReplaceAll(content, old, new)
			modified = true
		}
		// Replace parameter names in .go func declarations: ", g " or ", g," → ", gv " or ", gv,"
		old = ", " + reserved + " "
		new = ", " + replacement + " "
		if strings.Contains(content, old) {
			content = strings.ReplaceAll(content, old, new)
			modified = true
		}
		old = ", " + reserved + ","
		new = ", " + replacement + ","
		if strings.Contains(content, old) {
			content = strings.ReplaceAll(content, old, new)
			modified = true
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
					if profile != nil && (profile.MathStrategy != "promoted" || profile.NativeArithmetic) {
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
	filename := filepath.Join(outputDir, fmt.Sprintf("c_wrappers_%s_%s.gen.go", targetSuffix, archSuffix))
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

			// Build param list: one unsafe.Pointer per struct param
			var paramNames []string
			for _, p := range pf.Params {
				if isGenericStructPtr(p.Type) {
					paramNames = append(paramNames, p.Name)
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

	filename := filepath.Join(asmDir, fmt.Sprintf("c_struct_wrappers_%s_%s.gen.go", targetSuffix, archSuffix))
	if err := os.WriteFile(filename, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("write struct asm passthrough: %w", err)
	}
	fmt.Printf("Generated: %s\n", filename)
	return nil
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

	// Check if we need the hwy import for half-precision types
	needsHwy := false
	for _, pf := range funcs {
		for _, et := range getCElemTypes(&pf) {
			if isHalfPrecisionType(et) {
				profile := GetCProfile(target.Name, et)
				if profile != nil && (profile.MathStrategy != "promoted" || profile.NativeArithmetic) {
					needsHwy = true
					break
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
	fmt.Fprintf(&buf, "package %s\n\n", g.PackageOut)
	fmt.Fprintf(&buf, "import (\n")
	fmt.Fprintf(&buf, "\t\"unsafe\"\n\n")
	if needsHwy {
		fmt.Fprintf(&buf, "\t\"github.com/ajroetker/go-highway/hwy\"\n")
	}
	fmt.Fprintf(&buf, "\t\"%s\"\n", asmImport)
	fmt.Fprintf(&buf, ")\n\n")

	// init() to override dispatch variables
	fmt.Fprintf(&buf, "func init() {\n")
	for _, pf := range funcs {
		for _, elemType := range getCElemTypes(&pf) {
			profile := GetCProfile(target.Name, elemType)
			if profile == nil {
				continue
			}
			if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
				continue
			}
			dispatchVar := buildDispatchVarName(pf.Name, elemType)
			adapterName := buildAdapterFuncName(pf.Name, elemType)
			fmt.Fprintf(&buf, "\t%s = %s\n", dispatchVar, adapterName)
		}
	}
	fmt.Fprintf(&buf, "}\n\n")

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

	filename := filepath.Join(g.OutputDir, fmt.Sprintf("z_c_%s_%s.gen.go", targetSuffix, archSuffix))
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
			if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
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
				} else if p.Type == "int" || p.Type == "int64" {
					ptrName := "p" + p.Name
					paramDefs = append(paramDefs, ptrName+" unsafe.Pointer")
					paramNames = append(paramNames, ptrName)
				} else if p.Type == "T" {
					// By-value scalar param — use concrete type
					scalarType := slicePassthroughScalarType(elemType)
					paramDefs = append(paramDefs, p.Name+" "+scalarType)
					paramNames = append(paramNames, p.Name)
				}
			}

			fmt.Fprintf(&buf, "// %s calls the %s SIMD assembly implementation.\n",
				exportedName, strings.ToUpper(target.Name))
			fmt.Fprintf(&buf, "func %s(%s) {\n",
				exportedName, strings.Join(paramDefs, ", "))
			fmt.Fprintf(&buf, "\t%s(%s)\n", asmName, strings.Join(paramNames, ", "))
			fmt.Fprintf(&buf, "}\n\n")
		}
	}

	filename := filepath.Join(asmDir, fmt.Sprintf("c_slice_passthrough_%s_%s.gen.go", targetSuffix, archSuffix))
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

	// Check if we need the hwy import for half-precision types
	needsHwy := false
	for _, pf := range funcs {
		for _, et := range getCElemTypes(&pf) {
			if isHalfPrecisionType(et) {
				profile := GetCProfile(target.Name, et)
				if profile != nil && (profile.MathStrategy != "promoted" || profile.NativeArithmetic) {
					needsHwy = true
					break
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
	fmt.Fprintf(&buf, "package %s\n\n", g.PackageOut)
	fmt.Fprintf(&buf, "import (\n")
	fmt.Fprintf(&buf, "\t\"unsafe\"\n\n")
	if needsHwy {
		fmt.Fprintf(&buf, "\t\"github.com/ajroetker/go-highway/hwy\"\n")
	}
	fmt.Fprintf(&buf, "\t\"%s\"\n", asmImport)
	fmt.Fprintf(&buf, ")\n\n")

	// init() to override dispatch variables
	fmt.Fprintf(&buf, "func init() {\n")
	for _, pf := range funcs {
		for _, elemType := range getCElemTypes(&pf) {
			profile := GetCProfile(target.Name, elemType)
			if profile == nil {
				continue
			}
			if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
				continue
			}
			dispatchVar := buildDispatchVarName(pf.Name, elemType)
			adapterName := buildAdapterFuncName(pf.Name, elemType)
			fmt.Fprintf(&buf, "\t%s = %s\n", dispatchVar, adapterName)
		}
	}
	fmt.Fprintf(&buf, "}\n\n")

	// Adapter functions: Go slice+int params → unsafe.Pointer → asm call
	for _, pf := range funcs {
		for _, elemType := range getCElemTypes(&pf) {
			profile := GetCProfile(target.Name, elemType)
			if profile == nil {
				continue
			}
			if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
				continue
			}
			emitSliceZCAdapterFunc(&buf, &pf, elemType)
		}
	}

	filename := filepath.Join(g.OutputDir, fmt.Sprintf("z_c_slices_%s_%s.gen.go", targetSuffix, archSuffix))
	if err := os.WriteFile(filename, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("write z_c slice dispatch: %w", err)
	}
	fmt.Printf("Generated: %s\n", filename)
	return nil
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

	fmt.Fprintf(buf, "func %s(%s) {\n", adapterName, goSig)

	// Classify params.
	var sliceParams []string
	var intParams []string
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			sliceParams = append(sliceParams, p.Name)
		} else if p.Type == "int" || p.Type == "int64" {
			intParams = append(intParams, p.Name)
		}
	}

	// Guard against empty slices to prevent &slice[0] panic.
	// Int params are not guarded — the C code handles zero dimensions,
	// and some int params (e.g., phase) are legitimately 0.
	if len(sliceParams) > 0 {
		var checks []string
		for _, sp := range sliceParams {
			checks = append(checks, "len("+sp+") == 0")
		}
		fmt.Fprintf(buf, "\tif %s {\n", strings.Join(checks, " || "))
		fmt.Fprintf(buf, "\t\treturn\n")
		fmt.Fprintf(buf, "\t}\n")
	}

	// Convert int params to int64 for unsafe.Pointer
	for _, ip := range intParams {
		fmt.Fprintf(buf, "\t%sVal := int64(%s)\n", ip, ip)
	}

	// Call asm passthrough
	fmt.Fprintf(buf, "\tasm.%s(\n", asmExportedName)
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&%s[0]),\n", p.Name)
		} else if p.Type == "int" || p.Type == "int64" {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&%sVal),\n", p.Name)
		} else if p.Type == "T" {
			// By-value scalar param — cast half-precision types
			goScalarType := astWrapperGoScalarType(elemType)
			if goScalarType == "hwy.Float16" || goScalarType == "hwy.BFloat16" {
				fmt.Fprintf(buf, "\t\tuint16(%s),\n", p.Name)
			} else {
				fmt.Fprintf(buf, "\t\t%s,\n", p.Name)
			}
		}
	}
	fmt.Fprintf(buf, "\t)\n")
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
	return name + "_" + cTypePublicSuffix(elemType)
}

// buildDispatchVarName builds the dispatch variable name from a base function name.
// E.g., BaseForwardICT + float32 → ForwardICTFloat32
func buildDispatchVarName(baseName, elemType string) string {
	name := strings.TrimPrefix(baseName, "Base")
	return name + typeNameToDispatchSuffix(elemType)
}

// buildAdapterFuncName builds the unexported adapter function name.
// E.g., BaseForwardICT + float32 → forwardICTAsmF32
func buildAdapterFuncName(baseName, elemType string) string {
	name := strings.TrimPrefix(baseName, "Base")
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

	// Build parameter list with specialized types
	var params []string
	for _, p := range pf.Params {
		paramType := specializeStructPtrType(p.Type, elemType)
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

	// Call the asm/ exported function
	asmExportedName := structAsmExportedName(pf.Name, elemType)
	fmt.Fprintf(buf, "\tasm.%s(\n", asmExportedName)
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&c%s),\n", p.Name)
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
// BaseExpVec -> ExpCF32, BaseGELU -> GELUCF32
func buildCPublicName(baseName, elemType string) string {
	name := strings.TrimPrefix(baseName, "Base")
	name = strings.TrimSuffix(name, "Vec")

	typeSuffix := cTypePublicSuffix(elemType)

	return name + "C" + typeSuffix
}

// cAsmFuncName creates the assembly function name.
// BaseExpVec, float32, neon -> exp_c_f32_neon
func cAsmFuncName(baseName, elemType, targetSuffix string) string {
	name := strings.TrimPrefix(baseName, "Base")
	name = strings.TrimSuffix(name, "Vec")
	name = strings.ToLower(name)

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
func goReturnTypeForWrapper(goType string) string {
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
		} else if p.Type == "int" || p.Type == "int64" {
			intParams = append(intParams, p.Name)
		}
	}

	// Determine if we need a hidden length param (no explicit int params but has slices)
	needsHiddenLen := len(intParams) == 0 && len(sliceParams) > 0
	firstSlice := ""
	if needsHiddenLen && len(sliceParams) > 0 {
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
			returnType = goReturnTypeForWrapper(pf.Returns[0].Type)
		} else {
			var retTypes []string
			for _, ret := range pf.Returns {
				retTypes = append(retTypes, goReturnTypeForWrapper(ret.Type))
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
	if needsHiddenLen {
		fmt.Fprintf(buf, "\tif len(%s) == 0 {\n", firstSlice)
		if hasReturns {
			// Return zero values
			var zeros []string
			for range pf.Returns {
				zeros = append(zeros, "0")
			}
			fmt.Fprintf(buf, "\t\treturn %s\n", strings.Join(zeros, ", "))
		} else {
			fmt.Fprintf(buf, "\t\treturn\n")
		}
		fmt.Fprintf(buf, "\t}\n")
	}

	// Guard against empty slices to prevent &slice[0] panic.
	if len(sliceParams) > 0 {
		var checks []string
		for _, sp := range sliceParams {
			checks = append(checks, "len("+sp+") == 0")
		}
		fmt.Fprintf(buf, "\tif %s {\n", strings.Join(checks, " || "))
		if hasReturns {
			var zeros []string
			for range pf.Returns {
				zeros = append(zeros, "0")
			}
			fmt.Fprintf(buf, "\t\treturn %s\n", strings.Join(zeros, ", "))
		} else {
			fmt.Fprintf(buf, "\t\treturn\n")
		}
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
			for range pf.Returns {
				zeros = append(zeros, "0")
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

	// Hidden length param
	if needsHiddenLen {
		fmt.Fprintf(buf, "\tlenVal := int64(len(%s))\n", firstSlice)
	}

	// Output variables for return values.
	// GOAT always uses 64-bit (long *) for output pointers, so we declare
	// int64 variables and cast to the actual return type.
	if hasReturns {
		for _, ret := range pf.Returns {
			name := ret.Name
			if name == "" {
				name = "result"
			}
			fmt.Fprintf(buf, "\tvar out_%s int64\n", name)
		}
	}

	// Call assembly function
	fmt.Fprintf(buf, "\t%s(\n", asmName)

	// Regular params
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&%s[0]),\n", p.Name)
		} else if p.Type == "int" || p.Type == "int64" {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&%sVal),\n", p.Name)
		} else if p.Type == "T" {
			// By-value scalar element-type param (e.g., float16_t coeff).
			// Cast to underlying type for asm stub compatibility
			// (e.g., hwy.Float16 → uint16).
			goScalarType := astWrapperGoScalarType(elemType)
			if goScalarType == "hwy.Float16" || goScalarType == "hwy.BFloat16" {
				fmt.Fprintf(buf, "\t\tuint16(%s),\n", p.Name)
			} else {
				fmt.Fprintf(buf, "\t\t%s,\n", p.Name)
			}
		}
	}
	// Hidden length param
	if needsHiddenLen {
		fmt.Fprintf(buf, "\t\tunsafe.Pointer(&lenVal),\n")
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

	// Return with type narrowing from int64 to actual return type
	if hasReturns {
		var retExprs []string
		for _, ret := range pf.Returns {
			name := ret.Name
			if name == "" {
				name = "result"
			}
			goType := goReturnTypeForWrapper(ret.Type)
			if goType == "int64" || goType == "int" {
				retExprs = append(retExprs, goType+"(out_"+name+")")
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
	sort.Strings(names)
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
	var groups []string
	var currentNames []string
	var currentType string

	for _, p := range pf.Params {
		var paramType string
		if strings.HasPrefix(p.Type, "[]") {
			paramType = goSliceType
		} else if p.Type == "int" || p.Type == "int64" {
			paramType = "int"
		} else if p.Type == "T" {
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
