package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// targetPriority returns a priority value for SIMD targets.
// Higher values are checked first in dispatch code.
// This ensures AVX512 is checked before AVX2 (since AVX512 CPUs also have AVX2).
func targetPriority(name string) int {
	switch name {
	case "AVX512":
		return 3
	case "AVX2":
		return 2
	case "NEON":
		return 1
	default:
		return 0
	}
}

// ContribPackages tracks which contrib subpackages are needed for imports.
type ContribPackages struct {
	Math    bool // contrib/math (Exp, Log, Sin, etc.)
	Vec     bool // contrib/vec (Dot product, Norm, etc.)
	MatVec  bool // contrib/matvec (Matrix-vector ops)
	Matmul  bool // contrib/matmul (Matrix multiplication)
	Algo    bool // contrib/algo (Transform utilities)
	Image   bool // contrib/image (Image processing)
	Bitpack bool // contrib/bitpack (Bit packing)
	Sort    bool // contrib/sort (Sorting algorithms)
	HwyPkg  bool // hwy package functions (Pow2, etc.) used in SIMD targets
	StdMath bool // stdlib math package (math.Inf, math.NaN, etc.)
	HwyCore bool // hwy core ops (Load, Store, Add, etc.) that need vec package
}

// detectContribPackages analyzes parsed functions to determine which contrib subpackages are used.
// Returns the combined packages across all targets (for backward compatibility).
func detectContribPackages(funcs []ParsedFunc, targets []Target) ContribPackages {
	pkgs := ContribPackages{}

	for _, pf := range funcs {
		for _, call := range pf.HwyCalls {
			// Check each target's OpMap for this function
			for _, target := range targets {
				if opInfo, ok := target.OpMap[call.FuncName]; ok {
					switch opInfo.SubPackage {
					case "math":
						pkgs.Math = true
					case "vec":
						pkgs.Vec = true
					case "matvec":
						pkgs.MatVec = true
					case "matmul":
						pkgs.Matmul = true
					case "algo":
						pkgs.Algo = true
					case "image":
						pkgs.Image = true
					case "bitpack":
						pkgs.Bitpack = true
					case "sort":
						pkgs.Sort = true
					}
					// HwyPkg is set per-target in detectContribPackagesForTarget
				}
			}
		}
	}

	return pkgs
}

// detectContribPackagesForTarget analyzes parsed functions to determine packages used for a specific target.
func detectContribPackagesForTarget(funcs []ParsedFunc, target Target) ContribPackages {
	pkgs := ContribPackages{}

	// Known stdlib math functions that might be called
	stdMathFuncs := map[string]bool{
		"Inf": true, "NaN": true, "IsInf": true, "IsNaN": true,
		"Floor": true, "Ceil": true, "Round": true, "Trunc": true,
		"Sqrt": true, "Cbrt": true, "Abs": true,
		"Sin": true, "Cos": true, "Tan": true,
		"Asin": true, "Acos": true, "Atan": true, "Atan2": true,
		"Sinh": true, "Cosh": true, "Tanh": true,
		"Asinh": true, "Acosh": true, "Atanh": true,
		"Exp": true, "Exp2": true, "Expm1": true,
		"Log": true, "Log2": true, "Log10": true, "Log1p": true,
		"Pow": true, "Hypot": true, "Erf": true, "Erfc": true,
		"Copysign": true, "Signbit": true, "Max": true, "Min": true,
	}

	for _, pf := range funcs {
		for _, call := range pf.HwyCalls {
			// Check for stdmath FIRST - it's always stdlib, never in OpMap
			if call.Package == "stdmath" {
				pkgs.StdMath = true
				continue
			}
			// Check for stdlib math package calls BEFORE OpMap lookup
			// This is needed because OpMap might have entries like "Sqrt" for hwy.Sqrt,
			// but we need to distinguish math.Sqrt (stdlib) from hwy.Sqrt
			if call.Package == "math" {
				if stdMathFuncs[call.FuncName] {
					// This is a stdlib math function call (math.Sqrt, math.Inf, etc.)
					pkgs.StdMath = true
					continue
				} else if strings.HasPrefix(call.FuncName, "Base") {
					// This is a contrib math function reference (like math.BaseExpVec)
					pkgs.Math = true
					continue
				}
			}
			if opInfo, ok := target.OpMap[call.FuncName]; ok {
				switch opInfo.SubPackage {
				case "math":
					pkgs.Math = true
				case "vec":
					pkgs.Vec = true
				case "matvec":
					pkgs.MatVec = true
				case "matmul":
					pkgs.Matmul = true
				case "algo":
					pkgs.Algo = true
				case "image":
					pkgs.Image = true
				case "bitpack":
					pkgs.Bitpack = true
				case "sort":
					pkgs.Sort = true
				}
				// Check if this is a hwy package function (like RoundToEven for AVX512) for this target
				if opInfo.Package == "hwy" && !opInfo.IsMethod && target.Name != "Fallback" {
					pkgs.HwyPkg = true
				}
				// Track if core hwy operations are used (Load, Store, Add, etc.)
				if call.Package == "hwy" {
					pkgs.HwyCore = true
				}
			} else if call.Package == "algo" && strings.HasPrefix(call.FuncName, "Base") {
				// This is a contrib algo function reference (like algo.BaseApply)
				pkgs.Algo = true
			} else if call.Package == "hwy" {
				// Other hwy package references (operations not in OpMap)
				pkgs.HwyCore = true
				// For non-Fallback targets, we need hwy import for unmapped operations
				if target.Name != "Fallback" {
					pkgs.HwyPkg = true
				}
			}
		}

		// Check if this function will generate Float16/BFloat16 specializations.
		// For SIMD targets: need hwy package for hwy.Vec[hwy.Float16] since archsimd doesn't have native half-precision support.
		// For Fallback targets: need hwy package for the hwy.Float16/hwy.BFloat16 types themselves.
		if len(pf.TypeParams) > 0 {
			concreteTypes := GetConcreteTypes(pf.TypeParams[0].Constraint)
			for _, ct := range concreteTypes {
				if ct == "hwy.Float16" || ct == "hwy.BFloat16" {
					if target.Name != "Fallback" {
						pkgs.HwyPkg = true
					} else {
						// Fallback needs hwy import for Float16/BFloat16 types
						pkgs.HwyCore = true
					}
					break
				}
			}
		}
	}

	return pkgs
}

// deriveDispatchPrefix extracts a unique prefix from function names for dispatch file naming.
// E.g., "BaseBlockedMatMul" → "blockedmatmul", "BaseMatMul" → "matmul"
func deriveDispatchPrefix(funcs []ParsedFunc) string {
	if len(funcs) == 0 {
		return ""
	}
	// Use the first function's name (without "Base" prefix)
	name := funcs[0].Name
	if strings.HasPrefix(name, "Base") {
		name = name[4:]
	}
	return strings.ToLower(name)
}

// EmitDispatcher generates the runtime dispatch file(s).
// This generates architecture-specific dispatch files:
// - dispatch_{prefix}_amd64.gen.go for AVX2/AVX512
// - dispatch_{prefix}_arm64.gen.go for NEON
// - dispatch_{prefix}.gen.go for fallback-only (no build tags)
// If dispatchName is empty, derives prefix from function names.
func EmitDispatcher(funcs []ParsedFunc, targets []Target, pkgName, outPath, dispatchName string) error {
	// Use provided dispatch name or derive from function names
	prefix := dispatchName
	if prefix == "" {
		prefix = deriveDispatchPrefix(funcs)
	}

	// Group targets by architecture
	amd64Targets := []Target{}
	arm64Targets := []Target{}
	hasFallback := false

	for _, target := range targets {
		switch target.Arch() {
		case "amd64":
			amd64Targets = append(amd64Targets, target)
		case "arm64":
			arm64Targets = append(arm64Targets, target)
		default:
			if target.Name == "Fallback" {
				hasFallback = true
			}
		}
	}

	// Generate amd64 dispatch if we have amd64 targets
	if len(amd64Targets) > 0 {
		if err := emitArchDispatcher(funcs, amd64Targets, hasFallback, pkgName, outPath, "amd64", prefix); err != nil {
			return err
		}
	}

	// Generate arm64 dispatch if we have arm64 targets
	if len(arm64Targets) > 0 {
		if err := emitArchDispatcher(funcs, arm64Targets, hasFallback, pkgName, outPath, "arm64", prefix); err != nil {
			return err
		}
	}

	// Generate universal fallback dispatch if Fallback target is present
	// This covers architectures not handled above (e.g. 386, riscv64)
	// AND amd64/arm64 builds where SIMD is disabled or not supported by build tags
	if hasFallback {
		var constraints []string
		if len(arm64Targets) > 0 {
			constraints = append(constraints, "!arm64")
		}
		// Note: amd64 dispatch requires goexperiment.simd, so fallback handles !(amd64 && simd)
		if len(amd64Targets) > 0 {
			constraints = append(constraints, "!(amd64 && goexperiment.simd)")
		}

		buildTag := strings.Join(constraints, " && ")
		
		// Use "other" suffix if we have constraints, to avoid conflict with arch-specific files
		suffix := ""
		if buildTag != "" {
			suffix = "_other"
		}

		if err := emitFallbackOnlyDispatcher(funcs, pkgName, outPath, prefix, suffix, buildTag); err != nil {
			return err
		}
	}

	return nil
}

// hasVecInSignature checks if a function has hwy.Vec anywhere in its parameters or returns.
// This includes:
// - Direct Vec params: v hwy.Vec[T]
// - Function params containing Vec: fn func(hwy.Vec[T]) hwy.Vec[T]
// - Return types containing Vec
//
// Functions with Vec in their signature cannot have dispatch generated because:
// - The concrete Vec type differs per architecture (archsimd.Float32x8 vs asm.Float32x4)
// - Function type parameters containing Vec are especially problematic since you can't
//   assign func(archsimd.Float32x8) to func(hwy.Vec[float32])
func hasVecInSignature(pf ParsedFunc) bool {
	for _, param := range pf.Params {
		if strings.Contains(param.Type, "hwy.Vec[") || strings.Contains(param.Type, "Vec[") {
			return true
		}
	}
	for _, ret := range pf.Returns {
		if strings.Contains(ret.Type, "hwy.Vec[") || strings.Contains(ret.Type, "Vec[") {
			return true
		}
	}
	return false
}

// filterDispatchableFuncs returns only functions that should have dispatch generated.
// Functions with Vec anywhere in their signature are excluded because:
// - Vec→Vec functions have dispatch at the Transform layer
// - Function type parameters with Vec can't be unified across architectures
func filterDispatchableFuncs(funcs []ParsedFunc) []ParsedFunc {
	var result []ParsedFunc
	for _, pf := range funcs {
		if !hasVecInSignature(pf) {
			result = append(result, pf)
		}
	}
	return result
}

// emitArchDispatcher generates an architecture-specific dispatch file.
func emitArchDispatcher(funcs []ParsedFunc, archTargets []Target, hasFallback bool, pkgName, outPath, arch, prefix string) error {
	// Filter out Vec→Vec functions - they don't need dispatch
	dispatchableFuncs := filterDispatchableFuncs(funcs)
	if len(dispatchableFuncs) == 0 {
		// No dispatchable functions, skip dispatch file generation
		return nil
	}

	var buf bytes.Buffer

	// Determine build tag based on architecture
	// amd64 requires goexperiment.simd for archsimd
	// arm64 uses our asm package which doesn't require the SIMD experiment
	var buildTag string
	if arch == "amd64" {
		buildTag = arch + " && goexperiment.simd"
	} else {
		buildTag = arch
	}

	// File header with build tag
	fmt.Fprintf(&buf, "// Code generated by hwygen. DO NOT EDIT.\n")
	fmt.Fprintf(&buf, "//go:build %s\n", buildTag)
	fmt.Fprintf(&buf, "\npackage %s\n\n", pkgName)

	// Check if any function has type params (needs hwy import for generic dispatcher)
	hasGenerics := false
	for _, pf := range dispatchableFuncs {
		if len(pf.TypeParams) > 0 {
			hasGenerics = true
			break
		}
	}

	// Imports - amd64 needs archsimd, arm64 doesn't
	// hwy needed for generic dispatcher type constraint
	fmt.Fprintf(&buf, "import (\n")
	fmt.Fprintf(&buf, "\t\"os\"\n")
	if hasGenerics {
		fmt.Fprintf(&buf, "\n\t\"github.com/ajroetker/go-highway/hwy\"\n")
	}
	if arch == "amd64" {
		fmt.Fprintf(&buf, "\t\"simd/archsimd\"\n")
	}
	fmt.Fprintf(&buf, ")\n\n")

	// Declare function variables
	for _, pf := range dispatchableFuncs {
		isGeneric := len(pf.TypeParams) > 0
		var concreteTypes []string
		if isGeneric {
			concreteTypes = GetConcreteTypes(pf.TypeParams[0].Constraint)
		} else {
			concreteTypes = []string{"float32"}
		}

		for _, elemType := range concreteTypes {
			funcName := buildDispatchFuncName(pf.Name, elemType, isGeneric)
			signature := buildFuncSignature(pf, elemType)
			fmt.Fprintf(&buf, "var %s func%s\n", funcName, signature)
		}
	}
	fmt.Fprintf(&buf, "\n")

	// Generate generic dispatcher functions for generic source functions
	for _, pf := range dispatchableFuncs {
		if len(pf.TypeParams) > 0 {
			emitGenericDispatcher(&buf, pf)
		}
	}

	// Capitalize prefix for function names (e.g., "matmul" -> "Matmul")
	capPrefix := strings.Title(prefix)

	// Generate init() function
	fmt.Fprintf(&buf, "func init() {\n")
	fmt.Fprintf(&buf, "\tif os.Getenv(\"HWY_NO_SIMD\") != \"\" {\n")
	fmt.Fprintf(&buf, "\t\tinit%sFallback()\n", capPrefix)
	fmt.Fprintf(&buf, "\t\treturn\n")
	fmt.Fprintf(&buf, "\t}\n")

	// Add CPU detection for each target.
	// IMPORTANT: Sort targets so more capable SIMD is checked first.
	// AVX512 CPUs also have AVX2, so we must check AVX512 before AVX2.
	sortedTargets := make([]Target, len(archTargets))
	copy(sortedTargets, archTargets)
	sort.Slice(sortedTargets, func(i, j int) bool {
		return targetPriority(sortedTargets[i].Name) > targetPriority(sortedTargets[j].Name)
	})

	for _, target := range sortedTargets {
		switch target.Name {
		case "AVX512":
			fmt.Fprintf(&buf, "\tif archsimd.X86.AVX512() {\n")
			fmt.Fprintf(&buf, "\t\tinit%sAVX512()\n", capPrefix)
			fmt.Fprintf(&buf, "\t\treturn\n")
			fmt.Fprintf(&buf, "\t}\n")
		case "AVX2":
			fmt.Fprintf(&buf, "\tif archsimd.X86.AVX2() {\n")
			fmt.Fprintf(&buf, "\t\tinit%sAVX2()\n", capPrefix)
			fmt.Fprintf(&buf, "\t\treturn\n")
			fmt.Fprintf(&buf, "\t}\n")
		case "NEON":
			// Check if hwy actually detected NEON (lanes >= 4 for float32).
			// On emulators or fallback environments, hwy may report scalar mode.
			fmt.Fprintf(&buf, "\t// Check if hwy actually detected NEON (lanes >= 4 for float32).\n")
			fmt.Fprintf(&buf, "\t// On emulators or fallback environments, hwy may report scalar mode.\n")
			fmt.Fprintf(&buf, "\tlanes := hwy.Zero[float32]().NumLanes()\n")
			fmt.Fprintf(&buf, "\tif lanes < 4 {\n")
			fmt.Fprintf(&buf, "\t\tinit%sFallback()\n", capPrefix)
			fmt.Fprintf(&buf, "\t\treturn\n")
			fmt.Fprintf(&buf, "\t}\n")
			fmt.Fprintf(&buf, "\tinit%sNEON()\n", capPrefix)
			fmt.Fprintf(&buf, "\treturn\n")
		}
	}

	// Add fallback at end for x86 (in case no SIMD detected)
	// ARM64 handles fallback in the NEON case above
	if arch != "arm64" && hasFallback {
		fmt.Fprintf(&buf, "\tinit%sFallback()\n", capPrefix)
	}
	fmt.Fprintf(&buf, "}\n\n")

	// Generate init functions for each target
	for _, target := range archTargets {
		initFuncName := "init" + capPrefix + target.Name
		fmt.Fprintf(&buf, "func %s() {\n", initFuncName)

		for _, pf := range dispatchableFuncs {
			isGeneric := len(pf.TypeParams) > 0
			var concreteTypes []string
			if isGeneric {
				concreteTypes = GetConcreteTypes(pf.TypeParams[0].Constraint)
			} else {
				concreteTypes = []string{"float32"}
			}

			for _, elemType := range concreteTypes {
				dispatchName := buildDispatchFuncName(pf.Name, elemType, isGeneric)
				// Functions with interface type parameters only have fallback implementations
				// so use fallback even for SIMD targets
				var implName string
				if hasInterfaceTypeParams(pf.TypeParams) {
					implName = pf.Name + "_fallback"
				} else {
					implName = pf.Name + target.Suffix()
				}
				if elemType != "float32" && isGeneric {
					implName = implName + "_" + typeNameToSuffix(elemType)
				}
				fmt.Fprintf(&buf, "\t%s = %s\n", dispatchName, implName)
			}
		}

		fmt.Fprintf(&buf, "}\n\n")
	}

	// Generate fallback init function if needed
	if hasFallback {
		fmt.Fprintf(&buf, "func init%sFallback() {\n", capPrefix)
		for _, pf := range dispatchableFuncs {
			isGeneric := len(pf.TypeParams) > 0
			var concreteTypes []string
			if isGeneric {
				concreteTypes = GetConcreteTypes(pf.TypeParams[0].Constraint)
			} else {
				concreteTypes = []string{"float32"}
			}

			for _, elemType := range concreteTypes {
				dispatchName := buildDispatchFuncName(pf.Name, elemType, isGeneric)
				implName := pf.Name + "_fallback"
				if elemType != "float32" && isGeneric {
					implName = implName + "_" + typeNameToSuffix(elemType)
				}
				fmt.Fprintf(&buf, "\t%s = %s\n", dispatchName, implName)
			}
		}
		fmt.Fprintf(&buf, "}\n")
	}

	// Format the code
	formatted, err := format.Source(buf.Bytes())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: formatting failed: %v\n", err)
		formatted = buf.Bytes()
	}

	// Write to file with prefix and architecture suffix
	filename := filepath.Join(outPath, fmt.Sprintf("dispatch_%s_%s.gen.go", prefix, arch))
	if err := os.WriteFile(filename, formatted, 0644); err != nil {
		return fmt.Errorf("write dispatcher: %w", err)
	}

	return nil
}

// emitFallbackOnlyDispatcher generates a dispatch file with no build tags for fallback-only builds.
func emitFallbackOnlyDispatcher(funcs []ParsedFunc, pkgName, outPath, prefix, suffix, buildTag string) error {
	// Filter out Vec→Vec functions - they don't need dispatch
	dispatchableFuncs := filterDispatchableFuncs(funcs)
	if len(dispatchableFuncs) == 0 {
		// No dispatchable functions, skip dispatch file generation
		return nil
	}

	var buf bytes.Buffer

	// Check if any function has type params (needs hwy import for generic dispatcher)
	hasGenerics := false
	for _, pf := range dispatchableFuncs {
		if len(pf.TypeParams) > 0 {
			hasGenerics = true
			break
		}
	}

	fmt.Fprintf(&buf, "// Code generated by hwygen. DO NOT EDIT.\n")
	if buildTag != "" {
		fmt.Fprintf(&buf, "//go:build %s\n", buildTag)
	}
	fmt.Fprintf(&buf, "\npackage %s\n\n", pkgName)

	fmt.Fprintf(&buf, "import (\n")
	fmt.Fprintf(&buf, "\t\"os\"\n")
	if hasGenerics {
		fmt.Fprintf(&buf, "\n\t\"github.com/ajroetker/go-highway/hwy\"\n")
	}
	fmt.Fprintf(&buf, ")\n\n")

	// Declare function variables
	for _, pf := range dispatchableFuncs {
		isGeneric := len(pf.TypeParams) > 0
		var concreteTypes []string
		if isGeneric {
			concreteTypes = GetConcreteTypes(pf.TypeParams[0].Constraint)
		} else {
			concreteTypes = []string{"float32"}
		}

		for _, elemType := range concreteTypes {
			funcName := buildDispatchFuncName(pf.Name, elemType, isGeneric)
			signature := buildFuncSignature(pf, elemType)
			fmt.Fprintf(&buf, "var %s func%s\n", funcName, signature)
		}
	}
	fmt.Fprintf(&buf, "\n")

	// Generate generic dispatcher functions for generic source functions
	for _, pf := range dispatchableFuncs {
		if len(pf.TypeParams) > 0 {
			emitGenericDispatcher(&buf, pf)
		}
	}

	// Capitalize prefix for function names
	capPrefix := strings.Title(prefix)

	// Simple init that just uses fallback
	fmt.Fprintf(&buf, "func init() {\n")
	fmt.Fprintf(&buf, "\t_ = os.Getenv // silence unused import\n")
	fmt.Fprintf(&buf, "\tinit%sFallback()\n", capPrefix)
	fmt.Fprintf(&buf, "}\n\n")

	fmt.Fprintf(&buf, "func init%sFallback() {\n", capPrefix)
	for _, pf := range dispatchableFuncs {
		isGeneric := len(pf.TypeParams) > 0
		var concreteTypes []string
		if isGeneric {
			concreteTypes = GetConcreteTypes(pf.TypeParams[0].Constraint)
		} else {
			concreteTypes = []string{"float32"}
		}

		for _, elemType := range concreteTypes {
			dispatchName := buildDispatchFuncName(pf.Name, elemType, isGeneric)
			implName := pf.Name + "_fallback"
			if elemType != "float32" && isGeneric {
				implName = implName + "_" + typeNameToSuffix(elemType)
			}
			fmt.Fprintf(&buf, "\t%s = %s\n", dispatchName, implName)
		}
	}
	fmt.Fprintf(&buf, "}\n")

	formatted, err := format.Source(buf.Bytes())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: formatting failed: %v\n", err)
		formatted = buf.Bytes()
	}

	filename := filepath.Join(outPath, fmt.Sprintf("dispatch_%s%s.gen.go", prefix, suffix))
	if err := os.WriteFile(filename, formatted, 0644); err != nil {
		return fmt.Errorf("write dispatcher: %w", err)
	}

	return nil
}

// EmitTarget generates a target-specific implementation file.
// sourceImports contains the imports from the original source file that should be preserved
// if they're still used after transformation (e.g., "unsafe", "math/bits").
func EmitTarget(funcs []*ast.FuncDecl, target Target, pkgName, baseName, outPath string, contribPkgs ContribPackages, hoistedConsts []HoistedConst, sourceImports map[string]string) error {
	var buf bytes.Buffer

	// File header
	fmt.Fprintf(&buf, "// Code generated by hwygen. DO NOT EDIT.\n")
	if target.BuildTag != "" {
		fmt.Fprintf(&buf, "//go:build %s\n", target.BuildTag)
	}
	fmt.Fprintf(&buf, "\npackage %s\n\n", pkgName)

	// Build import list
	imports := []string{}

	if target.Name != "Fallback" {
		// Import the appropriate vector package only if core hwy ops are used
		if contribPkgs.HwyCore {
			switch target.VecPackage {
			case "archsimd":
				imports = append(imports, `"simd/archsimd"`)
			case "asm":
				imports = append(imports, `"github.com/ajroetker/go-highway/hwy/asm"`)
			}
		}
		// Add sync import for AVX-512 lazy initialization of hoisted constants
		if target.Name == "AVX512" && len(hoistedConsts) > 0 {
			imports = append(imports, `"sync"`)
		}
		// Add hwy package import if hwy functions are used (e.g., Pow2)
		if contribPkgs.HwyPkg {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy"`)
		}
		// Add contrib subpackage imports for SIMD targets
		if contribPkgs.Math {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/math"`)
		}
		if contribPkgs.Vec {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/vec"`)
		}
		if contribPkgs.MatVec {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/matvec"`)
		}
		if contribPkgs.Matmul {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/matmul"`)
		}
		if contribPkgs.Algo {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/algo"`)
		}
		if contribPkgs.Image {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/image"`)
		}
		if contribPkgs.Bitpack {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/bitpack"`)
		}
		if contribPkgs.Sort {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/sort"`)
		}
		// stdmath only if explicitly needed (math.Inf, math.NaN, etc. were found)
		if contribPkgs.StdMath {
			imports = append(imports, `stdmath "math"`)
		}
	} else {
		// Fallback uses the hwy package directly for core ops only if core ops are used
		if contribPkgs.HwyCore {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy"`)
		}
		// Fallback also uses contrib subpackages for their portable generic implementations
		if contribPkgs.Math {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/math"`)
		}
		if contribPkgs.Vec {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/vec"`)
		}
		if contribPkgs.MatVec {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/matvec"`)
		}
		if contribPkgs.Matmul {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/matmul"`)
		}
		if contribPkgs.Algo {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/algo"`)
		}
		if contribPkgs.Image {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/image"`)
		}
		if contribPkgs.Bitpack {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/bitpack"`)
		}
		if contribPkgs.Sort {
			imports = append(imports, `"github.com/ajroetker/go-highway/hwy/contrib/sort"`)
		}
		// Include stdmath if the source file uses stdlib math functions
		if contribPkgs.StdMath {
			imports = append(imports, `stdmath "math"`)
		}
	}

	// Add source imports that are still used after transformation
	// Walk the AST to find which packages are actually referenced
	usedPkgs := collectUsedPackages(funcs)

	// These are stdlib packages that don't get transformed (unlike hwy -> asm)
	preservedImports := map[string]bool{
		"unsafe":          true,
		"math/bits":       true,
		"encoding/binary": true,
	}
	for localName, importPath := range sourceImports {
		if preservedImports[importPath] && usedPkgs[localName] {
			if localName == importPath || localName == "" || localName == filepath.Base(importPath) {
				imports = append(imports, fmt.Sprintf(`"%s"`, importPath))
			} else {
				imports = append(imports, fmt.Sprintf(`%s "%s"`, localName, importPath))
			}
		}
	}

	// Sort imports for consistency
	sort.Strings(imports)

	// Write imports
	fmt.Fprintf(&buf, "import (\n")
	for _, imp := range imports {
		fmt.Fprintf(&buf, "\t%s\n", imp)
	}
	fmt.Fprintf(&buf, ")\n\n")

	// Emit hoisted constants as package-level pre-broadcasted vectors
	if len(hoistedConsts) > 0 && target.Name != "Fallback" {
		emitHoistedConstants(&buf, hoistedConsts, target, baseName)
	}

	// Print each function
	fset := token.NewFileSet()
	for _, funcDecl := range funcs {
		// For AVX-512 with hoisted constants, inject lazy init call at function start
		if target.Name == "AVX512" && len(hoistedConsts) > 0 && funcDecl.Body != nil {
			injectHoistedConstInit(funcDecl, baseName)
		}
		if err := printer.Fprint(&buf, fset, funcDecl); err != nil {
			return fmt.Errorf("print function: %w", err)
		}
		fmt.Fprintf(&buf, "\n\n")
	}

	// Format the code
	formatted, err := format.Source(buf.Bytes())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: formatting failed for %s: %v\n", target.Name, err)
		formatted = buf.Bytes()
	}

	// Determine output filename
	filename := filepath.Join(outPath, baseName+target.Suffix()+".gen.go")

	// Write to file
	if err := os.WriteFile(filename, formatted, 0644); err != nil {
		return fmt.Errorf("write target file: %w", err)
	}

	return nil
}

// emitHoistedConstants writes package-level var declarations for hoisted vector constants.
// For AVX-512, uses lazy initialization with sync.Once to avoid executing AVX-512 instructions
// at package init time (which would crash on machines without AVX-512 support).
// The baseName parameter is used to generate unique names for _hoistOnce and _initHoistedConstants
// to avoid conflicts when multiple source files have hoisted constants.
func emitHoistedConstants(buf *bytes.Buffer, consts []HoistedConst, target Target, baseName string) {
	if target.Name == "AVX512" {
		// Generate unique prefix from baseName (e.g., "varint_base" -> "_varintBase")
		prefix := "_" + toCamelCase(baseName)
		onceName := prefix + "HoistOnce"
		initName := prefix + "InitHoistedConstants"

		// AVX-512: Use lazy initialization to avoid init-time crashes on non-AVX512 machines
		fmt.Fprintf(buf, "// Hoisted constants - lazily initialized on first use to avoid init-time crashes\n")
		fmt.Fprintf(buf, "var (\n")
		for _, c := range consts {
			// Extract the vector type from the broadcast function (e.g., "archsimd.BroadcastFloat32x16" -> "archsimd.Float32x16")
			vecType := strings.Replace(c.Broadcast, "Broadcast", "", 1)
			fmt.Fprintf(buf, "\t%s %s\n", c.VarName, vecType)
		}
		fmt.Fprintf(buf, "\t%s sync.Once\n", onceName)
		fmt.Fprintf(buf, ")\n\n")

		// Generate the lazy init function
		fmt.Fprintf(buf, "func %s() {\n", initName)
		fmt.Fprintf(buf, "\t%s.Do(func() {\n", onceName)
		for _, c := range consts {
			fmt.Fprintf(buf, "\t\t%s = %s(%s)\n", c.VarName, c.Broadcast, c.Value)
		}
		fmt.Fprintf(buf, "\t})\n")
		fmt.Fprintf(buf, "}\n\n")
	} else {
		// Other targets: Initialize at package init time (safe for widely-supported instruction sets)
		fmt.Fprintf(buf, "// Hoisted constants - pre-broadcasted at package init time\n")
		fmt.Fprintf(buf, "var (\n")
		for _, c := range consts {
			// e.g., var BaseSigmoid_one_f32 = archsimd.BroadcastFloat32x8(1.0)
			fmt.Fprintf(buf, "\t%s = %s(%s)\n", c.VarName, c.Broadcast, c.Value)
		}
		fmt.Fprintf(buf, ")\n\n")
	}
}

// injectHoistedConstInit prepends a call to the init function at the start of a function.
// This is used for AVX-512 to ensure lazy initialization of hoisted constants.
// The baseName parameter determines the unique init function name.
func injectHoistedConstInit(funcDecl *ast.FuncDecl, baseName string) {
	if funcDecl.Body == nil {
		return
	}

	// Generate the same init function name used by emitHoistedConstants
	prefix := "_" + toCamelCase(baseName)
	initName := prefix + "InitHoistedConstants"

	// Create the call expression
	initCall := &ast.ExprStmt{
		X: &ast.CallExpr{
			Fun:  &ast.Ident{Name: initName},
			Args: nil,
		},
	}

	// Prepend to function body
	funcDecl.Body.List = append([]ast.Stmt{initCall}, funcDecl.Body.List...)
}

// toCamelCase converts a snake_case string to camelCase.
// E.g., "varint_base" -> "varintBase", "maskedvbyte_base" -> "maskedvbyteBase"
func toCamelCase(s string) string {
	parts := strings.Split(s, "_")
	for i := 1; i < len(parts); i++ {
		if len(parts[i]) > 0 {
			parts[i] = strings.ToUpper(parts[i][:1]) + parts[i][1:]
		}
	}
	return strings.Join(parts, "")
}

// emitGenericDispatcher generates a generic function that dispatches based on type.
// For a function like BaseMatMul[T hwy.Floats], this generates:
//
//	func MatMul[T hwy.Floats](a, b, c []T, m, n, k int) {
//	    switch any(a).(type) {
//	    case []float32:
//	        MatMulFloat32(any(a).([]float32), any(b).([]float32), any(c).([]float32), m, n, k)
//	    case []float64:
//	        MatMulFloat64(any(a).([]float64), any(b).([]float64), any(c).([]float64), m, n, k)
//	    }
//	}
func emitGenericDispatcher(buf *bytes.Buffer, pf ParsedFunc) {
	hasInterfaceParams := hasInterfaceTypeParams(pf.TypeParams)
	genericName := buildGenericFuncName(pf.Name, hasInterfaceParams)
	concreteTypes := GetConcreteTypes(pf.TypeParams[0].Constraint)

	// Function signature with type parameters
	fmt.Fprintf(buf, "// %s is the generic API that dispatches to the appropriate SIMD implementation.\n", genericName)
	fmt.Fprintf(buf, "func %s[", genericName)

	// Build type parameter list - include all type parameters
	for i, tp := range pf.TypeParams {
		if i > 0 {
			fmt.Fprintf(buf, ", ")
		}
		fmt.Fprintf(buf, "%s %s", tp.Name, tp.Constraint)
	}
	fmt.Fprintf(buf, "](")

	// Parameters
	for i, param := range pf.Params {
		if i > 0 {
			fmt.Fprintf(buf, ", ")
		}
		fmt.Fprintf(buf, "%s %s", param.Name, param.Type)
	}
	fmt.Fprintf(buf, ")")

	// Return type if any
	if len(pf.Returns) > 0 {
		if len(pf.Returns) == 1 && pf.Returns[0].Name == "" {
			fmt.Fprintf(buf, " %s", pf.Returns[0].Type)
		} else {
			fmt.Fprintf(buf, " (")
			for i, ret := range pf.Returns {
				if i > 0 {
					fmt.Fprintf(buf, ", ")
				}
				if ret.Name != "" {
					fmt.Fprintf(buf, "%s ", ret.Name)
				}
				fmt.Fprintf(buf, "%s", ret.Type)
			}
			fmt.Fprintf(buf, ")")
		}
	}

	fmt.Fprintf(buf, " {\n")

	// Find the first parameter with a generic type to use for type switch
	var switchParam string
	var switchParamType string
	for _, param := range pf.Params {
		// Check if this parameter contains a type parameter
		if containsTypeParam(param.Type, pf.TypeParams) {
			switchParam = param.Name
			switchParamType = param.Type
			break
		}
	}

	if switchParam == "" {
		// No generic parameter found, use first parameter
		switchParam = pf.Params[0].Name
		switchParamType = pf.Params[0].Type
	}

	// Generate type switch
	fmt.Fprintf(buf, "\tswitch any(%s).(type) {\n", switchParam)

	for _, elemType := range concreteTypes {
		// emitGenericDispatcher is only called for generic functions, so isGeneric is always true
		dispatchName := buildDispatchFuncName(pf.Name, elemType, true)
		// Use specializeType to get the concrete type for the case statement
		caseType := specializeType(switchParamType, pf.TypeParams, elemType)

		fmt.Fprintf(buf, "\tcase %s:\n", caseType)

		// Check if any return type is a type parameter that needs wrapping
		needsReturnWrap := false
		for _, ret := range pf.Returns {
			if containsTypeParam(ret.Type, pf.TypeParams) {
				needsReturnWrap = true
				break
			}
		}

		// Handle multiple return values that need type conversion
		if needsReturnWrap && len(pf.Returns) > 1 {
			// Generate temp variable assignment: _r0, _r1 := Func(...)
			fmt.Fprintf(buf, "\t\t")
			for i := range pf.Returns {
				if i > 0 {
					fmt.Fprintf(buf, ", ")
				}
				fmt.Fprintf(buf, "_r%d", i)
			}
			fmt.Fprintf(buf, " := %s(", dispatchName)

			for i, param := range pf.Params {
				if i > 0 {
					fmt.Fprintf(buf, ", ")
				}
				if containsTypeParam(param.Type, pf.TypeParams) {
					concreteParamType := specializeType(param.Type, pf.TypeParams, elemType)
					fmt.Fprintf(buf, "any(%s).(%s)", param.Name, concreteParamType)
				} else if param.Type == "T" {
					fmt.Fprintf(buf, "any(%s).(%s)", param.Name, elemType)
				} else if isInterfaceTypeParam(param.Type, pf.TypeParams) {
					concreteType := specializeType(getConstraintForParam(param.Type, pf.TypeParams), pf.TypeParams, elemType)
					fmt.Fprintf(buf, "any(%s).(%s)", param.Name, concreteType)
				} else {
					fmt.Fprintf(buf, "%s", param.Name)
				}
			}
			fmt.Fprintf(buf, ")\n")

			// Generate return statement: return any(_r0).(T), any(_r1).(T)
			fmt.Fprintf(buf, "\t\treturn ")
			for i, ret := range pf.Returns {
				if i > 0 {
					fmt.Fprintf(buf, ", ")
				}
				if containsTypeParam(ret.Type, pf.TypeParams) {
					fmt.Fprintf(buf, "any(_r%d).(%s)", i, ret.Type)
				} else {
					fmt.Fprintf(buf, "_r%d", i)
				}
			}
			fmt.Fprintf(buf, "\n")
		} else {
			// Single return or no return - use simpler inline form
			if len(pf.Returns) > 0 {
				if needsReturnWrap {
					fmt.Fprintf(buf, "\t\treturn any(")
				} else {
					fmt.Fprintf(buf, "\t\treturn ")
				}
			} else {
				fmt.Fprintf(buf, "\t\t")
			}
			fmt.Fprintf(buf, "%s(", dispatchName)

			for i, param := range pf.Params {
				if i > 0 {
					fmt.Fprintf(buf, ", ")
				}
				// Check if this parameter needs type assertion
				if containsTypeParam(param.Type, pf.TypeParams) {
					// Use specializeType to get the correct concrete type
					concreteParamType := specializeType(param.Type, pf.TypeParams, elemType)
					fmt.Fprintf(buf, "any(%s).(%s)", param.Name, concreteParamType)
				} else if param.Type == "T" {
					fmt.Fprintf(buf, "any(%s).(%s)", param.Name, elemType)
				} else if isInterfaceTypeParam(param.Type, pf.TypeParams) {
					// Interface type parameter (e.g., P constrained by Predicate[T])
					// Need to assert to concrete interface type (e.g., Predicate[float32])
					concreteType := specializeType(getConstraintForParam(param.Type, pf.TypeParams), pf.TypeParams, elemType)
					fmt.Fprintf(buf, "any(%s).(%s)", param.Name, concreteType)
				} else {
					fmt.Fprintf(buf, "%s", param.Name)
				}
			}
			fmt.Fprintf(buf, ")")
			if needsReturnWrap {
				fmt.Fprintf(buf, ").(%s)", pf.Returns[0].Type)
			}
			fmt.Fprintf(buf, "\n")
		}
	}

	fmt.Fprintf(buf, "\t}\n")

	// Add default return if function has return values
	if len(pf.Returns) > 0 {
		fmt.Fprintf(buf, "\tpanic(\"unreachable\")\n")
	}

	fmt.Fprintf(buf, "}\n\n")
}

// buildDispatchFuncName creates the public function name for the dispatcher.
// For generic functions (multiple type variants):
//
//	BaseSigmoid[float32] -> SigmoidFloat32
//	BaseSigmoid[float64] -> SigmoidFloat64
//
// For non-generic functions (single concrete type):
//
//	BaseDecodeStreamVByte32Into -> DecodeStreamVByte32Into
func buildDispatchFuncName(baseName, elemType string, isGeneric bool) string {
	// Remove "Base" prefix
	name := strings.TrimPrefix(baseName, "Base")

	// Only add type suffix for generic functions (which have multiple type variants)
	if isGeneric {
		name = name + typeNameToSuffix(elemType)
	}

	return name
}

// buildGenericFuncName creates the generic function name (without type suffix).
// BaseSigmoid -> Sigmoid
// BaseAll -> AllP (functions with interface type params get P suffix)
func buildGenericFuncName(baseName string, hasInterfaceParams bool) string {
	name := strings.TrimPrefix(baseName, "Base")
	if hasInterfaceParams {
		name = name + "P"
	}
	return name
}

// buildFuncSignature builds a function signature string from ParsedFunc.
func buildFuncSignature(pf ParsedFunc, elemType string) string {
	var buf bytes.Buffer

	// Parameters
	buf.WriteString("(")
	for i, param := range pf.Params {
		if i > 0 {
			buf.WriteString(", ")
		}
		paramType := specializeType(param.Type, pf.TypeParams, elemType)
		buf.WriteString(param.Name)
		buf.WriteString(" ")
		buf.WriteString(paramType)
	}
	buf.WriteString(")")

	// Return values
	if len(pf.Returns) > 0 {
		buf.WriteString(" ")
		if len(pf.Returns) == 1 && pf.Returns[0].Name == "" {
			// Single unnamed return
			retType := specializeType(pf.Returns[0].Type, pf.TypeParams, elemType)
			buf.WriteString(retType)
		} else {
			// Multiple or named returns
			buf.WriteString("(")
			for i, ret := range pf.Returns {
				if i > 0 {
					buf.WriteString(", ")
				}
				if ret.Name != "" {
					buf.WriteString(ret.Name)
					buf.WriteString(" ")
				}
				retType := specializeType(ret.Type, pf.TypeParams, elemType)
				buf.WriteString(retType)
			}
			buf.WriteString(")")
		}
	}

	return buf.String()
}

// getBaseFilename extracts the base filename without extension.
func getBaseFilename(path string) string {
	base := filepath.Base(path)
	ext := filepath.Ext(base)
	return base[:len(base)-len(ext)]
}

// isInterfaceTypeParam checks if a parameter type is an interface type parameter
// (e.g., "P" where P is constrained by Predicate[T]).
func isInterfaceTypeParam(paramType string, typeParams []TypeParam) bool {
	for _, tp := range typeParams {
		if tp.Name == paramType {
			// Check if this is NOT an element type constraint
			if !strings.Contains(tp.Constraint, "Lanes") &&
				!strings.Contains(tp.Constraint, "Floats") &&
				!strings.Contains(tp.Constraint, "Integers") &&
				!strings.Contains(tp.Constraint, "SignedInts") &&
				!strings.Contains(tp.Constraint, "UnsignedInts") {
				return true
			}
		}
	}
	return false
}

// getConstraintForParam returns the constraint for a type parameter.
func getConstraintForParam(paramType string, typeParams []TypeParam) string {
	for _, tp := range typeParams {
		if tp.Name == paramType {
			return tp.Constraint
		}
	}
	return paramType
}

// containsTypeParam checks if a type string contains any of the element type parameters.
// This includes patterns like:
// - "T" (standalone type param)
// - "[]T" (slice of type param)
// - "*Image[T]" (pointer to generic type)
// - "func(T) T" (function with type params)
func containsTypeParam(typeStr string, typeParams []TypeParam) bool {
	for _, tp := range typeParams {
		// Check if this is an element type parameter (not an interface constraint)
		if !strings.Contains(tp.Constraint, "Lanes") &&
			!strings.Contains(tp.Constraint, "Floats") &&
			!strings.Contains(tp.Constraint, "Integers") &&
			!strings.Contains(tp.Constraint, "SignedInts") &&
			!strings.Contains(tp.Constraint, "UnsignedInts") {
			continue // Skip interface type params
		}

		// Check various patterns where the type param might appear
		paramName := tp.Name

		// Exact match (just "T")
		if typeStr == paramName {
			return true
		}

		// In brackets [T] (generic type argument)
		if strings.Contains(typeStr, "["+paramName+"]") {
			return true
		}

		// Slice type []T
		if strings.Contains(typeStr, "[]"+paramName) {
			return true
		}

		// Function parameter/return (T, or (T or ,T or T))
		if strings.Contains(typeStr, "("+paramName) ||
			strings.Contains(typeStr, ","+paramName) ||
			strings.Contains(typeStr, " "+paramName) ||
			strings.HasSuffix(typeStr, paramName) {
			return true
		}
	}
	return false
}

// collectUsedPackages walks the AST of the given functions and returns
// a set of package names that are referenced (e.g., "unsafe" from "unsafe.Slice").
func collectUsedPackages(funcs []*ast.FuncDecl) map[string]bool {
	used := make(map[string]bool)

	for _, fn := range funcs {
		ast.Inspect(fn, func(n ast.Node) bool {
			// Look for selector expressions like "unsafe.Slice" or "bits.Len32"
			if sel, ok := n.(*ast.SelectorExpr); ok {
				if ident, ok := sel.X.(*ast.Ident); ok {
					used[ident.Name] = true
				}
			}
			return true
		})
	}

	return used
}
