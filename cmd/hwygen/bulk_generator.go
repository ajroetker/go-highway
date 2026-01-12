package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// runBulkMode generates bulk C code for NEON targets, compiles with GOAT,
// and generates Go wrapper functions.
func (g *Generator) runBulkMode(result *ParseResult) error {
	// Find NEON target
	var neonTarget Target
	hasNEON := false
	for _, name := range g.Targets {
		if name == "neon" {
			target, err := GetTarget(name)
			if err != nil {
				return err
			}
			neonTarget = target
			hasNEON = true
			break
		}
	}

	if !hasNEON {
		return fmt.Errorf("bulk mode requires 'neon' target (got: %v)", g.Targets)
	}

	// Collect all eligible functions
	var vecFuncs []ParsedFunc    // Vec→Vec functions
	var sliceFuncs []ParsedFunc  // Slice→Slice functions (composite like GELU)

	for _, pf := range result.Funcs {
		if IsBulkEligible(&pf) {
			vecFuncs = append(vecFuncs, pf)
		} else if IsSliceFunction(&pf) {
			sliceFuncs = append(sliceFuncs, pf)
		}
	}

	totalFuncs := len(vecFuncs) + len(sliceFuncs)
	if totalFuncs == 0 {
		return fmt.Errorf("no bulk-eligible functions found")
	}

	fmt.Printf("Found %d bulk-eligible functions:\n", totalFuncs)
	for _, pf := range vecFuncs {
		fmt.Printf("  - %s (Vec→Vec)\n", pf.Name)
	}
	for _, pf := range sliceFuncs {
		fmt.Printf("  - %s (Slice→Slice)\n", pf.Name)
	}

	// Track generated C files for GOAT compilation
	var cFiles []string

	// Generate C code for Vec→Vec functions
	for _, pf := range vecFuncs {
		elemTypes := getElemTypes(&pf)
		for _, elemType := range elemTypes {
			emitter := NewCEmitter(g.PackageOut, elemType, neonTarget)
			cFile, err := emitter.EmitBulkC(&pf, g.OutputDir)
			if err != nil {
				return fmt.Errorf("emit bulk C for %s (%s): %w", pf.Name, elemType, err)
			}
			cFiles = append(cFiles, cFile)
		}
	}

	// Generate C code for Slice→Slice (composite) functions
	for _, pf := range sliceFuncs {
		elemTypes := getElemTypes(&pf)
		for _, elemType := range elemTypes {
			emitter := NewCEmitter(g.PackageOut, elemType, neonTarget)
			cFile, err := emitter.EmitCompositeC(&pf, g.OutputDir)
			if err != nil {
				return fmt.Errorf("emit composite C for %s (%s): %w", pf.Name, elemType, err)
			}
			cFiles = append(cFiles, cFile)
		}
	}

	// Compile all C files with GOAT
	fmt.Printf("\nCompiling %d C files with GOAT...\n", len(cFiles))
	for _, cFile := range cFiles {
		if err := runGOAT(cFile); err != nil {
			return fmt.Errorf("GOAT compile %s: %w", cFile, err)
		}
		fmt.Printf("  Compiled: %s\n", filepath.Base(cFile))
	}

	// Clean up C and object files (Go build doesn't like them)
	for _, cFile := range cFiles {
		os.Remove(cFile)
		os.Remove(strings.TrimSuffix(cFile, ".c") + ".o")
	}

	// Generate unified wrapper file
	allFuncs := append(vecFuncs, sliceFuncs...)
	if err := g.emitWrappers(allFuncs, neonTarget); err != nil {
		return fmt.Errorf("emit wrappers: %w", err)
	}

	return nil
}

// getElemTypes returns the concrete element types for a function.
func getElemTypes(pf *ParsedFunc) []string {
	if len(pf.TypeParams) > 0 {
		return GetConcreteTypes(pf.TypeParams[0].Constraint)
	}
	return []string{"float32"}
}

// IsSliceFunction checks if a function operates on slices (not Vec).
// These are composite functions like GELU that process entire arrays.
func IsSliceFunction(pf *ParsedFunc) bool {
	hasSliceParam := false
	for _, param := range pf.Params {
		if strings.HasPrefix(param.Type, "[]") {
			hasSliceParam = true
			break
		}
	}
	// Must have slice params and NOT have Vec in signature
	return hasSliceParam && !hasVecInSignature(*pf)
}

// runGOAT invokes the GOAT tool to compile a C file to Go assembly.
// It uses `go tool github.com/gorse-io/goat` which requires goat to be
// declared as a tool dependency in go.mod (via `go get -tool`).
func runGOAT(cFile string) error {
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

	cmd := exec.Command(goBin, "tool", "github.com/gorse-io/goat", absCFile,
		"-O3",
		"-o", filepath.Dir(absCFile),
		"-e=--target=arm64",
		"-e=-march=armv8-a+simd+fp",
		"-e=-fno-builtin-memset",
	)

	cmd.Dir = modRoot
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%w: %s", err, string(output))
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
	base := filepath.Base(dir)
	// Replace hyphens with underscores (GOAT behavior)
	return strings.ReplaceAll(base, "-", "_")
}

// emitWrappers generates a Go file with wrapper functions that call the assembly.
func (g *Generator) emitWrappers(funcs []ParsedFunc, target Target) error {
	var buf bytes.Buffer

	// GOAT derives package name from output directory, so we must match that
	pkgName := goatPackageName(g.OutputDir)

	// File header
	fmt.Fprintf(&buf, "//go:build !noasm && arm64\n")
	fmt.Fprintf(&buf, "// Code generated by hwygen -bulk. DO NOT EDIT.\n\n")
	fmt.Fprintf(&buf, "package %s\n\n", pkgName)
	fmt.Fprintf(&buf, "import \"unsafe\"\n\n")

	// Note: GOAT generates the assembly function stubs in separate .go files,
	// so we don't declare them here.

	// Emit wrapper functions
	fmt.Fprintf(&buf, "// Public wrapper functions\n")
	for _, pf := range funcs {
		elemTypes := getElemTypes(&pf)
		for _, elemType := range elemTypes {
			emitWrapperFunc(&buf, &pf, elemType)
		}
	}

	// Format and write
	filename := filepath.Join(g.OutputDir, "bulk_wrappers_neon_arm64.go")
	if err := os.WriteFile(filename, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("write wrappers: %w", err)
	}

	fmt.Printf("Generated: %s\n", filename)
	return nil
}

// emitWrapperFunc generates a single wrapper function.
func emitWrapperFunc(buf *bytes.Buffer, pf *ParsedFunc, elemType string) {
	// Build function name: BaseExpVec -> ExpBulkF32 or ExpBulkF64
	publicName := buildPublicName(pf.Name, elemType)
	asmName := bulkAsmFuncName(pf.Name, elemType)
	sliceType := "[]" + elemType

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

	fmt.Fprintf(buf, "// %s computes %s for entire arrays using NEON SIMD.\n", publicName, pf.Name)
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

// buildPublicName creates the public function name.
// BaseExpVec -> ExpBulkF32, BaseGELU -> GELUBulkF32
func buildPublicName(baseName, elemType string) string {
	name := strings.TrimPrefix(baseName, "Base")
	name = strings.TrimSuffix(name, "Vec")

	typeSuffix := "F32"
	if elemType == "float64" {
		typeSuffix = "F64"
	}

	return name + "Bulk" + typeSuffix
}

// bulkAsmFuncName creates the assembly function name.
// BaseExpVec, float32 -> exp_bulk_f32_neon
func bulkAsmFuncName(baseName, elemType string) string {
	name := strings.TrimPrefix(baseName, "Base")
	name = strings.TrimSuffix(name, "Vec")
	name = strings.ToLower(name)

	typeSuffix := "f32"
	if elemType == "float64" {
		typeSuffix = "f64"
	}

	return name + "_bulk_" + typeSuffix + "_neon"
}
